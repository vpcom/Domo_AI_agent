"""Module for the Domo assistant."""

from __future__ import annotations

import json
import uuid

from assistant.audit import log_event, log_state_snapshot
from assistant.planner import PlanningError, plan_goal
from assistant.registry import LLM_TASKS, TOOLS
from assistant.runtime import (
    approve_plan,
    create_agent_state,
    record_planning_error,
    reset_agent_state,
    run_next_step,
    start_new_goal,
    store_validated_plan,
)
from assistant.schemas import AgentState, ChatMessage, PlanStep, UiEvent


CONFIRMATION_WORDS = {"yes", "y", "ok",
                      "okay", "confirm", "approved", "go ahead"}
AUTO_EXECUTE_WRITE_CONFIDENCE = 0.85
LEGACY_READ_ONLY_TOOL_NAMES = {"read_documents"}
LEGACY_WRITE_TOOL_NAMES = {"write_document",
                           "copy_file", "run_job_agent", "create_job_files"}

"""Create an empty chat history list."""


def create_chat_history() -> list[ChatMessage]:
    return []


"""Create an empty list of UI events."""


def create_ui_events() -> list[UiEvent]:
    return []


"""Create a new unique turn identifier."""


def new_turn_id() -> str:
    return str(uuid.uuid4())


def handle_user_message(
    state: AgentState,
    chat_history: list[ChatMessage],
    ui_events: list[UiEvent],
    user_input: str,
) -> None:
    """Handle the latest user message and update state."""
    turn_id = new_turn_id()
    normalized_input = user_input.strip()
    if not normalized_input:
        return

    chat_history.append(ChatMessage(
        role="user", content=user_input, turn_id=turn_id))
    log_event("user_message_received",
              session_id=state.session_id, turn_id=turn_id)

    if state.status == "waiting" and _is_confirmation_message(normalized_input):
        _execute_current_plan(
            state,
            chat_history,
            ui_events,
            turn_id=turn_id,
            approval_message="Plan approved.",
        )
        return

    if state.status == "waiting":
        _append_event(
            ui_events,
            "planner",
            "Pending plan replaced by a new request.",
            step_id=None,
        )

    start_new_goal(state, user_input)
    _append_event(
        ui_events,
        "system",
        "Received user request.",
        detail="User message captured for planning.",
        expanded_text=user_input,
        step_id=None,
    )
    log_state_snapshot(state, reason="goal_started", turn_id=turn_id)
    _append_state_snapshot_event(
        ui_events,
        state,
        message="State updated after goal start.",
    )

    try:
        plan_draft, output_root, planner_trace = plan_goal(user_input)
        for trace_event in planner_trace:
            _append_event(
                ui_events,
                "planner",
                trace_event["message"],
                detail=trace_event.get("detail", ""),
                expanded_text=trace_event.get("expanded_text", ""),
                step_id=None,
            )
        store_validated_plan(
            state,
            plan_draft.normalized_goal,
            plan_draft.plan,
            output_root=output_root,
        )
        _append_event(
            ui_events,
            "planner",
            f"Prepared a {len(state.plan)}-step plan.",
            detail=_plan_summary(state.plan),
            step_id=None,
        )
        log_state_snapshot(state, reason="plan_stored", turn_id=turn_id)
        _append_state_snapshot_event(
            ui_events,
            state,
            message="State updated after plan storage.",
        )

        auto_execute, auto_reason = _should_auto_execute_plan(
            state.plan,
            plan_draft.confidence,
        )
        if auto_execute:
            _execute_current_plan(
                state,
                chat_history,
                ui_events,
                turn_id=turn_id,
                approval_message="Plan auto-approved for direct execution.",
                approval_detail=auto_reason,
            )
            return

        assistant_text = build_waiting_message(
            state,
            confidence=plan_draft.confidence,
        )
    except Exception as exc:
        if isinstance(exc, PlanningError):
            for trace_event in exc.trace:
                _append_event(
                    ui_events,
                    "planner",
                    trace_event["message"],
                    detail=trace_event.get("detail", ""),
                    expanded_text=trace_event.get("expanded_text", ""),
                    step_id=None,
                )
        record_planning_error(state, str(exc))
        _append_event(
            ui_events,
            "error",
            "Planning failed.",
            detail=str(exc),
            step_id=None,
        )
        log_state_snapshot(state, reason="planning_failed", turn_id=turn_id)
        _append_state_snapshot_event(
            ui_events,
            state,
            message="State updated after planning failure.",
        )
        assistant_text = (
            "I could not prepare a valid plan for that request.\n\n"
            f"Error: {state.last_error}"
        )

    chat_history.append(
        ChatMessage(role="assistant", content=assistant_text, turn_id=turn_id)
    )


def build_waiting_message(
    state: AgentState,
    *,
    confidence: float | None = None,
) -> str:
    """Build waiting message."""
    summary = _plan_summary(state.plan)
    lines = [
        "I prepared this plan:",
        "",
        summary,
        "",
    ]
    if confidence is not None:
        lines.append(f"Planner confidence: {confidence:.2f}")
    if _plan_contains_write(state.plan):
        lines.append(
            "This plan will write files, so reply `yes`, `ok`, or `confirm` to run it."
        )
    else:
        lines.append("Reply `yes`, `ok`, or `confirm` to run it.")
    return "\n".join(lines)


def build_assistant_message(state: AgentState) -> str:
    """Build assistant message."""
    if state.status == "error":
        return f"Execution failed.\n\nError: {state.last_error}"

    if not state.plan:
        return "No plan is available."

    last_step = state.plan[min(
        max(state.current_step - 1, 0), len(state.plan) - 1)]
    if last_step.output:
        display_text = str(
            last_step.output.get("metadata", {}).get("display_text", "")
        ).strip()
        if display_text:
            return display_text

    return f"Completed {len(state.plan)} step(s)."


def build_state_view_model(state: AgentState) -> dict:
    """Build the view model used by the UI."""
    return {
        "session_id": state.session_id,
        "status": state.status,
        "goal": state.goal.model_dump(mode="json"),
        "current_step": state.current_step,
        "plan": [
            {
                "step_id": step.step_id,
                "description": step.description,
                "type": step.type,
                "tool_name": step.tool_name,
                "status": step.status,
                "retry_count": step.retry_count,
                "started_at": step.started_at.isoformat() if step.started_at else None,
                "finished_at": step.finished_at.isoformat() if step.finished_at else None,
            }
            for step in state.plan
        ],
        "memory": state.memory.model_dump(mode="json"),
        "last_error": state.last_error,
    }


def reset_session(
    state: AgentState,
    chat_history: list[ChatMessage],
    ui_events: list[UiEvent],
) -> None:
    """Reset session."""
    reset_agent_state(state)
    chat_history.clear()
    ui_events.clear()
    log_state_snapshot(state, reason="session_reset")


def _execute_current_plan(
    state: AgentState,
    chat_history: list[ChatMessage],
    ui_events: list[UiEvent],
    *,
    turn_id: str,
    approval_message: str,
    approval_detail: str = "",
) -> None:
    """Return execute current plan."""
    _append_event(
        ui_events,
        "planner",
        approval_message,
        detail=approval_detail,
        step_id=None,
    )
    approve_plan(state)
    log_state_snapshot(state, reason="plan_approved", turn_id=turn_id)
    _append_state_snapshot_event(
        ui_events,
        state,
        message="State updated after plan approval.",
    )

    executed_steps: list[PlanStep] = []
    while state.status == "executing":
        step = run_next_step(state, TOOLS, LLM_TASKS)
        if step is None:
            break
        executed_steps.append(step.model_copy(deep=True))
        log_state_snapshot(
            state,
            reason=f"step_{step.step_id}_completed",
            turn_id=turn_id,
        )
        _append_state_snapshot_event(
            ui_events,
            state,
            message=f"State updated after step {step.step_id}.",
            step_id=step.step_id,
        )

    _append_execution_events(ui_events, executed_steps)
    log_state_snapshot(state, reason="execution_finished", turn_id=turn_id)
    _append_state_snapshot_event(
        ui_events,
        state,
        message="State updated after execution finished.",
    )
    chat_history.append(
        ChatMessage(
            role="assistant",
            content=build_assistant_message(state),
            turn_id=turn_id,
        )
    )


def _plan_summary(plan: list[PlanStep]) -> str:
    """Return plan summary."""
    return "\n".join(
        f"{index}. [{step.type}] `{step.tool_name}` - {step.description}"
        for index, step in enumerate(plan, start=1)
    )


def _append_execution_events(ui_events: list[UiEvent], steps: list[PlanStep]) -> None:
    """Return append execution events."""
    for step in steps:
        if step.status == "done":
            detail = ""
            if step.output is not None:
                detail = str(step.output.get(
                    "metadata", {}).get("display_text", ""))
            _append_event(
                ui_events,
                "execution",
                f"Completed step {step.step_id}: {step.description}",
                detail=detail,
                step_id=step.step_id,
            )
        elif step.status == "failed":
            _append_event(
                ui_events,
                "error",
                f"Failed step {step.step_id}: {step.description}",
                detail=step.output.get("metadata", {}).get("display_text", "")
                if step.output
                else "",
                step_id=step.step_id,
            )


def _append_event(
    ui_events: list[UiEvent],
    category: str,
    message: str,
    *,
    detail: str = "",
    expanded_text: str = "",
    step_id: int | None,
) -> None:
    """Return append event."""
    ui_events.append(
        UiEvent(
            event_id=str(uuid.uuid4()),
            category=category,
            message=message,
            detail=detail,
            expanded_text=expanded_text,
            step_id=step_id,
        )
    )


def _append_state_snapshot_event(
    ui_events: list[UiEvent],
    state: AgentState,
    *,
    message: str,
    step_id: int | None = None,
) -> None:
    """Return append state snapshot event."""
    _append_event(
        ui_events,
        "state",
        message,
        detail=f"status={state.status}, current_step={state.current_step}",
        expanded_text=json.dumps(
            state.model_dump(mode="json"),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        step_id=step_id,
    )


def _should_auto_execute_plan(
    plan: list[PlanStep],
    confidence: float,
) -> tuple[bool, str]:
    """Return should auto execute plan."""
    if not plan:
        return False, ""
    if all(_is_read_only_step(step) for step in plan):
        return True, "Read-only plan."
    if (
        all(_is_read_only_step(step) or _is_write_step(step) for step in plan)
        and _plan_contains_write(plan)
        and confidence >= AUTO_EXECUTE_WRITE_CONFIDENCE
    ):
        return True, f"High-confidence write plan (confidence={confidence:.2f})."
    return False, ""


def _plan_contains_write(plan: list[PlanStep]) -> bool:
    """Return plan contains write."""
    return any(_is_write_step(step) for step in plan)


def _is_read_only_step(step: PlanStep) -> bool:
    """Return whether read only step."""
    if step.type == "llm":
        return True
    spec = TOOLS.get(step.tool_name)
    if spec is not None:
        risks = tuple(getattr(spec, "risks", ()))
        group = getattr(spec, "group", "")
        if group in {"read_tools", "transform_tools"}:
            return True
        if risks and "write" not in risks and "subprocess" not in risks:
            return True
    if step.tool_name == "search_web":
        return not bool(step.inputs.get("output_path"))
    return step.tool_name in LEGACY_READ_ONLY_TOOL_NAMES


def _is_write_step(step: PlanStep) -> bool:
    """Return whether write step."""
    if step.type != "tool":
        return False
    spec = TOOLS.get(step.tool_name)
    if spec is not None:
        risks = tuple(getattr(spec, "risks", ()))
        if risks and ("write" in risks or "subprocess" in risks):
            return True
    if step.tool_name == "search_web":
        return bool(step.inputs.get("output_path"))
    return step.tool_name in LEGACY_WRITE_TOOL_NAMES


def _is_confirmation_message(user_input: str) -> bool:
    """Return whether confirmation message."""
    lowered = user_input.strip().lower().rstrip(".!?")
    return lowered in CONFIRMATION_WORDS


__all__ = [
    "build_assistant_message",
    "build_state_view_model",
    "create_agent_state",
    "create_chat_history",
    "create_ui_events",
    "handle_user_message",
    "new_turn_id",
    "reset_session",
]
