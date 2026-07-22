"""Deterministic core state transitions and step execution."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
import uuid

from assistant.policy import (
    GOAL_REFERENCE_PATTERN,
    MEMORY_REFERENCE_PATTERN,
    STEP_REFERENCE_PATTERN,
    validate_and_normalize_tool_inputs,
)
from assistant.schemas import AgentState, ArtifactRecord, PlanStep, utc_now


def create_agent_state() -> AgentState:
    """Create a fully initialized agent state with a unique session id."""

    return AgentState(session_id=str(uuid.uuid4()))


def start_new_goal(state: AgentState, user_input: str) -> None:
    """Reset goal-scoped fields and enter the planning state."""

    state.status = "planning"
    state.goal.user_input = user_input
    state.goal.normalized_goal = ""
    state.plan = []
    state.current_step = 0
    state.memory.working_memory = {}
    state.memory.artifacts = []
    state.last_error = None


def store_validated_plan(
    state: AgentState,
    normalized_goal: str,
    plan: Iterable[object],
    *,
    output_root: Path | None = None,
) -> None:
    """Store a validated plan without changing the state schema."""

    state.goal.normalized_goal = normalized_goal
    state.plan = []

    for step in plan:
        if isinstance(step, PlanStep):
            state.plan.append(step.model_copy(deep=True))
            continue

        if hasattr(step, "model_dump"):
            payload = step.model_dump()
        else:
            payload = dict(step)

        state.plan.append(
            PlanStep(
                step_id=payload["step_id"],
                description=payload["description"],
                type=payload["type"],
                tool_name=payload["tool_name"],
                inputs=deepcopy(payload.get("inputs", {})),
                status="pending",
                output=None,
                started_at=None,
                finished_at=None,
                retry_count=0,
            )
        )

    state.current_step = 0
    state.last_error = None
    state.status = "waiting"
    if output_root is not None:
        state.memory.working_memory["output_root"] = str(output_root)


def record_planning_error(state: AgentState, error_message: str) -> None:
    """Keep the agent in planning and persist the latest planning error."""

    state.status = "planning"
    state.plan = []
    state.current_step = 0
    state.last_error = error_message


def approve_plan(state: AgentState) -> None:
    """Move a prepared plan from waiting into executing."""

    if not state.plan:
        raise ValueError("There is no plan to approve.")

    state.status = "executing"
    state.last_error = None


def run_next_step(
    state: AgentState,
    tool_registry: dict,
    llm_registry: dict,
) -> PlanStep | None:
    """Execute the current step and advance the state machine by one step."""

    if state.status != "executing":
        return None

    if state.current_step >= len(state.plan):
        state.status = "done"
        return None

    step = state.plan[state.current_step]
    step.status = "running"
    step.retry_count += 1
    step.started_at = step.started_at or utc_now()
    step.finished_at = None

    try:
        resolved_inputs = _resolve_inputs(step.inputs, state)
        if step.type == "tool":
            output_root = state.memory.working_memory.get("output_root")
            normalized_inputs = validate_and_normalize_tool_inputs(
                step.tool_name,
                resolved_inputs,
                output_root=Path(output_root) if isinstance(output_root, str) else None,
                allow_references=False,
            )
            output = tool_registry[step.tool_name].function(**normalized_inputs)
        else:
            validated_inputs = llm_registry[step.tool_name].input_model.model_validate(
                resolved_inputs
            )
            output = llm_registry[step.tool_name].function(
                **validated_inputs.model_dump(exclude_none=True)
            )

        if not isinstance(output, dict) or "result" not in output or "metadata" not in output:
            raise ValueError(
                f"Step `{step.tool_name}` returned an invalid output contract."
            )

        step.inputs = resolved_inputs
        step.output = output
        step.status = "done"
        step.finished_at = utc_now()
        _capture_artifacts(state, step)
        state.memory.working_memory["last_step_id"] = step.step_id
        state.memory.working_memory["last_step_output"] = deepcopy(output["result"])
        state.current_step += 1
        state.last_error = None
        if state.current_step >= len(state.plan):
            state.status = "done"
        return step
    except Exception as exc:
        step.status = "failed"
        step.finished_at = utc_now()
        state.last_error = str(exc)
        state.status = "error"
        return step


def run_until_blocked(
    state: AgentState,
    tool_registry: dict,
    llm_registry: dict,
) -> list[PlanStep]:
    """Run steps until the agent reaches done, error, or another blocked state."""

    executed_steps: list[PlanStep] = []
    while state.status == "executing":
        step = run_next_step(state, tool_registry, llm_registry)
        if step is None:
            break
        executed_steps.append(step)
        if state.status in {"done", "error", "waiting"}:
            break
    return executed_steps


def reset_agent_state(state: AgentState) -> None:
    """Reset an existing state object back to the initial structure."""

    fresh = create_agent_state()
    state.session_id = fresh.session_id
    state.status = fresh.status
    state.goal = fresh.goal
    state.plan = fresh.plan
    state.current_step = fresh.current_step
    state.memory = fresh.memory
    state.last_error = fresh.last_error


def _resolve_inputs(value: object, state: AgentState):
    """Recursively resolve goal, memory, and prior-step references."""

    if isinstance(value, dict):
        return {key: _resolve_inputs(item, state) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_inputs(item, state) for item in value]
    if isinstance(value, str):
        return _resolve_reference(value, state)
    return value


def _resolve_reference(value: str, state: AgentState):
    """Resolve a single reference string against the current agent state."""

    goal_match = GOAL_REFERENCE_PATTERN.fullmatch(value)
    if goal_match:
        return getattr(state.goal, goal_match.group(1))

    memory_match = MEMORY_REFERENCE_PATTERN.fullmatch(value)
    if memory_match:
        key = memory_match.group(1)
        if key not in state.memory.working_memory:
            raise ValueError(f"Missing memory reference: {value}")
        return state.memory.working_memory[key]

    step_match = STEP_REFERENCE_PATTERN.fullmatch(value)
    if step_match:
        step_id = int(step_match.group(1))
        path = step_match.group(2)
        if step_id >= len(state.plan):
            raise ValueError(f"Unknown step reference: {value}")

        step = state.plan[step_id]
        if step.output is None:
            raise ValueError(f"Referenced step has no output yet: {value}")

        current = step.output
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Missing reference path `{path}` in `{value}`")
            current = current[part]
        return deepcopy(current)

    return value


def _capture_artifacts(state: AgentState, step: PlanStep) -> None:
    """Append structured artifact records from a completed step output."""

    if step.output is None:
        return

    metadata = step.output.get("metadata", {})
    raw_artifacts = metadata.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        return

    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        state.memory.artifacts.append(
            ArtifactRecord(
                name=str(item.get("name", "")) or f"artifact-{step.step_id}",
                kind=str(item.get("kind", "artifact")),
                path=str(item.get("path", "")),
                step_id=step.step_id,
                metadata=dict(item.get("metadata", {})),
            )
        )
