from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assistant.controller import (
    create_agent_state,
    create_chat_history,
    create_ui_events,
    handle_user_message,
    new_turn_id,
    reset_session,
)
from assistant.planner import PlanningError
from assistant.schemas import ChatMessage, PlanDraft, PlanStepDraft, UiEvent, WriteDocumentArgs


class ConversationRuntimeTests(unittest.TestCase):
    def test_initial_agent_state_is_fully_initialized(self):
        state = create_agent_state()

        self.assertEqual(state.status, "planning")
        self.assertEqual(state.goal.user_input, "")
        self.assertEqual(state.goal.normalized_goal, "")
        self.assertEqual(state.plan, [])
        self.assertEqual(state.current_step, 0)
        self.assertEqual(state.memory.working_memory, {})
        self.assertEqual(state.memory.artifacts, [])
        self.assertIsNone(state.last_error)

    def test_llm_question_auto_executes_without_confirmation(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()
        output_root = Path("/tmp/domo-output")

        with patch(
            "assistant.controller.plan_goal",
            return_value=(
                PlanDraft(
                    normalized_goal="answer the question",
                    confidence=0.20,
                    plan=[
                        PlanStepDraft(
                            step_id=0,
                            description="Answer the question",
                            type="llm",
                            tool_name="answer_question",
                            inputs={"question": "@goal:user_input"},
                        )
                    ],
                ),
                output_root,
                [
                    {
                        "message": "Planner prompt prepared.",
                        "detail": "",
                        "expanded_text": "PROMPT",
                    },
                    {
                        "message": "Planner raw response received.",
                        "detail": "",
                        "expanded_text": '{"normalized_goal":"answer the question","confidence":0.2,"plan":[]}',
                    },
                ],
            ),
        ), patch("assistant.controller.log_state_snapshot"), patch(
            "assistant.llm_tasks.call_llm",
            return_value="Paris.",
        ):
            handle_user_message(
                state,
                chat_history,
                ui_events,
                "What is the capital of France?",
            )

        self.assertEqual(state.status, "done")
        self.assertEqual(state.goal.user_input, "What is the capital of France?")
        self.assertEqual(state.goal.normalized_goal, "answer the question")
        self.assertEqual(len(state.plan), 1)
        self.assertEqual(state.plan[0].status, "done")
        self.assertEqual(state.current_step, 1)
        self.assertEqual(state.memory.working_memory["output_root"], str(output_root))
        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[-1].content, "Paris.")
        self.assertTrue(any(event.category == "state" for event in ui_events))
        self.assertTrue(any(event.message == "Planner prompt prepared." for event in ui_events))
        self.assertTrue(any(event.message == "Planner raw response received." for event in ui_events))
        self.assertTrue(any(event.message == "Plan auto-approved for direct execution." for event in ui_events))

    def test_low_confidence_write_waits_for_confirmation_then_executes(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()
        fake_tools = {
            "write_document": SimpleNamespace(
                function=lambda destination_path, content: {
                    "result": {
                        "destination_path": destination_path,
                        "content": content,
                    },
                    "metadata": {
                        "display_text": "Wrote note.",
                        "artifacts": [],
                    },
                },
                input_model=WriteDocumentArgs,
            )
        }

        with patch(
            "assistant.controller.plan_goal",
            return_value=(
                PlanDraft(
                    normalized_goal="write note",
                    confidence=0.40,
                    plan=[
                        PlanStepDraft(
                            step_id=0,
                            description="Write the note",
                            type="tool",
                            tool_name="write_document",
                            inputs={
                                "destination_path": "data/outputs/note.txt",
                                "content": "Hello",
                            },
                        )
                    ],
                ),
                Path("/tmp/domo-output"),
                [],
            ),
        ), patch("assistant.controller.log_state_snapshot"), patch(
            "assistant.controller.TOOLS",
            fake_tools,
        ):
            handle_user_message(state, chat_history, ui_events, "Write a note")
            self.assertEqual(state.status, "waiting")
            handle_user_message(state, chat_history, ui_events, "yes")

        self.assertEqual(state.status, "done")
        self.assertEqual(state.current_step, 1)
        self.assertEqual(state.plan[0].status, "done")
        self.assertEqual(chat_history[-1].content, "Wrote note.")
        self.assertGreaterEqual(len(ui_events), 2)
        self.assertTrue(any(event.message == "Plan approved." for event in ui_events))

    def test_waiting_non_confirmation_replaces_plan(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()

        first_plan = (
            PlanDraft(
                normalized_goal="first goal",
                confidence=0.35,
                plan=[
                    PlanStepDraft(
                        step_id=0,
                        description="Write the first note",
                        type="tool",
                        tool_name="write_document",
                        inputs={
                            "destination_path": "data/outputs/first.txt",
                            "content": "first",
                        },
                    )
                ],
            ),
            Path("/tmp/first-output"),
            [],
        )
        second_plan = (
            PlanDraft(
                normalized_goal="second goal",
                confidence=0.40,
                plan=[
                    PlanStepDraft(
                        step_id=0,
                        description="Write the second note",
                        type="tool",
                        tool_name="write_document",
                        inputs={
                            "destination_path": "data/outputs/second.txt",
                            "content": "second",
                        },
                    )
                ],
            ),
            Path("/tmp/second-output"),
            [],
        )

        with patch(
            "assistant.controller.plan_goal",
            side_effect=[first_plan, second_plan],
        ), patch("assistant.controller.log_state_snapshot"):
            handle_user_message(state, chat_history, ui_events, "Write the first note")
            handle_user_message(state, chat_history, ui_events, "Write the second note instead")

        self.assertEqual(state.status, "waiting")
        self.assertEqual(state.goal.user_input, "Write the second note instead")
        self.assertEqual(state.goal.normalized_goal, "second goal")
        self.assertEqual(state.plan[0].tool_name, "write_document")
        self.assertEqual(state.memory.working_memory["output_root"], "/tmp/second-output")

    def test_high_confidence_write_auto_executes_without_confirmation(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()
        fake_tools = {
            "write_document": SimpleNamespace(
                function=lambda destination_path, content: {
                    "result": {
                        "destination_path": destination_path,
                        "content": content,
                    },
                    "metadata": {
                        "display_text": "Wrote note immediately.",
                        "artifacts": [],
                    },
                },
                input_model=WriteDocumentArgs,
            )
        }

        with patch(
            "assistant.controller.plan_goal",
            return_value=(
                PlanDraft(
                    normalized_goal="write note",
                    confidence=0.95,
                    plan=[
                        PlanStepDraft(
                            step_id=0,
                            description="Write the note",
                            type="tool",
                            tool_name="write_document",
                            inputs={
                                "destination_path": "data/outputs/note.txt",
                                "content": "Hello",
                            },
                        )
                    ],
                ),
                Path("/tmp/domo-output"),
                [],
            ),
        ), patch("assistant.controller.log_state_snapshot"), patch(
            "assistant.controller.TOOLS",
            fake_tools,
        ):
            handle_user_message(state, chat_history, ui_events, "Write a note")

        self.assertEqual(state.status, "done")
        self.assertEqual(state.current_step, 1)
        self.assertEqual(chat_history[-1].content, "Wrote note immediately.")
        self.assertTrue(any(event.message == "Plan auto-approved for direct execution." for event in ui_events))

    def test_planning_failure_keeps_prompt_and_response_trace_in_ui_events(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()

        with patch(
            "assistant.controller.plan_goal",
            side_effect=PlanningError(
                "Planner response was not valid JSON.",
                trace=[
                    {
                        "message": "Planner prompt prepared.",
                        "detail": "",
                        "expanded_text": "PROMPT",
                    },
                    {
                        "message": "Planner raw response received.",
                        "detail": "",
                        "expanded_text": "not json",
                    },
                ],
            ),
        ), patch("assistant.controller.log_state_snapshot"):
            handle_user_message(state, chat_history, ui_events, "Hello")

        self.assertEqual(state.status, "planning")
        self.assertEqual(
            state.last_error,
            "Planner response was not valid JSON.",
        )
        self.assertTrue(any(event.message == "Planner prompt prepared." for event in ui_events))
        self.assertTrue(any(event.message == "Planner raw response received." for event in ui_events))
        self.assertTrue(any(event.message == "Planning failed." for event in ui_events))

    def test_reset_session_clears_interface_state(self):
        state = create_agent_state()
        chat_history = create_chat_history()
        ui_events = create_ui_events()
        chat_history.append(
            ChatMessage(role="assistant", content="stale", turn_id=new_turn_id())
        )
        ui_events.append(
            UiEvent(event_id="event-1", category="system", message="stale")
        )

        with patch("assistant.controller.log_state_snapshot"):
            reset_session(state, chat_history, ui_events)

        self.assertEqual(state.status, "planning")
        self.assertEqual(chat_history, [])
        self.assertEqual(ui_events, [])


if __name__ == "__main__":
    unittest.main()
