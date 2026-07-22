from pathlib import Path
from types import SimpleNamespace
import unittest

from assistant.runtime import (
    approve_plan,
    create_agent_state,
    run_until_blocked,
    start_new_goal,
    store_validated_plan,
)
from assistant.schemas import AnswerQuestionArgs, PlanStepDraft, WriteDocumentArgs


class AgentCoreTests(unittest.TestCase):
    def test_store_validated_plan_initializes_pending_steps(self):
        state = create_agent_state()
        start_new_goal(state, "Say hello")

        store_validated_plan(
            state,
            "say hello",
            [
                PlanStepDraft(
                    step_id=0,
                    description="Echo",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/note.txt",
                        "content": "hello",
                    },
                )
            ],
            output_root=Path("/tmp/domo-output"),
        )

        self.assertEqual(state.status, "waiting")
        self.assertEqual(state.current_step, 0)
        self.assertEqual(state.plan[0].status, "pending")
        self.assertEqual(state.memory.working_memory["output_root"], "/tmp/domo-output")

    def test_run_until_blocked_executes_steps_and_captures_artifacts(self):
        state = create_agent_state()
        start_new_goal(state, "Create note then summarize")
        store_validated_plan(
            state,
            "create note then summarize",
            [
                PlanStepDraft(
                    step_id=0,
                    description="Write a note",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/note.txt",
                        "content": "hello",
                    },
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Summarize previous output",
                    type="llm",
                    tool_name="answer_question",
                    inputs={"question": "@step:0.output.result.text"},
                ),
            ],
            output_root=Path("/tmp/domo-output"),
        )
        approve_plan(state)

        tool_registry = {
            "write_document": SimpleNamespace(
                function=lambda destination_path, content: {
                    "result": {"text": content, "destination_path": destination_path},
                    "metadata": {
                        "display_text": f"Wrote {content}",
                        "artifacts": [
                            {
                                "name": "note.txt",
                                "kind": "file",
                                "path": destination_path,
                                "metadata": {},
                            }
                        ],
                    },
                },
                input_model=WriteDocumentArgs,
            )
        }
        llm_registry = {
            "answer_question": SimpleNamespace(
                function=lambda question: {
                    "result": {"text": question.upper()},
                    "metadata": {
                        "display_text": question.upper(),
                        "artifacts": [],
                    },
                },
                input_model=AnswerQuestionArgs,
            )
        }

        executed_steps = run_until_blocked(state, tool_registry, llm_registry)

        self.assertEqual(state.status, "done")
        self.assertEqual(state.current_step, 2)
        self.assertEqual([step.status for step in executed_steps], ["done", "done"])
        self.assertEqual(state.plan[1].output["result"]["text"], "HELLO")
        self.assertEqual(len(state.memory.artifacts), 1)
        self.assertEqual(state.memory.artifacts[0].path, "/tmp/domo-output/note.txt")

    def test_run_until_blocked_stops_on_error(self):
        state = create_agent_state()
        start_new_goal(state, "Fail")
        store_validated_plan(
            state,
            "fail",
            [
                PlanStepDraft(
                    step_id=0,
                    description="Explode",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/note.txt",
                        "content": "boom",
                    },
                )
            ],
            output_root=Path("/tmp/domo-output"),
        )
        approve_plan(state)

        tool_registry = {
            "write_document": SimpleNamespace(
                function=lambda destination_path, content: (_ for _ in ()).throw(RuntimeError("boom")),
                input_model=WriteDocumentArgs,
            )
        }

        executed_steps = run_until_blocked(state, tool_registry, {})

        self.assertEqual(len(executed_steps), 1)
        self.assertEqual(state.status, "error")
        self.assertEqual(state.plan[0].status, "failed")
        self.assertEqual(state.last_error, "boom")


if __name__ == "__main__":
    unittest.main()
