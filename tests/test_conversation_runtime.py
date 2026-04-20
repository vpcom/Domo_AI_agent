import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from assistant import runtime
from assistant.schemas import (
    CreateJobFilesArgs,
    MatchCvArgs,
    PlannedToolCall,
    WriteDocumentArgs,
)


def _planner_response(
    *,
    action=None,
    arguments=None,
    steps=None,
    turn_intent="confirm",
    assistant_message="",
    missing_fields=None,
    confidence=0.91,
    reasoning="Structured planner decision.",
    confirmation_required=None,
    activity_events=None,
):
    if confirmation_required is None:
        confirmation_required = turn_intent == "confirm" and action is not None

    payload = {
        "assistant_message": assistant_message,
        "turn_intent": turn_intent,
        "action": action,
        "arguments": arguments or {},
        "steps": steps or [],
        "missing_fields": missing_fields or [],
        "confidence": confidence,
        "reasoning": reasoning,
        "confirmation_required": confirmation_required,
    }
    if activity_events is not None:
        payload["activity_events"] = activity_events
    return json.dumps(payload)


class ConversationRuntimeTests(unittest.TestCase):
    def test_initial_job_search_loads_defaults_and_prepares_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="run_job_agent",
                reasoning="The user is asking to search for jobs.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertTrue(result.confirmation_required)
        self.assertIsNotNone(state.pending_tool_call)
        self.assertEqual(state.confirmation_state, "awaiting_confirmation")
        self.assertEqual(state.context.parameters["role"].source, "default")
        self.assertEqual(state.context.parameters["role"].status, "pending")
        self.assertEqual(state.context.parameters["location"].source, "default")
        self.assertEqual(state.context.request["selected_workflow"].value, "run_job_agent")
        self.assertEqual(
            state.context.request["confirmation_state"].value,
            "awaiting_confirmation",
        )
        self.assertTrue(
            any(
                event.summary == "Planner proposed `run_job_agent`."
                and event.detail == "The user is asking to search for jobs."
                for event in result.activity_events
            )
        )
        self.assertTrue(
            any(
                event.summary == "Planner prompt prepared." and event.raw_lines
                for event in result.activity_events
            )
        )
        self.assertTrue(
            any(
                event.summary == "Planner raw response received." and event.raw_lines
                for event in result.activity_events
            )
        )
        self.assertFalse(
            any(
                event.summary == "Received chat message."
                for event in result.activity_events
            )
        )

    def test_planner_prompt_examples_are_generic(self):
        state = runtime.create_conversation_state()
        prompt = runtime._build_prompt(state, "Analyze this document")

        self.assertIn("Company - Role", prompt)
        self.assertNotIn("Natzka", prompt)
        self.assertIn("not limited to job-related requests", prompt)
        self.assertIn("PENDING PLAN JSON", prompt)
        self.assertIn('"turn_intent": "respond" | "clarify" | "confirm" | "execute"', prompt)
        self.assertIn("`steps` is the canonical executable plan", prompt)
        self.assertIn("Every executable workflow, including a single action", prompt)
        self.assertIn("Top-level `action` and `arguments` are optional legacy mirrors", prompt)
        self.assertIn("{working_folder}", prompt)
        self.assertIn("shared staged folder", prompt)
        self.assertIn("Never ask the user for it and never list it in `missing_fields`", prompt)
        self.assertIn(
            "The runtime validates planned earlier outputs even before they exist on disk",
            prompt,
        )
        self.assertIn('Do not write placeholder strings like `"Full pasted job text here"`', prompt)
        self.assertIn(
            f"data/outputs/{runtime.TODAY_STAMP} - Ahead Health - Senior Product Engineer/job_description_raw.txt",
            prompt,
        )
        self.assertIn("can return their findings directly in chat", prompt)
        self.assertIn("save the findings to a file once you confirm", prompt)
        self.assertNotIn("MUST put it in the top-level `action` field", prompt)

    def test_yes_executes_only_when_confirmation_is_pending(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                turn_intent="respond",
                assistant_message="There is nothing pending to confirm yet.",
                confirmation_required=False,
            ),
        ), patch("assistant.runtime.log_activity_event"):
            no_pending = runtime.plan_chat_turn(
                state,
                "yes",
                turn_id="turn-1",
            )
        self.assertEqual(no_pending.turn_intent, "respond")
        self.assertIn("nothing pending", no_pending.assistant_message.lower())

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user confirmed the already pending validated plan.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                result,
                turn_id="turn-2",
            )
            confirmed = runtime.plan_chat_turn(
                state,
                "yes",
                turn_id="turn-3",
            )

        self.assertEqual(confirmed.turn_intent, "execute")
        self.assertIsNotNone(confirmed.proposed_tool_call)

    def test_ok_executes_when_confirmation_is_pending(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user confirmed the already pending validated plan.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                result,
                turn_id="turn-1",
            )
            confirmed = runtime.plan_chat_turn(
                state,
                "ok",
                turn_id="turn-2",
            )

        self.assertEqual(confirmed.turn_intent, "execute")
        self.assertIsNotNone(confirmed.proposed_tool_call)

    def test_i_confirm_executes_when_confirmation_is_pending(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(
                    action="search_web",
                    arguments={"query": "mickey"},
                ),
                _planner_response(
                    action="search_web",
                    arguments={"query": "mickey"},
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user confirmed the already pending validated plan.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search the web for mickey",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search the web for mickey",
                result,
                turn_id="turn-1",
            )
            confirmed = runtime.plan_chat_turn(
                state,
                "I confirm",
                turn_id="turn-2",
            )

        self.assertEqual(confirmed.turn_intent, "execute")
        self.assertIsNotNone(confirmed.proposed_tool_call)
        self.assertEqual(confirmed.proposed_tool_call.tool_name, "search_web")

    def test_ok_with_punctuation_executes_when_confirmation_is_pending(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(
                    action="search_web",
                    arguments={"query": "mickey"},
                ),
                _planner_response(
                    action="search_web",
                    arguments={"query": "mickey"},
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user confirmed the already pending validated plan.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search the web for mickey",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search the web for mickey",
                result,
                turn_id="turn-1",
            )
            confirmed = runtime.plan_chat_turn(
                state,
                "ok, search, I am ready",
                turn_id="turn-2",
            )

        self.assertEqual(confirmed.turn_intent, "execute")
        self.assertIsNotNone(confirmed.proposed_tool_call)
        self.assertEqual(confirmed.proposed_tool_call.tool_name, "search_web")

    def test_person_lookup_is_planned_by_llm_as_search_web(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="search_web",
                arguments={"query": "Bob Smith"},
                assistant_message="I can search the web for Bob Smith once you confirm.",
                reasoning="The user asked about a person and that requires web search.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Hey Domo! Do you know Bob Smith",
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertTrue(result.confirmation_required)
        self.assertIsNotNone(result.proposed_tool_call)
        self.assertEqual(result.proposed_tool_call.tool_name, "search_web")
        self.assertEqual(result.proposed_tool_call.parameters.query, "Bob Smith")
        self.assertIn("search the web for Bob Smith", result.assistant_message)

    def test_generic_capabilities_answer_is_planned_by_llm_and_not_job_only(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                turn_intent="respond",
                assistant_message=(
                    "I can answer open questions in chat, search the web, inspect project files and data, "
                    "summarize or evaluate documents, and write new files under data/outputs/."
                ),
                confirmation_required=False,
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "What can you do?",
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "respond")
        self.assertIn("search the web", result.assistant_message)
        self.assertIn("inspect project files", result.assistant_message)
        self.assertNotIn("job-related requests", result.assistant_message.lower())

    def test_execute_intent_for_changed_plan_is_downgraded_to_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    arguments={"remote_only": True},
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user wants to run the revised plan immediately.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                initial,
                turn_id="turn-1",
            )
            revised = runtime.plan_chat_turn(
                state,
                "Actually make it remote only",
                turn_id="turn-2",
            )

        self.assertEqual(revised.turn_intent, "confirm")
        self.assertTrue(revised.confirmation_required)
        self.assertIsNotNone(revised.proposed_tool_call)
        self.assertTrue(revised.proposed_tool_call.parameters.remote_only)

    def test_follow_up_uses_current_workflow_context(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    arguments={"remote_only": True},
                    reasoning="The user wants to keep the same search and add a remote-only filter.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                initial,
                turn_id="turn-1",
            )
            follow_up = runtime.plan_chat_turn(
                state,
                "same search but remote only",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "same search but remote only",
                follow_up,
                turn_id="turn-2",
            )

        self.assertEqual(follow_up.turn_intent, "confirm")
        self.assertEqual(state.context.request["selected_workflow"].value, "run_job_agent")
        self.assertTrue(state.context.parameters["remote_only"].value)
        self.assertEqual(state.pending_tool_call.parameters.remote_only, True)
        self.assertEqual(
            state.pending_tool_call.parameters.role,
            runtime.JOB_SEARCH_DEFAULTS["role"],
        )

    def test_revision_after_confirmation_rebuilds_pending_call(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    arguments={"location": "City Alpha"},
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                initial,
                turn_id="turn-1",
            )
            revised = runtime.plan_chat_turn(
                state,
                "location City Alpha",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "location City Alpha",
                revised,
                turn_id="turn-2",
            )

        self.assertEqual(revised.turn_intent, "confirm")
        self.assertEqual(state.confirmation_state, "awaiting_confirmation")
        self.assertEqual(state.pending_tool_call.parameters.location, "City Alpha")

    def test_rejection_with_revised_request_switches_workflow(self):
        state = runtime.create_conversation_state()
        planned_call = PlannedToolCall(
            tool_name="create_job_files",
            parameters=CreateJobFilesArgs(job_folder="/tmp/company-alpha"),
            request_id="req-123",
        )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(action="run_job_agent"),
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "prepare me job search document for the company-alpha job",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "prepare me job search document for the company-alpha job",
                initial,
                turn_id="turn-1",
            )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="create_job_files",
                arguments={"job_folder": "company-alpha"},
                reasoning="The user wants local document generation for an existing job folder.",
            ),
        ), patch(
            "assistant.runtime.plan_tool_call",
            return_value=planned_call,
        ), patch("assistant.runtime.log_activity_event"):
            revised = runtime.plan_chat_turn(
                state,
                "No, find the folder in jobs containing the company name company-alpha and prepare documents for this job offer",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "No, find the folder in jobs containing the company name company-alpha and prepare documents for this job offer",
                revised,
                turn_id="turn-2",
            )

        self.assertEqual(revised.turn_intent, "confirm")
        self.assertEqual(state.context.request["selected_workflow"].value, "create_job_files")
        self.assertEqual(state.context.parameters["job_folder"].value, "/tmp/company-alpha")
        self.assertEqual(state.pending_tool_call.tool_name, "create_job_files")
        self.assertEqual(state.pending_tool_call.parameters.job_folder, "/tmp/company-alpha")
        self.assertNotIn("role", state.context.parameters)

    def test_match_cv_without_job_folder_stays_in_clarification(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="match_cv",
                turn_intent="clarify",
                confirmation_required=False,
                missing_fields=["job_folder"],
                confidence=0.84,
                reasoning="CV matching needs a target job folder.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "match cv",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "match cv",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "clarify")
        self.assertEqual(result.missing_fields, ["job_folder"])
        self.assertEqual(
            state.context.request["open_question"].value,
            "Which job folder should I use?",
        )
        self.assertIsNone(state.pending_tool_call)

    def test_llm_inferred_values_remain_pending_until_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="run_job_agent",
                arguments={"role": "Application Engineer"},
                reasoning="The user expressed interest in backend systems, which maps to a platform role.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "I would like to work on backend systems",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "I would like to work on backend systems",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(state.context.parameters["role"].source, "inferred")
        self.assertEqual(state.context.parameters["role"].status, "pending")

    def test_invalid_action_degrades_to_clarification(self):
        state = runtime.create_conversation_state()
        bad_payload = """
        {
          "assistant_message": "I can help you analyze the job ad you pasted.",
          "turn_intent": "confirm",
          "action": "analyze_job_ad",
          "arguments": {
            "role": "Role Alpha"
          },
          "missing_fields": [],
          "confidence": 0.72,
          "reasoning": "This looks like a job-ad analysis request.",
          "confirmation_required": true
        }
        """

        with patch("assistant.runtime.call_llm", return_value=bad_payload), patch(
            "assistant.runtime.log_activity_event"
        ):
            result = runtime.plan_chat_turn(
                state,
                "i found a job add, can you analyse it if I copy paste it here?",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "i found a job add, can you analyse it if I copy paste it here?",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "clarify")
        self.assertFalse(result.confirmation_required)
        self.assertIsNone(state.pending_tool_call)
        self.assertIn("analyze the job ad", state.messages[-1].content.lower())
        self.assertIn("summarize_documents", state.messages[-1].content)
        self.assertIsNone(state.context.request["selected_workflow"].value)
        self.assertEqual(state.context.request["open_question"].value, state.messages[-1].content)
        self.assertTrue(
            any(
                "Closest supported actions" in event.detail
                for event in result.activity_events
            )
        )

    def test_nested_fenced_json_response_is_salvaged(self):
        state = runtime.create_conversation_state()
        nested_payload = """
        {
          "assistant_message": "I can help analyze the job advertisement you provided. To do this, I suggest creating a new job file and pasting the text into it. Then, I will use the `evaluate_documents` action to evaluate the document against the criteria of being a Role Alpha in City Beta. Here's my proposed workflow:\n\n```json\n{\n  \\"assistant_message\\": \\"I can help analyze the job advertisement you provided. I will create a new job file, paste the text into it, and then evaluate the document against the criteria of being a Role Alpha in City Beta.\\",\n  \\"turn_intent\\": \\"respond\\",\n  \\"action\\": \\"create_job_files\\",\n  \\"arguments\\": {\n    \\"job_folder\\": \\"data/outputs/new_job\\"\n  },\n  \\"confirmation_required\\": true\n}\n```",
          "turn_intent": "clarify",
          "missing_fields": [],
          "confidence": 0.95,
          "reasoning": "The user provided a job advertisement and asked for it to be analyzed. To do this, I propose creating a new job file and pasting the text into it, then using the `evaluate_documents` action to evaluate the document against the criteria of being a Role Alpha in City Beta."
        }
        """

        with patch("assistant.runtime.call_llm", return_value=nested_payload), patch(
            "assistant.runtime.log_activity_event"
        ):
            result = runtime.plan_chat_turn(
                state,
                "i found a job add, can you analyse it if I copy paste it here?",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "i found a job add, can you analyse it if I copy paste it here?",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertIsNotNone(state.pending_tool_call)
        self.assertEqual(state.pending_tool_call.tool_name, "create_job_files")
        self.assertEqual(state.context.request["selected_workflow"].value, "create_job_files")

    def test_write_document_prepares_confirmation_without_exposing_full_content_in_context(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="write_document",
                arguments={
                    "destination_path": "data/outputs/20260412 - Company Alpha - Application Engineer/job_description_raw.txt",
                    "content": "Application Engineer role\\nBuild services\\nShip systems",
                },
                reasoning="The user wants to save pasted job text into a local file.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Save this job ad as a file in jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Save this job ad as a file in jobs",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(state.pending_tool_call.tool_name, "write_document")
        self.assertIn("destination_path", state.context.parameters)
        self.assertNotIn("content", state.context.parameters)

    def test_write_document_output_is_normalized_under_timestamp_root(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="write_document",
                arguments={
                    "destination_path": "data/outputs/job_description_raw.txt",
                    "content": "Application Engineer role\\nBuild services\\nShip systems",
                },
                reasoning="The user wants to save pasted job text into a local file.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Save this job ad as a file",
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertIsNotNone(result.proposed_tool_call)
        destination_path = result.proposed_tool_call.parameters.destination_path
        self.assertRegex(
            destination_path,
            r"data/outputs/\d{8}_\d{6}/job_description_raw\.txt$",
        )

    def test_steps_with_null_top_level_action_prepare_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                            "content": "Long pasted job description text\nwith multiple lines\nand details",
                        },
                    }
                ],
                assistant_message="Please confirm if I should proceed.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Here is the job ad text with enough detail to save",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Here is the job ad text with enough detail to save",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertIsNotNone(state.pending_tool_call)
        self.assertEqual(state.pending_tool_call.tool_name, "write_document")
        self.assertTrue(
            any(
                event.summary == "Aligned top-level planner action with the first planned step."
                for event in result.activity_events
            )
        )

    def test_runnable_clarify_plan_with_internal_placeholders_is_promoted_to_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/job_description_raw.txt",
                            "content": "About the job\nWe are Ahead Health\nSenior Product Engineer\n",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {
                            "job_folder": "{{last_job_folder}}",
                        },
                    },
                    {
                        "action": "match_cv",
                        "arguments": {
                            "job_folder": "{{last_job_folder}}",
                        },
                    },
                ],
                turn_intent="clarify",
                assistant_message=(
                    "I can process this job advertisement for you. Here's the plan:\n"
                    "1. Write the job description into a file.\n"
                    "2. Create a new job folder with the processed data.\n"
                    "3. Match the CVs with the processed job folder."
                ),
                missing_fields=["last_job_folder"],
                confirmation_required=False,
                reasoning="The user pasted a job ad and wants downstream job processing.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Please process this job ad and match the CVs.",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Please process this job ad and match the CVs.",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertTrue(result.confirmation_required)
        self.assertEqual(len(result.proposed_tool_calls), 3)
        self.assertEqual(state.pending_tool_call.tool_name, "write_document")
        self.assertEqual(state.pending_tool_calls[-1].tool_name, "match_cv")
        self.assertEqual(result.missing_fields, [])
        self.assertTrue(
            any(
                event.summary == "Promoted a runnable planner clarification to confirmation."
                for event in result.activity_events
            )
        )

    def test_working_folder_missing_field_is_dropped_for_staged_job_plan(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": f"data/outputs/{runtime.TODAY_STAMP} - Ahead Health - Senior Product Engineer/job_description_raw.txt",
                            "content": (
                                "About the job\n"
                                "We are Ahead Health\n"
                                "Senior Product Engineer\n"
                                "Build product features, maintain services, and collaborate across the stack.\n"
                            ),
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {},
                    },
                    {
                        "action": "match_cv",
                        "arguments": {},
                    },
                ],
                turn_intent="clarify",
                assistant_message="I can stage the pasted job ad and process it once you confirm.",
                missing_fields=["working_folder"],
                confirmation_required=False,
                reasoning="The user pasted a new job ad and wants downstream job processing.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Help me get a job for the following job ad: About the job We are Ahead Health...",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Help me get a job for the following job ad: About the job We are Ahead Health...",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(result.missing_fields, [])
        self.assertEqual(len(state.pending_tool_calls), 3)
        self.assertIn("working_folder", state.context.parameters)
        self.assertTrue(
            any(
                event.summary == "Dropped internal planner missing fields."
                and "working_folder" in event.detail
                for event in result.activity_events
            )
        )

    def test_staged_job_flow_autofills_working_folder_for_create_job_files(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/job_description_raw.txt",
                            "content": "About the job\nWe are Ahead Health\nSenior Product Engineer\n",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {},
                    },
                    {
                        "action": "evaluate_documents",
                        "arguments": {
                            "input_path": "{last_written_document}",
                            "instructions": "Analyze this job advertisement against the candidate profile.",
                        },
                    },
                ],
                turn_intent="confirm",
                assistant_message=(
                    "I can stage the job description, create the job folder, and analyze it once you confirm."
                ),
                missing_fields=[],
                confirmation_required=True,
                reasoning="The user pasted a job ad and wants the assistant to help with it.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Help me get this job: About the job We are Ahead Health...",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Help me get this job: About the job We are Ahead Health...",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertTrue(result.confirmation_required)
        self.assertEqual(len(state.pending_tool_calls), 3)
        write_call = state.pending_tool_calls[0]
        create_call = state.pending_tool_calls[1]
        evaluate_call = state.pending_tool_calls[2]
        working_folder = str(Path(write_call.parameters.destination_path).parent)
        self.assertEqual(create_call.tool_name, "create_job_files")
        self.assertEqual(create_call.parameters.job_folder, working_folder)
        self.assertEqual(evaluate_call.parameters.input_path, write_call.parameters.destination_path)
        self.assertEqual(state.context.parameters["working_folder"].value, working_folder)
        self.assertTrue(
            any(
                event.summary == "Filled `create_job_files.job_folder` from the staged working folder."
                for event in result.activity_events
            )
        )

    def test_copy_file_can_read_planned_earlier_output_under_shared_timestamp_root(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": (
                                f"data/outputs/{runtime.TODAY_STAMP} - Ahead Health - "
                                "Senior Product Engineer/job_description_raw.txt"
                            ),
                            "content": (
                                "About the job\n"
                                "We are Ahead Health\n"
                                "Senior Product Engineer\n"
                                "Build product features and maintain services.\n"
                            ),
                        },
                    },
                    {
                        "action": "copy_file",
                        "arguments": {
                            "source_path": (
                                f"data/outputs/{runtime.TODAY_STAMP} - Ahead Health - "
                                "Senior Product Engineer/job_description_raw.txt"
                            ),
                            "destination_path": (
                                f"data/outputs/{runtime.TODAY_STAMP} - Ahead Health - "
                                "Senior Product Engineer/job_description_copy.txt"
                            ),
                        },
                    },
                ],
                turn_intent="confirm",
                assistant_message="I can stage the job ad and copy the staged file once you confirm.",
                missing_fields=[],
                confirmation_required=True,
                reasoning="The user wants a staged copy of the pasted job ad.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Help me stage this job ad and keep a raw copy.",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Help me stage this job ad and keep a raw copy.",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(len(state.pending_tool_calls), 2)
        write_call = state.pending_tool_calls[0]
        copy_call = state.pending_tool_calls[1]
        self.assertEqual(copy_call.tool_name, "copy_file")
        self.assertEqual(copy_call.parameters.source_path, write_call.parameters.destination_path)
        self.assertRegex(
            copy_call.parameters.source_path,
            r"data/outputs/\d{8}_\d{6}/\d{8} - Ahead Health - Senior Product Engineer/job_description_raw\.txt$",
        )
        self.assertRegex(
            copy_call.parameters.destination_path,
            r"data/outputs/\d{8}_\d{6}/\d{8} - Ahead Health - Senior Product Engineer/job_description_copy\.txt$",
        )

    def test_first_step_wins_when_top_level_action_conflicts(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="create_job_files",
                arguments={"job_folder": "data/inputs/jobs/wrong-folder"},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                            "content": "Long pasted job description text\nwith multiple lines\nand details",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {"job_folder": "{{last_job_folder}}"},
                    },
                ],
                assistant_message="Please confirm if I should proceed.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Here is the job ad text with enough detail to save",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Here is the job ad text with enough detail to save",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(state.pending_tool_call.tool_name, "write_document")
        self.assertTrue(
            any(
                event.summary == "Aligned top-level planner action with the first planned step."
                and "create_job_files" in event.detail
                and "write_document" in event.detail
                for event in result.activity_events
            )
        )

    def test_multi_step_plan_is_prepared_and_uses_last_step_as_selected_workflow(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="write_document",
                arguments={
                    "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                    "content": "Company\nRole\nFull pasted text",
                },
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                            "content": "Company\nRole\nFull pasted text",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {"job_folder": "{{last_job_folder}}"},
                    },
                    {
                        "action": "match_cv",
                        "arguments": {"job_folder": "{{last_job_folder}}"},
                    },
                ],
                reasoning="The user pasted a new job ad and wants downstream job processing.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Analyze this pasted job ad and find the best CV match",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Analyze this pasted job ad and find the best CV match",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(len(result.proposed_tool_calls), 3)
        self.assertEqual(state.confirmation_state, "awaiting_confirmation")
        self.assertEqual(len(state.pending_tool_calls), 3)
        self.assertEqual(state.pending_tool_call.tool_name, "write_document")
        self.assertEqual(state.pending_tool_calls[-1].tool_name, "match_cv")
        self.assertEqual(state.context.request["selected_workflow"].value, "match_cv")
        self.assertIn("3-step plan", result.assistant_message)
        self.assertIn("job_folder", state.context.parameters)

    def test_missing_write_document_content_is_filled_from_pasted_user_message(self):
        state = runtime.create_conversation_state()
        pasted_job_ad = (
            "Company: Company Alpha\n"
            "Role: Role Alpha\n"
            "Location: City Beta\n"
            "About the job\n"
            "We are looking for an engineer to build product features, maintain services, and collaborate across the stack.\n"
            "Requirements\n"
            "Framework A, Language B, Runtime C, CI/CD, API design, and strong collaboration skills.\n"
            "Nice to have\n"
            "Analytics platform experience and startup comfort.\n"
        )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {
                            "job_folder": "data/outputs/20260412 - Company - Role",
                        },
                    },
                    {
                        "action": "evaluate_documents",
                        "arguments": {
                            "input_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                            "instructions": "Analyze this job advertisement.",
                        },
                    },
                ],
                assistant_message="Please confirm if I should proceed.",
                reasoning="The user pasted a job ad and wants it analyzed.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                pasted_job_ad,
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                pasted_job_ad,
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(len(state.pending_tool_calls), 3)
        self.assertEqual(state.pending_tool_calls[0].tool_name, "write_document")
        self.assertEqual(state.pending_tool_calls[0].parameters.content, pasted_job_ad)
        self.assertEqual(state.pending_tool_calls[-1].tool_name, "evaluate_documents")
        self.assertTrue(
            any(
                event.summary == "Filled `write_document.content` from the latest user message."
                for event in result.activity_events
            )
        )

    def test_single_brace_placeholders_and_internal_steps_are_reconciled(self):
        state = runtime.create_conversation_state()
        pasted_job_ad = (
            "Company: Company Beta\n"
            "Role: Role Beta\n"
            "Location: City Gamma\n"
            "About the job\n"
            "We are looking for an engineer to build product features, maintain services, and collaborate across the stack.\n"
            "Requirements\n"
            "Framework A, Language B, Runtime C, CI/CD, API design, and strong collaboration skills.\n"
            "Nice to have\n"
            "Analytics platform experience and startup comfort.\n"
        )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action=None,
                arguments={},
                steps=[
                    {
                        "action": "write_document",
                        "arguments": {
                            "destination_path": "data/outputs/20260412 - Company Beta - Role Beta/job_description_raw.txt",
                        },
                    },
                    {
                        "action": "create_job_files",
                        "arguments": {
                            "job_folder": "{last_job_folder}",
                        },
                    },
                    {
                        "action": "match_cv",
                        "arguments": {
                            "job_folder": "{last_job_folder}",
                        },
                    },
                ],
                assistant_message="Please confirm if this plan is acceptable.",
                reasoning="The user pasted a job ad and wants downstream job processing.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                pasted_job_ad,
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                pasted_job_ad,
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "confirm")
        self.assertEqual(len(state.pending_tool_calls), 3)
        self.assertEqual(
            state.pending_tool_calls[1].parameters.job_folder,
            state.pending_tool_calls[0].parameters.destination_path.rsplit("/", 1)[0],
        )
        self.assertEqual(
            state.pending_tool_calls[2].parameters.job_folder,
            state.pending_tool_calls[1].parameters.job_folder,
        )

    def test_missing_write_document_content_without_pasted_text_stays_in_clarification(self):
        state = runtime.create_conversation_state()
        planner_message = "Please confirm if this plan is acceptable."

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(
                    action=None,
                    arguments={},
                    steps=[
                        {
                            "action": "write_document",
                            "arguments": {
                                "destination_path": "data/outputs/20260412 - Company - Role/job_description_raw.txt",
                            },
                        },
                        {
                            "action": "create_job_files",
                            "arguments": {
                                "job_folder": "data/outputs/20260412 - Company - Role",
                            },
                        },
                    ],
                    assistant_message=planner_message,
                    reasoning="The user pasted a job ad and wants it analyzed.",
                ),
                _planner_response(
                    action=None,
                    turn_intent="respond",
                    assistant_message="There is nothing pending to confirm yet.",
                    confirmation_required=False,
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "analyze this please",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "analyze this please",
                result,
                turn_id="turn-1",
            )
            no_pending = runtime.plan_chat_turn(
                state,
                "ok",
                turn_id="turn-2",
            )

        self.assertEqual(result.turn_intent, "clarify")
        self.assertIsNone(state.pending_tool_call)
        self.assertEqual(state.pending_tool_calls, [])
        self.assertIn("document content", result.assistant_message.lower())
        self.assertNotEqual(result.assistant_message, planner_message)
        self.assertNotIn("confirm", result.assistant_message.lower())
        self.assertEqual(no_pending.turn_intent, "respond")
        self.assertIn("nothing pending", no_pending.assistant_message.lower())

    def test_low_confidence_stays_in_clarification(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="run_job_agent",
                confidence=0.2,
                reasoning="The request might be about browsing roles, but the intent is ambiguous.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "maybe look around for work",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "maybe look around for work",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "clarify")
        self.assertIsNone(state.pending_tool_call)
        self.assertIn("not confident", state.context.request["open_question"].value.lower())

    def test_malformed_arguments_are_blocked_before_confirmation(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="run_job_agent",
                arguments={"remote_only": "maybe"},
                confidence=0.93,
                reasoning="The user appears to want a job search but supplied an invalid boolean flag.",
            ),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search for jobs and set remote only to maybe",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs and set remote only to maybe",
                result,
                turn_id="turn-1",
            )

        self.assertEqual(result.turn_intent, "clarify")
        self.assertIn("failed validation", result.assistant_message)
        self.assertIsNone(state.pending_tool_call)

    def test_execution_captures_activity_and_output_folder(self):
        state = runtime.create_conversation_state()

        with patch(
            "assistant.runtime.call_llm",
            side_effect=[
                _planner_response(action="run_job_agent"),
                _planner_response(
                    action="run_job_agent",
                    turn_intent="execute",
                    assistant_message="",
                    reasoning="The user confirmed the already pending validated plan.",
                ),
            ],
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                initial,
                turn_id="turn-1",
            )
            confirmed = runtime.plan_chat_turn(
                state,
                "yes",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "yes",
                confirmed,
                turn_id="turn-2",
            )

        with patch(
            "assistant.runtime.execute_tool_call",
            return_value=iter(
                [
                    "Starting job workflow...\n",
                    "Done. Output written to: data/outputs/20260405_120000\n",
                ]
            ),
        ), patch("assistant.runtime.log_activity_event") as log_activity_event:
            execution_result = runtime.execute_pending_tool_call(
                state,
                turn_id="turn-2",
            )
            runtime.apply_execution_result(
                state,
                execution_result,
                turn_id="turn-2",
            )

        self.assertEqual(
            state.context.execution["last_output_folder"].value,
            "data/outputs/20260405_120000",
        )
        self.assertEqual(state.context.request["run_status"].value, "completed")
        self.assertIn("finished", state.messages[-1].content)
        self.assertTrue(
            any(event.raw_lines for event in execution_result.activity_events)
        )
        self.assertGreaterEqual(log_activity_event.call_count, 2)

    def test_search_web_execution_prints_results_in_chat_when_no_output_file(self):
        state = runtime.create_conversation_state()
        state.pending_tool_calls = [
            PlannedToolCall(
                tool_name="search_web",
                parameters=runtime.TOOLS["search_web"].arg_model(
                    query="Mickey Mouse",
                    max_results=3,
                ),
                request_id="req-1",
            ),
        ]
        state.pending_tool_call = state.pending_tool_calls[0]
        state.confirmation_state = "confirmed"

        with patch(
            "assistant.runtime.execute_tool_call",
            return_value=iter(
                [
                    "Starting web search workflow...\n",
                    "Query: Mickey Mouse\n",
                    "Max results: 3\n",
                    "Found 2 result(s) for: Mickey Mouse\n",
                    "1. Mickey Mouse & Friends | Official Disney Site\n",
                    "   URL: https://mickey.disney.com/\n",
                    "2. Mickey Mouse - Wikipedia\n",
                    "   URL: https://en.wikipedia.org/wiki/Mickey_Mouse\n",
                    "Workflow finished.\n",
                ]
            ),
        ), patch("assistant.runtime.log_activity_event"):
            execution_result = runtime.execute_pending_tool_call(
                state,
                turn_id="turn-1",
            )
            runtime.apply_execution_result(
                state,
                execution_result,
                turn_id="turn-1",
            )

        self.assertIn("Found 2 result(s) for: Mickey Mouse", state.messages[-1].content)
        self.assertIn("https://mickey.disney.com/", state.messages[-1].content)
        self.assertNotIn("`search_web` finished.", state.messages[-1].content)

    def test_execution_collapses_nested_output_path_to_timestamp_root(self):
        state = runtime.create_conversation_state()
        state.pending_tool_calls = [
            PlannedToolCall(
                tool_name="write_document",
                parameters=WriteDocumentArgs(
                    destination_path="/Users/z/dev/domo/Domo_AI_agent/data/outputs/20260412_120000/job/job_description_raw.txt",
                    content="Job text",
                ),
                request_id="req-1",
            ),
        ]
        state.pending_tool_call = state.pending_tool_calls[0]
        state.confirmation_state = "confirmed"

        with patch(
            "assistant.runtime.execute_tool_call",
            return_value=iter(
                ["Wrote document\n", "Output written to: data/outputs/20260412_120000/job/job_description_raw.txt\n"]
            ),
        ), patch("assistant.runtime.log_activity_event"):
            execution_result = runtime.execute_pending_tool_call(
                state,
                turn_id="turn-1",
            )
            runtime.apply_execution_result(
                state,
                execution_result,
                turn_id="turn-1",
            )

        self.assertEqual(
            state.context.execution["last_output_folder"].value,
            "data/outputs/20260412_120000",
        )
        self.assertTrue(
            re.search(
                r"Output written to `data/outputs/20260412_120000`",
                state.messages[-1].content,
            )
        )

    def test_multi_step_execution_runs_each_validated_call_in_order(self):
        state = runtime.create_conversation_state()
        state.pending_tool_calls = [
            PlannedToolCall(
                tool_name="write_document",
                parameters=WriteDocumentArgs(
                    destination_path="/Users/z/dev/domo/Domo_AI_agent/data/outputs/20260412 - Company - Role/job_description_raw.txt",
                    content="Job text",
                ),
                request_id="req-1",
            ),
            PlannedToolCall(
                tool_name="create_job_files",
                parameters=CreateJobFilesArgs(
                    job_folder="/Users/z/dev/domo/Domo_AI_agent/data/outputs/20260412 - Company - Role",
                ),
                request_id="req-2",
            ),
            PlannedToolCall(
                tool_name="match_cv",
                parameters=MatchCvArgs(
                    job_folder="/Users/z/dev/domo/Domo_AI_agent/data/outputs/20260412 - Company - Role",
                    cvs_folder="/Users/z/dev/domo/Domo_AI_agent/data/inputs/cvs",
                ),
                request_id="req-3",
            ),
        ]
        state.pending_tool_call = state.pending_tool_calls[0]
        state.confirmation_state = "confirmed"

        with patch(
            "assistant.runtime.execute_tool_call",
            side_effect=[
                iter(["Wrote document\n"]),
                iter(["Created job files\n"]),
                iter(["Matched CVs\n", "Output written to: data/outputs/20260412_120000\n"]),
            ],
        ) as execute_tool_call, patch("assistant.runtime.log_activity_event"):
            execution_result = runtime.execute_pending_tool_call(
                state,
                turn_id="turn-1",
            )
            runtime.apply_execution_result(
                state,
                execution_result,
                turn_id="turn-1",
            )

        self.assertEqual(execute_tool_call.call_count, 3)
        self.assertEqual(state.context.request["run_status"].value, "completed")
        self.assertEqual(
            state.context.execution["last_output_folder"].value,
            "data/outputs/20260412_120000",
        )
        self.assertIn("3-step plan finished", state.messages[-1].content)
        self.assertEqual(state.pending_tool_calls, [])
        self.assertIsNone(state.pending_tool_call)

    def test_switching_workflows_clears_stale_parameters(self):
        state = runtime.create_conversation_state()
        planned_call = PlannedToolCall(
            tool_name="create_job_files",
            parameters=CreateJobFilesArgs(job_folder="/tmp/company-alpha"),
            request_id="req-456",
        )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(action="run_job_agent"),
        ), patch("assistant.runtime.log_activity_event"):
            initial = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                initial,
                turn_id="turn-1",
            )

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(
                action="create_job_files",
                arguments={"job_folder": "company-alpha"},
            ),
        ), patch(
            "assistant.runtime.plan_tool_call",
            return_value=planned_call,
        ), patch("assistant.runtime.log_activity_event"):
            switched = runtime.plan_chat_turn(
                state,
                "create job files from folder 'company-alpha'",
                turn_id="turn-2",
            )
            runtime.apply_turn_result(
                state,
                "create job files from folder 'company-alpha'",
                switched,
                turn_id="turn-2",
            )

        self.assertEqual(state.context.request["selected_workflow"].value, "create_job_files")
        self.assertEqual(state.context.parameters["job_folder"].value, "/tmp/company-alpha")
        self.assertNotIn("role", state.context.parameters)
        self.assertNotIn("location", state.context.parameters)

    def test_reset_conversation_state_clears_session_state(self):
        state = runtime.create_conversation_state()
        original_session_id = state.session_id

        with patch(
            "assistant.runtime.call_llm",
            return_value=_planner_response(action="run_job_agent"),
        ), patch("assistant.runtime.log_activity_event"):
            result = runtime.plan_chat_turn(
                state,
                "Search for jobs",
                turn_id="turn-1",
            )
            runtime.apply_turn_result(
                state,
                "Search for jobs",
                result,
                turn_id="turn-1",
            )

        runtime.reset_conversation_state(state)

        self.assertNotEqual(original_session_id, state.session_id)
        self.assertEqual(state.messages, [])
        self.assertEqual(state.activity_events, [])
        self.assertEqual(state.confirmation_state, "idle")
        self.assertIsNone(state.pending_tool_call)
        self.assertEqual(state.context.request["run_status"].value, "idle")


if __name__ == "__main__":
    unittest.main()
