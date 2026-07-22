from pathlib import Path
import unittest
from unittest.mock import patch

from assistant.planner import plan_goal, validate_plan_draft
from assistant.schemas import PlanDraft, PlanStepDraft


class PlannerValidationTests(unittest.TestCase):
    def test_rejects_unknown_tool(self):
        draft = PlanDraft(
            normalized_goal="do something",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Unknown",
                    type="tool",
                    tool_name="not_a_tool",
                    inputs={},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "Unknown or disallowed tool"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_unknown_llm_task(self):
        draft = PlanDraft(
            normalized_goal="do something",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Unknown",
                    type="llm",
                    tool_name="not_a_task",
                    inputs={},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "Unknown or disallowed llm task"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_missing_required_inputs(self):
        draft = PlanDraft(
            normalized_goal="write a file",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Write",
                    type="tool",
                    tool_name="write_document",
                    inputs={"destination_path": "data/outputs/note.txt"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing required inputs"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_rank_cvs_string_document_inputs_with_references(self):
        draft = PlanDraft(
            normalized_goal="rank films",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for films",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "Bruce Lee filmography"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Rank the collected films",
                    type="llm",
                    tool_name="rank_cvs",
                    inputs={
                        "job_documents": "Bruce Lee filmography",
                        "cv_documents": "@step:0.output.result.results",
                    },
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "job_documents.*must be a list"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_unknown_result_field_reference_for_known_step(self):
        draft = PlanDraft(
            normalized_goal="write film count",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for films",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "Bruce Lee films"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Extract the film count",
                    type="llm",
                    tool_name="summarize_text",
                    inputs={
                        "documents": "@step:0.output.result.results",
                        "instructions": "Extract the number of films.",
                    },
                ),
                PlanStepDraft(
                    step_id=2,
                    description="Write the number of films into a new output file",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/bruce-lee-films.txt",
                        "content": "@step:1.output.result.content",
                    },
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "summarize_text.*summary"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_answer_reference_after_evaluate_text(self):
        draft = PlanDraft(
            normalized_goal="identify current president",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for the current U.S. President",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "current us president"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Evaluate search results",
                    type="llm",
                    tool_name="evaluate_text",
                    inputs={
                        "documents": "@step:0.output.result.results",
                        "instructions": "Identify the current U.S. President.",
                    },
                ),
                PlanStepDraft(
                    step_id=2,
                    description="Write the identified President to an output file",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/president.txt",
                        "content": "@step:1.output.result.answer",
                    },
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "evaluate_text.*report"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_wildcard_step_reference(self):
        draft = PlanDraft(
            normalized_goal="identify current president",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for the current U.S. President",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "who is the current us president"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Provide the President from the search results",
                    type="llm",
                    tool_name="answer_question",
                    inputs={"question": "@step:0.output.result.results[*].title"},
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "list indexing or wildcards"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_accepts_search_results_as_summarize_text_documents(self):
        draft = PlanDraft(
            normalized_goal="identify current president",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for the current U.S. President",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "who is the current us president"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Identify the President from the search results",
                    type="llm",
                    tool_name="summarize_text",
                    inputs={
                        "documents": "@step:0.output.result.results",
                        "instructions": "Identify the current U.S. President by name.",
                    },
                ),
            ],
        )

        validated = validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

        self.assertEqual(
            validated[1].inputs["documents"],
            "@step:0.output.result.results",
        )

    def test_accepts_generated_document_set_multi_file_plan(self):
        draft = PlanDraft(
            normalized_goal="write one summary file per Bruce Lee movie",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Search for Bruce Lee movies",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "best Bruce Lee movies"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Generate one summary document per movie",
                    type="llm",
                    tool_name="generate_document_set",
                    inputs={
                        "source_documents": "@step:0.output.result.results",
                        "instructions": "Create 5 text documents, one per Bruce Lee movie.",
                    },
                ),
                PlanStepDraft(
                    step_id=2,
                    description="Write the generated summary files",
                    type="tool",
                    tool_name="write_generated_documents",
                    inputs={
                        "output_dir": "data/outputs/bruce_lee_movie_summaries",
                        "documents": "@step:1.output.result.documents",
                    },
                ),
            ],
        )

        validated = validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

        self.assertEqual(validated[1].tool_name, "generate_document_set")
        self.assertEqual(validated[2].tool_name, "write_generated_documents")
        self.assertEqual(
            validated[2].inputs["output_dir"],
            "/tmp/domo-output/bruce_lee_movie_summaries",
        )
        self.assertEqual(
            validated[2].inputs["documents"],
            "@step:1.output.result.documents",
        )

    def test_rejects_search_query_from_answer_question_step(self):
        draft = PlanDraft(
            normalized_goal="search web",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Parse user's search query",
                    type="llm",
                    tool_name="answer_question",
                    inputs={"question": "@goal:user_input"},
                ),
                PlanStepDraft(
                    step_id=1,
                    description="Search web using the parsed query",
                    type="tool",
                    tool_name="search_web",
                    inputs={"query": "@step:0.output.result.text"},
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "search_web.query.*answer_question"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_legacy_placeholders(self):
        draft = PlanDraft(
            normalized_goal="copy previous output",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Write",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/note.txt",
                        "content": "{{bad}}",
                    },
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "Legacy placeholder syntax"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_forward_step_references(self):
        draft = PlanDraft(
            normalized_goal="summarize docs",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Summarize future result",
                    type="llm",
                    tool_name="summarize_text",
                    inputs={"documents": "@step:0.output.result.documents"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "earlier step"):
            validate_plan_draft(draft, output_root=Path("/tmp/domo-output"))

    def test_rejects_local_file_read_when_job_text_is_embedded_in_prompt(self):
        draft = PlanDraft(
            normalized_goal="create job search document",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Read the job description",
                    type="tool",
                    tool_name="read_text_file",
                    inputs={"path": "data/inputs/jobs/Legartis_Frontend_Developer.txt"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "already contains source text"):
            validate_plan_draft(
                draft,
                output_root=Path("/tmp/domo-output"),
                user_input=(
                    "Create a job search document from this job description:\n"
                    "Frontend Developer at Legartis\n"
                    "Build UI components and work closely with product and design."
                ),
            )

    def test_normalizes_output_paths_under_shared_timestamp_root(self):
        draft = PlanDraft(
            normalized_goal="write a file",
            plan=[
                PlanStepDraft(
                    step_id=0,
                    description="Write",
                    type="tool",
                    tool_name="write_document",
                    inputs={
                        "destination_path": "data/outputs/job/summary.txt",
                        "content": "hello",
                    },
                )
            ],
        )

        validated = validate_plan_draft(
            draft,
            output_root=Path("/tmp/domo-output"),
        )

        self.assertEqual(
            validated[0].inputs["destination_path"],
            "/tmp/domo-output/job/summary.txt",
        )

    def test_plan_goal_retries_after_invalid_json_response(self):
        valid_response = """
        {
          "normalized_goal": "answer directly",
          "plan": [
            {
              "step_id": 0,
              "description": "Answer the question",
              "type": "llm",
              "tool_name": "answer_question",
              "inputs": {
                "question": "@goal:user_input"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            side_effect=["not json", valid_response],
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, output_root, trace = plan_goal("What is Python?")

        self.assertEqual(draft.normalized_goal, "answer directly")
        self.assertEqual(draft.plan[0].tool_name, "answer_question")
        self.assertEqual(output_root, Path("/tmp/domo-output"))
        self.assertTrue(any(item["message"] == "Planner prompt prepared." for item in trace))
        self.assertTrue(any(item["message"] == "Planner raw response received." for item in trace))
        self.assertTrue(any(item["message"] == "Planner validation failed." for item in trace))

    def test_plan_goal_normalizes_malformed_tool_steps(self):
        malformed_response = """
        {
          "normalized_goal": "search and save results",
          "confidence": 0.92,
          "plan": [
            {
              "step_id": 0,
              "description": "Search the web",
              "type": "tool",
              "tool_name": "search_web",
              "inputs": {
                "query": "@goal:user_input"
              }
            },
            {
              "step_id": 1,
              "description": "Write the search results",
              "type": "write_search_results_as_tool",
              "inputs": {
                "destination_path": "data/outputs/search/frontend_developer.md",
                "query": "@step:0.output.result.query",
                "results": "@step:0.output.result.results"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            return_value=malformed_response,
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, output_root, _trace = plan_goal("Find frontend developer jobs")

        self.assertEqual(draft.normalized_goal, "search and save results")
        self.assertEqual(draft.plan[1].type, "tool")
        self.assertEqual(draft.plan[1].tool_name, "write_search_results")
        self.assertEqual(
            draft.plan[1].inputs["destination_path"],
            "/tmp/domo-output/search/frontend_developer.md",
        )
        self.assertEqual(
            draft.plan[1].inputs["query"],
            "@step:0.output.result.query",
        )
        self.assertEqual(
            draft.plan[1].inputs["results"],
            "@step:0.output.result.results",
        )
        self.assertEqual(output_root, Path("/tmp/domo-output"))

    def test_plan_goal_normalizes_llm_task_when_type_is_wrong(self):
        malformed_response = """
        {
          "normalized_goal": "summarize the documents",
          "confidence": 0.75,
          "plan": [
            {
              "step_id": 0,
              "description": "Read the documents",
              "type": "tool",
              "tool_name": "read_documents",
              "inputs": {
                "input_path": "data/inputs/documents"
              }
            },
            {
              "step_id": 1,
              "description": "Summarize the documents",
              "type": "tool",
              "tool_name": "summarize_text",
              "inputs": {
                "documents": "@step:0.output.result.documents"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            return_value=malformed_response,
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ), patch(
            "assistant.planner.validate_and_normalize_tool_inputs",
            side_effect=lambda tool_name, inputs, **kwargs: dict(inputs),
        ):
            draft, _output_root, _trace = plan_goal("Summarize these documents")

        self.assertEqual(draft.plan[1].tool_name, "summarize_text")
        self.assertEqual(draft.plan[1].type, "llm")
        self.assertEqual(
            draft.plan[1].inputs["documents"],
            "@step:0.output.result.documents",
        )

    def test_plan_goal_infers_missing_summarize_text_tool_name(self):
        malformed_response = """
        {
          "normalized_goal": "Find and summarize 5 top Bruce Lee movies",
          "confidence": 0.7,
          "plan": [
            {
              "step_id": 0,
              "description": "Search the web for the best Bruce Lee movies",
              "type": "tool",
              "tool_name": "search_web",
              "inputs": {
                "query": "best bruce lee movies"
              }
            },
            {
              "step_id": 1,
              "description": "Summarize the search results",
              "type": "llm",
              "inputs": {
                "documents": "@step:0.output.result.results",
                "instructions": "List and summarize the top 5 Bruce Lee movies."
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            return_value=malformed_response,
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, _output_root, _trace = plan_goal("Find 5 top Bruce Lee movies")

        self.assertEqual(draft.plan[1].type, "llm")
        self.assertEqual(draft.plan[1].tool_name, "summarize_text")

    def test_plan_goal_repairs_incomplete_generated_summary_write(self):
        bad_response = """
        {
          "normalized_goal": "Find and summarize 5 top Bruce Lee movies",
          "confidence": 0.7,
          "plan": [
            {
              "step_id": 0,
              "description": "Search the web for the best Bruce Lee movies",
              "type": "tool",
              "tool_name": "search_web",
              "inputs": {
                "query": "best bruce lee movies"
              }
            },
            {
              "step_id": 1,
              "description": "Summarize the search results",
              "type": "llm",
              "inputs": {
                "documents": "@step:0.output.result.results",
                "instructions": "List and summarize the top 5 Bruce Lee movies."
              }
            },
            {
              "step_id": 2,
              "description": "Write a summary for each movie in separate text files",
              "type": "tool",
              "tool_name": "write_document"
            }
          ]
        }
        """
        good_response = """
        {
          "normalized_goal": "Find and summarize 5 top Bruce Lee movies",
          "confidence": 0.75,
          "plan": [
            {
              "step_id": 0,
              "description": "Search the web for the best Bruce Lee movies",
              "type": "tool",
              "tool_name": "search_web",
              "inputs": {
                "query": "best bruce lee movies"
              }
            },
            {
              "step_id": 1,
              "description": "Summarize the search results",
              "type": "llm",
              "tool_name": "summarize_text",
              "inputs": {
                "documents": "@step:0.output.result.results",
                "instructions": "List and summarize the top 5 Bruce Lee movies."
              }
            },
            {
              "step_id": 2,
              "description": "Write the summarized results",
              "type": "tool",
              "tool_name": "write_document",
              "inputs": {
                "destination_path": "data/outputs/bruce_lee_movie_summaries.txt",
                "content": "@step:1.output.result.summary"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            side_effect=[bad_response, good_response],
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, _output_root, trace = plan_goal(
                "find on internet the 5 best bruce lee movies and write a summary of each in text files"
            )

        self.assertEqual(draft.plan[1].tool_name, "summarize_text")
        self.assertEqual(draft.plan[2].tool_name, "write_document")
        self.assertEqual(
            draft.plan[2].inputs["content"],
            "@step:1.output.result.summary",
        )
        self.assertTrue(any(item["message"] == "Planner validation failed." for item in trace))

    def test_plan_goal_retries_when_model_uses_fake_read_step_for_prompt_text(self):
        bad_response = """
        {
          "normalized_goal": "create job search document for the given job ad",
          "confidence": 0.9,
          "plan": [
            {
              "step_id": 0,
              "description": "Read the job ad",
              "type": "tool",
              "tool_name": "read_text",
              "inputs": {
                "text": "@goal:user_input"
              }
            },
            {
              "step_id": 1,
              "description": "Clean the job ad text",
              "type": "tool",
              "tool_name": "clean_job_description",
              "inputs": {
                "raw_job_text": "@step:0.output.result.text"
              }
            },
            {
              "step_id": 2,
              "description": "Build the application notes text",
              "type": "tool",
              "tool_name": "build_application_notes_from_job_description",
              "inputs": {
                "cleaned_job_description": "@step:1.output.result.text"
              }
            },
            {
              "step_id": 3,
              "description": "Write the generated note",
              "type": "tool",
              "tool_name": "write_document",
              "inputs": {
                "destination_path": "data/outputs/job_search_document.md",
                "content": "@step:2.output.result.text"
              }
            }
          ]
        }
        """
        good_response = """
        {
          "normalized_goal": "create job search document for the given job ad",
          "confidence": 0.82,
          "plan": [
            {
              "step_id": 0,
              "description": "Clean the pasted job ad text",
              "type": "tool",
              "tool_name": "clean_job_description",
              "inputs": {
                "raw_job_text": "@goal:user_input"
              }
            },
            {
              "step_id": 1,
              "description": "Build the application notes text",
              "type": "tool",
              "tool_name": "build_application_notes_from_job_description",
              "inputs": {
                "cleaned_job_text": "@step:0.output.result.cleaned_text"
              }
            },
            {
              "step_id": 2,
              "description": "Write the generated note",
              "type": "tool",
              "tool_name": "write_document",
              "inputs": {
                "destination_path": "data/outputs/job_search_document.md",
                "content": "@step:1.output.result.info"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            side_effect=[bad_response, good_response],
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, output_root, trace = plan_goal(
                "Create a job search document from this job description:\n"
                "Frontend Developer at Legartis\n"
                "Build UI components and work closely with product and design."
            )

        self.assertEqual(draft.plan[0].tool_name, "clean_job_description")
        self.assertEqual(draft.plan[0].inputs["raw_job_text"], "@goal:user_input")
        self.assertEqual(
            draft.plan[1].inputs["cleaned_job_text"],
            "@step:0.output.result.cleaned_text",
        )
        self.assertEqual(
            draft.plan[2].inputs["content"],
            "@step:1.output.result.info",
        )
        self.assertEqual(output_root, Path("/tmp/domo-output"))
        self.assertTrue(any(item["message"] == "Planner validation failed." for item in trace))

    def test_plan_goal_retries_when_model_invents_input_file_for_prompt_content(self):
        bad_response = """
        {
          "normalized_goal": "create job search document",
          "confidence": 0.8,
          "plan": [
            {
              "step_id": 0,
              "description": "Read the job description",
              "type": "tool",
              "tool_name": "read_text_file",
              "inputs": {
                "path": "data/inputs/jobs/Legartis_Frontend_Developer.txt"
              }
            }
          ]
        }
        """
        good_response = """
        {
          "normalized_goal": "create job search document",
          "confidence": 0.72,
          "plan": [
            {
              "step_id": 0,
              "description": "Clean the job description from the prompt",
              "type": "tool",
              "tool_name": "clean_job_description",
              "inputs": {
                "raw_job_text": "@goal:user_input"
              }
            },
            {
              "step_id": 1,
              "description": "Build application notes",
              "type": "tool",
              "tool_name": "build_application_notes_from_job_description",
              "inputs": {
                "cleaned_job_text": "@step:0.output.result.cleaned_text"
              }
            }
          ]
        }
        """

        with patch(
            "assistant.planner.call_llm",
            side_effect=[bad_response, good_response],
        ), patch(
            "assistant.planner.build_timestamped_output_root",
            return_value=Path("/tmp/domo-output"),
        ):
            draft, output_root, trace = plan_goal(
                "Create a job search document from this job description:\n"
                "Frontend Developer at Legartis\n"
                "Build UI components and work closely with product and design."
            )

        self.assertEqual(draft.plan[0].tool_name, "clean_job_description")
        self.assertEqual(draft.plan[0].inputs["raw_job_text"], "@goal:user_input")
        self.assertEqual(
            draft.plan[1].tool_name,
            "build_application_notes_from_job_description",
        )
        self.assertTrue(any(item["message"] == "Planner validation failed." for item in trace))
        self.assertEqual(output_root, Path("/tmp/domo-output"))


if __name__ == "__main__":
    unittest.main()
