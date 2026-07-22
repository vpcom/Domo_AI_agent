import unittest

from assistant.registry import COMMANDS, LLM_TASKS, READ_TOOLS, TOOLS, TRANSFORM_TOOLS, WRITE_TOOLS


class WorkflowRegistryTests(unittest.TestCase):
    def test_tool_registry_exposes_strict_capabilities(self):
        self.assertIn("search_web", TOOLS)
        self.assertIn("write_document", TOOLS)
        self.assertIn("read_documents", TOOLS)
        self.assertIn("inspect_path", TOOLS)
        self.assertIn("list_directory", TOOLS)
        self.assertIn("read_text_file", TOOLS)
        self.assertIn("read_json_file", TOOLS)
        self.assertIn("resolve_job_folder_hint", TOOLS)
        self.assertIn("clean_job_description", TOOLS)
        self.assertIn("write_json_file", TOOLS)
        self.assertIn("write_search_results", TOOLS)
        self.assertIn("write_generated_documents", TOOLS)
        self.assertNotIn("summarize_documents", TOOLS)
        self.assertNotIn("evaluate_documents", TOOLS)
        self.assertNotIn("match_cv", TOOLS)

        for spec in TOOLS.values():
            self.assertTrue(callable(spec.function))
            self.assertTrue(spec.allowed)
            self.assertTrue(spec.description)
            self.assertIsNotNone(spec.input_model)
            self.assertFalse(spec.account_access)
            self.assertTrue(spec.group)
            self.assertTrue(spec.kind)
            self.assertTrue(spec.approval)

    def test_tool_registry_groups_are_exposed(self):
        self.assertIn("search_web", READ_TOOLS)
        self.assertIn("clean_job_description", TRANSFORM_TOOLS)
        self.assertIn("write_document", WRITE_TOOLS)
        self.assertIn("run_job_agent", COMMANDS)

    def test_llm_task_registry_exposes_explicit_llm_steps(self):
        self.assertIn("answer_question", LLM_TASKS)
        self.assertIn("summarize_text", LLM_TASKS)
        self.assertIn("evaluate_text", LLM_TASKS)
        self.assertIn("generate_document_set", LLM_TASKS)
        self.assertIn("rank_cvs", LLM_TASKS)

        for spec in LLM_TASKS.values():
            self.assertTrue(callable(spec.function))
            self.assertTrue(spec.allowed)
            self.assertTrue(spec.description)
            self.assertIsNotNone(spec.input_model)


if __name__ == "__main__":
    unittest.main()
