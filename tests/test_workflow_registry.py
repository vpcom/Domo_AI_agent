import unittest
from unittest.mock import patch

from assistant.config import get_config_path, get_display_path
from assistant.registry import TOOLS, WORKFLOWS


class WorkflowRegistryTests(unittest.TestCase):
    def test_registry_uses_workflow_entrypoints_for_both_actions(self):
        self.assertIs(TOOLS["run_job_agent"].executor, WORKFLOWS["run_job_agent"])
        self.assertIs(TOOLS["match_cv"].executor, WORKFLOWS["match_cv"])

    def test_run_job_agent_workflow_has_consistent_wrapper_output(self):
        with patch(
            "workflows.run_job_agent_workflow.run_job_agent",
            return_value=iter(["tool chunk\n"]),
        ):
            chunks = list(TOOLS["run_job_agent"].executor(folder_path=None))

        self.assertEqual(chunks[0], "Starting job workflow...\n")
        self.assertEqual(
            chunks[1],
            f"Running default online job search from {get_display_path(get_config_path())}\n",
        )
        self.assertEqual(chunks[2], "tool chunk\n")
        self.assertEqual(chunks[-1], "Workflow finished.\n")

    def test_match_cv_workflow_has_consistent_wrapper_output(self):
        with patch(
            "workflows.match_cv_workflow.match_cv",
            return_value=iter(["tool chunk\n"]),
        ):
            chunks = list(
                TOOLS["match_cv"].executor(
                    job_folder="/tmp/job-folder",
                    cvs_folder="/tmp/cvs-folder",
                )
            )

        self.assertEqual(chunks[0], "Starting CV matching workflow...\n")
        self.assertEqual(chunks[1], "Resolved job input: /tmp/job-folder\n")
        self.assertEqual(chunks[2], "Resolved CV input: /tmp/cvs-folder\n")
        self.assertEqual(chunks[3], "tool chunk\n")
        self.assertEqual(chunks[-1], "Workflow finished.\n")


if __name__ == "__main__":
    unittest.main()
