import unittest
from unittest.mock import patch

from assistant.config import get_config_path, get_display_path
from assistant.registry import TOOLS, WORKFLOWS


class WorkflowRegistryTests(unittest.TestCase):
    def test_registry_uses_workflow_entrypoints_for_all_actions(self):
        self.assertIs(TOOLS["run_job_agent"].executor, WORKFLOWS["run_job_agent"])
        self.assertIs(TOOLS["create_job_files"].executor, WORKFLOWS["create_job_files"])
        self.assertIs(TOOLS["match_cv"].executor, WORKFLOWS["match_cv"])
        self.assertIs(TOOLS["copy_file"].executor, WORKFLOWS["copy_file"])
        self.assertIs(TOOLS["write_document"].executor, WORKFLOWS["write_document"])
        self.assertIs(TOOLS["read_documents"].executor, WORKFLOWS["read_documents"])
        self.assertIs(
            TOOLS["summarize_documents"].executor,
            WORKFLOWS["summarize_documents"],
        )
        self.assertIs(
            TOOLS["evaluate_documents"].executor,
            WORKFLOWS["evaluate_documents"],
        )

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

    def test_run_job_agent_workflow_surfaces_search_overrides(self):
        with patch(
            "workflows.run_job_agent_workflow.run_job_agent",
            return_value=iter(["tool chunk\n"]),
        ):
            chunks = list(
                TOOLS["run_job_agent"].executor(
                    role="Application Engineer",
                    location="City Alpha",
                    ignore_location=True,
                    remote_only=False,
                )
            )

        self.assertEqual(chunks[0], "Starting job workflow...\n")
        self.assertEqual(
            chunks[1],
            f"Running default online job search from {get_display_path(get_config_path())}\n",
        )
        self.assertEqual(
            chunks[2],
            "Search overrides: role=Application Engineer, location=City Alpha, ignore_location=True, remote_only=False\n",
        )
        self.assertEqual(chunks[3], "tool chunk\n")
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

    def test_create_job_files_workflow_has_consistent_wrapper_output(self):
        with patch(
            "workflows.create_job_files_workflow.create_job_files",
            return_value=iter(["tool chunk\n"]),
        ):
            chunks = list(
                TOOLS["create_job_files"].executor(
                    job_folder="/tmp/job-folder",
                )
            )

        self.assertEqual(chunks[0], "Starting create_job_files workflow...\n")
        self.assertEqual(chunks[1], "Resolved job input: /tmp/job-folder\n")
        self.assertEqual(chunks[2], "tool chunk\n")
        self.assertEqual(chunks[-1], "Workflow finished.\n")


if __name__ == "__main__":
    unittest.main()
