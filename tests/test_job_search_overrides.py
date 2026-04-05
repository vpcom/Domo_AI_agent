import os
import unittest
from unittest.mock import patch

from assistant.policy import plan_tool_call
from assistant.schemas import CreateJobFilesArgs, RunJobAgentArgs
from tools.job.discover_jobs import load_inputs_config, select_best_jobs
from tools.job.run_job_agent import run_job_agent


class JobSearchOverrideTests(unittest.TestCase):
    def test_load_inputs_config_applies_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "DOMO_JOB_SEARCH_OVERRIDES_JSON": (
                    '{"role":"Backend Engineer","ignore_location":true,"remote_only":false}'
                )
            },
            clear=False,
        ):
            config = load_inputs_config()

        self.assertEqual(config["role"], "Backend Engineer")
        self.assertTrue(config["ignore_location"])
        self.assertFalse(config["remote_only"])

    def test_select_best_jobs_respects_strict_location_by_default(self):
        jobs = [
            {"title": "Backend Engineer", "location": "Zurich", "company": "A", "source": "x"},
            {"title": "Backend Engineer", "location": "Berlin", "company": "B", "source": "x"},
        ]

        selected = select_best_jobs(jobs, "Backend Engineer", "Zurich", 5)

        self.assertEqual([job["company"] for job in selected], ["A"])
        self.assertEqual(selected[0]["search_strategy"], "strict")

    def test_select_best_jobs_can_ignore_location_when_configured(self):
        jobs = [
            {"title": "Backend Engineer", "location": "Berlin", "company": "B", "source": "x"},
        ]

        selected = select_best_jobs(
            jobs,
            "Backend Engineer",
            "Zurich",
            5,
            ignore_location=True,
        )

        self.assertEqual([job["company"] for job in selected], ["B"])
        self.assertEqual(selected[0]["search_strategy"], "ignore_location")

    def test_select_best_jobs_remote_only_still_requires_role_match(self):
        jobs = [
            {"title": "Backend Engineer", "location": "Remote - Europe", "company": "A", "source": "x"},
            {"title": "Designer", "location": "Remote - Europe", "company": "B", "source": "x"},
            {"title": "Backend Engineer", "location": "Berlin", "company": "C", "source": "x"},
        ]

        selected = select_best_jobs(
            jobs,
            "Backend Engineer",
            "Zurich",
            5,
            remote_only=True,
        )

        self.assertEqual([job["company"] for job in selected], ["A"])
        self.assertEqual(selected[0]["search_strategy"], "remote_only")

    def test_plan_tool_call_preserves_online_search_overrides(self):
        planned = plan_tool_call(
            "run_job_agent",
            RunJobAgentArgs(
                role="Backend Engineer",
                location="Amsterdam",
                ignore_location=True,
                remote_only=False,
            ),
            "Search for backend engineer roles in Amsterdam but ignore location",
            "req-1",
        )

        self.assertEqual(planned.parameters.role, "Backend Engineer")
        self.assertEqual(planned.parameters.location, "Amsterdam")
        self.assertTrue(planned.parameters.ignore_location)
        self.assertFalse(planned.parameters.remote_only)

    def test_plan_tool_call_rejects_search_overrides_for_local_folder_mode(self):
        with self.assertRaises(ValueError):
            plan_tool_call(
                "run_job_agent",
                RunJobAgentArgs(
                    folder_path="data/jobs/job_search",
                    remote_only=True,
                ),
                "Run the local job folder and make it remote only",
                "req-2",
            )

    def test_plan_tool_call_accepts_create_job_files_for_local_job_folder(self):
        planned = plan_tool_call(
            "create_job_files",
            CreateJobFilesArgs(job_folder="data/jobs"),
            "Process the local job folder and create the job files",
            "req-3",
        )

        self.assertTrue(planned.parameters.job_folder.endswith("data/jobs"))

    def test_run_job_agent_passes_overrides_via_subprocess_env(self):
        fake_process = type(
            "FakeProcess",
            (),
            {
                "stdout": iter(["ok\n"]),
                "wait": lambda self, timeout=None: 0,
                "returncode": 0,
            },
        )()

        with patch("tools.job.run_job_agent.subprocess.Popen", return_value=fake_process) as popen:
            list(
                run_job_agent(
                    role="Backend Engineer",
                    location="Amsterdam",
                    ignore_location=True,
                    remote_only=False,
                )
            )

        env = popen.call_args.kwargs["env"]
        self.assertIn("DOMO_JOB_SEARCH_OVERRIDES_JSON", env)
        self.assertIn('"role": "Backend Engineer"', env["DOMO_JOB_SEARCH_OVERRIDES_JSON"])
        self.assertIn('"ignore_location": true', env["DOMO_JOB_SEARCH_OVERRIDES_JSON"])


if __name__ == "__main__":
    unittest.main()
