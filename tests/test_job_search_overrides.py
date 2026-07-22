import os
import unittest
from unittest.mock import patch

from tools.job.discover_jobs import discover_jobs_from_config, load_inputs_config
from tools.job.run_job_agent import run_job_agent


class JobSearchOverrideTests(unittest.TestCase):
    def test_load_inputs_config_applies_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "DOMO_JOB_SEARCH_OVERRIDES_JSON": (
                    '{"role":"Application Engineer","ignore_location":true,"remote_only":false}'
                )
            },
            clear=False,
        ):
            config = load_inputs_config()

        self.assertEqual(config["role"], "Application Engineer")
        self.assertTrue(config["ignore_location"])
        self.assertFalse(config["remote_only"])

    def test_discover_jobs_from_config_passes_location_flags(self):
        config = {
            "role": "Application Engineer",
            "location": "City Alpha",
            "ignore_location": True,
            "remote_only": False,
            "sources": ["greenhouse"],
            "max_results_per_source": 5,
            "max_jobs": 2,
            "max_company_attempts_per_source": 3,
            "companies": {"greenhouse": ["company-alpha"]},
        }

        with patch("tools.job.discover_jobs.discover_jobs", return_value=[]) as discover_jobs:
            discover_jobs_from_config(config)

        args = discover_jobs.call_args.args
        self.assertEqual(args[0], "application engineer")
        self.assertEqual(args[1], "city alpha")
        self.assertTrue(args[2])
        self.assertFalse(args[3])
        self.assertEqual(args[4], ["greenhouse"])

    def test_run_job_agent_passes_overrides_and_returns_structured_output(self):
        fake_process = type(
            "FakeProcess",
            (),
            {
                "stdout": iter(["ok\n", "Done. Output written to: data/outputs/20260426_120000\n"]),
                "wait": lambda self, timeout=None: 0,
                "returncode": 0,
            },
        )()

        with patch("tools.job.run_job_agent.subprocess.Popen", return_value=fake_process) as popen:
            result = run_job_agent(
                role="Application Engineer",
                location="City Alpha",
                ignore_location=True,
                remote_only=False,
            )

        env = popen.call_args.kwargs["env"]
        self.assertIn("DOMO_JOB_SEARCH_OVERRIDES_JSON", env)
        self.assertIn('"role": "Application Engineer"', env["DOMO_JOB_SEARCH_OVERRIDES_JSON"])
        self.assertEqual(
            result["result"]["output_root"],
            "data/outputs/20260426_120000",
        )
        self.assertEqual(
            result["metadata"]["artifacts"][0]["path"],
            "data/outputs/20260426_120000",
        )


if __name__ == "__main__":
    unittest.main()
