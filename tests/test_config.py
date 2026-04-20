import unittest

from assistant.config import (
    get_config_path,
    get_display_path,
    get_job_search_config,
    get_ollama_config,
    get_paths,
    get_prompt_override_fields,
    is_debug_enabled,
)
from tools.job.discover_jobs import load_inputs_config


class ConfigTests(unittest.TestCase):
    def test_main_config_file_is_config_yaml(self):
        self.assertEqual(get_config_path().name, "config.yaml")
        self.assertEqual(get_display_path(get_config_path()), "config.yaml")

    def test_paths_are_loaded_from_shared_config(self):
        paths = get_paths()

        self.assertTrue(str(paths["inputs_root"]).endswith("data/inputs"))
        self.assertTrue(str(paths["jobs_root"]).endswith("data/inputs/jobs"))
        self.assertTrue(str(paths["documents_root"]).endswith("data/inputs/documents"))
        self.assertTrue(str(paths["outputs_root"]).endswith("data/outputs"))
        self.assertTrue(str(paths["cvs_root"]).endswith("data/inputs/cvs"))

    def test_job_search_loader_uses_shared_config(self):
        self.assertEqual(load_inputs_config(), get_job_search_config())
        self.assertFalse(get_job_search_config()["ignore_location"])
        self.assertFalse(get_job_search_config()["remote_only"])

    def test_debug_is_general_and_disabled_by_default(self):
        self.assertFalse(is_debug_enabled())

    def test_ollama_settings_are_loaded_from_shared_config(self):
        ollama = get_ollama_config()

        self.assertEqual(ollama["base_url"], "http://localhost:11434")
        self.assertEqual(ollama["model"], "mistral")

    def test_prompt_override_fields_are_explicit_in_config(self):
        self.assertEqual(
            get_prompt_override_fields("run_job_agent"),
            ["role", "location", "ignore_location", "remote_only"],
        )
        self.assertEqual(get_prompt_override_fields("create_job_files"), [])
        self.assertEqual(get_prompt_override_fields("match_cv"), [])


if __name__ == "__main__":
    unittest.main()
