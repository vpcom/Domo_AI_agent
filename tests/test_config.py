import unittest

from assistant.config import (
    get_config_path,
    get_display_path,
    get_job_search_config,
    get_ollama_config,
    get_paths,
    is_debug_enabled,
)
from tools.job.discover_jobs import load_inputs_config


class ConfigTests(unittest.TestCase):
    def test_main_config_file_is_domo_config_yaml(self):
        self.assertEqual(get_config_path().name, "domo_config.yaml")
        self.assertEqual(get_display_path(get_config_path()), "domo_config.yaml")

    def test_paths_are_loaded_from_shared_config(self):
        paths = get_paths()

        self.assertTrue(str(paths["jobs_root"]).endswith("data/jobs"))
        self.assertTrue(str(paths["outputs_root"]).endswith("data/outputs"))
        self.assertTrue(str(paths["cvs_root"]).endswith("data/cvs"))

    def test_job_search_loader_uses_shared_config(self):
        self.assertEqual(load_inputs_config(), get_job_search_config())

    def test_debug_is_general_and_disabled_by_default(self):
        self.assertFalse(is_debug_enabled())

    def test_ollama_settings_are_loaded_from_shared_config(self):
        ollama = get_ollama_config()

        self.assertEqual(ollama["base_url"], "http://localhost:11434")
        self.assertEqual(ollama["model"], "mistral")


if __name__ == "__main__":
    unittest.main()
