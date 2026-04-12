import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.job.clean_job_description import clean_job_description
from tools.job.discover_jobs import save_discovered_job
from tools.job.text_normalization import normalize_job_posting_text


HTML_SAMPLE = (
    "&lt;p&gt;Company Alpha is recognized on "
    "&lt;a href=&quot;https://example.com/awards/platform-100&quot; target=&quot;_blank&quot;&gt;"
    "Platform 100 List&lt;/a&gt; and received the 2024 "
    "&lt;a href=&quot;https://example.com/awards/product-breakthrough&quot;&gt;"
    "Product Breakthrough Award&lt;/a&gt;.&lt;/p&gt;"
    "&lt;ul&gt;&lt;li&gt;Build backend systems&lt;/li&gt;&lt;li&gt;Ship product features&lt;/li&gt;&lt;/ul&gt;"
)


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_job_posting_text_decodes_entities_and_strips_tags(self):
        normalized = normalize_job_posting_text(HTML_SAMPLE)

        self.assertNotIn("&lt;", normalized)
        self.assertNotIn("&quot;", normalized)
        self.assertNotIn("<a", normalized)
        self.assertIn("Platform 100 List", normalized)
        self.assertIn("Product Breakthrough Award", normalized)
        self.assertIn("- Build backend systems", normalized)
        self.assertIn("- Ship product features", normalized)

    def test_save_discovered_job_writes_normalized_text_to_both_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_file = root / "job_description_raw.txt"
            metadata_file = root / "job_metadata.json"
            job = {
                "company": "Company Alpha",
                "title": "Application Engineer II",
                "location": "City Alpha",
                "source": "greenhouse",
                "description": HTML_SAMPLE,
            }

            save_discovered_job(raw_file, metadata_file, job)

            raw_contents = raw_file.read_text(encoding="utf-8")
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

            self.assertNotIn("&lt;", raw_contents)
            self.assertNotIn("&quot;", raw_contents)
            self.assertEqual(metadata["description"], raw_contents)

    def test_clean_job_description_normalizes_before_and_after_model_call(self):
        with patch(
            "tools.job.clean_job_description.call_llm",
            return_value="&lt;p&gt;Role Summary&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Build APIs&lt;/li&gt;&lt;/ul&gt;",
        ):
            cleaned = clean_job_description(HTML_SAMPLE)

        self.assertEqual(cleaned, "Role Summary\n- Build APIs")


if __name__ == "__main__":
    unittest.main()
