import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.job.generate_application_materials import generate_application_materials, run
from tools.job.models import JobState


class GenerateApplicationMaterialsTests(unittest.TestCase):
    def test_generate_application_materials_returns_single_info_document(self):
        response = """
        {
          "summary": "Build product features for enterprise users.",
          "skills": ["Python", "SQL"],
          "cv_summary": "Backend-focused engineer with product delivery experience.",
          "key_strengths": ["Systems thinking", "Cross-functional delivery"],
          "cv_base_texts": "Led backend feature development and partner integrations.",
          "cover_letter": "I am interested in this role because it fits my experience."
        }
        """

        with patch("tools.job.generate_application_materials.call_llm", return_value=response):
            materials = generate_application_materials("Cleaned job description text")

        self.assertEqual(set(materials.keys()), {"info"})
        info = materials["info"]
        self.assertIn("SUMMARY\nBuild product features for enterprise users.", info)
        self.assertIn("KEY SKILLS\n- Python\n- SQL", info)
        self.assertIn("CV SUMMARY\nBackend-focused engineer with product delivery experience.", info)
        self.assertIn("KEY STRENGTHS\n- Systems thinking\n- Cross-functional delivery", info)
        self.assertIn("CV BASE TEXTS\nLed backend feature development and partner integrations.", info)
        self.assertIn(
            "COVER LETTER\nI am interested in this role because it fits my experience.",
            info,
        )

    def test_run_writes_only_info_file(self):
        response = """
        {
          "summary": "Summary text",
          "skills": ["Skill A", "Skill B"],
          "cv_summary": "CV summary text",
          "key_strengths": ["Strength A"],
          "cv_base_texts": "Base text",
          "cover_letter": "Cover letter text"
        }
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cleaned_file = root / "cleaned_job_description.txt"
            cleaned_file.write_text("Cleaned job description", encoding="utf-8")
            state = JobState(
                folder=root,
                raw_file=root / "job_description_raw.txt",
                metadata_file=root / "job_metadata.json",
                cleaned_file=cleaned_file,
                pdf_file=root / "job_description.pdf",
                info_file=root / "info.txt",
            )

            with patch("tools.job.generate_application_materials.call_llm", return_value=response):
                run(state)

            self.assertTrue((root / "info.txt").exists())
            self.assertFalse((root / "application_notes.txt").exists())
            self.assertFalse((root / "summary.txt").exists())
            self.assertFalse((root / "skills.txt").exists())
            self.assertFalse((root / "sample_cv.txt").exists())


if __name__ == "__main__":
    unittest.main()
