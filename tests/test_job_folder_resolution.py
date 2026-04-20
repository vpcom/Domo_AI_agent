import tempfile
import unittest
from datetime import date
from pathlib import Path

from reportlab.pdfgen import canvas

from tools.job.job_folder_resolution import (
    find_best_matching_job_folder,
    resolve_job_folder_hint,
)
from tools.job.local_job_inputs import infer_local_pdf_metadata, resolve_local_job_inputs


class JobFolderResolutionTests(unittest.TestCase):
    def test_find_best_matching_job_folder_prefers_closest_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_root = Path(tmp_dir)
            (jobs_root / "20260321 - company-alpha - Application Engineer").mkdir()
            newest = jobs_root / "20260329 - company-alpha - Application Engineer II"
            newest.mkdir()
            (jobs_root / "20260319 - company-beta - Growth Engineer").mkdir()

            resolved = find_best_matching_job_folder(
                "company-alpha",
                jobs_root,
                today=date(2026, 3, 29),
            )

            self.assertEqual(resolved, newest)

    def test_resolve_job_folder_hint_can_use_company_name_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            jobs_root = project_root / "data" / "inputs" / "jobs"
            jobs_root.mkdir(parents=True)
            target = jobs_root / "20260329 - company-alpha - Application Engineer II"
            target.mkdir()

            resolved = resolve_job_folder_hint(
                "company-alpha",
                project_root,
                jobs_root,
                today=date(2026, 3, 29),
            )

            self.assertEqual(resolved, target.resolve())

    def test_resolve_local_job_inputs_bootstraps_raw_and_metadata_from_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "20260329 - company-alpha - Application Engineer II"
            folder.mkdir()
            pdf_path = folder / "job_description.pdf"

            pdf = canvas.Canvas(str(pdf_path))
            pdf.drawString(72, 800, "Application Engineer II")
            pdf.drawString(72, 780, "Build services and ship backend systems.")
            pdf.save()

            resolved = resolve_local_job_inputs(folder)

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.mode, "raw")
            self.assertIsNotNone(resolved.raw_text)
            self.assertIn("Application Engineer II", resolved.raw_text)
            self.assertEqual(resolved.metadata["company"].lower(), "company-alpha")
            self.assertFalse((folder / "job_description_raw.txt").exists())
            self.assertFalse((folder / "job_metadata.json").exists())

    def test_resolve_local_job_inputs_bootstraps_raw_from_spaced_text_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "20260329 - Recruiting Partner - UI Engineer"
            folder.mkdir()
            text_path = folder / "job description.txt"
            text_path.write_text(
                "UI Engineer\nBuild modern interface applications.",
                encoding="utf-8",
            )

            resolved = resolve_local_job_inputs(folder)

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.mode, "raw")
            self.assertIsNotNone(resolved.raw_text)
            self.assertIn("UI Engineer", resolved.raw_text)
            self.assertEqual(resolved.metadata["company"].lower(), "recruiting partner")
            self.assertFalse((folder / "job_description_raw.txt").exists())
            self.assertFalse((folder / "job_metadata.json").exists())

    def test_infer_local_pdf_metadata_uses_folder_name(self):
        metadata = infer_local_pdf_metadata(
            Path("20260329 - company-alpha - Application Engineer II"),
            "Example description",
        )

        self.assertEqual(metadata["company"], "company-alpha")
        self.assertEqual(metadata["title"], "Application Engineer II")
        self.assertEqual(metadata["source"], "local_pdf")


if __name__ == "__main__":
    unittest.main()
