import tempfile
import unittest
from datetime import date
from pathlib import Path

from reportlab.pdfgen import canvas

from tools.job.job_folder_resolution import (
    find_best_matching_job_folder,
    resolve_job_folder_hint,
)
from tools.job.local_job_inputs import ensure_local_job_inputs, infer_local_pdf_metadata


class JobFolderResolutionTests(unittest.TestCase):
    def test_find_best_matching_job_folder_prefers_closest_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jobs_root = Path(tmp_dir)
            (jobs_root / "20260321 - checkr - Backend Engineer").mkdir()
            newest = jobs_root / "20260329 - checkr - Backend Software Engineer II"
            newest.mkdir()
            (jobs_root / "20260319 - elevenlabs - Growth Engineer").mkdir()

            resolved = find_best_matching_job_folder(
                "checkr",
                jobs_root,
                today=date(2026, 3, 29),
            )

            self.assertEqual(resolved, newest)

    def test_resolve_job_folder_hint_can_use_company_name_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            jobs_root = project_root / "data" / "jobs"
            jobs_root.mkdir(parents=True)
            target = jobs_root / "20260329 - checkr - Backend Software Engineer II"
            target.mkdir()

            resolved = resolve_job_folder_hint(
                "checkr",
                project_root,
                jobs_root,
                today=date(2026, 3, 29),
            )

            self.assertEqual(resolved, target.resolve())

    def test_ensure_local_job_inputs_bootstraps_raw_and_metadata_from_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "20260329 - checkr - Backend Software Engineer II"
            folder.mkdir()
            pdf_path = folder / "job_description.pdf"

            pdf = canvas.Canvas(str(pdf_path))
            pdf.drawString(72, 800, "Backend Software Engineer II")
            pdf.drawString(72, 780, "Build APIs and ship backend systems.")
            pdf.save()

            ensure_local_job_inputs(folder)

            raw_file = folder / "job_description_raw.txt"
            metadata_file = folder / "job_metadata.json"
            self.assertTrue(raw_file.exists())
            self.assertTrue(metadata_file.exists())
            self.assertIn("Backend Software Engineer II", raw_file.read_text(encoding="utf-8"))
            self.assertIn('"company": "checkr"', metadata_file.read_text(encoding="utf-8").lower())

    def test_ensure_local_job_inputs_bootstraps_raw_from_spaced_text_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "20260329 - TEKsystems for UBS - Senior Frontend Developer"
            folder.mkdir()
            text_path = folder / "job description.txt"
            text_path.write_text(
                "Senior Frontend Developer\nBuild modern React applications.",
                encoding="utf-8",
            )

            ensure_local_job_inputs(folder)

            raw_file = folder / "job_description_raw.txt"
            metadata_file = folder / "job_metadata.json"
            self.assertTrue(raw_file.exists())
            self.assertTrue(metadata_file.exists())
            self.assertIn("Senior Frontend Developer", raw_file.read_text(encoding="utf-8"))
            self.assertIn('"company": "teksystems for ubs"', metadata_file.read_text(encoding="utf-8").lower())

    def test_infer_local_pdf_metadata_uses_folder_name(self):
        metadata = infer_local_pdf_metadata(
            Path("20260329 - checkr - Backend Software Engineer II"),
            "Example description",
        )

        self.assertEqual(metadata["company"], "checkr")
        self.assertEqual(metadata["title"], "Backend Software Engineer II")
        self.assertEqual(metadata["source"], "local_pdf")


if __name__ == "__main__":
    unittest.main()
