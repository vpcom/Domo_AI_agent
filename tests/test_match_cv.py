import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from reportlab.pdfgen import canvas

from assistant.config import get_paths
from tools.job import match_cv


def _write_pdf(file_path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(file_path))
    y = 800
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 20
    pdf.save()


class MatchCvParsingTests(unittest.TestCase):
    def test_interpret_llm_evaluation_parses_json_directly(self):
        response = """
        {
          "score": 8.5,
          "strengths": ["Framework A", "Language B"],
          "weaknesses": ["Limited backend depth"],
          "fit_summary": "Strong frontend match with enough full-stack overlap."
        }
        """

        interpreted = match_cv._interpret_llm_evaluation(response)

        self.assertEqual(interpreted["score"], 8.5)
        self.assertEqual(
            interpreted["fit_summary"],
            "Strong frontend match with enough full-stack overlap.",
        )
        self.assertIsNone(interpreted["repair_response"])

    def test_interpret_llm_evaluation_recovers_summary_and_repairs_score(self):
        response = (
            "The CV provided has a strong background in software engineering, "
            "with experience in both frontend and backend development, including "
            "Framework A, State Tool B, Runtime C, Language D, Platform E, and others. The candidate also "
            "has experience with developer-assistance tools and modern API integrations."
        )

        with patch.object(
            match_cv,
            "call_llm",
            return_value=(
                '{"score": 8, "strengths": ["Strong frontend background"], '
                '"weaknesses": ["Backend depth could be stronger"], '
                '"fit_summary": "Strong frontend-leaning full-stack fit."}'
            ),
        ):
            interpreted = match_cv._interpret_llm_evaluation(response)

        self.assertEqual(interpreted["score"], 8.0)
        self.assertEqual(
            interpreted["fit_summary"],
            "Strong frontend-leaning full-stack fit.",
        )
        self.assertIsNotNone(interpreted["repair_response"])

    def test_interpret_llm_evaluation_extracts_inline_score_without_repair(self):
        response = (
            "Overall match: 7.5/10. Strong frontend framework and product engineering experience, "
            "but weaker explicit backend scaling experience."
        )

        interpreted = match_cv._interpret_llm_evaluation(response)

        self.assertEqual(interpreted["score"], 7.5)
        self.assertEqual(
            interpreted["fit_summary"],
            "Overall match: 7.5/10. Strong frontend framework and product engineering experience, but weaker explicit backend scaling experience.",
        )
        self.assertIsNone(interpreted["repair_response"])

    def test_match_cv_bootstraps_job_text_from_spaced_pdf_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            job_folder = temp_root / "20260329 - Recruiting Partner - UI Engineer"
            job_folder.mkdir()
            _write_pdf(
                job_folder / "job description.pdf",
                [
                    "UI Engineer",
                    "Build modern interface applications.",
                ],
            )

            cvs_folder = temp_root / "cvs"
            cvs_folder.mkdir()
            outputs_root = temp_root / "outputs"
            outputs_root.mkdir()
            _write_pdf(
                cvs_folder / "Candidate.pdf",
                [
                    "Example Candidate",
                    "Interface application engineer",
                ],
            )

            with patch.object(
                match_cv,
                "call_llm",
                return_value=(
                    '{"score": 8.5, "strengths": ["Framework A"], '
                    '"weaknesses": ["Limited finance context"], '
                    '"fit_summary": "Strong frontend fit."}'
                ),
            ), patch.object(
                match_cv,
                "get_paths",
                return_value={
                    "data_root": temp_root,
                    "inputs_root": temp_root / "inputs",
                    "jobs_root": temp_root / "jobs",
                    "documents_root": temp_root / "documents",
                    "outputs_root": outputs_root,
                    "cvs_root": cvs_folder,
                    "logs_root": outputs_root / "logs",
                },
            ):
                output = "".join(match_cv.match_cv(str(job_folder), str(cvs_folder)))

            self.assertIn("Loading job description: UI Engineer", output)
            output_folders = list(outputs_root.glob("*/20260329 - Recruiting Partner - UI Engineer"))
            self.assertEqual(len(output_folders), 1)
            output_folder = output_folders[0]
            self.assertTrue((output_folder / "job_description_raw.txt").exists())
            self.assertTrue((output_folder / "job_metadata.json").exists())
            self.assertTrue((output_folder / "cv_match_analysis.json").exists())
            self.assertTrue((output_folder / "cv_match_summary.txt").exists())
            self.assertFalse((job_folder / "job_description_raw.txt").exists())


if __name__ == "__main__":
    unittest.main()
