import unittest
from unittest.mock import patch

from tools.job import match_cv


class MatchCvParsingTests(unittest.TestCase):
    def test_interpret_llm_evaluation_parses_json_directly(self):
        response = """
        {
          "score": 8.5,
          "strengths": ["React", "TypeScript"],
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
            "React, Redux, Node.js, Python, Java, and others. The candidate also "
            "has experience in AI, having worked with GitHub Copilot, ChatGPT, "
            "and integrating the OpenAI API."
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
            "Overall match: 7.5/10. Strong React and product engineering experience, "
            "but weaker explicit backend scaling experience."
        )

        interpreted = match_cv._interpret_llm_evaluation(response)

        self.assertEqual(interpreted["score"], 7.5)
        self.assertEqual(
            interpreted["fit_summary"],
            "Overall match: 7.5/10. Strong React and product engineering experience, but weaker explicit backend scaling experience.",
        )
        self.assertIsNone(interpreted["repair_response"])


if __name__ == "__main__":
    unittest.main()
