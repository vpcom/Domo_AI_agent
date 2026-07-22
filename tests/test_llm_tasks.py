import unittest
from unittest.mock import patch

from assistant import llm_tasks


class LlmTaskTests(unittest.TestCase):
    def test_answer_question_returns_capability_summary_for_capability_query(self):
        with patch("assistant.llm_tasks.call_llm") as call_llm:
            result = llm_tasks.answer_question("What can you do?")

        call_llm.assert_not_called()
        self.assertIn("registered capabilities", result["result"]["text"].lower())
        self.assertIn("read tools", result["result"]["text"].lower())
        self.assertIn("write tools", result["result"]["text"].lower())
        self.assertIn("I will not do these things:", result["metadata"]["display_text"])

    def test_answer_question_returns_structured_output(self):
        with patch("assistant.llm_tasks.call_llm", return_value="Paris."):
            result = llm_tasks.answer_question("What is the capital of France?")

        self.assertEqual(result["result"]["text"], "Paris.")
        self.assertEqual(result["metadata"]["display_text"], "Paris.")

    def test_answer_question_prompt_includes_capability_constraints(self):
        with patch("assistant.llm_tasks.call_llm", return_value="No.") as call_llm:
            llm_tasks.answer_question("What is the capital of France?")

        prompt = call_llm.call_args.args[0]
        self.assertIn("Registered capabilities:", prompt)
        self.assertIn("Forbidden actions:", prompt)
        self.assertIn("Never claim capabilities outside the list below.", prompt)

    def test_answer_question_refuses_forbidden_request_without_llm(self):
        with patch("assistant.llm_tasks.call_llm") as call_llm:
            result = llm_tasks.answer_question("Delete this file for me.")

        call_llm.assert_not_called()
        self.assertIn("do not delete", result["result"]["text"].lower())

    def test_summarize_text_returns_summary(self):
        with patch("assistant.llm_tasks.call_llm", return_value="Short summary."):
            result = llm_tasks.summarize_text(
                [{"path": "a.txt", "content": "alpha"}],
                instructions="Summarize briefly.",
            )

        self.assertEqual(result["result"]["summary"], "Short summary.")

    def test_summarize_text_accepts_search_results(self):
        with patch("assistant.llm_tasks.call_llm", return_value="Five films.") as call_llm:
            result = llm_tasks.summarize_text(
                [
                    {
                        "title": "Bruce Lee filmography",
                        "url": "https://example.com/bruce-lee-filmography",
                    }
                ],
                instructions="Extract the found film count.",
            )

        prompt = call_llm.call_args.args[0]
        self.assertIn("FILE: https://example.com/bruce-lee-filmography", prompt)
        self.assertIn("Title: Bruce Lee filmography", prompt)
        self.assertEqual(result["result"]["summary"], "Five films.")

    def test_evaluate_text_formats_report(self):
        response = """
        {
          "results": [
            {
              "path": "a.txt",
              "score": 9.0,
              "reasoning": "Best match",
              "highlights": ["Strong fit"]
            }
          ]
        }
        """

        with patch("assistant.llm_tasks.call_llm", return_value=response):
            result = llm_tasks.evaluate_text(
                [{"path": "a.txt", "content": "alpha"}],
                instructions="Rank by relevance.",
            )

        self.assertIn("a.txt -> score: 9.0", result["metadata"]["display_text"])

    def test_generate_document_set_returns_documents(self):
        response = """
        {
          "documents": [
            {
              "filename": "enter-the-dragon.txt",
              "content": "A concise summary."
            }
          ]
        }
        """

        with patch("assistant.llm_tasks.call_llm", return_value=response):
            result = llm_tasks.generate_document_set(
                [
                    {
                        "title": "Enter the Dragon",
                        "url": "https://example.com/enter-the-dragon",
                    }
                ],
                instructions="Create one file per movie.",
            )

        self.assertEqual(
            result["result"]["documents"][0]["filename"],
            "enter-the-dragon.txt",
        )
        self.assertIn("enter-the-dragon.txt", result["metadata"]["display_text"])

    def test_rank_cvs_returns_ranked_report(self):
        response = """
        {
          "best_cv": "candidate.pdf",
          "results": [
            {
              "path": "candidate.pdf",
              "score": 8.5,
              "reasoning": "Strong match"
            }
          ]
        }
        """

        with patch("assistant.llm_tasks.call_llm", return_value=response):
            result = llm_tasks.rank_cvs(
                [{"path": "job.txt", "content": "backend engineer"}],
                [{"path": "candidate.pdf", "content": "backend engineer experience"}],
            )

        self.assertEqual(result["result"]["best_cv"], "candidate.pdf")
        self.assertIn("Best CV: candidate.pdf", result["metadata"]["display_text"])


if __name__ == "__main__":
    unittest.main()
