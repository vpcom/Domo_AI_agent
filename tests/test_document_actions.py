import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant.config import get_paths
from assistant.registry import TOOLS


class DocumentActionTests(unittest.TestCase):
    def setUp(self):
        self.data_root = get_paths()["data_root"]

    def test_write_document_workflow_writes_file(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            destination = Path(tmp_dir) / "note.txt"

            chunks = list(
                TOOLS["write_document"].executor(
                    destination_path=str(destination),
                    content="Hello from the test suite.",
                )
            )

            self.assertTrue(destination.exists())
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "Hello from the test suite.",
            )
            self.assertEqual(chunks[0], "Starting write_document workflow...\n")
            self.assertIn("Output written to:", "".join(chunks))

    def test_copy_file_workflow_copies_source_file(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.txt"
            destination = root / "copies" / "copied.txt"
            source.write_text("source content", encoding="utf-8")

            chunks = list(
                TOOLS["copy_file"].executor(
                    source_path=str(source),
                    destination_path=str(destination),
                )
            )

            self.assertEqual(destination.read_text(encoding="utf-8"), "source content")
            self.assertEqual(chunks[0], "Starting copy_file workflow...\n")
            self.assertIn("Workflow finished.\n", chunks[-1])

    def test_read_documents_workflow_reads_folder(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.md").write_text("beta", encoding="utf-8")

            chunks = list(
                TOOLS["read_documents"].executor(
                    input_path=str(root),
                    recursive=False,
                )
            )

            joined = "".join(chunks)
            self.assertIn("Loaded 2 document(s)", joined)
            self.assertIn("alpha", joined)
            self.assertIn("beta", joined)

    def test_summarize_documents_workflow_can_write_output(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            input_file = root / "input.txt"
            output_file = root / "summary.txt"
            input_file.write_text("This is a source document.", encoding="utf-8")

            with patch("tools.document_actions.call_llm", return_value="Short summary."):
                chunks = list(
                    TOOLS["summarize_documents"].executor(
                        input_path=str(input_file),
                        instructions="Summarize in one line.",
                        output_path=str(output_file),
                        recursive=False,
                    )
                )

            self.assertEqual(output_file.read_text(encoding="utf-8"), "Short summary.\n")
            self.assertIn("Summary generated.", "".join(chunks))
            self.assertIn("Output written to:", "".join(chunks))

    def test_evaluate_documents_workflow_formats_ranked_output(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "a.txt").write_text("Alpha profile", encoding="utf-8")
            (docs / "b.txt").write_text("Beta profile", encoding="utf-8")
            output_file = root / "ranking.txt"

            response = """
            {
              "results": [
                {
                  "path": "a.txt",
                  "score": 9.0,
                  "reasoning": "Best match",
                  "highlights": ["Strong fit"]
                },
                {
                  "path": "b.txt",
                  "score": 6.5,
                  "reasoning": "Partial match",
                  "highlights": ["Needs review"]
                }
              ]
            }
            """

            with patch("tools.document_actions.call_llm", return_value=response):
                chunks = list(
                    TOOLS["evaluate_documents"].executor(
                        input_path=str(docs),
                        instructions="Rank by relevance.",
                        output_path=str(output_file),
                        recursive=False,
                    )
                )

            report = output_file.read_text(encoding="utf-8")
            self.assertIn("a.txt -> score: 9.0", report)
            self.assertIn("Reasoning: Best match", report)
            self.assertIn("b.txt -> score: 6.5", report)
            self.assertIn("Evaluation generated.", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
