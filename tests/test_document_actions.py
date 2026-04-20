import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant.config import get_paths
from assistant.registry import TOOLS


class DocumentActionTests(unittest.TestCase):
    def setUp(self):
        self.data_root = get_paths()["data_root"]
        self.outputs_root = get_paths()["outputs_root"]

    def test_write_document_workflow_writes_file(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
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

    def test_write_document_workflow_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
            destination = Path(tmp_dir) / "note.txt"
            destination.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                list(
                    TOOLS["write_document"].executor(
                        destination_path=str(destination),
                        content="Hello from the test suite.",
                    )
                )

    def test_copy_file_workflow_copies_source_file(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as source_tmp_dir, tempfile.TemporaryDirectory(
            dir=self.outputs_root
        ) as output_tmp_dir:
            root = Path(source_tmp_dir)
            source = root / "source.txt"
            destination = Path(output_tmp_dir) / "copies" / "copied.txt"
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

    def test_read_documents_workflow_reads_code_files(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.py").write_text("print('hello')\n", encoding="utf-8")

            chunks = list(
                TOOLS["read_documents"].executor(
                    input_path=str(root / "example.py"),
                    recursive=False,
                )
            )

            joined = "".join(chunks)
            self.assertIn("example.py", joined)
            self.assertIn("print('hello')", joined)

    def test_summarize_documents_workflow_can_write_output(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as input_tmp_dir, tempfile.TemporaryDirectory(
            dir=self.outputs_root
        ) as output_tmp_dir:
            input_root = Path(input_tmp_dir)
            output_root = Path(output_tmp_dir)
            input_file = input_root / "input.txt"
            output_file = output_root / "summary.txt"
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
        with tempfile.TemporaryDirectory(dir=self.data_root) as input_tmp_dir, tempfile.TemporaryDirectory(
            dir=self.outputs_root
        ) as output_tmp_dir:
            root = Path(input_tmp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "a.txt").write_text("Alpha profile", encoding="utf-8")
            (docs / "b.txt").write_text("Beta profile", encoding="utf-8")
            output_file = Path(output_tmp_dir) / "ranking.txt"

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

    def test_search_web_workflow_returns_results(self):
        response_html = """
        <html>
          <body>
            <a class="result__a" href="https://example.com/one">Result One</a>
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fu3%3DaHR0cHMlM0ElMkYlMkZleGFtcGxlLmNvbSUyRnR3bw%253D%253D">Result Two</a>
          </body>
        </html>
        """

        fake_response = type(
            "FakeResponse",
            (),
            {
                "text": response_html,
                "raise_for_status": lambda self: None,
            },
        )()

        with patch("tools.web_search.requests.post", return_value=fake_response):
            chunks = list(
                TOOLS["search_web"].executor(
                    query="example query",
                    max_results=2,
                )
            )

        joined = "".join(chunks)
        self.assertIn("Found 2 result(s) for: example query", joined)
        self.assertIn("Result One", joined)
        self.assertIn("https://example.com/two", joined)

    def test_search_web_workflow_surfaces_provider_challenge(self):
        challenge_html = """
        <html>
          <body>
            <div class="anomaly-modal">
              Unfortunately, bots use DuckDuckGo too.
            </div>
          </body>
        </html>
        """

        fake_response = type(
            "FakeResponse",
            (),
            {
                "text": challenge_html,
                "raise_for_status": lambda self: None,
            },
        )()

        with patch("tools.web_search.requests.post", return_value=fake_response):
            with self.assertRaisesRegex(
                RuntimeError,
                "human-verification challenge",
            ):
                list(
                    TOOLS["search_web"].executor(
                        query="mickey",
                        max_results=2,
                    )
                )


if __name__ == "__main__":
    unittest.main()
