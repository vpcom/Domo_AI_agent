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

    def test_write_document_returns_structured_output(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
            destination = Path(tmp_dir) / "note.txt"

            result = TOOLS["write_document"].function(
                destination_path=str(destination),
                content="Hello from the test suite.",
            )

            self.assertTrue(destination.exists())
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "Hello from the test suite.",
            )
            self.assertEqual(result["result"]["destination_path"], str(destination))
            self.assertIn("display_text", result["metadata"])
            self.assertEqual(result["metadata"]["artifacts"][0]["path"], str(destination))

    def test_copy_file_returns_structured_output(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as source_tmp_dir, tempfile.TemporaryDirectory(
            dir=self.outputs_root
        ) as output_tmp_dir:
            source = Path(source_tmp_dir) / "source.txt"
            destination = Path(output_tmp_dir) / "copies" / "copied.txt"
            source.write_text("source content", encoding="utf-8")

            result = TOOLS["copy_file"].function(
                source_path=str(source),
                destination_path=str(destination),
            )

            self.assertEqual(destination.read_text(encoding="utf-8"), "source content")
            self.assertEqual(result["result"]["destination_path"], str(destination))
            self.assertEqual(result["metadata"]["artifacts"][0]["path"], str(destination))

    def test_read_documents_returns_documents_list(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.md").write_text("beta", encoding="utf-8")

            result = TOOLS["read_documents"].function(
                input_path=str(root),
                recursive=False,
            )

            self.assertEqual(len(result["result"]["documents"]), 2)
            self.assertIn("alpha", result["metadata"]["display_text"])
            self.assertIn("beta", result["metadata"]["display_text"])

    def test_search_web_returns_structured_results(self):
        response_html = """
        <html>
          <body>
            <a class="result__a" href="https://example.com/one">Result One</a>
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Result Two</a>
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
            result = TOOLS["search_web"].function(
                query="example query",
                max_results=2,
            )

        self.assertEqual(len(result["result"]["results"]), 2)
        self.assertIn("Found 2 result(s) for: example query", result["metadata"]["display_text"])
        self.assertIn("https://example.com/two", result["metadata"]["display_text"])


if __name__ == "__main__":
    unittest.main()
