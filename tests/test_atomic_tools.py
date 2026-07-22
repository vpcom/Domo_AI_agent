import json
import tempfile
import unittest
from pathlib import Path

from assistant.config import get_paths
from assistant.registry import TOOLS


class AtomicToolTests(unittest.TestCase):
    def setUp(self):
        paths = get_paths()
        self.data_root = paths["data_root"]
        self.outputs_root = paths["outputs_root"]

    def test_inspect_path_reports_file_metadata(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            target = Path(tmp_dir) / "sample.txt"
            target.write_text("hello", encoding="utf-8")

            result = TOOLS["inspect_path"].function(path=str(target))

        self.assertTrue(result["result"]["exists"])
        self.assertTrue(result["result"]["is_file"])
        self.assertEqual(result["result"]["name"], "sample.txt")

    def test_list_directory_returns_entries(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")

            result = TOOLS["list_directory"].function(path=str(root))

        names = [entry["name"] for entry in result["result"]["entries"]]
        self.assertEqual(names, ["a.txt", "b.txt"])

    def test_read_json_file_returns_payload(self):
        with tempfile.TemporaryDirectory(dir=self.data_root) as tmp_dir:
            target = Path(tmp_dir) / "sample.json"
            target.write_text(json.dumps({"name": "domo"}), encoding="utf-8")

            result = TOOLS["read_json_file"].function(path=str(target))

        self.assertEqual(result["result"]["payload"]["name"], "domo")

    def test_write_json_file_writes_new_output(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
            target = Path(tmp_dir) / "sample.json"

            result = TOOLS["write_json_file"].function(
                destination_path=str(target),
                payload={"name": "domo"},
            )

            self.assertTrue(target.exists())
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"name": "domo"},
            )
            self.assertEqual(result["metadata"]["artifacts"][0]["path"], str(target))

    def test_write_search_results_writes_output_file(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
            target = Path(tmp_dir) / "results.txt"

            result = TOOLS["write_search_results"].function(
                destination_path=str(target),
                query="example query",
                results=[
                    {"title": "Result One", "url": "https://example.com/one"},
                    {"title": "Result Two", "url": "https://example.com/two"},
                ],
            )

            content = target.read_text(encoding="utf-8")
            self.assertIn("Found 2 result(s) for: example query", content)
            self.assertIn("https://example.com/two", content)
            self.assertEqual(result["metadata"]["artifacts"][0]["path"], str(target))

    def test_write_generated_documents_writes_multiple_safe_files(self):
        with tempfile.TemporaryDirectory(dir=self.outputs_root) as tmp_dir:
            output_dir = Path(tmp_dir) / "generated"

            result = TOOLS["write_generated_documents"].function(
                output_dir=str(output_dir),
                documents=[
                    {
                        "filename": "Enter the Dragon.txt",
                        "content": "Martial arts tournament summary.",
                    },
                    {
                        "filename": "../Way of the Dragon",
                        "content": "Rome-set action summary.",
                    },
                    {
                        "filename": "Way of the Dragon.txt",
                        "content": "Duplicate title summary.",
                    },
                ],
            )

            first = output_dir / "Enter-the-Dragon.txt"
            second = output_dir / "Way-of-the-Dragon.txt"
            third = output_dir / "Way-of-the-Dragon-2.txt"
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())
            self.assertIn("Martial arts", first.read_text(encoding="utf-8"))
            self.assertEqual(len(result["metadata"]["artifacts"]), 3)


if __name__ == "__main__":
    unittest.main()
