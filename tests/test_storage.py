import json
import tempfile
import unittest
from pathlib import Path

from storage import (
    ExperimentNotFoundError,
    delete_experiment,
    list_experiments,
    load_experiment,
    save_experiment,
)


class TestStorage(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_round_trip(self):
        results = [{"technique": "zero_shot", "response": "Hi", "evaluation_score": 8.0}]
        metadata = {"task": "Say hi", "model": "gemini-test"}

        saved = save_experiment(results, metadata, data_dir=self.data_dir)
        loaded = load_experiment(saved["id"], data_dir=self.data_dir)

        self.assertEqual(loaded, saved)
        self.assertEqual(len(saved["id"]), 32)
        self.assertIn("T", saved["created_at"])
        self.assertEqual(loaded["metadata"]["model"], "gemini-test")
        self.assertEqual(loaded["results"], results)

    def test_ids_are_unique(self):
        ids = {save_experiment([], {}, data_dir=self.data_dir)["id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_list_is_newest_first_and_skips_bad_files(self):
        first = save_experiment([], {"task": "a"}, data_dir=self.data_dir)
        second = save_experiment([{"technique": "x", "response": "y"}], {"task": "b"},
                                 data_dir=self.data_dir)
        Path(self.data_dir, "broken.json").write_text("{not json", encoding="utf-8")

        summaries = list_experiments(data_dir=self.data_dir)

        self.assertEqual([s["id"] for s in summaries], [second["id"], first["id"]])
        self.assertEqual(summaries[0]["result_count"], 1)

    def test_missing_experiment_raises(self):
        with self.assertRaises(ExperimentNotFoundError):
            load_experiment("0" * 32, data_dir=self.data_dir)

    def test_invalid_id_is_rejected(self):
        for bad in ["../secrets", "abc", None]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    load_experiment(bad, data_dir=self.data_dir)

    def test_delete(self):
        saved = save_experiment([], {}, data_dir=self.data_dir)
        delete_experiment(saved["id"], data_dir=self.data_dir)
        with self.assertRaises(ExperimentNotFoundError):
            load_experiment(saved["id"], data_dir=self.data_dir)

    def test_no_temp_files_left_behind(self):
        save_experiment([], {}, data_dir=self.data_dir)
        self.assertEqual(list(Path(self.data_dir).glob("*.tmp")), [])
        for path in Path(self.data_dir).glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
