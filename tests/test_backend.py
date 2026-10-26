import tempfile
import unittest

from backend import run_full_experiment
from storage import load_experiment
from tests.fakes import FakeClient, bad_request_error, judge_reply, make_response


def no_sleep(_seconds):
    pass


class TestRunFullExperiment(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_runs_validates_judges_and_saves(self):
        client = FakeClient([
            make_response("Positive", 10, 2),
            bad_request_error(),
            judge_reply(9, 9, 9, 9),
        ])
        variants = {"zero_shot": "Classify: I love it", "few_shot": "Examples... Classify"}

        record = run_full_experiment(
            "Classify: I love it", "Text classification", variants, client,
            model="gen", judge_model="judge", reference_answer="Positive",
            data_dir=self.data_dir, sleep=no_sleep,
        )

        ok, failed = record["results"]
        self.assertEqual(ok["evaluation_score"], 9.0)
        self.assertTrue(ok["validation"]["passed"])
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["evaluation_score"])
        self.assertEqual(record["metadata"]["model"], "gen")
        self.assertEqual(record["metadata"]["techniques"], ["zero_shot", "few_shot"])
        self.assertEqual(load_experiment(record["id"], data_dir=self.data_dir), record)

    def test_without_evaluation_or_saving(self):
        client = FakeClient([make_response("Answer")])

        record = run_full_experiment(
            "Explain ML", "General question answering", {"zero_shot": "Explain ML"},
            client, model="gen", judge_model="judge", evaluate=False, save=False,
        )

        self.assertIsNone(record["id"])
        self.assertIsNone(record["results"][0]["evaluation_score"])
        self.assertEqual(len(client.models.calls), 1)


if __name__ == "__main__":
    unittest.main()
