
import unittest

from results_dashboard import prepare_results


class TestResultsDashboard(unittest.TestCase):

    def test_empty_results_returns_empty_dataframe(self):
        df = prepare_results([])

        self.assertTrue(df.empty)
        self.assertIn("evaluation_score", df.columns)

    def test_valid_results_are_prepared(self):
        results = [
            {
                "technique": "zero_shot",
                "response": "Machine learning is a type of AI.",
                "latency_seconds": 1.2,
                "input_tokens": 12,
                "output_tokens": 20,
                "evaluation_score": 8.5,
            }
        ]

        df = prepare_results(results)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["technique"], "zero_shot")
        self.assertEqual(df.iloc[0]["evaluation_score"], 8.5)

    def test_missing_optional_fields_are_added(self):
        results = [
            {
                "technique": "zero_shot",
                "response": "Example response",
            }
        ]

        df = prepare_results(results)

        self.assertIn("latency_seconds", df.columns)
        self.assertIn("input_tokens", df.columns)
        self.assertIn("output_tokens", df.columns)
        self.assertIn("evaluation_score", df.columns)

    def test_missing_required_field_raises_error(self):
        with self.assertRaises(ValueError):
            prepare_results([{"technique": "zero_shot"}])


if __name__ == "__main__":
    unittest.main()
