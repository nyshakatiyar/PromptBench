import unittest

from evaluator import (
    RUBRIC,
    MalformedEvaluationError,
    build_judge_prompt,
    evaluate_result,
    overall_score,
    parse_judge_response,
)
from experiment_runner import RunResult
from tests.fakes import FakeClient, judge_reply, make_response, rate_limit_error


def no_sleep(_seconds):
    pass


def success_result():
    return RunResult(technique="zero_shot", prompt="Explain ML", response="ML learns from data.")


class TestParseJudgeResponse(unittest.TestCase):

    def test_valid_json(self):
        scores, feedback = parse_judge_response(
            '{"relevance": 8, "completeness": 7, "clarity": 9, '
            '"instruction_following": 10, "feedback": "Good."}'
        )
        self.assertEqual(scores["instruction_following"], 10.0)
        self.assertEqual(feedback, "Good.")

    def test_json_inside_code_fences_and_prose(self):
        text = ('Here you go:\n```json\n{"relevance": 5, "completeness": 5, '
                '"clarity": 5, "instruction_following": 5}\n```')
        scores, feedback = parse_judge_response(text)
        self.assertEqual(set(scores), set(RUBRIC))
        self.assertIsNone(feedback)

    def test_malformed_cases_raise(self):
        cases = [
            "",
            "not json at all",
            "{relevance: 8}",
            "[1, 2, 3]",
            '{"relevance": 8, "completeness": 7, "clarity": 9}',
            '{"relevance": "high", "completeness": 7, "clarity": 9, "instruction_following": 8}',
            '{"relevance": 11, "completeness": 7, "clarity": 9, "instruction_following": 8}',
            '{"relevance": 0, "completeness": 7, "clarity": 9, "instruction_following": 8}',
            '{"relevance": true, "completeness": 7, "clarity": 9, "instruction_following": 8}',
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(MalformedEvaluationError):
                    parse_judge_response(text)


class TestEvaluateResult(unittest.TestCase):

    def test_scores_are_filled_in(self):
        client = FakeClient([judge_reply(8, 7, 9, 8)])

        result = evaluate_result(success_result(), "Explain ML", client, "judge", sleep=no_sleep)

        self.assertEqual(result.evaluation_score, 8.0)
        self.assertEqual(result.judge_scores["clarity"], 9.0)
        self.assertEqual(result.judge_feedback, "Clear and on task.")
        call = client.models.calls[0]
        self.assertEqual(call["model"], "judge")
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertEqual(call["config"].temperature, 0)

    def test_malformed_judge_output_leaves_score_empty(self):
        client = FakeClient([make_response("I think it is pretty good!")])

        result = evaluate_result(success_result(), "Explain ML", client, "judge", sleep=no_sleep)

        self.assertIsNone(result.evaluation_score)
        self.assertTrue(result.judge_feedback.startswith("Evaluation failed"))

    def test_judge_rate_limit_is_retried(self):
        client = FakeClient([rate_limit_error(), judge_reply()])

        result = evaluate_result(success_result(), "Explain ML", client, "judge", sleep=no_sleep)

        self.assertIsNotNone(result.evaluation_score)

    def test_judge_api_failure_is_recorded(self):
        client = FakeClient([rate_limit_error()] * 4)

        result = evaluate_result(success_result(), "Explain ML", client, "judge", sleep=no_sleep)

        self.assertIsNone(result.evaluation_score)
        self.assertIn("429", result.judge_feedback)

    def test_failed_runs_are_not_judged(self):
        client = FakeClient([])
        failed = RunResult(technique="zero_shot", prompt="p", status="error", error="boom")

        result = evaluate_result(failed, "task", client, "judge")

        self.assertIsNone(result.evaluation_score)
        self.assertEqual(client.models.calls, [])


class TestHelpers(unittest.TestCase):

    def test_overall_score_is_mean(self):
        self.assertEqual(overall_score({"a": 7, "b": 8, "c": 8, "d": 9}), 8.0)

    def test_judge_prompt_mentions_every_criterion(self):
        prompt = build_judge_prompt("task", "prompt", "response")
        for name in RUBRIC:
            self.assertIn(name, prompt)


if __name__ == "__main__":
    unittest.main()
