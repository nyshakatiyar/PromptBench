import unittest
from types import SimpleNamespace

from experiment_runner import (
    call_with_retries,
    extract_usage,
    is_rate_limit_error,
    run_experiment,
    run_prompt,
)
from tests.fakes import (
    FakeClient,
    bad_request_error,
    make_response,
    rate_limit_error,
    server_error,
)


def no_sleep(_seconds):
    pass


class TestRunPrompt(unittest.TestCase):

    def test_success_captures_response_tokens_and_latency(self):
        client = FakeClient([make_response("Hello", 12, 30)])
        ticks = iter([1.0, 1.75])

        result = run_prompt(
            client, "zero_shot", "Say hi", "gemini-test",
            sleep=no_sleep, clock=lambda: next(ticks),
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.response, "Hello")
        self.assertEqual(result.input_tokens, 12)
        self.assertEqual(result.output_tokens, 30)
        self.assertEqual(result.total_tokens, 42)
        self.assertEqual(result.latency_seconds, 0.75)
        self.assertEqual(result.attempts, 1)
        self.assertIsNone(result.error)
        self.assertEqual(client.models.calls[0]["model"], "gemini-test")
        self.assertEqual(client.models.calls[0]["contents"], "Say hi")

    def test_missing_usage_metadata_leaves_tokens_empty(self):
        client = FakeClient([make_response("Hi", usage=False)])

        result = run_prompt(client, "zero_shot", "Say hi", "m", sleep=no_sleep)

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.total_tokens)

    def test_partial_usage_metadata(self):
        client = FakeClient([make_response("Hi", prompt_tokens=5, output_tokens=None)])

        result = run_prompt(client, "zero_shot", "Say hi", "m", sleep=no_sleep)

        self.assertEqual(result.input_tokens, 5)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.total_tokens)

    def test_rate_limit_is_retried_then_succeeds(self):
        delays = []
        client = FakeClient([rate_limit_error(), rate_limit_error(), make_response("Ok")])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=delays.append)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(delays, [2.0, 4.0])

    def test_rate_limit_exhausts_retries(self):
        client = FakeClient([rate_limit_error()] * 4)

        result = run_prompt(client, "zero_shot", "p", "m", max_retries=3, sleep=no_sleep)

        self.assertEqual(result.status, "rate_limited")
        self.assertEqual(result.attempts, 4)
        self.assertIn("429", result.error)
        self.assertEqual(result.response, "")

    def test_server_error_is_retried(self):
        client = FakeClient([server_error(), make_response("Ok")])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=no_sleep)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.attempts, 2)

    def test_client_error_is_not_retried(self):
        client = FakeClient([bad_request_error()])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=no_sleep)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(client.models.calls), 1)

    def test_unexpected_exception_is_recorded(self):
        client = FakeClient([ConnectionError("network down")])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=no_sleep)

        self.assertEqual(result.status, "error")
        self.assertIn("network down", result.error)

    def test_empty_response_is_flagged(self):
        client = FakeClient([make_response(None)])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=no_sleep)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.response, "")

    def test_text_property_that_raises_is_flagged_empty(self):
        class Blocked:
            usage_metadata = None

            @property
            def text(self):
                raise ValueError("No candidates")

        client = FakeClient([Blocked()])

        result = run_prompt(client, "zero_shot", "p", "m", sleep=no_sleep)

        self.assertEqual(result.status, "empty")


class TestRunExperiment(unittest.TestCase):

    def test_runs_every_variant_and_isolates_failures(self):
        client = FakeClient([make_response("A"), bad_request_error(), make_response("C")])
        variants = {"zero_shot": "p1", "few_shot": "p2", "role_prompting": "p3"}
        progress = []

        results = run_experiment(
            variants, client, "m", sleep=no_sleep,
            on_progress=lambda i, n, t: progress.append((i, n, t)),
        )

        self.assertEqual([r.technique for r in results], list(variants))
        self.assertEqual([r.status for r in results], ["success", "error", "success"])
        self.assertEqual(progress[1], (2, 3, "few_shot"))

    def test_result_dict_has_dashboard_fields(self):
        client = FakeClient([make_response("A")])

        row = run_experiment({"zero_shot": "p"}, client, "m")[0].to_dict()

        for key in ["technique", "prompt", "response", "latency_seconds",
                    "input_tokens", "output_tokens", "evaluation_score"]:
            self.assertIn(key, row)


class TestHelpers(unittest.TestCase):

    def test_is_rate_limit_error(self):
        self.assertTrue(is_rate_limit_error(rate_limit_error()))
        self.assertTrue(is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED")))
        self.assertFalse(is_rate_limit_error(bad_request_error()))

    def test_extract_usage_ignores_non_integer_values(self):
        response = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count="12", candidates_token_count=None, total_token_count=None,
        ))
        self.assertEqual(extract_usage(response), (None, None, None))

    def test_call_with_retries_reraises_non_retryable(self):
        def fail():
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            call_with_retries(fail, sleep=no_sleep)


if __name__ == "__main__":
    unittest.main()
