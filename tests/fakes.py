"""Fake Gemini client used by the backend tests."""

import json
from types import SimpleNamespace

from google.genai import errors


def make_response(text="A response.", prompt_tokens=10, output_tokens=20, usage=True):
    if not usage:
        return SimpleNamespace(text=text, usage_metadata=None)
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            total_token_count=(
                prompt_tokens + output_tokens
                if prompt_tokens is not None and output_tokens is not None
                else None
            ),
        ),
    )


def rate_limit_error():
    return errors.ClientError(
        429,
        {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
    )


def server_error():
    return errors.ServerError(
        503, {"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}}
    )


def bad_request_error():
    return errors.ClientError(
        400, {"error": {"code": 400, "message": "Bad request", "status": "INVALID_ARGUMENT"}}
    )


def judge_reply(relevance=8, completeness=7, clarity=9, instruction_following=8,
                feedback="Clear and on task."):
    return make_response(
        json.dumps(
            {
                "relevance": relevance,
                "completeness": completeness,
                "clarity": clarity,
                "instruction_following": instruction_following,
                "feedback": feedback,
            }
        )
    )


class FakeModels:
    """Returns queued outcomes in order. An Exception in the queue is raised."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self.outcomes:
            raise AssertionError("FakeModels ran out of queued outcomes.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


class EchoModels:
    """Answers every generation call, and every judge call with fixed scores."""

    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if config is not None and getattr(config, "response_mime_type", None) == "application/json":
            return judge_reply()
        return make_response(f"Answer to: {contents[:40]}", 15, 25)


class EchoClient:
    def __init__(self):
        self.models = EchoModels()
