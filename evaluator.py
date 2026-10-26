"""LLM-as-judge evaluation for PromptBench responses."""

import json
import re
import time
from typing import Callable

from experiment_runner import RunResult, call_with_retries


RUBRIC = {
    "relevance": (
        "Does the response address the original task and stay on topic?"
    ),
    "completeness": (
        "Does it cover everything the task asks for, without major gaps?"
    ),
    "clarity": (
        "Is it well organised, easy to follow and free of confusing wording?"
    ),
    "instruction_following": (
        "Does it obey every explicit instruction and constraint in the "
        "prompt, including format, length and tone?"
    ),
}

SCORE_MIN = 1
SCORE_MAX = 10


class MalformedEvaluationError(ValueError):
    """Raised when the judge's output cannot be read as a valid rubric score."""


def build_judge_prompt(task: str, prompt: str, response: str) -> str:
    """Build the prompt sent to the judge model."""

    criteria = "\n".join(
        f'- "{name}": {description}' for name, description in RUBRIC.items()
    )
    keys = ", ".join(f'"{name}"' for name in RUBRIC)

    return (
        "You are an impartial evaluator of AI responses.\n\n"
        f"Original task:\n{task}\n\n"
        f"Prompt given to the model:\n{prompt}\n\n"
        f"Model response:\n{response}\n\n"
        f"Score the response on each criterion from {SCORE_MIN} (very poor) "
        f"to {SCORE_MAX} (excellent):\n{criteria}\n\n"
        "Judge the response against the original task and the prompt. "
        "Do not reward length for its own sake.\n\n"
        f"Return only a JSON object with the integer keys {keys} and a "
        '"feedback" key holding one or two sentences of justification.'
    )


def _extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of the judge's reply."""

    if not text or not text.strip():
        raise MalformedEvaluationError("Judge returned an empty response.")

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise MalformedEvaluationError("Judge response contains no JSON object.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise MalformedEvaluationError(f"Judge returned invalid JSON: {error}")

    if not isinstance(data, dict):
        raise MalformedEvaluationError("Judge response is not a JSON object.")
    return data


def parse_judge_response(text: str) -> tuple[dict[str, float], str | None]:
    """Validate the judge's JSON and return (scores, feedback)."""

    data = _extract_json_object(text)

    missing = [name for name in RUBRIC if name not in data]
    if missing:
        raise MalformedEvaluationError(
            f"Judge response is missing scores for: {', '.join(missing)}"
        )

    scores = {}
    for name in RUBRIC:
        value = data[name]
        if isinstance(value, bool):
            raise MalformedEvaluationError(f"Score for {name} is not a number.")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise MalformedEvaluationError(f"Score for {name} is not a number.")
        if not SCORE_MIN <= value <= SCORE_MAX:
            raise MalformedEvaluationError(
                f"Score for {name} is outside {SCORE_MIN}-{SCORE_MAX}: {value}"
            )
        scores[name] = value

    feedback = data.get("feedback")
    return scores, feedback if isinstance(feedback, str) else None


def overall_score(scores: dict[str, float]) -> float:
    """Average the rubric scores into a single evaluation score."""

    return round(sum(scores.values()) / len(scores), 2)


def judge_response(
    client,
    judge_model: str,
    task: str,
    prompt: str,
    response: str,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, float], str | None]:
    """Ask the judge model to score one response against the rubric."""

    from google.genai import types

    judge_prompt = build_judge_prompt(task, prompt, response)
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )

    reply, _ = call_with_retries(
        lambda: client.models.generate_content(
            model=judge_model, contents=judge_prompt, config=config
        ),
        max_retries=max_retries,
        sleep=sleep,
    )
    try:
        text = reply.text or ""
    except (AttributeError, ValueError):
        text = ""
    return parse_judge_response(text)


def evaluate_result(
    result: RunResult,
    task: str,
    client,
    judge_model: str,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Fill in a result's judge scores. Failures are recorded, not raised."""

    if result.status != "success":
        return result

    try:
        scores, feedback = judge_response(
            client,
            judge_model,
            task,
            result.prompt,
            result.response,
            max_retries=max_retries,
            sleep=sleep,
        )
    except MalformedEvaluationError as error:
        result.judge_feedback = f"Evaluation failed: {error}"
        return result
    except Exception as error:
        result.judge_feedback = (
            f"Evaluation failed: {type(error).__name__}: {error}"
        )
        return result

    result.judge_scores = scores
    result.judge_feedback = feedback
    result.evaluation_score = overall_score(scores)
    return result
