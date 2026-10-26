"""Run prompt variants against Gemini and capture responses."""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable


RATE_LIMIT_CODES = {429}
RETRYABLE_CODES = {429, 500, 502, 503, 504}


@dataclass
class RunResult:
    """One prompt variant's response, in the dashboard's result format."""

    technique: str
    prompt: str
    response: str = ""
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model: str = ""
    status: str = "success"
    error: str | None = None
    attempts: int = 0
    evaluation_score: float | None = None
    judge_scores: dict = field(default_factory=dict)
    judge_feedback: str | None = None
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _error_code(error: Exception) -> int | None:
    """Return the HTTP status code of a Gemini API error, if any."""

    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def is_rate_limit_error(error: Exception) -> bool:
    if _error_code(error) in RATE_LIMIT_CODES:
        return True
    text = str(error).upper()
    return "RESOURCE_EXHAUSTED" in text or "RATE LIMIT" in text


def is_retryable_error(error: Exception) -> bool:
    return is_rate_limit_error(error) or _error_code(error) in RETRYABLE_CODES


def extract_usage(response) -> tuple[int | None, int | None, int | None]:
    """Read token counts from a Gemini response, tolerating missing fields."""

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None, None

    def as_int(value):
        return value if isinstance(value, int) else None

    input_tokens = as_int(getattr(usage, "prompt_token_count", None))
    output_tokens = as_int(getattr(usage, "candidates_token_count", None))
    total_tokens = as_int(getattr(usage, "total_token_count", None))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens


def extract_text(response) -> str:
    """Return the response text, or an empty string if Gemini sent none."""

    try:
        text = response.text
    except (AttributeError, ValueError):
        # .text can raise when a response was blocked or has no candidates.
        return ""
    return text or ""


def call_with_retries(
    func: Callable[[], object],
    max_retries: int = 3,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call func, retrying rate limits and server errors with backoff.

    Returns (result, attempts). Re-raises the last error when retries run out.
    """

    attempt = 0
    while True:
        attempt += 1
        try:
            return func(), attempt
        except Exception as error:
            if attempt > max_retries or not is_retryable_error(error):
                error.attempts = attempt
                raise
            sleep(base_delay * (2 ** (attempt - 1)))


def run_prompt(
    client,
    technique: str,
    prompt: str,
    model: str,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> RunResult:
    """Send one prompt to Gemini and capture the response and metrics."""

    result = RunResult(technique=technique, prompt=prompt, model=model)

    def request():
        start = clock()
        response = client.models.generate_content(model=model, contents=prompt)
        return response, clock() - start

    try:
        (response, latency), attempts = call_with_retries(
            request, max_retries=max_retries, sleep=sleep
        )
    except Exception as error:
        result.status = "rate_limited" if is_rate_limit_error(error) else "error"
        result.error = f"{type(error).__name__}: {error}"
        result.attempts = getattr(error, "attempts", 1)
        return result

    result.attempts = attempts
    result.latency_seconds = round(latency, 3)
    result.response = extract_text(response)
    (
        result.input_tokens,
        result.output_tokens,
        result.total_tokens,
    ) = extract_usage(response)

    if not result.response.strip():
        result.status = "empty"
        result.error = "Model returned no text (possibly blocked)."

    return result


def run_experiment(
    variants: dict[str, str],
    client,
    model: str,
    max_retries: int = 3,
    on_progress: Callable[[int, int, str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RunResult]:
    """Run every selected prompt variant and return one result per variant.

    A failure in one variant is recorded on its result and does not stop
    the others.
    """

    results = []
    total = len(variants)

    for index, (technique, prompt) in enumerate(variants.items(), start=1):
        if on_progress:
            on_progress(index, total, technique)
        results.append(
            run_prompt(
                client,
                technique,
                prompt,
                model,
                max_retries=max_retries,
                sleep=sleep,
            )
        )

    return results
