"""Rule-based checks that complement the LLM judge."""

import json
import re


STRUCTURED_OUTPUT_KEYS = {
    "answer": str,
    "key_points": list,
    "limitations": list,
}

CLASSIFICATION_CATEGORY = "Text classification"
EXTRACTION_CATEGORY = "Information extraction"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_LIMIT_PATTERN = (
    r"(?:in|within|under|at most|no more than|maximum of|max|less than|fewer than)"
    r"\s+(\d+|{words})\s+{unit}"
)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def parse_json(text: str):
    """Return (data, error) for a response that should be JSON."""

    try:
        return json.loads(_strip_code_fences(text)), None
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error.msg} (line {error.lineno})"


def check_structured_output(text: str) -> list[str]:
    """Check the JSON shape the structured_output technique asks for."""

    data, error = parse_json(text)
    if error:
        return [error]
    if not isinstance(data, dict):
        return ["JSON response is not an object."]

    issues = []
    for key, expected_type in STRUCTURED_OUTPUT_KEYS.items():
        if key not in data:
            issues.append(f'Missing JSON key "{key}".')
        elif not isinstance(data[key], expected_type):
            issues.append(
                f'JSON key "{key}" should be a {expected_type.__name__}.'
            )
    if text.strip().startswith("```"):
        issues.append("Response used Markdown code fences despite instructions.")
    return issues


def _parse_limit(task: str, unit: str) -> int | None:
    words = "|".join(_NUMBER_WORDS)
    pattern = _LIMIT_PATTERN.format(words=words, unit=unit)
    match = re.search(pattern, task, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+(?:\s|$)", text.strip()) if part.strip()])


def check_constraints(task: str, text: str) -> list[str]:
    """Check length limits stated in the task, such as "in 50 words"."""

    issues = []
    if not text.strip():
        return ["Response is empty."]

    word_limit = _parse_limit(task, r"words?")
    if word_limit is not None:
        words = count_words(text)
        if words > word_limit:
            issues.append(f"Response has {words} words; limit is {word_limit}.")

    sentence_limit = _parse_limit(task, r"sentences?")
    if sentence_limit is not None:
        sentences = count_sentences(text)
        if sentences > sentence_limit:
            issues.append(
                f"Response has {sentences} sentences; limit is {sentence_limit}."
            )

    return issues


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text)).strip().lower()


def token_f1(prediction: str, reference: str) -> float:
    """Word-overlap F1 between a response and a reference answer."""

    pred_tokens = _normalise(prediction).split()
    ref_tokens = _normalise(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    remaining = list(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


def check_reference(category: str, text: str, reference: str) -> dict:
    """Compare a response with a reference answer for the task category."""

    if category == CLASSIFICATION_CATEGORY:
        label = _normalise(reference)
        answer = _normalise(text)
        correct = answer == label or re.search(rf"\b{re.escape(label)}\b", answer) is not None
        return {"method": "label_match", "correct": bool(correct)}

    if category == EXTRACTION_CATEGORY:
        expected, ref_error = parse_json(reference)
        actual, error = parse_json(text)
        if ref_error or not isinstance(expected, dict):
            return {"method": "field_match", "correct": None,
                    "issue": "Reference answer is not a JSON object."}
        if error or not isinstance(actual, dict):
            return {"method": "field_match", "correct": False, "field_accuracy": 0.0}
        matched = sum(
            1 for key, value in expected.items()
            if key in actual and _normalise(actual[key]) == _normalise(value)
        )
        accuracy = round(matched / len(expected), 3) if expected else 1.0
        return {"method": "field_match", "correct": accuracy == 1.0,
                "field_accuracy": accuracy}

    return {"method": "token_f1", "f1": token_f1(text, reference)}


def answer_text(technique: str, text: str) -> str:
    """Return the part of a response that holds the actual answer."""

    if technique == "structured_output":
        data, error = parse_json(text)
        if not error and isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"]
    return text


def validate_response(
    technique: str,
    category: str,
    task: str,
    text: str,
    reference_answer: str | None = None,
) -> dict:
    """Run every rule-based check that applies and summarise the results."""

    checks = {}
    issues = []

    if technique == "structured_output":
        found = check_structured_output(text)
        checks["json_valid"] = not any(i.startswith("Invalid JSON") for i in found)
        issues.extend(found)
    elif category == EXTRACTION_CATEGORY:
        _, error = parse_json(text)
        checks["json_valid"] = error is None
        if error:
            issues.append(error)

    answer = answer_text(technique, text)
    constraint_issues = check_constraints(task, answer)
    checks["constraints_met"] = not constraint_issues
    issues.extend(constraint_issues)

    if reference_answer and reference_answer.strip():
        checks["reference"] = check_reference(category, answer, reference_answer)
        if checks["reference"].get("correct") is False:
            issues.append("Response does not match the reference answer.")

    return {"passed": not issues, "issues": issues, "checks": checks}
