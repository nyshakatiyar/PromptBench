"""Run, evaluate and save a full PromptBench experiment."""

import time
from typing import Callable

from evaluator import evaluate_result
from experiment_runner import run_experiment
from storage import save_experiment
from validators import validate_response


def run_full_experiment(
    task: str,
    task_category: str,
    variants: dict[str, str],
    client,
    model: str,
    judge_model: str,
    reference_answer: str | None = None,
    evaluate: bool = True,
    save: bool = True,
    data_dir=None,
    max_retries: int = 3,
    on_progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run every variant, score it and optionally save the experiment.

    Returns the experiment record: id, created_at, metadata and results,
    where results use the dashboard's result format.
    """

    def report(message: str):
        if on_progress:
            on_progress(message)

    results = run_experiment(
        variants,
        client,
        model,
        max_retries=max_retries,
        on_progress=lambda i, n, t: report(f"Running {t} ({i}/{n})"),
        sleep=sleep,
    )

    for result in results:
        if result.status == "success":
            result.validation = validate_response(
                result.technique,
                task_category,
                task,
                result.response,
                reference_answer=reference_answer,
            )
        if evaluate:
            report(f"Evaluating {result.technique}")
            evaluate_result(
                result,
                task,
                client,
                judge_model,
                max_retries=max_retries,
                sleep=sleep,
            )

    metadata = {
        "task": task,
        "task_category": task_category,
        "techniques": list(variants),
        "model": model,
        "judge_model": judge_model if evaluate else None,
        "reference_answer": reference_answer or None,
    }
    rows = [result.to_dict() for result in results]

    if save:
        return save_experiment(rows, metadata, data_dir=data_dir)
    return {"id": None, "created_at": None, "metadata": metadata, "results": rows}
