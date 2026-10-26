"""Save and load PromptBench experiments as JSON files."""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ExperimentNotFoundError(KeyError):
    """Raised when an experiment ID has no saved file."""


def _data_dir(data_dir: str | os.PathLike | None) -> Path:
    if data_dir is None:
        from config import get_settings

        data_dir = get_settings().data_dir
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _experiment_path(experiment_id: str, data_dir) -> Path:
    if not isinstance(experiment_id, str) or not _ID_PATTERN.match(experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    return _data_dir(data_dir) / f"{experiment_id}.json"


def new_experiment_id() -> str:
    return uuid.uuid4().hex


def save_experiment(
    results: list[dict],
    metadata: dict,
    data_dir: str | os.PathLike | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Write an experiment to disk and return the saved record."""

    record = {
        "id": experiment_id or new_experiment_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata),
        "results": list(results),
    }
    path = _experiment_path(record["id"], data_dir)

    # Write to a temp file first so a crash never leaves a half-written file.
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return record


def load_experiment(
    experiment_id: str, data_dir: str | os.PathLike | None = None
) -> dict:
    """Load one saved experiment by ID."""

    path = _experiment_path(experiment_id, data_dir)
    if not path.exists():
        raise ExperimentNotFoundError(experiment_id)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def list_experiments(data_dir: str | os.PathLike | None = None) -> list[dict]:
    """Return summaries of saved experiments, newest first.

    Unreadable files are skipped rather than breaking the list.
    """

    summaries = []
    for path in _data_dir(data_dir).glob("*.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            summaries.append(
                {
                    "id": record["id"],
                    "created_at": record["created_at"],
                    "metadata": record.get("metadata", {}),
                    "result_count": len(record.get("results", [])),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue

    summaries.sort(key=lambda item: item["created_at"], reverse=True)
    return summaries


def delete_experiment(
    experiment_id: str, data_dir: str | os.PathLike | None = None
) -> None:
    path = _experiment_path(experiment_id, data_dir)
    if not path.exists():
        raise ExperimentNotFoundError(experiment_id)
    path.unlink()
