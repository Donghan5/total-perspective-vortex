from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULT_KEY_FIELDS = (
    "subject_id",
    "experiment_name",
    "test_run",
)

RESULT_FIELDS = (
    "subject_id",
    "experiment_name",
    "train_runs",
    "test_run",
    "n_train_epochs",
    "n_test_epochs",
    "accuracy",
)


def build_expected_evaluation_keys(
    subject_ids: Iterable[int],
    experiments: dict[str, list[int]],
) -> set[tuple[int, str, int]]:
    """Build the expected subject/experiment/test-run key set."""
    return {
        (int(subject_id), experiment_name, int(test_run))
        for subject_id in subject_ids
        for experiment_name, run_ids in experiments.items()
        for test_run in run_ids
    }


def validate_evaluation_results(
    results: list[dict[str, Any]],
    expected_keys: set[tuple[int, str, int]] | None = None,
) -> dict[str, Any]:
    """Validate row-level evaluation results before persistence."""
    actual_keys: set[tuple[int, str, int]] = set()
    duplicate_keys: list[tuple[int, str, int]] = []

    for row_index, result in enumerate(results):
        missing_fields = [field for field in RESULT_FIELDS if field not in result]
        if missing_fields:
            raise ValueError(
                f"Result row {row_index} is missing fields: {missing_fields}."
            )

        key = (
            int(result["subject_id"]),
            str(result["experiment_name"]),
            int(result["test_run"]),
        )

        if key in actual_keys:
            duplicate_keys.append(key)
        actual_keys.add(key)

        accuracy = float(result["accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                f"Result row {row_index} has invalid accuracy: {accuracy}."
            )

    if duplicate_keys:
        raise ValueError(
            "Duplicate evaluation keys found: "
            f"{sorted(set(duplicate_keys))[:10]}."
        )

    missing_keys: set[tuple[int, str, int]] = set()
    unexpected_keys: set[tuple[int, str, int]] = set()

    if expected_keys is not None:
        missing_keys = expected_keys - actual_keys
        unexpected_keys = actual_keys - expected_keys

    return {
        "row_count": len(results),
        "unique_key_count": len(actual_keys),
        "duplicate_key_count": 0,
        "expected_key_count": (
            len(expected_keys) if expected_keys is not None else None
        ),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "missing_keys": [list(key) for key in sorted(missing_keys)],
        "unexpected_keys": [list(key) for key in sorted(unexpected_keys)],
    }


def get_git_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the current Git commit and whether the working tree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return {
            "commit": commit,
            "dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unknown",
            "dirty": None,
        }


def summarize_evaluation_results(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the same aggregate metrics used by the console summary."""
    accuracies = [float(result["accuracy"]) for result in results]

    if not accuracies:
        return {
            "successful_evaluations": 0,
            "errored_evaluations": len(errors),
            "mean_accuracy": None,
            "median_accuracy": None,
            "standard_deviation": None,
            "minimum_accuracy": None,
            "maximum_accuracy": None,
            "mean_accuracy_by_experiment": {},
        }

    experiment_names = sorted(
        {str(result["experiment_name"]) for result in results}
    )
    experiment_means = {
        experiment_name: statistics.fmean(
            float(result["accuracy"])
            for result in results
            if result["experiment_name"] == experiment_name
        )
        for experiment_name in experiment_names
    }

    return {
        "successful_evaluations": len(results),
        "errored_evaluations": len(errors),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "standard_deviation": statistics.pstdev(accuracies),
        "minimum_accuracy": min(accuracies),
        "maximum_accuracy": max(accuracies),
        "mean_accuracy_by_experiment": experiment_means,
    }


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _to_json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            _to_json_compatible(payload),
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    temporary_path.replace(path)


def _write_results_csv(
    path: Path,
    results: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> None:
    fieldnames = (
        "config_id",
        "config_hash",
        "timestamp_utc",
        "git_commit",
        "git_dirty",
        *RESULT_FIELDS,
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "config_id": run_metadata["config_id"],
                "config_hash": run_metadata["config_hash"],
                "timestamp_utc": run_metadata["timestamp_utc"],
                "git_commit": run_metadata["git"]["commit"],
                "git_dirty": run_metadata["git"]["dirty"],
                **result,
            }
            row["train_runs"] = json.dumps(
                result["train_runs"],
                separators=(",", ":"),
            )
            writer.writerow({field: row.get(field) for field in fieldnames})

    temporary_path.replace(path)


def save_evaluation_run(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    config_id: str,
    config: dict[str, Any],
    expected_keys: set[tuple[int, str, int]] | None = None,
    output_dir: Path = RESULTS_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Persist row-level results, errors, metadata, and validation details."""
    if not config_id or any(character.isspace() for character in config_id):
        raise ValueError("config_id must be a non-empty string without spaces.")

    validation = validate_evaluation_results(results, expected_keys)
    if not errors and (
        validation["missing_key_count"] > 0
        or validation["unexpected_key_count"] > 0
    ):
        raise ValueError(
            "Evaluation key validation failed without recorded errors: "
            f"{validation}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    timestamp_utc = timestamp.isoformat()
    timestamp_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    normalized_config = _to_json_compatible(config)
    config_payload = json.dumps(
        normalized_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    config_hash = hashlib.sha256(
        config_payload.encode("utf-8")
    ).hexdigest()[:12]

    run_name = f"{config_id}_{config_hash}_{timestamp_token}"
    result_path = output_dir / f"{run_name}_evaluations.csv"
    error_path = output_dir / f"{run_name}_errors.json"
    metadata_path = output_dir / f"{run_name}_metadata.json"

    run_metadata = {
        "schema_version": 1,
        "run_name": run_name,
        "config_id": config_id,
        "config_hash": config_hash,
        "timestamp_utc": timestamp_utc,
        "git": get_git_state(project_root),
        "config": normalized_config,
        "summary": summarize_evaluation_results(results, errors),
        "validation": validation,
        "files": {
            "evaluations_csv": str(result_path),
            "errors_json": str(error_path),
            "metadata_json": str(metadata_path),
        },
    }

    _write_results_csv(result_path, results, run_metadata)
    _write_json(error_path, errors)
    _write_json(metadata_path, run_metadata)

    return {
        "evaluations": result_path,
        "errors": error_path,
        "metadata": metadata_path,
    }
