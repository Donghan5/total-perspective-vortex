import csv
import json
from pathlib import Path

import pytest

from src.results import (
    build_expected_evaluation_keys,
    save_evaluation_run,
    summarize_evaluation_results,
    validate_evaluation_results,
)


def make_result(
    subject_id: int,
    experiment_name: str,
    test_run: int,
    accuracy: float,
) -> dict:
    return {
        "subject_id": subject_id,
        "experiment_name": experiment_name,
        "train_runs": [7, 11],
        "test_run": test_run,
        "n_train_epochs": 30,
        "n_test_epochs": 15,
        "accuracy": accuracy,
    }


def test_build_expected_evaluation_keys():
    experiments = {
        "experiment_a": [3, 7, 11],
        "experiment_b": [4, 8, 12],
    }

    keys = build_expected_evaluation_keys([1, 2], experiments)

    assert len(keys) == 12
    assert (1, "experiment_a", 3) in keys
    assert (2, "experiment_b", 12) in keys


def test_validate_evaluation_results_rejects_duplicate_keys():
    result = make_result(1, "experiment_a", 3, 0.6)

    with pytest.raises(ValueError, match="Duplicate evaluation keys"):
        validate_evaluation_results([result, result.copy()])


def test_validate_evaluation_results_reports_missing_keys():
    expected_keys = {
        (1, "experiment_a", 3),
        (1, "experiment_a", 7),
    }
    results = [make_result(1, "experiment_a", 3, 0.6)]

    validation = validate_evaluation_results(results, expected_keys)

    assert validation["row_count"] == 1
    assert validation["unique_key_count"] == 1
    assert validation["missing_key_count"] == 1
    assert validation["missing_keys"] == [[1, "experiment_a", 7]]


def test_summarize_evaluation_results_matches_population_std():
    results = [
        make_result(1, "experiment_a", 3, 0.5),
        make_result(2, "experiment_a", 3, 1.0),
    ]

    summary = summarize_evaluation_results(results, [])

    assert summary["successful_evaluations"] == 2
    assert summary["errored_evaluations"] == 0
    assert summary["mean_accuracy"] == pytest.approx(0.75)
    assert summary["median_accuracy"] == pytest.approx(0.75)
    assert summary["standard_deviation"] == pytest.approx(0.25)
    assert summary["minimum_accuracy"] == pytest.approx(0.5)
    assert summary["maximum_accuracy"] == pytest.approx(1.0)
    assert summary["mean_accuracy_by_experiment"]["experiment_a"] == pytest.approx(0.75)


def test_save_evaluation_run_writes_pairable_artifacts(tmp_path: Path):
    results = [
        make_result(1, "experiment_a", 3, 0.6),
        make_result(1, "experiment_a", 7, 0.8),
    ]
    expected_keys = {
        (1, "experiment_a", 3),
        (1, "experiment_a", 7),
    }

    paths = save_evaluation_run(
        results=results,
        errors=[],
        config_id="test_config",
        config={
            "csp": {
                "n_components": 4,
                "reg": 0.01,
            }
        },
        expected_keys=expected_keys,
        output_dir=tmp_path,
        project_root=tmp_path,
    )

    assert paths["evaluations"].exists()
    assert paths["errors"].exists()
    assert paths["metadata"].exists()

    with paths["evaluations"].open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["config_id"] == "test_config"
    assert rows[0]["subject_id"] == "1"
    assert rows[0]["experiment_name"] == "experiment_a"
    assert rows[0]["test_run"] == "3"
    assert json.loads(rows[0]["train_runs"]) == [7, 11]

    with paths["metadata"].open(encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["config_id"] == "test_config"
    assert metadata["validation"]["row_count"] == 2
    assert metadata["validation"]["missing_key_count"] == 0
    assert metadata["validation"]["unexpected_key_count"] == 0
    assert metadata["summary"]["mean_accuracy"] == pytest.approx(0.7)


def test_save_evaluation_run_rejects_unexplained_missing_rows(tmp_path: Path):
    results = [make_result(1, "experiment_a", 3, 0.6)]
    expected_keys = {
        (1, "experiment_a", 3),
        (1, "experiment_a", 7),
    }

    with pytest.raises(ValueError, match="key validation failed"):
        save_evaluation_run(
            results=results,
            errors=[],
            config_id="test_config",
            config={},
            expected_keys=expected_keys,
            output_dir=tmp_path,
            project_root=tmp_path,
        )
