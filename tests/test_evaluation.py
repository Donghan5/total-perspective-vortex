import numpy as np
import pytest
from src import evaluation


class FakePipeline:
    def fit(self, X, y):
        return self

    def score(self, X, y):
        return 0.5


def test_evaluate_subject_experiment_returns_all_18_folds(
        monkeypatch,
) -> None:
    run_cache = {
        run_id: (
            np.full((4, 2, 3), run_id, dtype=float),
            np.array([0, 1, 0, 1], dtype=np.int64),
        )
        for run_id in evaluation.ALL_TASK_RUNS
    }

    monkeypatch.setattr(
        evaluation,
        "create_pipeline",
        lambda: FakePipeline(),
    )

    results, errors = evaluation.evaluate_subject_experiment(
        subject_id=1,
        run_cache=run_cache,
    )

    assert errors == []
    assert len(results) == 18
    assert {
        (result["experiment_id"], result["held_out_index"])
        for result in results
    } == {
        (experiment_id, held_out_index)
        for experiment_id in range(6)
        for held_out_index in range(3)
    }

    for result in results:
        assert result["subject_id"] == 1
        assert 0.0 <= result["accuracy"] <= 1.0
        assert set(result["train_runs"]).isdisjoint(
            result["test_runs"]
        )

def test_combine_modality_repetitions_labels_actual_and_imagined():
    run_cache = {
        3: (
            np.full((2, 1, 1), 3.0),
            np.array([10, 10]),
        ),
        4: (
            np.full((3, 1, 1), 4.0),
            np.array([20, 20, 20]),
        )
    }

    X, y = evaluation.combine_modality_repetitions(run_cache, [(3, 4)])

    np.testing.assert_array_equal(
        X[:, 0, 0],
        np.array([3.0, 3.0, 4.0, 4.0, 4.0])
    )

    np.testing.assert_array_equal(
        y,
        np.array([0, 0, 1, 1, 1]),
    )

def test_all_task_runs_contains_every_supported_run():
    assert evaluation.ALL_TASK_RUNS == tuple(range(3, 15))

def test_combine_cached_runs_rejects_empty_runs():
    with pytest.raises(
        ValueError,
        match="run_ids must contain at least one run",
    ):
        evaluation.combine_cached_runs({}, [])

def test_combine_cached_runs_reject_missing_runs():
    run_cache = {
        3: (
            np.zeros((2, 1, 1)),
            np.array([0, 1]),
        ),
    }
    with pytest.raises(
        ValueError,
        match=r"Runs not found in cache: \[4\]",
    ):
        evaluation.combine_cached_runs(run_cache, [3, 4])

def test_combine_cached_runs_preserves_requested_order():
    run_cache = {
        3: (
            np.full((2, 1, 1), 3.0),
            np.array([10, 11]),
        ),
        4: (
            np.full((1, 1, 1), 4.0),
            np.array([20]),
        ),
    }

    X, y = evaluation.combine_cached_runs(run_cache, [4, 3])

    np.testing.assert_array_equal(
        X[:, 0, 0],
        np.array([4.0, 3.0, 3.0])
    )
    np.testing.assert_array_equal(
        y,
        np.array([20, 10, 11])
    )