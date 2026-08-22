import numpy as np

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
