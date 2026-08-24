import pytest
from mybci import print_evaluation_summary
from src.evaluation import evaluate_all_experiments

@pytest.mark.parametrize(
    ("test_runs", "expected_output"),
    [
        ([14], "Test runs: R14"),
        ([13, 14], "Test runs: R13, R14"),
        (None, "Test runs: None"),
    ]
)
def test_print_evaluation_summary_formats_test_runs(
        test_runs,
        expected_output,
        capsys,
):
    errors = [
        {
            "subject_id": 1,
            "experiment_id": None,
            "experiment_name": None,
            "held_out_index": None,
            "test_runs": test_runs,
            "error": "test failure",
        }
    ]

    print_evaluation_summary([], errors)

    captured = capsys.readouterr()

    assert expected_output in captured.out
    assert "test failure" in captured.out


def test_print_evaluation_summary_prints_mean_of_experiment_means(capsys):
    results = [
        {"experiment_name": "experiment_0", "accuracy": 0.0},
        {"experiment_name": "experiment_0", "accuracy": 0.0},
        {"experiment_name": "experiment_1", "accuracy": 1.0},
        {"experiment_name": "experiment_2", "accuracy": 1.0},
        {"experiment_name": "experiment_3", "accuracy": 1.0},
        {"experiment_name": "experiment_4", "accuracy": 1.0},
        {"experiment_name": "experiment_5", "accuracy": 1.0},
    ]

    print_evaluation_summary(results, [])

    captured = capsys.readouterr()

    assert "Mean accuracy of 6 experiments: 0.8333" in captured.out


def test_subject_error_test_runs_schema(monkeypatch):
    def raise_loading_error(subject_id):
        raise RuntimeError("loading failed")
    monkeypatch.setattr(
        "src.evaluation.build_subject_run_cache",
        raise_loading_error,
    )
    results, errors = evaluate_all_experiments(subject_range=[1])
    assert results == []
    assert len(errors) == 1

    error = errors[0]

    assert error.keys() == {
        "subject_id",
        "experiment_id",
        "experiment_name",
        "held_out_index",
        "test_runs",
        "error",
    }
    assert error["subject_id"] == 1
    assert error["test_runs"] is None
    assert "loading failed" in error["error"]
