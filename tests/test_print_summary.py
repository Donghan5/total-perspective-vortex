import pytest
from mybci import print_evaluation_summary

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

def test_subject_error_test_runs_schema(monkeypatch)