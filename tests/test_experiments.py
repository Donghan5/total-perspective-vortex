import pytest
import re

from src.experiments import (
    EXPERIMENTS,
    resolve_single_run_task,
)

from src.evaluation import get_train_runs

def test_exactly_six_experiments():
    assert len(EXPERIMENTS) == 6, "There should be exactly six experiments defined."

def test_fold_runs_do_not_overlap():
    for exp in EXPERIMENTS.values():
        for held_out_index in range(len(exp.repetitions)):
            train_runs, test_runs = exp.get_fold_runs(held_out_index)
            assert set(train_runs).isdisjoint(test_runs), (
                f"Training and test runs overlap for experiment {exp.name} "
                f"with held_out_index {held_out_index}."
            )

def test_modality_fold_mapping():
    train_runs, test_runs = EXPERIMENTS[4].get_fold_runs(2)

    assert train_runs == [3, 4, 7, 8]
    assert test_runs == [11, 12]

@pytest.mark.parametrize(
    ("test_run", "expected_task_name", "expected_run_ids"),
    [
        (
            3,
            "actual_left_vs_right_fist",
            (3, 7, 11),
        ),
        (
            7,
            "actual_left_vs_right_fist",
            (3, 7, 11),
        ),
        (
            11,
            "actual_left_vs_right_fist",
            (3, 7, 11),
        ),
        (
            4,
            "imagined_left_vs_right_fist",
            (4, 8, 12),
        ),
        (
            8,
            "imagined_left_vs_right_fist",
            (4, 8, 12),
        ),
        (
            12,
            "imagined_left_vs_right_fist",
            (4, 8, 12),
        ),
        (
            5,
            "actual_fists_vs_feet",
            (5, 9, 13),
        ),
        (
            9,
            "actual_fists_vs_feet",
            (5, 9, 13),
        ),
        (
            13,
            "actual_fists_vs_feet",
            (5, 9, 13),
        ),
        (
            6,
            "imagined_fists_vs_feet",
            (6, 10, 14),
        ),
        (
            10,
            "imagined_fists_vs_feet",
            (6, 10, 14),
        ),
        (
            14,
            "imagined_fists_vs_feet",
            (6, 10, 14),
        )
    ]
)
def test_resolve_single_run_task(
        test_run,
        expected_task_name,
        expected_run_ids,
):
    task_name, run_ids = resolve_single_run_task(test_run)

    assert task_name == expected_task_name
    assert run_ids == expected_run_ids
    assert test_run in run_ids

@pytest.mark.parametrize(
    ("test_run", "expected_task_name", "expected_train_runs"),
    [
        (3, "actual_left_vs_right_fist", [7, 11]),
        (7, "actual_left_vs_right_fist", [3, 11]),
        (11, "actual_left_vs_right_fist", [3, 7]),
        (4, "imagined_left_vs_right_fist", [8, 12]),
        (8, "imagined_left_vs_right_fist", [4, 12]),
        (12, "imagined_left_vs_right_fist", [4, 8]),
        (5, "actual_fists_vs_feet", [9, 13]),
        (9, "actual_fists_vs_feet", [5, 13]),
        (13, "actual_fists_vs_feet", [5, 9]),
        (6, "imagined_fists_vs_feet", [10, 14]),
        (10, "imagined_fists_vs_feet", [6, 14]),
        (14, "imagined_fists_vs_feet", [6, 10]),
    ]
)
def test_get_train_runs(
        test_run,
        expected_task_name,
        expected_train_runs,
):
    task_name, train_runs = get_train_runs(test_run)
    assert task_name == expected_task_name
    assert train_runs == expected_train_runs
    assert test_run not in train_runs
    assert len(train_runs) == 2

@pytest.mark.parametrize("test_run", [2, 15])
def test_resolve_single_run_task_rejects_unknown_run(test_run):
    expected_message = (
        f"Unknown test_run: {test_run}. "
        "Supported runs are 3 through 14."
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        resolve_single_run_task(test_run)