import pytest

from src.experiments import EXPERIMENTS

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