import numpy as np

from src.preprocessing import preprocess_eeg_data
from src.pipeline.pipeline import create_pipeline
from src.experiments import (
    EXPERIMENTS,
    ExperimentSpec,
    resolve_single_run_task
)

ALL_TASK_RUNS = tuple(sorted({
    run_id
    for experiment in EXPERIMENTS.values()
    for repetition in experiment.repetitions
    for run_id in repetition
}))

EEGDataset = tuple[np.ndarray, np.ndarray]
RunCache = dict[int, EEGDataset]


def build_subject_run_cache(
        subject_id: int,
) -> RunCache:
    """ Build a cache of preprocessed data for all runs of a subject. """
    return {
        run_id: preprocess_eeg_data(subject_id, run_id) for run_id in ALL_TASK_RUNS
    }

def combine_cached_runs(
        run_cache: RunCache,
        run_ids: list[int]
) -> EEGDataset:
    if not run_ids:
        raise ValueError("run_ids must contain at least one run.")
    missing_runs = [run_id for run_id in run_ids if run_id not in run_cache]
    if missing_runs:
        raise ValueError(f"Runs not found in cache: {missing_runs}")

    X = np.concatenate([run_cache[run_id][0] for run_id in run_ids], axis=0)

    y = np.concatenate([run_cache[run_id][1] for run_id in run_ids], axis=0)

    return X, y

def get_train_runs(
        test_run: int,
) -> tuple[str, list[int]]:
    task_name, run_ids = resolve_single_run_task(test_run)

    train_runs = [
        run_id
        for run_id in run_ids
        if run_id != test_run
    ]

    if len(train_runs) != 2:
        raise ValueError(
            f"Expected exactly 2 training runs for {test_run}, "
            f"got {len(train_runs)} runs."
        )

    return task_name, train_runs

def combine_modality_repetitions(
        run_cache: RunCache,
        repetitions: list[tuple[int, ...]],
) -> EEGDataset:
    if not repetitions:
        raise ValueError(
            "repetitions must contain at least one repetitions."
        )
    X_list = []
    y_list = []

    for repetition in repetitions:
        if len(repetition) != 2:
            raise ValueError(
                "A modality repetition must contain "
                "one actual run and one imagined run."
            )

        actual_run, imagined_run = repetition

        missing_runs = [
            run_id
            for run_id in (actual_run, imagined_run)
            if run_id not in run_cache
        ]

        if missing_runs:
            raise ValueError(
                f"Runs not found in cache: {missing_runs}"
            )

        X_actual = run_cache[actual_run][0]
        X_imagined = run_cache[imagined_run][0]


        X_list.extend([X_actual, X_imagined])
        y_list.extend([
            np.zeros(
                len(X_actual),
                dtype=np.int64
            ),
            np.ones(
                len(X_imagined),
                dtype=np.int64
            )
        ])

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def evaluate_held_out_fold(
        subject_id: int,
        experiment: ExperimentSpec,
        held_out_index: int,
        run_cache: RunCache
) -> dict:
    """
    Evaluate the model on a held-out run.
    Just handle one subject and one test run
    """
    train_runs, test_runs = experiment.get_fold_runs(held_out_index)

    if experiment.label_strategy == "event":
        X_train, y_train = combine_cached_runs(run_cache, train_runs)
        X_test, y_test = combine_cached_runs(run_cache, test_runs)
    elif experiment.label_strategy == "modality":
        train_repetitions, test_repetitions = experiment.get_fold_repetitions(held_out_index)
        listed_test_repetitions = [test_repetitions]
        X_train, y_train = combine_modality_repetitions(run_cache, train_repetitions)
        X_test, y_test = combine_modality_repetitions(run_cache, listed_test_repetitions)
    else:
        raise ValueError(
            "Unknown label strategy: "
            f"{experiment.label_strategy}"
        )
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    return {
        "subject_id": subject_id,
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.name,
        "held_out_index": held_out_index,
        "train_runs": train_runs,
        "test_runs": test_runs,
        "n_train_epochs": len(y_train),
        "n_test_epochs": len(y_test),
        "accuracy": float(accuracy),
    }

def evaluate_subject_experiment(subject_id: int, run_cache: RunCache) -> tuple[list[dict], list[dict]]:
    """Evaluate all runs of a subject for each experiment."""
    results, errors = [], []
    for experiment in EXPERIMENTS.values():
        for held_out_index in range(len(experiment.repetitions)):
            try:
                result = evaluate_held_out_fold(subject_id, experiment, held_out_index, run_cache)
                results.append(result)
            except Exception as e:
                errors.append({
                    "subject_id": subject_id,
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.name,
                    "held_out_index": held_out_index,
                    "test_runs": list(
                        experiment.repetitions[held_out_index]
                    ),
                    "error": (
                        "Failed to evaluate held-out fold: "
                        f"{e}"
                    ),
                })
    return results, errors


def evaluate_all_experiments(
        subject_range=range(1, 110),
) -> tuple[list[dict], list[dict]]:
    """
    Args:
    - subject_range: Range of subject IDs to evaluate.

    Evaluate all experiments and return the results as a list of a dictionary
    """
    subject_ids = list(subject_range)

    results, errors = [], []

    for subject_id in subject_ids:
        try:
            run_cache = build_subject_run_cache(subject_id)
            subject_results, subject_errors = evaluate_subject_experiment(subject_id, run_cache)
            results.extend(subject_results)
            errors.extend(subject_errors)
        except Exception as e:
            errors.append({
                "subject_id": subject_id,
                "experiment_id": None,
                "experiment_name": None,
                "held_out_index": None,
                "test_runs": None,
                "error": (
                    "Failed to evaluate subject: "
                    f"{e}"
                )
            })

    return results, errors

def print_evaluation_summary(
        results: list[dict],
        errors: list[dict],
) -> None:

    accuracies = [
        result["accuracy"]
        for result in results
    ]

    print("----- Overall Evalutaion Summary -----")
    print(f"Successful evalutaions: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Mean accuarcy: {np.mean(accuracies):.4f}")
    print(f"Median accuarcy: {np.median(accuracies):.4f}")
    print(f"std: {np.std(accuracies):.4f}")
    print(f"min: {np.min(accuracies):.4f}")
    print(f"max: {np.max(accuracies):.4f}")

    print("\n=== Mean Accuracy by Experiment ===")

    experiment_means = []

    for experiment in EXPERIMENTS.values():
        scores = [
            result["accuracy"]
            for result in results
            if result["experiment_id"]
            == experiment.experiment_id
        ]

        if not scores:
            continue

        mean_accuracy = np.mean(scores)
        experiment_means.append(mean_accuracy)

        print(
            f"Experiment {experiment.experiment_id} "
            f"({experiment.name})"
            f"{mean_accuracy:.4f}"
        )

        if experiment_means:
            print(
                "\nMean of experiment_means: "
                f"{np.mean(experiment_means):.4f}"
            )

            print(
                "Experiment with mean accuracy over equal than 60%: "
                f"{sum(score >= 0.60 for score in experiment_means)}"
                f"/{len(experiment_means)}"
            )

    print("\n=== Mean Accuracy by Experiment ===")

    subject_means = []

    subject_ids = sorted({
        result["subject_id"]
        for result in results
    })

    for subject_id in subject_ids:
        scores = [
            result["accuracy"]
            for result in results
            if result["subject_id"] == subject_id
        ]

        if not scores:
            continue

        subject_means.append(
            float(np.mean(scores))
        )

    if subject_means:
        print(
            f"Subject evaluated: "
            f"{len(subject_means)}/109"
        )

        print(
            "Subject with mean accuracy over equal than 60%: "
            f"{sum(score >= 0.60 for score in subject_means)}"
            f"/{len(subject_means)}"
        )

        print(
            "Maximum subject mean accuracy: "
            f"{max(subject_means):.4f}"
        )