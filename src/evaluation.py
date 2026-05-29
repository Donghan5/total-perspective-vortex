import numpy as np
from sklearn.model_selection import cross_val_score

from src.pipeline import create_pipeline
from src.preprocessing import preprocess_subject_runs


EXPERIMENTS = {
    "actual_left_vs_right_fist": [3, 7, 11],
    "imagined_left_vs_right_fist": [4, 8, 12],
    "actual_fists_vs_feet": [5, 9, 13],
    "imagined_fists_vs_feet": [6, 10, 14],
}


def find_experiment_from_run(run_id: int) -> tuple[str, list[int]]:
    for experiment_name, run_ids in EXPERIMENTS.items():
        if run_id in run_ids:
            return experiment_name, run_ids

    raise ValueError(f"Run ID {run_id} does not belong to any experiment.")


def get_train_runs(test_run: int) -> tuple[str, list[int]]:
    """
    Return the experiment name and training runs after holding out one test run.
    """
    experiment_name, run_ids = find_experiment_from_run(test_run)
    train_runs = [run_id for run_id in run_ids if run_id != test_run]

    return experiment_name, train_runs


def evaluate_held_out_run(subject_id: int, test_run: int) -> dict:
    """
    Evaluate one subject on one unseen held-out run.
    """
    experiment_name, train_runs = get_train_runs(test_run)

    X_train, y_train = preprocess_subject_runs(subject_id, train_runs)
    X_test, y_test = preprocess_subject_runs(subject_id, [test_run])

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    accuracy = float(pipeline.score(X_test, y_test))

    return {
        "subject_id": subject_id,
        "experiment": experiment_name,
        "train_runs": train_runs,
        "test_run": test_run,
        "accuracy": accuracy,
    }


def evaluate_subject_all_experiments(subject_id: int) -> list[dict]:
    """
    Evaluate one subject on every configured experiment and held-out run.
    """
    results = []

    for run_ids in EXPERIMENTS.values():
        for test_run in run_ids:
            results.append(evaluate_held_out_run(subject_id, test_run))

    return results


def evaluate_all_experiments(
    subject_range=range(1, 110),
) -> list[dict]:
    """
    Evaluate all configured experiments for all requested subjects.
    """
    all_results = []

    for subject_id in subject_range:
        all_results.extend(evaluate_subject_all_experiments(subject_id))

    return all_results


def evaluate_cross_validation_baseline(
    pipeline,
    run_ids: list[int],
    subject_range=range(1, 110),
) -> dict[int, float | None]:
    """
    Existing epoch-level cross-validation baseline.
    This is not held-out run evaluation.
    """
    subject_ids = list(subject_range)
    total_subjects = len(subject_ids)
    mean_acc_dict = {}

    for subject_id in subject_ids:
        try:
            X, y = preprocess_subject_runs(subject_id, run_ids)
            scores = cross_val_score(pipeline, X, y, cv=5)

            mean_accuracy = float(scores.mean())
            mean_acc_dict[subject_id] = mean_accuracy

            print(
                f"Subject {subject_id:03d}: "
                f"mean accuracy = {mean_accuracy:.3f}"
            )

        except Exception as error:
            print(f"Subject {subject_id:03d}: ERROR {error}")
            mean_acc_dict[subject_id] = None

    valid_scores = [
        score for score in mean_acc_dict.values()
        if score is not None
    ]

    if not valid_scores:
        raise RuntimeError("No subject could be evaluated.")

    print("=== Cross-Validation Baseline Result ===")
    print(f"Subjects evaluated: {len(valid_scores)}/{total_subjects}")
    print()
    print(f"Mean:   {np.mean(valid_scores):.4f}")
    print(f"Median: {np.median(valid_scores):.4f}")
    print(f"Std:    {np.std(valid_scores):.4f}")
    print(f"Min:    {np.min(valid_scores):.4f}")
    print(f"Max:    {np.max(valid_scores):.4f}")
    print()
    print(f">= 60%: {sum(score >= 0.6 for score in valid_scores)}/{total_subjects}")
    print(f">= 70%: {sum(score >= 0.7 for score in valid_scores)}/{total_subjects}")
    print(f">= 80%: {sum(score >= 0.8 for score in valid_scores)}/{total_subjects}")

    return mean_acc_dict