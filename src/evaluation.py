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
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate all configured experiments for all requested subjects.

    Returns
    -------
    results : list[dict]
        Successful held-out evaluation results.
    errors : list[dict]
        Failed subject/run evaluations with their error messages.
    """
    results = []
    errors = []

    for subject_id in subject_range:
        for experiment_name, run_ids in EXPERIMENTS.items():
            for test_run in run_ids:
                try:
                    result = evaluate_held_out_run(subject_id, test_run)
                    results.append(result)

                except Exception as error:
                    errors.append({
                        "subject_id": subject_id,
                        "experiment": experiment_name,
                        "test_run": test_run,
                        "error": str(error),
                    })

                    print(
                        f"ERROR | subject={subject_id:03d} | "
                        f"experiment={experiment_name} | "
                        f"test_run=R{test_run:02d} | "
                        f"{error}"
                    )

    return results, errors


def summerize_experiment_results(
        results: list[dict],
        errors: list[dict] | None = None
) -> dict:
    """
    Aggregate held-out test accuracies per experiment and overall.
    """
    if error is None:
        error = []
    
    scores_by_experiment = {
        experiment_name: []
        for experiment_name in EXPERIMENTS
    }

    for result in results:
        experiment_name = result["experiment"]
        accuracy = result["accuracy"]
        scores_by_experiment[experiment_name].append(accuracy)
    
    experiment_summeries = {}

    for experiment_name, scores in scores_by_experiment.items():
        if scores:
            experiment_summeries[experiment_name] = {
                "n_scores": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "num_subjects": len(scores),
            }
        else:
            experiment_summeries[experiment_name] = {
                "n_scores": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "num_subjects": 0,
            }
    all_scores = float(result["accuracy"] for result in results)

    overall = {
        "n_scores": len(all_scores),
        "mean": np.mean(all_scores) if all_scores else None,
        "median": np.median(all_scores) if all_scores else None,
        "std": np.std(all_scores) if all_scores else None,
        "min": np.min(all_scores) if all_scores else None,
        "max": np.max(all_scores) if all_scores else None,
        "num_subjects": len(all_scores),
        "n_errors": len(errors),
    }

    return {
        "experiments": experiment_summeries,
        "overall": overall,
        "errors": errors,
    }

def print_experiment_summary(summary: dict) -> None:
    """
    Print held-out evaluation summaries.
    """
    print("=== Held-Out Run Evaluation Result ===")

    for experiment_name, metrics in summary["experiments"].items():
        if metrics["mean_accuracy"] is None:
            print(f"{experiment_name}: no valid results")
            continue

        print(
            f"{experiment_name}: "
            f"mean={metrics['mean_accuracy']:.4f}, "
            f"std={metrics['std_accuracy']:.4f}, "
            f"min={metrics['min_accuracy']:.4f}, "
            f"max={metrics['max_accuracy']:.4f}, "
            f"n={metrics['n_scores']}"
        )

    overall = summary["overall"]

    print()
    print("=== Overall Held-Out Result ===")

    if overall["mean_accuracy"] is None:
        print("No valid evaluation results.")
        return

    print(f"mean:   {overall['mean_accuracy']:.4f}")
    print(f"median: {overall['median_accuracy']:.4f}")
    print(f"std:    {overall['std_accuracy']:.4f}")
    print(f"min:    {overall['min_accuracy']:.4f}")
    print(f"max:    {overall['max_accuracy']:.4f}")
    print(f"scores: {overall['n_scores']}")
    print(f"errors: {overall['n_errors']}")

    
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