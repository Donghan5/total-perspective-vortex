# Import necessary libraries
import argparse

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

from src.preprocessing import preprocess_subject_runs
from src.pipeline import create_pipeline

from src.prediction import predict_stream
from src.utils import get_model_path

from src.evaluation import (
    evaluate_all_experiments,
    get_train_runs,
)

from src.dataset.physionet import load_physionet_epochs
from src.dataset.validation import validate_eeg_dataset

from src.pipeline.bonus_pipeline import create_wavelet_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG Motor Imagery Classification with CSP + LDA"
    )

    parser.add_argument("subject_id", type=int, help="Subject ID (1 to 109)", nargs="?")
    parser.add_argument("run_id", type=int, help="Run ID (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, or 14)", nargs="?")
    parser.add_argument("mode", choices=["train", "predict"], nargs="?")
    parser.add_argument("--pipeline", choices=["csp", "wavelet"], default="csp", help="Pipeline type: 'csp' or 'wavelet' (default: 'csp')")
    
    return parser.parse_args()

def create_selected_pipeline(pipeline_name: str, sfreq: float = 160.0):
    """
    Create a pipeline based on the selected pipeline name.
    Default is the standard CSP + LDA pipeline.
    """
    if (pipeline_name == "csp"):
        return create_pipeline()
    elif (pipeline_name == "wavelet"):
        return create_wavelet_pipeline(sfreq=sfreq)
    else:
        raise ValueError(f"Unknown pipeline name: {pipeline_name}. Supported names are 'csp' and 'wavelet'.")

def train_model(subject_id: int, test_run: int, pipeline_name: str = "csp", sfreq: float = 160.0) -> None:
    """
    Train the model on the training runs for a given test run.
    """
    # Get the training runs for the given test run
    experiment_name, train_runs = get_train_runs(test_run)

    # Guard against leakage
    if test_run in train_runs:
        raise ValueError(f"Test run {test_run} cannot be in the training runs {train_runs}")
    
    X_train, y_train = preprocess_subject_runs(subject_id, train_runs)

    train_dataset = load_physionet_epochs(
        subject_id=subject_id,
        run_ids=train_runs,
        task_name=experiment_name
    )
    
    pipeline = create_selected_pipeline(pipeline_name, sfreq=sfreq)

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    pipeline.fit(X_train, y_train)

    artifact = {
        "pipeline": pipeline,
        "pipeline_name": pipeline_name,
        "subject_id": subject_id,
        "experiment": experiment_name,
        "train_runs": train_runs,
        "test_run": test_run
    }

    model_path = get_model_path(subject_id, test_run, pipeline_name)
    joblib.dump(artifact, model_path)

    print(f"Experiment: {experiment_name}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Subject: S{subject_id:03d}")
    print(f"Training runs: {train_runs}")
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {np.mean(scores):.4f}")
    print(f"Model saved to: {model_path}")

def print_evaluation_summary(results: list[dict], errors: list[dict]) -> None:
    """
    Helper function to print the evaluation summary in a readable format.
    """
    print("=== Evaluation Summary ===")
    print(f"Successful evaluations: {len(results)}")
    print(f"Errored evaluations: {len(errors)}")

    if results:
        accuracies = [r["accuracy"] for r in results]
        print(f"Mean accuracy: {np.mean(accuracies):.4f}")
        print(f"Median accuracy: {np.median(accuracies):.4f}")
        print(f"Standard deviation: {np.std(accuracies):.4f}")
        print(f"Minimum accuracy: {np.min(accuracies):.4f}")
        print(f"Maximum accuracy: {np.max(accuracies):.4f}")

        print("\n=== Mean accuracy by experiment ===")
        experiment_names = sorted(set(r["experiment_name"] for r in results))
        for exp_name in experiment_names:
            scores = [
                r["accuracy"] for r in results
                if r["experiment_name"] == exp_name
            ]
            print(f"  {exp_name}: mean accuracy = {np.mean(scores):.4f}")

    if errors:
        print("\n=== First Errors ===")
        for error in errors[:10]:
            print(
                f"Subject S{error['subject_id']:03d}, "
                f"Experiment: {error['experiment_name']}, "
                f"Test run: R{error['test_run']:02d}, "
                f"Error: {error['error']}"
            )

    
def run_full_evaluation() -> None:
    """
    Run the full evaluation for all subjects and test runs.
    """
    results, errors = evaluate_all_experiments()
    print_evaluation_summary(results, errors)

def main() -> None:
    args = parse_args()

    if args.subject_id is None and args.run_id is None and args.mode is None:
        run_full_evaluation()
        return

    if args.subject_id is None or args.run_id is None or args.mode is None:
        raise SystemExit(
            "Usage:\n"
            "   python mybci.py <subject_id> <held_out_run> train\n"
            "   python mybci.py <subject_id> <held_out_run> train --pipeline wavelet\n"
            "   python mybci.py <subject_id> <held_out_run> predict\n"
            "   python mybci.py\n"
        )

    # Mode selecting
    if args.mode == "train":
        train_model(args.subject_id, args.run_id, args.pipeline, sfreq=args.sfreq)
    elif args.mode == "predict":
        predict_stream(args.subject_id, args.run_id, args.pipeline)

if __name__ == "__main__":
    main()