# Import necessary libraries
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from src.preprocessing import preprocess_subject_runs
from src.pipeline import create_pipeline

# Define constants
MODEL_DIR = Path("models")

from src.evaluation import (
    evaluate_all_experiments,
    get_train_runs,
    print_experiment_results,
    summarize_experiment_results
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG Motor Imagery Classification with CSP + LDA"
    )

    parser.add_argument("subject_id", type=int, help="Subject ID (1 to 109)", nargs="?")
    parser.add_argument("run_id", type=int, help="Run ID (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, or 14)", nargs="?")
    parser.add_argument("mode", choices=["train", "predict", "evaluate"], nargs="?")
    
    return parser.parse_args()

def get_model_path(subject_id: int, test_run: int) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"S{subject_id:03d}_R{test_run:02d}_pipeline.joblib"


def train_model(subject_id: int, test_run: int) -> None:
    """
    Train the model on the training runs for a given test run.
    """
    # Get the training runs for the given test run
    experiment_name, train_runs = get_train_runs(test_run)

    # Guard against leakage
    if test_run in train_runs:
        raise ValueError(f"Test run {test_run} cannot be in the training runs {train_runs}")
    
    X_train, y_train = preprocess_subject_runs(subject_id, train_runs)

    pipeline = create_pipeline()

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    pipeline.fit(X_train, y_train)

    artifact = {
        "pipeline": pipeline,
        "subject_id": subject_id,
        "experiment": experiment_name,
        "train_runs": train_runs,
        "test_run": test_run,
    }

    model_path = get_model_path(subject_id, test_run)
    joblib.dump(artifact, model_path)

    print(f"Experiment: {experiment_name}")
    print(f"Subject: S{subject_id:03d}")
    print(f"Training runs: {train_runs}")
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {np.mean(scores):.4f}")
    print(f"Model saved to: {model_path}")

def predict_stream(subject_id: int, test_run: int) -> None:
    """
    Predict the labels for the test run using the trained model.
    """

    # Load the trained model
    model_path = get_model_path(subject_id, test_run)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Load the trained model artifact
    artifact = joblib.load(model_path)

    # checking the metadata of the loaded artifact
    if artifact["subject_id"] != subject_id or artifact["test_run"] != test_run:
        raise ValueError(f"Loaded model metadata does not match the requestted subject_id {subject_id} and test_run {test_run}.")
    
    # Extract the pipeline from the artifact
    pipeline = artifact["pipeline"]

    # Load only the test run data
    X_test, y_test = preprocess_subject_runs(subject_id, [test_run])

    # Predict epoch by epoch
    predictions, latencies = [], []

    # Measure latency for each epoch
    for epoch, truth in zip(X_test, y_test):
        chunk = epoch[np.newaxis, ...]  # Add batch dimension

        start = time.perf_counter()
        pred = pipeline.predict(chunk)[0]
        elapsed = time.perf_counter() - start

        predictions.append(pred)
        latencies.append(elapsed)

    # Calculate accuracy and average latency
    accuracy = accuracy_score(y_test, predictions)
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    # Print final accuracy and max latency
    print(f"Subject: S{subject_id:03d}")
    print(f"Test run: R{test_run:02d}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average latency per epoch: {avg_latency:.4f} seconds")
    print(f"Maximum latency for any epoch: {max_latency:.4f} seconds")
    
def run_full_evaluation() -> None:
    """
    Run the full evaluation for all subjects and test runs.
    """
    results, errors = evaluate_all_experiments()
    summary = summarize_experiment_results(results, errors)
    print_experiment_results(summary)

def main() -> None:
    args = parse_args()

    if args.subject_id is None and args.run_id is None and args.mode is None:
        run_full_evaluation()
        return

    if args.mode == "evaluate" and args.subject_id is None and args.run_id is None:
        run_full_evaluation()
        return

    if args.subject_id is None or args.run_id is None or args.mode is None:
        raise SystemExit(
            "Usage:\n"
            "   python mybci.py <subject_id> <held_out_run> <mode> (train, predict)\n"
            "   python mybci.py\n"
            "   python mybci.py evaluate\n\n"
        )

    # Mode selecting
    if args.mode == "train":
        train_model(args.subject_id, args.run_id)
    elif args.mode == "predict":
        predict_stream(args.subject_id, args.run_id)
    elif args.mode == "evaluate":
        run_full_evaluation()

if __name__ == "__main__":
    main()