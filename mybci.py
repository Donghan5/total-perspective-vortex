# Import necessary libraries
import argparse

import joblib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from src.pipeline.pipeline import create_pipeline

from src.prediction import predict_stream
from src.utils import get_model_path

from src.evaluation import (
    evaluate_all_experiments,
    get_train_runs,
)

from src.dataset.physionet import load_physionet_epochs
from src.dataset.validation import validate_eeg_dataset

from src.pipeline.bonus_pipeline import create_wavelet_pipeline

def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the BCI pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "EEG Motor Imagery Classification with CSP + LDA.\n"
            "Train and test subject-specific BCI classifiers on PhysioNet EEGMMIDB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full CSP evaluation:
    python mybci.py

  Train the required CSP pipeline:
    python mybci.py 4 14 train

  Predict with the required CSP pipeline:
    python mybci.py 4 14 predict

  Train the optional Morlet wavelet pipeline:
    python mybci.py 4 14 train --pipeline wavelet

  Predict with the optional Morlet wavelet pipeline:
    python mybci.py 4 14 predict --pipeline wavelet
        """
    )

    parser.add_argument(
        "subject_id",
        type=int,
        nargs="?",
        metavar="SUBJECT_ID",
        help="Subject ID from 1 to 109.",
    )

    parser.add_argument(
        "run_id",
        type=int,
        nargs="?",
        metavar="HELD_OUT_RUN",
        help="Held-out run ID. Supported motor-task runs: 3 to 14.",
    )

    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        nargs="?",
        metavar="MODE",
        help="Execution mode: train or predict.",
    )

    parser.add_argument(
        "--pipeline",
        choices=["csp", "wavelet"],
        default="csp",
        help="Pipeline type to use. Default: csp.",
    )

    return parser

def validate_args(
        args: argparse.Namespace,
        parser: argparse.ArgumentParser
) -> None:
    """
    Validate the command-line arguments.
    """
    positional_args = [args.subject_id, args.run_id, args.mode]
    provided_count = sum(arg is not None for arg in positional_args)

    if provided_count not in (0, 3):
        parser.error(
            "Error: You must provide either all three positional arguments "
            "(subject_id, held_out_run, mode) or none of them."
        )

    if provided_count == 0 and args.pipeline != "csp":
        parser.error(
            "--pipeline can only be used with train or predict mode. "
            "Full evaluation currently uses the CSP pipeline."
        )

def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    return args

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


def print_train_model_results(
        experiment_name: str,
        pipeline_name: str,
        subject_id: int,
        train_runs: list[int],
        scores: np.ndarray,
        model_path: str
):
    print(f"Experiment: {experiment_name}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Subject: S{subject_id:03d}")
    print(f"Training runs: {train_runs}")
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {np.mean(scores):.4f}")
    print(f"Model saved to: {model_path}")

def train_model(
        subject_id: int, 
        test_run: int, 
        pipeline_name: str = "csp"
) -> None:
    """
    Train the model on the training runs for a given test run.
    """
    # Get the training runs for the given test run
    experiment_name, train_runs = get_train_runs(test_run)

    # Guard against leakage
    if test_run in train_runs:
        raise ValueError(f"Test run {test_run} cannot be in the training runs {train_runs}")

    train_dataset = load_physionet_epochs(
        subject_id=subject_id,
        run_ids=train_runs,
        task_name=experiment_name
    )

    cv = LeaveOneGroupOut()

    validate_eeg_dataset(train_dataset)
    
    pipeline = create_selected_pipeline(pipeline_name, sfreq=train_dataset.sfreq)

    scores = cross_val_score(pipeline, train_dataset.X, train_dataset.y, cv=cv, groups=train_dataset.groups)

    pipeline.fit(train_dataset.X, train_dataset.y)

    artifact = {
        "pipeline": pipeline,
        "pipeline_name": pipeline_name,
        "subject_id": subject_id,
        "experiment": experiment_name,
        "train_runs": train_runs,
        "test_run": test_run,
        "data_source": "PhysioNet EEGMMIDB",
    }

    model_path = get_model_path(subject_id, test_run, pipeline_name)
    joblib.dump(artifact, model_path)

    print_train_model_results(
        experiment_name=experiment_name,
        pipeline_name=pipeline_name,
        subject_id=subject_id,
        train_runs=train_runs,
        scores=scores,
        model_path=model_path
    )

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

    if args.subject_id is None:
        run_full_evaluation()
        return
    
    # Mode selecting
    if args.mode == "train":
        train_model(args.subject_id, args.run_id, args.pipeline)
    elif args.mode == "predict":
        predict_stream(args.subject_id, args.run_id, args.pipeline)

if __name__ == "__main__":
    main()