# Import necessary libraries
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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

    parser.add_argument("subject_id", type=int, help="Subject ID (1 to 109)")
    parser.add_argument("run_id", type=int, help="Run ID (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, or 14)")
    parser.add_argument("mode", choices=["train", "predict", "evaluate"], nargs="?")
    
    return parser.parse_args()

def get_model_path(subject_id: int, test_run: int) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"S{subject_id}_R{test_run:02d}_pipeline.joblib"


def train_model(subject_id: int, test_run: int) -> None:
    """
    Train the model on the training runs for a given test run.
    """