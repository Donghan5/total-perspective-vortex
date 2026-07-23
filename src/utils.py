from pathlib import Path
import numpy as np

MODEL_DIR = Path("models")

def get_model_path(subject_id: int, test_run: int, pipeline_name: str = "csp") -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"S{subject_id:03d}_R{test_run:02d}_{pipeline_name}_pipeline.joblib"

def estimate_covariance(epoch: np.ndarray) -> np.ndarray:
    centered = epoch - epoch.mean(axis=1, keepdims=True)
    return centered @ centered.T / (epoch.shape[1] - 1)