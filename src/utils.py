from pathlib import Path

MODEL_DIR = Path("models")

def get_model_path(subject_id: int, test_run: int) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"S{subject_id:03d}_R{test_run:02d}_pipeline.joblib"