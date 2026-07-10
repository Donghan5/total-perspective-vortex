import numpy as np

from src.dataset.base import EEGDataset

def load_npz_epochs(path: str) -> EEGDataset:
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    sfreq = float(data["sfreq"])

    if "ch_names" in data:
        ch_names = [str(ch) for ch in data["ch_names"]]
    else:
        ch_names = [f"ch_{i}" for i in range(X.shape[1])]

    groups = data["groups"] if "groups" in data else None

    return EEGDataset(
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=ch_names,
        groups=groups,
        task_name="npz_dataset"
    )