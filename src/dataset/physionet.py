import numpy as np

from src.dataset.base import EEGDataset
from src.preprocessing import preprocess_eeg_data, load_raw_eeg

def load_physionet_epochs(
        subject_id: int,
        run_ids: list[int],
        task_name: str | None = None,
) -> EEGDataset:
    X_list, y_list, groups = [], [], []

    ch_names, sfreq = None, None

    for run_id in run_ids:
        raw = load_raw_eeg(subject_id, run_id)
        if ch_names is None:
            ch_names = raw.ch_names
            sfreq = raw.info["sfreq"]

        X_run, y_run = preprocess_eeg_data(raw)
        X_list.append(X_run)
        y_list.append(y_run)
        groups.append(np.full(y_run.shape, run_id))

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    groups = np.asarray(groups)

    return EEGDataset(
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=ch_names,
        groups=groups,
        task_name=task_name
    )