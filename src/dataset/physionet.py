import numpy as np

from src.dataset.base import EEGDataset
from src.preprocessing import extract_epochs, filter_raw_eeg, load_raw_eeg

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
            ch_names = list(raw.ch_names)
            sfreq = float(raw.info["sfreq"])

        raw_filtered = filter_raw_eeg(raw)
        X_run, y_run = extract_epochs(raw_filtered)
        X_list.append(X_run)
        y_list.append(y_run)
        groups.extend([run_id] * len(y_run))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.asarray(groups)

    return EEGDataset(
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=ch_names,
        subject_id=subject_id,
        run_ids=run_ids,
        groups=groups,
        task_name=task_name
    )