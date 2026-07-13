import numpy as np

from src.dataset.base import EEGDataset


def validate_eeg_dataset(dataset: EEGDataset) -> None:
    """
    Validate the EEGDataset object to ensure it meets the expected structure and constraints.
    """

    X, y = dataset.X, dataset.y

    if X.ndim != 3:
        raise ValueError(f"Expected X to be a 3D array, but got shape {X.shape}.")
    
    if y.ndim != 1:
        raise ValueError(f"Expected y to be a 1D array, but got shape {y.shape}.")
    
    if len(X) != len(y):
        raise ValueError(f"Number of samples in X ({len(X)}) does not match number of labels in y ({len(y)}).")
    
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values (NaN or Inf).")
    
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values (NaN or Inf).")
    
    if dataset.sfreq <= 0:
        raise ValueError(f"Sampling frequency (sfreq) must be positive, but got {dataset.sfreq}.")
    
    if dataset.ch_names is None:
        raise ValueError("ch_names must not be None.")

    if len(dataset.ch_names) != X.shape[1]:
        raise ValueError(
            f"ch_names length mismatch: "
            f"len(ch_names)={len(dataset.ch_names)}, X channels={X.shape[1]}"
        )

    if len(np.unique(y)) != 2:
        raise ValueError(f"Expected binary labels, but got {len(np.unique(y))} unique values.")
    
    if dataset.groups is not None:
        groups = np.asarray(dataset.groups)

        if groups.ndim != 1:
            raise ValueError(f"Expected groups to be a 1D array, but got shape {groups.shape}.")
        if len(groups) != len(y):
            raise ValueError(f"Number of samples in groups ({len(groups)}) does not match number of labels in y ({len(y)}).")