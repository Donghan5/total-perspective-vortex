from dataclass import dataclass
import numpy as np

@dataclass
class EEGDataset:
    X: np.ndarray
    y: np.ndarray
    sfreq: float
    ch_names: list[str]
    groups: np.ndarray | None = None
    subject_id: int | str | None = None
    run_ids: list[int] | None = None
    task_name: str | None = None

class BaseEEGLoader:
    """
    Base class for EEG dataset loaders. Parent class for each dataset loader.
    """
    def load_epochs(self, *args, **kwargs) -> EEGDataset:
        raise NotImplementedError("Subclasses must implement this method.")