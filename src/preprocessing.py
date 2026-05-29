from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "physionet.org" / "files" / "eegmmidb" / "1.0.0"

def build_edf_path(subject_id: int, run_id: int) -> Path:
	"""
	Args:
	- subject_id: 1 to 109
	- run_id: motor imagery actual movement (eg: [3, 7, 11]) --> R03
	Returns:
	- Path to the EDF file.
	"""
	if not 1 <= subject_id <= 109:
		raise ValueError("subject_id must be between 1 and 109")
	if not 1 <= run_id <= 14:
		raise ValueError("run_id must be between 1 and 14")
	
	subject_dir = f"S{subject_id:03d}"
	filename = f"S{subject_id:03d}R{run_id:02d}.edf"
	return DATA_ROOT / subject_dir / filename

def load_raw_eeg(subject_id: int, run_id: int) -> mne.io.BaseRaw:
	"""
	Args:
	- subject_id: 1 to 109
	- run_id: motor imagery actual movement (eg: [3, 7, 11]) --> R03
	Returns:
	- Loaded raw EEG data.
	"""
	edf_path = build_edf_path(subject_id, run_id)

	if not edf_path.exists():
		raise FileNotFoundError(
			f"EDF file not found: {edf_path}\n"
			"Download the dataset before running code"	
		)

	return mne.io.read_raw_edf(edf_path, preload=True)

def filter_raw_eeg(
		raw: mne.io.BaseRaw,
		l_freq: float = 8.0,
		h_freq: float = 30.0
) -> mne.io.BaseRaw:
	"""
	Args:
	- raw: Raw EEG data.
	- l_freq: Low frequency cutoff.
	- h_freq: High frequency cutoff.
	Returns:
	- Filtered raw EEG data.
	"""
	return raw.copy().filter(l_freq=l_freq, h_freq=h_freq)

def extract_epochs(raw_filtered: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
	"""
	Args:
	- raw_filtered: Filtered raw EEG data.
	Returns:
	- Tuple of (X, y) where X is the EEG data and y is the labels.
	"""
	events, event_id = mne.events_from_annotations(raw_filtered)
	target_event_id = {'T1': event_id['T1'], 'T2': event_id['T2']}
	
	epochs = mne.Epochs(
		raw_filtered,
		events,
		event_id=target_event_id,
		tmin=0,	
		tmax=4,
		baseline=None,
		preload=True
	)

	X = epochs.get_data()
	y = epochs.events[:, 2] - 2 # 1 -> 0, 2 -> 1

	return X, y

def preprocess_eeg_data(subject_id: int, run_id: int) -> tuple[np.ndarray, np.ndarray]:
	"""
		Preprocessing EEG data.
	
	Args:
		subject_id: Subject ID.
		run_id: Run ID.
	
	Returns:
		Tuple of (X, y) where X is the EEG data and y is the labels.
	"""
	raw = load_raw_eeg(subject_id, run_id)
	raw_filtered = filter_raw_eeg(raw)
	return extract_epochs(raw_filtered)

def preprocess_subject_runs(subject_id: int, run_ids: list[int]):
	"""
		Args:
			subject_id: 1 to 109
			run_ids: motor imagery actual movement (eg: [3, 7, 11]) --> R03

		Returns:
			X: Tuples of X. (total_epochs, 64, 641)
			y: Tuples of y. (total_epochs,)
	"""

	# preparing list to store X and y (which will contain 50 epochs)
	X_list, y_list = [], []
	
	# loop (creation of path + preprocessing eeg data)
	for run_id in run_ids:
		X, y = preprocess_eeg_data(subject_id, run_id)
		X_list.append(X)
		y_list.append(y)

	# concatenating all X and y
	X = np.concatenate(X_list, axis=0)
	y = np.concatenate(y_list)

	return X, y


	