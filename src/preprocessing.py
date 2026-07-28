from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
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

def encode_binary_labels(
		event_codes: np.ndarray,
		target_event_id: dict[str, int],
) -> np.ndarray:
	"""
	Args: event_codes: Array of event codes.
		target_event_id: Dictionary mapping event names to event codes.
	"""
	event_codes = np.asarray(event_codes)

	if event_codes.ndim != 1:
		raise ValueError("event_codes must be a 1D array.")

	if event_codes.size == 0:
		raise ValueError("event_codes must not be empty.")

	if "T1" not in target_event_id or "T2" not in target_event_id:
		raise ValueError("target_event_id contain both T1 and T2.")
	
	t1_code = target_event_id['T1']
	t2_code = target_event_id['T2']

	if t1_code == t2_code:
		raise ValueError("T1 and T2 must have different event code.")

	valid_mask = np.isin(event_codes, [t1_code, t2_code])

	if not np.all(valid_mask):
		invalid_codes = event_codes[~valid_mask]
		raise ValueError(
			f"Invalid event codes found: {invalid_codes}. "
		)

	return (event_codes == t2_code).astype(np.int64)

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

	missing_events = {
		"T1",
		"T2",
	} - event_id.keys()

	if missing_events:
		raise ValueError(
			"This run does not contain the required"
			f"task annotations: {sorted(missing_events)}"
		)
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
	event_codes = epochs.events[:, 2]
	y = encode_binary_labels(event_codes, target_event_id)

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

def preprocess_subject_runs(
		subject_id: int,
		run_ids: list[int]
	) -> tuple[np.ndarray, np.ndarray]:
	"""
		Args:
			subject_id: 1 to 109
			run_ids: motor imagery actual movement (eg: [3, 7, 11]) --> R03

		Returns:
			X: Tuples of X. (total_epochs, 64, 641)
			y: Tuples of y. (total_epochs,)
	"""

	if not run_ids:
		raise ValueError(
			"run_ids must contain at least one run."
		)

	X_list, y_list = [], []
	
	# loop (creation of path + preprocessing eeg data)
	for run_id in run_ids:
		X, y = preprocess_eeg_data(subject_id, run_id)
		X_list.append(X)
		y_list.append(y)

	# concatenating all X and y
	X = np.concatenate(X_list, axis=0)
	y = np.concatenate(y_list, axis=0)

	return X, y
