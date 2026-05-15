import mne
import numpy as np
from sklearn.model_selection import cross_val_score

def preprocess_eeg_data(edf_path: str) -> tuple[np.ndarray, np.ndarray]:
	"""
		Preprocessing EEG data.
	
	Args:
		edf_path: Path to the EDF file.
	
	Returns:
		Tuple of (X, y) where X is the EEG data and y is the labels.
	"""
	raw = mne.io.read_raw_edf(edf_path, preload=True)
	raw_filtered = raw.copy().filter(l_freq=8, h_freq=30)
	
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
		path = f'../physionet.org/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run_id:02d}.edf'
		X, y = preprocess_eeg_data(path)
		X_list.append(X)
		y_list.append(y)

	# concatenating all X and y
	X = np.concatenate(X_list, axis=0)
	y = np.concatenate(y_list)

	return X, y

def evaluate_all_subjects(pipeline, run_ids: list[int], subject_range=range(1, 100)):
	"""
	
	Args:
		run_ids: motor imagery actual movement (eg: [3, 7, 11]) --> R03

	Returns:
		dict {subject_id: mean_accuracy}
	"""

	# preparing dict to store mean accuracy
	mean_acc_dict = {}

	for subject_id in subject_range:
		try:
			X, y = preprocess_subject_runs(subject_id, run_ids)
			scores = cross_val_score(pipeline, X, y, cv=5)
			mean_acc_dict[subject_id] = scores.mean()
			print(f"Subject {subject_id}: mean accuracy = {scores.mean():.3f}")
		except Exception as e:
			print(f"Subject {subject_id}: ERROR {e}")
			mean_acc_dict[subject_id] = None

	
	# temporary feature
	valid_scores = [s for s in mean_acc_dict.values() if s is not None]

	print(f"=== Evaluation Result ===")
	print(f"subject evaluated: {len(valid_scores)}/109")
	print(f"")
	print(f"mean:    {np.mean(valid_scores):.4f}")
	print(f"median:  {np.median(valid_scores):.4f}")
	print(f"std: {np.std(valid_scores):.4f}")
	print(f"min:    {np.min(valid_scores):.4f}")
	print(f"max:    {np.max(valid_scores):.4f}")
	print(f"")
	print(f">= 60%: {sum(s >= 0.6 for s in valid_scores)}/109")
	print(f">= 70%: {sum(s >= 0.7 for s in valid_scores)}/109")
	print(f">= 80%: {sum(s >= 0.8 for s in valid_scores)}/109")


	return mean_acc_dict

	