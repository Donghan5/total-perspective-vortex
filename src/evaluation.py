import numpy as np
from sklearn.model_selection import cross_val_score
from src.preprocessing import preprocess_subject_runs

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
