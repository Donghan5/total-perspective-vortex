import numpy as np
from sklearn.model_selection import cross_val_score
from src.preprocessing import preprocess_eeg_data, preprocess_subject_runs
from src.pipeline.pipeline import create_pipeline

EXPERIMENTS = {
	"actual_left_vs_right_fist": [3, 7, 11],
	"imagined_left_vs_right_fist": [4, 8, 12],
	"actual_fists_vs_feet": [5, 9, 13],
	"imagined_fists_vs_feet": [6, 10, 14],
}

ALL_TASK_RUNS = tuple(
      run_id
      for run_ids in EXPERIMENTS.values()
      for run_id in run_ids
)

EEGDataset = tuple[np.ndarray, np.ndarray]  # (X, y)
RunCache = dict[int, EEGDataset]  # {run_id: (X, y)}


def build_subject_run_cache(
            subject_id: int,
) -> RunCache:
      """
        Build a cache of preprocessed data for all runs of a subject.
      """
      return {
            run_id: preprocess_eeg_data(subject_id, run_id) for run_id in ALL_TASK_RUNS
	  }

def combine_cached_runs(
            run_cache: RunCache,
            run_ids: list[int]
) -> EEGDataset:
	"""
		Combine cached runs
	"""
	if not run_ids:
		raise ValueError("run_ids must contain at least one run.")

	missing_runs = [run_id for run_id in run_ids if run_id not in run_cache]
	if missing_runs:
		raise ValueError(f"Runs not found in cache: {missing_runs}")

	X = np.concatenate([run_cache[run_id][0] for run_id in run_ids], axis=0)

	y = np.concatenate([run_cache[run_id][1] for run_id in run_ids], axis=0)

	return X, y

def find_experiment_from_run(run_id: int) -> tuple[str, list[int]]:
    for exp_name, run_ids in EXPERIMENTS.items():
        if run_id in run_ids:
            return exp_name, run_ids
    raise ValueError(f"Run ID {run_id} does not belong to any experiment.")

def get_train_runs(test_run: int) -> tuple[str, list[int]]:
	"""
	Get the training runs for a given test run.
	"""
	exp_name, run_ids = find_experiment_from_run(test_run)
	train_runs = [run_id for run_id in run_ids if run_id != test_run]
    
	return exp_name, train_runs

def evaluate_held_out_run(subject_id: int, test_run: int, run_cache: RunCache) -> dict:
	"""
	Evaluate the model on a held-out run.
	Just handle one subject and one test run
	"""
	exp_name, train_runs = get_train_runs(test_run)

	X_train, y_train = combine_cached_runs(run_cache, train_runs)
	X_test, y_test = run_cache[test_run]

	pipeline = create_pipeline()
	pipeline.fit(X_train, y_train)
	accuracy = pipeline.score(X_test, y_test)

	return {
		"subject_id": subject_id, 
		"experiment_name": exp_name,
		"train_runs": train_runs,
		"test_run": test_run,
		"n_train_epochs": len(y_train),
		"n_test_epochs": len(y_test),
		"accuracy": float(accuracy)
	}

def evaluate_subject_experiment(subject_id: int, run_cache: RunCache) -> tuple[list[dict], list[dict]]:
    """
    Evaluate all runs of a subject for each experiment.
    """
    results, errors = [], []
    for exp_name, run_ids in EXPERIMENTS.items():
        for test_run in run_ids:
            try:
                result = evaluate_held_out_run(subject_id, test_run, run_cache)
                results.append(result)
            except Exception as error:
                errors.append({
                    "subject_id": subject_id,
                    "experiment_name": exp_name,
                    "test_run": test_run,
                    "error": str(error)
                })
    return results, errors


def evaluate_all_experiments(
	subject_range=range(1, 110),
) -> tuple[list[dict], list[dict]]:
	"""
	Args:
	- subject_range: Range of subject IDs to evaluate.

	Evaluate all experiments and return the results as a list of a dictionary
	"""
	subject_ids = list(subject_range)

	results, errors = [], []

	for subject_id in subject_ids:
		try:
			run_cache = build_subject_run_cache(subject_id)
			subject_results, subject_errors = evaluate_subject_experiment(subject_id, run_cache)
			results.extend(subject_results)
			errors.extend(subject_errors)
		except Exception as e:
			errors.append({
				"subject_id": subject_id,
				"experiment_name": None,
				"test_run": None,
				"error": (
					"Failed to evaluate subject: "
					f"{e}"
				)
			})
            
	return results, errors

def _print_evaluation_summary(valid_scores: list[float], total_subjects: int) -> None:
	evaluated_subjects = len(valid_scores)

	print("=== Evaluation Result ===")
	print(f"subject evaluated: {evaluated_subjects}/{total_subjects}")
	print()
	print(f"mean:    {np.mean(valid_scores):.4f}")
	print(f"median:  {np.median(valid_scores):.4f}")
	print(f"std: {np.std(valid_scores):.4f}")
	print(f"min:    {np.min(valid_scores):.4f}")
	print(f"max:    {np.max(valid_scores):.4f}")
	print()
	print(f">= 60%: {sum(s >= 0.6 for s in valid_scores)}/{evaluated_subjects}")
	print(f">= 70%: {sum(s >= 0.7 for s in valid_scores)}/{evaluated_subjects}")
	print(f">= 80%: {sum(s >= 0.8 for s in valid_scores)}/{evaluated_subjects}")

# explotray baseline evaluation using cross validation on all runs of an experiment (in notebook files)
def evaluate_cross_validation_baseline(
	pipeline,
	run_ids: list[int],
	subject_range=range(1, 110)
) -> dict[int, float | None]:
	"""
    
	Args:
		 run_ids: motor imagery actual movement (eg: [3, 7, 11]) --> R03

	Returns:
		dict {subject_id: mean_accuracy}
	"""

	# preparing dict to store mean accuracy
	mean_acc_dict = {int, float | None}
	subject_ids = list(subject_range)
	total_subjects = len(subject_ids)

	for subject_id in subject_ids:
		try:
			X, y = preprocess_subject_runs(subject_id, run_ids)
			scores = cross_val_score(pipeline, X, y, cv=5)
			mean_score = float(scores.mean())
			mean_acc_dict[subject_id] = mean_score
			print(f"Subject {subject_id}: mean accuracy = {mean_score:.3f}")
		except Exception as e:
			print(f"Subject {subject_id}: ERROR {e}")
			mean_acc_dict[subject_id] = None

    
	valid_scores = [s for s in mean_acc_dict.values() if s is not None]

	if not valid_scores:
		print("No valid scores to evaluate.")
		return mean_acc_dict
	
	_print_evaluation_summary(valid_scores, total_subjects)
	return mean_acc_dict