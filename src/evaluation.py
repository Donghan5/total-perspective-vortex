import numpy as np
from sklearn.model_selection import cross_val_score
from src.preprocessing import preprocess_subject_runs
from src.pipeline import create_pipeline

EXPERIMENTS = {
	"actual_left_vs_right_fist": [3, 7, 11],
	"imagined_left_vs_right_fist": [4, 8, 12],
	"actual_fists_vs_feet": [5, 9, 13],
	"imagined_fists_vs_feet": [6, 10, 14],
}

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

def evaluate_held_out_run(subject_id: int, test_run: int) -> dict:
    """
    Evaluate the model on a held-out run.
    Just handle one subject and one test run
    """
    exp_name, train_runs = get_train_runs(test_run)

    X_train, y_train = preprocess_subject_runs(subject_id, train_runs)
    X_test, y_test = preprocess_subject_runs(subject_id, [test_run])

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    return {
        "subject_id": subject_id, 
        "test_run": test_run,
        "experiment_name": exp_name,
        "accuracy": accuracy
    }

def evaluate_subject_experiment(subject_id: int) -> tuple[list[dict], list[dict]]:
    """
    Evaluate all runs of a subject for each experiment.
    """
    results, errors = [], []
    for exp_name, run_ids in EXPERIMENTS.items():
        for test_run in run_ids:
            try:
                result = evaluate_held_out_run(subject_id, test_run)
                result["experiment"] = exp_name
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

	Evaluate all experiments and reutrn the results as a list of a dictionary
	"""

	results, errors = [], []

	for subject_id in subject_range:
		subject_results, subject_errors = evaluate_subject_experiment(subject_id)
		results.extend(subject_results)
		errors.extend(subject_errors)
            
	return results, errors

def evaluate_cross_validation_baseline(
	pipeline,
	run_ids: list[int],
	subject_range=range(1, 110)
):
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