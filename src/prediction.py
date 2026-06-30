import time
import joblib
from sklearn.metrics import accuracy_score
from src.playback import playback_epochs
from src.preprocessing import preprocess_subject_runs
from src.utils import get_model_path
import numpy as np

def print_prediction_results(
        predictions: list, y_test: list
) -> None:
    """
    Print the prediction results in a readable format.
    Following the format from subject:
    epoch nb: [prediction] [truth] equal (boolean type)
    """
    print("Epoch nb: [prediction] [truth] equal?")
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        print(f"Epoch {i:02d}: [{pred}] [{truth}] {pred == truth}")


def predict_stream(subject_id: int, test_run: int, pipeline_name: str = "csp") -> None:
    """
    Predict the labels for the test run using the trained model.
    """

    # Load the trained model
    model_path = get_model_path(subject_id, test_run, pipeline_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Load the trained model artifact
    artifact = joblib.load(model_path)

    # checking the metadata of the loaded artifact
    if artifact["subject_id"] != subject_id or artifact["test_run"] != test_run or artifact["pipeline_name"] != pipeline_name:
        raise ValueError(f"Loaded model metadata does not match the requested subject_id {subject_id}, test_run {test_run}, and pipeline_name {pipeline_name}.")
    
    # Extract the pipeline from the artifact
    pipeline = artifact["pipeline"]

    if artifact.get("pipeline_name") != pipeline_name:
        raise ValueError(f"Loaded model pipeline name does not match the requested pipeline_name {pipeline_name}.")

    # Load only the test run data
    X_test, y_test = preprocess_subject_runs(subject_id, [test_run])

    # Predict epoch by epoch
    predictions, latencies = playback_epochs(X_test, pipeline)

    # Calculate accuracy and average latency
    accuracy = accuracy_score(y_test, predictions)
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    # Print final accuracy and max latency
    print_prediction_results(predictions, y_test)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Average Latency per Epoch: {avg_latency:.4f} seconds")
    print(f"Maximum Latency for any Epoch: {max_latency:.4f} seconds")
    print(f"2s latency constraint satisfied: {max_latency < 2.0}")