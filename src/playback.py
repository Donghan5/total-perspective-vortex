import time
from typing import Any
import numpy as np

def playback_epochs(
        X_test: np.ndarray, 
        pipeline: Any
) -> tuple[list, list[float]]:
    """
    Playback the test epochs one by one and measure the latency of each prediction.
    Real-time constraint: prediction latency is measured after the epoch chunk is sent to the pipeline.
    """
    if len(X_test) == 0:
        raise ValueError("No test epochs provided for playback.")
    
    predictions, latencies = [], []

    # Measure latency for each epoch
    for epoch in X_test:
        chunk = epoch[np.newaxis, ...]  # Add batch dimension

        start = time.perf_counter()
        pred = pipeline.predict(chunk)[0]
        elapsed = time.perf_counter() - start

        predictions.append(pred)
        latencies.append(elapsed)
    

    return predictions, latencies