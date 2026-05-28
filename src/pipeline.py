from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.csp import CSP

def create_pipeline(
        n_components: int = 4,
        eps: float = 1e-10,
) -> Pipeline:
    """
    Create a machine learning pipeline with CSP and LDA.

    Args:
    - n_components: Number of CSP components to keep.
    - eps: Small constant to avoid log(0) in feature extraction.

    Returns:
    - A scikit-learn Pipeline object.
    """
    return Pipeline([
        ('csp', CSP(n_components=n_components, eps=eps)),
        ('lda', LinearDiscriminantAnalysis())
    ])