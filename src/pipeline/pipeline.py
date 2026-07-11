from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from src.csp import CSP

def create_pipeline(
        n_components: int = 4,
        eps: float = 1e-10,
        csp_reg: float = 0.0
) -> Pipeline:
    """
    Create a machine learning pipeline with CSP and LDA.

    Args:
    - n_components: Number of CSP components to keep.
    - eps: Small constant to avoid log(0) in feature extraction.
    - csp_reg: Tikhonov-style diagonal loading strength for CSP covariance regularization.
               0.0 disables regularization.

    Returns:
    - A scikit-learn Pipeline object.
    """
    return Pipeline([
        ('csp', CSP(n_components=n_components, eps=eps, reg=csp_reg)),
        ('lda', LinearDiscriminantAnalysis())
    ])