from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.wavelet_transformer import MorletWaveletTransformer

def create_wavelet_pipeline() -> Pipeline:
    """
    Create a machine learning pipeline that includes Morlet wavelet feature extraction,
    standard scaling, and Linear Discriminant Analysis (LDA) for classification.
    """

    return Pipeline([
        ('wavelet', MorletWaveletTransformer()),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])


def create_wavelet_select_pipeline(k: int = 20) -> Pipeline:
    """
    Create a Morlet wavelet pipeline with feature selection.
    
    This reduces the high-dimensional wavelet feature vector before LDA
    """
    return Pipeline([
        ("wavelet", MorletWaveletTransformer()),
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("lda", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])