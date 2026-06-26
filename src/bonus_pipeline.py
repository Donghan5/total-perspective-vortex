from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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