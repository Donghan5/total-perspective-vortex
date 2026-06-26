from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from src.wavelet_features import wavelet_features_epochs
class MorletWaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            sfreq: float = 160.0,
            freqs=None,
            bands=None,
            w0: float = 6.0,
            width: float = 5.0,
            corrected: bool = True,
            eps: float = 1e-10
    ):
        self.sfreq = sfreq
        self.freqs = freqs
        self.bands = bands
        self.w0 = w0
        self.width = width
        self.corrected = corrected
        self.eps = eps

    def fit(self, X, y=None):
        """
        Validate the input data and initialize the frequency grid used for wavelet feature extraction.
        """
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError("Input data must be a 3D array (epochs x channels x time).")
        
        self.freqs_ = (
            np.asarray(self.freqs, dtype=float)
            if self.freqs is not None
            else np.arange(8.0, 31.0, 1.0)  # Default frequencies from 8 to 30 Hz
        )

        if np.any(self.freqs_ <= 0):
            raise ValueError("Frequencies must be positive.")
        
        if self.freqs_.ndim != 1:
            raise ValueError("Frequencies must be a 1D array.")

        return self
    
    def transform(self, X):
        """
        Extract Morlet wavelet band-power features from EEG epochs.
        """
        X = np.asarray(X, dtype=float)

        if not hasattr(self, 'freqs_'):
            raise RuntimeError("The transformer has not been fitted yet. Call 'fit' before 'transform'.")
        
        if X.ndim != 3:
            raise ValueError("Input data must be a 3D array (epochs x channels x time).")
        
        return wavelet_features_epochs(
            epochs=X,
            freqs=self.freqs_,
            sfreq=self.sfreq,
            w0=self.w0,
            width=self.width,
            corrected=self.corrected,
            bands=self.bands,
            eps=self.eps
        )