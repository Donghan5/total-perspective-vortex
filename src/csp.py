# import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import scikit learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.linalg import eigh

# import mne
import mne

DEFAULT_CSP_REG = 1e-2

class CSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4, eps=1e-10, reg=DEFAULT_CSP_REG):
		self.n_components = n_components
		self.eps = eps
		self.reg = reg

	def fit(self, X, y):
		"""
			Weight fitting
		"""
		# checking input dimensions
		if len(np.unique(y)) != 2:
			raise ValueError("CSP only supports binary classification.")

		# Calculate cov of each class
		X_class0 = X [y == 0]
		cov_0 = np.mean([self.estimate_covariance(epoch) for epoch in X_class0], axis=0)

		X_class1 = X [y == 1]
		cov_1 = np.mean([self.estimate_covariance(epoch) for epoch in X_class1], axis=0)

		cov_0 = self.regularize_covariance(cov_0)
		cov_1 = self.regularize_covariance(cov_1)

		# Genealized eigenvalue problem
		eigenvalues, eigenvectors = eigh(cov_0, cov_0 + cov_1)

		# Select index to store
		n_half = self.n_components // 2
		top_indices = (list(range(n_half)) + list(range(-n_half, 0)))

		self.filters_ = eigenvectors[:, top_indices]

		return self
	
	def transform(self, X):
		"""
			Applying weight
		"""
		X_csp = np.array([self.filters_.T @ epoch for epoch in X])
		
		features = np.log(np.var(X_csp, axis=2) + self.eps)
		return features

	@staticmethod
	def estimate_covariance(epoch: np.ndarray) -> np.ndarray:
		if epoch.ndim != 2:
			raise ValueError("Input epoch must be a 2D array.")

		n_samples = epoch.shape[1]
		if n_samples < 2:
			raise ValueError("Each epoch must have at least two samples to compute covariance.")

		centered = epoch - epoch.mean(axis=1, keepdims=True)
		cov = (centered @ centered.T) / (n_samples - 1)
		return cov

	def regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
		"""
			Regularize covariance matrix to ensure numerical stability.
		"""
		if self.reg <= 0:
			return cov
		
		n_channels = cov.shape[0]
		scale = np.trace(cov) / n_channels
		reg_cov = cov + self.reg * scale * np.eye(n_channels)

		return reg_cov

