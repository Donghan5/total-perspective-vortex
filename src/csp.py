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

class CSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4, eps=1e-10):
		self.n_components = n_components
		self.eps = eps

	def fit(self, X, y):
		"""
			Weight fitting
		"""
		# Calculate cov of each class
		X_class0 = X [y == 0]
		cov_0 = np.mean([np.cov(epoch) for epoch in X_class0], axis=0)

		X_class1 = X [y == 1]
		cov_1 = np.mean([np.cov(epoch) for epoch in X_class1], axis=0)

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
