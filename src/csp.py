# import basic libraries
import numpy as np

# import scikit learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import eigh

DEFAULT_CSP_REG = 1e-2

class CSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4, eps=1e-10, reg=DEFAULT_CSP_REG):
		self.n_components = n_components
		self.eps = eps
		self.reg = reg


	def _validate_regularization(self) -> None:
		numeric_types = (
			int,
			float,
			np.integer,
			np.floating,
		)
		"""
			Validate the regularization parameter.
		"""
		if isinstance(self.reg, (bool, np.bool_)) or not isinstance(self.reg, numeric_types):
			raise ValueError("Regularization parameter must be a number.")

		if not np.isfinite(self.reg) or self.reg < 0:
			raise ValueError("Regularization parameter must be finite and non-negative.")


	def _validate_fit_inputs(self, X, y):
		"""
			Verify input data
			X : 3D array of shape (n_epochs, n_channels, n_samples)
			y: 1D array of shape (n_epochs,)
		"""
		if X.ndim != 3:
			raise ValueError("Input X must be a 3D array of shape (n_epochs, n_channels, n_samples).")
		
		if y.ndim != 1:
			raise ValueError("Input y must be a 1D array of shape (n_epochs,).")
		
		if X.shape[0] != y.shape[0]:
			raise ValueError("Number of epochs in X and y must match.")

		if X.shape[2] < 2:
			raise ValueError("Each epoch must have at least two samples to compute covariance.")
		
		if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(self.n_components, (int, np.integer)):
			raise ValueError("Number of components must be an integer.")
		
		if self.n_components <= 0:
			raise ValueError("Number of components must be a positive integer.")
		
		if self.n_components % 2 != 0:
			raise ValueError("Number of components must be an even integer.")
		
		if self.n_components > X.shape[1]:
			raise ValueError("Number of components cannot exceed the number of channels.")

		numeric_types = (
			int,
			float,
			np.integer,
			np.floating,
		)

		if isinstance(self.eps, (bool, np.bool_)) or not isinstance(self.eps, numeric_types):
			raise ValueError("Epsilon must be a number.")
		
		if not np.isfinite(self.eps) or self.eps <= 0:
			raise ValueError("Epsilon must be finite and positive.")
		
		self._validate_regularization()

	def fit(self, X, y):
		"""
			Weight fitting
		"""
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		self._validate_fit_inputs(X, y)
		
		classes = np.unique(y)

		# checking input dimensions
		if len(classes) != 2:
			raise ValueError("CSP only supports binary classification.")

		self.classes_ = classes
		self.n_channels_in_ = X.shape[1]
		class_0, class_1 = classes

		# Calculate cov of each class
		X_class0 = X[y == class_0]
		cov_0 = np.mean([self.estimate_covariance(epoch) for epoch in X_class0], axis=0)

		X_class1 = X[y == class_1]
		cov_1 = np.mean([self.estimate_covariance(epoch) for epoch in X_class1], axis=0)

		cov_0 = self.regularize_covariance(cov_0)
		cov_1 = self.regularize_covariance(cov_1)

		# Genealized eigenvalue problem
		eigenvalues, eigenvectors = eigh(cov_0, cov_0 + cov_1)

		# Select index to store
		n_half = self.n_components // 2
		selected_indices = (list(range(n_half)) + list(range(-n_half, 0)))

		self.filters_ = eigenvectors[:, selected_indices]
		self.eigenvalues_ = eigenvalues[selected_indices]

		return self
	
	def transform(self, X):
		"""
			Applying weight
		"""
		check_is_fitted(
			self,
			attributes=["filters_", "classes_", "n_channels_in_"],
		)

		X = np.asarray(X, dtype=float)

		if X.ndim != 3:
			raise ValueError("Input X must be a 3D array of shape (n_epochs, n_channels, n_samples).")

		if X.shape[1] != self.n_channels_in_:
			raise ValueError(f"Input X must have {self.n_channels_in_} channels, but got {X.shape[1]}.")

		if X.shape[2] == 0:
			raise ValueError("Each epoch must contain at least one sample.")

		X_csp = np.array([self.filters_.T @ epoch for epoch in X])
		
		features = np.log(np.var(X_csp, axis=2) + self.eps)
		return features

	@staticmethod
	def estimate_covariance(epoch: np.ndarray) -> np.ndarray:
		"""
			Estimate the covariance matrix of a single epoch.
		"""
		epoch = np.asarray(epoch, dtype=float)

		if epoch.ndim != 2:
			raise ValueError("Input epoch must be a 2D array.")

		n_samples = epoch.shape[1]
		if n_samples < 2:
			raise ValueError("Each epoch must have at least two samples to compute covariance.")

		centered = epoch - epoch.mean(axis=1, keepdims=True)
		cov = (centered @ centered.T) / (n_samples - 1)
		trace = np.trace(cov)

		if not np.isfinite(trace) or trace <= 0:
			raise ValueError("Epoch covariance trace must be finite and positive.")
		return cov / trace

	def regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
		"""
			Regularize covariance matrix to ensure numerical stability.
		"""
		self._validate_regularization()
		if self.reg == 0:
			return cov
		
		n_channels = cov.shape[0]
		scale = np.trace(cov) / n_channels
		reg_cov = cov + self.reg * scale * np.eye(n_channels)

		return reg_cov

