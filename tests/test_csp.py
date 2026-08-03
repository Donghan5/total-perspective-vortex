import numpy as np
import pytest

from src.csp import CSP, DEFAULT_CSP_REG

def test_estimate_covariance_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    epoch = rng.normal(size=(64, 641))

    csp = CSP()

    actual = csp.estimate_covariance(epoch)
    expected = np.cov(epoch)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

def test_estimate_covariance_known_values() -> None:
    epoch = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    ])

    expected = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    actual = CSP.estimate_covariance(epoch)

    np.testing.assert_allclose(actual, expected)

def test_csp_regularization_is_enabled_by_default() -> None:
    csp = CSP()

    assert csp.reg == pytest.approx(DEFAULT_CSP_REG)
    assert csp.reg > 0.0

def test_regularize_covariance_adds_scaled_identity() -> None:
    covariance = np.array([
        [2.0, 0.5],
        [0.5, 1.0],
    ])

    reg = 0.1
    csp = CSP(reg=reg)

    actual = csp.regularize_covariance(covariance)

    n_channels = covariance.shape[0]
    scale = np.trace(covariance) / n_channels

    expected = covariance + reg * scale * np.eye(n_channels)

    np.testing.assert_allclose(actual, expected)

def test_regularization_can_be_disabled() -> None:
    covariance = np.array([
        [2.0, 0.5],
        [0.5, 4.0],
    ])

    csp = CSP(reg=0.0)

    actual = csp.regularize_covariance(covariance)

    np.testing.assert_allclose(actual, covariance)

def test_regularization_negative_raises_value_error() -> None:
    covariance = np.array([
        [2.0, 0.5],
        [0.5, 4.0],
    ])

    csp = CSP(reg=-0.1)

    with pytest.raises(ValueError, match="Regularization parameter must be finite and non-negative."):
        csp.regularize_covariance(covariance)

def test_estimate_covariance_invalid_input_raises_value_error() -> None:
    csp = CSP()

    # Test with 1D array
    epoch_1d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Input epoch must be a 2D array."):
        csp.estimate_covariance(epoch_1d)

    # Test with less than 2 samples
    epoch_few_samples = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="Each epoch must have at least two samples to compute covariance."):
        csp.estimate_covariance(epoch_few_samples)

def test_csp_fit_accepts_non_zero_one_binary_labels() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(6, 4, 50))
    y = np.array([1, 2, 1, 2, 1, 2])

    csp = CSP(n_components=2)
    csp.fit(X, y)

    np.testing.assert_array_equal(csp.classes_, np.array([1, 2]))
    assert csp.filters_.shape == (4, 2)

def test_normalize_covariance_test() -> None:
    covariance = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    actual = CSP.normalize_covariance_trace(covariance)

    expected = covariance / np.trace(covariance)
    np.testing.assert_allclose(actual, expected)

    assert np.trace(actual) == pytest.approx(1.0)