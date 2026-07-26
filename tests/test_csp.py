import numpy as np

from src.csp import CSP

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