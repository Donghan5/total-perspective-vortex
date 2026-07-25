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