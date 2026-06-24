import numpy as np

def mother_morlet(
        eta: np.ndarray,
        w0: float = 6.0,
        corrected: bool = True
) -> np.ndarray:
    normalization = np.pi ** (-0.25)
    complex_sinusoid = np.exp(1j * w0 * eta)    # Carrier wave

    if corrected:
        complex_sinusoid = complex_sinusoid - np.exp(-0.5 * w0 ** 2)  # Corrected carrier wave

    gaussian_window = np.exp(-0.5 * eta ** 2) 

    return normalization * complex_sinusoid * gaussian_window

