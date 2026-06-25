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

def frequency_to_scale(
        frequency: float,
        sampling_frequency: float,
        w0: float = 6.0,
) -> float:
    """
    Convert frequency to scale for Morlet wavelet.
    """
    if frequency <= 0:
        raise ValueError("Frequency must be positive.")
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    
    return (w0 ** sampling_frequency) / (2 * np.pi * frequency)

def morlet_wavelet_for_frequency(
        frequency: float,
        sampling_frequency: float,
        w0: float = 6.0,
        width: float = 5.0,
        corrected: bool = True,
        normalized_discrete: bool = True
) -> np.ndarray:
    """
    Generate a Morlet wavelet for a specific frequency.
    """
    scale = frequency_to_scale(frequency, sampling_frequency, w0)
    half_width = int(np.ceil(width * scale))
    t = np.arange(-half_width, half_width + 1)

    wavelet = (1.0 / np.sqrt(scale)) * mother_morlet(t / scale, w0=w0, corrected=corrected)
    if normalized_discrete:
        energy = np.sqrt(np.sum(np.abs(wavelet) ** 2))
        if energy > 0:
            wavelet = wavelet / energy

    return wavelet

def cwt_single_frequency(
        signal: np.ndarray,
        frequency: float,
        sampling_frequency: float,
        w0: float = 6.0,
        width: float = 5.0,
        corrected: bool = True
) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform for a single frequency.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    
    wavelet = morlet_wavelet_for_frequency(frequency, sampling_frequency, w0=w0, width=width, corrected=corrected)

    coeff = np.convolve(signal, np.conj(wavelet[::-1]), mode='same')

    return coeff

def cwt_frequencies(
        signal: np.ndarray,
        freqs: np.ndarray,
        sfreq: float,
        w0: float = 6.0,
        width: float = 5.0,
        corrected: bool = True
) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform for multiple frequencies.
    """
    signal = np.asarray(signal, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    
    coeffs = []
    for freq in freqs:
        coeff = cwt_single_frequency(signal, frequency=freq, sampling_frequency=sfreq, w0=w0, width=width, corrected=corrected)
        coeffs.append(coeff)

    return np.array(coeffs)

def wavelet_power(coeffs: np.ndarray) -> np.ndarray:
    """
    Complex feature extraction: Compute the wavelet power from the CWT coefficients.
    """
    return np.abs(coeffs) ** 2

### HAVE TO CHECK MATH THINGS FROM THIS FUNCTION
def band_power_features(
        power: np.ndarray,
        freqs: np.ndarray,
        bands=None,
        eps: float = 1e-10
) -> np.ndarray:
    """
    Compute band power features from wavelet power.
    """
    if bands is None:
        bands = [
            ("mu", 8.0, 13.0),
            ("low_beta", 13.0, 20.0),
            ("high_beta", 20.0, 30.0),
        ]

    features = []

    for _, low, high in bands:
        mask = (freqs >= low) & (freqs < high)

        if not np.any(mask):
            raise ValueError(f"No frequencies found in the band {low}-{high} Hz.")
        
        band_power = np.mean(power[mask])
        features.append(np.log(band_power + eps))  # Log-transform to stabilize variance

    return np.array(features)

def wavelet_features_epoch(
        epoch: np.ndarray,
        freqs: np.ndarray,
        sfreq: float,
        w0: float = 6.0,
        width: float = 5.0,
        corrected: bool = True,
        bands=None,
        eps: float = 1e-10
) -> np.ndarray:
    """
    Compute wavelet features for a single epoch.
    """
    epoch = np.asarray(epoch, dtype=float)

    if epoch.ndim != 2:
        raise ValueError("Input epoch must be a 2D array (channels x time).")
    
    channel_features = []

    for channel_signal in epoch:
        coeffs = cwt_frequencies(signal=channel_signal, freqs=freqs, sfreq=sfreq, w0=w0, width=width, corrected=corrected)
        power = wavelet_power(coeffs)
        features = band_power_features(power, freqs, bands=bands, eps=eps)
        channel_features.append(features)

    return np.concatenate(channel_features)

def wavelet_features_epochs(
        epochs: np.ndarray,
        freqs: np.ndarray,
        sfreq: float,
        w0: float = 6.0,
        width: float = 5.0,
        corrected: bool = True,
        bands=None,
        eps: float = 1e-10
) -> np.ndarray:
    """
    Compute wavelet features for multiple epochs.
    """
    epochs = np.asarray(epochs, dtype=float)

    if epochs.ndim != 3:
        raise ValueError("Input epochs must be a 3D array (epochs x channels x time).")
    
    all_features = []

    for epoch in epochs:
        features = wavelet_features_epoch(epoch, freqs=freqs, sfreq=sfreq, w0=w0, width=width, corrected=corrected, bands=bands, eps=eps)
        all_features.append(features)

    return np.array(all_features)