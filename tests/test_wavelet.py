import numpy as np
import pytest

from src.pipeline.bonus_pipeline import (
    create_wavelet_pipeline,
    create_wavelet_select_pipeline,
)
from src.wavelet_features import (
  band_power_features,
  cwt_single_frequency,
  cwt_frequencies,
  frequency_to_scale,
  morlet_wavelet_for_frequency,
  wavelet_features_epoch,
  wavelet_features_epochs,
  wavelet_power,
)

from src.wavelet_transformer import MorletWaveletTransformer


def test_frequency_to_scale_matches_formula():
    # Remains - if frequency is high, scale is going to be lower - frequency and sampling freq is not equal and lesser than 0
    actual = frequency_to_scale(
        frequency=10.0,
        sampling_frequency=160.0,
        w0=6.0
    )

    expected = (6.0 * 160.0) / (2 * np.pi * 10.0)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("frequency", [0.0, -1.0])
def test_frequency_to_scale_rejects_non_positive_frequency(frequency):
    with pytest.raises(ValueError, match="Frequency must be positive"):
        frequency_to_scale(frequency, sampling_frequency=160.0)


@pytest.mark.parametrize("sampling_frequency", [0.0, -160.0])
def test_frequency_to_scale_rejects_non_positive_sampling_frequency(
        sampling_frequency
):
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        frequency_to_scale(10.0, sampling_frequency=sampling_frequency)


def test_morlet_wavelet_has_unit_energy():
    wavelet = morlet_wavelet_for_frequency(
        frequency=10.0,
        sampling_frequency=160.0,
    )

    assert wavelet.ndim == 1
    assert len(wavelet) % 2 == 1
    assert np.iscomplexobj(wavelet)
    assert np.all(np.isfinite(wavelet))
    assert np.sum(np.abs(wavelet) ** 2) == pytest.approx(1.0)

def test_cwt_more_strongly_to_matching_frequency():
    sfreq = 160.0
    time = np.arange(640) / sfreq
    signal = np.sin(2 * np.pi * 10.0 * time)

    coef = cwt_frequencies(
        signal,
        freqs=np.array([10.0, 25.0]),
        sfreq=sfreq,
    )

    power = wavelet_power(coef)

    center = slice(100, -100)
    assert power[0, center].mean() > power[1, center].mean()


def test_cwt_frequencies_preserves_signal_length():
    signal = np.zeros(320)
    freqs = np.array([8.0, 10.0, 20.0])

    coefficients = cwt_frequencies(signal, freqs, sfreq=160.0)

    assert coefficients.shape == (3, 320)


def test_cwt_single_frequency_rejects_non_1d_signal():
    with pytest.raises(ValueError, match="Input signal must be a 1D array"):
        cwt_single_frequency(
            np.zeros((2, 320)),
            frequency=10.0,
            sampling_frequency=160.0,
        )


def test_wavelet_power_is_non_negative_and_scales_with_amplitude_squared():
    sfreq = 160.0
    time = np.arange(640) / sfreq
    signal = np.sin(2 * np.pi * 10.0 * time)

    original = wavelet_power(
        cwt_frequencies(signal, np.array([10.0]), sfreq=sfreq)
    )
    doubled = wavelet_power(
        cwt_frequencies(2.0 * signal, np.array([10.0]), sfreq=sfreq)
    )

    assert np.all(original >= 0)
    np.testing.assert_allclose(doubled, 4.0 * original)


def test_band_power_features_uses_half_open_band_and_log_mean():
    power = np.array([
        [1.0, 3.0],
        [5.0, 7.0],
        [100.0, 100.0],
    ])
    freqs = np.array([8.0, 12.0, 13.0])

    features = band_power_features(
        power,
        freqs,
        bands=[("mu", 8.0, 13.0)],
        eps=0.0,
    )

    assert features == pytest.approx([np.log(4.0)])


def test_band_power_features_rejects_band_without_frequencies():
    with pytest.raises(ValueError, match="No frequencies found"):
        band_power_features(
            np.ones((2, 10)),
            np.array([8.0, 10.0]),
            bands=[("gamma", 30.0, 40.0)],
        )


def test_wavelet_features_epochs_returns_expected_shape():
    rng = np.random.default_rng(42)
    epochs = rng.normal(size=(2, 3, 320))

    features = wavelet_features_epochs(
        epochs,
        freqs=np.array([8.0, 10.0, 15.0, 25.0]),
        sfreq=160.0,
    )

    assert features.shape == (2, 9)
    assert np.all(np.isfinite(features))


def test_wavelet_features_epoch_rejects_non_2d_input():
    with pytest.raises(ValueError, match="Input epoch must be a 2D array"):
        wavelet_features_epoch(
            np.zeros(320),
            freqs=np.arange(8.0, 31.0),
            sfreq=160.0,
        )


def test_wavelet_features_epochs_rejects_non_3d_input():
    with pytest.raises(ValueError, match="Input epochs must be a 3D array"):
        wavelet_features_epochs(
            np.zeros((2, 320)),
            freqs=np.arange(8.0, 31.0),
            sfreq=160.0,
        )


def test_transformer_fit_initializes_default_frequencies():
    transformer = MorletWaveletTransformer()

    returned = transformer.fit(np.zeros((2, 3, 320)))

    assert returned is transformer
    np.testing.assert_array_equal(transformer.freqs_, np.arange(8.0, 31.0))


@pytest.mark.parametrize(
    "freqs, message",
    [
        (np.array([0.0, 10.0]), "Frequencies must be positive"),
        (np.array([[8.0, 10.0]]), "Frequencies must be a 1D array"),
    ],
)
def test_transformer_fit_rejects_invalid_frequencies(freqs, message):
    transformer = MorletWaveletTransformer(freqs=freqs)

    with pytest.raises(ValueError, match=message):
        transformer.fit(np.zeros((2, 3, 320)))


def test_transformer_fit_transform_returns_expected_shape():
    transformer = MorletWaveletTransformer()

    features = transformer.fit_transform(np.zeros((2, 3, 320)))

    assert features.shape == (2, 9)
    assert np.all(np.isfinite(features))

def test_transform_rejects_call_before_fit():
    transformer = MorletWaveletTransformer()

    with pytest.raises(
            RuntimeError,
            match="The transformer has not been fitted yet. Call 'fit' before 'transform'."
    ):
        transformer.transform(np.zeros((1,2, 320)))


def test_transformer_fit_and_transform_reject_non_3d_input():
    transformer = MorletWaveletTransformer()

    with pytest.raises(ValueError, match="Input data must be a 3D array"):
        transformer.fit(np.zeros((2, 320)))

    transformer.fit(np.zeros((2, 3, 320)))
    with pytest.raises(ValueError, match="Input data must be a 3D array"):
        transformer.transform(np.zeros((2, 320)))

def test_wavelet_pipeline_has_expected_steps():
    pipeline = create_wavelet_pipeline(sfreq=160.0)

    assert list(pipeline.named_steps) == [
        "wavelet",
        "scaler",
        "lda"
    ]

    assert pipeline.named_steps["wavelet"].sfreq == 160.0


def test_wavelet_pipeline_can_fit_and_predict():
    rng = np.random.default_rng(42)
    epochs = rng.normal(size=(8, 3, 320))
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pipeline = create_wavelet_pipeline(sfreq=160.0)

    predictions = pipeline.fit(epochs, labels).predict(epochs)

    assert predictions.shape == (8,)
    assert set(predictions).issubset(set(labels))


def test_wavelet_select_pipeline_propagates_sampling_frequency():
    pipeline = create_wavelet_select_pipeline(sfreq=sfreq)

    assert pipeline.named_steps["wavelet"].sfreq == 128.0
