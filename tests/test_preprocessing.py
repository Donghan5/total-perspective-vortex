import numpy as np
import pytest

from src.preprocessing import encode_binary_labels

def test_encode_binary_labels_maps_t1_and_t2() -> None:
    event_codes = np.array([7, 13, 7, 13])
    target_event_id = {
        'T1': 7,
        'T2': 13
    }

    actual = encode_binary_labels(event_codes, target_event_id)

    expected = np.array([0, 1, 0, 1])

    np.testing.assert_array_equal(actual, expected)


def test_encode_binary_labels_rejects_unknown_codes() -> None:
    event_codes = np.array([7, 13, 99])
    target_event_id = {
        'T1': 7,
        'T2': 13
    }

    with pytest.raises(
        ValueError,
        match="Invalid event codes found"
    ):
        encode_binary_labels(event_codes, target_event_id)
    