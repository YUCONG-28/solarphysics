import numpy as np
import pytest
from solar_apps.workflows.radio.muser_preview import (
    _segments,
    _requested_fov,
    _requested_frequencies,
)
from solar_apps.workflows.radio.dispatcher import _COMMANDS


def test_gaps_are_not_bridged():
    segments = _segments(np.array([0.0, 1.0, 2.0, 8.0, 9.0]))
    assert [list(s) for s in segments] == [[0, 1, 2], [3, 4]]


def test_command_is_registered():
    assert _COMMANDS["muser-preview"] == (
        "solar_apps.workflows.radio.muser_preview",
        "main",
    )


def test_reference_fov_is_exact_and_rejects_invalid_bounds():
    assert _requested_fov({"fov_arcsec": [-10, 10, -20, 20]}) == [-10, 10, -20, 20]
    assert _requested_fov({}) is None
    with pytest.raises(ValueError):
        _requested_fov({"fov_arcsec": [10, -10, -20, 20]})


def test_frequencies_are_explicit_and_not_limited_to_one_observation():
    assert _requested_frequencies({"frequencies_mhz": [100, 200]}) == (100.0, 200.0)
    for frequencies in ([], [100, 100], [100, -1], [np.nan]):
        with pytest.raises(ValueError, match="distinct finite positive"):
            _requested_frequencies({"frequencies_mhz": frequencies})
