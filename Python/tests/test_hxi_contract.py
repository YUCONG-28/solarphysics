"""Synthetic HXI table contracts; no instrument calibration is assumed."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from solar_toolkit.xray_dem.hxi import (
    DEFAULT_HXI_EPOCH,
    HXI_ENERGY_CHANNELS,
    load_hxi_lightcurve,
)


def install_hdus(monkeypatch, offsets, counts):
    class HDUs:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __getitem__(self, index):
            if index == 1:
                return SimpleNamespace(data=SimpleNamespace(TIME=offsets))
            assert index == 3
            return SimpleNamespace(data={"CTS_THINTHICK": counts})

    monkeypatch.setattr("astropy.io.fits.open", lambda path: HDUs())


def test_hxi_aware_epoch_and_bounds_convert_to_naive_utc_without_mutating_input(
    monkeypatch,
):
    offsets = np.array([0.0, 1.0, 2.0])
    counts = np.arange(12).reshape(3, 4)
    original = counts.copy()
    install_hdus(monkeypatch, offsets, counts)
    epoch = datetime(2019, 1, 1, tzinfo=timezone(timedelta(hours=8)))
    selected = load_hxi_lightcurve(
        "synthetic.fits",
        epoch + timedelta(seconds=1),
        epoch + timedelta(seconds=1),
        epoch=epoch,
    )
    assert selected["times"] == [DEFAULT_HXI_EPOCH + timedelta(seconds=1)]
    assert tuple(selected["data"]) == HXI_ENERGY_CHANNELS
    for index, channel in enumerate(HXI_ENERGY_CHANNELS):
        np.testing.assert_array_equal(selected["data"][channel], [counts[1, index]])
        assert not np.shares_memory(selected["data"][channel], counts)
    np.testing.assert_array_equal(counts, original)
    np.testing.assert_array_equal(offsets, [0, 1, 2])


def test_hxi_aware_bound_works_with_historical_naive_epoch(monkeypatch):
    install_hdus(monkeypatch, np.array([0.0, 1.0]), np.ones((2, 4)))
    start = DEFAULT_HXI_EPOCH.replace(tzinfo=timezone.utc) + timedelta(seconds=1)
    result = load_hxi_lightcurve("synthetic.fits", start_time=start)
    assert result["times"] == [DEFAULT_HXI_EPOCH + timedelta(seconds=1)]


def test_hxi_reversed_window_fails_before_reading(monkeypatch):
    def forbidden(path):
        raise AssertionError("must validate bounds before reading")

    monkeypatch.setattr("astropy.io.fits.open", forbidden)
    with pytest.raises(ValueError, match="start_time"):
        load_hxi_lightcurve(
            "synthetic.fits",
            DEFAULT_HXI_EPOCH + timedelta(seconds=1),
            DEFAULT_HXI_EPOCH,
        )


@pytest.mark.parametrize(
    "offsets",
    [np.array([[0.0, 1.0]]), np.array([0.0, np.nan]), np.array([0.0, np.inf])],
)
def test_hxi_requires_finite_one_dimensional_second_offsets(monkeypatch, offsets):
    install_hdus(monkeypatch, offsets, np.ones((2, 4)))
    with pytest.raises(ValueError, match="TIME"):
        load_hxi_lightcurve("synthetic.fits")


@pytest.mark.parametrize(
    "counts", [np.ones(2), np.ones((3, 4)), np.ones((2, 3)), np.ones((2, 5))]
)
def test_hxi_requires_exactly_four_channels_and_one_row_per_time(monkeypatch, counts):
    install_hdus(monkeypatch, np.array([0.0, 1.0]), counts)
    with pytest.raises(ValueError, match="CTS_THINTHICK"):
        load_hxi_lightcurve("synthetic.fits")


def test_hxi_empty_selection_has_all_four_empty_channels(monkeypatch):
    install_hdus(monkeypatch, np.array([0.0, 1.0]), np.ones((2, 4)))
    result = load_hxi_lightcurve(
        "synthetic.fits", DEFAULT_HXI_EPOCH + timedelta(seconds=2)
    )
    assert result["times"] == []
    assert tuple(result["data"]) == HXI_ENERGY_CHANNELS
    assert all(values.shape == (0,) for values in result["data"].values())
