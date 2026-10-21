from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.radio.muser_raw_preview import (
    draw_original,
    missing_intervals,
    _json_safe,
    display_time_selection,
    compressed_time,
    actual_time,
    display_bin_slice,
)
from solar_apps.workflows.radio.dispatcher import _COMMANDS


def test_no_data_intervals_preserve_real_clock():
    edges = np.arange(7, dtype=float) + 100
    assert missing_intervals(edges, [False, False, True, False, True, False]) == [
        (100.0, 102.0),
        (103.0, 104.0),
        (105.0, 106.0),
    ]


def test_internal_gaps_removed_without_interpolation_and_endpoints_retained():
    edges = np.arange(10, dtype=float) + 100
    valid = np.array([False, True, True, False, False, True, False, True, False])
    keep, removed = display_time_selection(edges, valid, True)
    assert removed == [(103.0, 105.0), (106.0, 107.0)]
    np.testing.assert_array_equal(
        keep, [True, True, True, False, False, True, False, True, True]
    )
    display_edges = compressed_time(np.r_[edges[:-1][keep], edges[-1]], removed)
    np.testing.assert_array_equal(display_edges, np.arange(7) + 100.0)
    original_centers = (edges[:-1] + edges[1:]) / 2
    display_centers = compressed_time(original_centers[keep], removed)
    np.testing.assert_array_equal(
        actual_time(display_centers, removed), original_centers[keep]
    )
    values = np.array([np.nan, 3, 7, np.nan, np.nan, 2, np.nan, 9, np.nan])
    np.testing.assert_array_equal(values[keep], [np.nan, 3, 7, 2, 9, np.nan])
    assert actual_time(compressed_time(105.0, removed), removed) == 105.0


def test_all_missing_or_single_observation_never_compresses_outer_no_data():
    for valid in ([False] * 5, [False, False, True, False, False], [True] * 5):
        keep, removed = display_time_selection(np.arange(6), valid)
        assert keep.all()
        assert removed == []


def test_compression_can_be_disabled():
    keep, removed = display_time_selection(np.arange(4), [True, False, True], False)
    assert keep.all()
    assert removed == []


def test_default_keeps_physical_clock_and_frequency_is_explicit():
    from solar_apps.workflows.radio.muser_raw_preview import radio_bands

    keep, removed = display_time_selection(np.arange(4), [True, False, True])
    assert keep.all() and not removed
    with pytest.raises(KeyError, match="frequency_mhz"):
        radio_bands({"dart_directory": "selected-band"})
    with pytest.raises(ValueError, match="positive"):
        radio_bands({"frequency_mhz": np.nan, "dart_directory": "selected-band"})


def test_display_zoom_preserves_original_bins_and_only_compresses_internal_holes():
    edges = np.arange(10, dtype=float) + 100
    selected = display_bin_slice(edges, [102.2, 107.8])
    assert selected == slice(2, 8)
    values = np.array([8, 9, 2, np.nan, 4, 6, np.nan, 10, 30], dtype=float)
    selected_edges = edges[selected.start : selected.stop + 1]
    keep, removed = display_time_selection(
        selected_edges, np.isfinite(values[selected]), True
    )
    np.testing.assert_array_equal(values[selected][keep], [2, 4, 6, 10])
    assert removed == [(103.0, 104.0), (106.0, 107.0)]
    # Endpoints on existing bin edges do not include the following/previous bin.
    assert display_bin_slice(edges, [102, 108]) == selected
    assert display_bin_slice(edges, [100, 109]) == slice(0, 9)
    for bounds in ([99, 105], [102, 110], [105, 105], [106, 105], [np.nan, 107]):
        with pytest.raises(ValueError):
            display_bin_slice(edges, bounds)


def test_extended_fov_ticks_follow_the_requested_coordinates():
    fig, axes = plt.subplots(1, 2)
    prepared = SimpleNamespace(data=None, header=None, metadata={"status": "no_data"})
    draw_original(axes[0], axes[1], prepared, "DART", [-2000, 2000, -2000, 2000])
    np.testing.assert_array_equal(axes[0].get_xticks(), [-2000, -1000, 0, 1000, 2000])
    np.testing.assert_array_equal(axes[0].get_yticks(), [-2000, -1000, 0, 1000, 2000])
    assert axes[0].get_xlim() == axes[0].get_ylim() == (-2000, 2000)
    plt.close(fig)


def test_missing_instrument_does_not_remove_original_panel():
    fig, axes = plt.subplots(1, 2)
    prepared = SimpleNamespace(
        data=None,
        header=None,
        metadata={"status": "no_data", "reason": "Outside tolerance"},
    )
    record = draw_original(
        axes[0], axes[1], prepared, "DART", [-1200, 1200, -1200, 1200]
    )
    assert axes[0].get_xlim() == (-1200, 1200)
    assert axes[0].get_ylim() == (-1200, 1200)
    assert any(t.get_text() == "NO DATA" for t in axes[0].texts)
    assert record["status"] == "no_data"
    plt.close(fig)


def test_raw_pixels_and_signed_values_are_preserved():
    image = np.arange(16, dtype=float).reshape(4, 4) - 8
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CDELT1": 200.0,
            "CDELT2": 200.0,
            "BUNIT": "Jy/beam",
        }
    )
    prepared = SimpleNamespace(
        data=image.copy(),
        header=header,
        metadata={"utc": "2020-01-02T04:14:38.400", "delta_seconds": -0.036},
    )
    fig, axes = plt.subplots(1, 2)
    record = draw_original(
        axes[0], axes[1], prepared, "MUSER", [-1200, 1200, -1200, 1200]
    )
    np.testing.assert_array_equal(prepared.data, image)
    np.testing.assert_array_equal(axes[0].collections[0].get_array(), image)
    assert record["display"]["vmin"] < 0
    assert record["display"]["native_pixel_stride"] == 1
    plt.close(fig)


def test_json_missing_is_null_and_entrypoint_is_distinct():
    assert _json_safe({"values": np.array([1.0, np.nan, np.inf])}) == {
        "values": [1.0, None, None]
    }
    assert _COMMANDS["muser-raw-preview"][0].endswith(".muser_raw_preview")


def test_two_bands_use_explicit_muser_planes_and_dart_flux_with_six_panels(
    tmp_path, monkeypatch
):
    import csv
    from astropy.time import Time
    from solar_toolkit.radio import raw_comparison, cso_comparison
    from solar_apps.workflows.radio.muser_raw_preview import build_preview

    target = Time("2020-01-02T04:14:38").unix
    requested = []

    def image(*args, **kwargs):
        requested.append(kwargs.get("frequency_mhz"))
        return SimpleNamespace(data=None, header=None, metadata={"status": "no_data"})

    monkeypatch.setattr(raw_comparison, "prepare_aia", image)
    monkeypatch.setattr(raw_comparison, "prepare_muser", image)
    monkeypatch.setattr(raw_comparison, "prepare_dart", image)

    def spectrum(paths, bounds, bands_mhz, half_width_mhz):
        assert bands_mhz == [190.0, 285.0]
        return cso_comparison.CsoComparison(
            np.linspace(target - 1, target + 1, 5),
            np.array([100.0, 200.0, 300.0]),
            np.ones((2, 4)),
            np.ones((2, 4), dtype=int),
            np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
            np.ones((2, 4), dtype=int),
            {},
        )

    monkeypatch.setattr(cso_comparison, "read_cso_comparison", spectrum)
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda self, path, **kwargs: path.write_bytes(b"preview")
    )
    config = {
        "event_id": "test",
        "reference_utc": "2020-01-02T04:14:38",
        "window_seconds": 1,
        "aia_directory": "aia",
        "muser_directory": "muser",
        "dart_directory": "unused",
        "dart_mode": "I",
        "cso_files": [],
        "radio_bands": [
            {"dart_mhz": 190, "muser_mhz": 168.0078125, "dart_directory": "190"},
            {"dart_mhz": 285, "muser_mhz": 277.7734375, "dart_directory": "285"},
        ],
    }
    result = build_preview(config, tmp_path / "result")
    assert requested == [None, None, 168.0078125, 277.7734375, 190.0, 285.0]
    assert len(result["panels"]) == 6
    assert result["image_layout"] == "3 columns x 2 rows"
    assert result["frequency_marker_mhz"] == [190.0, 285.0]
    rows = list(csv.DictReader((tmp_path / "result/flux.csv").open()))
    assert [
        float(r["flux_sfu"]) for r in rows if float(r["frequency_mhz"]) == 285.0
    ] == [10, 20, 30, 40]
