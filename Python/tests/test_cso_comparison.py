import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_toolkit.radio.cso_comparison import normalize_spectrum_db
from solar_toolkit.radio.cso_comparison import (
    read_cso_comparison as _read_cso_comparison,
)


def read_cso_comparison(*args, **kwargs):
    kwargs.setdefault("bands_mhz", [238.0])
    return _read_cso_comparison(*args, **kwargs)


BASE = Time("2020-01-02").unix


def make_source(
    tmp_path, times, frequencies, total, name="source.fits", **header_changes
):
    header = fits.Header(
        {
            "DATE-OBS": "2020-01-02T00:00:00",
            "TIMESYS": "UTC",
            "POLARIZA": "RCP and LCP",
            "BUNIT": "SolarFluxUnit(SFU)",
        }
    )
    header.update(header_changes)
    total = np.asarray(total, dtype=float)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format=f"{len(times)}D", array=[times]),
            fits.Column(
                name="frequency", format=f"{len(frequencies)}D", array=[frequencies]
            ),
        ]
    )
    path = tmp_path / name
    fits.HDUList(
        [fits.PrimaryHDU(np.array([total / 4, 3 * total / 4]), header=header), table]
    ).writeto(path)
    return path


def test_native_weighted_band_mean_and_missing_bins(tmp_path):
    source = make_source(
        tmp_path,
        [0.01, 0.02, 0.21],
        [99.0, 100.0],
        [[2.0, 4.0, 10.0], [np.nan, 8.0, 20.0]],
    )
    result = read_cso_comparison([source], [BASE, BASE + 0.4], bands_mhz=[100.0])
    np.testing.assert_allclose(result.lightcurves_sfu[0, [0, 2]], [14 / 3, 15.0])
    np.testing.assert_array_equal(result.lightcurve_counts, [[3, 0, 2, 0]])
    assert np.isnan(result.lightcurves_sfu[0, [1, 3]]).all()
    np.testing.assert_allclose(result.spectrum_sfu[[0, 4], 0], [3.0, 8.0])
    np.testing.assert_array_equal(result.spectrum_counts[[0, 4], 0], [2, 1])
    assert result.time_edges[-1] == BASE + 0.4
    assert result.metadata["time_compression"] is False


def test_excludes_nonpositive_frequency_and_includes_band_edges(tmp_path):
    source = make_source(
        tmp_path,
        [0.01],
        [-1.0, 0.0, 237.0, 238.0, 239.0, 240.0],
        [[999.0], [999.0], [1.0], [2.0], [3.0], [100.0]],
    )
    result = read_cso_comparison([source], [BASE, BASE + 0.2])
    assert result.frequency_edges[0] == 237.0
    assert result.frequency_edges[-1] > 240.0
    assert result.lightcurves_sfu[0, 0] == 2.0
    assert result.lightcurve_counts[0, 0] == 3


def test_chunking_equivalent_and_sources_sorted_by_time(tmp_path):
    first = make_source(
        tmp_path,
        [0.01, 0.03],
        [237.0, 237.1, 238.0],
        [[1, 2], [3, np.nan], [4, 8]],
        name="z-first.fits",
    )
    second = make_source(
        tmp_path,
        [0.21, 0.23],
        [237.0, 237.1, 238.0],
        [[8, 16], [32, 64], [128, 256]],
        name="a-last.fits",
    )
    large = read_cso_comparison([second, first], [BASE, BASE + 0.3])
    tiny = read_cso_comparison([second, first], [BASE, BASE + 0.3], max_chunk_bytes=64)
    for key in (
        "spectrum_sfu",
        "spectrum_counts",
        "lightcurves_sfu",
        "lightcurve_counts",
    ):
        np.testing.assert_allclose(
            getattr(large, key), getattr(tiny, key), equal_nan=True
        )
    assert large.metadata["inputs"][0]["path"] == str(first)


def test_overlap_rejected(tmp_path):
    first = make_source(tmp_path, [0.01, 0.11], [238.0], [[1, 2]])
    other = make_source(tmp_path, [0.11, 0.21], [238.0], [[3, 4]], name="other.fits")
    with pytest.raises(ValueError, match="Overlapping"):
        read_cso_comparison([first, other], [BASE, BASE + 1])


@pytest.mark.parametrize(
    "changes",
    [{"TIMESYS": "TAI"}, {"POLARIZA": "LCP"}, {"BUNIT": "dB"}, {"BUNIT": "dB(SFU)"}],
)
def test_unreliable_source_rejected(tmp_path, changes):
    source = make_source(tmp_path, [0.01], [238.0], [[1]], **changes)
    with pytest.raises(ValueError, match="CSO requires"):
        read_cso_comparison([source], [BASE, BASE + 1])


def test_empty_and_outside_window_preserve_clock(tmp_path):
    source = make_source(tmp_path, [2.0, 3.0], [238.0], [[1, 2]])
    for paths in ([], [source]):
        result = read_cso_comparison(paths, [BASE, BASE + 0.2])
        assert result.spectrum_sfu.shape == (0, 2)
        assert result.lightcurves_sfu.shape == (1, 2)
        assert np.isnan(result.lightcurves_sfu).all()
        assert not result.lightcurve_counts.any()
        assert result.metadata["missing_time_bins"] == 2
    assert result.metadata["ignored_outside_window"] == [str(source)]


def test_exact_window_end_excluded_and_partial_bin_retained(tmp_path):
    source = make_source(tmp_path, [0.01, 0.21, 0.25], [238.0], [[1.0, 3.0, 999.0]])
    result = read_cso_comparison([source], [BASE, BASE + 0.25])
    np.testing.assert_allclose(result.lightcurves_sfu[0, [0, 2]], [1.0, 3.0])
    assert result.lightcurve_counts.sum() == 2
    assert result.time_edges[-1] == BASE + 0.25


def test_background_db_known_values_and_all_invalid():
    original = np.array([[1.0, 10.0, 100.0, np.nan], [0.0, -1.0, np.nan, np.inf]])
    db, meta = normalize_spectrum_db(original, percentile=0, upper_percentile=100)
    np.testing.assert_allclose(db[0, :3], [0.0, 10.0, 20.0])
    assert np.isnan(db[1]).all()
    assert meta["background_sfu"] == [1.0, None]
    assert meta["vmin"] == 0 and meta["vmax"] == 20.0
    db, meta = normalize_spectrum_db(np.full((2, 4), np.nan))
    assert np.isnan(db).all() and meta["vmax"] == 1.0
    assert np.isinf(original[1, 3])
