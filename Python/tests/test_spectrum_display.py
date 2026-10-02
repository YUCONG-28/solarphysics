import numpy as np
import pytest
from astropy.time import Time

from solar_toolkit.radio.spectrum_display import (
    compact_spectrum_time,
    relative_background_db,
)


def test_short_gap_estimates_preserve_original_and_long_gaps():
    raw = np.array([[1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0]])
    out, meta = relative_background_db(raw, np.arange(7) * 0.1, max_gap_seconds=0.21)
    assert meta["estimated_cells"] == 1
    assert np.isfinite(out[0, 1]) and np.isnan(out[0, 3:6]).all()
    assert np.isnan(raw[0, 1])


def test_relative_units_invariant_to_gain():
    raw = np.array([[1.0, 1.0, 1.0, 10.0]])
    a, _ = relative_background_db(raw, np.arange(4))
    b, _ = relative_background_db(raw * 1e7, np.arange(4))
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, [[0, 0, 0, 10]])


def test_compaction_preserves_finite_nonpositive_measurements_and_real_utc():
    base = float(Time("2000-01-01T00:00:00", scale="utc").unix)
    values = np.array([[0.0, np.nan, -2.0, 4.0], [1.0, np.nan, np.nan, 6.0]])
    original = values.copy()
    arrays = {"intensity": values, "time_edges_unix": base + np.arange(5)}
    field, edges, retained, marker, info = compact_spectrum_time(arrays, base + 2.25)
    np.testing.assert_equal(retained, [0, 2, 3])
    np.testing.assert_equal(edges, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_equal(field, original[:, [0, 2, 3]])
    np.testing.assert_equal(values, original)
    assert marker == pytest.approx(1.25)
    assert info["cursor_original_utc"] == "2000-01-01T00:00:02.250000Z"
    assert info["cursor_utc_roundtrip_error_seconds"] == 0
    assert info["time_axis_is_continuous_utc"] is False


def test_compacted_cursor_is_absent_for_missing_or_outside_original_time():
    arrays = {
        "intensity": np.array([[1.0, np.nan, 2.0]]),
        "time_edges_unix": np.arange(4, dtype=float),
    }
    with pytest.raises(ValueError, match="no spectrum data"):
        compact_spectrum_time(arrays, 1.5)
    assert compact_spectrum_time(arrays, 1.5, allow_missing_time=True)[3] is None
    assert (
        compact_spectrum_time(arrays, 5.0, allow_outside_window=True)[4][
            "cursor_display_status"
        ]
        == "image_time_outside_spectrum_window"
    )
    assert compact_spectrum_time(arrays, 3.0)[3] == pytest.approx(2.0)


def test_relative_estimation_uses_original_gap_duration():
    base = float(Time("2000-01-01T00:00:00", scale="utc").unix)
    values = np.array([[1.0, np.nan, 3.0]])
    result, meta = relative_background_db(
        values, base + np.array([0.0, 0.1, 10.0]), max_gap_seconds=0.3
    )
    assert np.isnan(result[0, 1])
    assert meta["estimated_cells"] == 0
    assert "original UTC retained" in meta["interpolation"]
