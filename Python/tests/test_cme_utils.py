from __future__ import annotations

import datetime as dt

import numpy as np
import pytest


def test_lasco_timestamp_scan_and_running_difference(tmp_path):
    from solar_toolkit.cme import (
        extract_lasco_timestamp,
        running_difference,
        scan_lasco_files,
    )

    first = tmp_path / "lasco_c2_20000101_044800.jp2"
    second = tmp_path / "lasco_c2_20000101_045000.jp2"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    assert extract_lasco_timestamp(first.name) == dt.datetime(2000, 1, 1, 4, 48)
    assert [path.name for path in scan_lasco_files(tmp_path)] == [
        first.name,
        second.name,
    ]
    assert np.array_equal(
        running_difference(np.array([[1, 4], [3, 8]]), np.array([[1, 1], [5, 2]])),
        np.array([[0, 3], [-2, 6]]),
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_running_difference_preserves_negative_values_and_inputs(dtype):
    from solar_toolkit.cme import running_difference

    current = np.array([[10, 30]], dtype=dtype)
    previous = np.array([[20, 10]], dtype=dtype)
    result = running_difference(current, previous)
    np.testing.assert_array_equal(result, [[-10, 20]])
    assert result.dtype.kind == "f"
    np.testing.assert_array_equal(current, [[10, 30]])
    np.testing.assert_array_equal(previous, [[20, 10]])


def test_running_difference_rejects_broadcasting():
    from solar_toolkit.cme import running_difference

    with pytest.raises(ValueError, match="shape"):
        running_difference(np.zeros((3, 3)), np.zeros(3))


def test_running_difference_preserves_both_masks_and_converts_units():
    import astropy.units as u
    from astropy.nddata import NDData

    from solar_toolkit.cme import running_difference

    current = NDData([1000.0, 2000.0, 3000.0], unit=u.m, mask=[False, True, False])
    previous = NDData([1.0, 1.0, 1.0], unit=u.km, mask=[False, False, True])
    np.testing.assert_array_equal(
        running_difference(current, previous), [0, np.nan, np.nan]
    )
    np.testing.assert_array_equal(current.data, [1000, 2000, 3000])
    np.testing.assert_array_equal(previous.data, [1, 1, 1])
    with pytest.raises(u.UnitConversionError):
        running_difference(1 * u.m, 1 * u.s)
    masked = np.ma.array([10, 1000], mask=[False, True])
    np.testing.assert_array_equal(running_difference(masked, [1, 1]), [9, np.nan])
