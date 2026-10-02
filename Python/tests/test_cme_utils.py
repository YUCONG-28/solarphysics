from __future__ import annotations

import datetime as dt

import numpy as np


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
