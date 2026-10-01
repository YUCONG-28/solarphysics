"""Exercise the Python 3.10 datetime surface under a newer test interpreter."""

from __future__ import annotations

import subprocess
import sys


def test_radio_import_and_utc_filenames_do_not_require_python311_alias():
    code = """
import datetime as dt
import numpy
from astropy.io import fits

# Third-party dependencies are loaded before simulating the 3.10 stdlib surface.
if hasattr(dt, "UTC"):
    del dt.UTC
from solar_toolkit.radio import dart_spectrogram
from solar_toolkit.visualization.image_naming import format_utc_filename_time
from solar_toolkit.net.observations import ObservationQueryV1

assert format_utc_filename_time(dt.datetime(2025, 1, 24, 4, 48)) == "20250124T044800Z"
query = ObservationQueryV1(
    "compat", "sdo-aia-euv", "2025-01-24T04:48:00Z", "2025-01-24T04:49:00Z"
)
assert query.start_utc.utcoffset() == dt.timedelta(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
