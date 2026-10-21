"""Exercise exact time, explicit units and missing samples without observations."""

from __future__ import annotations

import argparse
import json

import astropy.units as u
import numpy as np

from solar_toolkit.radio.newkirk import plasma_density_from_frequency_mhz
from solar_toolkit.time import normalize_time
from solar_toolkit.timeseries.processing import derivative_series


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    # This published leap-second boundary is a calendar counterexample, not
    # an observation or an event selection preset.
    start = normalize_time("2016-12-31T23:59:59")
    end = normalize_time("2017-01-01T00:00:00")
    elapsed = (end.tai_ns - start.tai_ns) / 1_000_000_000
    assert elapsed == 2
    assert end.shift(10 * u.ns).tai_ns - end.tai_ns == 10

    density = plasma_density_from_frequency_mhz(100 * u.MHz)
    assert density == plasma_density_from_frequency_mhz(0.1 * u.GHz)
    derivative = derivative_series([0, 1, 2], spacing_seconds=500 * u.ms)
    np.testing.assert_allclose(derivative, 2)
    missing = derivative_series(np.ma.array([999.0], mask=[True]))
    assert np.isnan(missing[0])
    print(
        json.dumps(
            {
                "synthetic": True,
                "elapsed_tai_seconds": elapsed,
                "resolved_interval_ns": 10,
                "equivalent_frequency_density_cm3": density,
                "derivative_per_second": derivative.tolist(),
                "missing_sample_preserved": bool(np.isnan(missing[0])),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
