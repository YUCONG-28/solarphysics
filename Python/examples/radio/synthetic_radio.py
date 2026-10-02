"""Compose a synthetic FITS reader, UTC matching and missing-column display.

Run with an explicit private output directory outside the source tree, or below
the repository's ignored Local directory. All values are generated with seed 0 at the neutral 2000-01-01 epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from solar_toolkit.radio.cso_extract import extract_spectrum
from solar_toolkit.radio.muser_comparison import nearest_index
from solar_toolkit.radio.spectrum_display import compact_spectrum_time

REQUIRES_LOCAL_DATA = False


def run_example(output: Path) -> dict:
    """Generate tiny inputs and compose independently reusable functions."""
    repository = Path(__file__).resolve().parents[3]
    private_root = repository / "Local"
    output = output.expanduser().resolve()
    if (
        output.is_relative_to(repository)
        and not output.is_relative_to(private_root.resolve())
    ) or output == private_root.resolve():
        raise ValueError(
            "Output must be outside the source tree or below its ignored Local directory"
        )
    output.mkdir(parents=True, exist_ok=False)
    base = float(Time("2000-01-01T00:00:00", scale="utc").unix)
    times = np.array([0.0, 0.1, 0.2, 0.6, 0.7, 0.8])
    frequencies = np.array([100.0, 200.0])
    rng = np.random.default_rng(0)
    data = rng.uniform(1.0, 5.0, size=(2, 2, len(times)))
    header = fits.Header(
        {
            "TIMESYS": "UTC",
            "POLARIZA": "RCP and LCP",
            "BUNIT": "SFU",
            "DATE_BEG": "2000-01-01T00:00:00.000",
            "DATE_END": "2000-01-01T00:00:00.800",
        }
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format="6D", unit="s", array=[times]),
            fits.Column(name="frequency", format="2D", unit="MHz", array=[frequencies]),
        ]
    )
    source = output / "synthetic.fits"
    fits.HDUList([fits.PrimaryHDU(data, header), table]).writeto(source)
    arrays, metadata = extract_spectrum(
        [str(source)],
        (base, base + 0.9),
        quantized=False,
        time_bin_seconds=0.1,
        frequency_bounds=(100.0, 200.0),
        frequency_bin_mhz=None,
    )
    image_times = base + times
    target = base + 0.6
    image_index = nearest_index(image_times, target, tolerance=0.01)
    field, display_edges, retained, marker, display = compact_spectrum_time(
        arrays, target
    )
    np.savez(
        output / "display_arrays.npz",
        intensity=field,
        display_edges_seconds=display_edges,
        retained_original_columns=retained,
    )
    summary = {
        "synthetic": True,
        "seed": 0,
        "epoch_utc": "2000-01-01T00:00:00Z",
        "matched_image_index": image_index,
        "original_image_utc": Time(image_times[image_index], format="unix").isot,
        "display_cursor_seconds": marker,
        "extraction": metadata,
        "display": display,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_example(args.output_dir)
    print(
        f"Matched synthetic image {result['matched_image_index']}; "
        f"display omitted {result['display']['removed_no_data_time_columns']} empty columns."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
