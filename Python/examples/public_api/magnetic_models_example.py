"""Generate synthetic inputs for PFSS band analysis and local field readers."""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from solar_toolkit.modeling.nlfff import read_nlfff_grid, read_sharp_cea_boundary
from solar_toolkit.modeling.pfss import load_bundle, save_bundle
from solar_toolkit.radio.fieldline_association import fieldline_band_segments


def run_example(output: Path) -> None:
    """Write synthetic inputs to a new local directory and check their contracts."""
    output.mkdir(parents=True, exist_ok=False)
    # At the second harmonic, the selected frequency band corresponds
    # to field strengths between roughly 10 and 15 G on this line.
    line = dict(
        fieldline_id="synthetic-line",
        trace_valid=True,
        xyz_carrington_rsun=np.array([[1.1, 0, 0], [1.5, 0, 0], [2.0, 0, 0]]),
        b_gauss=np.array([20.0, 12.0, 8.0]),
    )
    bundle = output / "pfss"
    save_bundle(
        bundle,
        [line],
        [{"hpc_arcsec": np.full((3, 2), np.nan)}],
        {"config": {"rss": 2.5}},
    )
    _, lines = load_bundle(bundle)
    result = fieldline_band_segments(
        lines[0]["xyz_carrington_rsun"],
        (56.0, 84.0),
        mechanism="cyclotron",
        harmonic=2,
        rss=2.5,
        field_strength_gauss=lines[0]["b_gauss"],
    )
    segments = result["segments"]
    assert len(segments) == 1
    np.testing.assert_allclose(segments[0]["frequency_mhz"][[0, -1]], [84, 56])

    header = fits.Header(
        {
            "CTYPE1": "CRLN-CEA",
            "CTYPE2": "CRLT-CEA",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 100.0,
            "CRVAL2": 0.0,
            "CDELT1": 0.03,
            "CDELT2": 0.03,
            "BUNIT": "G",
            "QUALITY": 0,
            "HARPNUM": 1,
            "T_REC": "2000.01.01_00:00:00_TAI",
        }
    )
    paths = []
    for component, value in (("Br", 30.0), ("Bt", -20.0), ("Bp", 10.0)):
        path = output / f"synthetic.{component}.fits"
        fits.writeto(path, np.full((3, 4), value), header)
        paths.append(path)
    boundary = read_sharp_cea_boundary(*paths)
    np.testing.assert_array_equal(boundary.field_G[:, 0, 0], [10.0, 20.0, 30.0])
    assert boundary.valid_mask.all()

    grid_path = output / "local_field.npz"
    metadata = {
        "schema_version": 1,
        "frame": "local_west_north_radial",
        "array_axis_order": "component,z,y,x",
        "field_unit": "G",
        "length_unit": "Mm",
    }
    np.savez(
        grid_path,
        metadata=json.dumps(metadata),
        volume=np.stack([boundary.field_G, boundary.field_G], axis=1),
        spacing=[0.5, 0.5, 1.0],
        observed=boundary.field_G,
    )
    grid = read_nlfff_grid(grid_path)
    np.testing.assert_array_equal(grid.volume_G[:, 0], boundary.field_G)
    np.testing.assert_allclose(grid.spacing_Mm, [0.5, 0.5, 1.0])
    print(f"{len(segments)} field-line intervals; field volume {grid.volume_G.shape}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_example(args.output_dir.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
