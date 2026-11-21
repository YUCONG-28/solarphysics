"""Inspect a SHARP CEA boundary or a saved local magnetic field grid."""

from __future__ import annotations

import argparse
import json
import sys

from solar_apps.platform.paths import normalize_allowed_roots, validate_allowed_path
from solar_apps.platform.paths.native_dialog import NativeDialogError


def main(argv=None):
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    boundary = commands.add_parser(
        "inspect-boundary", help="Check three SHARP CEA FITS files"
    )
    boundary.add_argument("--br", required=True)
    boundary.add_argument("--bt", required=True)
    boundary.add_argument("--bp", required=True)
    boundary.add_argument("--allowed-roots", nargs="+", required=True)
    result = commands.add_parser(
        "inspect-result", help="Check a regular-grid NPZ result"
    )
    result.add_argument("--result", required=True)
    result.add_argument("--allowed-roots", nargs="+", required=True)
    args = parser.parse_args(argv)

    from solar_toolkit.modeling.nlfff import read_nlfff_grid, read_sharp_cea_boundary

    try:
        roots = normalize_allowed_roots(args.allowed_roots)
        if args.command == "inspect-boundary":
            from solar_toolkit.io import resolve_fits_path

            paths = [
                validate_allowed_path(
                    resolve_fits_path(path), allowed_roots=roots, kind="file"
                )
                for path in (args.br, args.bt, args.bp)
            ]
            data = read_sharp_cea_boundary(*paths)
            valid = int(data.valid_mask.sum())
            payload = {
                "kind": "sharp_cea_boundary",
                "shape": list(data.field_G.shape),
                "field_unit": "G",
                "component_order": ["west", "north", "radial"],
                "valid_pixel_count": valid,
                "invalid_pixel_count": data.valid_mask.size - valid,
                "metadata": data.metadata,
            }
        else:
            path = validate_allowed_path(args.result, allowed_roots=roots, kind="file")
            grid = read_nlfff_grid(path)
            payload = {
                "kind": "nlfff_grid",
                "shape": list(grid.volume_G.shape),
                "field_unit": "G",
                "length_unit": "Mm",
                "spacing_xyz_Mm": grid.spacing_Mm.tolist(),
                "has_reference": grid.reference_G is not None,
                "has_observed": grid.observed_G is not None,
                "metadata": grid.metadata,
            }
        print(json.dumps(payload, indent=2, allow_nan=False))
    except (OSError, ValueError, NativeDialogError) as exc:
        parser.error(str(exc))
    return 0
