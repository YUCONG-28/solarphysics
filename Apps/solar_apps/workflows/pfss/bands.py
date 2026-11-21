"""Find conditional radio-frequency bands on saved PFSS field lines."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout, validate_private_output_path
from solar_apps.platform.paths import normalize_allowed_roots, validate_allowed_path
from solar_apps.platform.paths.native_dialog import NativeDialogError


def _source_surface(metadata, explicit_rss):
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("PFSS bundle config must be a JSON object")
    stored_rss = config.get("rss")
    if stored_rss is None:
        if explicit_rss is None:
            raise ValueError("Bundle has no source-surface radius. Supply --rss.")
        rss = explicit_rss
    else:
        if isinstance(stored_rss, bool) or not isinstance(stored_rss, (int, float)):
            raise ValueError("PFSS bundle source-surface radius must be a number")
        rss = float(stored_rss)
        if explicit_rss is not None and not math.isclose(explicit_rss, rss):
            raise ValueError(
                "--rss must agree with the source-surface radius in the bundle"
            )
    if not math.isfinite(rss) or rss <= 1:
        raise ValueError(
            "Source-surface radius must be finite and greater than 1 R_sun"
        )
    return rss


def _line_result(record, bounds, *, mechanism, harmonic, rss, newkirk_multiplier):
    from solar_toolkit.radio.fieldline_association import fieldline_band_segments

    result = {"fieldline_id": record["fieldline_id"], "segments": []}
    if not record.get("trace_valid"):
        return {
            **result,
            "status": "trace_invalid",
            "reason": record.get("reason", "invalid_trace"),
        }
    strengths = record.get("b_gauss")
    if mechanism == "cyclotron" and strengths is None:
        return {
            **result,
            "status": "field_strength_unavailable",
            "reason": record.get("field_strength_status", "not_recorded"),
        }
    bands = fieldline_band_segments(
        record["xyz_carrington_rsun"],
        bounds,
        mechanism=mechanism,
        harmonic=harmonic,
        rss=rss,
        newkirk_multiplier=newkirk_multiplier,
        field_strength_gauss=strengths if mechanism == "cyclotron" else None,
    )
    return {
        **result,
        "status": bands["metadata"]["status"],
        "segments": [
            {key: value.tolist() for key, value in segment.items()}
            for segment in bands["segments"]
        ],
        "diagnostics": bands["metadata"],
    }


def _write_new_file(output: Path, contents: str) -> None:
    # A hard link publishes a complete file and refuses a concurrent replacement.
    with tempfile.TemporaryDirectory(prefix=".pfss-bands-", dir=output.parent) as stage:
        staged = Path(stage) / "bands.json"
        staged.write_text(contents, encoding="utf-8")
        os.link(staged, output)


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--bundle", required=True, help="Completed PFSS bundle directory"
    )
    parser.add_argument("--frequency-bounds-mhz", type=float, nargs=2, required=True)
    parser.add_argument("--mechanism", choices=("plasma", "cyclotron"), required=True)
    parser.add_argument("--harmonic", type=int, required=True)
    parser.add_argument("--newkirk-multiplier", type=float, help="Required for plasma")
    parser.add_argument(
        "--rss", type=float, help="Radius in R_sun if absent from the bundle"
    )
    parser.add_argument(
        "--output", help="New JSON file outside the bundle (default: stdout)"
    )
    parser.add_argument("--allowed-roots", nargs="+", required=True)
    args = parser.parse_args(argv)
    lo, hi = args.frequency_bounds_mhz
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
        parser.error(
            "--frequency-bounds-mhz requires increasing, finite positive bounds"
        )
    if args.harmonic <= 0:
        parser.error("--harmonic must be a positive integer")
    multiplier = args.newkirk_multiplier
    if args.mechanism == "plasma":
        if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0:
            parser.error(
                "Plasma analysis requires a finite positive --newkirk-multiplier"
            )
    elif multiplier is not None:
        parser.error("--newkirk-multiplier is only used for plasma analysis")

    from solar_toolkit.modeling.pfss import load_bundle

    try:
        layout = RuntimeLayout.discover()
        roots = normalize_allowed_roots(
            args.allowed_roots, workspace_root=layout.repo_root
        )
        bundle = validate_allowed_path(
            args.bundle, allowed_roots=roots, kind="directory"
        )
        output = None
        if args.output:
            output = validate_allowed_path(
                args.output,
                allowed_roots=roots,
                kind="save_file",
                default_suffix=".json",
            )
            output = validate_private_output_path(output, repo_root=layout.repo_root)
            if output.is_relative_to(bundle):
                raise ValueError("Output must be outside the completed PFSS bundle")
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
        metadata, records = load_bundle(bundle)
        rss = _source_surface(metadata, args.rss)
        results = [
            _line_result(
                record,
                (lo, hi),
                mechanism=args.mechanism,
                harmonic=args.harmonic,
                rss=rss,
                newkirk_multiplier=multiplier,
            )
            for record in records
        ]
        payload = {
            "schema_version": 1,
            "bundle": str(bundle),
            "frame": "heliographic_carrington",
            "length_unit": "R_sun",
            "geometry_method": "conditional_model_candidate",
            "frequency_bounds_mhz": [lo, hi],
            "mechanism": args.mechanism,
            "harmonic": args.harmonic,
            "rss_rsun": rss,
            "newkirk_multiplier": multiplier,
            "fieldlines": results,
        }
        contents = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if output is None:
            print(contents, end="")
        else:
            _write_new_file(output, contents)
    except (OSError, ValueError, NativeDialogError) as exc:
        parser.error(str(exc))
    return 0
