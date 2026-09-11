# SPDX-License-Identifier: GPL-3.0-only
"""Process-isolated STEREO EUVI Plot worker for App 1.0."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

_EVENT_PREFIX = "APP_V1_EVENT "


def _event(kind: str, payload: dict) -> None:
    print(
        _EVENT_PREFIX
        + json.dumps(
            {"schema_version": 1, "kind": kind, "payload": payload},
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("overview", "movie"), default="overview")
    parser.add_argument("--wavelengths", default="171,195,284,304")
    parser.add_argument("--target-time", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--calibration", choices=("legacy", "secchi-prep"), default="legacy"
    )
    parser.add_argument("--ssw-root", default=None)
    parser.add_argument("--idl-executable", default="idl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from solar_apps.workflows.visualization.stereo_euvi_plot import (
            EuvPlotConfig,
            make_roi_movie,
            plot_euvi_overview,
        )
        import matplotlib

        matplotlib.use("Agg")
        input_dir = Path(args.input_dir).expanduser().resolve(strict=True)
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        wavelengths = tuple(
            int(x) for x in args.wavelengths.split(",") if x.strip()
        ) or (171, 195, 284, 304)
        if args.calibration == "secchi-prep":
            from solar_toolkit.map.secchi import prepare_euvi, runtime_environment

            if not args.ssw_root:
                raise ValueError(
                    "SECCHI_PREP requires --ssw-root and a working IDL installation"
                )
            runtime_environment(Path(args.ssw_root), args.idl_executable)
            files = sorted(
                {
                    p.resolve()
                    for pattern in ("*.fts", "*.fits", "*.fit")
                    for p in input_dir.rglob(pattern)
                }
            )
            from astropy.io import fits

            files = [
                p
                for p in files
                if int(fits.getheader(p).get("WAVELNTH", 0)) in wavelengths
            ]
            if not files:
                raise FileNotFoundError("No requested EUVI bands found for calibration")
            calibrated = output_dir / "calibrated"
            for index, path in enumerate(files, 1):
                prepare_euvi(
                    path,
                    calibrated,
                    ssw_root=Path(args.ssw_root),
                    idl_executable=args.idl_executable,
                )
                _event("progress", {"percent": int(50 * index / len(files))})
            input_dir = calibrated
        target_time = None
        if args.target_time:
            target_time = datetime.fromisoformat(
                args.target_time.replace("Z", "+00:00")
            )
        roi_bounds = None
        if args.roi:
            parts = [float(x) for x in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError("--roi must be xmin,xmax,ymin,ymax")
            roi_bounds = (parts[0], parts[1], parts[2], parts[3])
        config = EuvPlotConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            wavelengths=wavelengths,
            target_time=target_time,
            roi_bounds=roi_bounds,
            fps=args.fps,
        )
        paths = (
            plot_euvi_overview(config)
            if args.mode == "overview"
            else make_roi_movie(config)
        )
        for path in paths:
            _event(
                "artifact",
                {
                    "path": str(path),
                    "role": (
                        "stereo-euvi"
                        if args.mode == "overview"
                        else "stereo-euvi-movie"
                    ),
                },
            )
        _event("progress", {"percent": 100})
        _event("result", {"status": "succeeded", "artifact_count": len(paths)})
        return 0
    except Exception as exc:
        _event("log", {"level": "error", "message": str(exc)})
        _event("log", {"level": "debug", "message": traceback.format_exc()})
        _event("result", {"status": "failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
