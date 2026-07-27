# SPDX-License-Identifier: GPL-3.0-only
"""Headless scientific operations used by native App 1.0 pages."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUN_ID = os.environ.get("APP_V1_RUN_ID") or uuid.uuid4().hex


def _event(module_id: str, kind: str, **payload: object) -> None:
    value = {
        "schema_version": 1,
        "run_id": _RUN_ID,
        "module_id": module_id,
        "kind": kind,
        "payload": payload,
    }
    print(f"APP_V1_EVENT {json.dumps(value, allow_nan=False)}", flush=True)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _bad_frame_discover(args: argparse.Namespace) -> list[Path]:
    from solar_apps.frontends.radio_bad_frame_review.review import BadFrameReviewStore

    roots = tuple(Path(item) for item in args.allowed_root)
    store = BadFrameReviewStore(args.output_dir, roots)
    payload = store.discover(args.input_dir)
    return [_write_json(Path(args.output_dir) / "bad-frame-discovery.json", payload)]


def _dart_render(args: argparse.Namespace) -> list[Path]:
    from solar_apps.frontends.radio.dart_spectrogram.dart_spectrogram_app import (
        build_dart_artifact_filenames,
        build_dynamic_spectrum_png,
        build_narrowband_png,
        inspect_dart_dataset,
        parse_center_frequencies,
    )
    from solar_toolkit.radio.dart_spectrogram import (
        extract_dart_narrowband_lightcurves,
        read_dart_spectrogram_window,
    )

    summary = inspect_dart_dataset(args.input_dir)
    _event("dart-spectrogram", "progress", percent=25)
    window = read_dart_spectrogram_window(
        summary.directory,
        max_frequency_samples=args.max_frequency_samples,
        max_time_samples=args.max_time_samples,
        chunk_memory_mb=args.chunk_memory_mb,
    )
    _event("dart-spectrogram", "progress", percent=65)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    centers = (
        parse_center_frequencies(args.center_frequencies)
        if args.center_frequencies.strip()
        else ()
    )
    narrowband = (
        extract_dart_narrowband_lightcurves(
            summary.directory,
            centers,
            args.bandwidth_mhz,
        )
        if centers
        else None
    )
    product_keys = (
        ("dynamic_spectrum", "selected_spectrum", "narrowband_lightcurve")
        if narrowband is not None
        else ("dynamic_spectrum",)
    )
    filenames = build_dart_artifact_filenames(summary, product_keys)
    artifacts: list[Path] = []
    image = output / filenames["dynamic_spectrum"]
    image.write_bytes(
        build_dynamic_spectrum_png(
            window,
            narrowband,
            dpi=args.dpi,
            display_mode=args.display_mode,
        )
    )
    artifacts.append(image)
    if narrowband is not None:
        selected = output / filenames["selected_spectrum"]
        selected.write_bytes(
            build_dynamic_spectrum_png(
                window,
                narrowband,
                dpi=args.dpi,
                display_mode=args.display_mode,
                region_label="Selected narrowband region",
            )
        )
        lightcurve = output / filenames["narrowband_lightcurve"]
        lightcurve.write_bytes(
            build_narrowband_png(
                narrowband,
                dpi=args.dpi,
                display_mode=args.display_mode,
            )
        )
        artifacts.extend((selected, lightcurve))
    metadata = _write_json(
        output / "dart-dataset-summary.json",
        {
            **asdict(summary),
            "time_range_utc": [
                value.isoformat().replace("+00:00", "Z")
                for value in summary.time_range_utc
            ],
        },
    )
    return [*artifacts, metadata]


def _roi_run(args: argparse.Namespace) -> list[Path]:
    from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_application import (
        run_radio_roi_lightcurve,
    )

    command = [
        "--radio-dir",
        args.input_dir,
        "--out-dir",
        args.output_dir,
        "--polarization",
        args.polarization,
        "--roi-bounds",
        args.roi_bounds,
    ]
    if args.frequencies:
        command.extend(["--freqs", args.frequencies])
    products = run_radio_roi_lightcurve(command)
    return sorted(
        {
            Path(value)
            for key, value in products.items()
            if key != "output_dir" and Path(value).is_file()
        }
    )


def _radio_composite_discover(args: argparse.Namespace) -> list[Path]:
    suffixes = {".fits", ".fit", ".fts"}

    def files(root: str) -> list[str]:
        return [
            str(path)
            for path in sorted(Path(root).rglob("*"))
            if path.is_file() and path.suffix.casefold() in suffixes
        ]

    payload = {
        "schema_version": 1,
        "kind": "radio-composite-discovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "radio_root": str(Path(args.input_dir).resolve()),
        "dart_root": str(Path(args.secondary_dir).resolve()),
        "radio_files": files(args.input_dir),
        "dart_files": files(args.secondary_dir),
    }
    return [
        _write_json(
            Path(args.output_dir) / "radio-composite-discovery.json",
            payload,
        )
    ]


def _source_map_discover(args: argparse.Namespace) -> list[Path]:
    suffixes = {".fits", ".fit", ".fts"}
    candidates = [
        str(path)
        for path in sorted(Path(args.input_dir).rglob("*"))
        if path.is_file() and path.suffix.casefold() in suffixes
    ]
    return [
        _write_json(
            Path(args.output_dir) / "source-map-files.json",
            {
                "schema_version": 1,
                "kind": "source-map-file-discovery",
                "root": str(Path(args.input_dir).resolve()),
                "candidates": candidates,
            },
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "bad-frame-discover",
            "dart-render",
            "roi-run",
            "radio-composite-discover",
            "source-map-discover",
        ),
    )
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--secondary-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-root", action="append", default=[])
    parser.add_argument("--polarization", default="L+R")
    parser.add_argument("--roi-bounds", default="-300,-300,300,300")
    parser.add_argument("--frequencies", default="")
    parser.add_argument("--max-frequency-samples", type=int, default=1200)
    parser.add_argument("--max-time-samples", type=int, default=1200)
    parser.add_argument("--chunk-memory-mb", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--center-frequencies", default="")
    parser.add_argument("--bandwidth-mhz", type=float, default=2.0)
    parser.add_argument("--display-mode", choices=("db", "linear"), default="db")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operations: dict[str, Any] = {
        "bad-frame-discover": _bad_frame_discover,
        "dart-render": _dart_render,
        "roi-run": _roi_run,
        "radio-composite-discover": _radio_composite_discover,
        "source-map-discover": _source_map_discover,
    }
    _event(args.module_id, "progress", percent=5)
    _event(args.module_id, "log", message=f"Starting {args.operation}")
    artifacts = operations[args.operation](args)
    for index, artifact in enumerate(artifacts, start=1):
        kind = (
            "preview" if artifact.suffix.casefold() in {".png", ".jpg"} else "artifact"
        )
        _event(
            args.module_id,
            kind,
            path=str(artifact),
            media_type=_media_type(artifact),
        )
        _event(
            args.module_id,
            "progress",
            percent=70 + int(25 * index / max(len(artifacts), 1)),
        )
    _event(args.module_id, "result", artifact_count=len(artifacts))
    return 0


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
    }.get(path.suffix.casefold(), "application/octet-stream")


if __name__ == "__main__":
    raise SystemExit(main())
