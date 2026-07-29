# SPDX-License-Identifier: GPL-3.0-only
"""Build native Radio Composite reference, sequence, media, and ZIP products."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path

_PREFIX = "APP_V1_EVENT "


def _emit(kind: str, payload: dict[str, object]) -> None:
    print(
        _PREFIX
        + json.dumps(
            {
                "schema_version": 1,
                "module_id": "radio-composite",
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-dir", required=True)
    parser.add_argument("--dart-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frequencies", default="")
    parser.add_argument(
        "--polarization", choices=("RR+LL", "RR", "LL"), default="RR+LL"
    )
    parser.add_argument("--roi-bounds", default="-300,-300,300,300")
    parser.add_argument("--dart-bandwidth-mhz", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--transform", choices=("linear", "log10"), default="linear")
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-frames", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = _run(args, output)
        for path, role in result:
            _emit("artifact", {"path": str(path), "role": role})
        _emit("progress", {"percent": 100})
        _emit("result", {"status": "succeeded", "artifact_count": len(result)})
        return 0
    except Exception as exc:
        _emit("log", {"level": "error", "message": str(exc)})
        _emit("log", {"level": "debug", "message": traceback.format_exc()})
        _emit("result", {"status": "failed"})
        return 1


def _run(args: argparse.Namespace, output: Path) -> list[tuple[Path, str]]:
    from solar_apps.frontends.radio.composite_figure.composite_figure_application import (
        build_centered_frequency_bands,
        build_composite_artifacts,
        build_request_signature,
        save_composite_bundle,
    )
    from solar_apps.frontends.radio.composite_figure.composite_sequence import (
        SequenceExportOptions,
        common_candidate_time_coverage,
        export_composite_sequences,
        group_candidates_by_frequency,
        render_source_map_candidate,
        resolve_single_band_frequency_source,
    )
    from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_app import (
        build_file_manifest,
        discover_frequency_options,
    )
    from solar_apps.frontends.radio.source_map.service import (
        PathPolicy,
        discover_candidates,
        parse_request_config,
    )
    from solar_apps.workflows.radio.configs import DEFAULT_CONFIG_NAME
    from solar_toolkit.radio.dart_spectrogram import (
        DartNarrowbandResult,
        extract_dart_narrowband_lightcurves,
    )
    from solar_toolkit.radio.roi_lightcurve import (
        RadioRoi,
        extract_radio_roi_lightcurve,
    )

    radio = Path(args.radio_dir).expanduser().resolve()
    dart = Path(args.dart_dir).expanduser().resolve()
    manifest = build_file_manifest(
        radio,
        pattern=args.pattern,
        recursive=bool(args.recursive),
    )
    if manifest.empty:
        raise ValueError("No radio FITS files matched the selected input")
    available_frame = discover_frequency_options(
        radio,
        pattern=args.pattern,
        recursive=bool(args.recursive),
    )
    available = _frequency_values(available_frame)
    requested = _float_list(args.frequencies) or available
    frequencies = [
        frequency
        for frequency in requested
        if any(math.isclose(frequency, item, abs_tol=1e-6) for item in available)
    ]
    if not frequencies:
        raise ValueError("No requested radio frequency is available")
    policy = PathPolicy((radio, dart, output))
    manifest_times = {
        str(Path(row.path).resolve()): str(row.inferred_obs_time)
        for row in manifest.itertuples()
        if str(getattr(row, "inferred_obs_time", "")).strip()
    }
    configs: dict[float, dict[str, object]] = {}
    candidates: list[dict[str, object]] = []
    source_map_output = output / "source-maps"
    for frequency in frequencies:
        source = resolve_single_band_frequency_source(
            radio,
            manifest,
            frequency,
            polarization=args.polarization,
        )
        config = parse_request_config(
            {
                "config": DEFAULT_CONFIG_NAME,
                "mode": "single_band",
                "source_path": str(source),
                "output_dir": str(source_map_output),
                "frequencies": [frequency],
                "polarization": args.polarization,
                "gaussian_overlay": True,
                "cmap": "hot",
                "spectrogram_panel": False,
                "start_idx": 0,
                "end_idx": None,
            },
            policy=policy,
        )
        discovered = discover_candidates(config, policy=policy)
        _normalize_candidate_contract(discovered, frequency, manifest_times)
        configs[frequency] = config
        candidates.extend(discovered)
    _emit(
        "log",
        {
            "level": "info",
            "message": (
                f"Prepared {len(candidates)} Source Map candidate(s); "
                f"frequencies={','.join(f'{value:g}' for value in frequencies)}"
            ),
        },
    )
    grouped = group_candidates_by_frequency(candidates, frequencies)
    start, end = common_candidate_time_coverage(grouped)
    reference_frequency = frequencies[0]
    reference_candidate = grouped[reference_frequency][0]
    reference_time = _utc(reference_candidate["observation_time"])
    if reference_time < start:
        reference_candidate = next(
            item
            for item in grouped[reference_frequency]
            if _utc(item["observation_time"]) >= start
        )
        reference_time = _utc(reference_candidate["observation_time"])
    bounds = _bounds(args.roi_bounds)
    roi = RadioRoi.from_box(*bounds, label="App 1.0 ROI")
    radio_paths: dict[float, list[Path]] = {
        frequency: _manifest_paths(
            manifest,
            frequency=frequency,
            start=start,
            end=end,
        )
        for frequency in frequencies
    }
    radio_curves = {
        frequency: extract_radio_roi_lightcurve(
            radio,
            roi,
            files=paths,
            freqs=[frequency],
            polarization=args.polarization,
        )
        for frequency, paths in radio_paths.items()
    }
    bands = build_centered_frequency_bands(
        frequencies,
        float(args.dart_bandwidth_mhz),
    )
    dart_results: dict[float, DartNarrowbandResult] = {}
    for frequency, band in bands.items():
        dart_results[frequency] = extract_dart_narrowband_lightcurves(
            dart,
            [frequency],
            band.bandwidth_mhz,
            time_range_utc=(start, end),
        )
    shared_times = tuple(dart_results[reference_frequency].time_utc)
    combined_dart = DartNarrowbandResult(
        time_utc=shared_times,
        curves=tuple(dart_results[frequency].curves[0] for frequency in frequencies),
    )
    map_png, map_metadata, _map_result = render_source_map_candidate(
        configs[reference_frequency],
        reference_candidate,
        reference_frequency,
        args.transform,
        source_map_output,
        1,
    )
    signature = build_request_signature(
        {
            "frequencies": frequencies,
            "polarization": args.polarization,
            "roi": roi.to_json_dict(),
            "time_start": start,
            "time_end": end,
            "fps": args.fps,
            "stride": args.stride,
            "dpi": args.dpi,
            "transform": args.transform,
        },
        source_paths=[
            *[path for paths in radio_paths.values() for path in paths],
        ],
    )
    reference_dart = dart_results[reference_frequency]
    reference_bundle = build_composite_artifacts(
        map_png,
        map_metadata,
        radio_curves[reference_frequency],
        reference_dart,
        roi=roi,
        map_time=reference_time,
        map_frequency_mhz=reference_frequency,
        polarization=args.polarization,
        time_start=start,
        time_end=end,
        request_signature=signature,
        source_context={
            "radio_directory": str(radio),
            "dart_directory": str(dart),
        },
        dpi=int(args.dpi),
    )
    reference_dir = save_composite_bundle(reference_bundle, output / "reference")

    def progress(completed: int, total: int, message: str) -> None:
        percent = round(100 * completed / max(1, total))
        _emit("progress", {"percent": percent, "message": message})

    sequence = export_composite_sequences(
        output,
        source_configs=configs,
        candidates_by_frequency=grouped,
        radio_curves=radio_curves,
        dart_result=combined_dart,
        dart_results_by_frequency=dart_results,
        roi=roi,
        reference_frequency_mhz=reference_frequency,
        reference_time=reference_time,
        polarization=args.polarization,
        time_start=start,
        time_end=end,
        request_signature=signature,
        source_context={
            "radio_directory": str(radio),
            "dart_directory": str(dart),
        },
        options=SequenceExportOptions(
            fps=float(args.fps),
            stride=int(args.stride),
            dpi=int(args.dpi),
            transform=args.transform,
            save_video=bool(args.save_video),
            save_frames=bool(args.save_frames),
        ),
        reference_bundle=reference_bundle,
        progress=progress,
    )
    artifacts: list[tuple[Path, str]] = [
        (sequence.zip_path, "composite-sequence-zip"),
        (sequence.metadata_path, "composite-sequence-metadata"),
    ]
    artifacts.extend((path, "composite-video") for path in sequence.videos.values())
    artifacts.extend(
        (path, "composite-reference")
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".png"
    )
    for directory in sequence.frame_directories.values():
        first = next(iter(sorted(directory.glob("*.png"))), None)
        if first is not None:
            artifacts.append((first, "composite-frame-preview"))
    return artifacts


def _frequency_values(frame) -> list[float]:
    if frame is None or frame.empty:
        return []
    column = "freq_mhz" if "freq_mhz" in frame else frame.columns[0]
    values = frame[column]
    return sorted({float(value) for value in values if not math.isnan(float(value))})


def _normalize_candidate_contract(
    candidates: list[dict[str, object]],
    frequency_mhz: float,
    manifest_times: dict[str, str],
) -> None:
    """Keep single-band candidates in the manifest's MHz/time contract."""

    for candidate in candidates:
        candidate["id"] = f"{frequency_mhz:g}mhz-{candidate['id']}"
        # The manifest already resolved this single-band directory in MHz.
        # Some historical FITS headers store the same frequency in Hz, so
        # keep the candidate contract in the catalog's MHz unit.
        candidate["frequencies_mhz"] = [frequency_mhz]
        if candidate.get("observation_time"):
            continue
        candidate["observation_time"] = next(
            (
                manifest_times.get(str(Path(str(path)).resolve()))
                for path in candidate.get("paths", ())
                if manifest_times.get(str(Path(str(path)).resolve()))
            ),
            None,
        )


def _float_list(raw: str) -> list[float]:
    return [float(item) for item in str(raw).replace(",", " ").split() if item]


def _bounds(raw: str) -> tuple[float, float, float, float]:
    values = _float_list(raw)
    if len(values) != 4:
        raise ValueError("ROI bounds require left,bottom,right,top")
    left, bottom, right, top = values
    if left >= right or bottom >= top:
        raise ValueError("ROI bounds must have positive width and height")
    return left, bottom, right, top


def _utc(value: object) -> datetime:
    import pandas as pd

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime().astimezone(UTC)


def _manifest_paths(
    manifest,
    *,
    frequency: float,
    start: datetime,
    end: datetime,
) -> list[Path]:
    import numpy as np
    import pandas as pd

    frequency_column = (
        "inferred_freq_mhz" if "inferred_freq_mhz" in manifest else "freq_mhz"
    )
    values = pd.to_numeric(manifest[frequency_column], errors="coerce")
    times = pd.to_datetime(manifest.get("inferred_obs_time"), errors="coerce", utc=True)
    tolerance = max(1e-6, abs(float(frequency)) * 1e-5)
    mask = np.abs(values - float(frequency)) <= tolerance
    if times.notna().any():
        mask &= times.isna() | (
            (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        )
    paths = [Path(value) for value in manifest.loc[mask, "path"].astype(str)]
    if not paths:
        raise ValueError(f"No radio files match {frequency:g} MHz")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
