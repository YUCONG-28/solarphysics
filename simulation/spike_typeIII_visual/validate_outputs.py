"""Validate generated scientific data, PNG, GIF, and MP4 deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageSequence

EXPECTED_PNGS = {
    "harris_field.png",
    "tearing_structure.png",
    "current_density.png",
    "jet_structure.png",
    "electron_propagation.png",
    "typeIII_dynamic_spectrum.png",
    "mhd_diagnostics.png",
    "jet_diagnostics.png",
    "activity_timeline.png",
    "tearing_keyframes.png",
    "jet_keyframes.png",
    "electron_beam_keyframes.png",
    "typeIII_keyframes.png",
}
EXPECTED_ANIMATION_FRAMES = {
    "tearing": 30,
    "jet": 30,
    "electron_beam": 30,
    "typeIII": 30,
}
EXPECTED_GIFS = {
    f"{stem}.gif": minimum_frames
    for stem, minimum_frames in EXPECTED_ANIMATION_FRAMES.items()
}
EXPECTED_NPZ_FIELDS = {
    "x",
    "y",
    "time",
    "psi",
    "omega",
    "radio_time_s",
    "radio_frequency_mhz",
    "radio_intensity",
    "radio_ridge_frequency_mhz",
    "radio_injection_activity",
    "radio_jet_activity",
    "radio_conditioned_reconnection_activity",
    "spike_catalog",
    "jet_positive_speed",
    "jet_negative_speed",
    "jet_bidirectional_speed",
    "jet_activity_mhd",
    "reconnection_activity_mhd",
}
ATHENA_NPZ_FIELDS = {
    "rho",
    "pressure",
    "velocity_x",
    "velocity_y",
    "magnetic_x",
    "magnetic_y",
    "current_z",
    "internal_energy",
    "total_energy",
    "flux_difference",
    "xpoint_electric_field",
    "divergence_normalized_rms",
}
EXPECTED_DATA_FILES = {
    "data/diagnostics.csv",
    "data/mhd_snapshots.npz",
    "data/rmhd_fields.h5",
    "data/run_metadata.json",
}
BASE_MANIFEST_PATHS = {
    f"figures/{name}" for name in EXPECTED_PNGS
} | EXPECTED_DATA_FILES
SUPPORTED_ANIMATION_FORMATS = frozenset({"gif", "mp4"})
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
PRIVATE_PATH = re.compile(r"(?:/Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)")


def _expected_manifest_paths(
    animation_formats: tuple[str, ...],
    *,
    athena: bool = False,
) -> set[str]:
    animation_paths = {
        f"animations/{stem}.{animation_format}"
        for stem in EXPECTED_ANIMATION_FRAMES
        for animation_format in animation_formats
    }
    data_paths = {"data/mhd_bridge.h5"} if athena else set()
    return BASE_MANIFEST_PATHS | animation_paths | data_paths


# Historical outputs did not record export formats and were GIF-only.
EXPECTED_MANIFEST_PATHS = _expected_manifest_paths(("gif",))


def _animation_formats_from_metadata(
    metadata: dict[str, Any],
) -> tuple[tuple[str, ...], list[str]]:
    exports = metadata.get("exports")
    if not isinstance(exports, dict) or "animation_formats" not in exports:
        return ("gif",), []

    raw_formats = exports["animation_formats"]
    if not isinstance(raw_formats, list) or not all(
        isinstance(value, str) for value in raw_formats
    ):
        return (), ["metadata exports.animation_formats must be a list of strings."]

    animation_formats = tuple(raw_formats)
    errors: list[str] = []
    invalid = sorted(set(animation_formats) - SUPPORTED_ANIMATION_FORMATS)
    if invalid:
        errors.append(f"Unsupported metadata animation formats: {invalid}")
    if len(set(animation_formats)) != len(animation_formats):
        errors.append("metadata exports.animation_formats contains duplicates.")
    return animation_formats, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum_manifest(
    output_dir: Path,
    expected_paths: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    expected = EXPECTED_MANIFEST_PATHS if expected_paths is None else expected_paths
    errors: list[str] = []
    manifest_path = output_dir / "SHA256SUMS.txt"
    record: dict[str, Any] = {
        "name": manifest_path.name,
        "line_endings": None,
        "listed_files": [],
        "verified_files": 0,
    }
    if not manifest_path.is_file():
        errors.append("Missing checksum manifest: SHA256SUMS.txt")
        return record, errors

    raw = manifest_path.read_bytes()
    if b"\r" in raw:
        errors.append("Checksum manifest must use LF line endings.")
        record["line_endings"] = "contains-CR"
    else:
        record["line_endings"] = "LF"
    if raw and not raw.endswith(b"\n"):
        errors.append("Checksum manifest must end with an LF newline.")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"Checksum manifest is not valid UTF-8: {exc}")
        return record, errors

    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            errors.append(f"Malformed checksum line {line_number}: {line!r}")
            continue
        digest, relative_name = match.groups()
        relative_path = Path(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
        ):
            errors.append(
                f"Checksum path must be a normalized POSIX relative path: "
                f"{relative_name!r}"
            )
            continue
        if relative_name in entries:
            errors.append(f"Duplicate checksum entry: {relative_name}")
            continue
        entries[relative_name] = digest

    listed = set(entries)
    missing = sorted(expected - listed)
    unexpected = sorted(listed - expected)
    if missing:
        errors.append(f"Checksum manifest is missing entries: {missing}")
    if unexpected:
        errors.append(f"Checksum manifest has unexpected entries: {unexpected}")

    verified = 0
    for relative_name in sorted(listed & expected):
        path = output_dir / relative_name
        if not path.is_file():
            errors.append(f"Checksum target is missing: {relative_name}")
            continue
        actual = _sha256(path)
        if actual != entries[relative_name]:
            errors.append(f"Checksum mismatch: {relative_name}")
            continue
        verified += 1

    record["listed_files"] = sorted(entries)
    record["verified_files"] = verified
    return record, errors


def _inspect_mp4(path: Path) -> dict[str, Any]:
    reader = imageio.get_reader(path, format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        frame_count = int(reader.count_frames())
    finally:
        reader.close()
    return {
        "name": path.name,
        "size": list(metadata.get("size", ())),
        "frames": frame_count,
        "fps": float(metadata.get("fps", 0.0)),
        "codec": metadata.get("codec"),
        "bytes": path.stat().st_size,
    }


def validate(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = output_dir / "data" / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if not metadata_path.is_file():
        errors.append("Missing metadata: data/run_metadata.json")
    else:
        metadata_text = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_text)
        if PRIVATE_PATH.search(metadata_text):
            errors.append("run_metadata.json contains a personal absolute path.")
        schema_version = metadata.get("schema_version")
        if schema_version not in {2, 3, 4, 5, 6}:
            errors.append(
                "run_metadata.json must use schema_version 2, 3, 4, 5, or 6."
            )
        diagnostics = metadata.get("diagnostics", {})
        divergence = float(diagnostics.get("divergence_normalized_rms", np.inf))
        drift = abs(float(diagnostics.get("total_energy_drift_fraction", np.inf)))
        backend = metadata.get("mhd_backend")
        divergence_limit = 1e-6 if backend == "amrvac" else 1e-10
        if divergence >= divergence_limit:
            errors.append(f"Divergence residual is too large: {divergence}")
        drift_limit = 0.01 if backend == "athena" else 0.05
        if backend != "amrvac" and drift >= drift_limit:
            errors.append(f"Total-energy drift is too large: {drift}")
    animation_formats, format_errors = _animation_formats_from_metadata(metadata)
    errors.extend(format_errors)

    png_records: list[dict[str, Any]] = []
    figures_dir = output_dir / "figures"
    for name in sorted(EXPECTED_PNGS):
        path = figures_dir / name
        if not path.is_file():
            errors.append(f"Missing PNG: figures/{name}")
            continue
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=float)
            record = {
                "name": name,
                "size": list(image.size),
                "stddev": float(np.std(grayscale)),
                "bytes": path.stat().st_size,
            }
        if tuple(record["size"]) != (1600, 900):
            errors.append(f"Unexpected PNG size for {name}: {record['size']}")
        if record["stddev"] < 8.0:
            errors.append(f"PNG appears blank or nearly blank: {name}")
        png_records.append(record)

    gif_records: list[dict[str, Any]] = []
    mp4_records: list[dict[str, Any]] = []
    animations_dir = output_dir / "animations"
    if "gif" in animation_formats:
        for stem, minimum_frames in EXPECTED_ANIMATION_FRAMES.items():
            name = f"{stem}.gif"
            path = animations_dir / name
            if not path.is_file():
                errors.append(f"Missing GIF: animations/{name}")
                continue
            with Image.open(path) as image:
                frame_count = sum(1 for _ in ImageSequence.Iterator(image))
                record = {
                    "name": name,
                    "size": list(image.size),
                    "frames": frame_count,
                    "bytes": path.stat().st_size,
                }
            if tuple(record["size"]) != (960, 540):
                errors.append(f"Unexpected GIF size for {name}: {record['size']}")
            if frame_count < minimum_frames:
                errors.append(
                    f"GIF {name} has {frame_count} frames; "
                    f"expected >= {minimum_frames}."
                )
            gif_records.append(record)

    if "mp4" in animation_formats:
        for stem, minimum_frames in EXPECTED_ANIMATION_FRAMES.items():
            name = f"{stem}.mp4"
            path = animations_dir / name
            if not path.is_file():
                errors.append(f"Missing MP4: animations/{name}")
                continue
            try:
                record = _inspect_mp4(path)
            except (IndexError, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"Unable to decode MP4 {name}: {exc}")
                continue
            if tuple(record["size"]) != (960, 540):
                errors.append(f"Unexpected MP4 size for {name}: {record['size']}")
            if abs(record["fps"] - 10.0) > 0.1:
                errors.append(f"Unexpected MP4 frame rate for {name}: {record['fps']}")
            if "264" not in str(record["codec"]).lower():
                errors.append(f"Unexpected MP4 codec for {name}: {record['codec']}")
            if record["frames"] < minimum_frames:
                errors.append(
                    f"MP4 {name} has {record['frames']} frames; "
                    f"expected >= {minimum_frames}."
                )
            mp4_records.append(record)

    snapshot_path = output_dir / "data" / "mhd_snapshots.npz"
    npz_record: dict[str, Any] = {}
    if not snapshot_path.is_file():
        errors.append("Missing NPZ: data/mhd_snapshots.npz")
    else:
        with np.load(snapshot_path) as data:
            expected_npz = set(EXPECTED_NPZ_FIELDS)
            if metadata.get("mhd_backend") in {"athena", "amrvac"}:
                expected_npz.update(ATHENA_NPZ_FIELDS)
            missing = sorted(expected_npz - set(data.files))
            if missing:
                errors.append(f"NPZ is missing fields: {missing}")
            npz_record = {
                "fields": sorted(data.files),
                "snapshot_count": int(data["time"].size),
                "psi_shape": list(data["psi"].shape),
                "radio_shape": list(data["radio_intensity"].shape),
                "ridge_monotonic_decreasing": bool(
                    np.all(np.diff(data["radio_ridge_frequency_mhz"]) < 0.0)
                ),
            }
            if not npz_record["ridge_monotonic_decreasing"]:
                errors.append("Radio ridge is not strictly decreasing.")
            if not missing:
                catalog = np.asarray(data["spike_catalog"], dtype=float)
                if catalog.ndim != 2 or catalog.shape[1] != 5:
                    errors.append(
                        "spike_catalog must have shape (N, 5), including N=0."
                    )
                else:
                    radio_times = np.asarray(data["radio_time_s"], dtype=float)
                    ridge = np.asarray(
                        data["radio_ridge_frequency_mhz"],
                        dtype=float,
                    )
                    radio_config = metadata.get("config", {}).get("radio", {})
                    onset_start = float(radio_config.get("spike_onset_start_s", np.nan))
                    onset_end = min(
                        float(radio_config.get("spike_onset_cap_s", np.nan)),
                        float(radio_config.get("duration_s", np.nan))
                        * float(
                            radio_config.get(
                                "spike_onset_fraction",
                                np.nan,
                            )
                        ),
                    )
                    if catalog.size:
                        if not np.isfinite(catalog).all():
                            errors.append("spike_catalog contains non-finite values.")
                        spike_times = catalog[:, 0]
                        if not np.all(
                            (spike_times >= onset_start) & (spike_times <= onset_end)
                        ):
                            errors.append(
                                "Spike centers fall outside the onset window."
                            )
                        ridge_at_spikes = np.interp(
                            spike_times,
                            radio_times,
                            ridge,
                        )
                        if not np.all(catalog[:, 1] > ridge_at_spikes):
                            errors.append(
                                "At least one spike is not strictly above "
                                "the simultaneous Type III ridge."
                            )
                        coupling = metadata.get("config", {}).get("spike_coupling")
                        if coupling == "jet":
                            jet_config = metadata.get("config", {}).get(
                                "jet",
                                {},
                            )
                            q_jet = np.asarray(
                                data["radio_jet_activity"],
                                dtype=float,
                            )
                            q_reconnection = np.asarray(
                                data["radio_conditioned_reconnection_activity"],
                                dtype=float,
                            )
                            at_jet = np.interp(
                                spike_times,
                                radio_times,
                                q_jet,
                            )
                            at_reconnection = np.interp(
                                spike_times,
                                radio_times,
                                q_reconnection,
                            )
                            if not np.all(
                                at_jet >= float(jet_config.get("jet_threshold", np.inf))
                            ):
                                errors.append(
                                    "A conditioned spike is below the jet threshold."
                                )
                            if not np.all(
                                at_reconnection
                                >= float(
                                    jet_config.get(
                                        "reconnection_threshold",
                                        np.inf,
                                    )
                                )
                            ):
                                errors.append(
                                    "A conditioned spike is below the "
                                    "reconnection threshold."
                                )
                    expected_status = "events" if len(catalog) else "no_event"
                    actual_status = metadata.get("diagnostics", {}).get("event_status")
                    if actual_status != expected_status:
                        errors.append(
                            "metadata event_status is inconsistent with spike_catalog."
                        )

    if metadata.get("mhd_backend") in {"athena", "amrvac"}:
        bridge_path = output_dir / "data" / "mhd_bridge.h5"
        if not bridge_path.is_file():
            errors.append("Missing full-MHD bridge: data/mhd_bridge.h5")
        else:
            with h5py.File(bridge_path, "r") as bridge:
                bridge_schema = int(bridge.attrs.get("schema_version", 0))
                if bridge_schema not in {3, 4, 5}:
                    errors.append(
                        "Full-MHD bridge must use schema_version 3, 4, or 5."
                    )
                for name in (
                    "rho",
                    "pressure",
                    "velocity_x",
                    "velocity_y",
                    "magnetic_x",
                    "magnetic_y",
                    "current_z",
                    "time",
                ):
                    if name not in bridge:
                        errors.append(f"Athena bridge is missing dataset: {name}")
                if bridge_schema in {4, 5}:
                    for name in (
                        "velocity_z",
                        "magnetic_z",
                        "current_x",
                        "current_y",
                        "omega_x",
                        "omega_y",
                    ):
                        if name not in bridge:
                            errors.append(
                                f"2.5D full-MHD bridge is missing dataset: {name}"
                            )
                if bridge_schema == 5:
                    provenance = json.loads(
                        str(bridge.attrs.get("provenance_json", "{}"))
                    )
                    if metadata.get("mhd_backend") == "amrvac":
                        required = {
                            "solver",
                            "native_format_version",
                            "projection_method",
                            "energy_convention",
                        }
                        missing = sorted(required - set(provenance))
                        if missing:
                            errors.append(
                                f"AMRVAC bridge provenance is missing: {missing}"
                            )
                        if np.any(np.asarray(bridge["rho"]) <= 0.0):
                            errors.append("AMRVAC bridge contains non-positive density.")
                        if np.any(np.asarray(bridge["pressure"]) <= 0.0):
                            errors.append("AMRVAC bridge contains non-positive pressure.")

    expected_manifest_paths = _expected_manifest_paths(
        animation_formats,
        athena=metadata.get("mhd_backend") in {"athena", "amrvac"},
    )
    checksum_record, checksum_errors = _validate_checksum_manifest(
        output_dir,
        expected_manifest_paths,
    )
    errors.extend(checksum_errors)

    return {
        "ok": not errors,
        "output_dir": output_dir.name or ".",
        "errors": errors,
        "pngs": png_records,
        "gifs": gif_records,
        "mp4s": mp4_records,
        "animation_formats": list(animation_formats),
        "npz": npz_record,
        "metadata_diagnostics": metadata.get("diagnostics", {}),
        "checksums": checksum_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate(args.output_dir.resolve())
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
