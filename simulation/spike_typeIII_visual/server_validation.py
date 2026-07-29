"""Formal validation for the dual-partition CPU scientific delivery."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .validate_outputs import EXPECTED_PNGS

SCIENTIFIC_STEMS = (
    "causal_chain",
    "reconnection_topology",
    "bidirectional_outflow",
    "radio_event_control",
)
PRIVATE_TEXT = re.compile(
    r"(?:/Users/[^/\s]+|/home/[^/\s]+|/public/home/[^/\s]+|"
    r"[A-Za-z]:\\Users\\[^\\\s]+|\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(run: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    path = run / "SHA256SUMS.txt"
    listed: set[str] = set()
    verified = 0
    if not path.is_file():
        return {"verified": 0, "listed": 0, "complete": False}, ["Missing SHA256SUMS.txt"]
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        errors.append("SHA256SUMS.txt must use LF and end with LF.")
    for line in raw.decode("utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"Malformed checksum line: {line!r}")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"Unsafe checksum path: {relative}")
            continue
        listed.add(relative)
        target = run / relative
        if not target.is_file():
            errors.append(f"Missing listed file: {relative}")
        elif _sha256(target) != expected:
            errors.append(f"Checksum mismatch: {relative}")
        else:
            verified += 1
    actual = {
        path.relative_to(run).as_posix()
        for path in run.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS.txt", "validation_report.json"}
    }
    missing = sorted(actual - listed)
    unexpected = sorted(listed - actual)
    if missing:
        errors.append(f"Manifest coverage missing: {missing}")
    if unexpected:
        errors.append(f"Manifest has stale entries: {unexpected}")
    return {
        "verified": verified,
        "listed": len(listed),
        "complete": not missing and not unexpected,
    }, errors


def _probe_with_ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_packets,codec_name,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_packets"]),
        "codec": stream["codec_name"],
        "pixel_format": stream.get("pix_fmt"),
        "probe_backend": "ffprobe",
    }


def _probe_with_ffmpeg(path: Path) -> dict[str, Any]:
    """Inspect a video using the bundled imageio FFmpeg without decoding frames."""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Video validation requires ffprobe or imageio-ffmpeg."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "framehash",
            "-",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    video_lines = [
        line for line in result.stderr.splitlines() if "Video:" in line
    ]
    if not video_lines:
        raise RuntimeError(f"FFmpeg found no video stream in {path.name}.")
    video_line = video_lines[-1]
    codec_match = re.search(r"Video:\s*([^,\s]+)", video_line)
    pixel_match = re.search(r"Video:[^,]+,\s*([^,\s(]+)", video_line)
    size_match = re.search(r"(?<!\d)(\d{2,6})x(\d{2,6})(?!\d)", video_line)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", video_line)
    frame_count = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in result.stdout.splitlines()
    )
    if not all((codec_match, pixel_match, size_match, fps_match, frame_count)):
        raise RuntimeError(
            f"Unable to parse formal video metadata for {path.name}: "
            f"{video_line.strip()}"
        )
    return {
        "width": int(size_match.group(1)),
        "height": int(size_match.group(2)),
        "fps": float(fps_match.group(1)),
        "frames": frame_count,
        "codec": codec_match.group(1),
        "pixel_format": pixel_match.group(1),
        "probe_backend": "imageio-ffmpeg",
    }


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        return _probe_with_ffprobe(path, ffprobe)
    return _probe_with_ffmpeg(path)


def _privacy_scan(run: Path) -> list[str]:
    errors = []
    for path in run.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".json",
            ".csv",
            ".txt",
            ".md",
            ".log",
        }:
            text = path.read_text(encoding="utf-8", errors="replace")
            if PRIVATE_TEXT.search(text):
                errors.append(f"Private path/address in {path.relative_to(run)}")
    return errors


def validate_cpu_delivery(run_root: Path, *, write_report: bool = True) -> dict[str, Any]:
    """Validate the rendered CPU event delivery without CUDA assumptions."""

    run = Path(run_root).resolve()
    errors: list[str] = []
    metadata = json.loads((run / "data/run_metadata.json").read_text())
    diagnostics = metadata["diagnostics"]
    runtime = metadata["runtime"]
    exports = metadata["exports"]
    control_id = exports.get("control_run_id")
    control = run.parent / str(control_id) if control_id else None
    checks: dict[str, bool] = {
        "cpu_float64": (
            runtime["execution_backend"] == "torch"
            and runtime["execution_device"] == "cpu"
            and runtime["execution_precision"] == "float64"
        ),
        "fine_event_profile": metadata["config"]["profile"] == "rmhd-fine-event",
        "scientific_4k_render": exports["render_profile"] == "scientific-4k",
        "gif_and_mp4": set(exports["animation_formats"]) == {"gif", "mp4"},
        "event_contract": (
            diagnostics["event_status"] == "events"
            and int(diagnostics["event_count"]) == 12
            and float(diagnostics["minimum_topping_margin_mhz"]) > 0.0
            and float(diagnostics["jet_coincidence_fraction"]) == 1.0
        ),
        "divergence": float(diagnostics["divergence_normalized_rms"]) < 1e-10,
        "energy_budget": float(
            diagnostics["energy_budget_max_abs_fraction"]
        )
        < 2e-4,
    }

    hdf5_path = run / "data/rmhd_fields.h5"
    with h5py.File(hdf5_path, "r") as handle:
        checks["event_hdf5"] = bool(
            handle["state/psi"].shape == (401, 512, 1024)
            and handle["state/omega"].shape == (401, 512, 1024)
            and bool(np.isfinite(handle["state/psi"][-1]).all())
            and bool(np.isfinite(handle["state/omega"][-1]).all())
        )

    checks["control_contract"] = False
    if control is not None and control.is_dir():
        control_meta = json.loads(
            (control / "data/run_metadata.json").read_text()
        )
        control_diag = control_meta["diagnostics"]
        checks["control_contract"] = (
            control_meta["config"]["profile"] == "rmhd-fine-control"
            and control_diag["event_status"] == "no_event"
            and int(control_diag["event_count"]) == 0
        )

    science = json.loads(
        (run / "data/science_suite_summary.json").read_text()
    )
    benchmark = json.loads(
        (run / "data/cpu_partition_benchmark.json").read_text()
    )
    checks["science_suite"] = bool(science["passed"])
    checks["dual_partition_benchmark"] = (
        benchmark["repeats"] >= 3
        and benchmark["scientific_arrays_consistent"] is True
        and set(benchmark["records"]) == {"AMD7742", "E74809"}
    )

    rows = 0
    with (run / "data/diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = sum(1 for _ in csv.DictReader(stream))
    checks["diagnostic_rows"] = rows == 401

    png_records = {}
    for name in EXPECTED_PNGS:
        path = run / "figures" / name
        if not path.is_file():
            errors.append(f"Missing PNG: {name}")
            continue
        png_records[name] = list(Image.open(path).size)
    checks["static_4k"] = (
        len(png_records) == len(EXPECTED_PNGS)
        and all(size == [3840, 2160] for size in png_records.values())
    )

    media = {}
    media_ok = True
    for stem in SCIENTIFIC_STEMS:
        mp4 = run / "animations" / f"{stem}.mp4"
        master = run / "animations" / f"{stem}_master_ffv1.mkv"
        gif = run / "animations" / f"{stem}.gif"
        if not all(path.is_file() for path in (mp4, master, gif)):
            errors.append(f"Missing media set: {stem}")
            media_ok = False
            continue
        delivery_probe = _probe(mp4)
        master_probe = _probe(master)
        with Image.open(gif) as image:
            gif_record = {"size": list(image.size), "frames": image.n_frames}
        media[stem] = {
            "delivery": delivery_probe,
            "master": master_probe,
            "gif": gif_record,
        }
        media_ok &= (
            delivery_probe["width"] == 3840
            and delivery_probe["height"] == 2160
            and abs(delivery_probe["fps"] - 30.0) < 1e-6
            and delivery_probe["frames"] >= 401
            and delivery_probe["codec"] == "h264"
            and delivery_probe["pixel_format"] == "yuv420p"
            and master_probe["width"] == 3840
            and master_probe["height"] == 2160
            and master_probe["frames"] >= 401
            and master_probe["codec"] == "ffv1"
            and gif_record["size"] == [960, 540]
            and gif_record["frames"] >= 30
        )
    checks["formal_media"] = bool(media_ok)

    manifest_record, manifest_errors = _manifest(run)
    checks["manifest"] = not manifest_errors
    errors.extend(manifest_errors)
    privacy_errors = _privacy_scan(run)
    checks["privacy"] = not privacy_errors
    errors.extend(privacy_errors)
    for name, passed in checks.items():
        if not passed:
            errors.append(f"Failed check: {name}")

    report = {
        "schema": "rmhd-cpu-dual-partition-delivery-v1",
        "run_id": run.name,
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "manifest": manifest_record,
        "figures": png_records,
        "media": media,
        "scientific_boundary": (
            "2-D incompressible RMHD with phenomenological radio proxy; "
            "not a self-consistent 2.5-D coronal jet"
        ),
    }
    if write_report:
        (run / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
