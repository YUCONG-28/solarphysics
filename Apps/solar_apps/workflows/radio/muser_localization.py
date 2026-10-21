"""Explicit per-plane MUSER localization CLI with immutable input provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from solar_toolkit.radio.muser import read_muser_images
from solar_toolkit.radio.muser_localization import fit_radio_plane

__all__ = ["run_localization", "main"]


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _write_json(path, value):
    # Files are new and opened exclusively. A write failure propagates and no
    # completion marker is produced; it is never converted to a successful job.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _csv_cell(value):
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, (list, dict))
        else value
    )


def run_localization(config, output, *, base_directory="."):
    """Run explicit jobs into a new directory; preserve failed jobs as rows.

    Config: ``jobs`` is a list of ``path``, ``plane_index`` (zero based),
    ``roi_xyxy`` (integer half-open), and ``source_id`` records. ``fit_bg`` may
    be constant or plane at config level or per job. Relative input paths are
    resolved against ``base_directory``; no event or instrument defaults exist.
    """
    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("config.jobs must be a nonempty list")
    default_bg = config.get("fit_bg", "constant")
    if default_bg not in {"constant", "plane"}:
        raise ValueError("fit_bg must be constant or plane")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    _write_json(root / "resolved_config.json", config)
    base = Path(base_directory).resolve()
    rows, hashes = [], {}
    cached_path, cached_planes = None, None
    for index, job in enumerate(jobs):
        row = {
            "job_index": index,
            "source_id": job.get("source_id") if isinstance(job, dict) else None,
            "input_path": None,
            "input_sha256": None,
            "plane_index": None,
            "fit_success": False,
            "quality_status": "failed",
            "quality_reasons": [],
        }
        try:
            if not isinstance(job, dict):
                raise ValueError("Each job must be an object")
            if (
                not isinstance(job.get("source_id"), str)
                or not job["source_id"].strip()
            ):
                raise ValueError("source_id must be a nonempty string")
            source_path = Path(job["path"])
            if not source_path.is_absolute():
                source_path = base / source_path
            source_path = source_path.resolve()
            row["input_path"] = str(source_path)
            if source_path not in hashes:
                hashes[source_path] = _digest(source_path)
            row["input_sha256"] = hashes[source_path]
            plane_index = job["plane_index"]
            if (
                isinstance(plane_index, bool)
                or not isinstance(plane_index, int)
                or plane_index < 0
            ):
                raise ValueError("plane_index must be a nonnegative integer")
            row["plane_index"] = plane_index
            if source_path != cached_path:
                cached_planes = read_muser_images(source_path)
                cached_path = source_path
            if plane_index >= len(cached_planes):
                raise ValueError(
                    f"plane_index {plane_index} outside {len(cached_planes)} planes"
                )
            frequency, data, header, beam = cached_planes[plane_index]
            row.update(
                frequency_mhz=frequency,
                unit=header.get("BUNIT"),
                observation_time_header=header.get("DATE-OBS"),
                beam_raw={
                    key: value if math.isfinite(value) else None
                    for key, value in beam.items()
                },
                beam_unit_status="Not inferred by this workflow; raw metadata only",
                integration_time_status="Not inferred from cadence or filename",
            )
            row.update(
                fit_radio_plane(
                    data,
                    header,
                    job["roi_xyxy"],
                    background=job.get("fit_bg", default_bg),
                )
            )
        except Exception as exc:
            # Input/processing failures must not drop a source from the table.
            row.update(
                fit_success=False,
                quality_status="failed",
                quality_reasons=["input_or_processing_failure"],
                failure_detail=f"{type(exc).__name__}: {exc}",
            )
        _write_json(root / "jobs" / f"job_{index:06d}.json", row)
        rows.append(row)
    # Detect an input changed while it was being used before claiming completion.
    for path, digest in hashes.items():
        if _digest(path) != digest:
            raise RuntimeError(f"Input changed during execution: {path}")
    _write_json(root / "sources.json", rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "sources.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {key: _csv_cell(value) for key, value in row.items()} for row in rows
        )
    _write_json(
        root / "inputs.json",
        [{"path": str(path), "sha256": value} for path, value in hashes.items()],
    )
    failed = sum(not row["fit_success"] for row in rows)
    manifest = {
        "schema_version": 1,
        "status": "completed_with_failed_jobs" if failed else "completed",
        "job_count": len(rows),
        "failed_job_count": failed,
        "quality_review_count": sum(row["quality_status"] == "review" for row in rows),
        "files": [
            {"path": str(path.relative_to(root)), "sha256": _digest(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
    }
    _write_json(root / "COMPLETE.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    manifest = run_localization(
        config, args.output, base_directory=args.config.resolve().parent
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("status", "job_count", "failed_job_count")},
            ensure_ascii=False,
        )
    )
    return 2 if manifest["failed_job_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
