"""Freeze radio FITS selection by UTC identity and content hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STAMP = re.compile(
    r"_(\d{4})(\d{1,2})(\d{2})_(\d{6})_(\d{3})\.fits$", re.IGNORECASE
)


def require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{label} must use UTC (+00:00 or Z)")
    return value.astimezone(timezone.utc)


def utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def frozen_record_id(observed: datetime, relative_path: str) -> str:
    identity = f"{utc_z(observed)}\0{relative_path}"
    return "radio-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def observed_utc(path: Path) -> datetime:
    match = STAMP.search(path.name)
    if match is None:
        raise ValueError(f"Cannot derive UTC record identity: {path.name}")
    year, month, day, clock, millis = match.groups()
    start = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
    return start.replace(
        hour=int(clock[:2]), minute=int(clock[2:4]), second=int(clock[4:]),
        microsecond=int(millis) * 1000,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path, start: datetime, end: datetime) -> dict:
    start = require_utc(start, label="start_utc")
    end = require_utc(end, label="end_utc")
    if end <= start:
        raise ValueError("end_utc must be later than start_utc")
    records = []
    record_ids: set[str] = set()
    for path in sorted(root.glob("*MHz/*/*.fits")):
        observed = observed_utc(path)
        if not start <= observed < end:
            continue
        relative = path.relative_to(root).as_posix()
        record_id = frozen_record_id(observed, relative)
        if record_id in record_ids:
            raise ValueError(f"Duplicate frozen record_id: {record_id}")
        record_ids.add(record_id)
        records.append(
            {
                "record_id": record_id,
                "observed_utc": utc_z(observed),
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not records:
        raise ValueError("UTC interval selected no FITS records")
    return {
        "schema": "solar-radio-frozen-collection-v1",
        "selection": {"start_utc": utc_z(start), "end_utc": utc_z(end)},
        "record_count": len(records),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-utc", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end-utc", type=datetime.fromisoformat, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / ".frozen-collection-v1.json").resolve()
    payload = build(root, args.start_utc, args.end_utc)
    print(json.dumps({"output": str(output), "record_count": payload["record_count"]}, indent=2))
    if not args.apply:
        print("preview only; rerun with --apply")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
