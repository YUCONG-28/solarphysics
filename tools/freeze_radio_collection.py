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
    r"_(\d{4})(\d)(\d{2})_(\d{6})_(\d{3})\.fits$", re.IGNORECASE
)


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
    records = []
    for path in sorted(root.glob("*MHz/*/*.fits")):
        observed = observed_utc(path)
        if not start <= observed < end:
            continue
        relative = path.relative_to(root).as_posix()
        records.append({
            "record_id": "radio-" + hashlib.sha256(relative.encode()).hexdigest()[:24],
            "observed_utc": observed.isoformat().replace("+00:00", "Z"),
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    if not records:
        raise ValueError("UTC interval selected no FITS records")
    return {
        "schema": "solar-radio-frozen-collection-v1",
        "selection": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
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
