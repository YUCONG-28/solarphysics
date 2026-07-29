"""Standard-library verification for an extracted scientific delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    checksum_path = root / "SHA256SUMS.txt"
    errors: list[str] = []
    verified = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"Invalid checksum line: {line}")
            continue
        target = root / relative
        if not target.is_file():
            errors.append(f"Missing file: {relative}")
        elif _sha256(target) != expected:
            errors.append(f"Checksum mismatch: {relative}")
        else:
            verified += 1
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = int(manifest["file_count"])
        if declared != verified:
            errors.append(
                f"Manifest file_count={declared}, verified checksum files={verified}"
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid manifest.json: {exc}")
    return {
        "passed": not errors,
        "verified_files": verified,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args(argv)
    report = verify(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
