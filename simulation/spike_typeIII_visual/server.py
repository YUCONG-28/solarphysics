"""Local helpers for preparing CPU-only GridView/Slurm RMHD workflows.

This module deliberately contains no SSH, scheduler submission, or remote API
code.  It renders scripts for a human to inspect and paste into GridView.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

from .server_validation import validate_cpu_delivery

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PACKAGE_ROOT / "server" / "gridview"
PRIVATE = re.compile(
    rb"(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+|"
    rb"\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)
TIERS = {
    "report-lite": (
        "figures",
        "data/*.json",
        "data/*.csv",
        "data/*.npz",
        "SHA256SUMS.txt",
        "validation_report.json",
    ),
    "presentation": (
        "figures",
        "animations/*.gif",
        "animations/*.mp4",
        "*.pptx",
        "data/*.json",
        "data/*.csv",
        "data/*.npz",
        "SHA256SUMS.txt",
        "validation_report.json",
    ),
    "research-complete": (
        "figures",
        "animations",
        "data/*.json",
        "data/*.csv",
        "data/*.npz",
        "data/*.h5",
        "SHA256SUMS.txt",
        "validation_report.json",
    ),
}


def doctor() -> int:
    """Print a privacy-safe readiness report."""

    ffmpeg = shutil.which("ffmpeg") is not None
    if not ffmpeg and importlib.util.find_spec("imageio_ffmpeg"):
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
    checks = {
        "python": sys.version.split()[0],
        "numpy": bool(importlib.util.find_spec("numpy")),
        "h5py": bool(importlib.util.find_spec("h5py")),
        "torch": bool(importlib.util.find_spec("torch")),
        "ffmpeg": ffmpeg,
        "scheduler_job": bool(os.environ.get("SLURM_JOB_ID")),
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["numpy"] and checks["h5py"] else 2


def render_scripts(output_dir: Path) -> int:
    """Copy reviewed, account-neutral templates to a user-selected directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(TEMPLATE_ROOT.iterdir()):
        if source.is_file():
            target = output_dir / source.name
            shutil.copyfile(source, target)
            if target.suffix == ".sh":
                target.chmod(0o755)
    print(f"Rendered {len(list(output_dir.iterdir()))} files into {output_dir}")
    return 0


def validate(run_root: Path) -> int:
    """Run the existing scientific validator without scheduler interaction."""

    report = validate_cpu_delivery(run_root, write_report=True)
    passed = bool(report["passed"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def _tier_files(run_root: Path, tier: str) -> list[Path]:
    files: set[Path] = set()
    for pattern in TIERS[tier]:
        candidate = run_root / pattern
        if "*" not in pattern and candidate.is_dir():
            files.update(path for path in candidate.rglob("*") if path.is_file())
        elif "*" not in pattern and candidate.is_file():
            files.add(candidate)
        else:
            files.update(path for path in run_root.glob(pattern) if path.is_file())
    return sorted(files)


def package(run_root: Path, tier: str, output: Path) -> int:
    """Create a deterministic reviewed result archive for downloading."""

    files = _tier_files(run_root, tier)
    if not files:
        raise FileNotFoundError("No files matched the selected package tier.")
    for path in files:
        if (
            path.suffix.lower() in {".json", ".csv", ".txt", ".log", ".md"}
            and PRIVATE.search(path.read_bytes())
        ):
            raise ValueError(f"Privacy scan rejected {path.relative_to(run_root)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(run_root).as_posix()
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        payload = ("\n".join(checksums) + "\n").encode()
        info = tarfile.TarInfo("PACKAGE_SHA256SUMS.txt")
        info.size = len(payload)
        info.mtime = 0
        import io

        archive.addfile(info, io.BytesIO(payload))
    print(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    render = commands.add_parser("render-scripts")
    render.add_argument("--output-dir", type=Path, default=Path("server_scripts"))
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--run-root", type=Path, required=True)
    package_parser = commands.add_parser("package")
    package_parser.add_argument("--run-root", type=Path, required=True)
    package_parser.add_argument("--tier", choices=tuple(TIERS), default="report-lite")
    package_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "render-scripts":
        return render_scripts(args.output_dir)
    if args.command == "validate":
        return validate(args.run_root)
    return package(args.run_root, args.tier, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
