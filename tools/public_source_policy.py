"""Check Git blobs and distribution members against the public-source boundary.

This is a structural/content guard, not a replacement for human review or
Gitleaks. It never prints offending content, only paths and rule names.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from configparser import ConfigParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

BLOCKED_PARTS = {
    "local",
    "local-migration-backup",
    "outputs",
    "logs",
    "history",
    "legacy",
    "legacy_tests",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
}
BLOCKED_SUFFIXES = {
    ".avi",
    ".csv",
    ".db",
    ".fit",
    ".fits",
    ".gif",
    ".h5",
    ".hdf5",
    ".jpg",
    ".jpeg",
    ".json",
    ".jsonl",
    ".mkv",
    ".mov",
    ".mp4",
    ".nc",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".sqlite",
    ".tsv",
    ".webp",
    ".xls",
    ".xlsx",
    ".pyc",
    ".zip",
    ".whl",
    ".gz",
    ".yaml",
    ".yml",
}
ALLOWED_CONFIGS = {
    ".github/workflows/ci.yml",
    ".pre-commit-config.yaml",
    "Apps/configs/examples/paths.example.yaml",
    "Apps/environment.miniforge.yml",
    *(
        f"environment/locks/{target}/{name}"
        for target in ("linux-64-py314", "osx-arm64-py314")
        for name in ("lock-receipt.json", "pip-artifacts.json")
    ),
}
APP_ASSETS = {
    "frontends/image_viewer/templates/index.html",
    "frontends/image_viewer/static/style.css",
    "frontends/image_viewer/static/main.js",
    "frontends/radio/source_map/templates/index.html",
    "frontends/radio/source_map/static/app.js",
    "frontends/radio/source_map/static/style.css",
    "frontends/radio_bad_frame_review/templates/index.html",
    "frontends/radio_bad_frame_review/static/app.js",
    "frontends/radio_bad_frame_review/static/style.css",
    "frontends/workbench/templates/radio.html",
    "frontends/workbench/templates/index.html",
    "frontends/workbench/static/radio.js",
    "frontends/workbench/static/style.css",
    "frontends/workbench/static/radio_figure_composer.js",
    "frontends/workbench/static/main.js",
    "frontends/workbench/static/radio.css",
    "ui/theme_assets/state.js",
    "ui/theme_assets/theme.js",
    "ui/theme_assets/theme.css",
    "ui/media/mediabunny-MPL-2.0.txt",
    "ui/media/NOTICE.txt",
    "ui/media/native_path_dialog.js",
    "ui/media/browser_media.js",
    "ui/media/mediabunny-1.50.8.cjs",
    "frontends/app_v1/LICENSE.md",
}
PERSONAL_PATH = re.compile(
    r"(?:/(?:Users|home)/|\b[A-Z]:[\\/]+Users[\\/]+)"
    r"(?!<(?:user|username)>|%USERNAME%)[A-Za-z0-9_.-]+[\\/]",
    re.I,
)


def check_blob(name: str, data: bytes) -> list[str]:
    """Return rule names; never return the potentially private blob content."""
    path = PurePosixPath(name)
    reasons = []
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        reasons.append("unsafe path")
    if any(part.casefold() in BLOCKED_PARTS for part in path.parts):
        reasons.append("private/generated directory")
    if path.suffix.casefold() in BLOCKED_SUFFIXES and name not in ALLOWED_CONFIGS:
        reasons.append("data or unapproved configuration file")
    if path.name.casefold() in {"paths.local.yaml", "public_base.md", "sha256sums"}:
        reasons.append("private runtime record")
    if path.parts and re.fullmatch(r"20\d\d", path.parts[0]):
        reasons.append("observation directory")
    if b"\0" in data:
        reasons.append("binary content")
        return reasons
    text = data.decode("utf-8", errors="replace")
    if PERSONAL_PATH.search(text):
        reasons.append("personal absolute path")
    if path.suffix == ".ipynb":
        try:
            cells = json.loads(text)["cells"]
            if any(
                cell.get("outputs") or cell.get("execution_count") is not None
                for cell in cells
            ):
                reasons.append("executed notebook")
        except (ValueError, KeyError, TypeError):
            reasons.append("invalid notebook")
    return reasons


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if completed.returncode:
        raise RuntimeError("Git source inspection failed")
    return completed.stdout


def check_git(repo: Path, *, staged: bool = False, revision: str = "HEAD") -> list[str]:
    """Read index blobs for staged additions/changes, or a committed tree."""
    selected = None
    if staged:
        selected = set(
            _git(
                repo,
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMRTU",
                "--no-renames",
            ).split(b"\0")
        )
        records = _git(repo, "ls-files", "--stage", "-z")
    else:
        records = _git(repo, "ls-tree", "-r", "-z", revision)
    offenders = []
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        if selected is not None and raw_name not in selected:
            continue
        fields = metadata.split()
        mode = fields[0]
        oid = fields[1] if staged else fields[2]
        name = raw_name.decode("utf-8", errors="surrogateescape")
        if mode not in {b"100644", b"100755"} or (staged and fields[2] != b"0"):
            reasons = ["symlink, submodule or unresolved index entry"]
        else:
            reasons = check_blob(name, _git(repo, "cat-file", "blob", oid.decode()))
        if reasons:
            offenders.append(f"{name!r}: {', '.join(reasons)}")
    return offenders


def check_archive(archive: Path, project: str) -> list[str]:
    """Apply the same boundary to wheel/sdist content and check package assets."""
    prefix = "solar_apps/" if project == "Apps" else "solar_toolkit/"
    members = {}
    seen = set()

    def member_path(name):
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or any(":" in part for part in path.parts)
        ):
            raise ValueError("unsafe archive member")
        canonical = path.as_posix()
        if canonical in seen:
            raise ValueError("duplicate archive member")
        seen.add(canonical)
        return path

    if archive.suffix == ".whl":
        with ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = member_path(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("symlink in wheel")
                if not member.is_dir():
                    members[path.as_posix()] = bundle.read(member)
    else:
        roots = set()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                path = member_path(member.name)
                roots.add(path.parts[0])
                if len(roots) != 1:
                    raise ValueError("multiple source archive roots")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("non-regular source archive member")
                if len(path.parts) < 2:
                    raise ValueError("source archive files require a containing root")
                members["/".join(path.parts[1:])] = bundle.extractfile(member).read()
    offenders = []
    for name, data in members.items():
        reasons = check_blob(f"{project}/{name}", data)
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            reasons.append("unsafe archive path")
        if archive.suffix == ".whl" and not name.startswith(prefix):
            distribution = (
                "solarphysics_apps" if project == "Apps" else "solar_physics_toolkit"
            )
            if not (
                path.parts[0].startswith(distribution + "-")
                and path.parts[0].endswith(".dist-info")
            ):
                reasons.append("unexpected wheel namespace")
        if project == "Python" and name.endswith(".dist-info/entry_points.txt"):
            parser = ConfigParser()
            parser.read_string(data.decode("utf-8"))
            if parser.has_section("console_scripts"):
                reasons.append("library console entry point")
        if name.startswith(prefix):
            relative = name.removeprefix(prefix)
            if not relative.endswith(".py") and not (
                project == "Apps" and relative in APP_ASSETS
            ):
                reasons.append("unapproved package resource")
        if reasons:
            offenders.append(f"{name!r}: {', '.join(reasons)}")
    if not any(name.startswith(prefix) for name in members):
        offenders.append("package sources missing")
    if project == "Apps":
        missing = APP_ASSETS - {
            name.removeprefix(prefix) for name in members if name.startswith(prefix)
        }
        offenders.extend(f"missing asset: {name}" for name in sorted(missing))
        if not any(name.endswith("LICENSES/GPL-3.0-only.txt") for name in members):
            offenders.append("missing GPL license")
        if archive.suffix == ".whl":
            metadata = [
                data
                for name, data in members.items()
                if name.endswith(".dist-info/METADATA")
            ]
            if (
                len(metadata) != 1
                or b"License-Expression: MIT AND GPL-3.0-only AND MPL-2.0"
                not in metadata[0]
            ):
                offenders.append("incorrect application license expression")
    return offenders


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--revision", default="HEAD")
    mode.add_argument("--archive", type=Path)
    parser.add_argument("--project", choices=("Apps", "Python"), default="Python")
    args = parser.parse_args(argv)
    try:
        offenders = (
            check_archive(args.archive, args.project)
            if args.archive
            else check_git(args.repo, staged=args.staged, revision=args.revision)
        )
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(2, f"Public-source inspection failed: {exc}\n")
    for offender in offenders:
        print(offender)
    return int(bool(offenders))


if __name__ == "__main__":
    raise SystemExit(main())
