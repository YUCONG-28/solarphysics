"""Repository-level privacy guards for tracked files and attribution."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
MEDIA_AND_DATA_SUFFIXES = (
    ".avi",
    ".cdf",
    ".csv",
    ".db",
    ".docx",
    ".feather",
    ".fit",
    ".fits",
    ".fits.fz",
    ".fits.gz",
    ".fts",
    ".gif",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jp2",
    ".jpg",
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
    ".pickle",
    ".pkl",
    ".png",
    ".pptx",
    ".sqlite",
    ".sqlite3",
    ".tsv",
    ".webp",
    ".xls",
    ".xlsx",
)

# Every tracked media/data artifact in the Python and Apps source partitions
# must be individually reviewed. This is deliberately empty after the public
# repository privacy cleanup.
ALLOWED_MEDIA_AND_DATA_PATHS: frozenset[str] = frozenset()

PUBLIC_SOURCE_PREFIXES = ("Python/", "Apps/")
PRIVATE_REPOSITORY_PREFIXES = (
    "Local/",
    "Local-migration-backup/",
    "2023/",
    "2024/",
    "2025/",
    "2026/",
    "overview/",
)

SENSITIVE_BASENAME_GLOBS = (
    "*.cer",
    "*.crt",
    "*.jks",
    "*.kdbx",
    "*.keystore",
    "*.key",
    "*.p12",
    "*.pem",
    "*.pfx",
    "*credentials*.json",
    "auth*.json",
    "client_secret*.json",
    "cookies*.json",
    "cookies.txt",
    "id_ed25519*",
    "id_rsa*",
    "oauth*.json",
    "service-account*.json",
    "service_account*.json",
    "token*.json",
)
SENSITIVE_EXACT_BASENAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".yarnrc",
        ".yarnrc.yml",
        "credentials",
        "nuget.config",
        "pip.conf",
        "secrets.json",
    }
)
SENSITIVE_PATH_PREFIXES = (
    ".aws/",
    ".azure/",
    ".config/gcloud/",
    ".dvc/cache/",
    ".dvc/config.local",
    ".gem/credentials",
    ".kube/",
    ".mlflow/",
    ".oci/",
    ".wandb/",
    "lightning_logs/",
    "mlruns/",
    "ray_results/",
    "runs/",
    "wandb/",
)

WINDOWS_USER_PROFILE = re.compile(
    r"\b[A-Z]:[\\/]+Users[\\/]+"
    r"(?!(?:<user>|<username>|%USERNAME%)(?:[\\/]|$))"
    r"[^\\/\s`'\"]+",
    re.IGNORECASE,
)
PRIVATE_EMAIL = re.compile(
    r"\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@gmail\.com\b", re.IGNORECASE
)


def _tracked_paths() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available in this environment")

    tracked = [
        raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        for raw in result.stdout.split(b"\0")
        if raw
    ]
    return [
        path
        for path in tracked
        if path.startswith(PUBLIC_SOURCE_PREFIXES)
        if (REPO_ROOT / path).is_file() or (REPO_ROOT / path).is_symlink()
    ]


def _all_tracked_paths() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available in this environment")
    return [
        raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def _tracked_text(paths: list[str]):
    for path in paths:
        file_path = REPO_ROOT / path
        if file_path.is_symlink():
            yield path, str(file_path.readlink())
            continue
        raw = file_path.read_bytes()
        if b"\0" in raw:
            continue
        yield path, raw.decode("utf-8", errors="replace")


def test_git_does_not_track_unreviewed_generated_products():
    tracked = _tracked_paths()
    unexpected = [
        path
        for path in tracked
        if path.lower().endswith(MEDIA_AND_DATA_SUFFIXES)
        and path not in ALLOWED_MEDIA_AND_DATA_PATHS
    ]

    assert unexpected == []


def test_git_does_not_track_private_runtime_or_observation_trees():
    unexpected = [
        path
        for path in _all_tracked_paths()
        if path.startswith(PRIVATE_REPOSITORY_PREFIXES)
    ]

    assert unexpected == []


def test_git_does_not_track_sensitive_filenames():
    unexpected = []
    for path in _tracked_paths():
        normalized_path = path.lower()
        basename = Path(path).name.lower()
        if (
            basename in SENSITIVE_EXACT_BASENAMES
            or any(
                fnmatch.fnmatchcase(basename, pattern)
                for pattern in SENSITIVE_BASENAME_GLOBS
            )
            or normalized_path.startswith(SENSITIVE_PATH_PREFIXES)
        ):
            unexpected.append(path)

    assert unexpected == []


def test_tracked_text_does_not_expose_private_paths_or_email():
    research_root = "spike_" + "topping_type_III"
    runtime_marker = "codex-" + "runtimes"
    personal_marker = "211" + "29"
    unexpected = []

    for path, text in _tracked_text(_tracked_paths()):
        reasons = []
        if WINDOWS_USER_PROFILE.search(text):
            reasons.append("non-placeholder Windows user profile")
        if research_root.casefold() in text.casefold():
            reasons.append("private research root")
        if runtime_marker.casefold() in text.casefold():
            reasons.append("local Codex runtime path")
        if PRIVATE_EMAIL.search(text):
            reasons.append("private Gmail address")
        if personal_marker in text:
            reasons.append("personal author identifier")
        if reasons:
            unexpected.append(f"{path}: {', '.join(reasons)}")

    assert unexpected == []


def test_citation_keeps_public_academic_attribution():
    citation = (PYTHON_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert 'family-names: "Li"' in citation
    assert 'given-names: "Y."' in citation
    assert 'institution: "Shandong University"' in citation
    assert "https://github.com/YUCONG-28/solarphysics" in citation


POSIX_USER_PROFILE = re.compile(
    r"/(?:Users|home)/(?!<[^>]+>(?:/|$))[^/\s`\"']+",
    re.IGNORECASE,
)


def test_public_tree_has_no_research_record_partition():
    forbidden = ("Paper/", "tools/literature/", "Notes/", "Runtime/")
    assert not [path for path in _all_tracked_paths() if path.startswith(forbidden)]


def test_all_public_text_uses_machine_path_placeholders():
    paths = [path for path in _all_tracked_paths() if (REPO_ROOT / path).is_file()]
    unexpected = []
    for path, text in _tracked_text(paths):
        if POSIX_USER_PROFILE.search(text) or WINDOWS_USER_PROFILE.search(text):
            unexpected.append(path)
    assert unexpected == []


def test_public_notebooks_have_no_saved_outputs():
    import json

    for path in _all_tracked_paths():
        if not path.endswith(".ipynb") or not (REPO_ROOT / path).is_file():
            continue
        notebook = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None, path
                assert not cell.get("outputs"), path
