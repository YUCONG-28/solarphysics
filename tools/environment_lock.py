#!/usr/bin/env python3
"""Capture and validate platform-specific public research environments.

The lock separates two kinds of evidence deliberately:

* Conda packages are recorded as exact artifact URLs with SHA-256 fragments.
* PyPI packages are recorded as exact installed versions.  A version lock is
  replayable, but is not claimed to authenticate the wheel archive itself.

``capture`` is intentionally an explicit write operation.  It observes an
already-installed environment; it never installs, upgrades, or solves a
dependency.  ``check`` is entirely offline and validates both lock structure
and freshness against the tracked source specifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import Tag, sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ImportError as exc:  # pragma: no cover - exercised only outside the env
    raise SystemExit(
        "environment_lock.py requires the 'packaging' distribution; "
        "run it with the selected Miniforge environment"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = REPO_ROOT / "environment" / "locks"
SCHEMA = "solarphysics-environment-lock/v1"
LOCAL_DISTRIBUTIONS = {
    canonicalize_name("solar-physics-toolkit"),
    canonicalize_name("solarphysics-apps"),
}
SOURCE_PROFILES = {
    "Apps/pyproject.toml": ("dev",),
    "Python/pyproject.toml": ("dev", "quality-ml"),
}
MARKER_KEYS = (
    "implementation_name",
    "implementation_version",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_system",
    "python_full_version",
    "python_version",
    "sys_platform",
)
PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
HASHED_PIN_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+) " r"--hash=sha256:([0-9a-f]{64})$"
)
CONDA_ARTIFACT_PATTERN = re.compile(
    r"^https://conda\.anaconda\.org/conda-forge/"
    r"(?P<subdir>[a-z0-9_-]+)/[^#]+#(?P<sha256>[0-9a-f]{64})$"
)


class LockError(RuntimeError):
    """Raised when lock evidence is incomplete or inconsistent."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LockError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed.stdout


def _conda_run(conda: str, environment: str, *command: str) -> list[str]:
    return [conda, "run", "--no-capture-output", "-n", environment, *command]


def _stable_marker_environment() -> dict[str, str]:
    marker = default_environment()
    return {key: marker[key] for key in MARKER_KEYS}


def _validate_marker_target(
    platform_name: str, marker_environment: dict[str, str]
) -> None:
    expected = {
        "linux-64": ("linux", "posix", "Linux", {"x86_64", "AMD64"}),
        "linux-aarch64": ("linux", "posix", "Linux", {"aarch64", "arm64"}),
        "osx-arm64": ("darwin", "posix", "Darwin", {"arm64"}),
        "win-64": ("win32", "nt", "Windows", {"AMD64", "x86_64"}),
    }.get(platform_name)
    if expected is None:
        raise LockError(f"unsupported lock platform: {platform_name}")
    sys_platform, os_name, platform_system, machines = expected
    if (
        marker_environment.get("sys_platform") != sys_platform
        or marker_environment.get("os_name") != os_name
        or marker_environment.get("platform_system") != platform_system
        or marker_environment.get("platform_machine") not in machines
        or marker_environment.get("implementation_name") != "cpython"
        or marker_environment.get("platform_python_implementation") != "CPython"
    ):
        raise LockError(
            f"marker environment does not match Conda platform {platform_name}"
        )


def _tags_match_target(
    tags: frozenset[Tag], platform_name: str, python_full_version: str
) -> bool:
    major, minor, *_ = python_full_version.split(".")
    expected_python = f"cp{major}{minor}"
    platform_fragments = {
        "linux-64": ("x86_64",),
        "linux-aarch64": ("aarch64",),
        "osx-arm64": ("macosx", "arm64"),
        "win-64": ("win_amd64",),
    }.get(platform_name)
    if platform_fragments is None:
        return False
    for tag in tags:
        python_ok = tag.interpreter in {expected_python, "py3"}
        if not python_ok:
            continue
        if tag.platform == "any":
            return True
        if all(fragment in tag.platform for fragment in platform_fragments):
            return True
    return False


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise LockError(f"immutable receipt already exists with different bytes: {path}")
    _atomic_write(path, data)


def _source_hashes() -> dict[str, str]:
    paths = [
        "Apps/environment.miniforge.yml",
        "tools/environment_lock.py",
        *SOURCE_PROFILES,
    ]
    return {relative: _sha256_path(REPO_ROOT / relative) for relative in sorted(paths)}


def _target_from(platform_name: str, python_version: str) -> str:
    major, minor, *_ = python_version.split(".")
    return f"{platform_name}-py{major}{minor}"


def _conda_platform(explicit: str) -> str:
    for line in explicit.splitlines():
        if line.startswith("# platform: "):
            return line.removeprefix("# platform: ").strip()
    raise LockError("Conda explicit export has no '# platform:' header")


def _conda_artifacts(explicit: str, expected_platform: str) -> list[str]:
    lines = [line.strip() for line in explicit.splitlines()]
    if "@EXPLICIT" not in lines:
        raise LockError("Conda lock has no @EXPLICIT marker")
    artifacts = [line for line in lines if line and not line.startswith(("#", "@"))]
    if not artifacts:
        raise LockError("Conda lock contains no artifacts")
    for line in artifacts:
        match = CONDA_ARTIFACT_PATTERN.fullmatch(line)
        if not match:
            raise LockError(
                f"Conda artifact is not a conda-forge URL with SHA-256: {line}"
            )
        if match.group("subdir") not in {expected_platform, "noarch"}:
            raise LockError(
                f"Conda artifact subdir {match.group('subdir')!r} does not match "
                f"{expected_platform!r}"
            )
    return artifacts


def _freeze_to_pins(freeze: str, conda_distributions: set[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in freeze.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        is_editable = line.startswith("-e ")
        editable = line.removeprefix("-e ").strip()
        if "#egg=" in editable:
            name = editable.rsplit("#egg=", 1)[1].split("&", 1)[0]
            if canonicalize_name(name) in LOCAL_DISTRIBUTIONS:
                continue
        if is_editable:
            if editable.startswith("file://"):
                parsed = urlparse(editable)
                local_path = Path(unquote(parsed.path))
            elif "://" not in editable and not editable.startswith("git+"):
                local_path = Path(editable)
            else:
                local_path = None
            if local_path is not None and local_path.expanduser().resolve() in {
                (REPO_ROOT / "Apps").resolve(),
                (REPO_ROOT / "Python").resolve(),
            }:
                continue
            raise LockError(f"unrecognized editable distribution in pip freeze: {line}")
        match = PIN_PATTERN.fullmatch(line)
        if not match:
            name_part = line.split(" @ ", 1)[0]
            normalized = canonicalize_name(name_part)
            if normalized in conda_distributions or normalized in LOCAL_DISTRIBUTIONS:
                continue
            raise LockError(
                f"pip freeze entry is not an exact public version pin: {line}"
            )
        raw_name, version = match.groups()
        name = canonicalize_name(raw_name)
        if name in conda_distributions or name in LOCAL_DISTRIBUTIONS:
            continue
        if name in pins and pins[name] != version:
            raise LockError(
                f"duplicate pip distribution with different versions: {name}"
            )
        pins[name] = version
    if not pins:
        raise LockError("pip version lock would be empty")
    return dict(sorted(pins.items()))


def _validate_editable_projects(
    editable_json: str, *, require_current_checkout: bool
) -> dict[str, str]:
    payload = json.loads(editable_json)
    if not isinstance(payload, list):
        raise LockError("pip editable inventory is not a JSON list")
    expected_locations = {
        canonicalize_name("solar-physics-toolkit"): (REPO_ROOT / "Python").resolve(),
        canonicalize_name("solarphysics-apps"): (REPO_ROOT / "Apps").resolve(),
    }
    observed: dict[str, Path] = {}
    for record in payload:
        if not isinstance(record, dict):
            raise LockError("pip editable inventory has an invalid record")
        name = canonicalize_name(str(record.get("name", "")))
        location = record.get("editable_project_location")
        if not name or not isinstance(location, str) or not location:
            raise LockError("pip editable inventory has no stable name/location")
        if name in observed:
            raise LockError(f"duplicate editable distribution: {name}")
        observed[name] = Path(location).expanduser().resolve()
    if set(observed) != set(expected_locations):
        raise LockError(
            "editable set must contain exactly solar-physics-toolkit and "
            f"solarphysics-apps: observed={sorted(observed)}"
        )
    if require_current_checkout:
        wrong = {
            name: str(observed[name])
            for name in observed
            if observed[name] != expected_locations[name]
        }
        if wrong:
            raise LockError(
                f"editable projects do not point to this reviewed checkout: {wrong}"
            )
    return {name: str(location) for name, location in observed.items()}


def _render_pins(pins: dict[str, str], target: str) -> bytes:
    lines = [
        "# Generated by tools/environment_lock.py from an installed environment.",
        f"# target: {target}",
        "# Exact versions; wheel archives are not authenticated by this file.",
        "# Install with --no-deps --only-binary=:all: and finish with pip check.",
        "",
    ]
    lines.extend(f"{name}=={version}" for name, version in pins.items())
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_pins(content: str, expected_target: str) -> dict[str, str]:
    if f"# target: {expected_target}" not in content.splitlines():
        raise LockError(f"pip lock target header does not match {expected_target}")
    pins: dict[str, str] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        if not match:
            raise LockError(f"pip lock entry is not exactly pinned: {line}")
        raw_name, version = match.groups()
        name = canonicalize_name(raw_name)
        if name in pins:
            raise LockError(f"duplicate pip lock entry: {name}")
        pins[name] = version
    if not pins:
        raise LockError("pip lock contains no pins")
    if list(pins) != sorted(pins):
        raise LockError("pip lock entries are not sorted by canonical name")
    return pins


def _source_requirements() -> list[Requirement]:
    requirements: list[Requirement] = []
    for relative, extras in SOURCE_PROFILES.items():
        payload = tomllib.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        project = payload["project"]
        requirements.extend(
            Requirement(value) for value in project.get("dependencies", [])
        )
        optional = project.get("optional-dependencies", {})
        for extra in extras:
            requirements.extend(Requirement(value) for value in optional[extra])
    return requirements


def _validate_source_constraints(
    pins: dict[str, str], marker_environment: dict[str, str]
) -> None:
    missing: list[str] = []
    incompatible: list[str] = []
    python_version = Version(marker_environment["python_full_version"])
    for relative in SOURCE_PROFILES:
        payload = tomllib.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        requires_python = SpecifierSet(payload["project"]["requires-python"])
        if python_version not in requires_python:
            incompatible.append(
                f"{relative} requires Python {requires_python} "
                f"(target has {python_version})"
            )
    for requirement in _source_requirements():
        if requirement.marker and not requirement.marker.evaluate(
            environment=marker_environment
        ):
            continue
        name = canonicalize_name(requirement.name)
        version = pins.get(name)
        if version is None:
            missing.append(requirement.name)
        elif requirement.specifier and Version(version) not in requirement.specifier:
            incompatible.append(f"{requirement} (locked {version})")
    if missing:
        raise LockError(
            f"source requirements absent from pip lock: {sorted(set(missing))}"
        )
    if incompatible:
        raise LockError(
            f"source constraints reject locked versions: {sorted(set(incompatible))}"
        )


def _environment_lock_sha(receipt: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "environment_lock_sha256"
    }
    return _sha256_bytes(_canonical_json(payload))


def _capture(args: argparse.Namespace) -> int:
    conda = str(Path(args.conda).expanduser())
    environment = args.environment
    explicit = _run([conda, "list", "--explicit", "--sha256", "-n", environment])
    platform_name = _conda_platform(explicit)
    _conda_artifacts(explicit, platform_name)

    runtime = json.loads(
        _run(
            _conda_run(
                conda,
                environment,
                "python",
                "-c",
                (
                    "import json, platform; "
                    "from packaging.markers import default_environment; "
                    "marker = default_environment(); "
                    "keys = ('implementation_name', 'implementation_version', "
                    "'os_name', 'platform_machine', "
                    "'platform_python_implementation', 'platform_system', "
                    "'python_full_version', 'python_version', 'sys_platform'); "
                    "print(json.dumps({'python_full_version': platform.python_version(), "
                    "'implementation': platform.python_implementation(), "
                    "'marker_environment': {key: marker[key] for key in keys}}))"
                ),
            )
        )
    )
    if runtime["implementation"] != "CPython":
        raise LockError("only CPython environments are supported")
    _validate_marker_target(platform_name, runtime["marker_environment"])
    target = _target_from(platform_name, runtime["python_full_version"])
    if args.target and args.target != target:
        raise LockError(
            f"requested target {args.target!r} does not match runtime {target!r}"
        )
    if _stable_marker_environment() != runtime["marker_environment"]:
        raise LockError("capture process does not match the selected environment")

    conda_records = json.loads(_run([conda, "list", "--json", "-n", environment]))
    conda_distributions = {
        canonicalize_name(record["name"])
        for record in conda_records
        if record.get("channel") != "pypi"
    }
    installed_versions = {
        canonicalize_name(record["name"]): str(record["version"])
        for record in conda_records
    }

    _run(_conda_run(conda, environment, "python", "-m", "pip", "check"))
    editable_json = _run(
        _conda_run(
            conda,
            environment,
            "python",
            "-m",
            "pip",
            "list",
            "--editable",
            "--format=json",
        )
    )
    _validate_editable_projects(editable_json, require_current_checkout=True)
    freeze = _run(
        _conda_run(conda, environment, "python", "-m", "pip", "freeze", "--all")
    )
    pins = _freeze_to_pins(freeze, conda_distributions)
    constraint_versions = {**installed_versions, **pins}
    _validate_source_constraints(constraint_versions, runtime["marker_environment"])

    explicit_bytes = explicit.encode("utf-8")
    pins_bytes = _render_pins(pins, target)
    source_hashes = _source_hashes()
    conda_sha = _sha256_bytes(explicit_bytes)
    pip_sha = _sha256_bytes(pins_bytes)
    receipt = {
        "schema": SCHEMA,
        "target": target,
        "platform": platform_name,
        "python_full_version": runtime["python_full_version"],
        "implementation": runtime["implementation"],
        "marker_environment": runtime["marker_environment"],
        "profiles": {
            relative: list(extras)
            for relative, extras in sorted(SOURCE_PROFILES.items())
        },
        "source_files": source_hashes,
        "lock_files": {
            "conda-explicit.txt": conda_sha,
            "pip-pins.txt": pip_sha,
        },
        "conda_artifacts_exact": True,
        "conda_artifacts_sha256": True,
        "pip_versions_exact": True,
        "pip_artifacts_sha256": False,
        "capture_mode": "read-only-observation-of-installed-environment",
        "local_source_install": "editable-no-deps-no-build-isolation",
    }
    receipt["environment_lock_sha256"] = _environment_lock_sha(receipt)
    receipt_bytes = _canonical_json(receipt)
    destination = LOCK_ROOT / target

    print(f"target={target}")
    print(f"environment_lock_sha256={receipt['environment_lock_sha256']}")
    print(f"conda_artifacts={len(_conda_artifacts(explicit, platform_name))}")
    print(f"pip_pins={len(pins)}")
    print("pip_artifacts_sha256=false")
    if not args.apply:
        print(
            f"preview only; add --apply to write {destination.relative_to(REPO_ROOT)}"
        )
        return 0

    _atomic_write(destination / "conda-explicit.txt", explicit_bytes)
    _atomic_write(destination / "pip-pins.txt", pins_bytes)
    _atomic_write(destination / "lock-receipt.json", receipt_bytes)
    print(f"wrote {destination.relative_to(REPO_ROOT)}")
    return 0


def _check_target(target: str) -> dict[str, Any]:
    directory = LOCK_ROOT / target
    receipt_path = directory / "lock-receipt.json"
    if not receipt_path.is_file():
        raise LockError(f"missing receipt: {receipt_path.relative_to(REPO_ROOT)}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise LockError(f"unsupported receipt schema for {target}")
    if receipt.get("target") != target:
        raise LockError(f"receipt target does not match directory {target}")
    if not isinstance(receipt.get("pip_artifacts_sha256"), bool):
        raise LockError("pip artifact-hash status must be an explicit boolean")
    for field in (
        "conda_artifacts_exact",
        "conda_artifacts_sha256",
        "pip_versions_exact",
    ):
        if receipt.get(field) is not True:
            raise LockError(f"required lock property is not true for {target}: {field}")
    expected_profiles = {
        relative: list(extras) for relative, extras in sorted(SOURCE_PROFILES.items())
    }
    if receipt.get("profiles") != expected_profiles:
        raise LockError(f"dependency profiles disagree with policy for {target}")

    current_sources = _source_hashes()
    if receipt.get("source_files") != current_sources:
        raise LockError(f"{target} lock is stale relative to source environment files")

    lock_files = receipt.get("lock_files")
    required_lock_files = {"conda-explicit.txt", "pip-pins.txt"}
    sealed_lock_files = required_lock_files | {
        "pip-artifacts.json",
        "pip-hashed.txt",
    }
    lock_names = frozenset(lock_files) if isinstance(lock_files, dict) else frozenset()
    if not isinstance(lock_files, dict) or lock_names not in {
        frozenset(required_lock_files),
        frozenset(sealed_lock_files),
    }:
        raise LockError(f"invalid lock file map for {target}")
    for name, expected in lock_files.items():
        path = directory / name
        if not path.is_file():
            raise LockError(f"missing lock file: {path.relative_to(REPO_ROOT)}")
        actual = _sha256_path(path)
        if actual != expected:
            raise LockError(f"lock checksum mismatch for {path.relative_to(REPO_ROOT)}")

    explicit = (directory / "conda-explicit.txt").read_text(encoding="utf-8")
    platform_name = _conda_platform(explicit)
    if platform_name != receipt.get("platform"):
        raise LockError(f"Conda platform and receipt disagree for {target}")
    python_full_version = str(receipt.get("python_full_version", ""))
    if _target_from(platform_name, python_full_version) != target:
        raise LockError(
            f"target label does not match platform/Python receipt for {target}"
        )
    artifacts = _conda_artifacts(explicit, platform_name)
    python_artifacts = [
        line
        for line in artifacts
        if Path(line.split("#", 1)[0]).name.startswith("python-")
    ]
    if len(python_artifacts) != 1:
        raise LockError(f"expected one CPython artifact for {target}")
    conda_python_match = re.match(
        r"python-([0-9][^-]*)-",
        Path(python_artifacts[0].split("#", 1)[0]).name,
    )
    if not conda_python_match or Version(conda_python_match.group(1)) != Version(
        python_full_version
    ):
        raise LockError(f"Conda Python artifact and receipt disagree for {target}")

    marker_environment = receipt.get("marker_environment")
    if not isinstance(marker_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in marker_environment.items()
    ):
        raise LockError(f"invalid marker environment for {target}")
    if marker_environment.get("python_full_version") != python_full_version:
        raise LockError(f"marker environment Python and receipt disagree for {target}")
    _validate_marker_target(platform_name, marker_environment)

    pins = _parse_pins((directory / "pip-pins.txt").read_text(encoding="utf-8"), target)
    _validate_source_constraints(pins, marker_environment)
    sealed = set(lock_files) == sealed_lock_files
    if receipt.get("pip_artifacts_sha256") is not sealed:
        raise LockError(f"pip artifact-hash flag and lock files disagree for {target}")
    if sealed:
        _validate_pip_artifacts(directory, target, pins)
    expected_lock_sha = _environment_lock_sha(receipt)
    if receipt.get("environment_lock_sha256") != expected_lock_sha:
        raise LockError(f"combined environment lock SHA mismatch for {target}")
    return receipt


def _validate_pip_artifacts(
    directory: Path, target: str, pins: dict[str, str]
) -> dict[str, dict[str, Any]]:
    payload = json.loads((directory / "pip-artifacts.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "solarphysics-pip-artifacts/v1":
        raise LockError(f"unsupported pip artifact schema for {target}")
    if payload.get("target") != target:
        raise LockError(f"pip artifact target does not match {target}")
    artifact_platform = str(payload.get("platform", ""))
    artifact_python = str(payload.get("python_full_version", ""))
    if _target_from(artifact_platform, artifact_python) != target:
        raise LockError(f"pip artifact platform/Python does not match {target}")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise LockError(f"pip artifact list is invalid for {target}")
    artifacts: dict[str, dict[str, Any]] = {}
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            raise LockError(f"pip artifact record is invalid for {target}")
        name = canonicalize_name(str(artifact.get("name", "")))
        version = str(artifact.get("version", ""))
        filename = str(artifact.get("filename", ""))
        sha256 = str(artifact.get("sha256", ""))
        size = artifact.get("size")
        if not name or name in artifacts:
            raise LockError(
                f"duplicate or empty pip artifact name for {target}: {name}"
            )
        if filename != Path(filename).name or not filename.endswith(".whl"):
            raise LockError(f"unsafe pip artifact filename for {target}: {filename}")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise LockError(f"invalid pip artifact SHA-256 for {target}: {filename}")
        if not isinstance(size, int) or size <= 0:
            raise LockError(f"invalid pip artifact size for {target}: {filename}")
        if pins.get(name) != version:
            raise LockError(
                f"pip artifact does not match version pin: {name}=={version}"
            )
        parsed_name, parsed_version, _build, tags = parse_wheel_filename(filename)
        if canonicalize_name(parsed_name) != name or parsed_version != Version(version):
            raise LockError(f"wheel filename and artifact record disagree: {filename}")
        if not _tags_match_target(tags, artifact_platform, artifact_python):
            raise LockError(f"wheel filename is incompatible with target: {filename}")
        artifacts[name] = artifact
    if set(artifacts) != set(pins):
        raise LockError(
            f"pip artifacts do not cover the exact version-pin set for {target}"
        )

    hashed: dict[str, tuple[str, str]] = {}
    for raw in (directory / "pip-hashed.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = HASHED_PIN_PATTERN.fullmatch(line)
        if not match:
            raise LockError(f"invalid pip hashed-lock entry: {line}")
        raw_name, version, sha256 = match.groups()
        name = canonicalize_name(raw_name)
        if name in hashed:
            raise LockError(f"duplicate pip hashed-lock entry: {name}")
        hashed[name] = (version, sha256)
    expected_hashed = {
        name: (str(artifact["version"]), str(artifact["sha256"]))
        for name, artifact in artifacts.items()
    }
    if hashed != expected_hashed:
        raise LockError(f"pip hashed lock and artifact receipt disagree for {target}")
    return artifacts


def _seal_pip(args: argparse.Namespace) -> int:
    receipt = _check_target(args.target)
    if receipt["pip_artifacts_sha256"]:
        raise LockError(f"{args.target} already has a sealed pip artifact lock")
    directory = LOCK_ROOT / args.target
    pins = _parse_pins(
        (directory / "pip-pins.txt").read_text(encoding="utf-8"), args.target
    )
    wheelhouse = Path(args.wheelhouse).expanduser().resolve()
    if not wheelhouse.is_dir():
        raise LockError(f"wheelhouse is not a directory: {wheelhouse}")
    conda = str(Path(args.conda).expanduser())
    runtime_marker = json.loads(
        _run(
            _conda_run(
                conda,
                args.environment,
                "python",
                "-c",
                (
                    "import json; from packaging.markers import default_environment; "
                    f"keys = {MARKER_KEYS!r}; marker = default_environment(); "
                    "print(json.dumps({key: marker[key] for key in keys}))"
                ),
            )
        )
    )
    _validate_marker_target(receipt["platform"], runtime_marker)
    if runtime_marker != receipt["marker_environment"]:
        raise LockError(
            "selected sealing environment does not exactly match the receipt target"
        )
    if _stable_marker_environment() != runtime_marker:
        raise LockError(
            "seal-pip must itself run inside the selected target environment"
        )
    compatible_tags = set(sys_tags())
    artifacts: dict[str, dict[str, Any]] = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        raw_name, parsed_version, _build, tags = parse_wheel_filename(wheel.name)
        name = canonicalize_name(raw_name)
        if name not in pins:
            raise LockError(
                f"wheelhouse contains an unpinned distribution: {wheel.name}"
            )
        if parsed_version != Version(pins[name]):
            raise LockError(f"wheel version does not match the pin: {wheel.name}")
        if not compatible_tags.intersection(tags):
            raise LockError(
                f"wheel is incompatible with the sealing runtime: {wheel.name}"
            )
        if name in artifacts:
            raise LockError(f"wheelhouse contains multiple wheels for {name}")
        artifacts[name] = {
            "filename": wheel.name,
            "name": name,
            "sha256": _sha256_path(wheel),
            "size": wheel.stat().st_size,
            "version": pins[name],
        }
    if set(artifacts) != set(pins):
        missing = sorted(set(pins) - set(artifacts))
        raise LockError(f"wheelhouse does not cover every pip pin: missing={missing}")

    artifact_bytes = _canonical_json(
        {
            "schema": "solarphysics-pip-artifacts/v1",
            "target": args.target,
            "platform": receipt["platform"],
            "python_full_version": receipt["python_full_version"],
            "artifacts": [artifacts[name] for name in sorted(artifacts)],
        }
    )
    hashed_lines = [
        "# Generated by tools/environment_lock.py from a complete target wheelhouse.",
        f"# target: {args.target}",
        "# Every accepted wheel archive is authenticated by SHA-256.",
        "",
    ]
    hashed_lines.extend(
        f"{name}=={artifacts[name]['version']} "
        f"--hash=sha256:{artifacts[name]['sha256']}"
        for name in sorted(artifacts)
    )
    hashed_bytes = ("\n".join(hashed_lines) + "\n").encode("utf-8")
    lock_files = dict(receipt["lock_files"])
    lock_files.update(
        {
            "pip-artifacts.json": _sha256_bytes(artifact_bytes),
            "pip-hashed.txt": _sha256_bytes(hashed_bytes),
        }
    )
    receipt["lock_files"] = dict(sorted(lock_files.items()))
    receipt["pip_artifacts_sha256"] = True
    receipt["environment_lock_sha256"] = _environment_lock_sha(receipt)
    print(f"target={args.target}")
    print(f"pip_artifacts={len(artifacts)}")
    print(f"environment_lock_sha256={receipt['environment_lock_sha256']}")
    if not args.apply:
        print("preview only; add --apply to seal the committed pip lock")
        return 0
    _atomic_write(directory / "pip-artifacts.json", artifact_bytes)
    _atomic_write(directory / "pip-hashed.txt", hashed_bytes)
    _atomic_write(directory / "lock-receipt.json", _canonical_json(receipt))
    _check_target(args.target)
    print(f"sealed pip artifacts for {args.target}")
    return 0


def _check(args: argparse.Namespace) -> int:
    if args.target:
        targets = [args.target]
    else:
        targets = (
            sorted(path.name for path in LOCK_ROOT.iterdir() if path.is_dir())
            if LOCK_ROOT.is_dir()
            else []
        )
    if not targets:
        raise LockError("no committed platform locks found")
    for target in targets:
        receipt = _check_target(target)
        if args.require_artifact_hashes and not receipt["pip_artifacts_sha256"]:
            raise LockError(
                f"{target} has exact pip versions but no wheel SHA-256 lock"
            )
        wheel_status = "sealed" if receipt["pip_artifacts_sha256"] else "unavailable"
        print(
            f"ok {target}: {receipt['environment_lock_sha256']} "
            f"(pip wheel hashes: {wheel_status})"
        )
    return 0


def _verify_runtime(args: argparse.Namespace) -> int:
    receipt = _check_target(args.target)
    if not receipt["pip_artifacts_sha256"]:
        raise LockError(f"{args.target} is a version snapshot, not a sealed artifact lock")
    conda = str(Path(args.conda).expanduser())
    explicit = _run([conda, "list", "--explicit", "--sha256", "-n", args.environment])
    expected_explicit = (LOCK_ROOT / args.target / "conda-explicit.txt").read_text(
        encoding="utf-8"
    )
    actual_artifacts = _conda_artifacts(explicit, receipt["platform"])
    expected_artifacts = _conda_artifacts(expected_explicit, receipt["platform"])
    if actual_artifacts != expected_artifacts:
        raise LockError("runtime Conda artifacts do not match the committed lock")

    conda_records = json.loads(_run([conda, "list", "--json", "-n", args.environment]))
    conda_distributions = {
        canonicalize_name(record["name"])
        for record in conda_records
        if record.get("channel") != "pypi"
    }
    editable_json = _run(
        _conda_run(
            conda,
            args.environment,
            "python",
            "-m",
            "pip",
            "list",
            "--editable",
            "--format=json",
        )
    )
    editable_projects = _validate_editable_projects(
        editable_json, require_current_checkout=True
    )
    freeze = _run(
        _conda_run(conda, args.environment, "python", "-m", "pip", "freeze", "--all")
    )
    actual_pins = _freeze_to_pins(freeze, conda_distributions)
    expected_pins = _parse_pins(
        (LOCK_ROOT / args.target / "pip-pins.txt").read_text(encoding="utf-8"),
        args.target,
    )
    if actual_pins != expected_pins:
        missing = sorted(set(expected_pins) - set(actual_pins))
        unexpected = sorted(set(actual_pins) - set(expected_pins))
        changed = sorted(
            name
            for name in set(actual_pins) & set(expected_pins)
            if actual_pins[name] != expected_pins[name]
        )
        raise LockError(
            f"runtime pip pins differ: missing={missing}, unexpected={unexpected}, "
            f"changed={changed}"
        )
    _run(_conda_run(conda, args.environment, "python", "-m", "pip", "check"))
    result = {
        "schema": "solarphysics-environment-replay/v1",
        "target": args.target,
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment_lock_sha256": receipt["environment_lock_sha256"],
        "source_files": _source_hashes(),
        "editable_projects": editable_projects,
        "conda_artifacts_match": True,
        "pip_versions_match": True,
        "pip_artifact_hashes_sealed": True,
        "installed_pip_archive_sha256_reconstructed": False,
        "pip_check_passed": True,
        "limitation": (
            "pip cannot reconstruct an installed wheel archive hash from site-packages; "
            "installation must use pip --require-hashes and the sealed lock"
        ),
    }
    replay_sha = _sha256_bytes(_canonical_json(result))
    print(
        f"runtime metadata matches {args.target}: "
        f"{receipt['environment_lock_sha256']}"
    )
    print(f"replay_receipt_sha256={replay_sha}")
    if args.receipt:
        if not args.apply:
            print(f"preview only; add --apply to write {args.receipt}")
            return 0
        receipt_path = Path(args.receipt).expanduser().resolve()
        _write_once(receipt_path, _canonical_json(result))
        _write_once(
            receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
            f"{replay_sha}  {receipt_path.name}\n".encode("utf-8"),
        )
        print(f"wrote replay receipt {receipt_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate committed locks offline")
    check.add_argument("--target")
    check.add_argument("--require-artifact-hashes", action="store_true")
    check.set_defaults(handler=_check)

    capture = subparsers.add_parser(
        "capture", help="observe an installed environment and render its lock"
    )
    capture.add_argument("--conda", required=True)
    capture.add_argument("--environment", required=True)
    capture.add_argument("--target")
    capture.add_argument("--apply", action="store_true")
    capture.set_defaults(handler=_capture)

    seal = subparsers.add_parser(
        "seal-pip", help="hash a complete target wheelhouse and seal a version lock"
    )
    seal.add_argument("--target", required=True)
    seal.add_argument("--wheelhouse", required=True)
    seal.add_argument("--conda", required=True)
    seal.add_argument("--environment", required=True)
    seal.add_argument("--apply", action="store_true")
    seal.set_defaults(handler=_seal_pip)

    verify = subparsers.add_parser(
        "verify-runtime", help="compare an installed environment with a lock"
    )
    verify.add_argument("--conda", required=True)
    verify.add_argument("--environment", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument("--receipt")
    verify.add_argument("--apply", action="store_true")
    verify.set_defaults(handler=_verify_runtime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (LockError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"environment lock error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
