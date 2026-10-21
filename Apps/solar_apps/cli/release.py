"""Prepare version changes for review, then publish an explicitly merged commit.

All commands are read-only unless ``--execute`` is supplied. Preparation edits
only version/changelog files on a clean feature branch. Publication verifies
an immutable merged commit and its checks before creating a tag or Release;
it never commits to or pushes main. Matching remote objects make retries safe.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout

TOOLKIT_VERSION = "Python/solar_toolkit/_version.py"
APPS_VERSION = "Apps/solar_apps/_version.py"
CHANGELOG = "Python/CHANGELOG.md"

_VERSION_BUMPS = ("patch", "minor", "major")


def _repo_root() -> Path:
    return RuntimeLayout.discover().repo_root


def _read_version(path: Path) -> str:
    """Read the ``__version__`` assignment from a version module without import."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"no __version__ string found in {path}")


def _write_version(path: Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated = []
    replaced = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("__version__") and "=" in stripped:
            updated.append(f'__version__ = "{version}"\n')
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        raise ValueError(f"no __version__ assignment found in {path}")
    path.write_text("".join(updated), encoding="utf-8", newline="\n")


def _split_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"version {version!r} is not a numeric X.Y.Z triple")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _bump_version(version: str, bump: str) -> str:
    major, minor, patch = _split_version(version)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level {bump!r}")


def _current_version(repo_root: Path) -> str:
    toolkit = _read_version(repo_root / TOOLKIT_VERSION)
    apps = _read_version(repo_root / APPS_VERSION)
    if toolkit != apps:
        raise RuntimeError(
            f"version mismatch: toolkit={toolkit!r} apps={apps!r}; "
            "run tools release check for details"
        )
    return toolkit


def _changelog_release_block(repo_root: Path, new_version: str) -> tuple[str, str]:
    """Extract the Unreleased block and render the rewritten changelog.

    Returns ``(rewritten_text, unreleased_block)`` where the block is the body
    that was listed under ``## Unreleased``, without the heading. The released
    section is moved to the top (newest first) and a fresh ``## Unreleased``
    placeholder is inserted above it.
    """

    path = repo_root / CHANGELOG
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    heading_index = next(
        (i for i, line in enumerate(lines) if line.strip() == "## Unreleased"),
        None,
    )
    if heading_index is None:
        raise RuntimeError(f"{CHANGELOG} has no '## Unreleased' section")

    end_index = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if lines[index].strip().startswith("## "):
            end_index = index
            break

    body = [line for line in lines[heading_index + 1 : end_index]]
    while body and not body[-1].strip():
        body.pop()
    while body and not body[0].strip():
        body.pop(0)

    title_index = next(
        i for i, line in enumerate(lines) if line.strip() == "# Changelog"
    )
    released = f"## {new_version} - {date.today().isoformat()}"
    # Drop the old Unreleased heading and body, then insert the placeholder and
    # the released section right after the title.
    kept = [*lines[:heading_index], *lines[end_index:]]
    kept[title_index + 1 : title_index + 1] = [
        "",
        "## Unreleased",
        "",
        released,
        *body,
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + "\n", "\n".join(body).strip()


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _git(repo_root: Path, *args: str, dry_run: bool = False) -> str | None:
    print("$ git " + " ".join(args))
    if dry_run:
        return None
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


REQUIRED_CHECKS = frozenset(
    {
        "Gitleaks (complete history)",
        "Python public package (3.10)",
        "Python public package (3.11)",
        "Python public package (3.14)",
        "Apps (Windows Miniforge)",
        "Apps (macOS arm64 Miniforge)",
    }
)


def _check_clean(repo_root: Path, *, require_main: bool = True) -> list[str]:
    """Inspect local refs only; never fetch during a check or preview."""
    problems = []
    branch = (_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    if branch == "HEAD" or not branch:
        problems.append("detached HEAD; use a named branch for preparation")
    elif require_main and branch != "main":
        problems.append(f"not on main branch (current: {branch!r})")
    status = _git(repo_root, "status", "--short")
    if status and status.strip():
        problems.append("working tree is not clean:\n" + status.strip())
    if branch == "main":
        ahead = _git(repo_root, "rev-list", "--count", "origin/main..HEAD")
        behind = _git(repo_root, "rev-list", "--count", "HEAD..origin/main")
        if (ahead or "0").strip() != "0":
            problems.append("local main has unpushed commits")
        if (behind or "0").strip() != "0":
            problems.append("local main is behind origin/main; run: git pull --ff-only")
    return problems


def _strict_locks(repo_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/environment_lock.py"),
            "check",
            "--require-artifact-hashes",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "strict environment-lock validation failed; run tools/environment_lock.py check --require-artifact-hashes"
        )


def _optional_git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode in (1, 128) and args[0] == "rev-parse":
        return None
    raise RuntimeError(f"git {args[0]} inspection failed ({result.returncode})")


def _tag_targets(repo_root: Path, tag: str) -> tuple[str | None, str | None]:
    local = _optional_git(
        repo_root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"
    )
    lines = (
        _git(
            repo_root,
            "ls-remote",
            "--tags",
            "origin",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        )
        or ""
    )
    refs = {}
    for line in lines.splitlines():
        oid, ref = line.split()
        refs[ref] = oid
    remote = refs.get(f"refs/tags/{tag}^{{}}", refs.get(f"refs/tags/{tag}"))
    return local, remote


def _print_check(repo_root: Path) -> int:
    version = _current_version(repo_root)
    problems = _check_clean(repo_root, require_main=False)
    _strict_locks(repo_root)
    print(f"current version: {version}")
    print("strict environment locks: passed")
    print("local branch/tree state (cached refs): " + ("FAILED" if problems else "ok"))
    for problem in problems:
        print(f"  - {problem}")
    return int(bool(problems))


def _prepare_release(repo_root: Path, *, bump: str, note: str, execute: bool) -> int:
    if not note.strip():
        raise ValueError("--note must not be empty")
    problems = _check_clean(repo_root, require_main=False)
    if problems:
        raise RuntimeError("preparation preconditions failed: " + "; ".join(problems))
    branch = (_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    if execute and branch == "main":
        raise RuntimeError(
            "prepare --execute requires a clean feature branch; create a version PR branch first"
        )
    current = _current_version(repo_root)
    version = _bump_version(current, bump)
    rewritten, body = _changelog_release_block(repo_root, version)
    if not body:
        heading = f"## {version} - {date.today().isoformat()}"
        rewritten = rewritten.replace(heading, heading + "\n\n- " + note.strip(), 1)
    _strict_locks(repo_root)
    if any(_tag_targets(repo_root, f"v{version}")):
        raise RuntimeError(
            f"tag v{version} already exists; choose the correct next version"
        )
    paths = [repo_root / name for name in (TOOLKIT_VERSION, APPS_VERSION, CHANGELOG)]
    originals = {path: path.read_text(encoding="utf-8") for path in paths}
    proposed = {
        path: re.sub(r"(?m)^__version__\s*=.*$", f'__version__ = "{version}"', text)
        for path, text in originals.items()
        if path.name == "_version.py"
    }
    proposed[repo_root / CHANGELOG] = rewritten
    for path, text in proposed.items():
        print(
            "".join(
                difflib.unified_diff(
                    originals[path].splitlines(True),
                    text.splitlines(True),
                    fromfile=str(path.relative_to(repo_root)),
                    tofile=str(path.relative_to(repo_root)),
                )
            ),
            end="",
        )
    if execute:
        written = []
        try:
            for path, text in proposed.items():
                written.append(path)
                path.write_text(text, encoding="utf-8", newline="\n")
        except OSError:
            for path in written:
                path.write_text(originals[path], encoding="utf-8", newline="\n")
            raise
        print(
            f"prepared v{version}; review, commit and push these paths through a PR to main"
        )
    else:
        print(f"previewed preparation of v{version}; no files or refs changed")
    return 0


def _run_release(repo_root: Path, *, bump: str, note: str, execute: bool) -> int:
    """Compatibility alias: run prepares files and never publishes a release."""
    print(
        "release run is a compatibility alias for prepare; publication requires publish"
    )
    return _prepare_release(repo_root, bump=bump, note=note, execute=execute)


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _github_json(repo_root: Path, endpoint: str, *, missing_ok: bool = False):
    gh = _which("gh")
    if gh is None:
        raise RuntimeError("GitHub CLI gh is required to verify release provenance")
    result = subprocess.run(
        [gh, "api", endpoint],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        if missing_ok and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(
            "GitHub release verification failed; check connectivity and authentication"
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("GitHub release verification returned invalid JSON") from exc


def _release_notes(repo_root: Path, commit: str, version: str) -> str:
    text = _git(repo_root, "show", f"{commit}:{CHANGELOG}") or ""
    lines = text.splitlines()
    pattern = re.compile(r"^##\s+" + re.escape(version) + r"(?:\s+-.*)?$")
    for i, line in enumerate(lines):
        if pattern.fullmatch(line.strip()):
            end = next(
                (j for j in range(i + 1, len(lines)) if lines[j].startswith("## ")),
                len(lines),
            )
            body = "\n".join(lines[i + 1 : end]).strip()
            if body:
                return body
            break
    raise RuntimeError(f"merged changelog has no nonempty v{version} release section")


def _verify_checks(repo_root: Path, commit: str) -> None:
    result = _github_json(
        repo_root, f"repos/{{owner}}/{{repo}}/commits/{commit}/check-runs?per_page=100"
    )
    if not isinstance(result, dict) or not isinstance(result.get("check_runs"), list):
        raise RuntimeError("GitHub check response is incomplete")
    if result.get("total_count", 0) > len(result["check_runs"]):
        raise RuntimeError("GitHub check response is truncated; publication refused")
    latest = {}
    for check in result["check_runs"]:
        if not isinstance(check, dict) or not isinstance(check.get("name"), str):
            raise RuntimeError("GitHub check response contains invalid metadata")
        name = check.get("name")
        if name not in latest or check.get("id", 0) > latest[name].get("id", 0):
            latest[name] = check
    missing = REQUIRED_CHECKS - latest.keys()
    failed = [
        name
        for name, check in latest.items()
        if check.get("status") != "completed"
        or check.get("conclusion")
        not in (
            {"success"}
            if name in REQUIRED_CHECKS
            else {"success", "neutral", "skipped"}
        )
    ]
    if missing or failed:
        raise RuntimeError(
            "merged commit checks are not all successful; missing="
            + ",".join(sorted(missing))
            + "; incomplete/failed="
            + ",".join(sorted(failed))
        )


def _publish_release(
    repo_root: Path, *, version: str, commit: str, execute: bool
) -> int:
    _split_version(version)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("--commit must be the complete immutable merged commit SHA")
    commit = commit.lower()
    head = (_git(repo_root, "rev-parse", "HEAD") or "").strip()
    if head != commit or (_git(repo_root, "status", "--short") or "").strip():
        raise RuntimeError(
            "publish requires a clean checkout of --commit (a detached release checkout is supported)"
        )
    if _current_version(repo_root) != version:
        raise RuntimeError("checkout versions do not match --version")
    _strict_locks(repo_root)
    comparison = _github_json(
        repo_root, f"repos/{{owner}}/{{repo}}/compare/{commit}...main"
    )
    if not isinstance(comparison, dict) or comparison.get("status") not in {
        "ahead",
        "identical",
    }:
        raise RuntimeError("--commit is not contained in the current remote main")
    pulls = _github_json(
        repo_root, f"repos/{{owner}}/{{repo}}/commits/{commit}/pulls?per_page=100"
    )
    if not isinstance(pulls, list) or not any(
        pull.get("merged_at")
        and pull.get("base", {}).get("ref") == "main"
        and pull.get("merge_commit_sha") == commit
        for pull in pulls
    ):
        raise RuntimeError("--commit is not a verified merged PR commit on main")
    _verify_checks(repo_root, commit)
    notes = _release_notes(repo_root, commit, version)
    tag = f"v{version}"
    local, remote = _tag_targets(repo_root, tag)
    if any(target is not None and target != commit for target in (local, remote)):
        raise RuntimeError(
            f"tag {tag} conflicts with --commit; no refs or Release changed"
        )
    existing = _github_json(
        repo_root, f"repos/{{owner}}/{{repo}}/releases/tags/{tag}", missing_ok=True
    )
    if existing is not None and (
        not isinstance(existing, dict)
        or existing.get("tag_name") != tag
        or existing.get("draft")
    ):
        raise RuntimeError(
            "an incompatible or draft GitHub Release already exists; review it explicitly"
        )
    if existing is not None and remote is None:
        raise RuntimeError(
            "an existing Release has no verified remote tag; publication refused"
        )
    print(
        f"verified {tag} at {commit}; strict locks, merged PR and commit checks passed"
    )
    if not execute:
        print("publication preview only; no files, refs or Releases changed")
        return 0
    if remote is None:
        if local is None:
            _git(repo_root, "tag", "-a", tag, commit, "-m", notes)
        _git(repo_root, "push", "origin", f"refs/tags/{tag}")
    if existing is None:
        _run(
            [
                _which("gh") or "gh",
                "release",
                "create",
                tag,
                "--verify-tag",
                "--target",
                commit,
                "--title",
                tag,
                "--notes",
                notes,
            ],
            cwd=repo_root,
            dry_run=False,
        )
    else:
        print(f"existing Release reused: {existing.get('html_url', tag)}")
    print(f"release {tag} completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solar-apps release", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "check", help="Read-only version, clean-tree and strict-lock checks (no fetch)."
    )
    for name in ("prepare", "run"):
        prepare = subparsers.add_parser(
            name, help="Prepare version files for a PR; run is a compatibility alias."
        )
        prepare.add_argument("--bump", choices=_VERSION_BUMPS, required=True)
        prepare.add_argument("--note", required=True)
        prepare.add_argument(
            "--execute",
            action="store_true",
            help="Write version/changelog files on a clean feature branch; never commit or publish.",
        )
    publish = subparsers.add_parser(
        "publish", help="Verify and publish an explicitly merged immutable commit."
    )
    publish.add_argument("--version", required=True)
    publish.add_argument("--commit", required=True)
    publish.add_argument(
        "--execute",
        action="store_true",
        help="Create/push the verified tag and create its GitHub Release; matching objects are reused.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo_root = _repo_root()
        if args.command == "check":
            return _print_check(repo_root)
        if args.command == "publish":
            return _publish_release(
                repo_root,
                version=args.version,
                commit=args.commit,
                execute=args.execute,
            )
        handler = _run_release if args.command == "run" else _prepare_release
        return handler(repo_root, bump=args.bump, note=args.note, execute=args.execute)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
