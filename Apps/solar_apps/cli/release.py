"""Repository release command: version bump, changelog, tag, and GitHub release.

The release is fail-closed and dry-run by default. ``--execute`` is the only
path that writes the version files, updates the changelog, commits, tags,
pushes, or creates a GitHub release. Scientific and privacy boundaries are
preserved: the command never force-pushes, never rewrites history, and never
uploads anything beyond the reviewed version bump and changelog entries.
"""

from __future__ import annotations

import argparse
import ast
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


def _check_clean(repo_root: Path) -> list[str]:
    problems: list[str] = []
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD", dry_run=False)
    if (branch or "").strip() != "main":
        problems.append(f"not on main branch (current: {(branch or '').strip()!r})")
    status = _git(repo_root, "status", "--short", dry_run=False)
    if status and status.strip():
        problems.append("working tree is not clean:\n" + (status or "").strip())
    _git(repo_root, "fetch", "origin", dry_run=False)
    ahead = _git(repo_root, "rev-list", "--count", "HEAD..origin/main", dry_run=False)
    behind = _git(repo_root, "rev-list", "--count", "origin/main..HEAD", dry_run=False)
    if (ahead or "0").strip() != "0":
        problems.append("local main has unpushed commits")
    if (behind or "0").strip() != "0":
        problems.append("local main is behind origin/main; run: git pull --ff-only")
    return problems


def _print_check(repo_root: Path) -> int:
    try:
        version = _current_version(repo_root)
    except (RuntimeError, ValueError) as exc:
        print(f"release check: error: {exc}", file=sys.stderr)
        return 2
    problems = _check_clean(repo_root)
    print(f"current version: {version}")
    print("branch/tree state: " + ("ok" if not problems else "FAILED"))
    for problem in problems:
        print(f"  - {problem}")
    # Exact-environment gate: report the committed lock state without failing.
    locks = sorted((repo_root / "environment" / "locks").glob("*"))
    if locks:
        print(f"committed environment locks: {len(locks)} target(s)")
    else:
        print(
            "exact-environment gate: RED (no committed platform locks; "
            "see environment/README.md)"
        )
    return 0 if not problems else 1


def _run_release(repo_root: Path, *, bump: str, note: str, execute: bool) -> int:
    dry_run = not execute
    try:
        current = _current_version(repo_root)
        new_version = _bump_version(current, bump)
        rewritten, _ = _changelog_release_block(repo_root, new_version)
    except (RuntimeError, ValueError) as exc:
        print(f"release: error: {exc}", file=sys.stderr)
        return 2

    if not note.strip():
        print("release: error: --note is required", file=sys.stderr)
        return 2

    problems = _check_clean(repo_root)
    if problems:
        print("release: error: preconditions failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"release v{current} -> v{new_version} ({bump})"
        + (" [dry-run]" if dry_run else "")
    )

    print("would update:" if dry_run else "updating:")
    print(f"  {TOOLKIT_VERSION}")
    print(f"  {APPS_VERSION}")
    print(f"  {CHANGELOG}")
    if execute:
        _write_version(repo_root / TOOLKIT_VERSION, new_version)
        _write_version(repo_root / APPS_VERSION, new_version)
        (repo_root / CHANGELOG).write_text(rewritten, encoding="utf-8", newline="\n")

    tag = f"v{new_version}"
    _git(
        repo_root,
        "add",
        "--",
        TOOLKIT_VERSION,
        APPS_VERSION,
        CHANGELOG,
        dry_run=dry_run,
    )
    _git(
        repo_root,
        "commit",
        "-m",
        f"chore: release {tag}",
        dry_run=dry_run,
    )
    _git(repo_root, "tag", "-a", tag, "-m", note, dry_run=dry_run)
    _git(repo_root, "push", "origin", "main", dry_run=dry_run)
    _git(repo_root, "push", "origin", tag, dry_run=dry_run)

    gh = _which("gh")
    if gh:
        _run(
            [
                gh,
                "release",
                "create",
                tag,
                "--title",
                tag,
                "--notes",
                note,
            ],
            cwd=repo_root,
            dry_run=dry_run,
        )
    else:
        print("warning: gh CLI not found; GitHub release step skipped")

    print(
        f"release {tag} " + ("previewed (no changes made)" if dry_run else "completed")
    )
    return 0


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps release",
        description=(
            "Release the monorepository: bump the shared version, rewrite the "
            "changelog, tag, push, and create a GitHub release. Dry-run by default."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run release preconditions only.")
    check.set_defaults(handler=_print_check)

    run = subparsers.add_parser("run", help="Bump and release (dry-run by default).")
    run.add_argument("--bump", choices=_VERSION_BUMPS, required=True)
    run.add_argument("--note", required=True)
    run.add_argument(
        "--execute",
        action="store_true",
        help="Apply the release instead of previewing it.",
    )
    run.set_defaults(
        handler=lambda args, repo: _run_release(
            repo, bump=args.bump, note=args.note, execute=args.execute
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    try:
        if args.command == "check":
            return _print_check(repo_root)
        return args.handler(args, repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
