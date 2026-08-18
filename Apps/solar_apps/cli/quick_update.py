"""Fast daily-save, verify, push, and PR commands.

These commands stay within the repository workflow documented in
``WORKFLOW_README.md``: selective staging, reviewed commits, feature branches,
and Pull Requests against ``main``. They never run ``git add .``, never
force-push, and refuse to commit when the staged set looks like it carries
private data, generated artifacts, or unrelated changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from solar_apps.platform.environment import inspect_miniforge_runtime
from solar_apps.platform.layout import RuntimeLayout

_BLOCKED_SUFFIXES = {
    ".avi",
    ".csv",
    ".db",
    ".fit",
    ".fits",
    ".gif",
    ".h5",
    ".hdf5",
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
    ".pkl",
    ".png",
    ".sqlite",
    ".tsv",
    ".webp",
    ".xls",
    ".xlsx",
}
_BLOCKED_PARTS = {
    "Local",
    "Local-migration-backup",
    "outputs",
    "logs",
    "history",
    "legacy",
    "legacy_tests",
}


def _repo_root() -> Path:
    return RuntimeLayout.discover().repo_root


def _python() -> str:
    return str(inspect_miniforge_runtime().executable)


def _run_checked(command: list[str], *, cwd: Path) -> int:
    print("$ " + " ".join(command))
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _python_checks(repo_root: Path) -> int:
    python = _python()
    checks = [
        [python, "-m", "pip", "check"],
        [python, "-m", "compileall", "-q", "Python/solar_toolkit", "Python/tests"],
        [python, "-m", "ruff", "check", "Python/solar_toolkit", "Python/tests"],
    ]
    for command in checks:
        if _run_checked(command, cwd=repo_root) != 0:
            return 1
    return 0


def _git(
    repo_root: Path, *args: str, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=capture,
        text=True,
    )
    return completed


def _staged_has_private_content(repo_root: Path) -> list[str]:
    completed = _git(repo_root, "diff", "--cached", "--name-only", capture=True)
    names = [line for line in completed.stdout.splitlines() if line.strip()]
    offenders: list[str] = []
    for name in names:
        path = Path(name)
        if path.suffix.casefold() in _BLOCKED_SUFFIXES:
            offenders.append(f"{name}: blocked suffix {path.suffix}")
        if any(part.casefold() in _BLOCKED_PARTS for part in path.parts):
            offenders.append(f"{name}: private/ignored directory part")
    return offenders


def _require_clean_review(repo_root: Path) -> int:
    print("$ git status --short")
    completed = _git(repo_root, "status", "--short", capture=True)
    if completed.returncode != 0:
        return 1
    print(completed.stdout if completed.stdout.strip() else "(clean)")
    return 0


def _quick_check(repo_root: Path) -> int:
    return _python_checks(repo_root)


def _quick_save(repo_root: Path, *, message: str, paths: list[str]) -> int:
    if not message.strip():
        print("quick-save: error: -m/--message is required", file=sys.stderr)
        return 2
    if not paths:
        print(
            "quick-save: error: no paths given; stage explicit paths after --",
            file=sys.stderr,
        )
        return 2
    if _git(repo_root, "add", "--", *paths).returncode != 0:
        return 1
    offenders = _staged_has_private_content(repo_root)
    if offenders:
        print(
            "quick-save: error: staged paths look private or generated:",
            file=sys.stderr,
        )
        for offender in offenders:
            print(f"  - {offender}", file=sys.stderr)
        _git(repo_root, "restore", "--staged", "--", *paths)
        return 1
    print("$ git diff --cached --stat")
    stat = _git(repo_root, "diff", "--cached", "--stat", capture=True)
    print(stat.stdout if stat.stdout.strip() else "(nothing staged)")
    if _git(repo_root, "commit", "-m", message).returncode != 0:
        return 1
    print("committed.")
    return 0


def _quick_push(repo_root: Path) -> int:
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD", capture=True)
    branch_name = (branch.stdout or "").strip()
    if not branch_name:
        return 1
    if branch_name == "main":
        print(
            "quick-push: error: on main; create a feature branch first "
            "(see WORKFLOW_README.md section 2)",
            file=sys.stderr,
        )
        return 1
    if _git(repo_root, "push", "-u", "origin", branch_name).returncode != 0:
        return 1
    print("pushed. creating a Pull Request against main...")
    import shutil

    gh = shutil.which("gh")
    if not gh:
        print("warning: gh CLI not found; open a PR on GitHub manually")
        return 0
    return _run_checked(
        [gh, "pr", "create", "--base", "main", "--fill"],
        cwd=repo_root,
    )


def _quick_update(repo_root: Path, *, message: str, paths: list[str]) -> int:
    if _quick_check(repo_root) != 0:
        print("quick-update: checks failed; nothing committed", file=sys.stderr)
        return 1
    if _quick_save(repo_root, message=message, paths=paths) != 0:
        return 1
    return _quick_push(repo_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps quick",
        description="Fast verify, commit, push, and PR commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check", help="Run pip check, compileall, and ruff for public Python."
    )
    check.set_defaults(handler=lambda args, repo: _quick_check(repo))

    save = subparsers.add_parser(
        "save", help="Stage explicit paths, review, and commit."
    )
    save.add_argument("-m", "--message", required=True)
    save.add_argument("paths", nargs="+")
    save.set_defaults(
        handler=lambda args, repo: _quick_save(
            repo, message=args.message, paths=args.paths
        )
    )

    push = subparsers.add_parser(
        "push", help="Push the feature branch and open a PR against main."
    )
    push.set_defaults(handler=lambda args, repo: _quick_push(repo))

    update = subparsers.add_parser("update", help="check + save + push in one pass.")
    update.add_argument("-m", "--message", required=True)
    update.add_argument("paths", nargs="+")
    update.set_defaults(
        handler=lambda args, repo: _quick_update(
            repo, message=args.message, paths=args.paths
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    try:
        return args.handler(args, repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"quick: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
