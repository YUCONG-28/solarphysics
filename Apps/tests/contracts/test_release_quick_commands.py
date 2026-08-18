"""Contracts for the release and quick-update repository commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from solar_apps.cli import release
from solar_apps.cli import router

APPS_ROOT = Path(__file__).resolve().parents[2]


def test_tool_catalog_registers_release_and_quick() -> None:
    assert router.TOOL_TARGETS["release"] == "solar_apps.cli.release"
    assert router.TOOL_TARGETS["quick"] == "solar_apps.cli.quick_update"


def test_release_help_lists_check_and_run() -> None:
    help_text = release.build_parser().format_help()
    assert "check" in help_text
    assert "run" in help_text


def test_release_run_requires_bump_and_note_and_offers_execute() -> None:
    parser = release.build_parser()
    # argparse exposes sub-parser help only after selection; assert the flags
    # exist on the run sub-parser by inspecting the registered action strings.
    run_parser = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    choices = run_parser.choices
    assert "run" in choices
    run_help = choices["run"].format_help()
    assert "--execute" in run_help
    assert "--bump" in run_help
    assert "--note" in run_help


def test_quick_help_lists_commands() -> None:
    from solar_apps.cli import quick_update

    help_text = quick_update.build_parser().format_help()
    for command in ("check", "save", "push", "update"):
        assert command in help_text


def test_version_bump_levels() -> None:
    assert release._bump_version("0.3.0", "patch") == "0.3.1"
    assert release._bump_version("0.3.0", "minor") == "0.4.0"
    assert release._bump_version("0.3.0", "major") == "1.0.0"
    assert release._bump_version("0.3.9", "patch") == "0.3.10"


def test_version_bump_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        release._bump_version("0.3", "patch")
    with pytest.raises(ValueError):
        release._bump_version("abc", "patch")
    with pytest.raises(ValueError):
        release._bump_version("0.3.0", "nope")


def test_changelog_release_block_rewrites_in_order(tmp_path: Path) -> None:
    changelog = tmp_path / "Python" / "CHANGELOG.md"
    changelog.parent.mkdir(parents=True)
    changelog.write_text(
        "# Changelog\n\n"
        "## 0.3.0\n\n- shipped\n\n"
        "## Unreleased\n\n- new thing\n- another thing\n\n"
        "## 0.2.0\n\n- old\n",
        encoding="utf-8",
    )

    rewritten, block = release._changelog_release_block(tmp_path, "0.3.1")

    assert block == "- new thing\n- another thing"
    assert rewritten.index("## Unreleased") < rewritten.index("## 0.3.1")
    assert rewritten.index("## 0.3.1") < rewritten.index("## 0.3.0")
    assert rewritten.index("## 0.3.0") < rewritten.index("## 0.2.0")
    assert "## 0.3.1 - " in rewritten
    assert "- shipped" in rewritten
    assert "- old" in rewritten


def test_changelog_release_block_requires_unreleased(tmp_path: Path) -> None:
    changelog = tmp_path / "Python" / "CHANGELOG.md"
    changelog.parent.mkdir(parents=True)
    changelog.write_text("# Changelog\n\n## 0.3.0\n\n- shipped\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        release._changelog_release_block(tmp_path, "0.3.1")


def test_read_and_write_version_round_trip(tmp_path: Path) -> None:
    version_file = tmp_path / "_version.py"
    version_file.write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    assert release._read_version(version_file) == "0.3.0"
    release._write_version(version_file, "0.4.0")
    assert release._read_version(version_file) == "0.4.0"
