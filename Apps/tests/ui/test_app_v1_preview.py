"""Process-isolated PyQt6 checks; never mix PyQt6 with legacy Qt in pytest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from solar_apps.frontends.app_v1.timeline import (
    SQLiteTimelineIndex,
    TimeCoordinator,
    TimelineSample,
)

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PREFIX = "APP_V1_SMOKE "


def _run_smoke(tmp_path: Path, mode: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local"),
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "solar_apps.frontends.app_v1.cli",
            "--smoke-test",
            mode,
            "--no-show",
        ],
        cwd=APPS_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(
        (
            item
            for item in completed.stdout.splitlines()
            if item.startswith(SMOKE_PREFIX)
        ),
        None,
    )
    assert line is not None, completed.stdout
    return json.loads(line.removeprefix(SMOKE_PREFIX))


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_preview_help_accepts_all_themes(theme: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "solar_apps.frontends.app_v1.cli",
            "--theme",
            theme,
            "--help",
        ],
        cwd=APPS_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "frontend app-v1" in completed.stdout


def test_offscreen_preview_registers_ten_pages_and_switches_themes(
    tmp_path: Path,
) -> None:
    result = _run_smoke(tmp_path, "basic")

    assert len(result["registered_modules"]) == 10
    assert result["themes"]["light"] == "light"
    assert result["themes"]["dark"] == "dark"
    assert result["foreign_qt_loaded"] is False
    assert result["process_running"] is False
    assert result["dock_count"] == 6


def test_offscreen_preview_cancels_worker_without_leaving_a_process(
    tmp_path: Path,
) -> None:
    result = _run_smoke(tmp_path, "cancel")

    assert result["task_status"] == "cancelled"
    assert result["foreign_qt_loaded"] is False
    assert result["process_running"] is False
    assert not (tmp_path / "Local" / "outputs" / "app_v1").exists()


def test_offscreen_preview_broadcasts_restored_utc_selection(
    tmp_path: Path,
) -> None:
    local = tmp_path / "Local"
    index = SQLiteTimelineIndex(local / "state" / "app_v1" / "time_index.sqlite3")
    index.rebuild_source(
        source_id="base-source",
        module_id="source-trajectory",
        local_source=tmp_path / "radio",
        samples=[
            TimelineSample(
                locator="radio#1",
                observed_at_utc="2025-01-24T23:59:59Z",
            ),
            TimelineSample(
                locator="radio#2",
                observed_at_utc="2025-01-25T00:00:01Z",
            ),
        ],
    )
    index.rebuild_source(
        source_id="aia-source",
        module_id="image-viewer",
        local_source=tmp_path / "aia",
        samples=[
            TimelineSample(
                locator="aia#1",
                observed_at_utc="2025-01-24T23:59:58.5Z",
            )
        ],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("base-source")
    coordinator.register_source(
        "aia-source",
        offset_seconds=0.5,
        tolerance_seconds=0.0,
    )
    coordinator.set_base_source("base-source")
    coordinator.save(local / "workspaces" / "app_v1" / "preview.timeline.json")

    result = _run_smoke(tmp_path, "timeline")

    assert result["source_count"] == 2
    assert result["matched_count"] == 2
    assert result["synced_page_count"] == 9
    assert result["current_time_utc"] == "2025-01-24T23:59:59Z"


def test_offscreen_preview_shows_complete_confirmation_summary(
    tmp_path: Path,
) -> None:
    result = _run_smoke(tmp_path, "dialog")

    assert result["summary_fields"] == [
        "Module:",
        "Input:",
        "Parameters:",
        "Output:",
        "Workload:",
    ]
    assert result["accepted"] is False
    assert result["process_running"] is False


def test_offscreen_app_round_trips_project_and_parameter_preset(
    tmp_path: Path,
) -> None:
    result = _run_smoke(tmp_path, "project")

    assert result["project_file"] == "smoke-project.spapp.json"
    assert result["preset_file"] == "smoke-preset.json"
    assert result["restored_project_id"] == "smoke-project"
    assert result["restored_parameter"] == 2.5
    assert result["artifact_count"] == 0
    assert result["process_running"] is False


def test_offscreen_app_queues_batch_and_redraws_last_task(tmp_path: Path) -> None:
    result = _run_smoke(tmp_path, "workflow")

    assert result["task_count"] == 3
    assert result["modules"] == ["workbench", "radio-workspace", "radio-workspace"]
    assert result["statuses"] == ["succeeded", "succeeded", "succeeded"]
    assert result["redraw_created"] is True
    assert result["process_running"] is False


def test_offscreen_app_recovers_failed_task_with_original_parameters(
    tmp_path: Path,
) -> None:
    result = _run_smoke(tmp_path, "recovery")

    assert result["retry_count"] == 1
    assert result["retry_of"]
    assert result["statuses"] == ["failed", "succeeded"]
    assert result["marker_created"] is True
    assert result["process_running"] is False
