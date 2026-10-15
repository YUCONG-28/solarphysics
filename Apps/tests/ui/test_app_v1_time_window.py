"""Process-isolated UTC image-window controls in all App v1 themes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_time_window_controls_are_independent_of_browsing_and_restore(tmp_path: Path):
    script = r"""
import json, sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from solar_apps.frontends.app_v1.theme import AppV1ThemeController
from solar_apps.frontends.app_v1.timeline import SQLiteTimelineIndex, TimeCoordinator, TimelineSample
from solar_apps.frontends.app_v1.time_sync_ui import TimeSyncPanel

root = Path(sys.argv[1])
application = QApplication([])
theme = AppV1ThemeController(application)
index = SQLiteTimelineIndex(root / "timeline.sqlite3")
index.rebuild_source(source_id="images", module_id="image-composer", local_source=root,
    samples=[TimelineSample(str(number), timestamp) for number, timestamp in enumerate([
    "2025-01-24T04:48:29.999999999Z", "2025-01-24T04:48:30.000000000Z",
    "2025-01-24T04:48:30.000000001Z", "2025-01-24T04:49:00.000000000Z"])])
coordinator = TimeCoordinator(index)
coordinator.register_source("images")
panel = TimeSyncPanel(coordinator)
panel.resize(900, 460)
panel.show()
rebuilt = []
panel.plan_changed.connect(lambda plan: rebuilt.append(plan.fingerprint))
panel.time_start.setText("2025-01-24T04:48:30Z")
panel.time_end.setText("2025-01-24T04:49:00Z")
panel.apply_window.click()
application.processEvents()
assert len(coordinator.current_plan.rows) == 2
assert "2 output image(s)" in panel.window_status.text()
assert panel.time_start.text() == "2025-01-24T04:48:30.000000000Z"
saved = coordinator.to_dict()
for mode in ("Light", "Dark", "Auto"):
    theme.set_mode(mode)
    panel._step(1)
    panel._step(1)
    panel._step(-1)
    application.processEvents()
    assert coordinator.time_start == saved["time_start"]
    assert coordinator.time_end == saved["time_end"]
    assert panel.current_time.text() == "2025-01-24T04:48:30.000000000Z"
    for width in (480, 900):
        panel.resize(width, 460)
        application.processEvents()
        assert panel.time_start.isVisible() and panel.time_end.isVisible()
        assert panel.time_start.geometry().bottom() < panel.time_end.geometry().top()
        assert panel.time_end.geometry().bottom() < panel.current_time.geometry().top()
    panel.grab().save(str(root / (mode.lower() + ".png")))
panel.time_end.setText("2025-01-24T04:48:30Z")
panel.apply_window.click()
assert "time_start < time_end" in panel.match_summary.text()
assert coordinator.time_end == saved["time_end"]
coordinator.set_time_window("2025-01-24T04:50:00Z", "2025-01-24T04:51:00Z")
application.processEvents()
assert "No data inside" in panel.match_summary.text()
assert panel.current_time.text() == "No UTC time selected"
assert coordinator.current_time_utc is None
coordinator.load_dict(saved)
application.processEvents()
assert panel.time_start.text() == saved["time_start"]
assert panel.time_end.text() == saved["time_end"]
assert len(coordinator.current_plan.rows) == 2
panel.clear_window.click()
application.processEvents()
assert coordinator.time_start is None and coordinator.time_end is None
assert panel.time_start.text() == "" and panel.time_end.text() == ""
assert len(coordinator.current_plan.rows) == 4
assert len(rebuilt) == 4
panel.close()
application.processEvents()
print("TIME_WINDOW_QA " + json.dumps({"themes": ["Light", "Dark", "Auto"], "rows": 4,
    "window_independent": True, "half_open_nanoseconds": True, "empty_cleared": True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = next(
        line.removeprefix("TIME_WINDOW_QA ")
        for line in result.stdout.splitlines()
        if line.startswith("TIME_WINDOW_QA ")
    )
    assert json.loads(summary)["themes"] == ["Light", "Dark", "Auto"]
    assert all(
        (tmp_path / f"{theme}.png").is_file() for theme in ("light", "dark", "auto")
    )


def test_composer_registration_respects_own_window_without_overwriting_global(tmp_path):
    script = r"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication, QLabel
from solar_apps.frontends.app_v1.timeline import SQLiteTimelineIndex, TimeCoordinator
from solar_apps.frontends.app_v1.time_sync_ui import TimeSyncPanel
from solar_apps.frontends.app_v1.window import AppV1MainWindow
from solar_apps.frontends.image_composer.models import ComposerProject, FolderSource, ImageRecord, LayoutSlot

root = Path(sys.argv[1])
application = QApplication([])
index = SQLiteTimelineIndex(root / "composer-timeline.sqlite3")
coordinator = TimeCoordinator(index, time_start="2025-01-24T04:48:01Z", time_end="2025-01-24T04:48:02Z")
panel = TimeSyncPanel(coordinator)
records = [ImageRecord(number + 1, root / (str(number) + ".png"),
    datetime(2025, 1, 24, 4, 48, number, tzinfo=timezone.utc), "filename",
    timestamp_text=f"2025-01-24T04:48:0{number}.000000000Z") for number in range(4)]
folder = FolderSource("camera", root, "Camera", records, 1, 4, offset_seconds=1000)
project = ComposerProject(folders=[folder], slots=[LayoutSlot.create("camera", 1, x=0, y=0)])
project.matching.time_start = "2025-01-24T04:48:01Z"
project.matching.time_end = "2025-01-24T04:48:03Z"
cleared = []
messages = []
page = SimpleNamespace(time_status=QLabel(), phase4_panel=SimpleNamespace(project=project,
    clear_sync_selection=lambda: cleared.append(True)))
window = SimpleNamespace(module_pages={"image-composer": page},
    _composer_timeline_source_ids=set(), timeline_index=index, time_coordinator=coordinator, time_sync_panel=panel)
AppV1MainWindow._sync_composer_timeline_sources(window)
assert len(index.samples("composer-camera")) == 2
assert [item.locator for item in index.samples("composer-camera")] == [str(root / "1.png"), str(root / "2.png")]
assert len(coordinator.current_plan.rows) == 1
assert coordinator.current_plan.rows[0].master_frame.frame_id == str(root / "1.png")
assert coordinator.time_start == "2025-01-24T04:48:01.000000000Z"
assert coordinator.time_end == "2025-01-24T04:48:02.000000000Z"
coordinator.set_time_window()
assert len(coordinator.current_plan.rows) == 2
assert project.matching.time_end == "2025-01-24T04:48:03Z"
window.statusBar = lambda: SimpleNamespace(showMessage=lambda text: messages.append(text))
panel.plan_changed.connect(lambda plan: AppV1MainWindow._timeline_plan_changed(window, plan))
coordinator.set_time_window("2025-01-24T04:50:00Z", "2025-01-24T04:51:00Z")
assert cleared == [True]
assert "no data inside" in page.time_status.text()
assert "no images can be exported" in messages[-1]
panel.close()
application.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
