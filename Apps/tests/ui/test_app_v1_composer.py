"""Process-isolated PyQt6 Image Composer interaction smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

APPS_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "APP_V1_COMPOSER "


@pytest.mark.parametrize("theme_mode", ["light", "dark", "auto"])
def test_pyqt6_composer_adds_overlaps_grids_and_prepares_export(
    tmp_path: Path,
    theme_mode: str,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (80, 40), "orange").save(images / "camera_20260724_120000.png")
    Image.new("RGB", (80, 40), "blue").save(images / "camera_20260724_120001.png")
    script = r"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication, QMessageBox
from solar_apps.frontends.app_v1.phase4 import Phase4ComposerAdapter
from solar_apps.frontends.app_v1.phase4_page import Phase4ComposerPanel
from solar_apps.frontends.app_v1.theme import AppV1ThemeController
from solar_apps.frontends.image_composer.project import load_project
from solar_apps.platform.layout import RuntimeLayout

root = Path(sys.argv[1])
local = Path(sys.argv[2])
application = QApplication(["app-v1-composer-smoke"])
theme = AppV1ThemeController(application, initial_mode=sys.argv[3])
layout = RuntimeLayout.discover(
    Path.cwd().parent,
    environ={"SOLAR_APPS_LOCAL_ROOT": str(local)},
)
panel = Phase4ComposerPanel(
    Phase4ComposerAdapter(layout, allowed_roots=(root,))
)
folder = panel.add_folder(root)
panel.folder_start.setValue(1)
panel.folder_end.setValue(2)
panel.folder_offset.setValue(0.5)
first = panel.add_slot(folder.id, QPointF(10, 10), ordinal=2)
second = panel.add_slot(folder.id, QPointF(20, 20))
panel.scene.clearSelection()
panel._items[first.id].setSelected(True)
panel._items[second.id].setSelected(True)
panel.equal_size()
panel.auto_grid()
panel.change_layer(1)
panel.match_mode.setCurrentText("relative")
panel.match_tolerance.setValue(2.5)
panel.match_strict.setChecked(False)
panel.scene.clearSelection()
panel._items[first.id].setSelected(True)
panel.duplicate_selected_slot()
duplicate_id = panel._selected_items()[0].slot.id
panel.delete_selected_slot()
panel.set_current_time(
    datetime(2026, 7, 24, 12, 0, 1, 500000, tzinfo=timezone.utc)
)
launch = panel.adapter.build_static_export(panel.project, scale=2)
panel.project.export.output_format = "avi"
sequence = panel.adapter.build_sequence_export(
    panel.project,
    output_path=root / "composition.avi",
)
result = {
    "folder_count": len(panel.project.folders),
    "slot_count": len(panel.project.slots),
    "overlap_supported": first.id in panel._items and second.id in panel._items,
    "grid_positions": [[slot.x, slot.y] for slot in panel.project.slots],
    "z_indexes": sorted(slot.z_index for slot in panel.project.slots),
    "sync_ordinals": [slot.preview_ordinal for slot in panel.project.slots],
    "range": [folder.start_index, folder.end_index, folder.offset_seconds],
    "matching": [panel.project.matching.mode, panel.project.matching.tolerance_seconds, panel.project.matching.strict],
    "exact_drag_ordinal": first.preview_ordinal,
    "duplicate_deleted": duplicate_id not in panel._items,
    "sequence_suffix": sequence.arguments[sequence.arguments.index("--output") + 1],
    "output_local": str(launch.output_dir).startswith(str(local)),
    "foreign_qt": any(name.startswith("PySide6") or name.startswith("PyQt5") for name in sys.modules),
}
requested = []
panel.adapter.discard_prepared_export(launch)
panel.adapter.discard_prepared_export(sequence)
panel.task_requested.connect(requested.append)
queued = panel.adapter.build_static_export(panel.project)
queued_path = Path(queued.arguments[queued.arguments.index("--project") + 1])
panel.confirm = lambda *_args: True
panel._request(lambda: queued)
panel.project.slots[0].preview_ordinal = 1
cancelled = panel.adapter.build_static_export(panel.project)
cancelled_path = Path(cancelled.arguments[cancelled.arguments.index("--project") + 1])
panel.confirm = lambda *_args: False
panel._request(lambda: cancelled)
panel.adapter.discard_prepared_export(queued)
result["confirmed_launch_count"] = len(requested)
result["cancelled_snapshot_removed"] = not cancelled_path.exists()
result["queued_snapshot_ordinal"] = load_project(queued_path).slots[0].preview_ordinal
result["no_prepared_drafts"] = not panel.adapter._prepared_projects
QMessageBox.information = lambda *_args: QMessageBox.StandardButton.Ok
save_path = root / "composition.fic.json"
result["saved_clean"] = panel._save_project_to(save_path) and not panel._dirty
panel.folder_offset.setValue(0.75)
result["edited_dirty"] = panel._dirty
QMessageBox.warning = lambda *_args: QMessageBox.StandardButton.Cancel
result["new_cancel_preserves_project"] = not panel.new_project() and len(panel.project.slots) == 2
QMessageBox.warning = lambda *_args: QMessageBox.StandardButton.Discard
result["new_discards_project"] = panel.new_project() and not panel.project.slots
result["reloaded_clean"] = panel.import_project(save_path) and not panel._dirty
result["reloaded_slots"] = len(panel.project.slots)
result["reloaded_resolved"] = all(folder.resolved for folder in panel.project.folders)
result["theme_mode"] = theme.mode
panel.resize(1500, 1050)
panel.show()
application.processEvents()
result["rendered"] = panel.grab().save(sys.argv[4])
print("APP_V1_COMPOSER " + json.dumps(result, sort_keys=True))
panel.close()
panel.deleteLater()
application.processEvents()
"""
    environment = dict(os.environ)
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(images),
            str(tmp_path / "Local"),
            theme_mode,
            str(tmp_path / f"composer-{theme_mode}.png"),
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
        item for item in completed.stdout.splitlines() if item.startswith(PREFIX)
    )
    result = json.loads(line.removeprefix(PREFIX))
    assert result["folder_count"] == 1
    assert result["slot_count"] == 2
    assert result["overlap_supported"] is True
    assert result["z_indexes"] == [0, 1]
    assert result["sync_ordinals"] == [2, 2]
    assert result["range"] == [1, 2, 0.5]
    assert result["matching"] == ["relative", 2.5, False]
    assert result["exact_drag_ordinal"] == 2
    assert result["duplicate_deleted"] is True
    assert result["sequence_suffix"].endswith("composition.avi")
    assert result["output_local"] is True
    assert result["foreign_qt"] is False
    assert result["confirmed_launch_count"] == 1
    assert result["cancelled_snapshot_removed"] is True
    assert result["queued_snapshot_ordinal"] == 2
    assert result["no_prepared_drafts"] is True
    assert result["saved_clean"] is True
    assert result["edited_dirty"] is True
    assert result["new_cancel_preserves_project"] is True
    assert result["new_discards_project"] is True
    assert result["reloaded_clean"] is True
    assert result["reloaded_slots"] == 2
    assert result["reloaded_resolved"] is True
    assert result["theme_mode"] == theme_mode
    assert result["rendered"] is True
