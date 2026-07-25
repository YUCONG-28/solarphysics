"""Process-isolated PyQt6 Image Composer interaction smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

APPS_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "APP_V1_COMPOSER "


def test_pyqt6_composer_adds_overlaps_grids_and_prepares_export(
    tmp_path: Path,
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
from PyQt6.QtWidgets import QApplication
from solar_apps.frontends.app_v1.phase4 import Phase4ComposerAdapter
from solar_apps.frontends.app_v1.phase4_page import Phase4ComposerPanel
from solar_apps.platform.layout import RuntimeLayout

root = Path(sys.argv[1])
local = Path(sys.argv[2])
application = QApplication(["app-v1-composer-smoke"])
layout = RuntimeLayout.discover(
    Path.cwd().parent,
    environ={"SOLAR_APPS_LOCAL_ROOT": str(local)},
)
panel = Phase4ComposerPanel(
    Phase4ComposerAdapter(layout, allowed_roots=(root,))
)
folder = panel.add_folder(root)
first = panel.add_slot(folder.id, QPointF(10, 10))
second = panel.add_slot(folder.id, QPointF(20, 20))
panel.scene.clearSelection()
panel._items[first.id].setSelected(True)
panel._items[second.id].setSelected(True)
panel.equal_size()
panel.auto_grid()
panel.change_layer(1)
panel.set_current_time(datetime(2026, 7, 24, 12, 0, 1, tzinfo=timezone.utc))
launch = panel.adapter.build_static_export(panel.project, scale=2)
result = {
    "folder_count": len(panel.project.folders),
    "slot_count": len(panel.project.slots),
    "overlap_supported": first.id in panel._items and second.id in panel._items,
    "grid_positions": [[slot.x, slot.y] for slot in panel.project.slots],
    "z_indexes": sorted(slot.z_index for slot in panel.project.slots),
    "sync_ordinals": [slot.preview_ordinal for slot in panel.project.slots],
    "output_local": str(launch.output_dir).startswith(str(local)),
    "foreign_qt": any(name.startswith("PySide6") or name.startswith("PyQt5") for name in sys.modules),
}
print("APP_V1_COMPOSER " + json.dumps(result, sort_keys=True))
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
    assert result["output_local"] is True
    assert result["foreign_qt"] is False
