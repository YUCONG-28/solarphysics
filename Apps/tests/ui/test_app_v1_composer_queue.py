"""Process-isolated export queue, cancelled draft, and retry regression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

APPS_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "APP_V1_COMPOSER_QUEUE "


@pytest.mark.parametrize("output_format", ["png", "mp4", "avi"])
def test_queued_composer_and_retry_keep_original_snapshot(
    tmp_path: Path, output_format: str
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (40, 20), "red").save(images / "camera_20260724_120000.png")
    Image.new("RGB", (40, 20), "blue").save(images / "camera_20260724_120001.png")
    script = r"""
import json
import sys
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer
from solar_apps.frontends.app_v1 import tasks
from solar_apps.frontends.app_v1.phase4 import Phase4ComposerAdapter
from solar_apps.frontends.app_v1.tasks import TaskQueueController
from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.models import CanvasSettings, ComposerProject, FolderSource, LayoutSlot, MatchSettings
from solar_apps.platform.layout import RuntimeLayout

application = QCoreApplication(["composer-queue-regression"])
root = Path(sys.argv[1])
layout = RuntimeLayout.discover(Path.cwd().parent, environ={"SOLAR_APPS_LOCAL_ROOT": sys.argv[2]})
adapter = Phase4ComposerAdapter(layout, allowed_roots=(root,))
folder = FolderSource.create(root, scan_folder(root))
folder.end_index = 1
project = ComposerProject(
    canvas=CanvasSettings(width=64, height=48), folders=[folder],
    slots=[LayoutSlot.create(folder.id, 1, x=0, y=0, width=64, height=48)],
    matching=MatchSettings(master_folder_id=folder.id),
)
output_format = sys.argv[3]
if output_format != "png":
    project.export.output_format = output_format
tasks.selected_python_executable = lambda: Path(sys.executable)
controller = TaskQueueController(layout)
# Run source modules from this isolated checkout rather than an older editable install.
controller.set_working_directory(Path.cwd())

def drive_until(predicate):
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if predicate() else None)
    timer.start()
    QTimer.singleShot(10_000, loop.quit)
    loop.exec()
    timer.stop()
    if not predicate():
        raise RuntimeError("Queue did not reach expected state")

def terminal():
    return not controller.active_task_ids and all(
        record.status in {"succeeded", "failed", "cancelled"} for record in controller.records
    )

def prepare():
    if output_format == "png":
        return adapter.build_static_export(project)
    return adapter.build_sequence_export(project, fps=4, save_png_frames=True)

def decode_output(path):
    if output_format == "png":
        with Image.open(path) as image:
            assert image.size == (64, 48)
            return image.getpixel((32, 24)), 1
    import cv2
    reader = cv2.VideoCapture(str(path))
    colors = []
    try:
        while True:
            available, frame = reader.read()
            if not available:
                break
            assert frame.shape[:2] == (48, 64)
            colors.append(tuple(int(channel) for channel in frame[24, 32][::-1]))
    finally:
        reader.release()
    assert len(colors) == 1
    return colors[0], len(colors)

try:
    blocker = controller.enqueue_python_module(
        title="blocker", module_id="workbench",
        python_module="solar_apps.frontends.app_v1.task_worker",
        arguments=("--steps", "10", "--delay-ms", "80"),
    )
    drive_until(lambda: blocker.status == "running")
    launch = prepare()
    adapter.confirm_prepared_export(launch)
    first = controller.enqueue_batch([launch])[0]
    assert first.status == "queued"
    project.slots[0].preview_ordinal = 2
    folder.start_index = folder.end_index = 2
    draft = prepare()
    draft_path = Path(draft.arguments[draft.arguments.index("--project") + 1])
    adapter.discard_prepared_export(draft)
    adapter.save_workspace_project(project)
    drive_until(terminal)
    assert first.status == "succeeded", first.logs
    output = Path(launch.arguments[launch.arguments.index("--output") + 1])
    first_color, first_frames = decode_output(output)
    retry = controller.retry(first.task_id)
    drive_until(terminal)
    assert retry.status == "succeeded", retry.logs
    retry_color, retry_frames = decode_output(output)
    result = {
        "first_status": first.status, "retry_status": retry.status,
        "first_color": first_color, "retry_color": retry_color,
        "first_frames": first_frames, "retry_frames": retry_frames,
        "same_arguments": retry.arguments == first.arguments,
        "retry_of": retry.retry_of == first.task_id,
        "draft_removed": not draft_path.exists(),
        "no_children": not controller.process_running and not controller.active_task_ids,
    }
    print("APP_V1_COMPOSER_QUEUE " + json.dumps(result, sort_keys=True))
finally:
    controller.shutdown()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(images),
            str(tmp_path / "Local"),
            output_format,
        ],
        cwd=APPS_ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(
        line for line in completed.stdout.splitlines() if line.startswith(PREFIX)
    )
    result = json.loads(line.removeprefix(PREFIX))
    assert result["first_status"] == result["retry_status"] == "succeeded"
    assert result["first_color"] == result["retry_color"]
    assert all(
        abs(actual - expected) < 20
        for actual, expected in zip(result["first_color"], [255, 0, 0], strict=True)
    )
    assert result["first_frames"] == result["retry_frames"] == 1
    assert result["same_arguments"] is True
    assert result["retry_of"] is True
    assert result["draft_removed"] is True
    assert result["no_children"] is True
