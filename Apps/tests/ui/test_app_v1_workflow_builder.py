"""Offscreen checks for the shared schema form and visual workflow editor."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

from solar_apps.frontends.app_v1.function_catalog import (
    DEFAULT_FUNCTION_CATALOG,
    page_template,
)
from solar_apps.frontends.app_v1.flow_execution import FlowExecutionController
from solar_apps.frontends.app_v1.flows import (
    AppV1FlowV1,
    FlowEdgeV1,
    FlowNodeV1,
)
from solar_apps.frontends.app_v1.schema_form import SchemaForm
from solar_apps.frontends.app_v1.workflow_builder import WorkflowBuilder
from solar_apps.platform.layout import RuntimeLayout


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_schema_form_exposes_every_declared_parameter() -> None:
    application = _application()
    function = DEFAULT_FUNCTION_CATALOG.get("dart-spectrum")
    form = SchemaForm(allowed_roots=("/data",))
    form.set_function(function)
    form._editors["input_dir"].setText("/data/dart")

    assert set(form._editors) == {item.parameter_id for item in function.parameters}
    assert form.values()["dpi"] == 150
    assert form.findChildren(type(form.search))
    assert application is QApplication.instance()


def test_workflow_builder_loads_edits_and_restores_ten_page_template(
    tmp_path: Path,
    monkeypatch,
) -> None:
    application = _application()
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    layout = RuntimeLayout.discover(
        tmp_path / "repo",
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )
    builder = WorkflowBuilder(layout)
    flow = page_template("source-map")
    builder.load_flow(flow)
    builder.add_node("artifact-input", 50, 100)

    assert len(builder.flow.nodes) == 4
    assert builder.scene.items()
    assert builder.flow.schema_version == 1
    builder.undo()
    assert len(builder.flow.nodes) == 3
    builder.redo()
    assert len(builder.flow.nodes) == 4
    builder.shutdown()
    assert application is QApplication.instance()


def test_flow_executor_keeps_independent_branch_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    application = _application()
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    good = tmp_path / "good.png"
    good.write_bytes(b"not-decoded-by-worker")
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame.png").write_bytes(b"frame")
    layout = RuntimeLayout.discover()
    controller = FlowExecutionController(layout, DEFAULT_FUNCTION_CATALOG)
    flow = AppV1FlowV1(
        "branch-test",
        "Branch Test",
        nodes=(
            FlowNodeV1(
                "good",
                "artifact-input",
                {"path": str(good), "artifact_type": "image"},
            ),
            FlowNodeV1(
                "bad",
                "artifact-input",
                {"path": str(tmp_path / "missing.png"), "artifact_type": "table"},
            ),
            FlowNodeV1(
                "independent",
                "image-discover",
                {"input_dir": str(images), "recursive": False},
            ),
            FlowNodeV1("blocked", "newkirk-diagnostics"),
        ),
        edges=(FlowEdgeV1("bad", "artifact", "blocked", "drift"),),
        concurrency=2,
    )
    loop = QEventLoop()
    result: dict[str, int] = {}
    controller.flow_finished.connect(
        lambda summary: (result.update(summary), loop.quit())
    )
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(15_000)

    controller.run(flow)
    loop.exec()
    controller.shutdown()

    assert not timer.isActive() or result
    assert result["succeeded"] == 2
    assert result["failed"] == 1
    assert result["blocked"] == 1
    assert application is QApplication.instance()
