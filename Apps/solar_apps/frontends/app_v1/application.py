# SPDX-License-Identifier: GPL-3.0-only
"""PyQt6 application lifecycle for Solar Physics App 1.0."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from .components import RunConfirmationDialog
from .catalog import MODULES

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths import AllowedRootPolicyError, configured_allowed_roots

from .window import AppV1MainWindow

SMOKE_PREFIX = "APP_V1_SMOKE "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Apps/run.ps1 frontend app-v1",
        description="Launch the native Solar Physics App 1.0.",
    )
    parser.add_argument(
        "--theme",
        choices=("auto", "light", "dark", "dark-dimmed"),
        default="auto",
        help="Initial application chrome theme.",
    )
    parser.add_argument(
        "--auto-close-ms",
        type=int,
        default=0,
        help="Close automatically after this many milliseconds (smoke tests).",
    )
    parser.add_argument(
        "--smoke-test",
        choices=(
            "basic",
            "cancel",
            "dialog",
            "timeline",
            "project",
            "workflow",
            "recovery",
            "data-download",
        ),
        help="Run an offscreen lifecycle check and print a JSON result.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create the window without showing it.",
    )
    parser.add_argument(
        "--module",
        metavar="MODULE_ID",
        choices=tuple(module.module_id for module in MODULES),
        help="Select an App 1.0 module at startup.",
    )
    parser.add_argument(
        "--composer-project",
        type=Path,
        default=None,
        help="Open a schema-1 .fic.json project in the Image Composer.",
    )
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help=(
            "Path-separated directories available to the Image Composer. "
            "Defaults to the private Local configuration."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.auto_close_ms < 0:
        raise ValueError("--auto-close-ms cannot be negative")
    if args.composer_project is not None and args.module not in {
        None,
        "image-composer",
    }:
        print(
            "app-v1: error: --composer-project requires --module image-composer",
            file=sys.stderr,
        )
        return 2
    initial_module = "image-composer" if args.composer_project else args.module
    composer_roots: tuple[Path, ...] | None = None
    if initial_module == "image-composer" or args.allowed_roots is not None:
        try:
            configured_roots = configured_allowed_roots(cli_value=args.allowed_roots)
        except AllowedRootPolicyError as exc:
            print(f"Invalid allowed-root configuration: {exc}", file=sys.stderr)
            return 2
        if not configured_roots:
            print(
                "No application allowed roots are configured. Add apps.allowed_roots "
                "to the private Local config or pass --allowed-roots with "
                f"{os.pathsep!r} separators.",
                file=sys.stderr,
            )
            return 2
        composer_roots = tuple(configured_roots)
    application = QApplication.instance() or QApplication(["solar-physics-app-v1"])
    application.setApplicationName("Solar Physics App 1.0")
    application.setOrganizationName("solarphysics")
    layout = RuntimeLayout.discover().ensure()
    window = AppV1MainWindow(
        layout,
        initial_theme=args.theme,
        initial_module=initial_module or "workbench",
        composer_allowed_roots=composer_roots,
    )
    if args.composer_project is not None:
        if not window.load_composer_project(args.composer_project):
            return 2
    if not args.no_show:
        window.show()

    smoke: dict[str, object] = {}
    if args.smoke_test == "basic":
        _schedule_basic_smoke(application, window, smoke)
    elif args.smoke_test == "cancel":
        _schedule_cancel_smoke(application, window, smoke)
    elif args.smoke_test == "dialog":
        _schedule_dialog_smoke(application, window, smoke)
    elif args.smoke_test == "timeline":
        _schedule_timeline_smoke(application, window, smoke)
    elif args.smoke_test == "project":
        _schedule_project_smoke(application, window, smoke)
    elif args.smoke_test == "workflow":
        _schedule_workflow_smoke(application, window, smoke)
    elif args.smoke_test == "recovery":
        _schedule_recovery_smoke(application, window, smoke)
    elif args.smoke_test == "data-download":
        _schedule_data_download_smoke(application, window, smoke)
    elif args.auto_close_ms:
        QTimer.singleShot(args.auto_close_ms, window.close)
        QTimer.singleShot(args.auto_close_ms + 100, application.quit)

    exit_code = application.exec()
    if args.smoke_test:
        smoke.update(
            {
                "exit_code": int(exit_code),
                "platform": application.platformName(),
                "registered_modules": list(window.registered_module_ids),
                "foreign_qt_loaded": any(
                    name == "PyQt5"
                    or name.startswith("PyQt5.")
                    or name == "PySide6"
                    or name.startswith("PySide6.")
                    for name in sys.modules
                ),
                "forbidden_frontend_modules": [
                    name
                    for name in ("flask", "streamlit", "matplotlib", "PySide6")
                    if name in sys.modules
                ],
                "process_running": window.task_controller.process_running,
                "running_qprocess_count": (
                    window.task_controller.running_process_count
                ),
            }
        )
        print(SMOKE_PREFIX + json.dumps(smoke, sort_keys=True), flush=True)
    return int(exit_code)


def _schedule_basic_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    def run() -> None:
        modes: dict[str, str] = {}
        for mode in ("auto", "light", "dark", "dark_dimmed"):
            modes[mode] = window.set_theme(mode)
        result["themes"] = modes
        result["dock_count"] = len(window.findChildren(type(window.parameter_dock)))
        window.close()
        application.quit()

    QTimer.singleShot(50, run)
    QTimer.singleShot(8000, application.quit)


def _schedule_cancel_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    record = window.enqueue_demo_task("workbench", steps=200, delay_ms=20)
    cancel_scheduled = False

    def changed(task_id: str) -> None:
        nonlocal cancel_scheduled
        if task_id != record.task_id:
            return
        current = window.task_controller.task(task_id)
        if current.status == "running" and not cancel_scheduled:
            cancel_scheduled = True
            QTimer.singleShot(120, lambda: window.task_controller.cancel(task_id))
        if current.status in {"cancelled", "failed", "succeeded"}:
            result["task_status"] = current.status
            result["return_code"] = current.return_code
            QTimer.singleShot(0, window.close)
            QTimer.singleShot(20, application.quit)

    window.task_controller.task_changed.connect(changed)
    QTimer.singleShot(10000, application.quit)


def _schedule_dialog_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    summary = window.demo_confirmation_summary("workbench")
    result["summary_fields"] = [
        label
        for label in ("Module:", "Input:", "Parameters:", "Output:", "Workload:")
        if label in summary
    ]

    def accept_dialog() -> None:
        modal = application.activeModalWidget()
        if isinstance(modal, RunConfirmationDialog):
            modal.reject()

    def show_dialog() -> None:
        QTimer.singleShot(150, accept_dialog)
        accepted = window.confirm_and_enqueue_demo("workbench")
        result["accepted"] = accepted is not None
        window.close()
        application.quit()

    QTimer.singleShot(50, show_dialog)
    QTimer.singleShot(8000, application.quit)


def _schedule_timeline_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    def run() -> None:
        result["source_count"] = len(window.time_coordinator.sources)
        if window.time_coordinator.sources:
            selection = window.time_coordinator.step(1)
            result["current_time_utc"] = selection.current_time_utc.isoformat().replace(
                "+00:00", "Z"
            )
            result["matched_count"] = sum(
                locator is not None for locator in selection.matched_locators.values()
            )
            result["synced_page_count"] = sum(
                page.time_status is not None
                and result["current_time_utc"] in page.time_status.text()
                for page in window.module_pages.values()
            )
        window.close()
        application.quit()

    QTimer.singleShot(50, run)
    QTimer.singleShot(8000, application.quit)


def _schedule_project_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    def run() -> None:
        window.project_panel.project_id.setText("smoke-project")
        window.project_panel.project_name.setText("Smoke Project")
        window.parameter_document.setPlainText('{"cadence_seconds": 2.5}')
        window.project_panel.preset_id.setText("smoke-preset")
        preset_path = window.save_current_preset()
        project_path = window.save_current_project()
        window.parameter_document.setPlainText("{}")
        restored = window.load_current_project()
        result["project_file"] = project_path.name
        result["preset_file"] = preset_path.name
        result["restored_project_id"] = restored.project_id
        result["restored_parameter"] = window.project_parameters["workbench"][
            "cadence_seconds"
        ]
        result["artifact_count"] = len(restored.artifact_manifests)
        window.close()
        application.quit()

    QTimer.singleShot(50, run)
    QTimer.singleShot(8000, application.quit)


def _schedule_workflow_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    redraw_started = False

    def idle() -> None:
        nonlocal redraw_started
        records = window.task_controller.records
        if not redraw_started:
            redraw_started = True
            redrawn = window.redraw_last_task()
            result["redraw_created"] = redrawn is not None
            return
        result["modules"] = [record.module_id for record in records]
        result["statuses"] = [record.status for record in records]
        result["task_count"] = len(records)
        window.close()
        application.quit()

    window.task_controller.queue_idle.connect(idle)
    worker = "solar_apps.frontends.app_v1.task_worker"
    launches = (
        SimpleNamespace(
            title="Batch task one",
            module_id="workbench",
            python_module=worker,
            arguments=("--steps", "2", "--delay-ms", "0"),
            output_dir=None,
        ),
        SimpleNamespace(
            title="Batch task two",
            module_id="radio-workspace",
            python_module=worker,
            arguments=("--steps", "2", "--delay-ms", "0"),
            output_dir=None,
        ),
    )
    window.task_controller.enqueue_batch(launches)
    QTimer.singleShot(10000, application.quit)


def _schedule_recovery_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    recovery_started = False
    marker = window.runtime_paths.tmp_dir / f"recovery-{uuid.uuid4().hex}.marker"
    original = window.task_controller.enqueue_python_module(
        title="Recoverable task",
        module_id="workbench",
        python_module="solar_apps.frontends.app_v1.task_worker",
        arguments=(
            "--steps",
            "2",
            "--delay-ms",
            "0",
            "--fail-once-marker",
            str(marker),
        ),
    )

    def changed(task_id: str) -> None:
        nonlocal recovery_started
        record = window.task_controller.task(task_id)
        if task_id == original.task_id and record.status == "failed":
            recovered = window.recover_failed_tasks()
            recovery_started = True
            result["retry_count"] = len(recovered)
            result["retry_of"] = recovered[0].retry_of if recovered else None
            return
        if (
            recovery_started
            and record.retry_of == original.task_id
            and record.status == "succeeded"
        ):
            result["statuses"] = [
                item.status for item in window.task_controller.records
            ]
            result["marker_created"] = marker.is_file()
            marker.unlink(missing_ok=True)
            window.close()
            application.quit()

    window.task_controller.task_changed.connect(changed)
    QTimer.singleShot(10000, application.quit)


def _schedule_data_download_smoke(
    application: QApplication,
    window: AppV1MainWindow,
    result: dict[str, object],
) -> None:
    def run() -> None:
        page = window.module_pages["data-download"]
        panel = page.native_panels[0]
        result["module_id"] = panel.module_id
        result["product_count"] = panel.product.count()
        result["default_product"] = panel.product.currentData()
        result["download_enabled"] = panel.download_button.isEnabled()
        result["observation_root"] = panel.observation_root.text()
        window.close()
        application.quit()

    QTimer.singleShot(50, run)
    QTimer.singleShot(8000, application.quit)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SMOKE_PREFIX", "build_parser", "main"]
