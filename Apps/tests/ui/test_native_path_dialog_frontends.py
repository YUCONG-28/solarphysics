"""Native path routes are installed on every active Flask frontend."""

from __future__ import annotations

from pathlib import Path

import pytest

from solar_apps.frontends.radio_bad_frame_review.server import (
    create_app as create_bad_frame_app,
)
from solar_apps.frontends.image_viewer.server import (
    create_app as create_image_viewer_app,
)
from solar_apps.frontends.radio.source_map.server import (
    create_app as create_source_map_app,
)
from solar_apps.frontends.workbench.server import create_app as create_workbench_app

APPS_ROOT = Path(__file__).resolve().parents[2]


class _Selection:
    status = "cancelled"
    paths: tuple[Path, ...] = ()

    def to_dict(self):
        return {"ok": True, "status": self.status, "paths": []}


class _Service:
    supported = True

    def select(self, _payload):
        return _Selection()


def _assert_dialog_route(app) -> None:
    client = app.test_client()
    config = client.get("/api/native-path-dialog")
    assert config.status_code == 200
    token = config.get_json()["token"]
    response = client.post(
        "/api/native-path-dialog",
        json={"mode": "select_directory"},
        headers={"X-Native-Dialog-Token": token},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "cancelled"
    assert client.get("/api/native-path-dialog/client.js").status_code == 200


def test_all_flask_frontends_register_native_dialog_route(tmp_path: Path) -> None:
    service = _Service()
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    data_root.mkdir()
    output_root.mkdir()

    workbench = create_workbench_app(
        [data_root],
        repo_root=APPS_ROOT.parent,
        radio_output_root=output_root,
        stop_on_client_close=False,
        native_dialog_service=service,
    )
    try:
        apps = [
            workbench,
            create_image_viewer_app(
                [data_root],
                stop_on_client_close=False,
                native_dialog_service=service,
            ),
            create_bad_frame_app(
                [data_root],
                output_root=output_root,
                stop_on_client_close=False,
                native_dialog_service=service,
            ),
            create_source_map_app(
                [data_root],
                stop_on_client_close=False,
                native_dialog_service=service,
            ),
        ]
        for app in apps:
            _assert_dialog_route(app)
    finally:
        workbench.extensions["radio_workspace"]["close"]()


def test_shared_native_dialog_client_keeps_cancel_empty_and_deduplicates_windows_paths() -> (
    None
):
    source = (
        APPS_ROOT / "solar_apps" / "ui" / "media" / "native_path_dialog.js"
    ).read_text(encoding="utf-8")
    assert 'payload.status === "selected" ? (payload.paths || []) : []' in source
    assert 'replace(/\\//g, "\\\\")' in source
    assert ".toLocaleLowerCase()" in source
    assert "if (value && !seen.has(key))" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "solar_apps/frontends/workbench/static/main.js",
        "solar_apps/frontends/workbench/static/radio.js",
        "solar_apps/frontends/image_viewer/static/main.js",
        "solar_apps/frontends/radio/source_map/static/app.js",
    ],
)
def test_flask_path_buttons_use_native_selection_without_automatic_browser_fallback(
    relative_path: str,
) -> None:
    source = (APPS_ROOT / relative_path).read_text(encoding="utf-8")
    assert "SolarNativePathDialog.select" in source
    native_block = source[source.index("SolarNativePathDialog.select") :]
    catch_index = native_block.find("catch")
    assert catch_index >= 0
    assert "operation:" in native_block[:catch_index]
    assert "field:" in native_block[:catch_index]
    assert "/api/files/list" not in native_block[: catch_index + 200]


def test_bad_frame_review_root_picker_uses_shared_native_directory_contract() -> None:
    source = (
        APPS_ROOT
        / "solar_apps"
        / "frontends"
        / "radio_bad_frame_review"
        / "static"
        / "app.js"
    ).read_text(encoding="utf-8")
    start = source.index("async function openNativeRootDialog()")
    end = source.index("\nasync function loadDirectories", start)
    native_picker = source[start:end]

    assert 'mode: "select_directory"' in native_picker
    assert 'initialPath: $("#root-input").value' in native_picker
    assert 'operation: "scan-observation"' in native_picker
    assert 'field: "root-input"' in native_picker
    cancel_index = native_picker.index("if (!paths.length)")
    update_index = native_picker.index('$("#root-input").value = paths[0]')
    assert cancel_index < update_index
    assert native_picker.index("await discoverBands()") > update_index
    assert "await openInAppRootDialog(error)" in native_picker

    fallback_start = source.index("async function openInAppRootDialog(error)")
    fallback_end = source.index("\nasync function loadDirectories", fallback_start)
    fallback = source[fallback_start:fallback_end]
    assert "Windows folder dialog unavailable:" in fallback
    assert "dialog.showModal()" in fallback
    assert 'await loadDirectories($("#root-input").value.trim())' in fallback
