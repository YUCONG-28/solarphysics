"""Import, license, runtime, and JSON contracts for Solar Physics App 1.0."""

from __future__ import annotations

import ast
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from solar_apps.frontends.app_v1 import (
    MODULES,
    AppV1ProjectV1,
    AppV1RuntimePaths,
    ArtifactManifestV1,
    ArtifactProduct,
    InputReference,
    RunRequest,
    RunStatus,
    SyncSelection,
    TimelineSource,
    WorkerEventV1,
)
from solar_apps.platform.layout import RuntimeLayout

APPS_ROOT = Path(__file__).resolve().parents[2]
APP_V1_ROOT = APPS_ROOT / "solar_apps" / "frontends" / "app_v1"


def test_all_eleven_interfaces_have_unique_implementation_phases() -> None:
    assert len(MODULES) == 11
    assert len({module.module_id for module in MODULES}) == 11
    assert {module.target_phase for module in MODULES} <= {
        "1",
        "2A",
        "2B",
        "2C",
        "3",
        "4",
        "5",
    }
    assert all(
        module.legacy_interface
        for module in MODULES
        if module.module_id not in {"data-download", "image-composer"}
    )
    assert all(
        next(
            module for module in MODULES if module.module_id == module_id
        ).legacy_interface
        is None
        for module_id in ("data-download", "image-composer")
    )


def test_app_v1_import_safe_modules_import_no_qt_binding() -> None:
    forbidden = {"PyQt5", "PyQt6", "PySide6"}
    offenders: list[str] = []
    import_safe_paths = (
        APP_V1_ROOT / "__init__.py",
        APP_V1_ROOT / "catalog.py",
        APP_V1_ROOT / "contracts.py",
        APP_V1_ROOT / "runtime.py",
        APP_V1_ROOT / "cli.py",
    )
    for path in import_safe_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        if imported & forbidden:
            offenders.append(path.name)
    assert offenders == []


def test_pyqt6_is_declared_and_app_v1_has_scoped_gpl_notice() -> None:
    project = tomllib.loads((APPS_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert "PyQt6==6.9.1" in dependencies
    assert "PyQt6-Qt6==6.9.1" in dependencies
    assert not any(
        dependency.casefold().startswith(("pyside6", "shiboken6"))
        for dependency in dependencies
    )
    assert project["project"]["license"] == "MIT AND GPL-3.0-only AND MPL-2.0"
    assert project["build-system"]["requires"][0] == "setuptools>=77"
    license_files = set(project["project"]["license-files"])
    assert {
        "LICENSE",
        "LICENSES/GPL-3.0-only.txt",
        "solar_apps/frontends/app_v1/LICENSE.md",
        "solar_apps/ui/media/NOTICE.txt",
        "solar_apps/ui/media/mediabunny-MPL-2.0.txt",
    } <= license_files
    assert (
        (APPS_ROOT / "LICENSES" / "GPL-3.0-only.txt")
        .read_text(encoding="utf-8")
        .startswith("                    GNU GENERAL PUBLIC LICENSE")
    )
    license_text = (APP_V1_ROOT / "LICENSE.md").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: GPL-3.0-only" in license_text


def test_runtime_paths_extend_only_the_existing_local_layout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "private")}
    )
    paths = AppV1RuntimePaths.from_layout(layout).ensure()

    assert paths.outputs_dir == layout.outputs_dir / "app_v1"
    assert paths.observations_dir == layout.observations_dir
    assert paths.observations_dir.is_dir()
    assert paths.project_file("event-one").name == "event-one.spapp.json"
    assert paths.time_index_path == layout.state_dir / "app_v1" / "time_index.sqlite3"
    assert paths.run_output_dir("event-one", "run-one", "source-map").is_relative_to(
        layout.outputs_dir
    )
    with pytest.raises(ValueError):
        paths.project_file("../escape")


def test_versioned_contracts_are_utc_and_json_compatible() -> None:
    observed = datetime(2025, 1, 24, 4, 48, tzinfo=timezone.utc)
    source = InputReference(
        "radio-one",
        "radio-fits",
        "allowed-root://event/radio.fits",
        observed_at_utc=observed,
        metadata={"frequency_mhz": 149},
    )
    request = RunRequest(
        "run-one",
        "event-one",
        "source-map",
        inputs=(source,),
        parameters={"colormap": "hot"},
        requested_at_utc=observed,
    )
    manifest = ArtifactManifestV1(
        "event-one",
        "run-one",
        "source-map",
        RunStatus.SUCCEEDED,
        (source,),
        request.parameters,
        (ArtifactProduct("image", "images/map.png", "image/png"),),
        {"app": "1.0"},
        created_at_utc=observed,
        time_start_utc=observed,
        time_end_utc=observed,
    )
    timeline = TimelineSource(
        "radio-source", "source-map", (observed,), tolerance_seconds=2.0
    )
    selection = SyncSelection(
        "radio-source",
        observed,
        {"radio-source": "allowed-root://event/radio.fits"},
    )
    project = AppV1ProjectV1(
        "event-one",
        "Event One",
        modules=("source-map",),
        parameters={"source-map": request.parameters},
        timeline={"sources": [timeline.to_dict()]},
        layout={"active_module": "source-map"},
        artifact_manifests=("outputs/run-one/source-map/manifest.json",),
        saved_at_utc=observed,
    )

    assert request.to_dict()["requested_at_utc"].endswith("Z")
    assert manifest.to_dict()["products"][0]["relative_path"] == "images/map.png"
    assert selection.to_dict()["current_time_utc"] == "2025-01-24T04:48:00Z"
    assert project.to_dict()["schema_version"] == 1


def test_contracts_reject_unsafe_or_ambiguous_values() -> None:
    observed = datetime(2025, 1, 24, 4, 48, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="run-relative"):
        ArtifactProduct("image", "../map.png", "image/png")
    with pytest.raises(ValueError, match="terminal"):
        ArtifactManifestV1(
            "event-one",
            "run-one",
            "source-map",
            RunStatus.RUNNING,
            (),
            {},
            (),
            {},
            created_at_utc=observed,
        )
    with pytest.raises(ValueError, match="UTC offset"):
        TimelineSource(
            "radio-source",
            "source-map",
            (datetime(2025, 1, 24, 4, 48),),
        )


def test_worker_event_protocol_accepts_only_finite_version_one_json() -> None:
    event = WorkerEventV1(
        "run-one",
        "source-map",
        "preview",
        {"path": "outputs/source-map.png", "percent": 50},
    )

    assert event.to_dict()["kind"] == "preview"
    with pytest.raises(ValueError, match="Unsupported worker event kind"):
        WorkerEventV1("run-one", "source-map", "unknown", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite JSON"):
        WorkerEventV1("run-one", "source-map", "progress", {"percent": float("nan")})
