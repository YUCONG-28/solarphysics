"""App 1.0 project and preset persistence contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from solar_apps.frontends.app_v1.contracts import AppV1ProjectV1
from solar_apps.frontends.app_v1.project_store import AppV1ProjectStore
from solar_apps.frontends.app_v1.runtime import AppV1RuntimePaths
from solar_apps.platform.layout import RuntimeLayout


def _store(tmp_path: Path) -> AppV1ProjectStore:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )
    return AppV1ProjectStore(AppV1RuntimePaths.from_layout(layout))


def test_project_round_trip_is_versioned_and_contains_references_only(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    manifest = (
        store.runtime.outputs_dir
        / "campaign"
        / "run-1"
        / "source-map"
        / "manifest.json"
    )
    project = AppV1ProjectV1(
        project_id="campaign",
        name="Campaign",
        modules=("source-map", "image-composer"),
        parameters={"source-map": {"threshold_sigma": 6.0}},
        timeline={"schema_version": 1, "sources": []},
        layout={"dock_state_base64": "AA=="},
        artifact_manifests=(str(manifest),),
        saved_at_utc=datetime(2026, 7, 24, 12, tzinfo=timezone.utc),
    )

    target = store.save(project)
    restored = store.load("campaign")
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert target == store.runtime.workspaces_dir / "campaign.spapp.json"
    assert restored == store.from_dict(payload)
    assert restored.artifact_manifests == ("campaign/run-1/source-map/manifest.json",)
    assert "observation" not in payload
    assert not tuple(target.parent.glob(f".{target.name}.*.tmp"))


def test_project_rejects_artifact_references_outside_private_outputs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    project = AppV1ProjectV1(
        project_id="campaign",
        name="Campaign",
        artifact_manifests=(str(tmp_path / "raw" / "manifest.json"),),
    )

    with pytest.raises(ValueError, match="Local/outputs/app_v1"):
        store.save(project)


def test_parameter_presets_are_module_scoped_and_atomic(tmp_path: Path) -> None:
    store = _store(tmp_path)

    target = store.save_preset(
        "source-map",
        "publication",
        {"threshold_sigma": 6.0, "contours": [50, 70, 90]},
    )

    assert store.load_preset("source-map", "publication") == {
        "threshold_sigma": 6.0,
        "contours": [50, 70, 90],
    }
    assert target.name == "publication.json"
    assert not tuple(target.parent.glob(f".{target.name}.*.tmp"))
