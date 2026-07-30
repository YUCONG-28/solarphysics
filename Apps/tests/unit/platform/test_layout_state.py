from __future__ import annotations

import json
from pathlib import Path

import pytest

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.state import StateStore


def test_runtime_layout_defaults_to_repo_local_and_supports_override(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    default = RuntimeLayout.discover(repo, environ={})
    assert default.local_root == repo / "Local"
    private = tmp_path / "private"
    overridden = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(private)}
    ).ensure()
    assert overridden.local_root == private.resolve()
    assert overridden.config_path == private.resolve() / "configs" / "paths.local.yaml"
    assert all(
        path.is_dir()
        for path in (
            overridden.state_dir,
            overridden.workspaces_dir,
            overridden.outputs_dir,
            overridden.observations_dir,
            overridden.logs_dir,
            overridden.tmp_dir,
        )
    )


def test_runtime_layout_accepts_explicit_repository_environment(tmp_path: Path) -> None:
    repo = tmp_path / "relocated-workspace"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        environ={
            "SOLAR_APPS_REPO_ROOT": str(repo),
            "SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "runtime"),
        }
    )
    assert layout.repo_root == repo.resolve()
    assert layout.local_root == (tmp_path / "runtime").resolve()


def test_state_store_is_versioned_atomic_and_latest_only(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ui.json"
    store = StateStore(
        path,
        "frontend",
        allowed_keys=("theme", "fields"),
    )
    assert store.load({"theme": "auto"}) == {"theme": "auto"}
    store.save({"theme": "dark", "fields": {"input_path": "example.fits"}})
    store.update({"theme": "light"})
    assert store.load() == {
        "theme": "light",
        "fields": {"input_path": "example.fits"},
    }
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["namespace"] == "frontend"
    assert "history" not in payload
    assert not list(path.parent.glob("*.tmp"))


def test_state_store_requires_an_explicit_nonempty_allow_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-list"):
        StateStore(tmp_path / "state.json", "frontend", allowed_keys=())


@pytest.mark.parametrize(
    "invalid",
    (
        {"unexpected": True},
        {"fields": {"history": []}},
        {"fields": {"timestamp": "now"}},
        {"fields": {"task_id": "secret"}},
        {"fields": {"result": [1, 2, 3]}},
    ),
)
def test_state_store_rejects_non_ui_or_historical_data(
    tmp_path: Path, invalid: dict
) -> None:
    store = StateStore(
        tmp_path / "state.json",
        "frontend",
        allowed_keys=("theme", "fields"),
    )
    with pytest.raises(ValueError):
        store.save(invalid)


def test_state_store_bad_or_wrong_version_state_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path, "frontend", allowed_keys=("theme",))
    path.write_text("not json", encoding="utf-8")
    assert store.load({"theme": "auto"}) == {"theme": "auto"}
    path.write_text(
        json.dumps({"schema_version": 99, "namespace": "frontend", "data": {}}),
        encoding="utf-8",
    )
    assert store.load({"theme": "auto"}) == {"theme": "auto"}
