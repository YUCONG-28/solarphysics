from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from solar_apps.platform.layout import RuntimeLayout, validate_private_output_path
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


@pytest.mark.parametrize("relative", ("", "Apps", "Python", "tools", "docs", "outputs"))
def test_runtime_layout_rejects_public_repository_destinations_before_creation(
    tmp_path: Path, relative: str
) -> None:
    repo = tmp_path / "repo"
    candidate = repo / relative
    with pytest.raises(ValueError, match="inside Local/ or outside the repository"):
        RuntimeLayout.discover(repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(candidate)})
    assert not repo.exists()


def test_runtime_layout_accepts_private_subdirectories_and_relative_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    layout = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": "Local/nested"}
    )
    assert layout.local_root == repo / "Local" / "nested"
    assert not layout.local_root.exists()
    assert layout.ensure().outputs_dir.is_dir()
    with pytest.raises(ValueError, match="inside Local/"):
        RuntimeLayout.discover(repo, environ={"SOLAR_APPS_LOCAL_ROOT": "."})


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")


def test_runtime_layout_rejects_source_aliases_and_redirected_local(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "Python"
    source.mkdir(parents=True)
    external_alias = tmp_path / "alias"
    _symlink(external_alias, source)
    with pytest.raises(ValueError, match="inside Local/"):
        RuntimeLayout.discover(
            repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(external_alias)}
        )
    _symlink(repo / "Local", source)
    with pytest.raises(ValueError, match="inside Local/"):
        RuntimeLayout.discover(repo, environ={})
    assert list(source.iterdir()) == []


def test_runtime_layout_accepts_local_symlink_to_external_private_directory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    _symlink(repo / "Local", private)
    layout = RuntimeLayout.discover(repo, environ={}).ensure()
    assert layout.local_root == private.resolve()
    assert layout.state_dir.is_dir()


def test_runtime_layout_revalidates_all_directories_before_mkdir(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    layout = RuntimeLayout.discover(repo, environ={})
    altered = replace(layout, logs_dir=repo / "Apps" / "logs")
    with pytest.raises(ValueError, match="inside Local/"):
        altered.ensure()
    assert not repo.exists()


def test_runtime_layout_rejects_child_symlink_added_after_discovery(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "Apps"
    source.mkdir(parents=True)
    layout = RuntimeLayout.discover(repo, environ={})
    layout.local_root.mkdir()
    _symlink(layout.outputs_dir, source)
    with pytest.raises(ValueError, match="inside Local/"):
        layout.ensure()
    assert not layout.config_dir.exists()
    assert not layout.state_dir.exists()
    assert list(source.iterdir()) == []


def test_explicit_repository_does_not_require_a_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solar_apps.platform import layout as layout_module

    monkeypatch.setattr(
        layout_module, "__file__", str(tmp_path / "site-packages" / "layout.py")
    )
    repo = tmp_path / "workspace"
    private = tmp_path / "runtime"
    layout = RuntimeLayout.discover(
        environ={
            "SOLAR_APPS_REPO_ROOT": str(repo),
            "SOLAR_APPS_LOCAL_ROOT": str(private),
        }
    ).ensure()
    assert layout.local_root == private.resolve()
    assert not repo.exists()
    assert layout.outputs_dir.is_dir()


def test_private_output_validation_follows_nested_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "tools"
    source.mkdir(parents=True)
    local = repo / "Local"
    local.mkdir()
    _symlink(local / "redirect", source)
    with pytest.raises(ValueError, match="inside Local/"):
        validate_private_output_path(
            local / "redirect" / "new" / "state.json", repo_root=repo
        )
    assert not (source / "new").exists()


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
