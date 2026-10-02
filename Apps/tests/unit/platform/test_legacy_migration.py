from __future__ import annotations

import json
from pathlib import Path

import yaml

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.migration import RUNTIME_LAYOUT_VERSION, migrate_legacy_state
from solar_apps.platform.state import StateStore


def _layout(tmp_path: Path) -> RuntimeLayout:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    return RuntimeLayout.discover(repo, environ={}).ensure()


def test_legacy_config_and_home_settings_are_allow_listed_and_sources_unchanged(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    backup = layout.repo_root / "Local-migration-backup"
    source_config = backup / "configs" / "paths.local.yaml"
    source_config.parent.mkdir(parents=True)
    data_root = tmp_path / "observations"
    data_root.mkdir()
    old_output = layout.repo_root / "Local" / "outputs" / "radio"
    source_config.write_text(
        yaml.safe_dump(
            {
                "apps": {"allowed_roots": [str(data_root), str(layout.repo_root)]},
                "scripts": {"radio_source_map_plot": {"output_dir": str(old_output)}},
                "private_unknown": {"must_not_import": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    source_settings = home / ".solar_toolkit" / "radio_roi_lightcurve_app_settings.json"
    source_settings.parent.mkdir(parents=True)
    source_settings.write_text(
        json.dumps(
            {
                "metric": "mean",
                "radio_dir": str(data_root),
                "output_dir": str(tmp_path / "outside"),
                "operation_history": ["must not import"],
            }
        ),
        encoding="utf-8",
    )
    original_config = source_config.read_bytes()
    original_settings = source_settings.read_bytes()

    created = migrate_legacy_state(layout=layout, home=home, backup_root=backup)

    assert layout.config_path in created
    imported_config = yaml.safe_load(layout.config_path.read_text(encoding="utf-8"))
    assert "private_unknown" not in imported_config
    assert imported_config["apps"]["allowed_roots"] == [str(data_root.resolve())]
    assert imported_config["apps"]["runtime_layout_version"] == RUNTIME_LAYOUT_VERSION
    assert imported_config["scripts"]["radio_source_map_plot"]["output_dir"] == str(
        layout.outputs_dir / "radio"
    )
    state_path = layout.state_dir / "roi-lightcurve.json"
    assert state_path in created
    state = StateStore(
        state_path,
        "roi-lightcurve",
        allowed_keys=("fields", "theme", "legacy_imported"),
    ).load()
    assert state == {
        "fields": {"metric": "mean", "radio_dir": str(data_root)},
        "legacy_imported": True,
        "theme": "auto",
    }
    assert source_config.read_bytes() == original_config
    assert source_settings.read_bytes() == original_settings


def test_valid_existing_latest_state_is_never_overwritten(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    backup = layout.repo_root / "Local-migration-backup"
    (backup / "configs").mkdir(parents=True)
    (backup / "configs" / "paths.local.yaml").write_text(
        "apps:\n  allowed_roots: []\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    source = home / ".solar_toolkit" / "radio_source_trajectory_app.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"theme_mode": "dark", "fps": 20}), encoding="utf-8")
    target = layout.state_dir / "source-trajectory.json"
    store = StateStore(
        target,
        "source-trajectory",
        allowed_keys=("fields", "theme", "legacy_imported"),
    )
    expected = {"fields": {"fps": 30}, "theme": "light", "legacy_imported": True}
    store.save(expected)

    migrate_legacy_state(layout=layout, home=home, backup_root=backup)

    assert store.load() == expected


def test_existing_legacy_config_is_atomically_mapped_and_revalidated(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    valid_root = tmp_path / "observations"
    valid_root.mkdir()
    layout.config_path.write_text(
        yaml.safe_dump(
            {
                "apps": {"allowed_roots": [str(valid_root), str(layout.repo_root)]},
                "scripts": {
                    "radio_source_map_plot": {
                        "output_dir": str(
                            layout.repo_root / "Local" / "outputs" / "radio"
                        )
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    changed = migrate_legacy_state(
        layout=layout,
        home=tmp_path / "empty-home",
        backup_root=tmp_path / "missing-backup",
    )

    assert layout.config_path in changed
    imported = yaml.safe_load(layout.config_path.read_text(encoding="utf-8"))
    assert imported["apps"] == {
        "allowed_roots": [str(valid_root.resolve())],
        "runtime_layout_version": RUNTIME_LAYOUT_VERSION,
    }
    assert imported["scripts"]["radio_source_map_plot"]["output_dir"] == str(
        layout.outputs_dir / "radio"
    )
    assert not list(layout.config_dir.glob("*.tmp"))


def test_marked_current_config_is_never_rewritten(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    content = (
        "apps:\n"
        f"  runtime_layout_version: {RUNTIME_LAYOUT_VERSION}\n"
        "  allowed_roots: []\n"
        "scripts:\n"
        "  preserved: true\n"
    )
    layout.config_path.write_text(content, encoding="utf-8")

    changed = migrate_legacy_state(
        layout=layout,
        home=tmp_path / "empty-home",
        backup_root=tmp_path / "missing-backup",
    )

    assert layout.config_path not in changed
    assert layout.config_path.read_text(encoding="utf-8") == content


def test_nested_secrets_history_results_data_and_unknown_workflows_are_dropped(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    backup = layout.repo_root / "Local-migration-backup"
    source = backup / "configs" / "paths.local.yaml"
    source.parent.mkdir(parents=True)
    allowed = tmp_path / "observations"
    allowed.mkdir()
    source.write_text(
        yaml.safe_dump(
            {
                "apps": {
                    "allowed_roots": [str(allowed)],
                    "api_token": "do-not-copy",
                    "unknown": True,
                },
                "scripts": {
                    "sdo_aia_euv_processor": {
                        "data_path": str(allowed),
                        "output_dir": str(allowed / "output"),
                        "password": "do-not-copy",
                        "operation_history": ["do-not-copy"],
                        "unknown": "do-not-copy",
                    },
                    "radio_20000503_config": {
                        "output": {
                            "output_dir": str(allowed / "radio"),
                            "auth_token": "do-not-copy",
                        },
                        "user": {
                            "data": {"data_dir": str(allowed)},
                            "output": {
                                "output_dir": str(allowed / "user"),
                                "cookie": "do-not-copy",
                            },
                            "result": {"payload": "do-not-copy"},
                        },
                        "timestamp": "do-not-copy",
                    },
                    "unknown_workflow": {"output_dir": str(allowed / "unknown")},
                },
                "auth": {"secret": "do-not-copy"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    original = source.read_bytes()

    migrate_legacy_state(
        layout=layout,
        home=tmp_path / "empty-home",
        backup_root=backup,
    )

    imported = yaml.safe_load(layout.config_path.read_text(encoding="utf-8"))
    assert imported["apps"] == {
        "allowed_roots": [str(allowed.resolve())],
        "runtime_layout_version": RUNTIME_LAYOUT_VERSION,
    }
    assert imported["scripts"] == {
        "sdo_aia_euv_processor": {
            "data_path": str(allowed),
            "output_dir": str(allowed / "output"),
        },
    }

    def nested_keys(value: object) -> list[str]:
        if not isinstance(value, dict):
            return []
        return [
            *(str(key).casefold() for key in value),
            *(child for nested in value.values() for child in nested_keys(nested)),
        ]

    imported_keys = nested_keys(imported)
    for forbidden in (
        "token",
        "secret",
        "password",
        "cookie",
        "auth",
        "history",
        "timestamp",
        "task_id",
        "result",
    ):
        assert not any(forbidden in key for key in imported_keys)
    assert "data" not in imported_keys
    assert source.read_bytes() == original
