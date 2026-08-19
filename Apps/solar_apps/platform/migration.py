"""One-time, allow-listed import of private legacy application settings."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .layout import RuntimeLayout
from .paths.allowed_roots import AllowedRootPolicyError, normalize_allowed_roots
from .paths.native_dialog import is_path_within_roots
from .state import StateStore

_ROI_FIELDS = frozenset(
    {
        "display_bad_color",
        "display_colormap",
        "display_high_percentile",
        "display_low_percentile",
        "display_manual_max",
        "display_manual_min",
        "display_range_mode",
        "display_range_scope",
        "display_transform",
        "display_use_custom_fov",
        "display_x_max_arcsec",
        "display_x_min_arcsec",
        "display_y_max_arcsec",
        "display_y_min_arcsec",
        "metric",
        "output_dir",
        "page_size",
        "pair_time_tolerance_sec",
        "pattern",
        "polarization",
        "preview_max_side",
        "radio_dir",
        "recursive",
        "selected_freqs_mhz",
        "time_end",
        "time_start",
    }
)
_TRAJECTORY_FIELDS = frozenset(
    {
        "aia_dir",
        "aia_pattern",
        "centers",
        "compare_lr",
        "compare_tolerance_sec",
        "draw_lines",
        "facet_by",
        "fps",
        "frame_mode",
        "frequency_marker_symbols",
        "link_facet_views",
        "log_scale",
        "marker_size",
        "max_aia_dt_sec",
        "max_pixels",
        "percentile_limits",
        "playback_aia_max_pixels",
        "playback_min_step_sec",
        "playback_renderer",
        "plot_layout",
        "screen_fit",
        "selected_freqs",
        "selected_methods",
        "selected_pols",
        "show_debug_tables",
        "tail_n",
        "theme_mode",
        "time_end",
        "time_start",
        "trail_min_opacity",
        "use_aia",
        "video_browser_format",
        "video_fps",
        "video_height",
        "video_include_aia",
        "video_output_format",
        "video_output_path",
        "video_quality",
        "video_width",
        "wcs_mode",
    }
)
_LEGACY_SETTINGS = {
    "radio_roi_lightcurve_app_settings.json": ("roi-lightcurve", _ROI_FIELDS),
    "radio_source_trajectory_app.json": ("source-trajectory", _TRAJECTORY_FIELDS),
}
_PATH_FIELDS = frozenset({"aia_dir", "output_dir", "radio_dir", "video_output_path"})
RUNTIME_LAYOUT_VERSION = 2

_DANGEROUS_EXACT_KEYS = frozenset(
    {
        "data",
        "history",
        "log",
        "logs",
        "result",
        "results",
        "task_id",
        "task_ids",
        "timestamp",
        "timestamps",
    }
)
_DANGEROUS_KEY_FRAGMENTS = (
    "auth",
    "cookie",
    "history",
    "password",
    "result",
    "secret",
    "task_id",
    "timestamp",
    "token",
)

# Only machine-local settings consumed by known, still-supported workflows may
# cross the migration boundary. A set denotes scalar/list leaves; dictionaries
# describe the exact permitted nested shape.
_SCRIPT_SCHEMAS: dict[str, frozenset[str] | dict[str, Any]] = {
    "asos_hxi_goes_sxr_comparison": frozenset({"file_path"}),
    "asos_hxi_image_plot": frozenset({"file_path"}),
    "cso_radio_spectrogram_plot": frozenset({"file_path", "save_path"}),
    "cso_spectrogram_class": frozenset({"file_path"}),
    "dem_radio_source_overlay": frozenset(
        {"aia_fits_path", "radio_sources_dir", "tb_data_path"}
    ),
    "flare_aia_sxr_hxr_summary_plot": frozenset({"aia_dir", "hxi", "sxr"}),
    "goes_sxr_lightcurve_plot": frozenset({"file_path"}),
    "image_sequence_to_video": frozenset({"input_dir", "output_dir", "video_name"}),
    "neupert_sxr_derivative_hxr_comparison": frozenset({"data_file_path"}),
    "neupert_timing_error_analysis": frozenset({"file_path"}),
    "radio_source_map_plot": frozenset(
        {"data_dir", "multi_band_root", "output_dir", "single_file_path"}
    ),
    "sdo_aia_asos_hxi_overlay": frozenset(
        {"hxi_file_path", "hxi_file_path_pro", "input_dir_AIA", "output_dir"}
    ),
    "sdo_aia_base_difference": frozenset({"data_dir", "output_dir", "show_plot"}),
    "sdo_aia_dem_inversion": frozenset({"aia_fits_path", "tb_data_path"}),
    "sdo_aia_euv_processor": frozenset({"data_path", "output_dir"}),
    "sdo_aia_hmi_overlay": frozenset(
        {"input_dir_AIA", "input_dir_HMI", "output_dir", "show_plot"}
    ),
    "sdo_aia_lightcurve_extraction": frozenset(
        {"data_dir", "data_filename", "output_dir"}
    ),
    "sdo_aia_lightcurve_plot": frozenset({"input_dir", "output_dir"}),
    "sdo_aia_radio_hmi_overlay": frozenset(
        {"aia_base_dir", "hmi_base_dir", "output_dir", "radio_base_dir"}
    ),
    "sdo_aia_running_difference": frozenset({"data_dir", "output_dir", "show_plot"}),
    "sdo_aia_time_file_selector": frozenset(
        {"output_base_dir", "source_dir", "target_time", "tolerance"}
    ),
    "sdo_hmi_magnetogram_plot": frozenset({"data_dir", "output_dir", "show_plot"}),
    "soho_lasco_data_download": frozenset({"save_dir"}),
    "soho_lasco_image_plot": frozenset({"input_dir", "output_dir", "show_plot"}),
    "soho_lasco_running_difference": frozenset(
        {"input_dir", "output_dir", "show_plot"}
    ),
}

_EVENT_WORKFLOW_SCHEMA: dict[str, Any] = {
    "output": {"output_dir": None},
    "paths": {
        "aia_base_dir": None,
        "aia_panel_base_dir_template": None,
        "hmi_base_dir": None,
        "output_dir": None,
        "radio_base_dir": None,
    },
    "spectrogram": {"file_path": None, "file_paths": None},
}
_SCRIPT_SCHEMAS.update(
    {
        "radio_20250124_config": {
            "aia_multi_wave_gaussian_spectrogram": _EVENT_WORKFLOW_SCHEMA,
            "aia_radio_hmi": _EVENT_WORKFLOW_SCHEMA,
            "output": {"output_dir": None},
            "user": {
                "output": {"output_dir": None},
                "spectrogram": {"file_path": None, "file_paths": None},
            },
        },
        "radio_20250503_config": {
            "aia_multi_wave_raw_radio_spectrogram": _EVENT_WORKFLOW_SCHEMA,
            "aia_radio_hmi": _EVENT_WORKFLOW_SCHEMA,
            "aia_raw_radio_spectrogram": _EVENT_WORKFLOW_SCHEMA,
            "output": {"output_dir": None},
            "user": {
                "output": {"output_dir": None},
                "spectrogram": {"file_path": None, "file_paths": None},
            },
        },
    }
)


def _dangerous_key(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "_")
    return normalized in _DANGEROUS_EXACT_KEYS or any(
        fragment in normalized for fragment in _DANGEROUS_KEY_FRAGMENTS
    )


def _safe_leaf(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return list(value)
    return None


def _filter_known_mapping(
    raw: Mapping[str, Any],
    schema: frozenset[str] | Mapping[str, Any],
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    if isinstance(schema, frozenset):
        for key in schema:
            if key not in raw or _dangerous_key(key):
                continue
            value = _safe_leaf(raw[key])
            if value is not None or raw[key] is None:
                filtered[key] = value
        return filtered
    for key, child_schema in schema.items():
        if key not in raw or _dangerous_key(key):
            continue
        value = raw[key]
        if child_schema is None:
            safe_value = _safe_leaf(value)
            if safe_value is not None or value is None:
                filtered[key] = safe_value
        elif isinstance(value, Mapping):
            nested = _filter_known_mapping(value, child_schema)
            if nested:
                filtered[key] = nested
    return filtered


def _filter_scripts(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    filtered: dict[str, Any] = {}
    for section, schema in _SCRIPT_SCHEMAS.items():
        value = raw.get(section)
        if not isinstance(value, Mapping) or _dangerous_key(section):
            continue
        settings = _filter_known_mapping(value, schema)
        if settings:
            filtered[section] = settings
    return filtered


def _replace_prefix(value: Path, old: Path, new: Path) -> Path | None:
    try:
        relative = value.resolve(strict=False).relative_to(old.resolve(strict=False))
    except ValueError:
        return None
    return (new / relative).resolve(strict=False)


def _map_legacy_path(
    value: str,
    *,
    layout: RuntimeLayout,
    backup_root: Path,
) -> str:
    if not value:
        return value
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        first, *remaining = raw.parts
        relative_targets = {
            "configs": layout.config_dir,
            "logs": layout.logs_dir,
            "outputs": layout.outputs_dir,
            "tmp": layout.tmp_dir,
            "workspaces": layout.workspaces_dir,
            "solar_apps": layout.apps_root / "solar_apps",
            "examples": layout.apps_root / "examples",
        }
        target = relative_targets.get(first.casefold())
        return str(target.joinpath(*remaining)) if target is not None else value

    logical_old_root = layout.repo_root / "Local"
    mappings = (
        (logical_old_root / "configs", layout.config_dir),
        (logical_old_root / "logs", layout.logs_dir),
        (logical_old_root / "outputs", layout.outputs_dir),
        (logical_old_root / "tmp", layout.tmp_dir),
        (logical_old_root / "workspaces", layout.workspaces_dir),
        (logical_old_root / "solar_apps", layout.apps_root / "solar_apps"),
        (logical_old_root / "scripts", layout.apps_root / "solar_apps" / "workflows"),
        (logical_old_root / "examples", layout.apps_root / "examples"),
        (backup_root / "configs", layout.config_dir),
        (backup_root / "logs", layout.logs_dir),
        (backup_root / "outputs", layout.outputs_dir),
        (backup_root / "tmp", layout.tmp_dir),
        (backup_root / "solar_apps", layout.apps_root / "solar_apps"),
        (backup_root / "scripts", layout.apps_root / "solar_apps" / "workflows"),
        (backup_root / "examples", layout.apps_root / "examples"),
    )
    for old, new in mappings:
        mapped = _replace_prefix(raw, old, new)
        if mapped is not None:
            return str(mapped)
    return value


def _map_values(value: Any, *, layout: RuntimeLayout, backup_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _map_values(item, layout=layout, backup_root=backup_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _map_values(item, layout=layout, backup_root=backup_root) for item in value
        ]
    if isinstance(value, str):
        return _map_legacy_path(value, layout=layout, backup_root=backup_root)
    return value


def _safe_roots(data: Mapping[str, Any], *, layout: RuntimeLayout) -> tuple[Path, ...]:
    apps = data.get("apps", {})
    raw_roots = apps.get("allowed_roots", []) if isinstance(apps, Mapping) else []
    if not isinstance(raw_roots, list):
        return ()
    roots: list[Path] = []
    for raw in raw_roots:
        if not isinstance(raw, str):
            continue
        path = Path(raw).expanduser()
        if not path.is_dir():
            continue
        try:
            normalized = normalize_allowed_roots(
                [path], workspace_root=layout.repo_root
            )[0]
        except (AllowedRootPolicyError, IndexError):
            continue
        roots.append(normalized)
    return tuple(dict.fromkeys(roots))


def _migrate_config(
    *,
    layout: RuntimeLayout,
    backup_root: Path,
) -> bool:
    target_exists = layout.config_path.is_file()
    source = (
        layout.config_path
        if target_exists
        else backup_root / "configs" / "paths.local.yaml"
    )
    if not source.is_file():
        return False
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    if not isinstance(raw, Mapping):
        return False
    current_apps = raw.get("apps", {})
    if (
        target_exists
        and isinstance(current_apps, Mapping)
        and current_apps.get("runtime_layout_version") == RUNTIME_LAYOUT_VERSION
    ):
        return False
    raw_apps = raw.get("apps", {})
    imported: dict[str, Any] = {
        "apps": {
            "allowed_roots": (
                list(raw_apps.get("allowed_roots", []))
                if isinstance(raw_apps, Mapping)
                and isinstance(raw_apps.get("allowed_roots", []), list)
                else []
            )
        },
        "scripts": _filter_scripts(raw.get("scripts", {})),
    }
    imported = _map_values(imported, layout=layout, backup_root=backup_root)
    apps = imported.setdefault("apps", {})
    if isinstance(apps, dict):
        apps["allowed_roots"] = [
            str(path) for path in _safe_roots(imported, layout=layout)
        ]
        apps["runtime_layout_version"] = RUNTIME_LAYOUT_VERSION
    layout.config_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(imported, sort_keys=False, allow_unicode=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{layout.config_path.name}.",
            suffix=".tmp",
            dir=layout.config_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, layout.config_path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return True


def _load_allowed_roots(layout: RuntimeLayout) -> tuple[Path, ...]:
    try:
        data = yaml.safe_load(layout.config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return ()
    return _safe_roots(data, layout=layout) if isinstance(data, Mapping) else ()


def _valid_setting_path(key: str, value: str, roots: tuple[Path, ...]) -> bool:
    if key not in _PATH_FIELDS:
        return True
    if not roots:
        return False
    candidate = Path(value).expanduser().resolve(strict=False)
    anchor = candidate.parent if key == "video_output_path" else candidate
    if key in {"aia_dir", "radio_dir"} and not anchor.is_dir():
        return False
    if key == "output_dir" and not anchor.exists():
        existing = anchor
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        anchor = existing
    return anchor.exists() and is_path_within_roots(anchor, roots)


def _migrate_home_settings(
    *,
    layout: RuntimeLayout,
    home: Path,
    backup_root: Path,
) -> tuple[Path, ...]:
    source_dir = home / ".solar_toolkit"
    roots = _load_allowed_roots(layout)
    created: list[Path] = []
    for filename, (frontend, allowed_fields) in _LEGACY_SETTINGS.items():
        source = source_dir / filename
        target = layout.state_dir / f"{frontend}.json"
        store = StateStore(
            target,
            frontend,
            allowed_keys=("fields", "theme", "legacy_imported"),
        )
        existing = store.load({"__invalid__": True}) if target.exists() else None
        if (
            existing is not None and "__invalid__" not in existing
        ) or not source.is_file():
            continue
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, Mapping):
            continue
        fields: dict[str, Any] = {}
        theme = "auto"
        for key in allowed_fields:
            if key not in raw:
                continue
            value = _map_values(raw[key], layout=layout, backup_root=backup_root)
            if key == "theme_mode":
                if value in {"auto", "light", "dark"}:
                    theme = value
                continue
            if isinstance(value, str) and not _valid_setting_path(key, value, roots):
                continue
            fields[key] = value
        store.save({"fields": fields, "theme": theme, "legacy_imported": True})
        created.append(target)
    return tuple(created)


def migrate_legacy_state(
    *,
    layout: RuntimeLayout | None = None,
    home: str | os.PathLike[str] | None = None,
    backup_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Import safe latest settings once without modifying any legacy file."""

    selected = (layout or RuntimeLayout.discover()).ensure()
    backup = (
        Path(backup_root).expanduser().resolve(strict=False)
        if backup_root is not None
        else selected.repo_root / "Local-migration-backup"
    )
    created: list[Path] = []
    if _migrate_config(layout=selected, backup_root=backup):
        created.append(selected.config_path)
    created.extend(
        _migrate_home_settings(
            layout=selected,
            home=Path(home).expanduser().resolve(strict=False) if home else Path.home(),
            backup_root=backup,
        )
    )
    return tuple(created)


__all__ = ["RUNTIME_LAYOUT_VERSION", "migrate_legacy_state"]
