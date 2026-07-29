"""Configuration, candidate discovery, and filesystem policy for the app."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FITS_SUFFIXES = {".fit", ".fits", ".fts"}
CONFIG_PATTERN = re.compile(
    r"^solar_apps\.workflows\.radio\.configs\.[A-Za-z_][A-Za-z0-9_]*$"
)

COMMON_CONFIG_KEYS = {
    "mode",
    "single_file_path",
    "multi_band_root",
    "data_dir",
    "multi_band_freqs",
    "polarization",
    "combine_polarizations",
    "enable_gaussian_overlay",
    "enable_spectrogram_panel",
    "spectrogram_file_path",
    "radio_background_strategy",
    "background_use_for_display",
    "background_use_for_fit",
    "radio_cmap",
    "cmap",
    "color_range_mode",
    "fixed_vmin",
    "fixed_vmax",
    "radio_colorbar_unit",
    "spectrogram_colorbar_unit",
    "output_dir",
    "show_plot",
    "save_plot",
    "write_source_map_sidecar",
    "max_workers",
}


class PathPolicy:
    """Resolve app paths inside explicitly allowed roots."""

    def __init__(self, roots: Sequence[str | Path]) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        if not self.roots:
            raise ValueError("At least one allowed root is required")

    def resolve(
        self,
        value: Any,
        *,
        must_exist: bool = False,
        kind: str | None = None,
    ) -> Path:
        text = str(value or "").strip()
        if not text:
            raise ValueError("A filesystem path is required")
        path = Path(text).expanduser().resolve(strict=must_exist)
        if not any(path == root or root in path.parents for root in self.roots):
            raise PermissionError(
                f"Path is outside the configured allowed roots: {path}"
            )
        if kind == "file" and not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")
        if kind == "directory" and not path.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {path}")
        return path

    def list_directory(self, value: Any) -> list[dict[str, Any]]:
        root = self.resolve(value, must_exist=True, kind="directory")
        items = []
        for path in sorted(
            root.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())
        ):
            if path.is_file() and path.suffix.casefold() not in FITS_SUFFIXES | {
                ".png",
                ".json",
            }:
                continue
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "kind": "directory" if path.is_dir() else "file",
                    "suffix": path.suffix.casefold(),
                    "size": None if path.is_dir() else int(path.stat().st_size),
                }
            )
        return items[:1000]


def parse_request_config(
    payload: Mapping[str, Any], *, policy: PathPolicy
) -> dict[str, Any]:
    """Build one frozen source-map config from validated frontend controls."""

    from solar_apps.workflows.radio.configs import DEFAULT_CONFIG_NAME
    from solar_apps.workflows.radio import source_map_workflow as workflow
    from solar_toolkit.radio.config import load_radio_user_config

    config_name = str(payload.get("config") or DEFAULT_CONFIG_NAME).strip()
    if not CONFIG_PATTERN.fullmatch(config_name):
        raise ValueError(
            "Config must name a module under solar_apps.workflows.radio.configs"
        )
    user_config, _newkirk = load_radio_user_config(config_name)
    cfg = workflow.build_config(user_config, workflow.DEFAULT_CONFIG)
    cfg = workflow._migrate_config(cfg)
    cfg.update(
        validate_advanced_config(payload.get("advanced"), workflow.DEFAULT_CONFIG)
    )

    mode = str(payload.get("mode") or cfg.get("mode") or "single_band").strip().lower()
    if mode not in {"single_band", "multi_band"}:
        raise ValueError("mode must be single_band or multi_band")
    polarization = str(
        payload.get("polarization") or cfg.get("polarization") or "RR"
    ).upper()
    if polarization not in {"RR", "LL", "RR+LL"}:
        raise ValueError("polarization must be RR, LL, or RR+LL")

    source = policy.resolve(payload.get("source_path"), must_exist=True)
    if mode == "single_band" and not (source.is_file() or source.is_dir()):
        raise ValueError("Single-band source must be a FITS file or directory")
    if mode == "multi_band" and not source.is_dir():
        raise ValueError("Multi-band source must be a directory")
    output = policy.resolve(payload.get("output_dir"), must_exist=False)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")

    cfg.update(
        {
            "mode": mode,
            "polarization": polarization,
            "combine_polarizations": polarization == "RR+LL",
            "output_dir": str(output),
            "show_plot": False,
            "save_plot": True,
            "write_source_map_sidecar": True,
            "max_workers": 1,
            "enable_gaussian_overlay": _bool(
                payload.get("gaussian_overlay"),
                default=bool(cfg.get("enable_gaussian_overlay", True)),
            ),
            "enable_spectrogram_panel": _bool(
                payload.get("spectrogram_panel"), default=False
            ),
            "radio_cmap": _safe_cmap(
                payload.get("cmap") or cfg.get("radio_cmap") or cfg.get("cmap") or "hot"
            ),
            "color_range_mode": _color_range_mode(
                payload.get("color_range_mode") or cfg.get("color_range_mode") or "auto"
            ),
        }
    )
    if "start_idx" in payload:
        cfg["start_idx"] = _nonnegative_int(payload.get("start_idx"), default=0)
    if "end_idx" in payload:
        cfg["end_idx"] = _optional_end_index(payload.get("end_idx"))
    cfg["cmap"] = cfg["radio_cmap"]
    if cfg["end_idx"] is not None and cfg["end_idx"] < cfg["start_idx"]:
        raise ValueError("end_idx cannot precede start_idx")
    if mode == "single_band":
        cfg["single_file_path"] = str(source) if source.is_file() else None
        cfg["data_dir"] = str(source)
    else:
        cfg["multi_band_root"] = str(source)

    frequencies = _float_list(payload.get("frequencies"))
    if frequencies:
        cfg["multi_band_freqs"] = frequencies
    if mode == "multi_band" and not cfg.get("multi_band_freqs"):
        raise ValueError("At least one multi-band frequency is required")

    for request_key, config_key in (
        ("fixed_vmin", "fixed_vmin"),
        ("fixed_vmax", "fixed_vmax"),
    ):
        if payload.get(request_key) not in (None, ""):
            cfg[config_key] = _finite_float(payload[request_key], request_key)
    if cfg["color_range_mode"] == "fixed" and (
        cfg.get("fixed_vmin") is None or cfg.get("fixed_vmax") is None
    ):
        raise ValueError("Fixed color range requires both vmin and vmax")

    radio_unit = str(payload.get("radio_unit") or "").strip()
    spectrogram_unit = str(payload.get("spectrogram_unit") or "").strip()
    cfg["radio_colorbar_unit"] = radio_unit or None
    cfg["spectrogram_colorbar_unit"] = spectrogram_unit or None

    spectrogram_path = str(payload.get("spectrogram_path") or "").strip()
    if cfg["enable_spectrogram_panel"]:
        if spectrogram_path:
            cfg["spectrogram_file_path"] = str(
                policy.resolve(spectrogram_path, must_exist=True, kind="file")
            )
            cfg["spectrogram_file_paths"] = []
        else:
            configured = cfg.get("spectrogram_file_paths") or [
                cfg.get("spectrogram_file_path")
            ]
            cfg["spectrogram_file_paths"] = [
                str(policy.resolve(path, must_exist=True, kind="file"))
                for path in configured
                if path
            ]
            if cfg["spectrogram_file_paths"]:
                cfg["spectrogram_file_path"] = cfg["spectrogram_file_paths"][0]

    background_mode = str(payload.get("background_mode") or "off").strip().lower()
    if background_mode not in {"off", "noise_map_only", "local_mesh", "local_median"}:
        raise ValueError("Unsupported radio background mode")
    apply_display = _bool(payload.get("background_display"), default=False)
    apply_fit = _bool(payload.get("background_fit"), default=False)
    cfg["radio_background_strategy"] = background_mode
    cfg["radio_background_mode"] = background_mode
    cfg["radio_background_subtraction_mode"] = background_mode
    cfg["background_use_for_display"] = apply_display
    cfg["background_use_for_fit"] = apply_fit
    cfg["radio_background_force_off"] = background_mode == "off"
    cfg["enable_radio_background_subtraction"] = background_mode != "off"
    cfg["radio_background_workflow"] = (
        "display_and_fit"
        if apply_display and apply_fit
        else "display_only" if apply_display else "fit_only" if apply_fit else "off"
    )
    return cfg


def discover_candidates(
    cfg: dict[str, Any], *, policy: PathPolicy
) -> list[dict[str, Any]]:
    """Discover one-file candidates or synchronized multi-band slots."""

    from solar_apps.workflows.radio import source_map_workflow as workflow

    mode = cfg["mode"]
    if mode == "multi_band":
        slots = workflow._build_multi_band_slots(cfg)
        candidates = []
        for index, slot in enumerate(slots):
            frozen_slot: list[Any] = []
            paths: list[Path] = []
            for item in slot:
                if isinstance(item, (tuple, list)):
                    pair = [
                        policy.resolve(path, must_exist=True, kind="file")
                        for path in item
                    ]
                    frozen_slot.append([str(path) for path in pair])
                    paths.extend(pair)
                else:
                    path = policy.resolve(item, must_exist=True, kind="file")
                    frozen_slot.append(str(path))
                    paths.append(path)
            if not paths:
                continue
            candidates.append(
                _candidate_record(
                    f"slot-{index:04d}",
                    mode,
                    paths,
                    cfg,
                    slot_index=index,
                    slot=frozen_slot,
                )
            )
        return candidates

    source = Path(str(cfg.get("single_file_path") or cfg.get("data_dir")))
    if source.is_file():
        files = [policy.resolve(source, must_exist=True, kind="file")]
    else:
        directory = source
        polarization = cfg["polarization"]
        if (directory / ("RR" if polarization == "RR+LL" else polarization)).is_dir():
            directory = directory / ("RR" if polarization == "RR+LL" else polarization)
        directory = policy.resolve(directory, must_exist=True, kind="directory")
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in FITS_SUFFIXES
        )
        start = int(cfg.get("start_idx", 0) or 0)
        end = cfg.get("end_idx")
        files = files[start : int(end) if end is not None else None]

    candidates = []
    for index, source_file in enumerate(files):
        paths = [policy.resolve(source_file, must_exist=True, kind="file")]
        if cfg["polarization"] == "RR+LL":
            paths.append(_polarization_counterpart(paths[0], cfg, policy=policy))
        candidates.append(
            _candidate_record(
                f"file-{index:04d}", "single_band", paths, cfg, run_path=str(paths[0])
            )
        )
    return candidates


def public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) for key, value in candidate.items() if key != "slot"
    }


def validate_advanced_config(raw: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    if raw in (None, "", {}):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Advanced JSON is invalid: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Advanced JSON must be an object")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key not in defaults:
            raise ValueError(f"Unknown advanced source-map option: {key}")
        lowered = key.lower()
        if (
            key in COMMON_CONFIG_KEYS
            or key.startswith("_")
            or any(
                token in lowered for token in ("path", "file", "dir", "root", "output")
            )
        ):
            raise ValueError(f"Advanced JSON cannot override protected option: {key}")
        default = defaults[key]
        if default is not None:
            _validate_compatible_type(key, value, default)
        result[key] = copy.deepcopy(value)
    return result


def _candidate_record(
    candidate_id: str,
    mode: str,
    paths: Sequence[Path],
    cfg: Mapping[str, Any],
    *,
    slot_index: int | None = None,
    slot: list[Any] | None = None,
    run_path: str | None = None,
) -> dict[str, Any]:
    from astropy.io import fits
    from solar_apps.workflows.radio import source_map_workflow as workflow

    with fits.open(paths[0], memmap=True) as hdul:
        header = (
            hdul[1].header.copy()
            if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU)
            else hdul[0].header.copy()
        )
    observed = workflow.radio_datetime_from_header_or_path(header, str(paths[0]), cfg)
    frequencies = []
    for path in paths[::2] if cfg.get("polarization") == "RR+LL" else paths:
        with fits.open(path, memmap=True) as hdul:
            item_header = (
                hdul[1].header
                if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU)
                else hdul[0].header
            )
            frequency = workflow.get_freq_from_header(item_header)
            if frequency is not None:
                frequencies.append(float(frequency))
    result = {
        "id": candidate_id,
        "mode": mode,
        "title": (
            f"Source Map Slot {int(slot_index) + 1}"
            if slot_index is not None
            else f"Source Map {paths[0].name}"
        ),
        "paths": [str(path) for path in paths],
        "observation_time": observed.isoformat() if observed is not None else None,
        "frequencies_mhz": frequencies,
        "polarization": str(cfg.get("polarization")),
        "pairing_status": (
            "RR+LL matched"
            if cfg.get("combine_polarizations")
            else "single polarization"
        ),
    }
    if slot_index is not None:
        result["slot_index"] = int(slot_index)
        result["slot"] = slot
    if run_path is not None:
        result["run_path"] = run_path
    return result


def _polarization_counterpart(
    path: Path, cfg: Mapping[str, Any], *, policy: PathPolicy
) -> Path:
    rr = str(cfg.get("rr_dir_suffix", "RR"))
    ll = str(cfg.get("ll_dir_suffix", "LL"))
    if path.parent.name == rr:
        counterpart = path.parent.parent / ll / path.name
    elif path.parent.name == ll:
        counterpart = path.parent.parent / rr / path.name
    else:
        raise ValueError("RR+LL inputs must be stored in matched RR and LL directories")
    return policy.resolve(counterpart, must_exist=True, kind="file")


def _validate_compatible_type(key: str, value: Any, default: Any) -> None:
    if isinstance(default, bool):
        valid = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, str):
        valid = isinstance(value, str)
    elif isinstance(default, dict):
        valid = isinstance(value, dict)
    elif isinstance(default, (list, tuple)):
        valid = isinstance(value, list)
    else:
        valid = True
    if not valid:
        raise ValueError(f"Advanced option {key} has an incompatible value type")


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {
        "false",
        "0",
        "no",
        "off",
        "",
    }:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _float_list(value: Any) -> list[float | int]:
    if value in (None, "", []):
        return []
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ValueError("frequencies must be a comma-separated list")
    result: list[float | int] = []
    for item in items:
        if not str(item).strip():
            continue
        number = _finite_float(item, "frequency")
        result.append(int(number) if number.is_integer() else number)
    return result


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonnegative_int(value: Any, *, default: int) -> int:
    number = default if value in (None, "") else int(value)
    if number < 0:
        raise ValueError("start_idx must be non-negative")
    return number


def _optional_end_index(value: Any) -> int | None:
    if value in (None, "", -1, "-1"):
        return None
    number = int(value)
    if number < 0:
        raise ValueError("end_idx must be non-negative or omitted")
    return number


def _safe_cmap(value: Any) -> str:
    name = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("Invalid Matplotlib colormap name")
    return name


def _color_range_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"auto", "fixed", "global"}:
        raise ValueError("color_range_mode must be auto, fixed, or global")
    return mode
