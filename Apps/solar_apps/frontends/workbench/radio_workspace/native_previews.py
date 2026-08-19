"""Explicit, browser-native preview adapters for the radio workspace.

This module intentionally keeps its import surface lightweight. Scientific and
Plotly dependencies are imported only by the adapter that needs them, after the
caller explicitly requests a preview and all user-supplied paths have passed the
injected validator.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import mimetypes
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["build_native_preview"]

PathValidator = Callable[[str | Path], str | Path | None]


def build_native_preview(
    adapter: str,
    form: Mapping[str, Any] | None,
    *,
    validate_path: PathValidator,
) -> dict[str, Any]:
    """Build one explicitly requested native preview.

    Successful adapters return a JSON-compatible mapping with ``status`` set to
    ``"ready"``. Adapters that require state owned by another layer return an
    ``"unavailable"`` result rather than importing that layer. Invalid inputs
    and rejected paths raise an exception for the route boundary to report.
    """

    adapter_name = str(adapter or "").strip().lower()
    payload = dict(form or {})
    if not callable(validate_path):
        raise TypeError("validate_path must be callable")

    if adapter_name == "roi-selection":
        return _build_roi_selection_preview(payload, validate_path=validate_path)
    if adapter_name == "trajectory-media":
        return _build_trajectory_preview(payload, validate_path=validate_path)
    if adapter_name == "drift-selection":
        return _build_drift_selection_preview(payload, validate_path=validate_path)
    if adapter_name == "spectrogram-coverage":
        return _build_spectrogram_coverage_preview(payload, validate_path=validate_path)
    if adapter_name == "source-map-selection":
        return _build_source_map_selection_preview(payload, validate_path=validate_path)
    if adapter_name == "file-browser":
        return _build_file_browser(payload, validate_path=validate_path)
    if adapter_name in {"run-index", "artifact-index"}:
        return _unavailable_index(adapter_name)
    raise ValueError(f"Unknown native preview adapter: {adapter!r}")


def _build_source_map_selection_preview(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    """Scan source-map inputs, persist an explicit selection, and render one PNG."""

    from solar_apps.workflows.radio import source_map_workflow as workflow
    from solar_toolkit.radio.config import DEFAULT_CONFIG_NAME, load_radio_user_config

    config_name = str(form.get("config") or DEFAULT_CONFIG_NAME)
    event_user_config, _newkirk = load_radio_user_config(config_name)
    workspace_config = _source_map_adapter_config(form)
    user_config = _deep_merge_dict(event_user_config, workspace_config)
    cfg = workflow.build_config(user_config, workflow.DEFAULT_CONFIG)
    cfg["show_plot"] = False
    cfg["save_plot"] = True
    cfg["enable_spectrogram_panel"] = False
    cfg["max_workers"] = 1
    cfg["dpi"] = min(int(cfg.get("dpi", 120) or 120), 120)
    if str(cfg.get("polarization", "RR+LL")).upper() == "RR+LL":
        cfg["polarization"] = "RR+LL"
        cfg["combine_polarizations"] = True
    else:
        cfg["polarization"] = str(cfg.get("polarization", "RR")).upper()
        cfg["combine_polarizations"] = False

    mode = str(cfg.get("mode") or "multi_band")
    if mode == "multi_band":
        candidates = _source_map_multi_band_candidates(
            cfg, workflow=workflow, validate_path=validate_path
        )
    elif mode == "single_band":
        candidates = _source_map_single_band_candidates(
            cfg, workflow=workflow, validate_path=validate_path
        )
    else:
        raise ValueError("Source-map mode must be multi_band or single_band")
    if not candidates:
        raise RuntimeError("No source-map candidates were found.")

    selected_ids = _selected_source_map_candidate_ids(
        form.get("selected_source_map_json"), candidates
    )
    selected = next(item for item in candidates if item["id"] == selected_ids[0])
    image_url = _render_source_map_candidate(
        selected, cfg, workflow=workflow, validate_path=validate_path
    )
    return {
        "adapter": "source-map-selection",
        "status": "ready",
        "kind": "image",
        "title": selected["title"],
        "image_url": image_url,
        "candidates": candidates,
        "selected_candidate_ids": selected_ids,
        "metadata": {
            "mode": mode,
            "candidate_count": len(candidates),
            "candidate_limit": len(candidates),
            "frequency_mhz": selected.get("frequency_mhz"),
            "observation_time": selected.get("observation_time"),
            "polarization": selected.get("polarization"),
            "selection": selected.get("selection"),
            "title": selected["title"],
        },
    }


def _source_map_adapter_config(form: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("features", "display", "gaussian", "background", "spectrogram"):
        value = form.get(key)
        if isinstance(value, Mapping):
            result[key] = copy.deepcopy(dict(value))
    if form.get("mode") not in (None, ""):
        result["mode"] = str(form["mode"])
    data = copy.deepcopy(dict(form.get("data") or {}))
    mappings = {
        "single_file_path": "single_file_path",
        "radio_dir": "multi_band_root",
        "polarization": "polarization",
        "combine_polarizations": "combine_polarizations",
        "selected_source_map_json": "selected_source_map_json",
    }
    for source, target in mappings.items():
        if form.get(source) not in (None, ""):
            data[target] = form[source]
    if data:
        result["data"] = data
    return result


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _source_map_single_band_candidates(
    cfg: dict[str, Any], *, workflow: Any, validate_path: PathValidator
) -> list[dict[str, Any]]:
    raw_path = str(cfg.get("single_file_path") or "").strip()
    if not raw_path:
        raise ValueError("Select one Single radio FITS file before Preview.")
    path = _validated_path(raw_path, validate_path=validate_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source-map FITS file does not exist: {path}")
    return [
        _source_map_file_candidate(
            path,
            number=1,
            cfg=cfg,
            workflow=workflow,
            root=path.parent,
            validate_path=validate_path,
        )
    ]


def _source_map_multi_band_candidates(
    cfg: dict[str, Any], *, workflow: Any, validate_path: PathValidator
) -> list[dict[str, Any]]:
    root = _validated_path(cfg["multi_band_root"], validate_path=validate_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Multi-band radio folder does not exist: {root}")
    cfg["multi_band_root"] = str(root)
    _validate_source_map_band_dirs(cfg, validate_path=validate_path)
    slots = workflow._build_multi_band_slots(cfg)
    candidates: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        paths = _source_map_slot_paths(slot)
        validated = [
            _validated_path(path, validate_path=validate_path) for path in paths
        ]
        header = _source_map_header(validated[0])
        observed = workflow.radio_datetime_from_header_or_path(
            header, str(validated[0]), cfg
        )
        freq = workflow.get_freq_from_header(header)
        entries = [
            {
                "path": str(path),
                "relative_path": _relative_text(path, root),
            }
            for path in validated
        ]
        candidate_id = f"slot-{index:04d}"
        selection = _source_map_selection_payload(
            mode="multi_band",
            candidate_id=candidate_id,
            polarization=cfg.get("polarization"),
            slot_index=index,
            run_path=None,
            paths=[str(path) for path in validated],
        )
        candidates.append(
            {
                "id": candidate_id,
                "number": index + 1,
                "mode": "multi_band",
                "title": f"Source Map Slot #{index + 1}",
                "slot_index": index,
                "paths": [str(path) for path in validated],
                "entries": entries,
                "relative_path": "; ".join(item["relative_path"] for item in entries),
                "frequency_mhz": _finite_float_or_none(freq),
                "observation_time": _iso_or_none(observed),
                "polarization": str(cfg.get("polarization") or ""),
                "pairing_status": (
                    "RR+LL matched"
                    if cfg.get("combine_polarizations")
                    else "single polarization"
                ),
                "selection": selection,
            }
        )
    return candidates


def _validate_source_map_band_dirs(
    cfg: dict[str, Any], *, validate_path: PathValidator
) -> None:
    root = Path(str(cfg["multi_band_root"]))
    pattern = str(cfg.get("band_dir_pattern", "{freq}MHz/{polar}"))
    polarizations = (
        [cfg.get("rr_dir_suffix", "RR"), cfg.get("ll_dir_suffix", "LL")]
        if cfg.get("combine_polarizations") and cfg.get("polarization") == "RR+LL"
        else [cfg.get("polarization", "RR")]
    )
    for freq in cfg.get("multi_band_freqs", []):
        for polar in polarizations:
            folder = _validated_path(
                root / pattern.format(freq=freq, polar=polar),
                validate_path=validate_path,
            )
            if not folder.is_dir():
                raise NotADirectoryError(f"Radio band folder does not exist: {folder}")


def _source_map_file_candidate(
    path: Path,
    *,
    number: int,
    cfg: dict[str, Any],
    workflow: Any,
    root: Path,
    validate_path: PathValidator,
) -> dict[str, Any]:
    header = _source_map_header(path)
    observed = workflow.radio_datetime_from_header_or_path(header, str(path), cfg)
    freq = workflow.get_freq_from_header(header)
    paths = [str(path)]
    pairing_status = "single polarization"
    if cfg.get("combine_polarizations") and cfg.get("polarization") == "RR+LL":
        counterpart = _source_map_counterpart(path, cfg, validate_path=validate_path)
        paths.append(str(counterpart))
        pairing_status = f"RR+LL matched with {counterpart.parent.name}"
    candidate_id = f"file-{number:04d}"
    selection = _source_map_selection_payload(
        mode="single_band",
        candidate_id=candidate_id,
        polarization=cfg.get("polarization"),
        slot_index=None,
        run_path=str(path),
        paths=paths,
    )
    return {
        "id": candidate_id,
        "number": number,
        "mode": "single_band",
        "title": f"Source Map File #{number}",
        "run_path": str(path),
        "paths": paths,
        "relative_path": _relative_text(path, root),
        "frequency_mhz": _finite_float_or_none(freq),
        "observation_time": _iso_or_none(observed),
        "polarization": str(cfg.get("polarization") or ""),
        "pairing_status": pairing_status,
        "selection": selection,
    }


def _source_map_counterpart(
    path: Path, cfg: dict[str, Any], *, validate_path: PathValidator
) -> Path:
    rr_name = str(cfg.get("rr_dir_suffix", "RR"))
    ll_name = str(cfg.get("ll_dir_suffix", "LL"))
    parent_name = path.parent.name
    if parent_name == rr_name:
        counterpart = path.parent.parent / ll_name / path.name
    elif parent_name == ll_name:
        counterpart = path.parent.parent / rr_name / path.name
    else:
        raise ValueError(
            "RR+LL Preview requires the selected FITS file to live in an RR or LL folder."
        )
    validated = _validated_path(counterpart, validate_path=validate_path)
    if not validated.is_file():
        raise FileNotFoundError(
            f"RR+LL Preview could not find the matching polarization file: {validated}"
        )
    return validated


def _source_map_header(path: Path) -> Any:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        return (
            hdul[1].header.copy()
            if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU)
            else hdul[0].header.copy()
        )


def _source_map_slot_paths(slot: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in slot:
        if isinstance(item, (tuple, list)):
            paths.extend(str(value) for value in item)
        else:
            paths.append(str(item))
    return paths


def _source_map_selection_payload(
    *,
    mode: str,
    candidate_id: str,
    polarization: Any,
    slot_index: int | None,
    run_path: str | None,
    paths: list[str],
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "candidate_id": candidate_id,
        "paths": paths,
    }
    if slot_index is not None:
        item["slot_index"] = int(slot_index)
    if run_path:
        item["run_path"] = run_path
    return {
        "schema_version": 1,
        "mode": mode,
        "polarization": str(polarization or ""),
        "candidate_ids": [candidate_id],
        "items": [item],
    }


def _selected_source_map_candidate_ids(
    raw_value: Any, candidates: list[dict[str, Any]]
) -> list[str]:
    if raw_value in (None, ""):
        return [str(candidates[0]["id"])]
    if isinstance(raw_value, str):
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError("selected_source_map_json must be a JSON object.") from exc
    else:
        decoded = raw_value
    if not isinstance(decoded, Mapping):
        raise ValueError("selected_source_map_json must be a JSON object.")
    candidate_ids = decoded.get("candidate_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != 1:
        raise ValueError("Select exactly one source-map candidate.")
    candidate_id = str(candidate_ids[0])
    valid_ids = {str(item["id"]) for item in candidates}
    if candidate_id not in valid_ids:
        raise ValueError("The selected source-map candidate is stale. Preview again.")
    return [candidate_id]


def _render_source_map_candidate(
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    *,
    workflow: Any,
    validate_path: PathValidator,
) -> str:
    cfg = dict(cfg)
    cfg["output_dir"] = ""
    cfg["show_plot"] = False
    cfg["save_plot"] = True
    cfg["_interactive"] = False
    cfg["enable_spectrogram_panel"] = False
    with tempfile.TemporaryDirectory(prefix="radio-source-map-preview-") as tmp:
        cfg["output_dir"] = tmp
        if candidate["mode"] == "multi_band":
            slot = _source_map_slot_from_candidate(
                candidate, validate_path=validate_path
            )
            output = workflow.plot_multi_band_slot(
                int(candidate["slot_index"]), slot, tmp, cfg, vmin=None, vmax=None
            )
        else:
            path = _validated_path(candidate["run_path"], validate_path=validate_path)
            output = workflow.plot_single_band(
                str(path), tmp, cfg, vmin=None, vmax=None
            )
        output_path = Path(output)
        mime_type = mimetypes.guess_type(output_path.name)[0] or "image/png"
        return f"data:{mime_type};base64," + base64.b64encode(
            output_path.read_bytes()
        ).decode("ascii")


def _source_map_slot_from_candidate(
    candidate: dict[str, Any], *, validate_path: PathValidator
) -> list[Any]:
    entries = candidate.get("entries") or []
    paths = [
        _validated_path(item["path"], validate_path=validate_path) for item in entries
    ]
    if candidate.get("pairing_status") == "RR+LL matched":
        paired = []
        for index in range(0, len(paths), 2):
            paired.append((str(paths[index]), str(paths[index + 1])))
        return paired
    return [str(path) for path in paths]


def _build_spectrogram_coverage_preview(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    """Build an explicit multi-file CSO preview without crossing real gaps.

    The canonical radio spectrogram builder performs the FITS reads, frequency
    selection, polarization calculation, downsampling, and stitching.  This
    adapter adds only workspace path validation and a browser-safe Plotly view.
    Files separated by more than one second are rendered as separate heatmap
    traces, so Plotly cannot paint or interpolate across an unobserved interval.
    """

    primary = _required_validated_path(
        form, "primary_file", validate_path=validate_path
    )
    adjacent_values: list[str] = []
    if form.get("adjacent_file") not in (None, ""):
        adjacent_values.append(str(form["adjacent_file"]).strip())
    adjacent_values.extend(_json_path_list(form.get("adjacent_files_json")))
    validated_inputs = [primary]
    validated_inputs.extend(
        _validated_path(value, validate_path=validate_path) for value in adjacent_values
    )

    paths: list[Path] = []
    seen: set[Path] = set()
    for path in validated_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Spectrogram FITS file does not exist: {path}")
        if path.suffix.casefold() not in {".fit", ".fits", ".fts"}:
            raise ValueError(f"Spectrogram input must be a FITS file: {path.name}")
        canonical = path.resolve(strict=True)
        if canonical not in seen:
            seen.add(canonical)
            paths.append(canonical)

    if not paths:
        raise ValueError("Select a primary spectrogram FITS file")

    frequency_start = _bounded_float(
        form.get("frequency_start", 80.0), minimum=-1.0e9, maximum=1.0e9
    )
    frequency_end = _bounded_float(
        form.get("frequency_end", 340.0), minimum=-1.0e9, maximum=1.0e9
    )
    if frequency_start >= frequency_end:
        raise ValueError("frequency_start must be less than frequency_end")
    rebin_time = _bounded_int(form.get("rebin_time", 1000), minimum=1, maximum=10000)
    rebin_frequency = _bounded_int(
        form.get("rebin_frequency", 700), minimum=1, maximum=4096
    )
    polarization = str(form.get("polarization", "sum") or "sum").strip().lower()
    if polarization not in {"ll", "rr", "sum", "ratio"}:
        raise ValueError("polarization must be LL, RR, sum, or ratio")
    cmap = str(form.get("cmap", "jet") or "jet").strip().lower()
    if cmap not in {"jet", "viridis", "plasma", "inferno", "magma", "cividis"}:
        raise ValueError("Unsupported spectrogram color map")

    # Deliberately lazy: no FITS, NumPy, Matplotlib, or Plotly dependency is
    # loaded merely by enabling or expanding the owning workspace module.
    import numpy as np

    from solar_toolkit.radio import spectrogram

    from .figure_time import merge_coverage_segments

    file_metadata = [
        spectrogram._read_spectrogram_file_metadata(str(path)) for path in paths
    ]
    coverage_segments = merge_coverage_segments(
        [
            {
                "start_time_iso": _utc_iso(item["file_start"]),
                "end_time_iso": _utc_iso(item["file_end"]),
            }
            for item in file_metadata
        ],
        maximum_gap_s=1.0,
    )
    x_start_iso = coverage_segments[0]["start_time_iso"]
    x_end_iso = coverage_segments[-1]["end_time_iso"]
    cfg: dict[str, Any] = {
        "enable_spectrogram_panel": True,
        "spectrogram_file_paths": [str(path) for path in paths],
        "spectrogram_file_path": str(paths[0]),
        "spectrogram_time_display_mode": "full",
        # The preview must preserve (and display) gaps rather than disabling the
        # entire panel.  Separate Plotly traces below keep those gaps empty.
        "spectrogram_disable_on_time_mismatch": False,
        "spectrogram_f_start": frequency_start,
        "spectrogram_f_end": frequency_end,
        "spectrogram_rebin_t_target": rebin_time,
        "spectrogram_rebin_f_target": rebin_frequency,
        "spectrogram_polarization": polarization,
        "spectrogram_use_log10": _as_bool(
            form.get("use_log10"), default=polarization != "ratio"
        ),
        "spectrogram_cmap": cmap,
    }
    for source_name, target_name in (
        ("vmin", "spectrogram_vmin"),
        ("vmax", "spectrogram_vmax"),
    ):
        if form.get(source_name) not in (None, ""):
            cfg[target_name] = _bounded_float(
                form[source_name], minimum=-1.0e30, maximum=1.0e30
            )
    if (
        cfg.get("spectrogram_vmin") is not None
        and cfg.get("spectrogram_vmax") is not None
        and cfg["spectrogram_vmin"] >= cfg["spectrogram_vmax"]
    ):
        raise ValueError("vmin must be less than vmax")

    cache = spectrogram.build_spectrogram_cache(cfg)
    if cache is None or not cache.time_datetimes:
        raise RuntimeError("No compatible spectrogram data was available for Preview")

    traces: list[dict[str, Any]] = []
    time_values = np.asarray(
        [_utc_datetime(item).timestamp() for item in cache.time_datetimes],
        dtype=np.float64,
    )
    colorscale = cmap.capitalize() if cmap != "cividis" else "Cividis"
    for segment_index, segment in enumerate(coverage_segments):
        start_ts = _utc_datetime(segment["start_time_iso"]).timestamp()
        end_ts = _utc_datetime(segment["end_time_iso"]).timestamp()
        indices = np.flatnonzero((time_values >= start_ts) & (time_values <= end_ts))
        if not indices.size:
            continue
        z_values = np.asarray(cache.data[:, indices], dtype=np.float64)
        traces.append(
            {
                "type": "heatmap",
                "name": f"Observed segment {segment_index + 1}",
                "x": [_utc_iso(cache.time_datetimes[int(index)]) for index in indices],
                "y": [float(value) for value in np.asarray(cache.freq).ravel()],
                "z": [
                    [
                        float(value) if math.isfinite(float(value)) else None
                        for value in row
                    ]
                    for row in z_values.tolist()
                ],
                "colorscale": colorscale,
                "zmin": _finite_float_or_none(cache.vmin),
                "zmax": _finite_float_or_none(cache.vmax),
                "showscale": segment_index == 0,
                "connectgaps": False,
                "colorbar": {"title": cache.cbar_label},
                "hovertemplate": "%{x|%Y-%m-%d %H:%M:%S.%L UTC}<br>%{y:.3f} MHz<extra></extra>",
            }
        )
    if not traces:
        raise RuntimeError(
            "The stitched spectrogram contains no samples in real coverage"
        )

    gaps = [
        {
            "start_time_iso": coverage_segments[index]["end_time_iso"],
            "end_time_iso": coverage_segments[index + 1]["start_time_iso"],
        }
        for index in range(len(coverage_segments) - 1)
    ]
    figure = {
        "data": traces,
        "layout": {
            "title": {"text": cache.title},
            "xaxis": {
                "title": {"text": "Time (UTC)"},
                "type": "date",
                "range": [x_start_iso, x_end_iso],
            },
            "yaxis": {"title": {"text": "Frequency (MHz)"}},
            "margin": {"l": 74, "r": 36, "t": 58, "b": 62},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "showlegend": False,
        },
    }
    metadata = {
        "adapter": "spectrogram-coverage",
        "kind": "spectrogram",
        "title": cache.title,
        "x_start_iso": x_start_iso,
        "x_end_iso": x_end_iso,
        "coverage_segments": coverage_segments,
        "coverage_gaps": gaps,
        "source_count": len(paths),
        "source_names": [path.name for path in paths],
        "trace_count": len(traces),
        "frequency_min_mhz": float(np.nanmin(cache.freq)),
        "frequency_max_mhz": float(np.nanmax(cache.freq)),
        "x_axis_mapping": {
            "type": "utc-linear",
            "start_time_iso": x_start_iso,
            "end_time_iso": x_end_iso,
        },
        "interpolation": "none",
        "maximum_merged_gap_s": 1.0,
    }
    return {
        "adapter": "spectrogram-coverage",
        "status": "ready",
        "kind": "plotly",
        "title": cache.title,
        "figure": figure,
        "metadata": metadata,
        "x_start_iso": x_start_iso,
        "x_end_iso": x_end_iso,
        "coverage_segments": coverage_segments,
        "message": (
            "The explicit CSO Preview was rebuilt from validated inputs. "
            "Observed gaps longer than one second remain empty."
        ),
    }


def _json_path_list(value: Any) -> list[str]:
    if value in (None, "", [], ()):
        return []
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "adjacent_files_json must be a JSON array of FITS paths"
            ) from exc
    if not isinstance(decoded, (list, tuple)):
        raise TypeError("adjacent_files_json must be a JSON array of FITS paths")
    if len(decoded) > 31:
        raise ValueError("Select at most 31 adjacent spectrogram FITS files")
    paths: list[str] = []
    for item in decoded:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(
                "Every adjacent spectrogram path must be a non-empty string"
            )
        paths.append(item.strip())
    return paths


def _utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: Any) -> str:
    return _utc_datetime(value).isoformat().replace("+00:00", "Z")


def _build_roi_selection_preview(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    radio_dir = _required_validated_path(form, "radio_dir", validate_path=validate_path)
    if not radio_dir.is_dir():
        raise NotADirectoryError(f"Radio data folder does not exist: {radio_dir}")

    pattern = _safe_glob_pattern(form.get("pattern", "*.fits"))
    recursive = _as_bool(form.get("recursive", True), default=True)
    max_side = _bounded_int(form.get("max_side", 256), minimum=32, maximum=1024)
    low_percentile = _bounded_float(
        form.get("low_percentile", 1.0), minimum=0.0, maximum=100.0
    )
    high_percentile = _bounded_float(
        form.get("high_percentile", 99.7), minimum=0.0, maximum=100.0
    )
    if low_percentile >= high_percentile:
        raise ValueError("low_percentile must be less than high_percentile")
    roi_mode = str(form.get("roi_mode", "box") or "box").strip().lower()
    if roi_mode not in {"box", "lasso"}:
        raise ValueError("roi_mode must be 'box' or 'lasso'")

    # Deliberately lazy: importing these modules loads NumPy, Astropy, and Plotly.
    import numpy as np

    from solar_toolkit.radio.centers import iter_radio_images, select_radio_files
    from solar_toolkit.radio.roi_lightcurve import RadioRoi, build_radio_roi_mask
    from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_app import (
        build_reference_figure,
    )

    files = select_radio_files(
        radio_dir,
        pattern=pattern,
        recursive=recursive,
    )
    if not files:
        raise FileNotFoundError(
            f"No compatible FITS paths found under {radio_dir} matching {pattern!r}"
        )

    candidate_limit = 2000
    discovered_count = len(files)
    files = files[:candidate_limit]
    compatibility_probe = RadioRoi.from_box(-1.0, -1.0, 1.0, 1.0)
    candidates: list[dict[str, Any]] = []
    skipped: list[str] = []
    for discovered_path in files:
        source_path = _validated_path(discovered_path, validate_path=validate_path)
        if not source_path.is_file():
            skipped.append(f"{source_path.name}: not a file")
            continue

        images = iter_radio_images(source_path)
        compatible_item = None
        try:
            for item in images:
                try:
                    image = np.asarray(item.image)
                    if image.ndim != 2 or image.size == 0:
                        raise ValueError("not a non-empty 2D image")
                    if not bool(np.isfinite(image).any()):
                        raise ValueError("image contains no finite pixels")

                    # This public ROI API validates the HPLN/HPLT WCS contract.
                    build_radio_roi_mask(
                        item.header,
                        (1, 1),
                        compatibility_probe,
                    )
                except (TypeError, ValueError, RuntimeError):
                    continue
                compatible_item = item
                candidates.append(
                    {
                        "path": str(source_path),
                        "relative_path": _relative_text(source_path, radio_dir),
                        "name": source_path.name,
                        "hdu_index": int(item.hdu_index),
                        "image_shape": [int(value) for value in image.shape],
                        "polarization": str(item.pol),
                        "frequency_mhz": _finite_float_or_none(item.freq_mhz),
                        "observation_time": _iso_or_none(item.obs_time),
                    }
                )
                # The browser selects files, not individual HDUs. Use the first
                # ROI-compatible plane as that file's deterministic reference.
                break
        finally:
            close = getattr(images, "close", None)
            if callable(close):
                close()
        if compatible_item is None:
            skipped.append(f"{source_path.name}: no ROI-compatible image plane")

    if not candidates:
        last_reason = skipped[-1] if skipped else "no readable image plane"
        raise RuntimeError(
            "No ROI-compatible HPLN/HPLT radio image was found; "
            f"last reason: {last_reason}"
        )

    requested_paths = _selected_roi_preview_paths(
        form.get("selected_files_json"), validate_path=validate_path
    )
    if requested_paths is None:
        selected_indices = _representative_frequency_indices(candidates, maximum=9)
    else:
        candidate_indices = {
            Path(candidate["path"]).resolve(strict=False): index
            for index, candidate in enumerate(candidates)
        }
        selected_indices = []
        for requested in requested_paths:
            try:
                index = candidate_indices[requested]
            except KeyError as exc:
                raise ValueError(
                    "Every selected ROI preview file must be a compatible candidate "
                    "from the validated radio folder."
                ) from exc
            if index not in selected_indices:
                selected_indices.append(index)
        if not selected_indices:
            raise ValueError("Select at least one ROI reference file.")
        if len(selected_indices) > 9:
            raise ValueError("Select at most 9 ROI reference files.")

    selected_paths = [str(candidates[index]["path"]) for index in selected_indices]
    selected_path_set = set(selected_paths)
    for candidate in candidates:
        candidate["selected"] = candidate["path"] in selected_path_set

    selected_items = [
        _load_roi_candidate_image(
            candidates[index],
            validate_path=validate_path,
            compatibility_probe=compatibility_probe,
        )
        for index in selected_indices
    ]

    figures = [
        build_reference_figure(
            item,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            max_side=max_side,
            roi_mode=roi_mode,
            selection_enabled=True,
        )
        for item in selected_items
    ]
    figure = _build_roi_reference_grid(figures, selected_items, roi_mode=roi_mode)
    first_index = selected_indices[0]
    first_item = selected_items[0]
    first_candidate = candidates[first_index]
    return {
        "adapter": "roi-selection",
        "status": "ready",
        "kind": "plotly",
        "figure": _plotly_figure_dict(figure),
        "candidates": candidates,
        "selected_files": selected_paths,
        "selection": {
            "coordinate_system": "HPLN/HPLT arcsec",
            "default_mode": roi_mode,
            "supported_modes": ["box", "lasso"],
            "event_source": "plotly_selected",
            "x_field": "x",
            "y_field": "y",
        },
        "metadata": {
            "source_path": first_candidate["path"],
            "source_name": first_candidate["name"],
            "relative_path": first_candidate["relative_path"],
            "hdu_index": int(first_item.hdu_index),
            "image_shape": [int(value) for value in first_item.image.shape],
            "polarization": str(first_item.pol),
            "frequency_mhz": _finite_float_or_none(first_item.freq_mhz),
            "observation_time": _iso_or_none(first_item.obs_time),
            "pattern": pattern,
            "recursive": recursive,
            "candidate_count": len(candidates),
            "discovered_file_count": discovered_count,
            "candidate_limit": candidate_limit,
            "candidates_truncated": discovered_count > len(files),
            "selected_count": len(selected_items),
            "selection_limit": 9,
            "skipped_before_match": len(skipped),
            "selected_files_scope": "reference_preview_only",
        },
    }


def _load_roi_candidate_image(
    candidate: Mapping[str, Any],
    *,
    validate_path: PathValidator,
    compatibility_probe: Any,
) -> Any:
    """Reload one selected candidate without retaining all scanned FITS arrays."""

    import numpy as np

    from solar_toolkit.radio.centers import iter_radio_images
    from solar_toolkit.radio.roi_lightcurve import build_radio_roi_mask

    source_path = _validated_path(candidate["path"], validate_path=validate_path)
    expected_path = Path(str(candidate["path"])).resolve(strict=False)
    if source_path != expected_path:
        raise ValueError(
            "Validated ROI candidate path changed before preview rendering."
        )
    expected_hdu = int(candidate["hdu_index"])
    images = iter_radio_images(source_path)
    try:
        for item in images:
            if int(item.hdu_index) != expected_hdu:
                continue
            image = np.asarray(item.image)
            if image.ndim != 2 or image.size == 0 or not bool(np.isfinite(image).any()):
                break
            build_radio_roi_mask(item.header, (1, 1), compatibility_probe)
            return item
    finally:
        close = getattr(images, "close", None)
        if callable(close):
            close()
    raise RuntimeError(f"Selected ROI candidate is no longer readable: {source_path}")


def _selected_roi_preview_paths(
    raw_value: Any, *, validate_path: PathValidator
) -> list[Path] | None:
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "selected_files_json must be a JSON array of paths."
            ) from exc
    else:
        decoded = raw_value
    if not isinstance(decoded, list) or any(
        not isinstance(value, str) or not value.strip() for value in decoded
    ):
        raise ValueError("selected_files_json must be a JSON array of paths.")
    return [_validated_path(value, validate_path=validate_path) for value in decoded]


def _representative_frequency_indices(
    candidates: list[dict[str, Any]], *, maximum: int
) -> list[int]:
    selected: list[int] = []
    seen: set[float | str] = set()
    for index, candidate in enumerate(candidates):
        frequency = candidate.get("frequency_mhz")
        key: float | str = (
            round(float(frequency), 6) if frequency is not None else "unknown"
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(index)
        if len(selected) >= maximum:
            break
    return selected or [0]


def _build_roi_reference_grid(
    figures: list[Any], items: list[Any], *, roi_mode: str
) -> Any:
    if len(figures) == 1:
        return figures[0]

    from plotly.subplots import make_subplots

    count = len(figures)
    columns = min(3, count)
    rows = math.ceil(count / columns)
    titles = [
        (
            f"{_finite_float_or_none(item.freq_mhz):g} MHz | {item.pol}"
            if _finite_float_or_none(item.freq_mhz) is not None
            else f"Unknown frequency | {item.pol}"
        )
        for item in items
    ]
    grid = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for index, source in enumerate(figures):
        row = index // columns + 1
        column = index % columns + 1
        for trace in source.data:
            grid.add_trace(trace, row=row, col=column)
        axis_suffix = "" if index == 0 else str(index + 1)
        grid.update_xaxes(title_text="HPLN / arcsec", row=row, col=column)
        grid.update_yaxes(
            title_text="HPLT / arcsec",
            scaleanchor=f"x{axis_suffix}",
            scaleratio=1,
            row=row,
            col=column,
        )
    grid.update_layout(
        title="Multi-frequency ROI reference grid",
        dragmode="lasso" if str(roi_mode).lower() == "lasso" else "select",
        height=max(620, rows * 430),
        margin={"l": 55, "r": 20, "t": 85, "b": 50},
        showlegend=False,
    )
    return grid


def _build_trajectory_preview(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    centers_path = _required_validated_path(
        form, "centers", validate_path=validate_path
    )
    if not centers_path.is_file():
        raise FileNotFoundError(f"Center table does not exist: {centers_path}")
    if centers_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Center table must be CSV or XLSX/XLS")

    # Deliberately lazy: table readers and Plotly are only needed for this adapter.
    import pandas as pd

    from solar_toolkit.aia.background import scan_aia_folder
    from solar_apps.frontends.radio.source_trajectory.source_app import (
        build_preloaded_playback_payload,
    )
    from solar_toolkit.radio.trajectory import (
        filter_centers,
        frame_times,
        load_centers_table,
        select_visible_centers,
    )
    from solar_apps.workflows.visualization.radio_source_trajectory import (
        build_trajectory_figure,
    )

    centers = load_centers_table(
        centers_path,
        valid_only=_as_bool(form.get("valid_only", True), default=True),
    )
    centers = filter_centers(
        centers,
        freqs=_float_list(form.get("freqs")),
        polarizations=_text_list(form.get("polarizations")),
        center_methods=_text_list(form.get("center_methods")),
    )
    if centers.empty:
        raise ValueError("Center table contains no valid rows after filtering")

    times = frame_times(centers)
    if not times:
        raise ValueError("Center table contains no usable observation times")
    frame_time = _select_frame_time(times, form)
    mode = str(form.get("frame_mode", form.get("mode", "all")) or "all").strip().lower()
    tail_n = _bounded_int(form.get("tail_n", 5), minimum=1, maximum=10000)
    visible = select_visible_centers(
        centers,
        frame_time,
        mode=mode,
        tail_n=tail_n,
    )
    if visible.empty:
        raise ValueError("Selected trajectory frame contains no visible centers")

    figure, compare_table = build_trajectory_figure(
        visible,
        frame_time,
        draw_lines=_as_bool(form.get("draw_lines", True), default=True),
        compare_lr=_as_bool(form.get("compare_lr", False), default=False),
        compare_tolerance_sec=float(form.get("compare_tolerance_sec", 1.0)),
        height=_bounded_int(form.get("height", 760), minimum=320, maximum=2400),
        theme_mode=str(form.get("theme_mode", form.get("theme", "auto")) or "auto"),
        use_webgl=_as_bool(form.get("use_webgl", True), default=True),
        plot_layout=str(form.get("plot_layout", "overlay") or "overlay"),
        facet_by=str(form.get("facet_by", "freq_mhz") or "freq_mhz"),
        marker_size=_bounded_int(form.get("marker_size", 9), minimum=1, maximum=64),
    )
    playback_times = _sample_times(
        times,
        _bounded_int(form.get("max_playback_frames", 600), minimum=1, maximum=2000),
    )
    use_aia = _as_bool(form.get("use_aia", False), default=False)
    aia_table = None
    if use_aia:
        aia_dir = _required_validated_path(form, "aia_dir", validate_path=validate_path)
        if not aia_dir.is_dir():
            raise NotADirectoryError(f"AIA folder does not exist: {aia_dir}")
        aia_table = scan_aia_folder(
            aia_dir,
            pattern=str(form.get("aia_pattern", "*.fits") or "*.fits"),
        )
        if aia_table.empty:
            raise ValueError("AIA folder contains no time-resolvable FITS files")
    playback = build_preloaded_playback_payload(
        centers,
        playback_times,
        frame_mode=mode,
        tail_n=tail_n,
        aia_table=aia_table,
        use_aia=use_aia,
        max_aia_dt_sec=_bounded_float(
            form.get("max_aia_dt_sec", 3600.0), minimum=0.1, maximum=86400.0
        ),
        playback_aia_max_pixels=_bounded_int(
            form.get("aia_max_pixels", 384), minimum=64, maximum=2048
        ),
        theme_mode=str(form.get("theme_mode", form.get("theme", "auto")) or "auto"),
        draw_lines=_as_bool(form.get("draw_lines", True), default=True),
        fps=_bounded_float(form.get("fps", 2.0), minimum=0.2, maximum=60.0),
        plot_layout=str(form.get("plot_layout", "overlay") or "overlay"),
        facet_by=str(form.get("facet_by", "freq_mhz") or "freq_mhz"),
        marker_size=_bounded_int(form.get("marker_size", 9), minimum=1, maximum=64),
    )
    selected_timestamp = pd.Timestamp(frame_time)
    return {
        "adapter": "trajectory-media",
        "status": "ready",
        "kind": "plotly",
        "figure": _plotly_figure_dict(figure),
        "playback": playback,
        "metadata": {
            "source_path": str(centers_path),
            "source_name": centers_path.name,
            "row_count": int(len(centers)),
            "visible_row_count": int(len(visible)),
            "frame_count": int(len(times)),
            "playback_frame_count": int(len(playback_times)),
            "playback_sampled": len(playback_times) < len(times),
            "frame_time": selected_timestamp.isoformat(),
            "first_frame_time": pd.Timestamp(times[0]).isoformat(),
            "last_frame_time": pd.Timestamp(times[-1]).isoformat(),
            "mode": mode,
            "tail_n": tail_n,
            "lr_comparison_count": int(len(compare_table)),
        },
    }


def _build_drift_selection_preview(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    preview_path = _required_validated_path(
        form, "spectrogram_image", validate_path=validate_path
    )
    metadata_path = _required_validated_path(
        form, "spectrogram_metadata", validate_path=validate_path
    )
    if not preview_path.is_file():
        raise FileNotFoundError(f"Spectrogram image does not exist: {preview_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Spectrogram metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Spectrogram metadata must be a JSON object")
    x_start = metadata.get("x_start_iso") or metadata.get("spectrogram_time_start")
    x_end = metadata.get("x_end_iso") or metadata.get("spectrogram_time_end")
    f_min = metadata.get("f_min_mhz", metadata.get("spectrogram_f_start"))
    f_max = metadata.get("f_max_mhz", metadata.get("spectrogram_f_end"))
    if x_start in (None, "") or x_end in (None, ""):
        raise ValueError("Spectrogram metadata is missing x_start_iso/x_end_iso")
    try:
        f_min_value = float(f_min)
        f_max_value = float(f_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Spectrogram metadata is missing finite frequency bounds"
        ) from exc
    if not all(math.isfinite(value) for value in (f_min_value, f_max_value)):
        raise ValueError("Spectrogram frequency bounds must be finite")
    if f_min_value > f_max_value:
        f_min_value, f_max_value = f_max_value, f_min_value

    mime_type = mimetypes.guess_type(preview_path.name)[0] or "image/png"
    source = f"data:{mime_type};base64," + base64.b64encode(
        preview_path.read_bytes()
    ).decode("ascii")
    figure = {
        "data": [
            {
                "type": "heatmap",
                "x": [str(x_start), str(x_end)],
                "y": [f_min_value, f_max_value],
                "z": [[0.0, 0.0], [0.0, 0.0]],
                "opacity": 0.01,
                "showscale": False,
                "hovertemplate": "Time: %{x}<br>Frequency: %{y:.2f} MHz<extra></extra>",
            }
        ],
        "layout": {
            "title": "Drift-rate endpoint selection",
            "height": 650,
            "xaxis": {"title": "Time (UT)", "range": [str(x_start), str(x_end)]},
            "yaxis": {
                "title": "Frequency (MHz)",
                "range": [f_min_value, f_max_value],
            },
            "images": [
                {
                    "source": source,
                    "xref": "x",
                    "yref": "y",
                    "x": str(x_start),
                    "y": f_max_value,
                    "sizex": _datetime_span_milliseconds(x_start, x_end),
                    "sizey": f_max_value - f_min_value,
                    "xanchor": "left",
                    "yanchor": "top",
                    "sizing": "stretch",
                    "layer": "below",
                }
            ],
            "clickmode": "event",
        },
    }
    return {
        "adapter": "drift-selection",
        "status": "ready",
        "kind": "plotly",
        "figure": figure,
        "selection": {
            "mode": "two-point-lines",
            "event_source": "continuous-plot-click",
            "output_field": "drift_lines_json",
        },
        "metadata": {
            "source_path": str(preview_path),
            "metadata_path": str(metadata_path),
            "x_start_iso": str(x_start),
            "x_end_iso": str(x_end),
            "f_min_mhz": f_min_value,
            "f_max_mhz": f_max_value,
        },
    }


def _build_file_browser(
    form: Mapping[str, Any], *, validate_path: PathValidator
) -> dict[str, Any]:
    raw_path = form.get("path") or form.get("radio_dir")
    if raw_path in (None, ""):
        return {
            "adapter": "file-browser",
            "status": "unavailable",
            "kind": "file-browser",
            "items": [],
            "reason": "Choose a local folder to browse.",
        }

    root = _validated_path(raw_path, validate_path=validate_path)
    if not root.exists():
        raise FileNotFoundError(f"Browse path does not exist: {root}")
    limit = _bounded_int(form.get("limit", 200), minimum=1, maximum=1000)
    candidates = (
        [root]
        if root.is_file()
        else sorted(
            root.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())
        )
    )
    selected = candidates[:limit]
    items = [_file_item(path, root=root) for path in selected]
    return {
        "adapter": "file-browser",
        "status": "ready",
        "kind": "file-browser",
        "items": items,
        "metadata": {
            "root": str(root),
            "count": len(items),
            "total_count": len(candidates),
            "truncated": len(candidates) > len(items),
            "limit": limit,
        },
    }


def _unavailable_index(adapter: str) -> dict[str, Any]:
    label = "run manifests" if adapter == "run-index" else "artifact manifests"
    return {
        "adapter": adapter,
        "status": "unavailable",
        "kind": adapter,
        "items": [],
        "reason": f"{label.capitalize()} require an injected workspace store.",
    }


def _required_validated_path(
    form: Mapping[str, Any], key: str, *, validate_path: PathValidator
) -> Path:
    value = form.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required path field: {key}")
    return _validated_path(value, validate_path=validate_path)


def _validated_path(value: Any, *, validate_path: PathValidator) -> Path:
    validated = validate_path(value)
    resolved_value = value if validated is None else validated
    return Path(str(resolved_value)).expanduser().resolve(strict=False)


def _safe_glob_pattern(value: Any) -> str:
    pattern = str(value or "*.fits").strip()
    if not pattern:
        raise ValueError("FITS pattern cannot be empty")
    path = Path(pattern)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError("FITS pattern must stay within the validated radio folder")
    return pattern


def _select_frame_time(times: list[Any], form: Mapping[str, Any]) -> Any:
    requested = form.get("frame_time")
    if requested not in (None, ""):
        import pandas as pd

        timestamp = pd.Timestamp(requested)
        if pd.isna(timestamp):
            raise ValueError(f"Invalid frame_time: {requested!r}")
        return timestamp
    index = int(form.get("frame_index", -1))
    if index < 0:
        index += len(times)
    if not 0 <= index < len(times):
        raise IndexError(f"frame_index is outside 0..{len(times) - 1}")
    return times[index]


def _sample_times(times: list[Any], maximum: int) -> list[Any]:
    """Bound browser payload size while preserving the first and last frames."""

    if len(times) <= maximum:
        return list(times)
    if maximum == 1:
        return [times[-1]]
    step = (len(times) - 1) / (maximum - 1)
    indices = [round(index * step) for index in range(maximum)]
    return [times[index] for index in dict.fromkeys(indices)]


def _plotly_figure_dict(figure: Any) -> dict[str, Any]:
    """Return a plain mapping accepted by the standard-library JSON encoder."""

    return json.loads(figure.to_json())


def _file_item(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    if path.is_dir():
        kind = "directory"
        size: int | None = None
    elif path.is_file():
        kind = "file"
        size = int(stat.st_size)
    else:
        kind = "other"
        size = int(stat.st_size)
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": _relative_text(path, root),
        "kind": kind,
        "size": size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "suffix": path.suffix.lower(),
    }


def _relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    method = getattr(value, "isoformat", None)
    return str(method()) if callable(method) else str(value)


def _datetime_span_milliseconds(start: Any, end: Any) -> float:
    def parse(value: Any) -> datetime:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    span = (parse(end) - parse(start)).total_seconds() * 1000.0
    if not math.isfinite(span) or span <= 0:
        raise ValueError("Spectrogram time bounds must be increasing")
    return span


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"Expected an integer between {minimum} and {maximum}")
    return number


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"Expected a finite number between {minimum} and {maximum}")
    return number


def _text_list(value: Any) -> list[str] | None:
    if value in (None, "", [], ()):
        return None
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    result = [str(item).strip() for item in items if str(item).strip()]
    return result or None


def _float_list(value: Any) -> list[float] | None:
    items = _text_list(value)
    return [float(item) for item in items] if items else None
