"""Reference previews, light-curve presentation and prepared artifact exports.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_toolkit.radio.centers import (
    RadioImage,
)
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
)
from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_application import (
    PRODUCT_FILENAMES,
)
from solar_apps.workflows.radio.spatial_display import spatial_display_for_reference
from ._roi_app_helpers import (
    _expanded_lightcurve_limits,
)
from ._roi_app_helpers import _roi_app_facade


from ._roi_app_state import (
    _LIGHTCURVE_Y_AXIS_MODES,
    _LIGHTCURVE_ROBUST_MIN_SAMPLES,
    _ReferencePreview,
    _DisplayReferencePreview,
)


def build_reference_figure(
    item: RadioImage,
    *,
    roi: RadioRoi | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.7,
    max_side: int = 256,
    roi_mode: str = "box",
    display_config: dict[str, Any] | None = None,
    selection_enabled: bool = True,
):
    """Build the Plotly reference image used for ROI selection."""

    raw_preview = _prepare_reference_preview(item, max_side=max_side)
    preview = _DisplayReferencePreview(
        display_view=_display_array(raw_preview.raw_view, display_config),
        x_arcsec=raw_preview.x_arcsec,
        y_arcsec=raw_preview.y_arcsec,
    )
    return _roi_app_facade()._build_reference_figure_from_preview(
        item,
        preview,
        roi=roi,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        roi_mode=roi_mode,
        display_config=display_config,
        selection_enabled=selection_enabled,
    )


def _build_reference_figure_from_preview(
    item: RadioImage,
    preview: _DisplayReferencePreview,
    *,
    roi: RadioRoi | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.7,
    roi_mode: str = "box",
    display_config: dict[str, Any] | None = None,
    selection_enabled: bool = True,
):
    import plotly.graph_objects as go

    display_view = preview.display_view
    x_arcsec = preview.x_arcsec
    y_arcsec = preview.y_arcsec
    display_contract = spatial_display_for_reference(display_config)
    zmin, zmax = _display_limits_for_item(
        item,
        display_view,
        display_config,
        fallback_percentiles=(float(low_percentile), float(high_percentile)),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=display_view,
            x=x_arcsec[0, :],
            y=y_arcsec[:, 0],
            colorscale=display_contract.cmap,
            zmin=float(zmin),
            zmax=float(zmax),
            showscale=False,
            hovertemplate=(
                "x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{z:.4g} "
                f"{_display_colorbar_title(item, display_config)}<extra></extra>"
            ),
        )
    )
    if selection_enabled:
        fig.add_trace(
            go.Scattergl(
                x=x_arcsec.ravel(),
                y=y_arcsec.ravel(),
                mode="markers",
                marker={"size": 4, "opacity": 0.01, "color": "white"},
                name="Selection grid",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if roi is not None:
        _add_roi_shape(fig, roi)
    dragmode: str | bool = (
        ("lasso" if str(roi_mode).lower() == "lasso" else "select")
        if selection_enabled
        else False
    )
    fig.update_layout(
        title=_reference_title(item),
        xaxis_title="HPLN / arcsec",
        yaxis_title="HPLT / arcsec",
        dragmode=dragmode,
        height=620,
        margin={"l": 60, "r": 20, "t": 60, "b": 55},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _apply_plotly_fov(fig, display_config)
    return fig


def _normalize_lightcurve_plot_style(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"scatter", "line"}:
        raise ValueError(f"Unsupported light-curve plot style: {value!r}")
    return normalized


def _normalize_lightcurve_y_transform(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "log": "log10",
        "log10 positive": "log10",
        "log10-positive": "log10",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"linear", "log10"}:
        raise ValueError(f"Unsupported light-curve Y transform: {value!r}")
    return normalized


def _lightcurve_metric_unit(df: pd.DataFrame, metric: str) -> str:
    units = []
    if "bunit" in df.columns:
        units = [
            str(value).strip()
            for value in df["bunit"].dropna().unique()
            if str(value).strip()
        ]
    base_unit = units[0] if units else "FITS unit"
    if metric == "raw_sum":
        return f"{base_unit} * pixel"
    return base_unit


def _lightcurve_axis_label(
    df: pd.DataFrame,
    metric: str,
    y_transform: str,
) -> str:
    unit = _lightcurve_metric_unit(df, metric)
    if _normalize_lightcurve_y_transform(y_transform) == "log10":
        return f"log10({metric} / 1 [{unit}])"
    return f"{metric} ({unit})"


def _lightcurve_metric_frame(
    df: pd.DataFrame,
    metric: str,
    *,
    y_transform: str = "linear",
) -> pd.DataFrame:
    """Return the finite, time-resolved rows used by the light-curve plot."""

    if metric not in df.columns:
        raise ValueError(f"Light-curve metric is not present in the analysis: {metric}")
    data = df.copy()
    data["obs_time_dt"] = pd.to_datetime(data.get("obs_time"), errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    if "quality_flag" in data.columns:
        quality_ok = data["quality_flag"].astype(str).str.lower().eq("ok")
    else:
        quality_ok = pd.Series(True, index=data.index)
    values = data[metric].to_numpy(dtype=float, na_value=np.nan)
    valid = quality_ok & data["obs_time_dt"].notna() & np.isfinite(values)
    transform = _normalize_lightcurve_y_transform(y_transform)
    if transform == "log10":
        valid &= values > 0.0
    result = data.loc[valid].copy()
    if transform == "log10":
        result[metric] = np.log10(result[metric].to_numpy(dtype=float))
    return result


def _full_lightcurve_y_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return _expanded_lightcurve_limits(float(np.min(finite)), float(np.max(finite)))


def _robust_lightcurve_y_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    full_limits = _full_lightcurve_y_limits(finite)
    if full_limits is None or finite.size < _LIGHTCURVE_ROBUST_MIN_SAMPLES:
        return full_limits
    lower, upper = np.quantile(finite, [0.001, 0.999])
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
        return full_limits
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    central_span = float(q75 - q25)
    if central_span > 0 and float(upper - lower) > 100.0 * central_span:
        # A few finite calibration failures can occupy more than 0.1% of a
        # short single-frequency sequence. Keep the documented percentile
        # rule as the first pass, then use a wider 1% tail guard only when the
        # first-pass span is still two orders of magnitude above the IQR.
        guarded_lower, guarded_upper = np.quantile(finite, [0.01, 0.99])
        if np.isfinite(guarded_lower) and np.isfinite(guarded_upper):
            if guarded_lower < guarded_upper:
                lower, upper = guarded_lower, guarded_upper
    padding = 0.05 * float(upper - lower)
    return float(lower - padding), float(upper + padding)


def _coerce_lightcurve_y_limits(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        lower, upper = value
        lower = float(lower)
        upper = float(upper)
    except TypeError, ValueError:
        return None
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
        return None
    return lower, upper


def _resolve_lightcurve_y_limits(
    values: np.ndarray,
    mode: str,
    *,
    manual_limits: tuple[float, float] | None = None,
    previous_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Resolve display-only Y limits without changing scientific samples."""

    normalized_mode = str(mode)
    if normalized_mode not in _LIGHTCURVE_Y_AXIS_MODES:
        raise ValueError(f"Unsupported Y-axis range mode: {mode!r}")
    full_limits = _full_lightcurve_y_limits(values)
    robust_limits = _robust_lightcurve_y_limits(values)
    if normalized_mode == "Full data":
        return {
            "mode": normalized_mode,
            "limits": None,
            "display_limits": full_limits,
            "full_limits": full_limits,
            "robust_limits": robust_limits,
            "valid": True,
            "used_fallback": False,
        }
    if normalized_mode == "Robust auto":
        return {
            "mode": normalized_mode,
            "limits": robust_limits,
            "display_limits": robust_limits,
            "full_limits": full_limits,
            "robust_limits": robust_limits,
            "valid": True,
            "used_fallback": robust_limits == full_limits,
        }

    resolved_manual = _coerce_lightcurve_y_limits(manual_limits)
    if resolved_manual is not None:
        return {
            "mode": normalized_mode,
            "limits": resolved_manual,
            "display_limits": resolved_manual,
            "full_limits": full_limits,
            "robust_limits": robust_limits,
            "valid": True,
            "used_fallback": False,
        }
    fallback = _coerce_lightcurve_y_limits(previous_limits) or robust_limits
    return {
        "mode": normalized_mode,
        "limits": fallback,
        "display_limits": fallback,
        "full_limits": full_limits,
        "robust_limits": robust_limits,
        "valid": False,
        "used_fallback": True,
    }


def _lightcurve_diagnostics(
    df: pd.DataFrame,
    metric: str,
    limits: tuple[float, float] | None,
) -> dict[str, Any]:
    data = _lightcurve_metric_frame(df, metric)
    values = data[metric].to_numpy(dtype=float)
    full_limits = _full_lightcurve_y_limits(values)
    robust_limits = _robust_lightcurve_y_limits(values)
    display_limits = _coerce_lightcurve_y_limits(limits) or full_limits
    outside_mask = np.zeros(values.shape, dtype=bool)
    if display_limits is not None:
        outside_mask = (values < display_limits[0]) | (values > display_limits[1])
    outside = data.loc[outside_mask].copy()
    if not outside.empty and display_limits is not None:
        lower, upper = display_limits
        outside["_distance"] = np.maximum(
            lower - outside[metric].to_numpy(dtype=float),
            outside[metric].to_numpy(dtype=float) - upper,
        )
        outside = outside.sort_values("_distance", ascending=False).drop(
            columns="_distance"
        )
    columns = [
        column
        for column in (
            "obs_time",
            "freq_mhz",
            "polarization",
            metric,
            "filepath",
            "paired_filepath",
        )
        if column in outside.columns
    ]
    full_span = (
        float(full_limits[1] - full_limits[0]) if full_limits is not None else 0.0
    )
    robust_span = (
        float(robust_limits[1] - robust_limits[0]) if robust_limits is not None else 0.0
    )
    span_ratio = full_span / robust_span if robust_span > 0 else 1.0
    return {
        "valid_count": int(values.size),
        "negative_count": int(np.count_nonzero(values < 0)),
        "outside_count": int(np.count_nonzero(outside_mask)),
        "full_limits": full_limits,
        "robust_limits": robust_limits,
        "display_limits": display_limits,
        "span_ratio": float(span_ratio),
        "outside_rows": outside.loc[:, columns].head(20),
    }


def _lightcurve_frequencies(data: pd.DataFrame) -> list[float]:
    if "freq_mhz" not in data.columns:
        return []
    values = pd.to_numeric(data["freq_mhz"], errors="coerce").to_numpy(dtype=float)
    return sorted(float(value) for value in np.unique(values[np.isfinite(values)]))


def _frequency_rows(data: pd.DataFrame, freq_mhz: float) -> pd.DataFrame:
    values = pd.to_numeric(data.get("freq_mhz"), errors="coerce").to_numpy(dtype=float)
    return data.loc[np.isclose(values, float(freq_mhz), rtol=0.0, atol=1e-9)].copy()


def _canonical_frequency_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("frequencies", [])
    canonical: list[dict[str, Any]] = []
    for entry in entries if isinstance(entries, list) else []:
        try:
            freq_mhz = float(entry["freq_mhz"])
        except KeyError, TypeError, ValueError:
            continue
        if not np.isfinite(freq_mhz):
            continue
        limits = _coerce_lightcurve_y_limits(entry.get("limits"))
        display_limits = _coerce_lightcurve_y_limits(entry.get("display_limits"))
        canonical.append(
            {
                "freq_mhz": freq_mhz,
                "mode": str(entry.get("mode", "Robust auto")),
                "limits": list(limits) if limits is not None else None,
                "display_limits": (
                    list(display_limits) if display_limits is not None else None
                ),
                "valid": bool(entry.get("valid", True)),
                "outside_count": int(entry.get("outside_count", 0)),
            }
        )
    return sorted(canonical, key=lambda item: item["freq_mhz"])


def _frequency_limit_mapping(
    config: dict[str, Any],
) -> dict[float, tuple[float, float] | None]:
    return {
        float(entry["freq_mhz"]): _coerce_lightcurve_y_limits(entry.get("limits"))
        for entry in _canonical_frequency_configs(config)
    }


def _attach_auto_display_limits(
    reference_images: list[RadioImage],
    config: dict[str, Any],
    *,
    previews: list[_DisplayReferencePreview],
) -> None:
    if str(config.get("range_mode", "")).lower() != "auto percentile":
        return
    low = float(config.get("low_percentile", 1.0))
    high = float(config.get("high_percentile", 99.7))
    by_freq: dict[str, list[np.ndarray]] = {}
    for item, preview in zip(reference_images, previews, strict=True):
        finite = preview.display_view[np.isfinite(preview.display_view)]
        if finite.size:
            by_freq.setdefault(_display_frequency_key(item.freq_mhz), []).append(finite)
    if not by_freq:
        return
    if config.get("range_scope") == "Shared/global":
        all_values = np.concatenate(
            [np.concatenate(values) for values in by_freq.values()]
        )
        config["shared_limits"] = _clean_display_limits(
            np.nanpercentile(all_values, [low, high])
        )
        return
    config["limits_by_frequency"] = {
        key: _clean_display_limits(
            np.nanpercentile(np.concatenate(values), [low, high])
        )
        for key, values in by_freq.items()
    }


def _display_array(
    arr: np.ndarray, display_config: dict[str, Any] | None
) -> np.ndarray:
    return spatial_display_for_reference(display_config).transformed(arr)


def _display_limits_for_item(
    item: RadioImage,
    display_view: np.ndarray,
    display_config: dict[str, Any] | None,
    *,
    fallback_percentiles: tuple[float, float],
) -> tuple[float, float]:
    config = display_config or {}
    freq_key = _display_frequency_key(item.freq_mhz)
    limits_by_frequency = config.get("limits_by_frequency", {})
    if isinstance(limits_by_frequency, dict) and freq_key in limits_by_frequency:
        return _clean_display_limits(limits_by_frequency[freq_key])
    if config.get("shared_limits") is not None:
        return _clean_display_limits(config["shared_limits"])
    contract_config = dict(config)
    contract_config.setdefault("low_percentile", fallback_percentiles[0])
    contract_config.setdefault("high_percentile", fallback_percentiles[1])
    contract = spatial_display_for_reference(contract_config)
    raw_view = (
        np.power(10.0, display_view) if contract.transform == "log10" else display_view
    )
    return contract.display_limits(raw_view, band=freq_key)


def _transform_display_limit(value: float, display_config: dict[str, Any]) -> float:
    transform = str(display_config.get("transform", "Linear")).strip().lower()
    if transform in {"log10 positive", "log10", "log"}:
        return math.log10(value) if value > 0.0 else math.nan
    return float(value)


def _clean_display_limits(values: Any) -> list[float]:
    zmin, zmax = [float(item) for item in list(values)[:2]]
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return [0.0, 1.0]
    if zmin > zmax:
        zmin, zmax = zmax, zmin
    if zmin == zmax:
        pad = abs(zmin) * 0.01 or 1.0
        zmin -= pad
        zmax += pad
    return [float(zmin), float(zmax)]


def _display_frequency_key(freq_mhz: float) -> str:
    if not np.isfinite(freq_mhz):
        return "nan"
    return f"{float(freq_mhz):.6g}"


def _display_colorbar_title(
    item: RadioImage, display_config: dict[str, Any] | None
) -> str:
    unit = str(item.header.get("BUNIT", "")).strip() or "raw"
    transform = str((display_config or {}).get("transform", "Linear")).strip().lower()
    if transform in {"log10 positive", "log10", "log"}:
        return f"log10({unit})"
    return unit


def _apply_plotly_fov(fig: Any, display_config: dict[str, Any] | None) -> None:
    contract = spatial_display_for_reference(display_config)
    if contract.fov is None:
        return
    left, right, bottom, top = contract.fov
    fig.update_xaxes(range=[min(left, right), max(left, right)])
    fig.update_yaxes(range=[min(bottom, top), max(bottom, top)])


def _write_prepared_artifacts(
    artifacts: dict[str, bytes],
    output_dir: str | Path,
    *,
    filenames: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write prepared bytes to one independently allocated run directory."""

    unknown = sorted(set(artifacts) - set(PRODUCT_FILENAMES))
    if unknown:
        raise ValueError(f"Unknown prepared export products: {unknown}")
    if not artifacts:
        raise ValueError("No prepared export products are available to save.")
    for key, payload in artifacts.items():
        if not isinstance(payload, bytes):
            raise TypeError(f"Prepared artifact {key!r} is not bytes.")
    target_dir = _allocate_unique_run_directory(Path(output_dir).expanduser())
    resolved_names = dict(filenames or PRODUCT_FILENAMES)
    products: dict[str, Path] = {"output_dir": target_dir}
    for key in PRODUCT_FILENAMES:
        if key not in artifacts:
            continue
        payload = artifacts[key]
        path = target_dir / resolved_names[key]
        path.write_bytes(payload)
        products[key] = path
    return products


def _allocate_unique_run_directory(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = [f"radio_roi_lightcurve_{stamp}"] + [
        f"radio_roi_lightcurve_{stamp}_{index:03d}" for index in range(2, 1000)
    ]
    for name in names:
        candidate = base / name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a unique output directory under {base}")


def _run_metadata(
    selected_paths: list[str], settings: dict[str, Any]
) -> dict[str, Any]:
    return {
        "radio_dir": settings["radio_dir"],
        "pattern": settings["pattern"],
        "recursive": bool(settings["recursive"]),
        "selected_freqs_mhz": settings.get("selected_freqs_mhz", []),
        "time_start": settings.get("time_start", ""),
        "time_end": settings.get("time_end", ""),
        "selected_files": selected_paths,
        "selected_file_count": len(selected_paths),
        "selected_file_numbers": settings.get("selected_file_numbers", []),
        "reference_file": settings.get("reference_path", ""),
        "reference_images": settings.get("reference_images", []),
        "anchor_file_number": settings.get("reference_file_number", ""),
        "primary_reference_freq_mhz": settings.get("primary_reference_freq_mhz", ""),
        "display_config": settings.get("display_config", {}),
        "polarization": settings["polarization"],
        "pair_time_tolerance_sec": float(settings["pair_time_tolerance_sec"]),
        "metric": settings["metric"],
        "lightcurve_y_axis": settings.get("lightcurve_y_axis", {}),
    }


def _settings_with_reference(
    st: Any, settings: dict[str, Any], display_config: dict[str, Any]
) -> dict[str, Any]:
    enriched = dict(settings)
    enriched["reference_path"] = st.session_state.get("reference_path", "")
    enriched["reference_file_number"] = st.session_state.get(
        "reference_file_number", ""
    )
    enriched["primary_reference_freq_mhz"] = st.session_state.get(
        "primary_reference_freq_mhz", ""
    )
    enriched["reference_images"] = st.session_state.get("reference_metadata", [])
    enriched["display_config"] = display_config
    manifest = st.session_state.get("loaded_manifest")
    if manifest is not None:
        path_to_row = dict(
            zip(manifest["path"].astype(str), manifest["row"].astype(int), strict=False)
        )
        enriched["selected_file_numbers"] = [
            path_to_row.get(str(path))
            for path in st.session_state.get("selected_paths", [])
        ]
    return enriched


def _zip_artifacts(
    artifacts: dict[str, bytes],
    filenames: dict[str, str] | None = None,
) -> bytes:
    resolved_names = dict(filenames or PRODUCT_FILENAMES)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, payload in artifacts.items():
            archive.writestr(resolved_names[key], payload)
    return buffer.getvalue()


def _downsample_for_preview(
    arr: np.ndarray, *, max_side: int
) -> tuple[np.ndarray, slice, slice]:
    ny, nx = arr.shape
    stride = max(1, int(np.ceil(max(ny, nx) / max(1, int(max_side)))))
    y_slice = slice(0, ny, stride)
    x_slice = slice(0, nx, stride)
    return arr[y_slice, x_slice], y_slice, x_slice


def _preview_coordinate_grid(
    item: RadioImage,
    preview_shape: tuple[int, int],
    y_slice: slice,
    x_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    from solar_toolkit.radio.roi_lightcurve import _pixel_coordinates_hpc_arcsec

    y_indices = np.arange(item.image.shape[0])[y_slice][: preview_shape[0]]
    x_indices = np.arange(item.image.shape[1])[x_slice][: preview_shape[1]]
    y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing="ij")
    return _pixel_coordinates_hpc_arcsec(item.header, x_grid, y_grid)


def _prepare_reference_preview(
    item: RadioImage,
    *,
    max_side: int,
) -> _ReferencePreview:
    arr = np.asarray(item.image, dtype=float)
    view, y_slice, x_slice = _downsample_for_preview(arr, max_side=max_side)
    x_arcsec, y_arcsec = _preview_coordinate_grid(
        item,
        view.shape,
        y_slice,
        x_slice,
    )
    x_arcsec.setflags(write=False)
    y_arcsec.setflags(write=False)
    return _ReferencePreview(
        raw_view=view,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
    )


def _reference_previews_from_state(
    st: Any,
    reference_images: list[RadioImage],
    *,
    max_side: int,
) -> list[_ReferencePreview]:
    identity = tuple(
        (
            str(item.path),
            int(item.hdu_index),
            tuple(int(value) for value in item.image.shape),
            id(item.image),
        )
        for item in reference_images
    )
    cache_key = (
        str(st.session_state.get("reference_grid_signature", "")),
        int(max_side),
        identity,
    )
    cached = st.session_state.get("reference_preview_cache")
    if (
        st.session_state.get("reference_preview_cache_key") == cache_key
        and isinstance(cached, list)
        and len(cached) == len(reference_images)
    ):
        return cached
    previews = [
        _prepare_reference_preview(item, max_side=max_side) for item in reference_images
    ]
    st.session_state["reference_preview_cache_key"] = cache_key
    st.session_state["reference_preview_cache"] = previews
    return previews


def _add_roi_shape(fig: Any, roi: RadioRoi) -> None:
    vertices = list(roi.vertices_arcsec)
    xs = [item[0] for item in vertices] + [vertices[0][0]]
    ys = [item[1] for item in vertices] + [vertices[0][1]]
    fig.add_trace(
        {
            "type": "scatter",
            "x": xs,
            "y": ys,
            "mode": "lines",
            "line": {"color": "white", "width": 2},
            "name": "Active ROI",
            "hoverinfo": "skip",
        }
    )


def _reference_title(item: RadioImage) -> str:
    time_label = (
        item.obs_time.isoformat(timespec="milliseconds")
        if item.obs_time
        else "unknown time"
    )
    freq_label = (
        f"{item.freq_mhz:g} MHz" if np.isfinite(item.freq_mhz) else "unknown frequency"
    )
    source = str(getattr(item, "source_label", "") or "").strip()
    prefix = f"{source} | " if source and source.lower() != "main" else ""
    return f"{prefix}{freq_label} {item.pol} {time_label}"


__all__ = [
    "build_reference_figure",
    "_build_reference_figure_from_preview",
    "_normalize_lightcurve_plot_style",
    "_normalize_lightcurve_y_transform",
    "_lightcurve_metric_unit",
    "_lightcurve_axis_label",
    "_lightcurve_metric_frame",
    "_full_lightcurve_y_limits",
    "_robust_lightcurve_y_limits",
    "_coerce_lightcurve_y_limits",
    "_resolve_lightcurve_y_limits",
    "_lightcurve_diagnostics",
    "_lightcurve_frequencies",
    "_frequency_rows",
    "_canonical_frequency_configs",
    "_frequency_limit_mapping",
    "_attach_auto_display_limits",
    "_display_array",
    "_display_limits_for_item",
    "_transform_display_limit",
    "_clean_display_limits",
    "_display_frequency_key",
    "_display_colorbar_title",
    "_apply_plotly_fov",
    "_write_prepared_artifacts",
    "_allocate_unique_run_directory",
    "_run_metadata",
    "_settings_with_reference",
    "_zip_artifacts",
    "_downsample_for_preview",
    "_preview_coordinate_grid",
    "_prepare_reference_preview",
    "_reference_previews_from_state",
    "_add_roi_shape",
    "_reference_title",
]
