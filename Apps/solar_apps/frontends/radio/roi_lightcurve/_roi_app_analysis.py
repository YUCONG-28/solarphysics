"""ROI request and result signatures and bounded session artifact caches.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
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
    build_radio_roi_artifacts,
    build_radio_roi_product_filenames,
)
from ._roi_app_helpers import _roi_app_facade


from ._roi_app_state import (
    _ANALYSIS_REQUEST_VERSION,
    _LIGHTCURVE_CACHE_SIZE,
    _LIGHTCURVE_PRODUCT_KEYS,
    _LIGHTCURVE_DEFAULT_MARKER_SIZE,
)


def _dataset_signature(settings: dict[str, Any], manifest: pd.DataFrame) -> str:
    return str(
        hash(
            (
                settings["radio_dir"],
                settings["pattern"],
                bool(settings["recursive"]),
                tuple(
                    float(item) for item in settings.get("selected_freqs_mhz", []) or []
                ),
                settings["time_start"],
                settings["time_end"],
                len(manifest),
            )
        )
    )


def _analysis_signature(
    selected_paths: list[str],
    roi: RadioRoi,
    settings: dict[str, Any],
    *,
    file_identities: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "version": _ANALYSIS_REQUEST_VERSION,
        "selected_files": (
            file_identities
            if file_identities is not None
            else _selected_file_identities(selected_paths)
        ),
        "roi": roi.to_json_dict(),
        "polarization": settings["polarization"],
        "pair_time_tolerance_sec": float(settings["pair_time_tolerance_sec"]),
    }
    return _stable_sha256(payload)


def _analysis_context_signature(
    selected_paths: list[str],
    roi: RadioRoi,
    settings: dict[str, Any],
    *,
    selection_token: Any | None = None,
) -> str:
    """Hash request controls without touching the filesystem."""

    return _stable_sha256(
        {
            "version": _ANALYSIS_REQUEST_VERSION,
            "selection": (
                selection_token
                if selection_token is not None
                else [str(path) for path in selected_paths]
            ),
            "roi": roi.to_json_dict(),
            "polarization": settings["polarization"],
            "pair_time_tolerance_sec": float(settings["pair_time_tolerance_sec"]),
        }
    )


def _selected_input_size_from_manifest(
    st: Any,
    selected_paths: list[str],
) -> tuple[int, int]:
    """Summarize selected bytes from the loaded manifest without live stat calls."""

    cache_key = _stable_sha256(
        {
            "dataset_signature": st.session_state.get("dataset_signature", ""),
            "selection_revision": int(st.session_state.get("selection_revision", 0)),
            "selected_count": len(selected_paths),
        }
    )
    cached = st.session_state.get("analysis_input_summary")
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return int(cached["size_bytes"]), int(cached["unknown_count"])
    manifest = st.session_state.get("loaded_manifest")
    if (
        not isinstance(manifest, pd.DataFrame)
        or manifest.empty
        or "path" not in manifest.columns
        or "size_bytes" not in manifest.columns
    ):
        result = (0, len(selected_paths))
    else:
        sizes_by_path = dict(
            zip(
                manifest["path"].astype(str),
                pd.to_numeric(manifest["size_bytes"], errors="coerce"),
                strict=False,
            )
        )
        sizes = [sizes_by_path.get(str(path), math.nan) for path in selected_paths]
        result = (
            sum(int(size) for size in sizes if pd.notna(size)),
            sum(pd.isna(size) for size in sizes),
        )
    st.session_state["analysis_input_summary"] = {
        "key": cache_key,
        "size_bytes": int(result[0]),
        "unknown_count": int(result[1]),
    }
    return result


def _selected_file_identities(
    paths: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return ordered live file identities for analysis cache invalidation."""

    return [_file_identity(path) for path in paths]


def _file_identity(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except OSError, RuntimeError:
        resolved = candidate.absolute()
    try:
        stat = resolved.stat()
    except OSError:
        size_bytes = None
        mtime_ns = None
    else:
        size_bytes = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    return {
        "path": str(resolved),
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
    }


def _dataframe_content_signature(df: pd.DataFrame) -> str:
    """Build a stable content signature for a materialized analysis table."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in df.columns],
                "dtypes": [str(dtype) for dtype in df.dtypes],
                "shape": list(df.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    try:
        row_hashes = pd.util.hash_pandas_object(
            df,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
        digest.update(row_hashes.tobytes())
    except TypeError, ValueError:
        digest.update(df.to_csv(index=True).encode("utf-8"))
    return digest.hexdigest()


def _reference_file_identities(
    st: Any,
    references: list[RadioImage],
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = [
        {
            "reference_grid_signature": str(
                st.session_state.get("reference_grid_signature", "")
            )
        }
    ]
    identities.extend(
        {
            "path": str(reference.path),
            "hdu_index": int(reference.hdu_index),
            "shape": [int(value) for value in reference.image.shape],
        }
        for reference in references
    )
    identities.extend(
        {
            "path": str(item.get("path", "")),
            "paired_path": str(item.get("paired_path", "")),
        }
        for item in (st.session_state.get("reference_metadata", []) or [])
    )
    return identities


def _export_signature(
    *,
    analysis_result_signature: str,
    product_keys: tuple[str, ...],
    metric: str,
    reference_identities: list[dict[str, Any]],
    display_config: dict[str, Any],
    y_axis_mode: str = "Full data",
    lightcurve_y_limits: tuple[float, float] | None = None,
    lightcurve_frequency_y_limits: (
        dict[float, tuple[float, float] | None] | None
    ) = None,
    lightcurve_frequency_config: list[dict[str, Any]] | None = None,
    lightcurve_marker_size: float = _LIGHTCURVE_DEFAULT_MARKER_SIZE,
    lightcurve_detail_frequency_mhz: float | None = None,
    lightcurve_plot_style: str = "scatter",
    lightcurve_y_transform: str = "linear",
) -> str:
    canonical_frequency_limits = _canonical_frequency_limit_items(
        lightcurve_frequency_y_limits
    )
    selected_product_set = set(product_keys)
    effective_detail_frequency = (
        lightcurve_detail_frequency_mhz
        if "lightcurve_detail_png" in selected_product_set
        else None
    )
    payload = {
        "version": "radio-roi-export-v5",
        "analysis_result_signature": str(analysis_result_signature),
        "products": sorted(set(product_keys)),
        "metric": str(metric),
        "reference_files": reference_identities,
        "display_config": display_config,
        "lightcurve_y_axis": {
            "mode": str(y_axis_mode),
            "limits": lightcurve_y_limits,
            "frequency_limits": canonical_frequency_limits,
            "frequency_config": lightcurve_frequency_config or [],
            "marker_size": float(lightcurve_marker_size),
            "detail_frequency_mhz": effective_detail_frequency,
            "plot_style": _roi_app_facade()._normalize_lightcurve_plot_style(
                lightcurve_plot_style
            ),
            "y_transform": _roi_app_facade()._normalize_lightcurve_y_transform(
                lightcurve_y_transform
            ),
        },
    }
    return _stable_sha256(payload)


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frequency_limit_items(
    frequency_y_limits: dict[float, tuple[float, float] | None] | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_frequency, raw_limits in (frequency_y_limits or {}).items():
        try:
            frequency = float(raw_frequency)
        except TypeError, ValueError:
            continue
        if not np.isfinite(frequency):
            continue
        limits = _roi_app_facade()._coerce_lightcurve_y_limits(raw_limits)
        items.append(
            {
                "freq_mhz": frequency,
                "limits": list(limits) if limits is not None else None,
            }
        )
    return sorted(items, key=lambda item: item["freq_mhz"])


def _cached_lightcurve_png(
    st: Any,
    df: pd.DataFrame,
    roi: RadioRoi,
    *,
    analysis_result_signature: str,
    metric: str,
    y_axis_mode: str = "Full data",
    lightcurve_y_limits: tuple[float, float] | None = None,
    lightcurve_frequency_y_limits: (
        dict[float, tuple[float, float] | None] | None
    ) = None,
    lightcurve_frequency_config: list[dict[str, Any]] | None = None,
    lightcurve_marker_size: float = _LIGHTCURVE_DEFAULT_MARKER_SIZE,
    lightcurve_detail_frequency_mhz: float | None = None,
    lightcurve_plot_style: str = "scatter",
    lightcurve_y_transform: str = "linear",
    product_key: str = "lightcurve_png",
) -> bytes:
    if product_key not in _LIGHTCURVE_PRODUCT_KEYS:
        raise ValueError(f"Unsupported light-curve preview product: {product_key}")
    normalized_limits = _roi_app_facade()._coerce_lightcurve_y_limits(
        lightcurve_y_limits
    )
    normalized_plot_style = _roi_app_facade()._normalize_lightcurve_plot_style(
        lightcurve_plot_style
    )
    normalized_y_transform = _roi_app_facade()._normalize_lightcurve_y_transform(
        lightcurve_y_transform
    )
    canonical_frequency_limits = _canonical_frequency_limit_items(
        lightcurve_frequency_y_limits
    )
    effective_detail_frequency = (
        lightcurve_detail_frequency_mhz
        if product_key == "lightcurve_detail_png"
        else None
    )
    cache_frequency_limits = canonical_frequency_limits
    cache_frequency_config = lightcurve_frequency_config or []
    if effective_detail_frequency is not None:
        cache_frequency_limits = [
            item
            for item in canonical_frequency_limits
            if np.isclose(
                float(item["freq_mhz"]),
                float(effective_detail_frequency),
                rtol=0.0,
                atol=1e-6,
            )
        ]
        cache_frequency_config = [
            item
            for item in (lightcurve_frequency_config or [])
            if np.isclose(
                float(item.get("freq_mhz", np.nan)),
                float(effective_detail_frequency),
                rtol=0.0,
                atol=1e-6,
            )
        ]
    cache_key = _stable_sha256(
        {
            "analysis_result_signature": analysis_result_signature,
            "metric": metric,
            "y_axis_mode": str(y_axis_mode),
            "lightcurve_y_limits": normalized_limits,
            "frequency_limits": cache_frequency_limits,
            "frequency_config": cache_frequency_config,
            "marker_size": float(lightcurve_marker_size),
            "detail_frequency_mhz": effective_detail_frequency,
            "plot_style": normalized_plot_style,
            "y_transform": normalized_y_transform,
            "product_key": product_key,
            "version": "lightcurve-preview-v4",
        }
    )
    cache = st.session_state.setdefault("lightcurve_png_cache", {})
    cached = cache.get(cache_key)
    if isinstance(cached, bytes):
        return cached
    artifact_kwargs: dict[str, Any] = {
        "metric": metric,
        "lightcurve_y_limits": normalized_limits,
        "lightcurve_plot_style": normalized_plot_style,
        "lightcurve_y_transform": normalized_y_transform,
        "selected_products": (product_key,),
    }
    if canonical_frequency_limits:
        artifact_kwargs.update(
            {
                "lightcurve_frequency_y_limits": {
                    float(
                        item["freq_mhz"]
                    ): _roi_app_facade()._coerce_lightcurve_y_limits(item["limits"])
                    for item in canonical_frequency_limits
                },
                "lightcurve_marker_size": float(lightcurve_marker_size),
                "lightcurve_detail_frequency_mhz": (
                    float(effective_detail_frequency)
                    if effective_detail_frequency is not None
                    else None
                ),
            }
        )
    payload = build_radio_roi_artifacts(df, roi, **artifact_kwargs)[product_key]
    cache[cache_key] = payload
    while len(cache) > _LIGHTCURVE_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    return payload


def _build_cached_export_artifacts(
    st: Any,
    df: pd.DataFrame,
    roi: RadioRoi,
    *,
    selected_paths: list[str],
    references: list[RadioImage],
    settings: dict[str, Any],
    display_config: dict[str, Any],
    product_keys: tuple[str, ...],
    lightcurve_y_axis: dict[str, Any] | None = None,
) -> dict[str, bytes]:
    artifact_filenames = build_radio_roi_product_filenames(
        df,
        selected_products=product_keys,
        generated_at=datetime.now(timezone.utc),
    )
    st.session_state["export_artifact_filenames"] = artifact_filenames
    y_axis_config = dict(lightcurve_y_axis or {})
    y_axis_mode = str(y_axis_config.get("mode", "Full data"))
    lightcurve_y_limits = _roi_app_facade()._coerce_lightcurve_y_limits(
        y_axis_config.get("limits")
    )
    frequency_config = _roi_app_facade()._canonical_frequency_configs(y_axis_config)
    frequency_y_limits = _roi_app_facade()._frequency_limit_mapping(y_axis_config)
    marker_size = float(
        y_axis_config.get("marker_size", _LIGHTCURVE_DEFAULT_MARKER_SIZE)
    )
    detail_frequency = y_axis_config.get("detail_frequency_mhz")
    plot_style = _roi_app_facade()._normalize_lightcurve_plot_style(
        y_axis_config.get("plot_style", "scatter")
    )
    y_transform = _roi_app_facade()._normalize_lightcurve_y_transform(
        y_axis_config.get("y_transform", "linear")
    )
    plot_metadata: dict[str, Any] = {}
    if "plot_style" in y_axis_config:
        plot_metadata["plot_style"] = plot_style
    if "y_transform" in y_axis_config:
        plot_metadata["y_transform"] = y_transform
    if "metric_unit" in y_axis_config:
        plot_metadata["metric_unit"] = y_axis_config["metric_unit"]
    base_keys = tuple(
        key for key in product_keys if key not in _LIGHTCURVE_PRODUCT_KEYS
    )
    artifacts: dict[str, bytes] = {}
    if base_keys:
        artifacts.update(
            build_radio_roi_artifacts(
                df,
                roi,
                reference_images=references,
                display_config=display_config,
                run_metadata=_roi_app_facade()._run_metadata(
                    selected_paths,
                    {
                        **_roi_app_facade()._settings_with_reference(
                            st, settings, display_config
                        ),
                        "lightcurve_y_axis": (
                            {
                                "mode": y_axis_mode,
                                "limits": (
                                    list(lightcurve_y_limits)
                                    if lightcurve_y_limits is not None
                                    else None
                                ),
                                **plot_metadata,
                                "marker_size": marker_size,
                                "detail_frequency_mhz": detail_frequency,
                                "normalization": y_axis_config.get(
                                    "normalization",
                                    "per-frequency displayed Y limits mapped to 0-1",
                                ),
                                "frequencies": frequency_config,
                            }
                            if frequency_config
                            else {
                                "mode": y_axis_mode,
                                "limits": (
                                    list(lightcurve_y_limits)
                                    if lightcurve_y_limits is not None
                                    else None
                                ),
                                **plot_metadata,
                            }
                        ),
                    },
                ),
                metric=str(settings["metric"]),
                lightcurve_y_limits=lightcurve_y_limits,
                lightcurve_frequency_y_limits=frequency_y_limits,
                lightcurve_marker_size=marker_size,
                lightcurve_detail_frequency_mhz=detail_frequency,
                lightcurve_plot_style=plot_style,
                lightcurve_y_transform=y_transform,
                selected_products=base_keys,
                artifact_filenames={key: artifact_filenames[key] for key in base_keys},
            )
        )
    for product_key in _LIGHTCURVE_PRODUCT_KEYS:
        if product_key not in product_keys:
            continue
        preview_kwargs: dict[str, Any] = {
            "analysis_result_signature": str(
                st.session_state.get("analysis_result_signature")
                or _dataframe_content_signature(df)
            ),
            "metric": str(settings["metric"]),
            "y_axis_mode": y_axis_mode,
            "lightcurve_y_limits": lightcurve_y_limits,
        }
        if "plot_style" in y_axis_config:
            preview_kwargs["lightcurve_plot_style"] = plot_style
        if "y_transform" in y_axis_config:
            preview_kwargs["lightcurve_y_transform"] = y_transform
        if frequency_config:
            preview_kwargs.update(
                {
                    "lightcurve_frequency_y_limits": frequency_y_limits,
                    "lightcurve_frequency_config": frequency_config,
                    "lightcurve_marker_size": marker_size,
                    "lightcurve_detail_frequency_mhz": detail_frequency,
                    "product_key": product_key,
                }
            )
        elif product_key != "lightcurve_png":
            preview_kwargs["product_key"] = product_key
        artifacts[product_key] = _cached_lightcurve_png(
            st,
            df,
            roi,
            **preview_kwargs,
        )
    if "json" in artifacts:
        selection = json.loads(artifacts["json"].decode("utf-8"))
        selection["outputs"] = {
            key: artifact_filenames[key]
            for key in PRODUCT_FILENAMES
            if key in product_keys
        }
        artifacts["json"] = json.dumps(
            selection,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
    return {key: artifacts[key] for key in PRODUCT_FILENAMES if key in product_keys}


__all__ = [
    "_dataset_signature",
    "_analysis_signature",
    "_analysis_context_signature",
    "_selected_input_size_from_manifest",
    "_selected_file_identities",
    "_file_identity",
    "_dataframe_content_signature",
    "_reference_file_identities",
    "_export_signature",
    "_stable_sha256",
    "_canonical_frequency_limit_items",
    "_cached_lightcurve_png",
    "_build_cached_export_artifacts",
]
