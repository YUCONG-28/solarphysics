"""Gaussian fitting helpers extracted from radio source-map plotting."""

from __future__ import annotations

import csv
import datetime
import math
import os
import time
import warnings
from dataclasses import dataclass

import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import OptimizeWarning, curve_fit

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d

from . import gaussian_masks as _gaussian_mask_helpers
from .coordinates import (
    coordinate_roundtrip_error_pixel,
    pixel_to_data_coord,
    unravel_2d_index,
)
from .gaussian_background import _safe_rms_map, estimate_background_noise
from .gaussian_models import (
    elliptical_gaussian_2d_with_constant_bg,
    elliptical_gaussian_2d_with_plane_bg,
    gaussian_only_from_popt,
)
from .io import (
    GAUSSIAN_DIAGNOSTIC_FIELDS,
    MULTI_GAUSSIAN_DIAGNOSTIC_FIELDS,
)
from .io import plot_output_subdir as _plot_output_subdir

__all__ = [
    "GaussianFitResult",
    "GaussianSourceCandidate",
    "MultiGaussianFitResult",
    "_acquire_csv_lock",
    "_attach_candidate_metadata",
    "_attach_gaussian_fit_metadata",
    "_candidate_detection_masks",
    "_candidate_failure_row",
    "_center_peak_distance_arcsec",
    "_fit_failure_warning",
    "_gaussian_fit_diag_defaults",
    "_gaussian_fwhm_arcsec",
    "_gaussian_quality_config",
    "_limit_fit_pixels",
    "_local_peak_coordinates",
    "_multi_source_limit",
    "_release_csv_lock",
    "_requested_multi_source_count",
    "_roi_slices_from_mask",
    "_safe_rms_map",
    "_select_peak_connected_mask",
    "_set_gaussian_failure_diag",
    "_unravel_2d_index",
    "_update_gaussian_quality",
    "_watershed_candidate_labels",
    "_weighted_moment_initial_guess",
    "create_source_mask",
    "detect_gaussian_source_candidates",
    "elliptical_gaussian_2d",
    "elliptical_gaussian_2d_with_constant_bg",
    "elliptical_gaussian_2d_with_plane_bg",
    "estimate_background_noise",
    "fit_elliptical_gaussian_on_radio_image",
    "fit_multiple_gaussians_on_radio_image",
    "gaussian_only_from_popt",
    "multi_gaussian_diagnostics_rows",
    "overlay_gaussian_fit_on_axis",
    "overlay_multi_gaussian_fit_on_axis",
    "save_gaussian_diagnostics_row",
    "save_multi_gaussian_diagnostics_row",
]

_unravel_2d_index = unravel_2d_index
_select_peak_connected_mask = _gaussian_mask_helpers._select_peak_connected_mask
create_source_mask = _gaussian_mask_helpers.create_source_mask
_GAUSSIAN_WARNING_COUNTS = {}


@dataclass
class GaussianFitResult:
    model: np.ndarray
    gaussian_only_model: np.ndarray
    center_pixel: tuple[float, float]
    center_arcsec: tuple[float, float]
    sigma_pixel: tuple[float, float]
    theta_rad: float
    amplitude: float
    background_level: float | None
    noise_sigma: float | None
    snr: float | None
    residual_rms: float | None
    quality_flag: str
    covariance: np.ndarray | None
    mask_pixel_count: int
    source_file: str | None = None


@dataclass
class GaussianSourceCandidate:
    source_id: int
    rank: int
    peak_pixel: tuple[float, float]
    peak_arcsec: tuple[float, float]
    peak_value: float
    detection_snr: float | None
    mask_pixel_count: int
    mask: np.ndarray


@dataclass
class MultiGaussianFitResult:
    candidates: list[GaussianSourceCandidate]
    fit_results: list[GaussianFitResult]
    primary_result: GaussianFitResult | None
    source_count_mode: str
    requested_source_count: int | None
    detected_source_count: int
    missing_source_count: int
    failure_rows: list[dict]


# Canonical models, background estimators, and masks are imported above.


def _requested_multi_source_count(cfg):
    raw = cfg.get("multi_gaussian_source_count")
    if raw in (None, ""):
        return None
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return None
    return max(count, 1)


def _multi_source_limit(cfg):
    requested = _requested_multi_source_count(cfg)
    if requested is not None:
        return requested
    return max(int(cfg.get("multi_gaussian_max_sources", 3)), 1)


def _candidate_detection_masks(work, cfg, background_map=None, rms_map=None):
    finite = np.isfinite(work)
    finite_data = work[finite]
    if finite_data.size == 0:
        return None

    peak = float(np.nanmax(finite_data))
    min_peak_fraction = float(cfg.get("multi_gaussian_min_peak_fraction", 0.30))
    grow_peak_fraction = float(cfg.get("fit_grow_peak_fraction_threshold", 0.20))
    intensity_threshold = min_peak_fraction * peak
    grow_intensity_threshold = min(grow_peak_fraction, min_peak_fraction) * peak

    bg = None
    rms = None
    use_snr = False
    if (
        cfg.get("background_use_for_mask", True)
        and background_map is not None
        and rms_map is not None
    ):
        bg = np.asarray(background_map, dtype=np.float64)
        rms = _safe_rms_map(rms_map)
        use_snr = bg.shape == work.shape and rms.shape == work.shape

    if use_snr:
        snr_map = (work - bg) / rms
        core_mask = (
            finite
            & np.isfinite(snr_map)
            & (snr_map >= float(cfg.get("fit_snr_threshold", 5.0)))
            & (work > intensity_threshold)
        )
        grow_mask = (
            finite
            & np.isfinite(snr_map)
            & (snr_map >= float(cfg.get("fit_grow_snr_threshold", 3.0)))
            & (work > grow_intensity_threshold)
        )
    else:
        _, noise_sigma = estimate_background_noise(work)
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = max(float(np.nanstd(finite_data)), 1e-12)
        snr_map = (work - float(np.nanmedian(finite_data))) / noise_sigma
        core_threshold = max(
            float(cfg.get("fit_snr_threshold", 5.0)) * noise_sigma,
            intensity_threshold,
        )
        grow_threshold = max(
            float(cfg.get("fit_grow_snr_threshold", 3.0)) * noise_sigma,
            grow_intensity_threshold,
        )
        core_mask = finite & (work > core_threshold)
        grow_mask = finite & (work > grow_threshold)

    if not np.any(core_mask):
        return None
    return {
        "peak": peak,
        "core_mask": np.asarray(core_mask, dtype=np.bool_),
        "grow_mask": np.asarray(grow_mask | core_mask, dtype=np.bool_),
        "snr_map": snr_map,
    }


def _local_peak_coordinates(work, core_mask, min_distance, max_sources):
    peak_image = np.where(core_mask & np.isfinite(work), work, -np.inf)
    coords = None
    try:
        from skimage.feature import peak_local_max

        coords = peak_local_max(
            peak_image,
            min_distance=max(int(min_distance), 1),
            exclude_border=False,
            labels=np.asarray(core_mask, dtype=np.uint8),
            num_peaks=max(int(max_sources), 1),
        )
    except Exception:
        size = max(2 * int(min_distance) + 1, 3)
        local_max = core_mask & (peak_image == maximum_filter(peak_image, size=size))
        coords = np.column_stack(np.nonzero(local_max))

    if coords is None or len(coords) == 0:
        if not np.any(core_mask):
            return []
        y, x = _unravel_2d_index(int(np.nanargmax(peak_image)), work.shape)
        coords = np.asarray([[y, x]], dtype=np.intp)

    coord_list = [
        (int(y), int(x), float(work[int(y), int(x)]))
        for y, x in np.asarray(coords, dtype=np.intp)
        if np.isfinite(work[int(y), int(x)])
    ]
    coord_list.sort(key=lambda item: item[2], reverse=True)

    min_dist2 = float(max(int(min_distance), 1) ** 2)
    selected = []
    for y, x, value in coord_list:
        if all(
            (y - kept[0]) ** 2 + (x - kept[1]) ** 2 >= min_dist2 for kept in selected
        ):
            selected.append((y, x, value))
        if len(selected) >= int(max_sources):
            break
    return selected


def _watershed_candidate_labels(work, grow_mask, peak_coords, use_watershed):
    if len(peak_coords) == 0:
        return None
    if use_watershed and len(peak_coords) > 1:
        try:
            from skimage.segmentation import watershed

            markers = np.zeros(work.shape, dtype=np.int32)
            for idx, (y, x, _) in enumerate(peak_coords, start=1):
                markers[int(y), int(x)] = idx
            finite_work = np.where(
                np.isfinite(work), work, np.nanmin(work[np.isfinite(work)])
            )
            return watershed(-finite_work, markers=markers, mask=grow_mask)
        except Exception:
            pass

    labeled, _ = label(grow_mask)
    labels_out = np.zeros(work.shape, dtype=np.int32)
    used_component_labels = set()
    out_label = 1
    for y, x, _ in peak_coords:
        component_label = int(labeled[int(y), int(x)])
        if component_label <= 0 or component_label in used_component_labels:
            continue
        labels_out[labeled == component_label] = out_label
        used_component_labels.add(component_label)
        out_label += 1
    return labels_out


def detect_gaussian_source_candidates(
    data,
    extent,
    cfg,
    background_map=None,
    rms_map=None,
    image_origin=None,
    coordinate_header=None,
):
    from solar_toolkit._utils.numeric import numeric_data

    work = numeric_data(data)
    if work.ndim != 2:
        return []
    image_origin = image_origin or cfg.get("_current_radio_image_origin", "upper")
    masks = _candidate_detection_masks(
        work, cfg, background_map=background_map, rms_map=rms_map
    )
    if masks is None:
        return []

    max_sources = _multi_source_limit(cfg)
    min_distance = int(cfg.get("multi_gaussian_min_peak_distance_pixels", 6))
    peak_coords = _local_peak_coordinates(
        work, masks["core_mask"], min_distance, max_sources
    )
    labels_map = _watershed_candidate_labels(
        work,
        masks["grow_mask"],
        peak_coords,
        bool(cfg.get("multi_gaussian_use_watershed", True)),
    )
    if labels_map is None:
        return []

    candidates = []
    min_candidate_pixels = int(cfg.get("fit_min_mask_pixels", 20))
    for rank, (peak_y, peak_x, peak_value) in enumerate(peak_coords, start=1):
        label_value = int(labels_map[int(peak_y), int(peak_x)])
        if label_value <= 0:
            candidate_mask = np.zeros(work.shape, dtype=np.bool_)
            candidate_mask[int(peak_y), int(peak_x)] = True
        else:
            candidate_mask = np.asarray(labels_map == label_value, dtype=np.bool_)
        candidate_mask &= np.isfinite(work)
        mask_count = int(np.count_nonzero(candidate_mask))
        if mask_count < min_candidate_pixels:
            continue
        snr_value = masks["snr_map"][int(peak_y), int(peak_x)]
        detection_snr = float(snr_value) if np.isfinite(snr_value) else None
        peak_arcsec = pixel_to_data_coord(
            float(peak_x), float(peak_y), extent, work.shape, origin=image_origin
        )
        if coordinate_header is not None:
            from solar_toolkit.map.coordinates import hpc_pixel_coordinates

            peak_arcsec = tuple(
                float(value)
                for value in hpc_pixel_coordinates(coordinate_header, peak_x, peak_y)
            )
        candidates.append(
            GaussianSourceCandidate(
                source_id=rank,
                rank=rank,
                peak_pixel=(float(peak_x), float(peak_y)),
                peak_arcsec=peak_arcsec,
                peak_value=float(peak_value),
                detection_snr=detection_snr,
                mask_pixel_count=mask_count,
                mask=candidate_mask,
            )
        )
        if len(candidates) >= max_sources:
            break

    candidates.sort(key=lambda item: item.peak_value, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
        candidate.source_id = rank
    return candidates


def _fit_failure_warning(source_file, quality_flag, detail=""):
    key = str(quality_flag)
    _GAUSSIAN_WARNING_COUNTS[key] = _GAUSSIAN_WARNING_COUNTS.get(key, 0) + 1
    if _GAUSSIAN_WARNING_COUNTS[key] > 3:
        return
    name = os.path.basename(source_file) if source_file else "radio image"
    suffix = f" / {detail}" if detail else ""
    warnings.warn(
        f"Gaussian fit skipped for {name}: reason={quality_flag}{suffix}", stacklevel=2
    )


def _gaussian_fit_diag_defaults(cfg):
    return {
        "gaussian_fit_method": "skipped",
        "roi_used": False,
        "roi_shape": "",
        "fit_pixel_count_before_limit": 0,
        "fit_pixel_count_after_limit": 0,
        "maxfev": int(cfg.get("gaussian_fit_maxfev", 8000)),
        "initial_center_pixel": "",
        "initial_sigma_x_pixel": np.nan,
        "initial_sigma_y_pixel": np.nan,
        "normalization_scale": np.nan,
    }


def _roi_slices_from_mask(mask, shape, padding):
    if mask is None or not np.any(mask):
        return slice(0, shape[0]), slice(0, shape[1]), False
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return slice(0, shape[0]), slice(0, shape[1]), False
    pad = max(int(padding), 0)
    y0 = max(int(np.min(ys)) - pad, 0)
    y1 = min(int(np.max(ys)) + pad + 1, shape[0])
    x0 = max(int(np.min(xs)) - pad, 0)
    x1 = min(int(np.max(xs)) + pad + 1, shape[1])
    roi_used = (y0 > 0) or (x0 > 0) or (y1 < shape[0]) or (x1 < shape[1])
    return slice(y0, y1), slice(x0, x1), roi_used


def _weighted_moment_initial_guess(x, y, z, bg, nx, ny, peak_x, peak_y, cfg):
    weights = np.asarray(z, dtype=np.float64) - bg
    finite = np.isfinite(weights) & np.isfinite(x) & np.isfinite(y)
    weights = np.where(finite & (weights > 0), weights, 0.0)
    total = float(np.sum(weights))
    sigma_min = 1.0
    sigma_max = max(2.0, float(cfg.get("max_sigma_fraction", 0.5)) * max(nx, ny))
    if total > 0:
        cx = float(np.sum(x * weights) / total)
        cy = float(np.sum(y * weights) / total)
        var_x = float(np.sum(((x - cx) ** 2) * weights) / total)
        var_y = float(np.sum(((y - cy) ** 2) * weights) / total)
        sigma_x = math.sqrt(max(var_x, sigma_min**2))
        sigma_y = math.sqrt(max(var_y, sigma_min**2))
    else:
        cx, cy = float(peak_x), float(peak_y)
        sigma_x = max(min(nx, ny) / 12.0, sigma_min)
        sigma_y = sigma_x
    sigma_x = float(np.clip(sigma_x, sigma_min, sigma_max))
    sigma_y = float(np.clip(sigma_y, sigma_min, sigma_max))
    if not np.isfinite(cx) or not np.isfinite(cy):
        cx, cy = float(peak_x), float(peak_y)
    return cx, cy, sigma_x, sigma_y


def _limit_fit_pixels(x, y, z, peak_x, peak_y, max_pixels):
    count = int(z.size)
    max_pixels = int(max_pixels or 0)
    if max_pixels <= 0 or count <= max_pixels:
        return x, y, z, count, count
    intensity_rank = np.asarray(z, dtype=np.float64)
    finite_intensity = intensity_rank[np.isfinite(intensity_rank)]
    fill = float(np.nanmin(finite_intensity)) if finite_intensity.size else 0.0
    intensity_rank = np.where(np.isfinite(intensity_rank), intensity_rank, fill)
    dist2 = (x - peak_x) ** 2 + (y - peak_y) ** 2
    order = np.lexsort((dist2, -intensity_rank))
    keep = np.sort(order[:max_pixels])
    return x[keep], y[keep], z[keep], count, int(keep.size)


def _attach_gaussian_fit_metadata(result, cfg, mask_diag, fit_input_type, fit_meta):
    result.fit_input_type = fit_input_type
    result.background_use_for_mask = bool(cfg.get("background_use_for_mask", True))
    result.background_use_for_fit = bool(cfg.get("background_use_for_fit", False))
    result.background_strategy = cfg.get("radio_background_strategy", "noise_map_only")
    result.source_snr_peak = mask_diag.get("source_snr_peak", np.nan)
    result.source_snr_mean = mask_diag.get("source_snr_mean", np.nan)
    result.background_rms_median = mask_diag.get("background_rms_median", np.nan)
    result.background_level_median = mask_diag.get("background_level_median", np.nan)
    result.mask_method = mask_diag.get("mask_method", "")
    result.peak = mask_diag.get("peak", np.nan)
    result.threshold = mask_diag.get("threshold", np.nan)
    result.fit_peak_fraction_threshold_used = mask_diag.get(
        "fit_peak_fraction_threshold_used", np.nan
    )
    result.fit_peak_fraction_candidate_counts = mask_diag.get(
        "fit_peak_fraction_candidate_counts", ""
    )
    for key, value in fit_meta.items():
        setattr(result, key, value)
    return result


def _gaussian_fwhm_arcsec(fit_result, extent, img_shape):
    if hasattr(fit_result, "wcs_fwhm_arcsec"):
        return tuple(float(value) for value in fit_result.wcs_fwhm_arcsec)
    ny, nx = img_shape
    dx = abs((extent[1] - extent[0]) / max(nx, 1))
    dy = abs((extent[3] - extent[2]) / max(ny, 1))
    width = 2.355 * fit_result.sigma_pixel[0] * dx
    height = 2.355 * fit_result.sigma_pixel[1] * dy
    return float(width), float(height)


def _center_peak_distance_arcsec(fit_result):
    raw_center = getattr(fit_result, "raw_center_arcsec", None)
    if raw_center is None:
        return np.nan
    cx, cy = fit_result.center_arcsec
    rx, ry = raw_center
    fit_result.center_peak_dx_arcsec = float(cx - rx)
    fit_result.center_peak_dy_arcsec = float(cy - ry)
    return float(math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2))


def _gaussian_quality_config(cfg):
    quality_cfg = dict(cfg.get("gaussian_quality_requirements", {}) or {})
    quality_cfg.setdefault("require_quality_ok", True)
    quality_cfg.setdefault("max_fwhm_arcsec", cfg.get("max_fwhm_arcsec", 1800.0))
    quality_cfg.setdefault(
        "max_center_peak_distance_arcsec",
        cfg.get("max_center_peak_distance_arcsec", 300.0),
    )
    quality_cfg.setdefault("min_snr", cfg.get("fit_snr_threshold", 5.0))
    quality_cfg.setdefault("max_residual_rms_fraction", 0.8)
    return quality_cfg


def _update_gaussian_quality(fit_result, extent, img_shape, cfg):
    """Mark fits valid for display only when they satisfy compact-source criteria."""
    is_moment_fallback = fit_result.quality_flag == "moment_fallback"
    width, height = _gaussian_fwhm_arcsec(fit_result, extent, img_shape)
    fit_result.fwhm_width_arcsec = width
    fit_result.fwhm_height_arcsec = height
    fit_result.fwhm_major_arcsec = float(max(width, height))
    fit_result.fwhm_minor_arcsec = float(min(width, height))
    quality_cfg = _gaussian_quality_config(cfg)
    max_fwhm = float(
        quality_cfg.get("max_fwhm_arcsec", cfg.get("max_fwhm_arcsec", 1800.0))
    )
    fit_result.max_fwhm_arcsec = max_fwhm
    fit_result.fwhm_valid = bool(max(width, height) <= max_fwhm)
    fit_result.center_peak_distance_arcsec = _center_peak_distance_arcsec(fit_result)
    if not fit_result.fwhm_valid:
        fit_result.quality_flag = "unphysical_size"
        fit_result.quality_flag_detail = "skipped_large_fwhm"
        fit_result.overlay_valid = False
        fit_result.trajectory_valid = False
        return False
    max_dist = float(quality_cfg.get("max_center_peak_distance_arcsec", 300.0))
    max_dist = min(
        max_dist,
        float(cfg.get("gaussian_max_center_peak_distance_fraction_of_fwhm", 0.5))
        * fit_result.fwhm_minor_arcsec,
    )
    if (
        np.isfinite(fit_result.center_peak_distance_arcsec)
        and fit_result.center_peak_distance_arcsec > max_dist
    ):
        fit_result.quality_flag = "center_far_from_peak"
        fit_result.quality_flag_detail = "center_far_from_peak"
        fit_result.overlay_valid = False
        fit_result.trajectory_valid = False
        return False
    min_snr = float(quality_cfg.get("min_snr", cfg.get("fit_snr_threshold", 5.0)))
    if (
        fit_result.snr is not None
        and np.isfinite(fit_result.snr)
        and fit_result.snr < min_snr
    ):
        if is_moment_fallback:
            fit_result.quality_flag_detail = "low_snr"
            fit_result.overlay_valid = False
            fit_result.trajectory_valid = False
            return False
        fit_result.quality_flag = "low_snr"
        fit_result.quality_flag_detail = "low_snr"
        fit_result.overlay_valid = False
        fit_result.trajectory_valid = False
        return False
    max_resid = float(quality_cfg.get("max_residual_rms_fraction", 0.8))
    if (
        fit_result.residual_rms is not None
        and np.isfinite(fit_result.residual_rms)
        and np.isfinite(fit_result.amplitude)
        and abs(fit_result.amplitude) > 0
        and fit_result.residual_rms / abs(fit_result.amplitude) > max_resid
    ):
        fit_result.quality_flag = "high_residual"
        fit_result.quality_flag_detail = "high_residual"
        fit_result.overlay_valid = False
        fit_result.trajectory_valid = False
        return False
    if is_moment_fallback:
        fit_result.quality_flag_detail = getattr(
            fit_result, "quality_flag_detail", "moment_fallback"
        )
        overlay_valid = not quality_cfg.get("require_quality_ok", True)
        trajectory_valid = overlay_valid and bool(
            cfg.get("gaussian_allow_moment_fallback_for_trajectory", False)
        )
        fit_result.overlay_valid = (
            bool(overlay_valid)
            if cfg.get("gaussian_valid_only_for_overlay", True)
            else True
        )
        fit_result.trajectory_valid = (
            bool(trajectory_valid)
            if cfg.get("gaussian_valid_only_for_trajectory", True)
            else True
        )
        return bool(fit_result.overlay_valid)
    if quality_cfg.get("require_quality_ok", True) and fit_result.quality_flag != "ok":
        fit_result.overlay_valid = False
        fit_result.trajectory_valid = False
        return False
    valid = fit_result.quality_flag == "ok"
    fit_result.overlay_valid = (
        bool(valid) if cfg.get("gaussian_valid_only_for_overlay", True) else True
    )
    fit_result.trajectory_valid = (
        bool(valid) if cfg.get("gaussian_valid_only_for_trajectory", True) else True
    )
    return bool(fit_result.overlay_valid)


def _set_gaussian_failure_diag(
    cfg: dict, source_file, reason: str, mask_diag: dict | None = None, **extra
) -> None:
    mask_diag = mask_diag or {}
    diag = {
        "source_file": source_file or "",
        "reason": reason,
        "quality_flag": reason,
        "finite_pixel_count": extra.get("finite_pixel_count", ""),
        "mask_pixel_count": mask_diag.get(
            "mask_pixel_count", extra.get("mask_pixel_count", 0)
        ),
        "peak": mask_diag.get("peak", extra.get("peak", np.nan)),
        "background_level": mask_diag.get(
            "background_level", extra.get("background_level", np.nan)
        ),
        "noise_sigma": mask_diag.get("noise_sigma", extra.get("noise_sigma", np.nan)),
        "threshold": mask_diag.get("threshold", extra.get("threshold", np.nan)),
        "fit_input_type": extra.get(
            "fit_input_type", mask_diag.get("fit_input_type", "")
        ),
        "background_use_for_mask": extra.get(
            "background_use_for_mask", mask_diag.get("background_use_for_mask", "")
        ),
        "background_use_for_fit": extra.get(
            "background_use_for_fit", mask_diag.get("background_use_for_fit", "")
        ),
        "background_strategy": extra.get(
            "background_strategy", mask_diag.get("background_strategy", "")
        ),
        "source_snr_peak": mask_diag.get("source_snr_peak", np.nan),
        "source_snr_mean": mask_diag.get("source_snr_mean", np.nan),
        "background_rms_median": mask_diag.get("background_rms_median", np.nan),
        "background_level_median": mask_diag.get("background_level_median", np.nan),
        "mask_method": mask_diag.get("mask_method", ""),
        "fit_peak_fraction_threshold_used": mask_diag.get(
            "fit_peak_fraction_threshold_used",
            extra.get("fit_peak_fraction_threshold_used", np.nan),
        ),
        "fit_peak_fraction_candidate_counts": mask_diag.get(
            "fit_peak_fraction_candidate_counts",
            extra.get("fit_peak_fraction_candidate_counts", ""),
        ),
        "gaussian_fit_method": extra.get("gaussian_fit_method", "skipped"),
        "roi_used": extra.get("roi_used", False),
        "roi_shape": extra.get("roi_shape", ""),
        "fit_pixel_count_before_limit": extra.get("fit_pixel_count_before_limit", 0),
        "fit_pixel_count_after_limit": extra.get("fit_pixel_count_after_limit", 0),
        "maxfev": extra.get("maxfev", cfg.get("gaussian_fit_maxfev", 8000)),
        "initial_center_pixel": extra.get("initial_center_pixel", ""),
        "initial_sigma_x_pixel": extra.get("initial_sigma_x_pixel", np.nan),
        "initial_sigma_y_pixel": extra.get("initial_sigma_y_pixel", np.nan),
        "normalization_scale": extra.get("normalization_scale", np.nan),
    }
    cfg["_last_gaussian_failure_diag"] = diag


def fit_elliptical_gaussian_on_radio_image(
    data,
    extent,
    cfg,
    source_file=None,
    background_map=None,
    rms_map=None,
    fit_input_type="raw",
    image_origin=None,
    source_mask_override=None,
    coordinate_header=None,
):
    """Fit in native pixels, optionally calibrating output geometry with WCS.

    ``coordinate_header`` supplies FITS WCS for exact HPC centers/contours and
    local Jacobian-based angular widths. Without it, ``extent`` retains the
    historical affine coordinate convention. Masks are excluded; input data
    and headers are not modified. ``cfg`` receives fit diagnostics as before.
    """
    from solar_toolkit._utils.numeric import numeric_data

    work = numeric_data(data)
    image_origin = image_origin or cfg.get("_current_radio_image_origin", "upper")
    cfg.pop("_last_gaussian_failure_diag", None)
    fit_meta = _gaussian_fit_diag_defaults(cfg)
    if work.ndim != 2:
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_data",
            fit_input_type=fit_input_type,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_data", "non-2D data")
        return None
    finite_work = work[np.isfinite(work)]
    if finite_work.size == 0:
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_data",
            finite_pixel_count=0,
            fit_input_type=fit_input_type,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_data")
        return None
    peak = float(np.max(finite_work))

    ny, nx = work.shape
    y_pix = np.arange(ny, dtype=np.float64)
    x_pix = np.arange(nx, dtype=np.float64)
    X, Y = np.meshgrid(x_pix, y_pix)

    source_mask, mask_diag = create_source_mask(
        work, cfg, background_map=background_map, rms_map=rms_map
    )
    if source_mask_override is not None:
        override_mask = np.asarray(source_mask_override, dtype=np.bool_)
        if override_mask.shape != work.shape:
            raise ValueError(
                "source_mask_override shape must match the fitted radio image"
            )
        override_mask = np.asarray(override_mask & np.isfinite(work), dtype=np.bool_)
        override_count = int(np.count_nonzero(override_mask))
        source_mask = override_mask if override_count else None
        mask_diag["mask_pixel_count"] = override_count
        mask_diag["mask_method"] = (
            f"{mask_diag.get('mask_method', '')}+source_override"
            if mask_diag.get("mask_method")
            else "source_override"
        )
        if override_count:
            mask_values = work[override_mask]
            mask_diag["peak"] = float(np.nanmax(mask_values))
            if override_count >= int(cfg.get("fit_min_mask_pixels", 20)):
                mask_diag["quality_flag"] = "ok"
            else:
                mask_diag["quality_flag"] = "mask_too_small"
    mask_diag["finite_pixel_count"] = int(finite_work.size)
    mask_diag["fit_input_type"] = fit_input_type
    mask_diag["background_use_for_mask"] = bool(
        cfg.get("background_use_for_mask", True)
    )
    mask_diag["background_use_for_fit"] = bool(cfg.get("background_use_for_fit", False))
    mask_diag["background_strategy"] = cfg.get(
        "radio_background_strategy", "noise_map_only"
    )
    if cfg.get("fit_use_source_mask", True) and source_mask is None:
        if cfg.get("skip_low_quality_fit", True):
            reason = mask_diag.get("quality_flag", "mask_too_small")
            _set_gaussian_failure_diag(
                cfg,
                source_file,
                reason,
                mask_diag,
                finite_pixel_count=finite_work.size,
                **fit_meta,
            )
            _fit_failure_warning(source_file, reason)
            return None
        fit_mask = np.isfinite(work)
    elif cfg.get("fit_use_source_mask", True) and source_mask is not None:
        source_mask_bool = np.asarray(source_mask, dtype=np.bool_)
        fit_mask = source_mask_bool & np.isfinite(work)
    else:
        fit_mask = np.isfinite(work)

    if cfg.get("gaussian_fit_use_roi", True):
        roi_source = source_mask if source_mask is not None else fit_mask
        y_slice, x_slice, roi_used = _roi_slices_from_mask(
            roi_source, work.shape, cfg.get("gaussian_fit_roi_padding_pixels", 10)
        )
    else:
        y_slice, x_slice, roi_used = slice(0, ny), slice(0, nx), False
    x_offset = int(x_slice.start or 0)
    y_offset = int(y_slice.start or 0)
    roi_work = work[y_slice, x_slice]
    roi_fit_mask = fit_mask[y_slice, x_slice]
    roi_ny, roi_nx = roi_work.shape
    fit_meta["roi_used"] = bool(roi_used)
    fit_meta["roi_shape"] = f"{roi_ny}x{roi_nx}"

    y_roi = np.arange(roi_ny, dtype=np.float64)
    x_roi = np.arange(roi_nx, dtype=np.float64)
    X_roi, Y_roi = np.meshgrid(x_roi, y_roi)
    xy_fit = (X_roi[roi_fit_mask].ravel(), Y_roi[roi_fit_mask].ravel())
    z_fit = roi_work[roi_fit_mask].ravel()
    finite = np.isfinite(z_fit) & np.isfinite(xy_fit[0]) & np.isfinite(xy_fit[1])
    xy_fit = (xy_fit[0][finite], xy_fit[1][finite])
    z_fit = z_fit[finite]
    fit_meta["fit_pixel_count_before_limit"] = int(z_fit.size)
    mask_diag["mask_pixel_count"] = int(z_fit.size)
    if z_fit.size < int(cfg.get("fit_min_mask_pixels", 20)):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "mask_too_small",
            mask_diag,
            finite_pixel_count=finite_work.size,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "mask_too_small")
        return None

    local_peak = float(np.nanmax(z_fit))
    local_bg = float(mask_diag.get("background_level", np.nan))
    if not np.isfinite(local_bg):
        local_bg = float(np.nanmedian(z_fit))
    A0 = max(local_peak - local_bg, 1e-12)
    if not np.isfinite(A0):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_initial_guess",
            mask_diag,
            finite_pixel_count=finite_work.size,
            peak=peak,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_initial_guess")
        return None
    local_peak_idx = int(np.nanargmax(z_fit))
    peak_x = float(xy_fit[0][local_peak_idx])
    peak_y = float(xy_fit[1][local_peak_idx])
    xy_x, xy_y, z_fit, before_limit, after_limit = _limit_fit_pixels(
        xy_fit[0],
        xy_fit[1],
        z_fit,
        peak_x,
        peak_y,
        cfg.get("gaussian_fit_max_pixels", 5000),
    )
    xy_fit = (xy_x, xy_y)
    fit_meta["fit_pixel_count_before_limit"] = int(before_limit)
    fit_meta["fit_pixel_count_after_limit"] = int(after_limit)
    mask_diag["mask_pixel_count"] = int(after_limit)
    if z_fit.size < int(cfg.get("fit_min_mask_pixels", 20)):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "mask_too_small",
            mask_diag,
            finite_pixel_count=finite_work.size,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "mask_too_small")
        return None

    cx0, cy0, sigma_x0, sigma_y0 = _weighted_moment_initial_guess(
        xy_fit[0], xy_fit[1], z_fit, local_bg, roi_nx, roi_ny, peak_x, peak_y, cfg
    )
    fit_meta["initial_center_pixel"] = f"{cx0 + x_offset:.3f},{cy0 + y_offset:.3f}"
    fit_meta["initial_sigma_x_pixel"] = float(sigma_x0)
    fit_meta["initial_sigma_y_pixel"] = float(sigma_y0)

    if cfg.get("gaussian_fit_normalize_data", True):
        centered = z_fit - local_bg
        finite_centered = centered[np.isfinite(centered)]
        norm_scale = (
            float(np.nanpercentile(np.abs(finite_centered), 99))
            if finite_centered.size
            else np.nan
        )
        if not np.isfinite(norm_scale) or norm_scale <= 0:
            norm_scale = max(abs(A0), 1.0)
        z_curve = centered / norm_scale
        A0_curve = max(A0 / norm_scale, 1e-6)
        bg0_curve = 0.0
        bg_abs_limit = 10.0
    else:
        norm_scale = 1.0
        z_curve = z_fit
        A0_curve = A0
        bg0_curve = local_bg
        bg_abs_limit = max(abs(local_peak) * 10.0, abs(local_bg) * 10.0, 1.0)
    fit_meta["normalization_scale"] = float(norm_scale)

    background_model = cfg.get("fit_background_model", "constant")
    model_map = {
        "none": elliptical_gaussian_2d,
        "constant": elliptical_gaussian_2d_with_constant_bg,
        "plane": elliptical_gaussian_2d_with_plane_bg,
    }
    model_func = model_map.get(background_model, elliptical_gaussian_2d_with_plane_bg)
    if background_model not in model_map:
        background_model = "constant"
        model_func = model_map[background_model]

    sigma_upper = max(
        2.0, float(cfg.get("max_sigma_fraction", 0.5)) * max(roi_nx, roi_ny)
    )
    amp_upper = max(abs(A0_curve) * 10.0, 2.0)
    slope_limit = bg_abs_limit / max(roi_nx, roi_ny, 1)

    p0 = [A0_curve, float(cx0), float(cy0), sigma_x0, sigma_y0, 0.0]
    lower = [0.0, 0.0, 0.0, 0.5, 0.5, -np.pi / 2]
    upper = [amp_upper, roi_nx - 1, roi_ny - 1, sigma_upper, sigma_upper, np.pi / 2]
    if background_model == "constant":
        p0 += [bg0_curve]
        lower += [-bg_abs_limit]
        upper += [bg_abs_limit]
    elif background_model == "plane":
        p0 += [bg0_curve, 0.0, 0.0]
        lower += [-bg_abs_limit, -slope_limit, -slope_limit]
        upper += [bg_abs_limit, slope_limit, slope_limit]

    p0_arr = np.asarray(p0, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    p0_arr = np.minimum(np.maximum(p0_arr, lower_arr), upper_arr)
    if not np.all(np.isfinite(p0_arr)):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_initial_guess",
            mask_diag,
            finite_pixel_count=finite_work.size,
            peak=peak,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_initial_guess")
        return None
    initial_model = model_func(xy_fit, *p0_arr)
    if not np.all(np.isfinite(initial_model)):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_initial_residual",
            mask_diag,
            finite_pixel_count=finite_work.size,
            peak=peak,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_initial_residual")
        return None

    maxfev = int(cfg.get("gaussian_fit_maxfev", 8000))
    fit_meta["maxfev"] = maxfev
    fit_exception = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                model_func,
                xy_fit,
                z_curve,
                p0=p0_arr,
                bounds=(lower_arr, upper_arr),
                maxfev=maxfev,
            )
    except Exception as exc:
        fit_exception = exc
        popt = None
        pcov = None
    if popt is None or not np.all(np.isfinite(popt)):
        if cfg.get("gaussian_fit_fallback_to_moment", True):
            popt = p0_arr.copy()
            pcov = None
            fit_meta["gaussian_fit_method"] = "moment_fallback"
            reason = "fit_failed_moment_fallback"
        else:
            reason = "fit_failed_no_fallback"
            _set_gaussian_failure_diag(
                cfg,
                source_file,
                reason,
                mask_diag,
                finite_pixel_count=finite_work.size,
                peak=peak,
                **fit_meta,
            )
            _fit_failure_warning(
                source_file, reason, str(fit_exception or "non-finite result")
            )
            return None
    else:
        fit_meta["gaussian_fit_method"] = "curve_fit"

    X_local_full = X - x_offset
    Y_local_full = Y - y_offset
    model_curve = model_func((X_local_full, Y_local_full), *popt).reshape(work.shape)
    model_total = (
        model_curve * norm_scale + local_bg
        if cfg.get("gaussian_fit_normalize_data", True)
        else model_curve
    )
    if not np.all(np.isfinite(model_total[fit_mask])):
        _set_gaussian_failure_diag(
            cfg,
            source_file,
            "non_finite_fit_result",
            mask_diag,
            finite_pixel_count=finite_work.size,
            peak=peak,
            **fit_meta,
        )
        _fit_failure_warning(source_file, "non_finite_fit_result")
        return None
    gaussian_curve = gaussian_only_from_popt(
        (X_local_full, Y_local_full), popt, background_model
    ).reshape(work.shape)
    gaussian_only_model = (
        gaussian_curve * norm_scale
        if cfg.get("gaussian_fit_normalize_data", True)
        else gaussian_curve
    )
    x0_fit, y0_fit = float(popt[1]) + x_offset, float(popt[2]) + y_offset
    sigma_x, sigma_y = abs(float(popt[3])), abs(float(popt[4]))
    center_arcsec = pixel_to_data_coord(
        x0_fit, y0_fit, extent, work.shape, origin=image_origin
    )
    residual = work[fit_mask] - model_total[fit_mask]
    residual_rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else np.nan
    if (
        cfg.get("background_use_for_mask", True)
        and rms_map is not None
        and np.asarray(rms_map).shape == work.shape
    ):
        safe_rms = _safe_rms_map(rms_map)
        noise_values = safe_rms[fit_mask & np.isfinite(safe_rms)]
        noise_sigma = float(np.nanmedian(noise_values)) if noise_values.size else np.nan
    else:
        _, noise_sigma = estimate_background_noise(work, fit_mask)
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = mask_diag.get("noise_sigma", np.nan)
    amplitude = float(popt[0])
    amplitude_original = (
        amplitude * norm_scale
        if cfg.get("gaussian_fit_normalize_data", True)
        else amplitude
    )
    snr = (
        float(amplitude_original / noise_sigma)
        if np.isfinite(noise_sigma) and noise_sigma > 0
        else np.nan
    )
    quality_flag = (
        "moment_fallback"
        if fit_meta["gaussian_fit_method"] == "moment_fallback"
        else "ok"
    )
    quality_flag_detail = ""
    if np.isfinite(snr) and snr < cfg.get("fit_snr_threshold", 5.0):
        if quality_flag == "moment_fallback":
            quality_flag_detail = "low_snr"
        else:
            quality_flag = "low_snr"
            quality_flag_detail = "low_snr"
    if sigma_x > cfg.get("max_sigma_fraction", 0.5) * max(nx, ny) or sigma_y > cfg.get(
        "max_sigma_fraction", 0.5
    ) * max(nx, ny):
        quality_flag = "unphysical_size"
        quality_flag_detail = "unphysical_size"

    result = GaussianFitResult(
        model=model_total,
        gaussian_only_model=gaussian_only_model,
        center_pixel=(x0_fit, y0_fit),
        center_arcsec=center_arcsec,
        sigma_pixel=(sigma_x, sigma_y),
        theta_rad=float(popt[5]),
        amplitude=amplitude_original,
        background_level=(
            float(local_bg + popt[6] * norm_scale)
            if len(popt) >= 7 and cfg.get("gaussian_fit_normalize_data", True)
            else (float(popt[6]) if len(popt) >= 7 else None)
        ),
        noise_sigma=float(noise_sigma) if np.isfinite(noise_sigma) else None,
        snr=snr if np.isfinite(snr) else None,
        residual_rms=residual_rms if np.isfinite(residual_rms) else None,
        quality_flag=quality_flag,
        covariance=pcov,
        mask_pixel_count=int(after_limit),
        source_file=source_file,
    )
    if quality_flag_detail:
        result.quality_flag_detail = quality_flag_detail
    result.source_mask = source_mask
    result.image_origin = image_origin
    result.image_extent = extent
    result.initial_center_pixel = fit_meta.get("initial_center_pixel", "")
    try:
        result.initial_center_pixel_tuple = (
            float(cx0 + x_offset),
            float(cy0 + y_offset),
        )
        result.initial_center_arcsec = pixel_to_data_coord(
            result.initial_center_pixel_tuple[0],
            result.initial_center_pixel_tuple[1],
            extent,
            work.shape,
            origin=image_origin,
        )
    except Exception:
        result.initial_center_pixel_tuple = None
        result.initial_center_arcsec = None
    result.coordinate_roundtrip_error_pixel = coordinate_roundtrip_error_pixel(
        x0_fit, y0_fit, extent, work.shape, origin=image_origin
    )
    if fit_meta["gaussian_fit_method"] == "moment_fallback":
        result.reason = "fit_failed_moment_fallback"
        if not getattr(result, "quality_flag_detail", ""):
            result.quality_flag_detail = str(fit_exception or "curve_fit_failed")
    result = _attach_gaussian_fit_metadata(
        result, cfg, mask_diag, fit_input_type, fit_meta
    )
    if coordinate_header is not None:
        from .gaussian_coordinates import attach_wcs_geometry

        peak_y, peak_x = np.unravel_index(np.nanargmax(work), work.shape)
        attach_wcs_geometry(result, coordinate_header, raw_peak_pixel=(peak_x, peak_y))
        _update_gaussian_quality(result, extent, work.shape, cfg)
    return result


def _attach_candidate_metadata(
    fit_result, candidate, source_count_mode, requested_count
):
    fit_result.source_id = candidate.source_id
    fit_result.source_rank = candidate.rank
    fit_result.source_is_primary = candidate.rank == 1
    fit_result.source_count_mode = source_count_mode
    fit_result.requested_source_count = requested_count
    fit_result.source_peak_x_pixel = candidate.peak_pixel[0]
    fit_result.source_peak_y_pixel = candidate.peak_pixel[1]
    fit_result.source_peak_x_arcsec = candidate.peak_arcsec[0]
    fit_result.source_peak_y_arcsec = candidate.peak_arcsec[1]
    fit_result.source_detection_snr = candidate.detection_snr
    fit_result.source_candidate_pixel_count = candidate.mask_pixel_count
    fit_result.raw_center_arcsec = candidate.peak_arcsec
    fit_result.raw_center_pixel = candidate.peak_pixel
    fit_result.raw_peak_x_pixel = candidate.peak_pixel[0]
    fit_result.raw_peak_y_pixel = candidate.peak_pixel[1]
    fit_result.raw_peak_x_arcsec = candidate.peak_arcsec[0]
    fit_result.raw_peak_y_arcsec = candidate.peak_arcsec[1]
    fit_result.center_peak_distance_arcsec = _center_peak_distance_arcsec(fit_result)
    return fit_result


def _candidate_failure_row(candidate, fit_cfg, reason=None):
    fail = dict(fit_cfg.get("_last_gaussian_failure_diag", {}) or {})
    fail.setdefault("reason", reason or fail.get("quality_flag", "fit_failed"))
    fail.setdefault("quality_flag", fail["reason"])
    fail.update(
        {
            "source_id": candidate.source_id,
            "source_rank": candidate.rank,
            "source_is_primary": candidate.rank == 1,
            "source_peak_x_pixel": candidate.peak_pixel[0],
            "source_peak_y_pixel": candidate.peak_pixel[1],
            "source_peak_x_arcsec": candidate.peak_arcsec[0],
            "source_peak_y_arcsec": candidate.peak_arcsec[1],
            "source_detection_snr": candidate.detection_snr,
            "source_candidate_pixel_count": candidate.mask_pixel_count,
        }
    )
    return fail


def fit_multiple_gaussians_on_radio_image(
    data,
    extent,
    cfg,
    source_file=None,
    background_map=None,
    rms_map=None,
    fit_input_type="raw",
    image_origin=None,
    coordinate_header=None,
):
    """Fit native-pixel components and optionally calibrate every result by WCS.

    ``coordinate_header`` follows the single-source fitter's exact center and
    local angular-width contract, including candidate peak annotations. The
    omitted-header path retains the historical affine extent convention.
    """
    requested_count = _requested_multi_source_count(cfg)
    source_count_mode = "requested" if requested_count is not None else "auto"
    image_origin = image_origin or cfg.get("_current_radio_image_origin", "upper")

    if requested_count == 1:
        single_result = fit_elliptical_gaussian_on_radio_image(
            data,
            extent,
            cfg,
            source_file=source_file,
            background_map=background_map,
            rms_map=rms_map,
            fit_input_type=fit_input_type,
            image_origin=image_origin,
            coordinate_header=coordinate_header,
        )
        candidates = detect_gaussian_source_candidates(
            data,
            extent,
            {**cfg, "multi_gaussian_max_sources": 1},
            background_map=background_map,
            rms_map=rms_map,
            image_origin=image_origin,
            coordinate_header=coordinate_header,
        )
        if single_result is not None and candidates:
            _attach_candidate_metadata(
                single_result, candidates[0], source_count_mode, requested_count
            )
        return MultiGaussianFitResult(
            candidates=candidates,
            fit_results=[single_result] if single_result is not None else [],
            primary_result=single_result,
            source_count_mode=source_count_mode,
            requested_source_count=requested_count,
            detected_source_count=1 if single_result is not None else 0,
            missing_source_count=0 if single_result is not None else 1,
            failure_rows=[],
        )

    candidates = detect_gaussian_source_candidates(
        data,
        extent,
        cfg,
        background_map=background_map,
        rms_map=rms_map,
        image_origin=image_origin,
        coordinate_header=coordinate_header,
    )
    detected_count = len(candidates)
    if requested_count is not None:
        candidates = candidates[:requested_count]

    if not candidates:
        cfg["_last_gaussian_failure_diag"] = {
            "source_file": source_file or "",
            "reason": "no_multi_source_candidates",
            "quality_flag": "no_multi_source_candidates",
            "fit_input_type": fit_input_type,
            "gaussian_fit_method": "skipped",
        }
        return MultiGaussianFitResult(
            candidates=[],
            fit_results=[],
            primary_result=None,
            source_count_mode=source_count_mode,
            requested_source_count=requested_count,
            detected_source_count=0,
            missing_source_count=requested_count or 0,
            failure_rows=[],
        )

    fit_results = []
    failure_rows = []
    for candidate in candidates:
        fit_cfg = dict(cfg)
        fit_result = fit_elliptical_gaussian_on_radio_image(
            data,
            extent,
            fit_cfg,
            source_file=source_file,
            background_map=background_map,
            rms_map=rms_map,
            fit_input_type=fit_input_type,
            image_origin=image_origin,
            source_mask_override=candidate.mask,
            coordinate_header=coordinate_header,
        )
        if fit_result is None:
            failure_rows.append(_candidate_failure_row(candidate, fit_cfg))
            continue
        fit_results.append(
            _attach_candidate_metadata(
                fit_result, candidate, source_count_mode, requested_count
            )
        )

    missing_count = 0
    if requested_count is not None and detected_count < requested_count:
        missing_count = requested_count - detected_count
        for rank in range(detected_count + 1, requested_count + 1):
            failure_rows.append(
                {
                    "source_file": source_file or "",
                    "reason": "missing_requested_source",
                    "quality_flag": "missing_requested_source",
                    "source_id": rank,
                    "source_rank": rank,
                    "source_is_primary": False,
                }
            )

    primary_result = fit_results[0] if fit_results else None
    if primary_result is None and failure_rows:
        cfg["_last_gaussian_failure_diag"] = failure_rows[0]

    return MultiGaussianFitResult(
        candidates=candidates,
        fit_results=fit_results,
        primary_result=primary_result,
        source_count_mode=source_count_mode,
        requested_source_count=requested_count,
        detected_source_count=detected_count,
        missing_source_count=missing_count,
        failure_rows=failure_rows,
    )


def overlay_gaussian_fit_on_axis(ax, fit_result, extent, img_shape, cfg):
    if fit_result is None:
        return
    image_origin = getattr(
        fit_result, "image_origin", cfg.get("_current_radio_image_origin", "upper")
    )
    display_mode = cfg.get("gaussian_overlay_display_mode", "contours_and_fwhm")
    valid_modes = {
        "contours_and_fwhm",
        "contours_only",
        "fwhm_only",
        "center_only",
        "none",
    }
    if display_mode not in valid_modes:
        display_mode = "contours_and_fwhm"
    if display_mode == "none":
        return

    is_quality_ok = _update_gaussian_quality(fit_result, extent, img_shape, cfg)
    invalid_reason = (
        getattr(fit_result, "quality_flag_detail", "") or fit_result.quality_flag
    )
    if cfg.get("gaussian_hide_all_when_fit_invalid", True) and not is_quality_ok:
        message = (
            "Gaussian fit invalid: FWHM too large"
            if invalid_reason == "skipped_large_fwhm"
            else f"Gaussian fit invalid: {fit_result.quality_flag}"
        )
        ax.text(
            0.02,
            0.04,
            message,
            transform=ax.transAxes,
            fontsize=max(cfg.get("annotation_fontsize", 20) - 8, 8),
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )
        return

    allow_low_quality = bool(cfg.get("draw_low_quality_gaussian_contours", False))
    allow_shape_overlay = is_quality_ok or allow_low_quality
    draw_contours = (
        display_mode in {"contours_and_fwhm", "contours_only"}
        and cfg.get("draw_gaussian_contours", True)
        and allow_shape_overlay
    )
    draw_fwhm = (
        display_mode in {"contours_and_fwhm", "fwhm_only"}
        and cfg.get("draw_gaussian_fwhm_ellipse", True)
        and allow_shape_overlay
    )
    draw_center = (
        display_mode
        in {
            "contours_and_fwhm",
            "contours_only",
            "fwhm_only",
            "center_only",
        }
        and cfg.get("draw_gaussian_center", True)
        and is_quality_ok
    )
    if getattr(
        fit_result, "quality_flag_detail", ""
    ) == "skipped_large_fwhm" and cfg.get(
        "gaussian_hide_center_when_fwhm_too_large", True
    ):
        draw_center = False

    gaussian_model = fit_result.gaussian_only_model
    peak = (
        float(np.nanmax(gaussian_model))
        if np.any(np.isfinite(gaussian_model))
        else np.nan
    )
    if draw_contours and np.isfinite(peak) and peak > 0:
        levels = [level * peak for level in cfg.get("gaussian_contour_levels", [0.5])]
        levels = sorted({level for level in levels if np.isfinite(level) and level > 0})
        if levels:
            if hasattr(fit_result, "coordinate_header"):
                from solar_toolkit.map.coordinates import hpc_pixel_grid

                contour_args = (
                    *hpc_pixel_grid(fit_result.coordinate_header, img_shape),
                    gaussian_model,
                )
                contour_coordinates = {}
            else:
                contour_args = (gaussian_model,)
                contour_coordinates = {"extent": extent, "origin": image_origin}
            ax.contour(
                *contour_args,
                levels=levels,
                **contour_coordinates,
                colors=cfg.get("gaussian_contour_color", "white"),
                linewidths=cfg.get("gaussian_contour_linewidth", 2.0),
                alpha=cfg.get("gaussian_contour_alpha", 0.9),
            )

    if draw_center:
        cx, cy = fit_result.center_arcsec
        ax.scatter(
            [cx],
            [cy],
            marker=cfg.get("gaussian_center_marker", "x"),
            c=cfg.get("gaussian_center_color", "red"),
            s=cfg.get("gaussian_center_size", 100),
            linewidths=cfg.get("gaussian_center_linewidth", 2.5),
            zorder=8,
        )
        if cfg.get("label_gaussian_center", True):
            ax.annotate(
                "Gaussian center",
                xy=(cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                color=cfg.get("gaussian_center_color", "red"),
                fontsize=max(cfg.get("annotation_fontsize", 20) - 6, 8),
            )

    if draw_fwhm:
        if hasattr(fit_result, "coordinate_header"):
            from .gaussian_coordinates import wcs_fwhm_curve

            ax.plot(
                *wcs_fwhm_curve(fit_result),
                color=cfg.get("gaussian_fwhm_color", "lime"),
                linewidth=cfg.get("gaussian_fwhm_linewidth", 2.0),
                alpha=cfg.get("gaussian_fwhm_alpha", 0.9),
                zorder=7,
            )
            draw_fwhm = False
    if draw_fwhm:
        width = getattr(fit_result, "fwhm_width_arcsec", np.nan)
        height = getattr(fit_result, "fwhm_height_arcsec", np.nan)
        ellipse = patches.Ellipse(
            fit_result.center_arcsec,
            width=width,
            height=height,
            angle=np.degrees(fit_result.theta_rad),
            edgecolor=cfg.get("gaussian_fwhm_color", "lime"),
            facecolor="none",
            linewidth=cfg.get("gaussian_fwhm_linewidth", 2.0),
            alpha=cfg.get("gaussian_fwhm_alpha", 0.9),
            zorder=7,
        )
        ax.add_patch(ellipse)

    raw_center_arcsec = getattr(fit_result, "raw_center_arcsec", None)
    draw_peak_marker = cfg.get("draw_raw_peak_marker", False) or cfg.get(
        "draw_raw_vs_bg_center_shift", False
    )
    if draw_peak_marker and raw_center_arcsec is not None:
        cx, cy = fit_result.center_arcsec
        rx, ry = raw_center_arcsec
        ax.scatter([rx], [ry], marker="+", c="yellow", s=80, linewidths=2.0, zorder=8)
        if cfg.get("draw_fit_peak_distance", False) or cfg.get(
            "draw_raw_vs_bg_center_shift", False
        ):
            ax.plot(
                [rx, cx],
                [ry, cy],
                color="yellow",
                linestyle=":",
                linewidth=1.5,
                zorder=7,
            )

    if cfg.get("draw_coordinate_debug", False):
        initial_arcsec = getattr(fit_result, "initial_center_arcsec", None)
        if initial_arcsec is not None:
            ix, iy = initial_arcsec
            ax.scatter(
                [ix],
                [iy],
                marker="o",
                facecolors="none",
                edgecolors="cyan",
                s=90,
                linewidths=1.8,
                zorder=8,
            )
        source_mask = getattr(fit_result, "source_mask", None)
        if source_mask is not None and np.asarray(source_mask).shape == tuple(
            img_shape
        ):
            if hasattr(fit_result, "coordinate_header"):
                from solar_toolkit.map.coordinates import hpc_pixel_grid

                mask_args = (
                    *hpc_pixel_grid(fit_result.coordinate_header, img_shape),
                    np.asarray(source_mask, dtype=float),
                )
                mask_coordinates = {}
            else:
                mask_args = (np.asarray(source_mask, dtype=float),)
                mask_coordinates = {"extent": extent, "origin": image_origin}
            ax.contour(
                *mask_args,
                levels=[0.5],
                **mask_coordinates,
                colors="cyan",
                linewidths=1.0,
                alpha=0.8,
            )

    if fit_result.quality_flag != "ok" and display_mode != "none":
        ax.text(
            0.02,
            (
                0.10
                if getattr(fit_result, "quality_flag_detail", "")
                == "skipped_large_fwhm"
                else 0.04
            ),
            f"Gaussian fit: {fit_result.quality_flag}",
            transform=ax.transAxes,
            fontsize=max(cfg.get("annotation_fontsize", 20) - 8, 8),
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )


def overlay_multi_gaussian_fit_on_axis(ax, multi_fit_result, extent, img_shape, cfg):
    if multi_fit_result is None:
        return
    local_cfg = dict(cfg)
    local_cfg["label_gaussian_center"] = False
    for fit_result in multi_fit_result.fit_results:
        overlay_gaussian_fit_on_axis(ax, fit_result, extent, img_shape, local_cfg)
        if not cfg.get("draw_multi_gaussian_labels", True):
            continue
        if not getattr(fit_result, "overlay_valid", fit_result.quality_flag == "ok"):
            continue
        cx, cy = fit_result.center_arcsec
        ax.annotate(
            f"G{getattr(fit_result, 'source_rank', '')}",
            xy=(cx, cy),
            xytext=(8, 8),
            textcoords="offset points",
            color=cfg.get("gaussian_center_color", "red"),
            fontsize=max(cfg.get("annotation_fontsize", 20) - 6, 8),
            zorder=9,
        )


def _acquire_csv_lock(lock_path, timeout_seconds=60.0, stale_seconds=300.0):
    deadline = time.time() + float(timeout_seconds)
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
            return fd
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > float(stale_seconds):
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for diagnostics lock: {lock_path}"
                ) from None
            time.sleep(0.05)


def _release_csv_lock(lock_path, fd):
    try:
        if fd is not None:
            os.close(fd)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def save_gaussian_diagnostics_row(row, output_dir, cfg):
    diagnostics_dir = os.path.join(output_dir, _plot_output_subdir(cfg))
    os.makedirs(diagnostics_dir, exist_ok=True)
    csv_path = os.path.join(
        diagnostics_dir,
        cfg.get("gaussian_diagnostics_csv", "radio_gaussian_fit_diagnostics.csv"),
    )
    fieldnames = GAUSSIAN_DIAGNOSTIC_FIELDS
    lock_path = f"{csv_path}.lock"
    lock_fd = _acquire_csv_lock(lock_path)
    try:
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    existing_header = next(csv.reader(handle), [])
            except OSError:
                existing_header = []
            if existing_header and existing_header != fieldnames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                root, ext = os.path.splitext(csv_path)
                csv_path = f"{root}_{timestamp}{ext}"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    finally:
        _release_csv_lock(lock_path, lock_fd)


def _gaussian_result_diagnostics_row(
    fit_result, cfg, freq=None, time_str=None, polarization=None, bg_diag=None
):
    bg_diag = bg_diag or {}
    fit_input_type = getattr(
        fit_result,
        "fit_input_type",
        "background_subtracted" if cfg.get("background_use_for_fit", False) else "raw",
    )
    return {
        "source_file": fit_result.source_file,
        "time": time_str,
        "freq": freq,
        "polarization": polarization,
        "reason": getattr(fit_result, "reason", ""),
        "finite_pixel_count": "",
        "center_x_arcsec": fit_result.center_arcsec[0],
        "center_y_arcsec": fit_result.center_arcsec[1],
        "center_x_pixel": fit_result.center_pixel[0],
        "center_y_pixel": fit_result.center_pixel[1],
        "raw_peak_x_arcsec": getattr(fit_result, "raw_peak_x_arcsec", ""),
        "raw_peak_y_arcsec": getattr(fit_result, "raw_peak_y_arcsec", ""),
        "raw_peak_x_pixel": getattr(fit_result, "raw_peak_x_pixel", ""),
        "raw_peak_y_pixel": getattr(fit_result, "raw_peak_y_pixel", ""),
        "center_peak_dx_arcsec": getattr(fit_result, "center_peak_dx_arcsec", ""),
        "center_peak_dy_arcsec": getattr(fit_result, "center_peak_dy_arcsec", ""),
        "center_peak_distance_arcsec": getattr(
            fit_result, "center_peak_distance_arcsec", ""
        ),
        "sigma_x_pixel": fit_result.sigma_pixel[0],
        "sigma_y_pixel": fit_result.sigma_pixel[1],
        "fwhm_x_pixel": 2.355 * fit_result.sigma_pixel[0],
        "fwhm_y_pixel": 2.355 * fit_result.sigma_pixel[1],
        "fwhm_width_arcsec": getattr(fit_result, "fwhm_width_arcsec", ""),
        "fwhm_height_arcsec": getattr(fit_result, "fwhm_height_arcsec", ""),
        "fwhm_major_arcsec": getattr(fit_result, "fwhm_major_arcsec", ""),
        "fwhm_minor_arcsec": getattr(fit_result, "fwhm_minor_arcsec", ""),
        "max_fwhm_arcsec": getattr(
            fit_result, "max_fwhm_arcsec", cfg.get("max_fwhm_arcsec", "")
        ),
        "fwhm_valid": getattr(fit_result, "fwhm_valid", ""),
        "overlay_valid": getattr(fit_result, "overlay_valid", ""),
        "trajectory_valid": getattr(fit_result, "trajectory_valid", ""),
        "coordinate_roundtrip_error_pixel": getattr(
            fit_result, "coordinate_roundtrip_error_pixel", ""
        ),
        "theta_rad": fit_result.theta_rad,
        "amplitude": fit_result.amplitude,
        "background_level": fit_result.background_level,
        "noise_sigma": fit_result.noise_sigma,
        "snr": fit_result.snr,
        "residual_rms": fit_result.residual_rms,
        "mask_pixel_count": fit_result.mask_pixel_count,
        "quality_flag": fit_result.quality_flag,
        "quality_flag_detail": getattr(fit_result, "quality_flag_detail", ""),
        "background_strategy": getattr(
            fit_result, "background_strategy", cfg.get("radio_background_strategy", "")
        ),
        "background_use_for_mask": getattr(
            fit_result,
            "background_use_for_mask",
            cfg.get("background_use_for_mask", ""),
        ),
        "background_use_for_display": cfg.get("background_use_for_display", False),
        "background_use_for_fit": getattr(
            fit_result, "background_use_for_fit", cfg.get("background_use_for_fit", "")
        ),
        "display_input_type": cfg.get("display_input_type", "raw"),
        "background_mesh_size": cfg.get("background_mesh_size", ""),
        "background_rms_median": getattr(fit_result, "background_rms_median", ""),
        "background_level_median": getattr(fit_result, "background_level_median", ""),
        "source_snr_peak": getattr(fit_result, "source_snr_peak", ""),
        "source_snr_mean": getattr(fit_result, "source_snr_mean", ""),
        "mask_method": getattr(fit_result, "mask_method", ""),
        "fit_peak_fraction_threshold_used": getattr(
            fit_result, "fit_peak_fraction_threshold_used", ""
        ),
        "fit_peak_fraction_candidate_counts": getattr(
            fit_result, "fit_peak_fraction_candidate_counts", ""
        ),
        "background_enabled": bg_diag.get("background_enabled", False),
        "background_mode_requested": bg_diag.get("background_mode_requested", ""),
        "background_mode_used": bg_diag.get("background_mode_used", ""),
        "background_scale": bg_diag.get("background_scale", ""),
        "use_background_subtracted_for_gaussian_fit": cfg.get(
            "background_use_for_fit", False
        ),
        "fit_used_background_subtracted": cfg.get("background_use_for_fit", False),
        "fit_input_type": fit_input_type,
        "fit_background_model": cfg.get("fit_background_model", "constant"),
        "gaussian_fit_method": getattr(fit_result, "gaussian_fit_method", ""),
        "roi_used": getattr(fit_result, "roi_used", ""),
        "roi_shape": getattr(fit_result, "roi_shape", ""),
        "fit_pixel_count_before_limit": getattr(
            fit_result, "fit_pixel_count_before_limit", ""
        ),
        "fit_pixel_count_after_limit": getattr(
            fit_result, "fit_pixel_count_after_limit", ""
        ),
        "maxfev": getattr(fit_result, "maxfev", cfg.get("gaussian_fit_maxfev", "")),
        "initial_center_pixel": getattr(fit_result, "initial_center_pixel", ""),
        "initial_sigma_x_pixel": getattr(fit_result, "initial_sigma_x_pixel", ""),
        "initial_sigma_y_pixel": getattr(fit_result, "initial_sigma_y_pixel", ""),
        "normalization_scale": getattr(fit_result, "normalization_scale", ""),
        "peak": getattr(fit_result, "peak", ""),
        "threshold": getattr(fit_result, "threshold", ""),
    }


def multi_gaussian_diagnostics_rows(
    multi_fit_result,
    cfg,
    freq=None,
    time_str=None,
    polarization=None,
    bg_diag=None,
):
    if multi_fit_result is None:
        return []
    rows = []
    for fit_result in multi_fit_result.fit_results:
        row = _gaussian_result_diagnostics_row(
            fit_result, cfg, freq, time_str, polarization, bg_diag
        )
        row.update(
            {
                "source_id": getattr(fit_result, "source_id", ""),
                "source_rank": getattr(fit_result, "source_rank", ""),
                "source_is_primary": getattr(fit_result, "source_is_primary", ""),
                "source_count_mode": multi_fit_result.source_count_mode,
                "requested_source_count": (
                    multi_fit_result.requested_source_count
                    if multi_fit_result.requested_source_count is not None
                    else ""
                ),
                "detected_source_count": multi_fit_result.detected_source_count,
                "source_peak_x_pixel": getattr(fit_result, "source_peak_x_pixel", ""),
                "source_peak_y_pixel": getattr(fit_result, "source_peak_y_pixel", ""),
                "source_peak_x_arcsec": getattr(fit_result, "source_peak_x_arcsec", ""),
                "source_peak_y_arcsec": getattr(fit_result, "source_peak_y_arcsec", ""),
                "source_detection_snr": getattr(fit_result, "source_detection_snr", ""),
                "source_candidate_pixel_count": getattr(
                    fit_result, "source_candidate_pixel_count", ""
                ),
            }
        )
        rows.append(row)

    for fail in multi_fit_result.failure_rows:
        row = {field: "" for field in MULTI_GAUSSIAN_DIAGNOSTIC_FIELDS}
        row.update(
            {
                "source_file": fail.get("source_file", ""),
                "time": time_str,
                "freq": freq,
                "polarization": polarization,
                "reason": fail.get("reason", "fit_failed"),
                "quality_flag": fail.get("quality_flag", fail.get("reason", "")),
                "quality_flag_detail": fail.get("quality_flag_detail", ""),
                "finite_pixel_count": fail.get("finite_pixel_count", ""),
                "mask_pixel_count": fail.get("mask_pixel_count", ""),
                "background_level": fail.get("background_level", ""),
                "noise_sigma": fail.get("noise_sigma", ""),
                "threshold": fail.get("threshold", ""),
                "fit_input_type": fail.get("fit_input_type", ""),
                "gaussian_fit_method": fail.get("gaussian_fit_method", "skipped"),
                "source_id": fail.get("source_id", ""),
                "source_rank": fail.get("source_rank", ""),
                "source_is_primary": fail.get("source_is_primary", ""),
                "source_count_mode": multi_fit_result.source_count_mode,
                "requested_source_count": (
                    multi_fit_result.requested_source_count
                    if multi_fit_result.requested_source_count is not None
                    else ""
                ),
                "detected_source_count": multi_fit_result.detected_source_count,
                "source_peak_x_pixel": fail.get("source_peak_x_pixel", ""),
                "source_peak_y_pixel": fail.get("source_peak_y_pixel", ""),
                "source_peak_x_arcsec": fail.get("source_peak_x_arcsec", ""),
                "source_peak_y_arcsec": fail.get("source_peak_y_arcsec", ""),
                "source_detection_snr": fail.get("source_detection_snr", ""),
                "source_candidate_pixel_count": fail.get(
                    "source_candidate_pixel_count", ""
                ),
            }
        )
        rows.append(row)
    return rows


def save_multi_gaussian_diagnostics_row(row, output_dir, cfg):
    diagnostics_dir = os.path.join(output_dir, _plot_output_subdir(cfg))
    os.makedirs(diagnostics_dir, exist_ok=True)
    csv_path = os.path.join(
        diagnostics_dir,
        cfg.get(
            "multi_gaussian_diagnostics_csv",
            "radio_multi_gaussian_fit_diagnostics.csv",
        ),
    )
    fieldnames = MULTI_GAUSSIAN_DIAGNOSTIC_FIELDS
    lock_path = f"{csv_path}.lock"
    lock_fd = _acquire_csv_lock(lock_path)
    try:
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    existing_header = next(csv.reader(handle), [])
            except OSError:
                existing_header = []
            if existing_header and existing_header != fieldnames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                root, ext = os.path.splitext(csv_path)
                csv_path = f"{root}_{timestamp}{ext}"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    finally:
        _release_csv_lock(lock_path, lock_fd)
