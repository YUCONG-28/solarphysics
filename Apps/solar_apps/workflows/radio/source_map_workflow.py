# 模块用途: 处理太阳射电 FITS 图像/频谱并绘制射电源图。
# 主要输入: 射电图像、频谱数据和拟合/等值线参数。
# 主要输出/运行说明: 输出单频或多频射电源图，可包含高斯拟合轮廓。
"""
Created on Sun Nov 23 00:19:30 2025

radio_source_map_plot.py: Process solar radio spectrogram (FITS files) and generate single-band or multi-band composite images.
Supports two operation modes: single-band mode (process FITS files one by one) and multi-band mode (synthesize multiple bands at the same time).
Supports parallel processing, automatic memory safety detection, multiple color range modes, and generates images with solar limb, coordinate grid, and directional markers.

Gaussian overlay experimental version:
This script keeps the original radio source map plotting workflow and adds background-corrected elliptical Gaussian fitting directly on radio images. The fitted Gaussian contour, fitted center, FWHM ellipse, and diagnostics can be overlaid on the original radio source map.

新增功能: 左右旋数据加和（可配置加权平均）
"""

__all__ = [
    "DEFAULT_CONFIG",
    "build_config",
    "main",
    "resolve_spatial_display",
    "run_source_map",
]

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from importlib import import_module
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from tqdm import tqdm

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import build_scientific_image_filename
from solar_toolkit.radio.gaussian import (
    fit_multiple_gaussians_on_radio_image as _fit_multiple_gaussians_on_radio_image,
)
from solar_toolkit.radio.gaussian import (
    multi_gaussian_diagnostics_rows as _multi_gaussian_diagnostics_rows,
)
from solar_toolkit.radio.gaussian import (
    overlay_multi_gaussian_fit_on_axis as _overlay_multi_gaussian_fit_on_axis,
)
from solar_toolkit.radio.gaussian import (
    save_multi_gaussian_diagnostics_row as _save_multi_gaussian_diagnostics_row,
)
from .artifacts import (
    apply_colorbar_tick_notation,
    colorbar_label,
    resolve_colorbar_unit,
    save_figure_artifact,
    tick_notation_for_transform,
)
from .spatial_display import SpatialRadioDisplay, spatial_display_for_source_map

# Canonical science helpers are bound before the historical orchestration is
# defined so annotations and monkeypatch anchors keep resolving in this module.
_canonical_gaussian_model = import_module("solar_toolkit.modeling.gaussian")
_canonical_radio_coordinates = import_module("solar_toolkit.radio.coordinates")
_canonical_radio_drift = import_module(
    "solar_apps.workflows.radio.drift_selection_application"
)
_canonical_radio_gaussian = import_module("solar_toolkit.radio.gaussian")
_canonical_radio_gaussian_background = import_module(
    "solar_toolkit.radio.gaussian_background"
)
_canonical_radio_gaussian_masks = import_module("solar_toolkit.radio.gaussian_masks")
_canonical_radio_gaussian_models = import_module("solar_toolkit.radio.gaussian_models")
_canonical_radio_io = import_module("solar_toolkit.radio.io")
_canonical_radio_output_paths = import_module("solar_toolkit.radio.output_paths")
_canonical_radio_spectrogram = import_module("solar_toolkit.radio.spectrogram")

GaussianFitResult = _canonical_radio_gaussian.GaussianFitResult
SpectrogramCache = _canonical_radio_spectrogram.SpectrogramCache
DriftRateResult = _canonical_radio_drift.DriftRateResult
GAUSSIAN_DIAGNOSTIC_FIELDS = _canonical_radio_io.GAUSSIAN_DIAGNOSTIC_FIELDS
DRIFT_RATE_DIAGNOSTIC_FIELDS = _canonical_radio_io.DRIFT_RATE_DIAGNOSTIC_FIELDS

elliptical_gaussian_2d = _canonical_gaussian_model.elliptical_gaussian_2d
elliptical_gaussian_2d_with_constant_bg = (
    _canonical_radio_gaussian_models.elliptical_gaussian_2d_with_constant_bg
)
elliptical_gaussian_2d_with_plane_bg = (
    _canonical_radio_gaussian_models.elliptical_gaussian_2d_with_plane_bg
)
gaussian_only_from_popt = _canonical_radio_gaussian_models.gaussian_only_from_popt
estimate_background_noise = (
    _canonical_radio_gaussian_background.estimate_background_noise
)
_safe_rms_map = _canonical_radio_gaussian_background._safe_rms_map
_unravel_2d_index = _canonical_radio_coordinates.unravel_2d_index
_true_indices = _canonical_gaussian_model.true_indices
_select_peak_connected_mask = (
    _canonical_radio_gaussian_masks._select_peak_connected_mask
)
create_source_mask = _canonical_radio_gaussian_masks.create_source_mask
_gaussian_fit_diag_defaults = _canonical_radio_gaussian._gaussian_fit_diag_defaults
_roi_slices_from_mask = _canonical_radio_gaussian._roi_slices_from_mask
_weighted_moment_initial_guess = (
    _canonical_radio_gaussian._weighted_moment_initial_guess
)
_limit_fit_pixels = _canonical_radio_gaussian._limit_fit_pixels
_attach_gaussian_fit_metadata = _canonical_radio_gaussian._attach_gaussian_fit_metadata
_gaussian_fwhm_arcsec = _canonical_radio_gaussian._gaussian_fwhm_arcsec
_center_peak_distance_arcsec = _canonical_radio_gaussian._center_peak_distance_arcsec
_gaussian_quality_config = _canonical_radio_gaussian._gaussian_quality_config
_update_gaussian_quality = _canonical_radio_gaussian._update_gaussian_quality
_set_gaussian_failure_diag = _canonical_radio_gaussian._set_gaussian_failure_diag
fit_elliptical_gaussian_on_radio_image = (
    _canonical_radio_gaussian.fit_elliptical_gaussian_on_radio_image
)
overlay_gaussian_fit_on_axis = _canonical_radio_gaussian.overlay_gaussian_fit_on_axis
_acquire_csv_lock = _canonical_radio_gaussian._acquire_csv_lock
_release_csv_lock = _canonical_radio_gaussian._release_csv_lock
save_gaussian_diagnostics_row = _canonical_radio_gaussian.save_gaussian_diagnostics_row

pixel_to_data_coord = _canonical_radio_coordinates.pixel_to_data_coord
data_coord_to_pixel = _canonical_radio_coordinates.data_coord_to_pixel
coordinate_roundtrip_error_pixel = (
    _canonical_radio_coordinates.coordinate_roundtrip_error_pixel
)
_parse_datetime_value = _canonical_radio_io.parse_datetime_value
_index_range_from_values = _canonical_radio_io.index_range_from_values
_index_range_from_time_values = _canonical_radio_io.index_range_from_time_values
_spectrogram_panel_enabled = _canonical_radio_output_paths.spectrogram_panel_enabled
_normalize_spectrogram_paths = _canonical_radio_spectrogram._normalize_spectrogram_paths
_read_spectrogram_file_metadata = (
    _canonical_radio_spectrogram._read_spectrogram_file_metadata
)
resolve_spectrogram_time_window_multi = (
    _canonical_radio_spectrogram.resolve_spectrogram_time_window_multi
)
_spectrogram_overlap_segments = (
    _canonical_radio_spectrogram._spectrogram_overlap_segments
)
_rebinned_axis_values = _canonical_radio_spectrogram._rebinned_axis_values
_read_rebinned_plane = _canonical_radio_spectrogram._read_rebinned_plane
build_spectrogram_cache = _canonical_radio_spectrogram.build_spectrogram_cache
_spectrogram_time_locator = _canonical_radio_spectrogram._spectrogram_time_locator
_date_num_to_datetime = _canonical_radio_spectrogram._date_num_to_datetime
_spectrogram_display_data_extent = (
    _canonical_radio_spectrogram._spectrogram_display_data_extent
)

_datetime_iso_ms = _canonical_radio_drift._datetime_iso_ms
_drift_line_time = _canonical_radio_drift._drift_line_time
calculate_drift_rate_from_line = _canonical_radio_drift.calculate_drift_rate_from_line
_mark_drift_range_warnings = _canonical_radio_drift._mark_drift_range_warnings
_spectrogram_coord_from_pixel = _canonical_radio_drift._spectrogram_coord_from_pixel
assert_spectrogram_mapping_not_flipped = (
    _canonical_radio_drift.assert_spectrogram_mapping_not_flipped
)
save_drift_selection_json = _canonical_radio_drift.save_drift_selection_json
load_drift_selection_json = _canonical_radio_drift.load_drift_selection_json
_load_drift_selection_payload = _canonical_radio_drift._load_drift_selection_payload
render_spectrogram_selection_preview = (
    _canonical_radio_drift.render_spectrogram_selection_preview
)
_drift_selection_html = _canonical_radio_drift._drift_selection_html
launch_drift_selection_server = _canonical_radio_drift.launch_drift_selection_server
overlay_drift_rate_results = _canonical_radio_drift.overlay_drift_rate_results
save_drift_rate_diagnostics_once = (
    _canonical_radio_drift.save_drift_rate_diagnostics_once
)

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

# ============================================================
#   ★ All configurable parameters are centralized here, no need to dive into code to adjust ★
# ============================================================
USER_CONFIG = {}

DEFAULT_CONFIG = {
    # ---------- 坐标轴和颜色条数字颜色 ----------
    "tick_color": "black",  # 坐标轴刻度数字颜色
    "colorbar_tick_color": "white",  # 颜色条刻度数字颜色
    # ---------- 坐标轴刻度配置 ----------
    "x_tick_step": 0,  # x轴刻度显示步长（单位：角秒），0表示自动计算
    "y_tick_step": 0,  # y轴刻度显示步长（单位：角秒），0表示自动计算
    "tick_label_rotation": 0,  # 刻度标签旋转角度（度），0表示不旋转
    "hide_inner_ticks": True,  # 是否隐藏内部子图的刻度标签（只显示边缘子图）
    # ---------- 时间解析配置 ----------
    # 支持的日期格式:
    #   "6digit": YYDDD (6位，如202553表示2025年第53天)
    #   "7digit": YYYYDDD (7位，如2025124表示2025年第124天)
    #   "8digit": YYYYDDDD (8位，不常见)
    #   "auto": 自动检测（默认）
    "date_format": "auto",  # "auto", "yyyyddd", "yyyymdd", "yymmdd_or_yyddd"
    # 文件名时间解析模式（正则表达式）
    "filename_patterns": {
        "with_ms": r"_(\d{6,8})_(\d{6})_(\d{1,3})",  # 带毫秒
        "without_ms": r"_(\d{6,8})_(\d{6})",  # 不带毫秒
    },
    # 时间解析容错模式
    "time_parsing_fallback": True,  # 如果精确解析失败，是否尝试宽松解析
    # ---------- Operation mode ----------
    # "single_band": single-band mode (similar to original)
    # "multi_band": multi-band synthesis mode (similar to sdo_aia_euv_processor.py)
    "mode": "multi_band",  # "single_band" or "multi_band"
    # ---------- Polarization configuration ----------
    # "RR": right circular polarization
    # "LL": left circular polarization
    # "RR+LL": sum of right and left circular polarization
    "polarization": "RR+LL",  # "RR", "LL", or "RR+LL"
    # ---------- 左右旋数据加和配置 ----------
    "combine_polarizations": True,  # 是否启用左右旋数据加和功能
    "rr_dir_suffix": "RR",  # 右旋数据目录后缀
    "ll_dir_suffix": "LL",  # 左旋数据目录后缀
    "weighted_average": False,  # 是否使用加权平均（True）或简单相加（False）
    "rr_weight": 0.5,  # 右旋权重（加权平均时使用）
    "ll_weight": 0.5,  # 左旋权重（加权平均时使用）
    "save_individual_pols": False,  # 是否同时保存单独的RR、LL图像
    "time_tolerance_seconds": 0.001,  # 时间对齐容差（秒）
    "enable_raw_quality_filter": False,
    "raw_quality_bad_frame_output_subdir": "raw_quality_bad_frames",
    # ---------- Single-band mode configuration ----------
    # Single-file mode: if you only want to plot a single file, fill in the full absolute path here
    "single_file_path": "data/radio/149MHz/RR/example.fits",
    # Batch single-band mode: directory containing FITS files
    "data_dir": "data/radio/149MHz/RR",
    # File range (only effective in batch mode)
    "start_idx": 1588,  # start index (inclusive)
    "end_idx": 1666,  # end index (exclusive)
    "study_mode": None,  # must be explicit: exploratory or confirmatory
    # ---------- Multi-band mode configuration ----------
    "multi_band_root": "data/radio",
    "multi_band_freqs": [149, 164, 190, 205, 223, 238],
    "band_dir_pattern": "{freq}MHz/{polar}",
    "multi_band_output_subdir": "multi_band_{polar}",
    "multi_band_layout": "auto",
    "multi_band_time_tolerance_seconds": 0.1,
    # ---------- 多波段颜色范围 ----------
    # True: 每个波段使用独立的颜色范围
    # False: 所有波段使用统一的全局颜色范围
    "use_per_band_colormap": True,
    # ---------- 波段颜色范围计算方法 ----------
    # "percentile": 使用固定百分位数（5%-95%，范围太小时自动调整到1%-99%）
    # "fixed_percentile": 使用用户自定义的百分位数
    # "minmax": 使用最小最大值方法
    "per_band_range_method": "fixed_percentile",
    # 当使用fixed_percentile方法时的百分位数设置
    # 例如: [5, 95] 表示使用5%到95%的数据范围
    "per_band_percentiles": [99.7, 99.99],
    # 当数据范围太小时的最小对数范围阈值
    "min_log_range": 0.001,
    # ---------- 多波段子图间距配置 ----------
    "multi_band_wspace": 0.0,  # 子图之间的水平间距（0表示无间隙）
    "multi_band_hspace": 0.0,  # 子图之间的垂直间距（0表示无间隙）
    "multi_band_layout_engine": "manual_zero_gap",
    "multi_band_zero_gap": True,
    "multi_band_aspect_mode": "equal_compact",
    "multi_band_auto_fig_height": True,
    "radio_grid_left": 0.06,
    "radio_grid_right": 0.98,
    "radio_grid_top": 0.92,
    "radio_grid_bottom": 0.30,
    "radio_hide_inner_ticklabels": True,
    "radio_hide_overlapping_edge_ticklabels": True,
    "radio_use_global_axis_labels": True,
    "radio_show_internal_spines": True,
    "radio_tick_prune_tolerance": 1e-6,
    "radio_global_xlabel": "x (arcsec)",
    "radio_global_ylabel": "y (arcsec)",
    "radio_global_xlabel_mode": "auto",
    "radio_spectrogram_label_gap_fraction": 0.55,
    "radio_global_xlabel_min_y_gap": 0.018,
    "radio_global_xlabel_offset": 0.015,
    "radio_global_ylabel_offset": 0.035,
    "radio_tick_step_auto_target": 5,
    "radio_panel_left": 0.055,
    "radio_panel_right": 0.985,
    "radio_panel_top": 0.925,
    "radio_panel_bottom": 0.34,
    "radio_panel_anchor": "center",
    "radio_panel_allow_shrink_width": True,
    "radio_panel_allow_shrink_height": True,
    "radio_remove_axis_margins": True,
    "radio_force_exact_xlim_ylim": True,
    # ---------- 颜色条位置配置 ----------
    "colorbar_position": [
        0.75,
        0.05,
        0.22,
        0.03,
    ],  # [x, y, width, height] 相对于子图内部
    # ---------- Output configuration ----------
    "output_dir": "outputs/radio/source_maps",
    "multi_band_also_save_single": False,
    # ---------- Color range ----------
    # "auto": adjust per frame automatically
    # "global": fixed to global min/max values
    # "fixed": fixed to fixed_vmin / fixed_vmax
    "color_range_mode": "fixed",
    "fixed_vmin": 0,
    "fixed_vmax": 4 * 1e9,
    # ---------- Image display limits ----------
    "use_custom_lim": True,
    "custom_xlim": (400, 2000),
    "custom_ylim": (-1000, 600),
    # ---------- Image appearance ----------
    "fig_size": (18, 16),
    "multi_band_fig_size": (24, 16),
    "dpi": 300,
    "radio_cmap": "hot",
    "cmap": "hot",
    "background_bad_color": "#000080",
    "radio_colorbar_unit": None,
    "scale_factor": 3.5,
    # ---------- Annotation styles ----------
    "title_fontsize": 24,
    "label_fontsize": 28,
    "tick_fontsize": 22,
    "legend_fontsize": 18,
    "annotation_fontsize": 20,
    # ---------- Parallel processing ----------
    # max_workers: number of parallel worker processes
    #   None  → program automatically calculates safe upper limit based on 【available memory / estimated per-frame memory】
    #   integer  → force specified (e.g., setting to 4 will use at most 4 cores)
    #   Note: setting too high may cause out-of-memory crashes, it is recommended to start with None for auto estimation
    "max_workers": None,
    # memory_per_worker_mb: estimated memory per worker (MB)
    #   used for automatic safe max_workers calculation, can be adjusted according to actual FITS file size
    #   None → automatically estimated from file size (×20 safety margin)
    "memory_per_worker_mb": None,
    # ---------- Radio background subtraction ----------
    "radio_background_strategy": "noise_map_only",
    "background_use_for_mask": True,
    "background_use_for_display": False,
    "background_use_for_fit": False,
    "display_input_type": "raw",
    "fit_input_type": "raw_with_local_baseline",
    "background_mesh_size": 96,
    "background_mesh_step": 48,
    "background_sigma_clip": 3.0,
    "background_sigma_clip_iters": 3,
    "background_min_valid_pixels": 20,
    "background_interpolate_order": 1,
    "background_fit_validation_enabled": True,
    "background_max_center_shift_arcsec": 500.0,
    "background_clip_negative_for_display_only": True,
    "radio_background_force_off": False,
    "radio_background_workflow": "off",
    "enable_radio_background_subtraction": False,
    "radio_background_subtraction_mode": "local_median",
    "display_background_subtracted_image": False,
    "use_background_subtracted_for_gaussian_fit": False,
    "save_background_subtracted_image": False,
    "save_estimated_background_map": False,
    "background_pre_event_seconds": 60.0,
    "background_min_frames": 5,
    "background_local_median_size": 31,
    "background_scale_mode": "offsource_median",
    "clip_negative_after_background": True,
    "save_background_diagnostics": False,
    "background_diagnostics_csv": "radio_background_subtraction_diagnostics.csv",
    "background_subtract_before_polarization_combine": False,
    # ---------- Gaussian fitting overlay ----------
    "enable_gaussian_overlay": True,
    "draw_gaussian_contours": True,
    "draw_gaussian_center": True,
    "draw_gaussian_fwhm_ellipse": True,
    "gaussian_overlay_display_mode": "fwhm_only",
    "draw_raw_vs_bg_center_shift": False,
    "draw_fit_residual_panel": True,
    "gaussian_contour_levels": [0.5],
    "draw_low_quality_gaussian_contours": False,
    "max_fwhm_arcsec": 1800.0,
    "max_center_peak_distance_arcsec": 300.0,
    "gaussian_hide_center_when_fwhm_too_large": True,
    "gaussian_hide_label_when_fwhm_too_large": True,
    "gaussian_hide_all_when_fit_invalid": True,
    "gaussian_quality_requirements": {
        "require_quality_ok": True,
        "max_fwhm_arcsec": 1800.0,
        "max_center_peak_distance_arcsec": 300.0,
        "min_snr": 5.0,
        "max_residual_rms_fraction": 0.8,
    },
    "gaussian_contour_color": "white",
    "gaussian_contour_linewidth": 2.0,
    "gaussian_contour_alpha": 0.9,
    "gaussian_center_marker": "x",
    "gaussian_center_color": "red",
    "gaussian_center_size": 100,
    "gaussian_center_linewidth": 2.5,
    "label_gaussian_center": True,
    "gaussian_fwhm_color": "lime",
    "gaussian_fwhm_linewidth": 2.0,
    "gaussian_fwhm_alpha": 0.9,
    "radio_background_mode": "local_median",
    "fit_use_source_mask": True,
    "fit_snr_threshold": 5.0,
    "fit_grow_snr_threshold": 3.0,
    "fit_peak_fraction_threshold": 0.40,
    "fit_grow_peak_fraction_threshold": 0.22,
    "fit_mask_target_min_pixels": 18,
    "fit_mask_target_max_pixels": 260,
    "fit_peak_fraction_threshold_min": 0.25,
    "fit_peak_fraction_threshold_max": 0.62,
    "fit_peak_fraction_threshold_step": 0.03,
    "fit_min_mask_pixels": 12,
    "fit_mask_dilation_pixels": 1,
    "fit_background_model": "constant",
    "gaussian_fit_maxfev": 8000,
    "gaussian_fit_use_roi": True,
    "gaussian_fit_roi_padding_pixels": 4,
    "gaussian_fit_max_pixels": 400,
    "gaussian_fit_normalize_data": True,
    "gaussian_fit_fallback_to_moment": True,
    "gaussian_fit_verbose": False,
    "max_sigma_fraction": 0.18,
    "gaussian_per_band_params": {},
    "gaussian_source_mode": "single",
    "multi_gaussian_source_count": None,
    "multi_gaussian_max_sources": 3,
    "multi_gaussian_min_peak_fraction": 0.30,
    "multi_gaussian_min_peak_distance_pixels": 6,
    "multi_gaussian_use_watershed": True,
    "multi_gaussian_diagnostics_csv": "radio_multi_gaussian_fit_diagnostics.csv",
    "draw_multi_gaussian_labels": True,
    "skip_low_quality_fit": True,
    "save_gaussian_diagnostics": True,
    "gaussian_diagnostics_csv": "radio_gaussian_fit_diagnostics.csv",
    # ---------- Spectrogram panel / video-style composite ----------
    # 上半部分为射电源图像，下半部分为动态频谱；频谱数据只缓存一次，逐帧只更新当前时间虚线。
    "enable_spectrogram_panel": True,
    "spectrogram_file_paths": [],
    "spectrogram_file_path": "data/radio/spectrogram.fits",
    "spectrogram_time_display_mode": "user",
    "spectrogram_time_start": "2025-01-24T04:48:30",  # 例如 "2025-05-03T07:16:00"；None 表示自动从射电图像时间范围推断
    "spectrogram_time_end": "2025-01-24T04:49:00",
    "spectrogram_time_margin_seconds": 30.0,
    "spectrogram_f_start": 80.0,
    "spectrogram_f_end": 340.0,
    "spectrogram_polarization": "sum",  # "LL", "RR", "sum", "ratio"
    "spectrogram_rebin_t_target": 1000,
    "spectrogram_rebin_f_target": 700,
    "spectrogram_chunk_mem_mb": 64,
    "spectrogram_cmap": "jet",
    "spectrogram_vmin": 2.5,
    "spectrogram_vmax": 4.5,
    "spectrogram_use_log10": True,
    "spectrogram_title": "CSO dynamic spectrum",
    "spectrogram_panel_height_ratio": 0.34,
    "spectrogram_hspace": 0.14,
    "spectrogram_line_color": "white",
    "spectrogram_line_style": "--",
    "spectrogram_line_width": 1.6,
    "spectrogram_line_alpha": 0.95,
    "spectrogram_draw_colorbar": True,
    "spectrogram_colorbar_label": r"log$_{10}$ intensity",
    "spectrogram_colorbar_unit": None,
    "spectrogram_xtick_format": "%H:%M:%S",
    "spectrogram_major_tick_seconds": 2,
    "spectrogram_auto_time_locator": True,
    "spectrogram_max_time_ticks": 34,
    "spectrogram_disable_on_time_mismatch": True,
    "spectrogram_clip_current_time_line": True,
    "spectrogram_show_out_of_range_time_note": True,
    "spectrogram_output_subdir": "radio_spectrogram_composite",
    # ---------- Output ----------
    "show_plot": False,
    "save_plot": True,
    "write_source_map_sidecar": False,
}
ADVANCED_CONFIG = DEFAULT_CONFIG


def _deep_update_dict(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update_dict(result[key], value)
        else:
            result[key] = value
    return result


def build_config(user_config, default_config):
    """Map nested USER_CONFIG back to the flat cfg keys used by legacy code."""
    cfg = dict(default_config)
    user_config = user_config or {}
    if user_config.get("mode") is not None:
        cfg["mode"] = user_config["mode"]
    if user_config.get("study_mode") is not None:
        cfg["study_mode"] = user_config["study_mode"]

    for key, value in user_config.get("data", {}).items():
        cfg[key] = value

    features = user_config.get("features", {})
    feature_map = {
        "gaussian_overlay": "enable_gaussian_overlay",
        "spectrogram_panel": "enable_spectrogram_panel",
        "save_gaussian_diagnostics": "save_gaussian_diagnostics",
        "save_individual_pols": "save_individual_pols",
        "raw_quality_filter": "enable_raw_quality_filter",
    }
    for user_key, flat_key in feature_map.items():
        if user_key in features:
            cfg[flat_key] = features[user_key]

    if "save_background_products" in features:
        save_bg = bool(features["save_background_products"])
        cfg["save_background_subtracted_image"] = save_bg
        cfg["save_estimated_background_map"] = save_bg

    raw_quality = user_config.get("raw_quality", {})
    if "enabled" in raw_quality:
        cfg["enable_raw_quality_filter"] = bool(raw_quality["enabled"])
    if "filter_bad_fits" in raw_quality:
        cfg["enable_raw_quality_filter"] = bool(raw_quality["filter_bad_fits"])
    if "bad_frame_output_subdir" in raw_quality:
        cfg["raw_quality_bad_frame_output_subdir"] = str(
            raw_quality["bad_frame_output_subdir"]
        )

    for key, value in user_config.get("display", {}).items():
        cfg[key] = value
    if "radio_cmap" in cfg:
        cfg["cmap"] = cfg["radio_cmap"]

    gaussian = user_config.get("gaussian", {})
    gaussian_map = {
        "overlay_display_mode": "gaussian_overlay_display_mode",
        "fit_use_source_mask": "fit_use_source_mask",
        "fit_snr_threshold": "fit_snr_threshold",
        "fit_grow_snr_threshold": "fit_grow_snr_threshold",
        "fit_peak_fraction_threshold": "fit_peak_fraction_threshold",
        "fit_grow_peak_fraction_threshold": "fit_grow_peak_fraction_threshold",
        "fit_mask_target_min_pixels": "fit_mask_target_min_pixels",
        "fit_mask_target_max_pixels": "fit_mask_target_max_pixels",
        "fit_min_mask_pixels": "fit_min_mask_pixels",
        "fit_peak_fraction_threshold_min": "fit_peak_fraction_threshold_min",
        "fit_peak_fraction_threshold_max": "fit_peak_fraction_threshold_max",
        "fit_peak_fraction_threshold_step": "fit_peak_fraction_threshold_step",
        "gaussian_per_band_params": "gaussian_per_band_params",
        "gaussian_fit_roi_padding_pixels": "gaussian_fit_roi_padding_pixels",
        "gaussian_fit_max_pixels": "gaussian_fit_max_pixels",
        "gaussian_diagnostics_csv": "gaussian_diagnostics_csv",
        "gaussian_source_mode": "gaussian_source_mode",
        "multi_gaussian_source_count": "multi_gaussian_source_count",
        "multi_gaussian_max_sources": "multi_gaussian_max_sources",
        "multi_gaussian_min_peak_fraction": "multi_gaussian_min_peak_fraction",
        "multi_gaussian_min_peak_distance_pixels": (
            "multi_gaussian_min_peak_distance_pixels"
        ),
        "multi_gaussian_use_watershed": "multi_gaussian_use_watershed",
        "multi_gaussian_diagnostics_csv": "multi_gaussian_diagnostics_csv",
        "draw_multi_gaussian_labels": "draw_multi_gaussian_labels",
        "max_sigma_fraction": "max_sigma_fraction",
        "fit_background_model": "fit_background_model",
        "max_fwhm_arcsec": "max_fwhm_arcsec",
        "max_center_peak_distance_arcsec": "max_center_peak_distance_arcsec",
        "draw_raw_peak_marker": "draw_raw_peak_marker",
        "draw_fit_peak_distance": "draw_fit_peak_distance",
        "draw_coordinate_debug": "draw_coordinate_debug",
    }
    for user_key, flat_key in gaussian_map.items():
        if user_key in gaussian:
            cfg[flat_key] = gaussian[user_key]

    background = user_config.get("background", {})
    background_correction = bool(features.get("background_correction", False))
    bg_mode = str(background.get("mode", "off") or "off").lower()
    bg_display = bool(background.get("apply_to_display", False))
    bg_fit = bool(background.get("apply_to_fit", False))
    if not background_correction:
        if bg_display or bg_fit or bg_mode != "off":
            warnings.warn(
                "background_correction=False; ignoring background apply/mode settings.",
                stacklevel=2,
            )
        bg_mode = "off"
        bg_display = False
        bg_fit = False
    cfg["radio_background_strategy"] = bg_mode
    cfg["radio_background_mode"] = bg_mode
    cfg["radio_background_subtraction_mode"] = bg_mode
    cfg["background_use_for_display"] = bg_display
    cfg["background_use_for_fit"] = bg_fit
    cfg["display_background_subtracted_image"] = bg_display
    cfg["use_background_subtracted_for_gaussian_fit"] = bg_fit
    cfg["background_subtract_before_polarization_combine"] = bool(
        background.get("apply_before_polarization_combine", False)
    )
    if not background_correction or bg_mode == "off":
        cfg["radio_background_workflow"] = "off"
        cfg["enable_radio_background_subtraction"] = False
        cfg["radio_background_force_off"] = True
    else:
        cfg["radio_background_force_off"] = False
        cfg["enable_radio_background_subtraction"] = True
        if bg_display and bg_fit:
            cfg["radio_background_workflow"] = "display_and_fit"
        elif bg_display:
            cfg["radio_background_workflow"] = "display_only"
        elif bg_fit:
            cfg["radio_background_workflow"] = "fit_only"
        else:
            cfg["radio_background_workflow"] = "off"

    spectrogram = user_config.get("spectrogram", {})
    spectrogram_map = {
        "file_paths": "spectrogram_file_paths",
        "file_path": "spectrogram_file_path",
        "time_display_mode": "spectrogram_time_display_mode",
        "time_start": "spectrogram_time_start",
        "time_end": "spectrogram_time_end",
        "f_start": "spectrogram_f_start",
        "f_end": "spectrogram_f_end",
        "polarization": "spectrogram_polarization",
        "vmin": "spectrogram_vmin",
        "vmax": "spectrogram_vmax",
        "use_log10": "spectrogram_use_log10",
        "cmap": "spectrogram_cmap",
        "colorbar_label": "spectrogram_colorbar_label",
        "colorbar_unit": "spectrogram_colorbar_unit",
    }
    for user_key, flat_key in spectrogram_map.items():
        if user_key in spectrogram:
            cfg[flat_key] = spectrogram[user_key]

    drift_rate = user_config.get("drift_rate", {})
    if drift_rate:
        cfg["enable_drift_rate_overlay"] = bool(drift_rate.get("enabled", False))
        cfg["drift_rate_mode"] = drift_rate.get("mode", "off")
        cfg["drift_rate_interactive"] = dict(drift_rate.get("interactive", {}) or {})
        drift_map = {
            "selection_json": "drift_rate_selection_json",
            "selection_preview_png": "drift_rate_selection_preview_png",
            "selection_metadata_json": "drift_rate_selection_metadata_json",
            "draw_lines": "draw_drift_rate_lines",
            "draw_endpoints": "draw_drift_rate_endpoints",
            "draw_label": "draw_drift_rate_label",
            "draw_selected_id": "draw_drift_rate_selected_id",
            "label_format": "drift_rate_label_format",
            "line_width": "drift_rate_line_width",
            "endpoint_marker": "drift_rate_endpoint_marker",
            "endpoint_size": "drift_rate_endpoint_size",
            "save_drift_diagnostics": "save_drift_rate_diagnostics",
            "drift_diagnostics_csv": "drift_rate_diagnostics_csv",
        }
        for user_key, flat_key in drift_map.items():
            if user_key in drift_rate:
                cfg[flat_key] = drift_rate[user_key]

    for key, value in user_config.get("output", {}).items():
        cfg[key] = value

    cfg.setdefault("gaussian_valid_only_for_overlay", True)
    cfg.setdefault("gaussian_valid_only_for_trajectory", True)
    cfg.setdefault("gaussian_max_center_peak_distance_fraction_of_fwhm", 0.5)
    cfg.setdefault("gaussian_allow_moment_fallback_for_trajectory", False)
    cfg.setdefault("draw_raw_peak_marker", True)
    cfg.setdefault("draw_fit_peak_distance", False)
    cfg.setdefault("draw_coordinate_debug", False)
    cfg.setdefault("preserve_fits_wcs_orientation", True)
    cfg.setdefault("radio_image_origin_mode", "auto")
    cfg.setdefault("draw_coordinate_corner_debug", False)
    cfg.setdefault("enable_drift_rate_overlay", False)
    cfg.setdefault("drift_rate_mode", "off")
    cfg.setdefault("drift_rate_interactive", {})
    cfg["drift_rate_interactive"].setdefault("launch_policy", "cli_only")
    cfg["drift_rate_interactive"].setdefault("reuse_existing_selection", True)
    cfg["drift_rate_interactive"].setdefault("overwrite_selection", False)
    cfg["drift_rate_interactive"].setdefault("print_usage_hint", True)
    cfg["drift_rate_interactive"].setdefault("auto_increment_port", True)
    cfg["drift_rate_interactive"].setdefault("max_port_tries", 20)
    cfg.setdefault(
        "drift_rate_selection_json", "spectrogram_drift_rate_manual_selection.json"
    )
    cfg.setdefault(
        "drift_rate_selection_preview_png",
        "drift_rate_selection_preview",
    )
    cfg.setdefault(
        "drift_rate_selection_metadata_json",
        "spectrogram_drift_rate_selection_metadata.json",
    )
    cfg.setdefault("draw_drift_rate_lines", True)
    cfg.setdefault("draw_drift_rate_endpoints", True)
    cfg.setdefault("draw_drift_rate_label", True)
    cfg.setdefault("save_drift_rate_diagnostics", False)
    cfg.setdefault(
        "drift_rate_diagnostics_csv", "radio_spectrogram_drift_rate_diagnostics.csv"
    )
    return resolve_spatial_display(cfg).apply_to_legacy_config(cfg)


def resolve_spatial_display(
    cfg: dict,
    *,
    transform: str | None = None,
    render_profile: str | None = None,
) -> SpatialRadioDisplay:
    """Resolve the shared contract without changing workflow science.

    Single-band Source Map output remains linear and multi-band output remains
    log10.  This scientific invariant has higher precedence than UI, CLI,
    event, or saved display settings.
    """

    mode = str(cfg.get("mode", "single_band") or "single_band").lower()
    scientific_transform = transform or ("log10" if mode == "multi_band" else "linear")
    profile = render_profile or str(cfg.get("render_profile", "export") or "export")
    return spatial_display_for_source_map(
        cfg,
        transform=scientific_transform,  # type: ignore[arg-type]
        render_profile=profile,  # type: ignore[arg-type]
    )


def _gaussian_band_key(freq):
    try:
        freq_float = float(freq)
    except (TypeError, ValueError):
        return str(freq)
    if np.isfinite(freq_float) and abs(freq_float - round(freq_float)) < 1e-6:
        return str(int(round(freq_float)))
    return f"{freq_float:g}"


_BAND_GAUSSIAN_QUALITY_KEYS = {
    "max_fwhm_arcsec": "max_fwhm_arcsec",
    "max_center_peak_distance_arcsec": "max_center_peak_distance_arcsec",
    "fit_snr_threshold": "min_snr",
}


def _merge_band_gaussian_quality_config(band_cfg: dict, overrides: dict) -> None:
    quality_cfg = dict(band_cfg.get("gaussian_quality_requirements", {}) or {})
    nested_quality = overrides.get("gaussian_quality_requirements", {})
    if isinstance(nested_quality, dict):
        quality_cfg.update(nested_quality)
    for override_key, quality_key in _BAND_GAUSSIAN_QUALITY_KEYS.items():
        if override_key in overrides:
            quality_cfg[quality_key] = overrides[override_key]
    if quality_cfg:
        band_cfg["gaussian_quality_requirements"] = quality_cfg


def config_for_gaussian_band(cfg: dict, freq) -> dict:
    band_cfg = dict(cfg)
    per_band = cfg.get("gaussian_per_band_params", {}) or {}
    if not isinstance(per_band, dict):
        return band_cfg
    candidates = [freq, _gaussian_band_key(freq)]
    try:
        freq_float = float(freq)
        if np.isfinite(freq_float) and abs(freq_float - round(freq_float)) < 1e-6:
            candidates.append(int(round(freq_float)))
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        if candidate in per_band and isinstance(per_band[candidate], dict):
            overrides = per_band[candidate]
            band_cfg.update(overrides)
            _merge_band_gaussian_quality_config(band_cfg, overrides)
            break
    band_cfg["_gaussian_band_freq"] = freq
    return band_cfg


CONFIG = build_config(USER_CONFIG, DEFAULT_CONFIG)
# ============================================================


_SPECTROGRAM_CACHE: SpectrogramCache | None = None
_DRIFT_RATE_RESULTS_CACHE: dict[tuple[str, str], list[DriftRateResult]] = {}
_DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS = set()


# ──────────────────────────────────────────────────────────────
# 动态频谱缓存与复合图工具
# ──────────────────────────────────────────────────────────────


def resolve_background_workflow(cfg: dict) -> str:
    if cfg.get("radio_background_force_off", False):
        return "off"
    strategy = str(cfg.get("radio_background_strategy", "") or "").lower()
    if strategy in {"off", "none", ""}:
        return "off"
    display = bool(
        cfg.get(
            "background_use_for_display",
            cfg.get("display_background_subtracted_image", False),
        )
    )
    fit = bool(
        cfg.get(
            "background_use_for_fit",
            cfg.get("use_background_subtracted_for_gaussian_fit", False),
        )
    )
    if strategy == "noise_map_only" and not display and not fit:
        return "off"
    if not cfg.get("enable_radio_background_subtraction", False) and not (
        display or fit
    ):
        return "off"
    if display and fit:
        return "display_and_fit"
    if display and not fit:
        return "display_only"
    if not display and fit:
        return "fit_only"
    return "off"


def background_enabled_for_display(cfg: dict) -> bool:
    return resolve_background_workflow(cfg) in {"display_only", "display_and_fit"}


def background_enabled_for_fit(cfg: dict) -> bool:
    return resolve_background_workflow(cfg) in {"fit_only", "display_and_fit"}


def background_workflow_enabled(cfg: dict) -> bool:
    return resolve_background_workflow(cfg) != "off"


def _gaussian_multi_source_enabled(cfg: dict) -> bool:
    return str(cfg.get("gaussian_source_mode", "single")).strip().lower() == "multi"


def _background_disabled_diag(source_file=None):
    return {
        "background_enabled": False,
        "background_mode_requested": "none",
        "background_mode_used": "none",
        "background_scale": 1.0,
        "background_file_count": 0,
        "warning": "",
        "source_file": source_file,
    }


def _plot_output_subdir(cfg: dict) -> str:
    """Choose an output subdirectory that reflects enabled overlays."""
    configured = str(cfg.get("analysis_subdir") or "").strip()
    if configured and configured.lower() != "auto":
        return configured
    use_gaussian = cfg.get("enable_gaussian_overlay", False)
    use_spec = _spectrogram_panel_enabled(cfg)
    show_bgsub = background_enabled_for_display(cfg)
    bgfit = background_enabled_for_fit(cfg) and use_gaussian and not show_bgsub
    if show_bgsub:
        parts = []
        if use_gaussian:
            parts.append("gaussian")
        if use_spec:
            parts.append("spectrogram")
        parts.append("background_subtracted")
        return "_".join(parts)
    if bgfit:
        return "gaussian_bgfit_overlay"
    if use_gaussian and use_spec:
        return "gaussian_spectrogram_overlay"
    if use_spec:
        return cfg.get("spectrogram_output_subdir", "radio_spectrogram_composite")
    if use_gaussian:
        return "gaussian_overlay"
    return "radio_source_maps"


def _output_suffix(cfg: dict) -> str:
    parts = []
    show_bgsub = background_enabled_for_display(cfg)
    bgfit = (
        background_enabled_for_fit(cfg)
        and cfg.get("enable_gaussian_overlay", False)
        and not show_bgsub
    )
    if show_bgsub:
        parts.append("bgsub")
    elif bgfit:
        parts.append("bgfit")
    if cfg.get("enable_gaussian_overlay", False):
        parts.append("gaussian")
    if _spectrogram_panel_enabled(cfg):
        parts.append("spectrogram")
    if parts:
        return "_" + "_".join(parts) + "_overlay"
    return ""


def _drift_output_path(cfg: dict, key: str) -> str:
    if key == "drift_rate_diagnostics_csv" and key not in cfg:
        key = "drift_diagnostics_csv"
    path = str(cfg.get(key, "") or "")
    if not path:
        path = str(DEFAULT_CONFIG.get(key, "") or key)
    if os.path.isabs(path):
        return path
    if key == "drift_rate_selection_json":
        return os.path.abspath(path)
    output_dir = cfg.get("output_dir") or os.getcwd()
    return os.path.join(output_dir, _plot_output_subdir(cfg), path)


def _radio_colormap(cfg: dict):
    return resolve_spatial_display(cfg).matplotlib_cmap()


def _extract_yyyymmdd_hint_from_path(path: str) -> datetime.date | None:
    """Extract the most specific YYYYMMDD date hint from a full path."""
    if not path:
        return None
    import re

    text = str(path).split("|")[0]
    candidates = re.findall(r"(?<!\d)(20\d{6})", text)
    for token in reversed(candidates):
        try:
            return datetime.datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            continue
    return None


def _parse_yyyyddd_date(date_part: str) -> datetime.date | None:
    try:
        year = int(date_part[:4])
        doy = int(date_part[4:])
        if doy < 1:
            return None
        date_value = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
        if date_value.year != year:
            return None
        return date_value
    except Exception:
        return None


def _parse_yyyymdd_date(date_part: str) -> datetime.date | None:
    try:
        year = int(date_part[:4])
        month = int(date_part[4])
        day = int(date_part[5:])
        return datetime.date(year, month, day)
    except Exception:
        return None


def _date_obs_hint_from_cfg(cfg: dict | None) -> datetime.date | None:
    if not cfg:
        return None
    parsed = _parse_datetime_value(cfg.get("spectrogram_time_start"))
    if parsed is not None:
        return parsed.date()
    path = cfg.get("spectrogram_file_path")
    hint = _extract_yyyymmdd_hint_from_path(path) if path else None
    if hint is not None:
        return hint
    if path and os.path.isfile(path):
        try:
            with fits.open(path, memmap=True) as hdul:
                parsed = _parse_datetime_value(
                    hdul[0].header.get("DATE-OBS") or hdul[0].header.get("DATE_OBS")
                )
            if parsed is not None:
                return parsed.date()
        except Exception:
            return None
    return None


def _parse_radio_date_part(
    date_part: str, path: str | None, cfg: dict | None
) -> datetime.date | None:
    cfg = cfg or CONFIG
    fmt = str(cfg.get("date_format", "auto")).lower()
    path_hint = _extract_yyyymmdd_hint_from_path(path) if path else None

    if len(date_part) == 8:
        try:
            return datetime.datetime.strptime(date_part, "%Y%m%d").date()
        except ValueError:
            return _parse_yyyyddd_date(date_part)

    if len(date_part) == 7:
        as_yyyymdd = _parse_yyyymdd_date(date_part)
        as_yyyyddd = _parse_yyyyddd_date(date_part)
        if fmt in {"yyyymdd", "yyyy-m-dd"}:
            return as_yyyymdd
        if fmt in {"yyyyddd", "yyyy-ddd", "7digit"}:
            return as_yyyyddd
        if fmt == "auto":
            # Seven digits are ambiguous (YYYYMDD versus YYYYDDD).  Auto mode
            # is deliberately calendar-first so unrelated dated parent
            # directories cannot silently change the scientific timestamp.
            # Day-of-year inputs remain available through date_format=yyyyddd.
            if as_yyyymdd is not None:
                return as_yyyymdd
            if path_hint is not None and as_yyyyddd == path_hint:
                return as_yyyyddd
            target = path_hint or _date_obs_hint_from_cfg(cfg)
            valid = [d for d in (as_yyyymdd, as_yyyyddd) if d is not None]
            if target is not None and valid:
                return min(valid, key=lambda d: abs((d - target).days))
            return as_yyyyddd
        return as_yyyymdd or as_yyyyddd

    if len(date_part) == 6:
        yy = int(date_part[:2])
        year = 2000 + yy
        as_yymmdd = None
        as_yyddd = None
        try:
            as_yymmdd = datetime.date(year, int(date_part[2:4]), int(date_part[4:6]))
        except ValueError:
            pass
        try:
            doy = int(date_part[2:])
            if doy >= 1:
                as_yyddd = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
                if as_yyddd.year != year:
                    as_yyddd = None
        except Exception:
            pass
        if fmt in {"yymmdd", "yymmdd_or_yyddd"}:
            return as_yymmdd or as_yyddd
        if fmt in {"yyddd", "6digit"}:
            return as_yyddd
        if fmt == "auto":
            target = path_hint or _date_obs_hint_from_cfg(cfg)
            valid = [d for d in (as_yymmdd, as_yyddd) if d is not None]
            if target is not None and valid:
                return min(valid, key=lambda d: abs((d - target).days))
            return as_yymmdd or as_yyddd
    return None


def _radio_datetime_from_filename(
    path: str | None, cfg: dict | None = None
) -> datetime.datetime | None:
    if not path:
        return None
    path = path.split("|")[0]
    name = os.path.basename(path)
    import re

    match = re.search(r"_(\d{6,8})_(\d{6})(?:_(\d{1,6}))?", name)
    if not match:
        return None
    date_part, time_part, frac_part = match.groups()
    try:
        base_date = _parse_radio_date_part(date_part, path, cfg)
        if base_date is None:
            return None
        hour = int(time_part[0:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        microsecond = 0
        if frac_part:
            microsecond = int(frac_part.ljust(6, "0")[:6])
        return datetime.datetime.combine(base_date, datetime.time()).replace(
            hour=hour, minute=minute, second=second, microsecond=microsecond
        )
    except Exception:
        return None


def radio_datetime_from_header_or_path(
    header=None, path: str | None = None, cfg: dict | None = None
) -> datetime.datetime | None:
    """Get observation datetime from FITS header first, then filename."""
    if header is not None:
        date_obs = str(header.get("DATE-OBS", header.get("DATE_OBS", ""))).strip()
        time_obs = str(header.get("TIME-OBS", header.get("TIME_OBS", ""))).strip()
        candidates = []
        if date_obs and time_obs and "T" not in date_obs and " " not in date_obs:
            candidates.append(f"{date_obs} {time_obs}")
        if date_obs:
            candidates.append(date_obs)
        for candidate in candidates:
            parsed = _parse_datetime_value(candidate)
            if parsed is not None:
                return parsed
    return _radio_datetime_from_filename(path, cfg)


def _radio_item_datetime(item, cfg: dict | None = None) -> datetime.datetime | None:
    """Parse datetime from a single path, RR/LL tuple, or diagnostic source string."""
    if isinstance(item, (tuple, list)) and item:
        return _radio_item_datetime(item[0], cfg)
    if not isinstance(item, str):
        return None
    path = item.split("|")[0]
    try:
        if os.path.isfile(path):
            with fits.open(path, memmap=True) as hdul:
                header = (
                    hdul[1].header
                    if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU)
                    else hdul[0].header
                )
            parsed = radio_datetime_from_header_or_path(header, path, cfg)
            if parsed is not None:
                return parsed
    except Exception:
        pass
    return _radio_datetime_from_filename(path, cfg)


def _derive_radio_time_range(
    items, margin_seconds: float = 30.0, cfg: dict | None = None
):
    """Derive [start, end] for the spectrogram panel from radio image files/slots."""
    times = []
    if not items:
        return None
    for item in items:
        if isinstance(item, list):
            for sub in item:
                dt = _radio_item_datetime(sub, cfg)
                if dt is not None:
                    times.append(dt)
        else:
            dt = _radio_item_datetime(item, cfg)
            if dt is not None:
                times.append(dt)
    if not times:
        return None
    margin = datetime.timedelta(seconds=float(margin_seconds))
    return min(times) - margin, max(times) + margin


def resolve_spectrogram_time_window(cfg, radio_time_range, dt_base, time_arr):
    finite_time = np.asarray(time_arr, dtype=np.float64)
    finite_time = finite_time[np.isfinite(finite_time)]
    if finite_time.size == 0:
        return None, None, "empty"
    file_start = dt_base + datetime.timedelta(seconds=float(np.nanmin(finite_time)))
    file_end = dt_base + datetime.timedelta(seconds=float(np.nanmax(finite_time)))
    mode = str(cfg.get("spectrogram_time_display_mode", "user") or "user").lower()
    if mode not in {"user", "auto_radio", "full"}:
        mode = "user"

    if mode == "full":
        t_start, t_end = file_start, file_end
    elif mode == "auto_radio":
        if radio_time_range is not None:
            margin = datetime.timedelta(
                seconds=float(cfg.get("spectrogram_time_margin_seconds", 30.0))
            )
            t_start = radio_time_range[0] - margin
            t_end = radio_time_range[1] + margin
        else:
            t_start, t_end = file_start, file_end
    else:
        t_start = _parse_datetime_value(cfg.get("spectrogram_time_start")) or file_start
        t_end = _parse_datetime_value(cfg.get("spectrogram_time_end")) or file_end

    if t_start > t_end:
        t_start, t_end = t_end, t_start
    print(f"[Spectrogram time window] mode={mode}, start={t_start}, end={t_end}")
    return t_start, t_end, mode


def _prepare_spectrogram_colorbar_label(cfg: dict):
    """Resolve the spectrogram unit locally without changing the public helper."""

    prepared = cfg.get("_spectrogram_unit_resolution")
    if isinstance(prepared, dict):
        return prepared
    headers = []
    for path in _normalize_spectrogram_paths(cfg):
        try:
            with fits.open(path, memmap=True) as hdul:
                headers.append(hdul[0].header.copy())
        except Exception as exc:
            warnings.warn(
                f"Could not inspect spectrogram BUNIT from {path}: {exc}",
                stacklevel=2,
            )
            headers.append(None)
    resolution = resolve_colorbar_unit(headers, cfg.get("spectrogram_colorbar_unit"))
    requested_pol = str(cfg.get("spectrogram_polarization", "sum") or "sum").lower()
    use_log = bool(cfg.get("spectrogram_use_log10", True)) and requested_pol not in {
        "ratio",
        "polarization_ratio",
    }
    cfg["spectrogram_colorbar_label"] = colorbar_label(
        resolution, transform="log10" if use_log else "linear"
    )
    cfg["_spectrogram_unit_resolution"] = resolution.to_dict()
    cfg["_spectrogram_unit_warnings"] = list(resolution.warnings)
    for message in resolution.warnings:
        warnings.warn(message, stacklevel=2)
    return cfg["_spectrogram_unit_resolution"]


def get_spectrogram_cache(cfg: dict) -> SpectrogramCache | None:
    global _SPECTROGRAM_CACHE
    if not _spectrogram_panel_enabled(cfg):
        return None
    if _SPECTROGRAM_CACHE is None:
        _prepare_spectrogram_colorbar_label(cfg)
        _SPECTROGRAM_CACHE = build_spectrogram_cache(cfg)
    return _SPECTROGRAM_CACHE


def get_or_load_drift_rate_results(
    cache, cfg, launch_func=None
) -> list[DriftRateResult]:
    if not cfg.get("enable_drift_rate_overlay", False):
        return []
    launch_func = launch_func or launch_drift_selection_server
    mode = str(cfg.get("drift_rate_mode", "off") or "off").lower()
    if mode == "off":
        return []
    selection_path = _drift_output_path(cfg, "drift_rate_selection_json")
    if cfg.get("_drift_selection_cli_path"):
        selection_path = cfg["_drift_selection_cli_path"]
    interactive = dict(cfg.get("drift_rate_interactive", {}) or {})
    launch_policy = str(interactive.get("launch_policy", "cli_only") or "cli_only")
    cache_key = (mode, os.path.abspath(selection_path), launch_policy)
    if cache_key in _DRIFT_RATE_RESULTS_CACHE:
        return _DRIFT_RATE_RESULTS_CACHE[cache_key]
    selection_exists = os.path.exists(selection_path)

    def _load_selection_payload():
        payload = _load_drift_selection_payload(selection_path)
        source_file = payload.get("source_file")
        if source_file and os.path.abspath(str(source_file)) != os.path.abspath(
            cache.source_file
        ):
            warnings.warn(
                "Drift-rate selection source_file differs from current "
                f"spectrogram_file_path: {source_file}",
                stacklevel=2,
            )
        return list(payload.get("lines", []) or [])

    if mode == "interactive_manual":
        if cfg.get("_select_drift_now", False):
            lines = launch_func(cache, cfg)
        elif launch_policy == "always":
            lines = launch_func(cache, cfg)
        elif launch_policy == "auto_if_missing" and not selection_exists:
            print(
                "[Drift selection] selection JSON not found; "
                "starting interactive selector..."
            )
            lines = launch_func(cache, cfg)
        elif selection_exists:
            lines = _load_selection_payload()
        else:
            hint = (
                "No drift-rate selection JSON found. Run:\n"
                "  python radio_source_map_plot_gaussian_overlay.py "
                "--select-drift --drift-port 8050\n"
                "or set drift_rate.interactive.launch_policy='auto_if_missing'."
            )
            if interactive.get("print_usage_hint", True):
                print(f"[Drift selection] {hint}")
            warnings.warn(
                f"No drift-rate selection JSON found: {selection_path}",
                stacklevel=2,
            )
            return []
    elif mode == "manual_json":
        if not selection_exists:
            warnings.warn(
                "No drift-rate selection JSON found for manual_json mode. Run: "
                "python radio_source_map_plot_gaussian_overlay.py --select-drift "
                "--drift-port 8050",
                stacklevel=2,
            )
            return []
        lines = _load_selection_payload()
    elif mode in {"auto_peak", "auto_ridge"}:
        warnings.warn(
            f"drift_rate_mode={mode!r} is reserved for future implementation.",
            stacklevel=2,
        )
        return []
    else:
        return []
    results = [calculate_drift_rate_from_line(line) for line in lines]
    results = _mark_drift_range_warnings(results, cache)
    _DRIFT_RATE_RESULTS_CACHE[cache_key] = results
    return results


def overlay_spectrogram_panel(ax, cfg: dict, current_time: datetime.datetime | None):
    """Draw cached dynamic spectrum and the current-time vertical dashed line."""
    cache = get_spectrogram_cache(cfg)
    if cache is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Spectrogram unavailable", ha="center", va="center")
        return None
    x_start, x_end = cache.display_time_nums
    display_data, extent, f_min, f_max = _spectrogram_display_data_extent(cache)
    im = ax.imshow(
        display_data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cache.cmap,
        vmin=cache.vmin,
        vmax=cache.vmax,
    )
    if cfg.get("enable_drift_rate_overlay", False):
        drift_results = get_or_load_drift_rate_results(cache, cfg)
        overlay_drift_rate_results(ax, drift_results, cfg)
        if cfg.get("save_drift_rate_diagnostics", False):
            save_drift_rate_diagnostics_once(drift_results, cfg, cache.source_file)
    if current_time is not None:
        current_num = mdates.date2num(current_time)
        in_range = x_start <= current_num <= x_end
        if in_range or not cfg.get("spectrogram_clip_current_time_line", True):
            ax.axvline(
                current_num,
                color=cfg.get("spectrogram_line_color", "white"),
                linestyle=cfg.get("spectrogram_line_style", "--"),
                linewidth=cfg.get("spectrogram_line_width", 1.6),
                alpha=cfg.get("spectrogram_line_alpha", 0.95),
                zorder=7,
            )
        elif cfg.get("spectrogram_show_out_of_range_time_note", True):
            ax.text(
                0.02,
                0.94,
                "Current frame time outside displayed spectrum range",
                transform=ax.transAxes,
                fontsize=max(cfg.get("annotation_fontsize", 20) - 10, 8),
                color=cfg.get("tick_color", "black"),
                va="top",
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none"),
            )
    ax.set_title(cache.title, fontsize=max(cfg.get("title_fontsize", 24) - 8, 9), pad=4)
    ax.set_ylabel(
        "Frequency (MHz)", fontsize=max(cfg.get("label_fontsize", 28) - 12, 9)
    )
    ax.set_xlabel("Time (UT)", fontsize=max(cfg.get("label_fontsize", 28) - 12, 9))
    ax.xaxis_date()
    span_seconds = float((x_end - x_start) * 86400.0)
    if np.isfinite(span_seconds) and span_seconds > 0:
        ax.xaxis.set_major_locator(_spectrogram_time_locator(cfg, span_seconds))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(cfg.get("spectrogram_xtick_format", "%H:%M:%S"))
    )
    ax.tick_params(
        labelsize=max(cfg.get("tick_fontsize", 22) - 10, 8),
        colors=cfg.get("tick_color", "black"),
    )
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(f_min, f_max)
    if cfg.get("spectrogram_draw_colorbar", True):
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label(
            cache.cbar_label, fontsize=max(cfg.get("tick_fontsize", 22) - 11, 8)
        )
        cbar.ax.tick_params(labelsize=max(cfg.get("tick_fontsize", 22) - 12, 8))
    return im


# ──────────────────────────────────────────────────────────────
# 内存与核心数工具
# ──────────────────────────────────────────────────────────────


def _estimate_safe_workers(file_list: list, requested, memory_per_worker_mb) -> int:
    """
    Estimate safe number of parallel worker processes based on available physical memory and estimated memory per worker.

    Parameters
    ----------
    file_list            : list of files to be processed (used to estimate memory per frame)
    requested            : max_workers specified by the user in CONFIG (None means auto)
    memory_per_worker_mb : estimated memory per worker (MB); None means auto estimation

    Returns
    -------
    safe number of workers (at least 1)
    """
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024**2)
    except ImportError:
        warnings.warn(
            "psutil not found, cannot auto-estimate memory safety; please run `pip install psutil`. "
            "This run will use a conservative max_workers=2.",
            stacklevel=2,
        )
        cpu_count = os.cpu_count() or 1
        return min(requested or 2, cpu_count)

    # 估算单帧占用内存
    if memory_per_worker_mb is None:
        if file_list:
            try:
                sample = file_list[:3]
                total_bytes = 0
                count = 0

                for item in sample:
                    if isinstance(item, tuple):
                        # 如果是元组，取第一个文件（RR文件）作为大小估计
                        file_path = item[0]
                    else:
                        file_path = item

                    try:
                        total_bytes += os.path.getsize(file_path)
                        count += 1
                    except (OSError, TypeError):
                        # 如果文件不存在或路径有问题，跳过
                        continue

                if count > 0:
                    avg_bytes = total_bytes / count
                    memory_per_worker_mb = avg_bytes * 20 / (1024**2)
                else:
                    memory_per_worker_mb = 500.0
            except (OSError, TypeError):
                memory_per_worker_mb = 500.0
        else:
            memory_per_worker_mb = 500.0

    # Keep 20% available memory as system buffer
    usable_mb = available_mb * 0.80
    mem_safe = max(1, int(usable_mb / memory_per_worker_mb))
    cpu_count = os.cpu_count() or 1

    if requested is not None:
        if requested > mem_safe:
            warnings.warn(
                f"[Memory warning] You set max_workers={requested}, "
                f"but according to available memory {available_mb:.0f} MB and per-worker estimate "
                f"{memory_per_worker_mb:.0f} MB, the safe upper limit is about {mem_safe}. "
                f"Automatically adjusted to {mem_safe}, please modify CONFIG['max_workers'].",
                stacklevel=2,
            )
            return mem_safe
        return requested

    auto = min(cpu_count, mem_safe)
    print(
        f"[Auto estimation] Available memory {available_mb:.0f} MB, "
        f"per worker estimate {memory_per_worker_mb:.0f} MB, "
        f"CPU cores {cpu_count}  →  max_workers = {auto}"
    )
    return auto


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────


def get_sorted_fits(
    data_dir: str,
    start: int,
    end: int | None,
    *,
    study_mode: str | None = None,
) -> list:
    """Return sorted list of FITS file paths within the specified range."""
    mode = _require_study_mode(study_mode)
    frozen = _frozen_files_for_band(Path(data_dir), required=mode == "confirmatory")
    if frozen is not None:
        return [str(path) for path in frozen]
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".fits")
    )
    if not all_files:
        raise FileNotFoundError(f"目录 {data_dir} 中未找到 FITS 文件")
    selected = all_files[start:end]
    if not selected:
        raise ValueError(f"索引范围 [{start}, {end}) 内没有文件，请检查参数")
    return selected


def read_fits(file_path: str):
    """
    Read FITS file, return (img_data 2D ndarray, header).
    Prefer ImageHDU (hdul[1]), otherwise use primary HDU (hdul[0]).
    memmap=True speeds up sequential reading of large files, .copy() ensures data remains valid after hdul is closed.
    """
    with fits.open(file_path, memmap=True) as hdul:
        if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU):
            data, header = hdul[1].data.copy(), hdul[1].header
        else:
            data, header = hdul[0].data.copy(), hdul[0].header

    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"数据维度异常：{data.ndim}D，需要 2D，文件：{file_path}")
    return data, header


def calc_extent(header, img_shape, cfg=None):
    return calc_image_extent_arcsec(header, img_shape, cfg)


def calc_image_extent_arcsec(header, img_shape, cfg=None) -> list[float]:
    """
    Calculate image extent in arcsec using matplotlib's standard order:
    [left, right, bottom, top].

    When preserve_fits_wcs_orientation=True, the edge order keeps the FITS
    CDELT signs. Matplotlib accepts inverted extents and the matching image
    origin is selected by get_radio_image_origin().
    """
    cfg = cfg or CONFIG
    try:
        crval1, crpix1, cdelt1 = header["CRVAL1"], header["CRPIX1"], header["CDELT1"]
        crval2, crpix2, cdelt2 = header["CRVAL2"], header["CRPIX2"], header["CDELT2"]
        ny, nx = img_shape
        x_edge0 = crval1 + (0.5 - crpix1) * cdelt1
        x_edge1 = crval1 + (nx + 0.5 - crpix1) * cdelt1
        y_edge0 = crval2 + (0.5 - crpix2) * cdelt2
        y_edge1 = crval2 + (ny + 0.5 - crpix2) * cdelt2
        if cfg.get("preserve_fits_wcs_orientation", True):
            return [float(x_edge0), float(x_edge1), float(y_edge0), float(y_edge1)]
        left, right = sorted((float(x_edge0), float(x_edge1)))
        bottom, top = sorted((float(y_edge0), float(y_edge1)))
        return [left, right, bottom, top]
    except KeyError:
        warnings.warn(
            "Header lacks WCS coordinate keywords, using default extent [-1500,1500]",
            stacklevel=2,
        )
        return [-1500.0, 1500.0, -1500.0, 1500.0]


def get_radio_image_origin(header, cfg) -> str:
    mode = str(cfg.get("radio_image_origin_mode", "auto") or "auto").lower()
    if mode in {"upper", "lower"}:
        return mode
    if cfg.get("preserve_fits_wcs_orientation", True):
        return "lower"
    return "upper"


def calc_radio_extent_and_origin(header, img_shape, cfg) -> tuple[list[float], str]:
    extent = calc_image_extent_arcsec(header, img_shape, cfg)
    origin = get_radio_image_origin(header, cfg)
    return extent, origin


def get_imshow_kwargs(
    data, extent, cfg, *, cmap=None, aspect="equal", origin=None
) -> dict:
    return {
        "extent": extent,
        "origin": origin or cfg.get("_current_radio_image_origin", "upper"),
        "cmap": cmap or _radio_colormap(cfg),
        "aspect": aspect,
    }


def add_radio_coordinate_corner_debug(ax, extent, shape, origin, cfg):
    if not cfg.get("draw_coordinate_corner_debug", False):
        return
    ny, nx = shape
    corners = [
        (0, 0, "pixel(0,0)"),
        (nx - 1, 0, f"pixel({nx - 1},0)"),
        (0, ny - 1, f"pixel(0,{ny - 1})"),
        (nx - 1, ny - 1, f"pixel({nx - 1},{ny - 1})"),
    ]
    for x_pix, y_pix, corner_label in corners:
        x_arc, y_arc = pixel_to_data_coord(x_pix, y_pix, extent, shape, origin)
        ax.scatter([x_arc], [y_arc], marker="s", s=26, c="cyan", zorder=10)
        ax.annotate(
            f"{corner_label}\n({x_arc:.1f}, {y_arc:.1f})",
            xy=(x_arc, y_arc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=max(cfg.get("annotation_fontsize", 20) - 12, 7),
            color="cyan",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
            zorder=11,
        )


def _radio_background_mode(cfg):
    return cfg.get(
        "radio_background_subtraction_mode",
        cfg.get("radio_background_mode", "local_median"),
    )


def _safe_background_median_size(cfg):
    size = int(cfg.get("background_local_median_size", 31))
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    return size


def _background_file_matches(path, band=None, polarization=None):
    text = os.path.normcase(path)
    if band is not None and band != "Unknown":
        band_text = str(int(band)) if isinstance(band, (int, float)) else str(band)
        if band_text and band_text not in text:
            return False
    if polarization:
        pol_text = str(polarization).upper()
        if "RR" in pol_text or "LL" in pol_text:
            wanted = "RR" if "RR" in pol_text else "LL"
            if wanted.lower() not in text:
                return False
    return True


def find_background_files_for_current_radio(
    current_radio_file: str,
    cfg: dict,
    band: str | float | None = None,
    polarization: str | None = None,
) -> list[str]:
    if not current_radio_file or "|" in current_radio_file:
        return []
    current_time = _radio_datetime_from_filename(current_radio_file, cfg)
    if current_time is None:
        return []
    directory = os.path.dirname(current_radio_file)
    try:
        names = os.listdir(directory)
    except OSError:
        return []

    window_seconds = float(cfg.get("background_pre_event_seconds", 60.0))
    current_abs = os.path.abspath(current_radio_file)
    matches = []
    for name in names:
        if not name.lower().endswith((".fits", ".fit", ".fts")):
            continue
        path = os.path.join(directory, name)
        if os.path.abspath(path) == current_abs:
            continue
        if not _background_file_matches(path, band, polarization):
            continue
        frame_time = _radio_datetime_from_filename(path, cfg)
        if frame_time is None or frame_time >= current_time:
            continue
        delta = (current_time - frame_time).total_seconds()
        if 0 < delta <= window_seconds:
            matches.append((frame_time, path))
    matches.sort()
    return [path for _, path in matches]


def build_radio_background_map(background_files, cfg, target_shape=None):
    frames = []
    for path in background_files:
        try:
            data, _header = read_fits(path)
        except Exception as exc:
            warnings.warn(f"Skipping background frame {path}: {exc}", stacklevel=2)
            continue
        if target_shape is not None and data.shape != target_shape:
            continue
        frames.append(np.asarray(data, dtype=np.float64))
    if len(frames) < int(cfg.get("background_min_frames", 5)):
        return None, len(frames)
    return np.nanmedian(np.stack(frames, axis=0), axis=0), len(frames)


def _estimate_background_scale(data, background_map, cfg):
    if cfg.get("background_scale_mode", "offsource_median") == "none":
        return 1.0
    finite = np.isfinite(data) & np.isfinite(background_map)
    if not np.any(finite):
        return 1.0
    peak = float(np.nanmax(data[finite]))
    offsource = finite & (data <= 0.30 * peak)
    if np.count_nonzero(offsource) < 10:
        offsource = finite
    data_med = float(np.nanmedian(data[offsource]))
    bg_med = float(np.nanmedian(background_map[offsource]))
    if not np.isfinite(data_med) or not np.isfinite(bg_med) or bg_med == 0:
        return 1.0
    scale = data_med / bg_med
    if not np.isfinite(scale) or scale <= 0 or scale > 10:
        return 1.0
    return float(scale)


def _subtract_with_background_map(work, background_map, cfg):
    scale = _estimate_background_scale(work, background_map, cfg)
    scaled_background = scale * background_map
    data_sub_raw = work - scaled_background
    _, noise_after = estimate_background_noise(data_sub_raw)
    data_sub = data_sub_raw
    if cfg.get("clip_negative_after_background", True):
        data_sub = np.where(data_sub < 0, 0.0, data_sub)
    return data_sub, scaled_background, scale, noise_after


def subtract_radio_background(
    data: np.ndarray,
    cfg: dict,
    source_file: str | None = None,
    band: str | float | None = None,
    polarization: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    work = np.asarray(data, dtype=np.float64)
    background_level, noise_before = estimate_background_noise(work)
    mode_requested = _radio_background_mode(cfg)
    enabled = background_workflow_enabled(cfg)
    diagnostics = {
        "background_enabled": enabled,
        "background_mode_requested": mode_requested,
        "background_mode_used": "none",
        "background_scale": 1.0,
        "background_level": background_level,
        "noise_sigma_before": noise_before,
        "noise_sigma_after": noise_before,
        "background_file_count": 0,
        "warning": "",
        "source_file": source_file,
    }
    if cfg.get("radio_background_force_off", False):
        diagnostics["background_enabled"] = False
        diagnostics["background_mode_used"] = "none"
        return work.copy(), None, diagnostics
    if not enabled or mode_requested == "none":
        diagnostics["background_enabled"] = False
        return work.copy(), None, diagnostics

    background_map = None
    mode_used = mode_requested
    warning = ""

    try:
        if mode_requested == "constant":
            background_map = np.full_like(work, background_level, dtype=np.float64)
        elif mode_requested == "local_median":
            size = _safe_background_median_size(cfg)
            filled = np.where(np.isfinite(work), work, background_level)
            background_map = median_filter(filled, size=size)
        elif mode_requested == "pre_event_median":
            background_files = find_background_files_for_current_radio(
                source_file, cfg, band, polarization
            )
            background_map, file_count = build_radio_background_map(
                background_files, cfg, target_shape=work.shape
            )
            diagnostics["background_file_count"] = file_count
            if background_map is None:
                print(
                    "[背景扣除] pre_event_median 背景帧不足，fallback 到 local_median"
                )
                warning = "pre_event_median insufficient; fallback to local_median"
                mode_used = "local_median"
                size = _safe_background_median_size(cfg)
                filled = np.where(np.isfinite(work), work, background_level)
                background_map = median_filter(filled, size=size)
            else:
                mode_used = "pre_event_median"
        elif mode_requested == "plane_only":
            diagnostics["background_mode_used"] = "plane_only"
            print(
                "[背景扣除] enabled=True, mode=plane_only, used=plane_only, "
                "scale=1.000, file_count=0"
            )
            return work.copy(), None, diagnostics
        else:
            warning = f"unknown mode {mode_requested!r}; fallback to local_median"
            mode_used = "local_median"
            size = _safe_background_median_size(cfg)
            filled = np.where(np.isfinite(work), work, background_level)
            background_map = median_filter(filled, size=size)
    except Exception as exc:
        warning = f"{mode_requested} failed: {exc}; fallback to constant"
        mode_used = "constant"
        try:
            background_map = np.full_like(work, background_level, dtype=np.float64)
        except Exception as const_exc:
            diagnostics["warning"] = f"{warning}; constant failed: {const_exc}"
            return work.copy(), None, diagnostics

    if background_map is None:
        diagnostics["warning"] = warning or "background map unavailable"
        return work.copy(), None, diagnostics

    data_sub, background_map, scale, noise_after = _subtract_with_background_map(
        work, background_map, cfg
    )
    diagnostics.update(
        {
            "background_mode_used": mode_used,
            "background_scale": scale,
            "noise_sigma_after": noise_after,
            "warning": warning,
        }
    )
    print(
        f"[背景扣除] enabled=True, mode={mode_requested}, used={mode_used}, "
        f"scale={scale:.3f}, file_count={diagnostics['background_file_count']}"
    )
    return data_sub, background_map, diagnostics


def subtract_radio_background_for_fit(data, cfg):
    old_cfg = dict(cfg)
    old_cfg["enable_radio_background_subtraction"] = True
    old_cfg["radio_background_subtraction_mode"] = cfg.get(
        "radio_background_mode",
        cfg.get("radio_background_subtraction_mode", "local_median"),
    )
    radio_fit_data, background_map, diagnostics = subtract_radio_background(
        data, old_cfg
    )
    return (
        radio_fit_data,
        background_map,
        diagnostics.get("background_level"),
        diagnostics.get("noise_sigma_before"),
    )


def _robust_median_mad(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(arr))
    return med, rms


def _sigma_clip_values(values, sigma=3.0, iters=3):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    sigma = float(sigma)
    iters = max(int(iters), 0)
    for _ in range(iters):
        med, rms = _robust_median_mad(arr)
        if not np.isfinite(rms) or rms <= 0 or not np.isfinite(med):
            break
        keep = np.abs(arr - med) <= sigma * rms
        if np.count_nonzero(keep) == arr.size:
            break
        arr = arr[keep]
        if arr.size == 0:
            break
    return arr


def _mesh_values_to_map(mesh_y, mesh_x, mesh_values, shape, fill_value):
    ny, nx = shape
    if len(mesh_values) == 0:
        return np.full(shape, fill_value, dtype=np.float64)
    mesh_y = np.asarray(mesh_y, dtype=np.float64)
    mesh_x = np.asarray(mesh_x, dtype=np.float64)
    values = np.asarray(mesh_values, dtype=np.float64)
    valid = np.isfinite(mesh_y) & np.isfinite(mesh_x) & np.isfinite(values)
    mesh_y = mesh_y[valid]
    mesh_x = mesh_x[valid]
    values = values[valid]
    if values.size == 0:
        return np.full(shape, fill_value, dtype=np.float64)

    y_unique = np.unique(mesh_y)
    x_unique = np.unique(mesh_x)
    grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float64)
    y_lookup = {v: i for i, v in enumerate(y_unique)}
    x_lookup = {v: i for i, v in enumerate(x_unique)}
    for yv, xv, val in zip(mesh_y, mesh_x, values, strict=False):
        grid[y_lookup[yv], x_lookup[xv]] = val

    global_fill = float(np.nanmedian(values)) if values.size else fill_value
    if not np.isfinite(global_fill):
        global_fill = fill_value
    grid = np.where(np.isfinite(grid), grid, global_fill)

    x_pixels = np.arange(nx, dtype=np.float64)
    y_pixels = np.arange(ny, dtype=np.float64)
    row_interp = np.empty((grid.shape[0], nx), dtype=np.float64)
    for i in range(grid.shape[0]):
        row_interp[i, :] = np.interp(x_pixels, x_unique, grid[i, :])
    out = np.empty((ny, nx), dtype=np.float64)
    for j in range(nx):
        out[:, j] = np.interp(y_pixels, y_unique, row_interp[:, j])
    return out


def estimate_background_rms_mesh(data, cfg, source_mask=None):
    work = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(work)
    finite_values = work[finite]
    global_bg, global_rms = _robust_median_mad(finite_values)
    if not np.isfinite(global_bg):
        global_bg = 0.0
    if not np.isfinite(global_rms) or global_rms <= 0:
        global_rms = 1.0

    mesh_size = max(int(cfg.get("background_mesh_size", 96)), 1)
    mesh_step = max(int(cfg.get("background_mesh_step", mesh_size)), 1)
    min_valid = max(int(cfg.get("background_min_valid_pixels", 20)), 1)
    sigma = float(cfg.get("background_sigma_clip", 3.0))
    iters = int(cfg.get("background_sigma_clip_iters", 3))
    exclude = np.zeros(work.shape, dtype=np.bool_)
    if source_mask is not None:
        exclude = np.asarray(source_mask, dtype=np.bool_)
        if exclude.shape != work.shape:
            exclude = np.zeros(work.shape, dtype=np.bool_)

    diagnostics = {
        "background_strategy": cfg.get("radio_background_strategy", "noise_map_only"),
        "background_mesh_size": mesh_size,
        "background_mesh_step": mesh_step,
        "background_rms_median": np.nan,
        "background_level_median": np.nan,
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "mesh_count": 0,
        "warning": "",
    }
    if work.ndim != 2 or finite_values.size == 0:
        diagnostics["warning"] = "non_finite_data"
        bg = np.full_like(work, global_bg, dtype=np.float64)
        rms = np.full_like(work, global_rms, dtype=np.float64)
        return bg, _safe_rms_map(rms), diagnostics

    ny, nx = work.shape
    y_starts = list(range(0, ny, mesh_step))
    x_starts = list(range(0, nx, mesh_step))
    mesh_y = []
    mesh_x = []
    bg_values = []
    rms_values = []
    for y0 in y_starts:
        y1 = min(y0 + mesh_size, ny)
        if y1 <= y0:
            continue
        for x0 in x_starts:
            x1 = min(x0 + mesh_size, nx)
            if x1 <= x0:
                continue
            box = work[y0:y1, x0:x1]
            valid = np.isfinite(box)
            if source_mask is not None:
                valid &= ~exclude[y0:y1, x0:x1]
            values = box[valid]
            if values.size < min_valid:
                continue
            clipped = _sigma_clip_values(values, sigma=sigma, iters=iters)
            if clipped.size < min_valid:
                continue
            bg, rms = _robust_median_mad(clipped)
            if not np.isfinite(bg) or not np.isfinite(rms) or rms <= 0:
                continue
            mesh_y.append(0.5 * (y0 + y1 - 1))
            mesh_x.append(0.5 * (x0 + x1 - 1))
            bg_values.append(bg)
            rms_values.append(rms)

    if len(bg_values) < 1:
        diagnostics["warning"] = "mesh_insufficient; fallback_global_median_mad"
        background_map = np.full_like(work, global_bg, dtype=np.float64)
        rms_map = np.full_like(work, global_rms, dtype=np.float64)
    else:
        background_map = _mesh_values_to_map(
            mesh_y, mesh_x, bg_values, work.shape, global_bg
        )
        rms_map = _mesh_values_to_map(
            mesh_y, mesh_x, rms_values, work.shape, global_rms
        )

    rms_map = _safe_rms_map(rms_map)
    diagnostics.update(
        {
            "background_rms_median": float(np.nanmedian(rms_map[np.isfinite(rms_map)])),
            "background_level_median": float(
                np.nanmedian(background_map[np.isfinite(background_map)])
            ),
            "mesh_count": int(len(bg_values)),
        }
    )
    return background_map, rms_map, diagnostics


def pixel_to_arcsec(x_pix, y_pix, extent, shape):
    return pixel_to_data_coord(x_pix, y_pix, extent, shape, origin="upper")


def _fit_failure_warning(source_file, quality_flag, detail=""):
    if isinstance(CONFIG, dict) and not CONFIG.get("gaussian_fit_verbose", False):
        counts = CONFIG.setdefault("_gaussian_warning_counts", {})
        key = str(quality_flag)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 3:
            return
    name = os.path.basename(source_file) if source_file else "radio image"
    suffix = f" / {detail}" if detail else ""
    warnings.warn(
        f"Gaussian fit skipped for {name}: reason={quality_flag}{suffix}", stacklevel=2
    )


def _gaussian_diagnostics_row(
    fit_result, cfg, freq=None, time_str=None, polarization=None, bg_diag=None
):
    bg_diag = bg_diag or {}
    if fit_result is None:
        fail = cfg.get("_last_gaussian_failure_diag", {})
        return {
            "source_file": fail.get("source_file", ""),
            "time": time_str,
            "freq": freq,
            "polarization": polarization,
            "reason": fail.get("reason", "fit_failed"),
            "finite_pixel_count": fail.get("finite_pixel_count", ""),
            "center_x_arcsec": "",
            "center_y_arcsec": "",
            "center_x_pixel": "",
            "center_y_pixel": "",
            "sigma_x_pixel": "",
            "sigma_y_pixel": "",
            "fwhm_x_pixel": "",
            "fwhm_y_pixel": "",
            "fwhm_width_arcsec": "",
            "fwhm_height_arcsec": "",
            "max_fwhm_arcsec": cfg.get("max_fwhm_arcsec", ""),
            "fwhm_valid": "",
            "center_peak_distance_arcsec": "",
            "theta_rad": "",
            "amplitude": "",
            "background_level": fail.get("background_level", ""),
            "noise_sigma": fail.get("noise_sigma", ""),
            "snr": "",
            "residual_rms": "",
            "mask_pixel_count": fail.get("mask_pixel_count", 0),
            "quality_flag": fail.get("quality_flag", fail.get("reason", "fit_failed")),
            "quality_flag_detail": fail.get("quality_flag_detail", ""),
            "background_strategy": fail.get(
                "background_strategy", cfg.get("radio_background_strategy", "")
            ),
            "background_use_for_mask": fail.get(
                "background_use_for_mask", cfg.get("background_use_for_mask", "")
            ),
            "background_use_for_display": cfg.get("background_use_for_display", False),
            "background_use_for_fit": fail.get(
                "background_use_for_fit", cfg.get("background_use_for_fit", "")
            ),
            "display_input_type": cfg.get("display_input_type", "raw"),
            "background_mesh_size": cfg.get("background_mesh_size", ""),
            "background_rms_median": fail.get("background_rms_median", ""),
            "background_level_median": fail.get("background_level_median", ""),
            "source_snr_peak": fail.get("source_snr_peak", ""),
            "source_snr_mean": fail.get("source_snr_mean", ""),
            "mask_method": fail.get("mask_method", ""),
            "fit_peak_fraction_threshold_used": fail.get(
                "fit_peak_fraction_threshold_used", ""
            ),
            "fit_peak_fraction_candidate_counts": fail.get(
                "fit_peak_fraction_candidate_counts", ""
            ),
            "background_enabled": bg_diag.get("background_enabled", False),
            "background_mode_requested": bg_diag.get("background_mode_requested", ""),
            "background_mode_used": bg_diag.get("background_mode_used", ""),
            "background_scale": bg_diag.get("background_scale", ""),
            "use_background_subtracted_for_gaussian_fit": cfg.get(
                "background_use_for_fit", False
            ),
            "fit_used_background_subtracted": cfg.get("background_use_for_fit", False),
            "fit_input_type": fail.get(
                "fit_input_type",
                (
                    "background_subtracted"
                    if cfg.get("background_use_for_fit", False)
                    else "raw"
                ),
            ),
            "fit_background_model": cfg.get("fit_background_model", "constant"),
            "gaussian_fit_method": fail.get("gaussian_fit_method", "skipped"),
            "roi_used": fail.get("roi_used", ""),
            "roi_shape": fail.get("roi_shape", ""),
            "fit_pixel_count_before_limit": fail.get(
                "fit_pixel_count_before_limit", ""
            ),
            "fit_pixel_count_after_limit": fail.get("fit_pixel_count_after_limit", ""),
            "maxfev": fail.get("maxfev", cfg.get("gaussian_fit_maxfev", "")),
            "initial_center_pixel": fail.get("initial_center_pixel", ""),
            "initial_sigma_x_pixel": fail.get("initial_sigma_x_pixel", ""),
            "initial_sigma_y_pixel": fail.get("initial_sigma_y_pixel", ""),
            "normalization_scale": fail.get("normalization_scale", ""),
            "peak": fail.get("peak", ""),
            "threshold": fail.get("threshold", ""),
        }
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
        "gaussian_fit_method": getattr(fit_result, "gaussian_fit_method", "curve_fit"),
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


def save_background_diagnostics_row(row, output_dir, cfg):
    diagnostics_dir = os.path.join(output_dir, _plot_output_subdir(cfg))
    os.makedirs(diagnostics_dir, exist_ok=True)
    csv_path = os.path.join(
        diagnostics_dir,
        cfg.get(
            "background_diagnostics_csv",
            "radio_background_subtraction_diagnostics.csv",
        ),
    )
    fieldnames = [
        "source_file",
        "time",
        "freq",
        "polarization",
        "background_enabled",
        "background_mode_requested",
        "background_mode_used",
        "background_file_count",
        "background_scale",
        "background_level",
        "noise_sigma_before",
        "noise_sigma_after",
        "raw_peak",
        "bgsub_peak",
        "raw_median",
        "bgsub_median",
        "display_background_subtracted_image",
        "use_background_subtracted_for_gaussian_fit",
        "warning",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _background_diagnostics_row(
    bg_diag, cfg, raw_data, bg_sub_data, freq=None, time_str=None, polarization=None
):
    return {
        "source_file": bg_diag.get("source_file", ""),
        "time": time_str,
        "freq": freq,
        "polarization": polarization,
        "background_enabled": bg_diag.get("background_enabled", False),
        "background_mode_requested": bg_diag.get("background_mode_requested", ""),
        "background_mode_used": bg_diag.get("background_mode_used", ""),
        "background_file_count": bg_diag.get("background_file_count", 0),
        "background_scale": bg_diag.get("background_scale", 1.0),
        "background_level": bg_diag.get("background_level", ""),
        "noise_sigma_before": bg_diag.get("noise_sigma_before", ""),
        "noise_sigma_after": bg_diag.get("noise_sigma_after", ""),
        "raw_peak": float(np.nanmax(raw_data)) if np.any(np.isfinite(raw_data)) else "",
        "bgsub_peak": (
            float(np.nanmax(bg_sub_data)) if np.any(np.isfinite(bg_sub_data)) else ""
        ),
        "raw_median": (
            float(np.nanmedian(raw_data)) if np.any(np.isfinite(raw_data)) else ""
        ),
        "bgsub_median": (
            float(np.nanmedian(bg_sub_data)) if np.any(np.isfinite(bg_sub_data)) else ""
        ),
        "display_background_subtracted_image": background_enabled_for_display(cfg),
        "use_background_subtracted_for_gaussian_fit": background_enabled_for_fit(cfg),
        "warning": bg_diag.get("warning", ""),
    }


def _save_background_products(
    bg_sub_data,
    background_map,
    extent,
    output_dir,
    base_name,
    cfg,
    image_origin=None,
    colorbar_text=None,
    *,
    observation_time=None,
    frequency_mhz=None,
    polarization=None,
    sequence=1,
    generated_at=None,
):
    should_save = not cfg.get("radio_background_force_off", False) and (
        cfg.get("save_background_subtracted_image", False)
        or cfg.get("save_estimated_background_map", False)
        or (
            background_workflow_enabled(cfg)
            and cfg.get("save_background_diagnostics", True)
        )
    )
    if not should_save:
        return
    maps_dir = os.path.join(output_dir, "background_subtracted_maps")
    os.makedirs(maps_dir, exist_ok=True)
    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)
    if cfg.get("save_background_subtracted_image", False):
        fig, ax = plt.subplots(figsize=cfg["fig_size"])
        im = ax.imshow(
            bg_sub_data,
            **get_imshow_kwargs(bg_sub_data, extent, cfg, origin=image_origin),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_text:
            cbar.set_label(colorbar_text)
        ax.set_title(f"{base_name} background subtracted")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                maps_dir,
                build_scientific_image_filename(
                    sequence=2 * sequence - 1,
                    start_time=observation_time,
                    instrument="radio",
                    channel=(
                        f"{float(frequency_mhz):g}mhz"
                        if frequency_mhz is not None
                        else None
                    ),
                    polarization=polarization,
                    product="background_subtracted_map",
                    generated_at=generated_at,
                ),
            ),
            dpi=cfg["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)
    if cfg.get("save_estimated_background_map", False) and background_map is not None:
        np.save(
            os.path.join(maps_dir, f"{base_name}_background_map.npy"), background_map
        )
        fig, ax = plt.subplots(figsize=cfg["fig_size"])
        im = ax.imshow(
            background_map,
            **get_imshow_kwargs(background_map, extent, cfg, origin=image_origin),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_text:
            cbar.set_label(colorbar_text)
        ax.set_title(f"{base_name} background map")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                maps_dir,
                build_scientific_image_filename(
                    sequence=2 * sequence,
                    start_time=observation_time,
                    instrument="radio",
                    channel=(
                        f"{float(frequency_mhz):g}mhz"
                        if frequency_mhz is not None
                        else None
                    ),
                    polarization=polarization,
                    product="estimated_background_map",
                    generated_at=generated_at,
                ),
            ),
            dpi=cfg["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)


def _attach_raw_peak_center(fit_result, img_data, extent, cfg=None, image_origin=None):
    if fit_result is None or not np.any(np.isfinite(img_data)):
        return
    image_origin = image_origin or getattr(
        fit_result,
        "image_origin",
        (cfg or {}).get("_current_radio_image_origin", "upper"),
    )
    finite_img = np.where(np.isfinite(img_data), img_data, -np.inf)
    raw_y, raw_x = _unravel_2d_index(int(np.argmax(finite_img)), img_data.shape)
    fit_result.raw_center_arcsec = pixel_to_data_coord(
        float(raw_x), float(raw_y), extent, img_data.shape, origin=image_origin
    )
    fit_result.raw_center_pixel = (float(raw_x), float(raw_y))
    fit_result.raw_peak_x_pixel = float(raw_x)
    fit_result.raw_peak_y_pixel = float(raw_y)
    fit_result.raw_peak_x_arcsec = fit_result.raw_center_arcsec[0]
    fit_result.raw_peak_y_arcsec = fit_result.raw_center_arcsec[1]
    fit_result.raw_peak_roundtrip_error_pixel = coordinate_roundtrip_error_pixel(
        raw_x, raw_y, extent, img_data.shape, origin=image_origin
    )
    if cfg is not None:
        _update_gaussian_quality(fit_result, extent, img_data.shape, cfg)


def _save_gaussian_residual_panel(
    radio_fit_data,
    fit_result,
    extent,
    output_path,
    cfg,
    title_prefix="Gaussian residual",
    colorbar_text=None,
    residual_output_path=None,
):
    if fit_result is None:
        return None
    residual = radio_fit_data - fit_result.model
    fig, ax = plt.subplots(figsize=cfg["fig_size"])
    finite = residual[np.isfinite(residual)]
    vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    image_origin = getattr(
        fit_result, "image_origin", cfg.get("_current_radio_image_origin", "upper")
    )
    im_kwargs = get_imshow_kwargs(
        residual, extent, cfg, cmap="RdBu_r", origin=image_origin
    )
    im_kwargs["vmin"] = -vmax
    im_kwargs["vmax"] = vmax
    im = ax.imshow(residual, **im_kwargs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_text:
        cbar.set_label(colorbar_text)
    ax.set_title(
        f"{title_prefix}  RMS={fit_result.residual_rms}  SNR={fit_result.snr}  {fit_result.quality_flag}",
        fontsize=cfg["title_fontsize"] - 4,
    )
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    residual_path = residual_output_path or output_path.replace(
        ".png", "_gaussian_residual.png"
    )
    plt.tight_layout()
    plt.savefig(residual_path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    return residual_path


def _global_range_one(fp):
    """
    Read a single file, return (nanmin, nanmax).
    For parallel calls, exceptions are caught independently.
    """
    try:
        # 处理元组情况：如果是元组，取第一个文件（RR文件）
        if isinstance(fp, tuple):
            file_path = fp[0]
        else:
            file_path = fp

        data, _ = read_fits(file_path)
        return float(np.nanmin(data)), float(np.nanmax(data))
    except Exception as e:
        warnings.warn(f"Skipping file {fp}: {e}", stacklevel=2)
        return None


def compute_global_range(
    file_list: list, fixed_vmin=None, fixed_vmax=None, max_workers: int = 4
):
    """
    Traverse all files in parallel to compute global [vmin, vmax].
    When fixed_vmin / fixed_vmax are not None, return directly and skip statistics.

    【Optimization】Use ThreadPoolExecutor:
      - Reading FITS + nanmin/nanmax is a mix of I/O and NumPy computation,
        ThreadPool can parallelize I/O, NumPy releases GIL so computation can also be parallel,
        and it avoids process fork overhead, memory usage is lower.
    """
    if fixed_vmin is not None and fixed_vmax is not None:
        return fixed_vmin, fixed_vmax

    mins, maxs = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_global_range_one, fp): fp for fp in file_list}
        with tqdm(
            total=len(file_list), desc="Computing global color range", unit="files"
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    mins.append(result[0])
                    maxs.append(result[1])
                pbar.update(1)

    gmin = fixed_vmin if fixed_vmin is not None else float(np.nanmin(mins))
    gmax = fixed_vmax if fixed_vmax is not None else float(np.nanmax(maxs))
    print(f"Global color range: [{gmin:.3e}, {gmax:.3e}]")
    return gmin, gmax


def get_time_from_header(header):
    """Extract observation time from FITS header."""
    date_obs = header.get("DATE-OBS", "Unknown")
    if "TIME-OBS" in header:
        return f"{date_obs} {header['TIME-OBS']}"
    return date_obs


def get_freq_from_header(header):
    """Extract frequency from FITS header."""
    return header.get("FREQ", header.get("FREQUENCY", None))


def get_polar_from_header(header):
    """Extract polarization information from FITS header."""
    return str(header.get("POLAR", "StokesI")).strip()


def _sorted_fits_for_band(
    band_dir: str,
    start_idx: int,
    end_idx,
    *,
    study_mode: str | None = None,
) -> list:
    """Get sorted list of FITS files in the specified band directory"""
    if not os.path.isdir(band_dir):
        raise ValueError(f"波段目录不存在：{band_dir}")

    mode = _require_study_mode(study_mode)
    frozen = _frozen_files_for_band(Path(band_dir), required=mode == "confirmatory")
    if frozen is not None:
        return [str(path) for path in frozen]

    all_files = sorted(
        os.path.join(band_dir, f)
        for f in os.listdir(band_dir)
        if f.lower().endswith(".fits")
    )
    if not all_files:
        raise ValueError(f"波段目录 {band_dir} 中未找到 FITS 文件")

    total = len(all_files)
    end = total if end_idx is None else min(end_idx, total)
    selected = all_files[start_idx:end]
    if not selected:
        raise ValueError(f"索引范围 [{start_idx}, {end}) 内没有文件")
    return selected


def _require_study_mode(value: object) -> str:
    mode = str(value or "").strip().casefold()
    if mode not in {"exploratory", "confirmatory"}:
        raise ValueError(
            "radio study_mode must be explicit: exploratory or confirmatory"
        )
    return mode


def _parse_utc_z(value: object, *, label: str) -> datetime.datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value)
        is None
    ):
        raise ValueError(f"Frozen collection {label} must be UTC with a Z suffix")
    try:
        parsed = datetime.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"Frozen collection {label} is invalid") from exc
    if parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"Frozen collection {label} must be UTC")
    return parsed


def _utc_z_millis(value: datetime.datetime) -> str:
    return (
        value.astimezone(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _expected_frozen_record_id(observed: datetime.datetime, relative_path: str) -> str:
    identity = f"{_utc_z_millis(observed)}\0{relative_path}"
    return "radio-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _frozen_files_for_band(
    band_dir: Path, *, required: bool = False
) -> list[Path] | None:
    """Return and verify an explicit collection, bypassing positional slices."""

    manifest = next(
        (
            parent / ".frozen-collection-v1.json"
            for parent in (band_dir, *band_dir.parents)
            if (parent / ".frozen-collection-v1.json").is_file()
        ),
        None,
    )
    if manifest is None:
        if required:
            raise FileNotFoundError(
                f"Confirmatory radio selection requires .frozen-collection-v1.json: "
                f"{band_dir}"
            )
        return None
    root = manifest.parent.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen collection must be a JSON object")
    if payload.get("schema") != "solar-radio-frozen-collection-v1":
        raise ValueError(f"Unsupported frozen collection: {manifest}")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Frozen collection selection must be an object")
    start = _parse_utc_z(selection.get("start_utc"), label="selection.start_utc")
    end = _parse_utc_z(selection.get("end_utc"), label="selection.end_utc")
    if end <= start:
        raise ValueError("Frozen collection selection must satisfy end_utc > start_utc")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen collection records must be a non-empty list")
    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(records)
    ):
        raise ValueError("Frozen collection record_count mismatch")
    resolved_band = band_dir.resolve()
    selected: list[tuple[datetime.datetime, str, Path]] = []
    record_ids: set[str] = set()
    paths: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Frozen collection record must be an object")
        record_id = item.get("record_id")
        if not isinstance(record_id, str) or not record_id.startswith("radio-"):
            raise ValueError("Frozen collection record_id is invalid")
        if record_id in record_ids:
            raise ValueError(f"Frozen collection duplicate record_id: {record_id}")
        record_ids.add(record_id)
        observed = _parse_utc_z(item.get("observed_utc"), label="record.observed_utc")
        if not start <= observed < end:
            raise ValueError(f"Frozen record outside selection interval: {record_id}")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Frozen record path is invalid: {record_id}")
        if relative_path in paths:
            raise ValueError(f"Frozen collection duplicate path: {relative_path}")
        paths.add(relative_path)
        expected_record_id = _expected_frozen_record_id(observed, relative_path)
        if record_id != expected_record_id:
            raise ValueError(f"Frozen record_id is not bound to UTC/path: {record_id}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Frozen collection path escaped its root") from exc
        size = item.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Frozen record bytes is invalid: {record_id}")
        digest_value = item.get("sha256")
        if (
            not isinstance(digest_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
        ):
            raise ValueError(f"Frozen record SHA is invalid: {record_id}")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Frozen collection file missing/size mismatch: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != digest_value:
            raise ValueError(f"Frozen collection SHA mismatch: {path}")
        if path.parent == resolved_band:
            selected.append((observed, record_id, path))
    if not selected:
        raise ValueError(f"Frozen collection has no records for {band_dir}")
    return [path for _observed, _record_id, path in sorted(selected)]


def _raw_quality_filter_enabled(cfg: dict) -> bool:
    return bool(cfg.get("enable_raw_quality_filter", False))


_RAW_QUALITY_BAD_REASONS_KEY = "_raw_quality_bad_file_reasons"
_RAW_QUALITY_FILE_FLAGS_KEY = "_raw_quality_file_quality_flags"


def _raw_quality_path_key(path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _iter_raw_quality_item_paths(item) -> list[str]:
    if isinstance(item, (tuple, list)):
        paths: list[str] = []
        for part in item:
            paths.extend(_iter_raw_quality_item_paths(part))
        return paths
    text = os.fspath(item)
    if "|" in text:
        return [part for part in text.split("|") if part]
    return [text]


def _remember_raw_quality_rows(cfg: dict, rows: list) -> None:
    bad_reasons = dict(cfg.get(_RAW_QUALITY_BAD_REASONS_KEY, {}) or {})
    file_flags = dict(cfg.get(_RAW_QUALITY_FILE_FLAGS_KEY, {}) or {})
    for row in rows:
        raw_path = str(row.source_file)
        path_key = _raw_quality_path_key(raw_path)
        file_flags[path_key] = str(row.quality_flag)
        if row.quality_flag == "bad":
            bad_reasons[path_key] = str(row.reason)
        else:
            bad_reasons.pop(path_key, None)
    cfg[_RAW_QUALITY_BAD_REASONS_KEY] = bad_reasons
    cfg[_RAW_QUALITY_FILE_FLAGS_KEY] = file_flags


def _raw_quality_bad_reasons_for_item(item, cfg: dict) -> list[str]:
    bad_reasons = cfg.get(_RAW_QUALITY_BAD_REASONS_KEY, {}) or {}
    reasons: list[str] = []
    seen = set()
    for path in _iter_raw_quality_item_paths(item):
        for key in (path, _raw_quality_path_key(path)):
            reason = bad_reasons.get(key)
            if reason and key not in seen:
                reasons.append(str(reason))
                seen.add(key)
    return reasons


def _raw_quality_item_is_bad(item, cfg: dict) -> bool:
    return bool(_raw_quality_bad_reasons_for_item(item, cfg))


def _raw_quality_bad_frame_output_dir(output_dir: str, cfg: dict, *parts) -> Path:
    bad_subdir = str(
        cfg.get("raw_quality_bad_frame_output_subdir", "raw_quality_bad_frames")
        or "raw_quality_bad_frames"
    ).strip()
    if not bad_subdir:
        bad_subdir = "raw_quality_bad_frames"
    return Path(output_dir) / _plot_output_subdir(cfg) / bad_subdir / Path(*parts)


def _filter_bad_radio_files(
    files: list, freq, polarization: str, cfg: dict, *, drop_bad: bool = False
) -> list:
    """Classify raw FITS quality, optionally dropping bad files for statistics only."""
    if not _raw_quality_filter_enabled(cfg):
        return files
    if not files:
        return files

    from solar_toolkit.radio.raw_quality import filter_bad_radio_fits_files

    result = filter_bad_radio_fits_files(
        files,
        frequency_mhz=float(freq),
        polarization=str(polarization),
    )
    _remember_raw_quality_rows(cfg, result.file_rows)
    rejected = result.rejected_rows
    if rejected:
        print(
            f"  Raw-quality filter {freq}MHz/{polarization}: "
            f"flagged {len(rejected)}/{len(files)} bad"
        )
        for row in rejected[:5]:
            print(f"    reject {os.path.basename(row.source_file)}: {row.reason}")
        if len(rejected) > 5:
            print(f"    ... {len(rejected) - 5} more rejected files")
    if drop_bad:
        return result.accepted_files
    return list(files)


# ──────────────────────────────────────────────────────────────
# 时间解析工具
# ──────────────────────────────────────────────────────────────


class TimeParser:
    """时间解析器，支持多种日期格式"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.date_format = cfg.get("date_format", "auto")
        self.fallback = cfg.get("time_parsing_fallback", True)

    def parse_date_part(self, date_str):
        """解析日期字符串，返回(年份, 天数)"""
        if len(date_str) == 6:
            # 格式: YYDDD (6位)
            year = int(date_str[0:2])
            # 假设20xx年
            full_year = 2000 + year if year < 100 else year
            day_of_year = int(date_str[2:])
            return full_year, day_of_year
        elif len(date_str) == 7:
            # 格式: YYYYDDD (7位)
            year = int(date_str[0:4])
            day_of_year = int(date_str[4:])
            return year, day_of_year
        elif len(date_str) == 8:
            # 格式: YYYYDDDD (8位，不常见）
            year = int(date_str[0:4])
            day_of_year = int(date_str[4:])
            return year, day_of_year
        else:
            raise ValueError(f"不支持的日期格式长度: {len(date_str)}位")

    def parse_time_from_filename(self, filename):
        """从文件名解析时间信息（精确到毫秒），支持多种格式。

        支持的格式:
        1. 6位日期+毫秒: 149MHz_202553_071600_353.fits
        2. 7位日期+毫秒: 149MHz_2025124_043739_681.fits
        3. 6位日期无毫秒: 149MHz_202553_071600.fits
        4. 7位日期无毫秒: 149MHz_2025124_043739.fits

        返回: (date_key, total_ms) 或 None
          - date_key : 用于跨天比较的日期键
          - total_ms : 当天从0点起的毫秒数
        """
        import re

        # 从配置获取正则表达式模式
        patterns = self.cfg.get("filename_patterns", {})
        pattern_with_ms = patterns.get("with_ms", r"_(\d{6,8})_(\d{6})_(\d{1,3})")
        pattern_without_ms = patterns.get("without_ms", r"_(\d{6,8})_(\d{6})")

        # 尝试带毫秒的模式
        match = re.search(pattern_with_ms, filename)
        if match:
            date_part = match.group(1)  # 如 "202553" 或 "2025124"
            time_part = match.group(2)  # 如 "071600"
            ms_str = match.group(3)  # 如 "353"

            # 解析时间部分
            hh = int(time_part[0:2])
            mm = int(time_part[2:4])
            ss = int(time_part[4:6])

            # Parse suffix as integer milliseconds: _13 means 13 ms, not 130 ms.
            ms = int(ms_str[:3])

            total_ms = (hh * 3600 + mm * 60 + ss) * 1000 + ms

            # 根据配置的日期格式生成date_key
            if self.date_format == "auto":
                # 自动根据长度选择
                date_key = date_part  # 使用原始字符串作为键
            else:
                # 使用指定格式解析并生成标准键
                year, day_of_year = self.parse_date_part(date_part)
                # 生成标准格式的日期键: YYYY-DDD
                date_key = f"{year:04d}-{day_of_year:03d}"

            return (date_key, total_ms)

        # 尝试不带毫秒的模式
        match = re.search(pattern_without_ms, filename)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            hh = int(time_part[0:2])
            mm = int(time_part[2:4])
            ss = int(time_part[4:6])
            total_ms = (hh * 3600 + mm * 60 + ss) * 1000

            if self.date_format == "auto":
                date_key = date_part
            else:
                year, day_of_year = self.parse_date_part(date_part)
                date_key = f"{year:04d}-{day_of_year:03d}"

            return (date_key, total_ms)

        # 容错模式：尝试更宽松的匹配
        if self.fallback:
            # 尝试匹配任何看起来像时间格式的部分
            fallback_pattern = r"_(\d{6,8})_(\d{6})"
            match = re.search(fallback_pattern, filename)
            if match:
                date_part = match.group(1)
                time_part = match.group(2)

                # 尝试解析时间
                try:
                    hh = int(time_part[0:2])
                    mm = int(time_part[2:4])
                    ss = int(time_part[4:6])
                    total_ms = (hh * 3600 + mm * 60 + ss) * 1000

                    if self.date_format == "auto":
                        date_key = date_part
                    else:
                        year, day_of_year = self.parse_date_part(date_part)
                        date_key = f"{year:04d}-{day_of_year:03d}"

                    return (date_key, total_ms)
                except (ValueError, IndexError):
                    pass

        return None


def _parse_time_from_filename(filename):
    """从文件名解析时间信息（精确到毫秒），用于时间对齐匹配。

    这是向后兼容的包装函数，使用新的TimeParser类。

    文件名格式:
      - 新格式6位日期: 149MHz_202553_071600_353.fits
      - 原格式7位日期: 149MHz_2025124_043739_681.fits

    返回: (date_str, total_ms) 或 None
      - date_str  : 日期字符串，用于跨天判断
      - total_ms  : 当天从0点起的毫秒数，用于数值比较
    """
    # 创建解析器实例，使用默认配置（日期格式自动检测）
    parser = TimeParser({"date_format": "auto", "time_parsing_fallback": True})

    result = parser.parse_time_from_filename(filename)
    if result:
        date_key, total_ms = result
        # 为了向后兼容，返回原始格式
        return (date_key, total_ms)
    return None


def create_time_parser(cfg=None):
    """创建时间解析器实例

    参数:
    ----------
    cfg : dict, 可选
        配置字典，如果为None则使用全局CONFIG

    返回:
    -------
    TimeParser 实例
    """
    if cfg is None:
        cfg = CONFIG
    return TimeParser(cfg)


def _check_time_alignment(
    rr_header, ll_header, rr_path, ll_path, tolerance_seconds=1.0
):
    """检查RR和LL文件的时间对齐情况。

    优先从文件名解析精确时间戳（毫秒级），
    次之从FITS header解析，最后退回字符串比较。
    """
    tolerance_ms = tolerance_seconds * 1000

    # ── 1. 优先用文件名时间戳（更精确、更可靠） ──────────────────
    rr_parsed = _parse_time_from_filename(os.path.basename(rr_path))
    ll_parsed = _parse_time_from_filename(os.path.basename(ll_path))

    if rr_parsed is not None and ll_parsed is not None:
        rr_date, rr_ms = rr_parsed
        ll_date, ll_ms = ll_parsed
        if rr_date != ll_date:
            print(
                f"警告: RR/LL文件日期不一致 (RR={rr_date}, LL={ll_date})\n"
                f"  RR: {os.path.basename(rr_path)}\n"
                f"  LL: {os.path.basename(ll_path)}"
            )
            return False
        diff_ms = abs(rr_ms - ll_ms)
        if diff_ms > tolerance_ms:
            print(
                f"警告: RR和LL文件时间差 {diff_ms:.1f} ms 超过容差 {tolerance_ms:.1f} ms\n"
                f"  RR: {os.path.basename(rr_path)}\n"
                f"  LL: {os.path.basename(ll_path)}"
            )
            return False
        return True

    # ── 2. 降级：从FITS header解析 ───────────────────────────────
    from astropy.time import Time

    rr_time_str = get_time_from_header(rr_header)
    ll_time_str = get_time_from_header(ll_header)

    if rr_time_str == "Unknown" or ll_time_str == "Unknown":
        # 无法从任何来源获取时间，假设对齐
        return True

    try:
        rr_time = Time(rr_time_str, format="isot", scale="utc")
        ll_time = Time(ll_time_str, format="isot", scale="utc")
        time_diff_s = abs((rr_time - ll_time).sec)
        if time_diff_s > tolerance_seconds:
            print(
                f"警告: RR和LL文件时间差 {time_diff_s:.3f} 秒超过容差 {tolerance_seconds} 秒\n"
                f"  RR: {rr_time_str}\n"
                f"  LL: {ll_time_str}"
            )
            return False
        return True
    except Exception as e:
        print(f"时间解析失败: {e}，使用字符串比较")
        return rr_time_str == ll_time_str


def _match_rr_ll_by_time(
    rr_files: list, ll_files: list, tolerance_ms: float = 10.0, cfg=None
):
    """根据文件名时间戳将RR与LL文件逐一配对（毫秒级精度）。

    参数:
    ----------
    rr_files      : RR文件路径列表（已排序）
    ll_files      : LL文件路径列表（已排序）
    tolerance_ms  : 时间匹配容差（毫秒），默认10ms
    cfg          : 配置字典，用于时间解析

    返回:
    -------
    matched_pairs : list of (rr_path, ll_path)
    """
    # 创建时间解析器
    parser = create_time_parser(cfg)

    # 构建LL时间索引: {(date_key, total_ms): ll_path}
    ll_index: dict = {}
    ll_no_parse: list = []
    for ll_path in ll_files:
        parsed = parser.parse_time_from_filename(os.path.basename(ll_path))
        if parsed is None:
            ll_no_parse.append(ll_path)
        else:
            key = parsed  # (date_key, total_ms)
            if key in ll_index:
                # 重复时间戳：保留先出现的（有序列表中索引更小的）
                pass
            else:
                ll_index[key] = ll_path

    if ll_no_parse:
        warnings.warn(
            f"有 {len(ll_no_parse)} 个LL文件无法从文件名解析时间，将被跳过。",
            stacklevel=2,
        )

    matched_pairs: list = []
    unmatched_rr: list = []

    # 将LL索引按日期分组，加速搜索
    from collections import defaultdict

    ll_by_date: dict = defaultdict(list)  # {date_key: [(total_ms, ll_path), ...]}
    for (date_key, total_ms), ll_path in ll_index.items():
        ll_by_date[date_key].append((total_ms, ll_path))
    # 每个日期内按ms排序
    for date_key in ll_by_date:
        ll_by_date[date_key].sort(key=lambda x: x[0])

    for rr_path in rr_files:
        parsed = parser.parse_time_from_filename(os.path.basename(rr_path))
        if parsed is None:
            unmatched_rr.append(rr_path)
            warnings.warn(
                f"RR文件 {os.path.basename(rr_path)} 无法解析时间，跳过。", stacklevel=2
            )
            continue

        rr_date_key, rr_ms = parsed

        # ① 精确匹配
        if (rr_date_key, rr_ms) in ll_index:
            matched_pairs.append((rr_path, ll_index[(rr_date_key, rr_ms)]))
            continue

        # ② 容差范围内最近邻匹配
        candidates = ll_by_date.get(rr_date_key, [])
        best_ll_path = None
        best_diff = float("inf")
        for ll_ms, ll_path in candidates:
            diff = abs(rr_ms - ll_ms)
            if diff < best_diff:
                best_diff = diff
                best_ll_path = ll_path

        if best_ll_path is not None and best_diff <= tolerance_ms:
            matched_pairs.append((rr_path, best_ll_path))
        else:
            unmatched_rr.append(rr_path)
            if best_diff != float("inf"):
                warnings.warn(
                    f"RR文件 {os.path.basename(rr_path)} 找不到时间匹配的LL文件 "
                    f"(最近差值={best_diff:.1f}ms > 容差={tolerance_ms:.1f}ms)，跳过。",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"RR文件 {os.path.basename(rr_path)} 在LL目录中找不到同日期文件，跳过。",
                    stacklevel=2,
                )

    if unmatched_rr:
        print(
            f"  时间匹配结果: 成功 {len(matched_pairs)} 对，"
            f"RR未匹配 {len(unmatched_rr)} 个。"
        )
    else:
        print(f"  时间匹配结果: 全部 {len(matched_pairs)} 对成功匹配。")

    return matched_pairs


def _radio_item_time_key(item, parser):
    path = item[0] if isinstance(item, tuple) else item
    return parser.parse_time_from_filename(os.path.basename(path))


def _build_slots_by_common_time(per_band: list, cfg: dict) -> list | None:
    """Build multi-band slots by matching nearest parsed times across all bands."""
    parser = create_time_parser(cfg)
    per_band_entries = []
    for band_items in per_band:
        entries = []
        for item in band_items:
            key = _radio_item_time_key(item, parser)
            if key is None:
                return None
            entries.append((key, item))
        per_band_entries.append(entries)

    if not per_band_entries:
        return []
    if any(not entries for entries in per_band_entries):
        return []

    tolerance_ms = float(cfg.get("multi_band_time_tolerance_seconds", 0.1)) * 1000.0
    reference_index = min(
        range(len(per_band_entries)), key=lambda index: len(per_band_entries[index])
    )
    reference_entries = sorted(
        per_band_entries[reference_index], key=lambda entry: (entry[0][0], entry[0][1])
    )
    used_by_band = [set() for _entries in per_band_entries]
    slots = []
    for ref_key, _ref_item in reference_entries:
        slot = []
        matched_indices = []
        for band_index, entries in enumerate(per_band_entries):
            match_index = _nearest_time_entry_index(
                entries, ref_key, used_by_band[band_index], tolerance_ms
            )
            if match_index is None:
                slot = []
                matched_indices = []
                break
            matched_indices.append(match_index)
            slot.append(entries[match_index][1])
        if slot:
            slot_times = [
                per_band_entries[band_index][match_index][0][1]
                for band_index, match_index in enumerate(matched_indices)
            ]
            if max(slot_times) - min(slot_times) > tolerance_ms:
                continue
            for band_index, match_index in enumerate(matched_indices):
                used_by_band[band_index].add(match_index)
            slots.append(slot)

    used_count = sum(len(used) for used in used_by_band)
    total_count = sum(len(entries) for entries in per_band_entries)
    dropped = total_count - used_count
    if dropped:
        print(
            "Dropped "
            f"{dropped} band-time entries because not every band has a usable match."
        )
    return slots


def _nearest_time_entry_index(entries: list, ref_key, used_indices: set, tolerance_ms):
    ref_date, ref_ms = ref_key
    best_index = None
    best_diff = float("inf")
    for index, (key, _item) in enumerate(entries):
        if index in used_indices:
            continue
        date_key, total_ms = key
        if date_key != ref_date:
            continue
        diff = abs(float(total_ms) - float(ref_ms))
        if diff < best_diff:
            best_index = index
            best_diff = diff
    if best_index is not None and best_diff <= tolerance_ms:
        return best_index
    return None


def _build_slots_by_position(per_band: list) -> list:
    lengths = [len(f) for f in per_band]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(
            f"Warning: number of files per band inconsistent, using the minimum count {min_len}"
        )
        per_band = [f[:min_len] for f in per_band]
    return [list(band_files) for band_files in zip(*per_band, strict=False)]


def _build_multi_band_slots(cfg: dict) -> list:
    """
    Build multi-band synthesis time slots (each slot contains files of each band at the same time).

    【Optimization】Inner time slot construction changed to zip(*per_band),
    eliminating nested for loops, slightly faster and more concise code.
    """
    root = cfg["multi_band_root"]
    freqs = cfg["multi_band_freqs"]
    pattern = cfg["band_dir_pattern"]
    polarization = cfg["polarization"]
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", None)

    # 检查是否启用左右旋数据加和
    combine_polarizations = cfg.get("combine_polarizations", False)
    time_tolerance = cfg.get("time_tolerance_seconds", 1.0)

    per_band = []
    for freq in freqs:
        if combine_polarizations and polarization == "RR+LL":
            # 左右旋数据加和模式：需要读取RR和LL两个目录的文件
            rr_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"])
            )
            ll_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["ll_dir_suffix"])
            )

            rr_files = _sorted_fits_for_band(
                rr_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            ll_files = _sorted_fits_for_band(
                ll_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            rr_files = _filter_bad_radio_files(
                rr_files, freq, cfg["rr_dir_suffix"], cfg
            )
            ll_files = _filter_bad_radio_files(
                ll_files, freq, cfg["ll_dir_suffix"], cfg
            )

            # ── 基于文件名时间戳做精确匹配（毫秒级） ──────────────
            tolerance_ms = time_tolerance * 1000  # 秒 → 毫秒
            print(
                f"  频率 {freq}MHz: RR={len(rr_files)} 文件, "
                f"LL={len(ll_files)} 文件，开始时间匹配 (容差={tolerance_ms:.1f}ms)..."
            )
            # 传递cfg给匹配函数，以便使用正确的时间解析配置
            combined_files = _match_rr_ll_by_time(rr_files, ll_files, tolerance_ms, cfg)

            if not combined_files:
                raise ValueError(
                    f"频率 {freq}MHz: RR和LL文件时间匹配失败，没有找到任何匹配对"
                )

            per_band.append(combined_files)
        else:
            # 普通模式：只读取指定偏振的文件
            band_dir = os.path.join(root, pattern.format(freq=freq, polar=polarization))
            files = _sorted_fits_for_band(
                band_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            files = _filter_bad_radio_files(files, freq, polarization, cfg)
            per_band.append(files)

    # ★ 优化：zip 直接转置二维列表，替代双层 for 循环
    slots = _build_slots_by_common_time(per_band, cfg)
    if slots is None:
        if _require_study_mode(cfg.get("study_mode")) == "confirmatory":
            raise ValueError(
                "Confirmatory multi-band selection requires parseable UTC times; "
                "positional fallback is forbidden"
            )
        print(
            "Warning: could not parse all radio times; "
            "falling back to positional slots."
        )
        slots = _build_slots_by_position(per_band)

    print(f"Built {len(slots)} time slots, each slot contains {len(freqs)} bands")
    print(f"Polarization: {polarization}")
    if combine_polarizations and polarization == "RR+LL":
        print("Mode: RR + LL data combination")
        if cfg.get("weighted_average", False):
            print(
                f"Weighted average: RR weight={cfg['rr_weight']}, LL weight={cfg['ll_weight']}"
            )
        else:
            print("Simple summation")
    return slots


def _workspace_source_map_selection(cfg: dict) -> dict | None:
    """Decode an explicit web-workspace source-map selection, if supplied."""

    raw_value = cfg.get("selected_source_map_json") or cfg.get("source_map_selection")
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        try:
            selection = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "selected_source_map_json must be valid JSON. Preview again."
            ) from exc
    else:
        selection = raw_value
    if not isinstance(selection, dict):
        raise ValueError("selected_source_map_json must be a JSON object.")
    schema_version = selection.get("schema_version", 1)
    if schema_version not in (1, "1"):
        raise ValueError("Unsupported source-map selection schema version.")
    mode = str(selection.get("mode") or "").strip()
    if mode not in {"single_band", "multi_band"}:
        raise ValueError("Source-map selection must declare single_band or multi_band.")
    items = selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Source-map selection must include at least one item.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Source-map selection items must be JSON objects.")
    return {
        "schema_version": 1,
        "mode": mode,
        "candidate_ids": list(selection.get("candidate_ids") or []),
        "items": items,
    }


def _normalized_selection_path(value) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _source_map_slot_file_paths(slot: list) -> list[str]:
    paths: list[str] = []
    for item in slot:
        if isinstance(item, (tuple, list)):
            paths.extend(os.fspath(path) for path in item)
        else:
            paths.append(os.fspath(item))
    return paths


def _candidate_slot_index(item: dict) -> int:
    raw_index = item.get("slot_index")
    if raw_index is None:
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id.startswith("slot-"):
            raw_index = candidate_id.partition("-")[2]
    try:
        return int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Multi-band source-map selection is missing a valid slot_index."
        ) from exc


def _selected_workspace_slot_items(
    slots: list, selection: dict
) -> list[tuple[int, list]]:
    if selection["mode"] != "multi_band":
        raise ValueError("Multi-band source-map run requires a multi-band selection.")
    selected: list[tuple[int, list]] = []
    seen_indices: set[int] = set()
    for item in selection["items"]:
        slot_index = _candidate_slot_index(item)
        if slot_index < 0 or slot_index >= len(slots):
            raise ValueError(
                "Selected source-map slot is no longer available. Preview again."
            )
        if slot_index in seen_indices:
            continue
        expected_paths = item.get("paths")
        if not isinstance(expected_paths, list) or not expected_paths:
            raise ValueError(
                "Selected source-map slot is missing its source paths. Preview again."
            )
        actual_paths = _source_map_slot_file_paths(slots[slot_index])
        if tuple(map(_normalized_selection_path, expected_paths)) != tuple(
            map(_normalized_selection_path, actual_paths)
        ):
            raise ValueError(
                "Selected source-map slot no longer matches the current input "
                "folder. Preview again."
            )
        selected.append((slot_index, slots[slot_index]))
        seen_indices.add(slot_index)
    if not selected:
        raise ValueError("No source-map slots were selected.")
    return selected


def _selected_workspace_files(selection: dict, cfg: dict) -> list[str]:
    if selection["mode"] != "single_band":
        raise ValueError("Single-band source-map run requires a single-band selection.")
    selected: list[str] = []
    combine = (
        bool(cfg.get("combine_polarizations")) and cfg.get("polarization") == "RR+LL"
    )
    for item in selection["items"]:
        paths = item.get("paths")
        run_path = item.get("run_path")
        if not run_path and isinstance(paths, list) and paths:
            run_path = paths[0]
        if not isinstance(run_path, str) or not run_path.strip():
            raise ValueError(
                "Selected source-map file is missing its run path. Preview again."
            )
        if not os.path.isfile(run_path):
            raise FileNotFoundError(
                f"Selected source-map FITS file does not exist: {run_path}"
            )
        if combine:
            if not isinstance(paths, list) or len(paths) < 2:
                raise ValueError(
                    "RR+LL source-map run requires a matched RR/LL preview selection."
                )
            missing = [path for path in paths if not os.path.isfile(os.fspath(path))]
            if missing:
                raise FileNotFoundError(
                    "RR+LL source-map run is missing matched polarization file(s): "
                    + ", ".join(map(str, missing))
                )
        selected.append(run_path)
    if not selected:
        raise ValueError("No source-map files were selected.")
    return selected


def _layout_grid(n: int):
    """Automatically calculate subplot layout"""
    if n <= 0:
        return 1, 1
    ncol = max(1, math.ceil(math.sqrt(n)))
    nrow = max(1, math.ceil(n / ncol))
    return nrow, ncol


def _get_radio_display_range(cfg, all_extents):
    if cfg.get("use_custom_lim", False):
        xlim = cfg.get("custom_xlim")
        ylim = cfg.get("custom_ylim")
        if xlim is not None and ylim is not None:
            return abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])
    if all_extents:
        extent = all_extents[0]
        return abs(extent[1] - extent[0]), abs(extent[2] - extent[3])
    return 1.0, 1.0


def _auto_multi_band_figure_size(cfg, nrow, ncol, all_extents):
    base_width, base_height = cfg.get("multi_band_fig_size", (24, 16))
    if not cfg.get("multi_band_auto_fig_height", True):
        return base_width, base_height
    x_range, y_range = _get_radio_display_range(cfg, all_extents)
    if x_range <= 0 or y_range <= 0 or nrow <= 0 or ncol <= 0:
        return base_width, base_height
    data_aspect = y_range / x_range
    radio_height = base_width * (nrow / ncol) * data_aspect
    if cfg.get("enable_spectrogram_panel", False):
        panel_ratio = float(cfg.get("spectrogram_panel_height_ratio", 0.34))
        total_height = radio_height + radio_height * panel_ratio + 1.2
    else:
        total_height = radio_height + 1.0
    return base_width, max(total_height, 4.0)


def _compute_manual_radio_panel_rect(
    fig, cfg, nrow, ncol, all_extents, spectrogram_enabled
):
    fig_width, fig_height = fig.get_size_inches()
    left = float(cfg.get("radio_panel_left", cfg.get("radio_grid_left", 0.06)))
    right = float(cfg.get("radio_panel_right", cfg.get("radio_grid_right", 0.98)))
    top = float(cfg.get("radio_panel_top", cfg.get("radio_grid_top", 0.92)))
    bottom = float(cfg.get("radio_panel_bottom", cfg.get("radio_grid_bottom", 0.30)))
    left = min(max(left, 0.0), 1.0)
    right = min(max(right, left + 1e-6), 1.0)
    bottom = min(max(bottom, 0.0), 1.0)
    top = min(max(top, bottom + 1e-6), 1.0)
    available_w = right - left
    available_h = top - bottom
    if nrow <= 0 or ncol <= 0:
        return left, bottom, available_w, available_h

    aspect_mode = str(cfg.get("multi_band_aspect_mode", "equal_compact")).lower()
    if aspect_mode == "fill":
        return left, bottom, available_w, available_h

    x_range, y_range = _get_radio_display_range(cfg, all_extents)
    if x_range <= 0 or y_range <= 0 or fig_width <= 0 or fig_height <= 0:
        return left, bottom, available_w, available_h
    data_aspect = y_range / x_range
    cell_w = available_w / ncol
    cell_h = cell_w * fig_width / fig_height * data_aspect
    required_h = cell_h * nrow
    final_left = left
    final_bottom = bottom
    final_w = available_w
    final_h = required_h
    anchor = str(cfg.get("radio_panel_anchor", "center") or "center").lower()

    if required_h <= available_h or not cfg.get(
        "radio_panel_allow_shrink_height", True
    ):
        if required_h > available_h:
            final_h = available_h
        if anchor in {"top", "upper"}:
            final_bottom = top - final_h
        elif anchor in {"bottom", "lower"}:
            final_bottom = bottom
        else:
            final_bottom = bottom + 0.5 * (available_h - final_h)
    else:
        cell_h = available_h / nrow
        cell_w = cell_h * fig_height / fig_width / data_aspect
        required_w = cell_w * ncol
        final_h = available_h
        final_w = min(required_w, available_w)
        if required_w <= available_w or cfg.get("radio_panel_allow_shrink_width", True):
            final_left = left + 0.5 * (available_w - final_w)
        else:
            final_left = left
            final_w = available_w

    return final_left, final_bottom, final_w, final_h


def _create_manual_radio_axes(fig, cfg, nrow, ncol, all_extents, spectrogram_enabled):
    left, bottom, width, height = _compute_manual_radio_panel_rect(
        fig, cfg, nrow, ncol, all_extents, spectrogram_enabled
    )
    cell_w = width / max(ncol, 1)
    cell_h = height / max(nrow, 1)
    axes = []
    for row in range(nrow):
        row_axes = []
        for col in range(ncol):
            x0 = left + col * cell_w
            y0 = bottom + (nrow - 1 - row) * cell_h
            ax = fig.add_axes([x0, y0, cell_w, cell_h])
            row_axes.append(ax)
        axes.append(row_axes)
    return np.array(axes)


def _apply_fixed_single_band_artifact_layout(
    fig,
    ax,
    cbar,
    *,
    intensity_unit: str | None,
    cfg,
) -> None:
    """Freeze sequence geometry and keep colorbar text legible on white."""

    figure_width, figure_height = (float(value) for value in fig.get_size_inches())
    if figure_width <= 0 or figure_height <= 0:
        raise ValueError("Source Map figure dimensions must be positive")
    x0, x1 = (float(value) for value in ax.get_xlim())
    y0, y1 = (float(value) for value in ax.get_ylim())
    x_span = abs(x1 - x0)
    y_span = abs(y1 - y0)
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Source Map world-coordinate ranges must be positive")

    slot_left = 0.085
    slot_bottom = 0.04
    slot_width = 0.78
    slot_height = 0.90
    figure_aspect = figure_width / figure_height
    data_aspect = x_span / y_span
    panel_width = min(slot_width, slot_height * data_aspect / figure_aspect)
    panel_height = panel_width * figure_aspect / data_aspect
    panel_left = slot_left + 0.5 * (slot_width - panel_width)
    panel_bottom = slot_bottom + 0.5 * (slot_height - panel_height)
    ax.set_position([panel_left, panel_bottom, panel_width, panel_height])

    colorbar_left = 0.895
    colorbar_width = 0.022
    cbar.ax.set_position([colorbar_left, panel_bottom, colorbar_width, panel_height])
    unit = str(intensity_unit or "").strip()
    label = f"Intensity [{unit}]" if unit else "Intensity"
    tick_fontsize = max(12, int(cfg.get("tick_fontsize", 16)) - 2)
    label_fontsize = max(14, int(cfg.get("label_fontsize", 18)) - 4)
    cbar.set_label(
        label,
        fontsize=label_fontsize,
        color="black",
        labelpad=12,
    )
    cbar.ax.tick_params(
        axis="y",
        which="both",
        labelsize=tick_fontsize,
        colors="black",
        length=6,
        width=1.2,
    )
    cbar.ax.yaxis.get_offset_text().set_color("black")
    cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    cbar.outline.set_edgecolor("black")


def _auto_tick_step(vmin, vmax, target=5):
    span = abs(vmax - vmin)
    if span <= 0 or not np.isfinite(span):
        return None
    raw = span / max(target, 1)
    exponent = math.floor(math.log10(raw))
    base = raw / (10**exponent)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exponent)


def _set_compact_radio_ticks(ax, cfg):
    x_tick_step = cfg.get("x_tick_step", 200)
    y_tick_step = cfg.get("y_tick_step", 200)
    target = int(cfg.get("radio_tick_step_auto_target", 5) or 5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if x_tick_step == 0:
        x_tick_step = _auto_tick_step(min(xlim), max(xlim), target)
    if y_tick_step == 0:
        y_tick_step = _auto_tick_step(min(ylim), max(ylim), target)
    if x_tick_step and x_tick_step > 0:
        x_start = math.ceil(min(xlim) / x_tick_step) * x_tick_step
        x_end = math.floor(max(xlim) / x_tick_step) * x_tick_step
        ax.set_xticks(np.arange(x_start, x_end + x_tick_step / 2, x_tick_step))
    if y_tick_step and y_tick_step > 0:
        y_start = math.ceil(min(ylim) / y_tick_step) * y_tick_step
        y_end = math.floor(max(ylim) / y_tick_step) * y_tick_step
        ax.set_yticks(np.arange(y_start, y_end + y_tick_step / 2, y_tick_step))


def _apply_compact_radio_axis_style(ax, row, col, nrow, ncol, cfg):
    hide_inner = cfg.get(
        "radio_hide_inner_ticklabels", cfg.get("hide_inner_ticks", True)
    )
    if hide_inner:
        if row < nrow - 1:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if col > 0:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    if cfg.get("radio_use_global_axis_labels", True):
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        if row == nrow - 1:
            ax.set_xlabel(
                cfg.get("radio_global_xlabel", "x (arcsec)"),
                fontsize=cfg["label_fontsize"] - 6,
            )
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(
                cfg.get("radio_global_ylabel", "y (arcsec)"),
                fontsize=cfg["label_fontsize"] - 6,
            )
        else:
            ax.set_ylabel("")
    if not cfg.get("radio_show_internal_spines", True):
        if col < ncol - 1:
            ax.spines["right"].set_visible(False)
        if row < nrow - 1:
            ax.spines["bottom"].set_visible(False)


def _prune_edge_ticklabels(ax, row, col, nrow, ncol, cfg):
    if not cfg.get("radio_hide_overlapping_edge_ticklabels", True):
        return
    tol = float(cfg.get("radio_tick_prune_tolerance", 1e-6))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmin, xmax = min(xlim), max(xlim)
    ymin, ymax = min(ylim), max(ylim)
    if row == nrow - 1:
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False):
            if col > 0 and abs(tick - xmin) <= max(tol, 1e-6 * max(abs(xmin), 1.0)):
                label.set_visible(False)
            if col < ncol - 1 and abs(tick - xmax) <= max(
                tol, 1e-6 * max(abs(xmax), 1.0)
            ):
                label.set_visible(False)
    if col == 0:
        for tick, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=False):
            if row > 0 and abs(tick - ymax) <= max(tol, 1e-6 * max(abs(ymax), 1.0)):
                label.set_visible(False)
            if row < nrow - 1 and abs(tick - ymin) <= max(
                tol, 1e-6 * max(abs(ymin), 1.0)
            ):
                label.set_visible(False)


def _add_global_radio_axis_labels(fig, axes, cfg, spectrogram_ax=None):
    if not cfg.get("radio_use_global_axis_labels", True):
        return
    xlabel_mode = str(cfg.get("radio_global_xlabel_mode", "auto") or "auto").lower()
    if xlabel_mode == "off":
        return
    fig.canvas.draw_idle()
    boxes = [
        ax.get_position() for row_axes in axes for ax in row_axes if ax.get_visible()
    ]
    if not boxes:
        return
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    top = max(b.y1 for b in boxes)
    show_xlabel = not (
        spectrogram_ax is not None and xlabel_mode == "hidden_when_spectrogram"
    )
    if show_xlabel:
        if spectrogram_ax is not None and xlabel_mode == "auto":
            spec_box = spectrogram_ax.get_position()
            spec_top = spec_box.y1
            gap_fraction = float(cfg.get("radio_spectrogram_label_gap_fraction", 0.55))
            min_gap = float(cfg.get("radio_global_xlabel_min_y_gap", 0.018))
            if bottom - spec_top < 2.0 * min_gap:
                show_xlabel = False
            else:
                label_y = spec_top + (bottom - spec_top) * gap_fraction
                label_y = max(label_y, spec_top + min_gap)
                label_y = min(label_y, bottom - min_gap)
                va = "center"
        else:
            label_y = bottom - float(cfg.get("radio_global_xlabel_offset", 0.015))
            va = "top"
    if show_xlabel:
        fig.text(
            0.5 * (left + right),
            label_y,
            cfg.get("radio_global_xlabel", "x (arcsec)"),
            ha="center",
            va=va,
            fontsize=cfg.get("label_fontsize", 28) - 6,
            color=cfg.get("tick_color", "black"),
        )
    fig.text(
        left - float(cfg.get("radio_global_ylabel_offset", 0.035)),
        0.5 * (bottom + top),
        cfg.get("radio_global_ylabel", "y (arcsec)"),
        ha="right",
        va="center",
        rotation=90,
        fontsize=cfg.get("label_fontsize", 28) - 6,
        color=cfg.get("tick_color", "black"),
    )


# ──────────────────────────────────────────────────────────────
# 输出目录预创建
# ──────────────────────────────────────────────────────────────


def _precreate_single_band_dirs(files: list, output_dir: str):
    """
    Pre-create all single-band output subdirectories in the main process,
    to avoid multiple subprocesses concurrently calling os.makedirs and competing for the file system.
    """
    for fp in files:
        try:
            # 处理元组情况：如果是元组，取第一个文件（RR文件）
            if isinstance(fp, tuple):
                file_path = fp[0]
            else:
                file_path = fp

            with fits.open(file_path, memmap=True) as hdul:
                hdr = (
                    hdul[1].header
                    if (len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU))
                    else hdul[0].header
                )
            freq = get_freq_from_header(hdr)
        except Exception:
            freq = None

        subdir = f"{int(freq)}MHz" if isinstance(freq, (int, float)) else "unknown"
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def _resolve_multi_band_output_dir(output_dir: str, cfg: dict) -> Path:
    """Resolve the actual multi-band output directory under the analysis subdir."""
    return Path(output_dir) / _plot_output_subdir(cfg) / _multi_band_output_subdir(cfg)


def _multi_band_output_subdir(cfg: dict) -> str:
    polarization = cfg.get("polarization", "RR")
    subdir_template = cfg.get("multi_band_output_subdir", "multi_band_{polar}")
    return str(subdir_template).format(polar=polarization)


def _precreate_multi_band_dir(output_dir: str, cfg: dict) -> str:
    """Pre-create the actual multi-band output subdirectory, return its path."""
    multi_output_dir = _resolve_multi_band_output_dir(output_dir, cfg)
    multi_output_dir.mkdir(parents=True, exist_ok=True)
    return str(multi_output_dir)


# ──────────────────────────────────────────────────────────────
# 颜色范围计算辅助函数
# ──────────────────────────────────────────────────────────────


def _calculate_range(data: np.ndarray, cfg: dict, is_global: bool = False) -> tuple:
    """
    计算数据范围。

    参数:
    ----------
    data : np.ndarray
        输入数据（通常是对数化后的数据）
    cfg : dict
        配置字典
    is_global : bool
        是否为全局范围计算

    返回:
    -------
    tuple : (vmin, vmax)
    """
    method = cfg.get("per_band_range_method", "fixed_percentile")
    min_log_range = cfg.get("min_log_range", 0.5)

    if len(data) == 0:
        return 0, 1

    if method == "percentile":
        # 使用5%和95%分位数
        low = np.percentile(data, 5)
        high = np.percentile(data, 95)

        # 如果范围太小，使用1%和99%分位数
        if high - low < min_log_range:
            low = np.percentile(data, 1)
            high = np.percentile(data, 99)

    elif method == "fixed_percentile":
        # 使用用户自定义的百分位数
        percentiles = cfg.get("per_band_percentiles", [66, 95])
        if len(percentiles) == 2:
            low = np.percentile(data, percentiles[0])
            high = np.percentile(data, percentiles[1])
        else:
            # 如果配置错误，使用默认值
            low = np.percentile(data, 5)
            high = np.percentile(data, 95)

        # 如果范围太小，给出警告但保持用户设置
        if high - low < min_log_range:
            warnings.warn(
                f"颜色范围过小 ({high - low:.3f})，考虑调整百分位数设置。", stacklevel=2
            )

    elif method == "minmax":
        # 使用最小最大值方法
        low = np.min(data)
        high = np.max(data)

        # 如果范围太小，给出警告
        if high - low < min_log_range:
            warnings.warn(
                f"颜色范围过小 ({high - low:.3f})，考虑使用百分位数方法。", stacklevel=2
            )
    else:
        # 默认使用5%和95%分位数
        low = np.percentile(data, 5)
        high = np.percentile(data, 95)

        if high - low < min_log_range:
            low = np.percentile(data, 1)
            high = np.percentile(data, 99)

    return low, high


def _calculate_per_band_ranges(all_log_data: list, cfg: dict) -> tuple:
    """
    为每个波段计算合适的对数数据范围。

    参数:
    ----------
    all_log_data : list
        所有波段的对数数据列表
    cfg : dict
        配置字典

    返回:
    -------
    tuple : (band_vmins, band_vmaxs)
    """
    band_vmins = []
    band_vmaxs = []

    for _idx, log_data in enumerate(all_log_data):
        valid_data = log_data[~np.isnan(log_data)]
        if len(valid_data) > 0:
            # 使用配置的方法计算范围
            vmin_band, vmax_band = _calculate_range(valid_data, cfg, is_global=False)
            band_vmins.append(vmin_band)
            band_vmaxs.append(vmax_band)
        else:
            # 如果没有有效数据，使用默认范围
            band_vmins.append(0)
            band_vmaxs.append(1)

    return band_vmins, band_vmaxs


def _resolve_multi_band_display_ranges(
    all_log_data: list[np.ndarray],
    cfg: dict,
    vmin=None,
    vmax=None,
) -> tuple[list[float], list[float]]:
    """Resolve the final log10 display range for every multi-band panel."""

    valid_parts = [data[np.isfinite(data)] for data in all_log_data]
    valid_parts = [data for data in valid_parts if data.size]
    if not valid_parts:
        return [0.0] * len(all_log_data), [1.0] * len(all_log_data)

    all_valid = np.concatenate(valid_parts)
    mode = str(cfg.get("color_range_mode", "auto") or "auto").lower()
    if mode == "fixed":
        raw_vmin = cfg.get("fixed_vmin")
        raw_vmax = cfg.get("fixed_vmax")
        if raw_vmin is None:
            raw_vmin = vmin
        if raw_vmax is None:
            raw_vmax = vmax
        low, high = _linear_limits_to_log10(raw_vmin, raw_vmax, all_valid)
        return [low] * len(all_log_data), [high] * len(all_log_data)

    if mode == "global":
        if vmin is not None and vmax is not None:
            low, high = _linear_limits_to_log10(vmin, vmax, all_valid)
        else:
            low, high = float(np.min(all_valid)), float(np.max(all_valid))
        return [low] * len(all_log_data), [high] * len(all_log_data)

    if mode != "auto":
        warnings.warn(
            f"Unknown multi-band color range mode {mode!r}; using auto.",
            stacklevel=2,
        )
    if (
        "fixed_band_vmins" in cfg
        and "fixed_band_vmaxs" in cfg
        and not background_enabled_for_display(cfg)
    ):
        band_vmins = [float(value) for value in cfg["fixed_band_vmins"]]
        band_vmaxs = [float(value) for value in cfg["fixed_band_vmaxs"]]
    else:
        band_vmins, band_vmaxs = _calculate_per_band_ranges(all_log_data, cfg)
    if cfg.get("use_per_band_colormap", True):
        return band_vmins, band_vmaxs

    low, high = _calculate_range(all_valid, cfg, is_global=True)
    return [float(low)] * len(all_log_data), [float(high)] * len(all_log_data)


def _linear_limits_to_log10(
    vmin, vmax, available_log_values: np.ndarray
) -> tuple[float, float]:
    if vmin is None or vmax is None:
        raise ValueError("A multi-band fixed range requires both vmin and vmax")
    raw_low = float(vmin)
    raw_high = float(vmax)
    if not np.isfinite(raw_low) or not np.isfinite(raw_high):
        raise ValueError("Multi-band color limits must be finite")
    if raw_low < 0 or raw_high <= 0 or raw_low >= raw_high:
        raise ValueError(
            "Multi-band color limits must satisfy 0 <= vmin < vmax and vmax > 0"
        )
    low = float(np.min(available_log_values)) if raw_low == 0 else math.log10(raw_low)
    high = math.log10(raw_high)
    if low >= high:
        raise ValueError("Multi-band color limits collapse after log10 conversion")
    return low, high


def _compute_fixed_band_ranges(cfg: dict) -> tuple:
    """
    为每个波段计算固定的颜色范围（基于该波段所有文件的数据）。

    参数:
    ----------
    cfg : dict
        配置字典

    返回:
    -------
    tuple : (band_vmins, band_vmaxs)
        每个波段的固定颜色范围
    """
    root = cfg["multi_band_root"]
    freqs = cfg["multi_band_freqs"]
    pattern = cfg["band_dir_pattern"]
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", None)

    combine_polarizations = cfg.get("combine_polarizations", False)
    polarization = cfg.get("polarization", "RR")

    band_vmins = []
    band_vmaxs = []

    print("开始计算每个波段的固定颜色范围...")

    for _freq_idx, freq in enumerate(tqdm(freqs, desc="计算波段颜色范围", unit="波段")):
        all_band_data = []

        if combine_polarizations and polarization == "RR+LL":
            # 读取RR和LL两个文件夹的所有文件
            rr_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"])
            )
            ll_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["ll_dir_suffix"])
            )

            # 获取两个文件夹的文件列表
            rr_files = _sorted_fits_for_band(
                rr_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            ll_files = _sorted_fits_for_band(
                ll_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )

            # ── 基于文件名时间戳做精确匹配 ─────────────────────────
            time_tolerance = cfg.get("time_tolerance_seconds", 1.0)
            tolerance_ms = time_tolerance * 1000
            rr_files = _filter_bad_radio_files(
                rr_files, freq, cfg["rr_dir_suffix"], cfg, drop_bad=True
            )
            ll_files = _filter_bad_radio_files(
                ll_files, freq, cfg["ll_dir_suffix"], cfg, drop_bad=True
            )
            matched_pairs = _match_rr_ll_by_time(rr_files, ll_files, tolerance_ms, cfg)

            if not matched_pairs:
                warnings.warn(
                    f"频率 {freq}MHz: RR和LL时间匹配失败，无有效数据", stacklevel=2
                )
                band_vmins.append(0)
                band_vmaxs.append(1)
                continue

            # 读取所有文件的数据
            for rr_path, ll_path in matched_pairs:
                try:
                    rr_data, rr_header = read_fits(rr_path)
                    ll_data, ll_header = read_fits(ll_path)

                    # 组合数据（加权平均或简单相加）
                    combined_data = _combine_polarization_data(rr_data, ll_data, cfg)

                    # 对数化处理
                    mask = combined_data > 0
                    log_data = np.full_like(combined_data, np.nan, dtype=np.float64)
                    log_data[mask] = np.log10(combined_data[mask])

                    # 收集有效数据
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 0:
                        all_band_data.extend(valid_data)

                except Exception as e:
                    warnings.warn(
                        f"读取文件时出错（频率 {freq}MHz）: {e}", stacklevel=2
                    )
                    continue
        else:
            # 普通模式：只读取指定偏振的文件
            band_dir = os.path.join(root, pattern.format(freq=freq, polar=polarization))
            files = _sorted_fits_for_band(
                band_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            files = _filter_bad_radio_files(
                files, freq, polarization, cfg, drop_bad=True
            )

            # 读取所有文件的数据
            for file_path in files:
                try:
                    img_data, header = read_fits(file_path)

                    # 对数化处理
                    mask = img_data > 0
                    log_data = np.full_like(img_data, np.nan, dtype=np.float64)
                    log_data[mask] = np.log10(img_data[mask])

                    # 收集有效数据
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 0:
                        all_band_data.extend(valid_data)

                except Exception as e:
                    warnings.warn(
                        f"读取文件时出错（频率 {freq}MHz）: {e}", stacklevel=2
                    )
                    continue

        # 计算该波段的固定颜色范围
        if len(all_band_data) > 0:
            all_band_data_array = np.array(all_band_data)
            vmin_band, vmax_band = _calculate_range(
                all_band_data_array, cfg, is_global=False
            )
            band_vmins.append(vmin_band)
            band_vmaxs.append(vmax_band)

            print(
                f"频率 {freq}MHz: 固定颜色范围 = [{vmin_band:.3f}, {vmax_band:.3f}] (基于{len(all_band_data)}个数据点)"
            )
        else:
            # 如果没有有效数据，使用默认范围
            band_vmins.append(0)
            band_vmaxs.append(1)
            print(f"频率 {freq}MHz: 警告 - 没有有效数据，使用默认范围")

    print("每个波段的固定颜色范围计算完成！")
    return band_vmins, band_vmaxs


# ──────────────────────────────────────────────────────────────
# 核心绘图函数（在子进程中执行）
# ──────────────────────────────────────────────────────────────


def _should_precompute_fixed_band_ranges(cfg: dict) -> bool:
    """Return whether multi-band plotting should reuse cross-frame color ranges."""

    return str(cfg.get("color_range_mode", "auto") or "auto").lower() != "auto"


def _combine_polarization_data(rr_data, ll_data, cfg):
    """组合RR和LL数据（加权平均或简单相加）"""
    weighted = cfg.get("weighted_average", False)

    if weighted:
        rr_weight = cfg.get("rr_weight", 0.5)
        ll_weight = cfg.get("ll_weight", 0.5)
        combined_data = rr_data * rr_weight + ll_data * ll_weight
    else:
        combined_data = rr_data + ll_data

    return combined_data


def plot_single_band(
    file_path: str,
    output_dir: str,
    cfg: dict,
    vmin=None,
    vmax=None,
    *,
    write_sidecar: bool | None = None,
    sequence: int = 1,
    generated_at: datetime.datetime | None = None,
) -> str:
    """
    Process and plot a single-band FITS file, save as PNG image.

    Note: Output directories have been pre-created by the main process, just concatenate paths here, no need to call os.makedirs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches

    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)

    # 检查是否启用左右旋数据加和
    combine_polarizations = cfg.get("combine_polarizations", False)

    if combine_polarizations:
        # 获取对应的左旋/右旋文件路径
        base_dir = os.path.dirname(os.path.dirname(file_path))  # 获取频率目录
        file_name = os.path.basename(file_path)

        # 根据当前文件路径判断是RR还是LL
        if cfg["rr_dir_suffix"] in file_path:
            # 当前是RR文件，查找对应的LL文件
            rr_path = file_path
            ll_path = os.path.join(base_dir, cfg["ll_dir_suffix"], file_name)
            current_pol = "RR"
        elif cfg["ll_dir_suffix"] in file_path:
            # 当前是LL文件，查找对应的RR文件
            ll_path = file_path
            rr_path = os.path.join(base_dir, cfg["rr_dir_suffix"], file_name)
            current_pol = "LL"
        else:
            # 无法确定偏振，使用单个文件
            img_data, header = read_fits(file_path)
            rr_data, ll_data = None, None
            current_pol = get_polar_from_header(header)
    else:
        img_data, header = read_fits(file_path)
        rr_data, ll_data = None, None
        current_pol = get_polar_from_header(header)

    if combine_polarizations and "rr_path" in locals() and "ll_path" in locals():
        # 读取两个偏振的数据并组合
        try:
            rr_data, rr_header = read_fits(rr_path)
            ll_data, ll_header = read_fits(ll_path)

            # 检查时间对齐
            time_tolerance = cfg.get("time_tolerance_seconds", 1.0)
            if not _check_time_alignment(
                rr_header, ll_header, rr_path, ll_path, time_tolerance
            ):
                warnings.warn(
                    f"RR和LL文件时间未对齐: {rr_path} vs {ll_path}", stacklevel=2
                )

            # 数据组合（加权平均或简单相加）
            # TODO: future version should subtract RR and LL backgrounds separately before combination.
            freq_for_bg = get_freq_from_header(rr_header) or "Unknown"
            if background_workflow_enabled(cfg) and cfg.get(
                "background_subtract_before_polarization_combine", True
            ):
                rr_sub, rr_background, rr_diag = subtract_radio_background(
                    rr_data,
                    cfg,
                    source_file=rr_path,
                    band=freq_for_bg,
                    polarization="RR",
                )
                ll_sub, ll_background, ll_diag = subtract_radio_background(
                    ll_data,
                    cfg,
                    source_file=ll_path,
                    band=freq_for_bg,
                    polarization="LL",
                )
                img_data_bgsub = _combine_polarization_data(rr_sub, ll_sub, cfg)
                background_map_for_slot = (
                    _combine_polarization_data(rr_background, ll_background, cfg)
                    if rr_background is not None and ll_background is not None
                    else None
                )
                bg_diag_for_slot = dict(rr_diag)
                bg_diag_for_slot.update(
                    {
                        "source_file": f"{rr_path}|{ll_path}",
                        "background_mode_used": (
                            f"RR:{rr_diag.get('background_mode_used')};"
                            f"LL:{ll_diag.get('background_mode_used')}"
                        ),
                        "background_file_count": rr_diag.get("background_file_count", 0)
                        + ll_diag.get("background_file_count", 0),
                        "background_scale": (
                            f"RR:{rr_diag.get('background_scale')};"
                            f"LL:{ll_diag.get('background_scale')}"
                        ),
                        "warning": "; ".join(
                            item
                            for item in [
                                rr_diag.get("warning", ""),
                                ll_diag.get("warning", ""),
                            ]
                            if item
                        ),
                    }
                )
            else:
                img_data_bgsub = None
                background_map_for_slot = None
                bg_diag_for_slot = None

            img_data = _combine_polarization_data(rr_data, ll_data, cfg)
            header = rr_header  # 使用RR文件的头文件
            polar_display = "RR+LL"

            # 如果需要保存单独的偏振图像
            save_individual = cfg.get("save_individual_pols", False)
            if save_individual:
                # 保存单独的RR图像
                rr_polar_display = "RR"
                rr_out_path = _save_single_pol_image(
                    rr_data,
                    rr_header,
                    output_dir,
                    cfg,
                    vmin,
                    vmax,
                    rr_polar_display,
                    file_name,
                    sequence=sequence,
                    generated_at=generated_at,
                )
                # 保存单独的LL图像
                ll_polar_display = "LL"
                ll_out_path = _save_single_pol_image(
                    ll_data,
                    ll_header,
                    output_dir,
                    cfg,
                    vmin,
                    vmax,
                    ll_polar_display,
                    file_name,
                    sequence=sequence,
                    generated_at=generated_at,
                )

        except FileNotFoundError as e:
            warnings.warn(f"无法找到对应的偏振文件: {e}，使用单个文件", stacklevel=2)
            if current_pol == "RR" and rr_data is not None:
                img_data, header = rr_data, rr_header
                polar_display = "RR"
            elif current_pol == "LL" and ll_data is not None:
                img_data, header = ll_data, ll_header
                polar_display = "LL"
            else:
                img_data, header = read_fits(file_path)
                polar_display = get_polar_from_header(header)
    else:
        polar_display = get_polar_from_header(header)

    contributing_headers = [header]
    artifact_source_files = [file_path]
    if polar_display == "RR+LL" and "rr_header" in locals() and "ll_header" in locals():
        contributing_headers = [rr_header, ll_header]
        artifact_source_files = [rr_path, ll_path]
    unit_resolution = resolve_colorbar_unit(
        contributing_headers, cfg.get("radio_colorbar_unit")
    )
    radio_colorbar_label = colorbar_label(unit_resolution, transform="linear")
    radio_tick_notation = tick_notation_for_transform("linear")
    for message in unit_resolution.warnings:
        warnings.warn(message, stacklevel=2)

    extent, image_origin = calc_radio_extent_and_origin(header, img_data.shape, cfg)
    cfg["_current_radio_image_origin"] = image_origin

    rsun_obs = header.get("RSUN_OBS", 960.0)
    freq = get_freq_from_header(header) or "Unknown"
    time_str = get_time_from_header(header)
    gaussian_cfg = config_for_gaussian_band(cfg, freq)
    gaussian_cfg["_current_radio_image_origin"] = image_origin
    file_name = os.path.basename(file_path)
    current_frame_time = radio_datetime_from_header_or_path(header, file_path, cfg)
    source_file_for_quality = file_path
    if combine_polarizations and "rr_path" in locals() and "ll_path" in locals():
        source_file_for_quality = f"{rr_path}|{ll_path}"
    raw_quality_bad_frame = _raw_quality_item_is_bad(source_file_for_quality, cfg)
    bg_sub_data = img_data
    background_map = None
    bg_diag = {}
    if background_workflow_enabled(cfg):
        if (
            polar_display == "RR+LL"
            and cfg.get("background_subtract_before_polarization_combine", True)
            and rr_data is not None
            and ll_data is not None
        ):
            rr_sub, rr_background, rr_diag = subtract_radio_background(
                rr_data, cfg, source_file=rr_path, band=freq, polarization="RR"
            )
            ll_sub, ll_background, ll_diag = subtract_radio_background(
                ll_data, cfg, source_file=ll_path, band=freq, polarization="LL"
            )
            bg_sub_data = _combine_polarization_data(rr_sub, ll_sub, cfg)
            background_map = (
                _combine_polarization_data(rr_background, ll_background, cfg)
                if rr_background is not None and ll_background is not None
                else None
            )
            bg_diag = dict(rr_diag)
            bg_diag.update(
                {
                    "source_file": f"{rr_path}|{ll_path}",
                    "background_mode_used": (
                        f"RR:{rr_diag.get('background_mode_used')};"
                        f"LL:{ll_diag.get('background_mode_used')}"
                    ),
                    "background_file_count": rr_diag.get("background_file_count", 0)
                    + ll_diag.get("background_file_count", 0),
                    "background_scale": (
                        f"RR:{rr_diag.get('background_scale')};"
                        f"LL:{ll_diag.get('background_scale')}"
                    ),
                    "warning": "; ".join(
                        item
                        for item in [
                            rr_diag.get("warning", ""),
                            ll_diag.get("warning", ""),
                        ]
                        if item
                    ),
                }
            )
        else:
            if polar_display == "RR+LL":
                # Less strict than per-polarization subtraction because RR and LL
                # backgrounds may have different levels before combination.
                pass
            bg_sub_data, background_map, bg_diag = subtract_radio_background(
                img_data,
                cfg,
                source_file=file_path,
                band=freq,
                polarization=polar_display,
            )
    else:
        bg_sub_data = img_data.copy()
        background_map = None
        bg_diag = {
            "background_enabled": False,
            "background_mode_requested": "none",
            "background_mode_used": "none",
            "source_file": file_path,
        }

    background_map_for_mask = None
    rms_map_for_mask = None
    background_diag_for_mask = {}
    strategy = str(
        cfg.get("radio_background_strategy", "noise_map_only") or "none"
    ).lower()
    if strategy in {"noise_map_only", "local_mesh"} or cfg.get(
        "background_use_for_mask", True
    ):
        background_map_for_mask, rms_map_for_mask, background_diag_for_mask = (
            estimate_background_rms_mesh(img_data, cfg)
        )
        bg_diag.update(background_diag_for_mask)
    if cfg.get("gaussian_fit_verbose", False):
        if cfg.get("gaussian_fit_verbose", False):
            print(
                "[Background strategy] "
                f"strategy={strategy}, mask={bool(cfg.get('background_use_for_mask', True))}, "
                f"display={bool(cfg.get('background_use_for_display', False))}, "
                f"fit={bool(cfg.get('background_use_for_fit', False))}"
            )

    if (
        cfg.get("background_use_for_display", False)
        and cfg.get("display_input_type", "raw") in {"excess", "background_subtracted"}
        and background_map_for_mask is not None
    ):
        display_data = img_data - background_map_for_mask
        if cfg.get("background_clip_negative_for_display_only", True):
            display_data = np.where(display_data < 0, 0.0, display_data)
    else:
        display_data = img_data
    if cfg.get("background_use_for_fit", False):
        fit_base_data = bg_sub_data
        fit_input_type = "background_subtracted"
    else:
        fit_base_data = img_data
        fit_input_type = "raw"

    fit_result = None
    multi_fit_result = None
    radio_fit_data = None
    if cfg.get("enable_gaussian_overlay", False) and not raw_quality_bad_frame:
        if gaussian_cfg.get("gaussian_fit_verbose", False):
            print(
                "[Gaussian input] using "
                f"{'background-subtracted radio image' if gaussian_cfg.get('background_use_for_fit', False) else 'raw radio image'} "
                "with local baseline model"
            )
        radio_fit_data = fit_base_data
        if _gaussian_multi_source_enabled(gaussian_cfg):
            multi_fit_result = _fit_multiple_gaussians_on_radio_image(
                radio_fit_data,
                extent=extent,
                cfg=gaussian_cfg,
                source_file=file_path,
                background_map=background_map_for_mask,
                rms_map=rms_map_for_mask,
                fit_input_type=fit_input_type,
                image_origin=image_origin,
            )
            fit_result = multi_fit_result.primary_result
        else:
            fit_result = fit_elliptical_gaussian_on_radio_image(
                radio_fit_data,
                extent=extent,
                cfg=gaussian_cfg,
                source_file=file_path,
                background_map=background_map_for_mask,
                rms_map=rms_map_for_mask,
                fit_input_type=fit_input_type,
                image_origin=image_origin,
            )
            _attach_raw_peak_center(
                fit_result, img_data, extent, gaussian_cfg, image_origin
            )

    # 将偏振显示名称
    if polar_display == "RR":
        polar_display = "Right Circular (RR)"
    elif polar_display == "LL":
        polar_display = "Left Circular (LL)"
    elif polar_display == "RR+LL":
        if cfg.get("weighted_average", False):
            rr_weight = cfg.get("rr_weight", 0.5)
            ll_weight = cfg.get("ll_weight", 0.5)
            polar_display = f"RR+LL Combined (weighted: RR={rr_weight}, LL={ll_weight})"
        else:
            polar_display = "RR+LL Combined (sum)"

    title = f"{file_name}   {freq} MHz  {polar_display}   {time_str}"

    if _spectrogram_panel_enabled(cfg):
        base_width, base_height = cfg["fig_size"]
        panel_ratio = float(cfg.get("spectrogram_panel_height_ratio", 0.34))
        fig = plt.figure(figsize=(base_width, base_height * (1.0 + panel_ratio)))
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[1.0, panel_ratio],
            hspace=float(cfg.get("spectrogram_hspace", 0.08)),
            left=float(cfg.get("radio_grid_left", 0.06)),
            right=float(cfg.get("radio_grid_right", 0.98)),
            top=float(cfg.get("radio_grid_top", 0.92)),
            bottom=float(cfg.get("radio_grid_bottom", 0.30)),
        )
        ax = fig.add_subplot(gs[0, 0])
        spectrogram_ax = fig.add_subplot(gs[1, 0])
    else:
        fig, ax = plt.subplots(
            figsize=cfg["fig_size"],
            dpi=(
                int(cfg["dpi"])
                if cfg.get("_artifact_fixed_panel_layout", False)
                else None
            ),
        )
        spectrogram_ax = None

    im_kwargs = get_imshow_kwargs(
        display_data, extent, cfg, aspect="equal", origin=image_origin
    )

    # 优先使用固定颜色范围
    if (
        cfg.get("color_range_mode") == "fixed"
        and "fixed_vmin" in cfg
        and "fixed_vmax" in cfg
    ):
        im_kwargs["vmin"] = cfg["fixed_vmin"]
        im_kwargs["vmax"] = cfg["fixed_vmax"]
    elif vmin is not None and vmax is not None:
        im_kwargs["vmin"] = vmin
        im_kwargs["vmax"] = vmax

    im = ax.imshow(display_data, **im_kwargs)
    add_radio_coordinate_corner_debug(ax, extent, img_data.shape, image_origin, cfg)
    if multi_fit_result is not None:
        _overlay_multi_gaussian_fit_on_axis(
            ax, multi_fit_result, extent, img_data.shape, gaussian_cfg
        )
    elif fit_result is not None:
        overlay_gaussian_fit_on_axis(
            ax, fit_result, extent, img_data.shape, gaussian_cfg
        )
    if (
        cfg.get("enable_gaussian_overlay", False)
        and not raw_quality_bad_frame
        and cfg.get("save_gaussian_diagnostics", True)
    ):
        save_gaussian_diagnostics_row(
            _gaussian_diagnostics_row(
                fit_result, gaussian_cfg, freq, time_str, polar_display, bg_diag
            ),
            output_dir,
            cfg,
        )
        if multi_fit_result is not None:
            for diagnostic_row in _multi_gaussian_diagnostics_rows(
                multi_fit_result, gaussian_cfg, freq, time_str, polar_display, bg_diag
            ):
                _save_multi_gaussian_diagnostics_row(diagnostic_row, output_dir, cfg)
    if cfg.get("save_background_diagnostics", True) and background_workflow_enabled(
        cfg
    ):
        save_background_diagnostics_row(
            _background_diagnostics_row(
                bg_diag, cfg, img_data, bg_sub_data, freq, time_str, polar_display
            ),
            output_dir,
            cfg,
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    apply_colorbar_tick_notation(cbar, transform="linear")
    cbar.set_label(radio_colorbar_label)
    cbar.ax.tick_params(
        labelsize=cfg["tick_fontsize"] - 4,
        colors=cfg.get("colorbar_tick_color", "yellow"),
    )
    # 添加颜色条刻度旋转（新增代码）
    tick_rotation = cfg.get("tick_label_rotation", 0)
    if tick_rotation != 0:
        cbar.ax.tick_params(axis="y", rotation=tick_rotation)

    ax.set_title(title, fontsize=cfg["title_fontsize"], fontweight="bold", pad=20)
    ax.set_xlabel("x (arcsec)", fontsize=cfg["label_fontsize"])
    ax.set_ylabel("y (arcsec)", fontsize=cfg["label_fontsize"])
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=cfg["tick_fontsize"],
        colors=cfg.get("tick_color", "yellow"),
    )

    # 设置坐标轴刻度步长（新增代码）
    x_tick_step = cfg.get("x_tick_step", 200)
    y_tick_step = cfg.get("y_tick_step", 200)

    # 获取当前坐标轴范围
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # 计算x轴刻度位置
    if x_tick_step and x_tick_step > 0:
        x_start = math.ceil(current_xlim[0] / x_tick_step) * x_tick_step
        x_end = math.floor(current_xlim[1] / x_tick_step) * x_tick_step
        x_ticks = np.arange(x_start, x_end + x_tick_step / 2, x_tick_step)
        ax.set_xticks(x_ticks)

    # 计算y轴刻度位置
    if y_tick_step and y_tick_step > 0:
        y_start = math.ceil(current_ylim[0] / y_tick_step) * y_tick_step
        y_end = math.floor(current_ylim[1] / y_tick_step) * y_tick_step
        y_ticks = np.arange(y_start, y_end + y_tick_step / 2, y_tick_step)
        ax.set_yticks(y_ticks)

    # 应用刻度标签旋转
    tick_rotation = cfg.get("tick_label_rotation", 0)
    if tick_rotation != 0:
        ax.tick_params(axis="x", rotation=tick_rotation)
        ax.tick_params(axis="y", rotation=tick_rotation)

    ax.add_patch(
        patches.Circle(
            (0, 0),
            radius=rsun_obs,
            edgecolor="white",
            facecolor="none",
            linewidth=3,
        )
    )

    half = rsun_obs
    ax.add_line(
        plt.Line2D(
            [-half, half], [0, 0], color="cyan", lw=1.5, linestyle="--", alpha=0.8
        )
    )
    ax.add_line(
        plt.Line2D(
            [0, 0], [-half, half], color="cyan", lw=1.5, linestyle="--", alpha=0.8
        )
    )

    offset_inner = rsun_obs + 50
    offset_text = rsun_obs + 150
    arrow_props = dict(arrowstyle="->", color="yellow", lw=2)
    fs = cfg["annotation_fontsize"]
    ax.annotate(
        "N",
        xy=(0, offset_inner),
        xytext=(0, offset_text),
        ha="center",
        va="bottom",
        fontsize=fs,
        color="yellow",
        arrowprops=arrow_props,
    )
    ax.annotate(
        "E",
        xy=(offset_inner, 0),
        xytext=(offset_text, 0),
        ha="left",
        va="center",
        fontsize=fs,
        color="yellow",
        arrowprops=arrow_props,
    )

    if cfg.get("use_custom_lim", False):
        ax.set_xlim(cfg["custom_xlim"])
        ax.set_ylim(cfg["custom_ylim"])
    else:
        sf = cfg["scale_factor"]
        ax.set_xlim(-rsun_obs * sf, rsun_obs * sf)
        ax.set_ylim(-rsun_obs * sf, rsun_obs * sf)

    ax.grid(True, alpha=0.3, linestyle=":", color="gray")
    legend_handles = [
        Line2D([0], [0], color="white", lw=3, label=f'Solar Limb (R={rsun_obs:.0f}")'),
        Line2D([0], [0], color="cyan", lw=1.5, linestyle="--", label="Solar Grid"),
    ]
    if fit_result is not None:
        display_mode = cfg.get("gaussian_overlay_display_mode", "contours_and_fwhm")
        if display_mode not in {
            "contours_and_fwhm",
            "contours_only",
            "fwhm_only",
            "center_only",
            "none",
        }:
            display_mode = "contours_and_fwhm"
        show_low_quality_shapes = fit_result.quality_flag == "ok" or cfg.get(
            "draw_low_quality_gaussian_contours", False
        )
        if (
            display_mode in {"contours_and_fwhm", "contours_only"}
            and cfg.get("draw_gaussian_contours", True)
            and show_low_quality_shapes
        ):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cfg.get("gaussian_contour_color", "white"),
                    lw=cfg.get("gaussian_contour_linewidth", 2.0),
                    label="Gaussian contours",
                )
            )
        if display_mode in {
            "contours_and_fwhm",
            "contours_only",
            "fwhm_only",
            "center_only",
        } and cfg.get("draw_gaussian_center", True):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cfg.get("gaussian_center_color", "red"),
                    marker=cfg.get("gaussian_center_marker", "x"),
                    linestyle="None",
                    markersize=8,
                    label="Gaussian center",
                )
            )
        if (
            display_mode in {"contours_and_fwhm", "fwhm_only"}
            and cfg.get("draw_gaussian_fwhm_ellipse", True)
            and show_low_quality_shapes
            and getattr(fit_result, "quality_flag_detail", "") != "skipped_large_fwhm"
        ):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cfg.get("gaussian_fwhm_color", "lime"),
                    lw=cfg.get("gaussian_fwhm_linewidth", 2.0),
                    label="Gaussian FWHM",
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=cfg["legend_fontsize"],
    )

    if spectrogram_ax is not None:
        overlay_spectrogram_panel(spectrogram_ax, cfg, current_frame_time)

    if cfg.get("_artifact_fixed_panel_layout", False):
        _apply_fixed_single_band_artifact_layout(
            fig,
            ax,
            cbar,
            intensity_unit=unit_resolution.unit,
            cfg=cfg,
        )
    else:
        plt.tight_layout()

    # ★ 优化：输出目录已预创建，直接拼接路径
    subdir = f"{int(freq)}MHz" if isinstance(freq, (int, float)) else "unknown"
    if raw_quality_bad_frame:
        overlay_dir = str(_raw_quality_bad_frame_output_dir(output_dir, cfg, subdir))
    else:
        overlay_dir = os.path.join(output_dir, _plot_output_subdir(cfg), subdir)
    os.makedirs(overlay_dir, exist_ok=True)
    suffix_tokens = tuple(
        token for token in _output_suffix(cfg).strip("_").split("_") if token
    )
    output_name = build_scientific_image_filename(
        sequence=2 * sequence - 1,
        start_time=current_frame_time,
        instrument="radio",
        channel=f"{float(freq):g}mhz" if isinstance(freq, (int, float)) else None,
        polarization=str(cfg.get("polarization") or polar_display),
        product="source_map",
        qualifiers=suffix_tokens,
        generated_at=generated_at,
    )
    out_path = os.path.join(overlay_dir, output_name)

    if cfg["save_plot"]:
        spatial_display = resolve_spatial_display(cfg, transform="linear")
        should_write_sidecar = (
            bool(cfg.get("write_source_map_sidecar", False))
            if write_sidecar is None
            else bool(write_sidecar)
        )
        artifact_warnings = list(unit_resolution.warnings)
        artifact_warnings.extend(cfg.get("_spectrogram_unit_warnings", []))
        save_figure_artifact(
            fig,
            out_path,
            dpi=cfg["dpi"],
            radio_axes=[ax],
            panel_metadata=[
                {
                    "id": "radio-1",
                    "frequency_mhz": (
                        float(freq) if isinstance(freq, (int, float)) else None
                    ),
                    "polarization": str(polar_display),
                    "transform": "linear",
                    "tick_notation": radio_tick_notation,
                    "colorbar_label": radio_colorbar_label,
                    "unit": unit_resolution.to_dict(),
                }
            ],
            mode="single_band",
            polarization=str(polar_display),
            source_files=artifact_source_files,
            write_sidecar=should_write_sidecar,
            warnings=artifact_warnings,
            display=spatial_display.sidecar_payload(),
            bbox_inches=cfg.get("_artifact_bbox_inches", "tight"),
            pad_inches=float(cfg.get("_artifact_pad_inches", 0.1)),
        )
        if (
            background_workflow_enabled(cfg)
            or cfg.get("save_background_subtracted_image", False)
            or cfg.get("save_estimated_background_map", False)
        ):
            _save_background_products(
                bg_sub_data,
                background_map,
                extent,
                output_dir,
                os.path.splitext(file_name)[0],
                cfg,
                image_origin,
                radio_colorbar_label,
                observation_time=current_frame_time,
                frequency_mhz=freq if isinstance(freq, (int, float)) else None,
                polarization=str(cfg.get("polarization") or polar_display),
                sequence=sequence,
                generated_at=generated_at,
            )
        if (
            cfg.get("draw_fit_residual_panel", False)
            and fit_result is not None
            and radio_fit_data is not None
        ):
            _save_gaussian_residual_panel(
                radio_fit_data,
                fit_result,
                extent,
                out_path,
                gaussian_cfg,
                colorbar_text=radio_colorbar_label,
                residual_output_path=os.path.join(
                    overlay_dir,
                    build_scientific_image_filename(
                        sequence=2 * sequence,
                        start_time=current_frame_time,
                        instrument="radio",
                        channel=(
                            f"{float(freq):g}mhz"
                            if isinstance(freq, (int, float))
                            else None
                        ),
                        polarization=str(cfg.get("polarization") or polar_display),
                        product="gaussian_residual",
                        generated_at=generated_at,
                    ),
                ),
            )

    if cfg["show_plot"] and cfg.get("_interactive", False):
        plt.show()

    plt.close(fig)
    return out_path


def _save_single_pol_image(
    img_data,
    header,
    output_dir,
    cfg,
    vmin,
    vmax,
    polar_display,
    base_filename,
    *,
    sequence=1,
    generated_at=None,
):
    """保存单独的偏振图像"""

    extent, image_origin = calc_radio_extent_and_origin(header, img_data.shape, cfg)
    unit_resolution = resolve_colorbar_unit([header], cfg.get("radio_colorbar_unit"))
    radio_colorbar_label = colorbar_label(unit_resolution, transform="linear")
    for message in unit_resolution.warnings:
        warnings.warn(message, stacklevel=2)
    rsun_obs = header.get("RSUN_OBS", 960.0)
    freq = get_freq_from_header(header) or "Unknown"
    time_str = get_time_from_header(header)
    observation_time = radio_datetime_from_header_or_path(
        header,
        base_filename,
        cfg,
    )
    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)

    # 修改文件名以包含偏振信息
    new_filename = build_scientific_image_filename(
        sequence=sequence,
        start_time=observation_time,
        instrument="radio",
        channel=f"{float(freq):g}mhz" if isinstance(freq, (int, float)) else None,
        polarization=polar_display,
        product="source_map",
        generated_at=generated_at,
    )

    # 创建子目录
    subdir = (
        f"{int(freq)}MHz_{polar_display}"
        if isinstance(freq, (int, float))
        else f"unknown_{polar_display}"
    )
    pol_output_dir = os.path.join(output_dir, "individual_pols", subdir)
    os.makedirs(pol_output_dir, exist_ok=True)

    # 创建图像
    fig, ax = plt.subplots(figsize=cfg["fig_size"])

    im_kwargs = get_imshow_kwargs(
        img_data, extent, cfg, aspect="equal", origin=image_origin
    )
    if vmin is not None:
        im_kwargs["vmin"] = vmin
    if vmax is not None:
        im_kwargs["vmax"] = vmax

    im = ax.imshow(img_data, **im_kwargs)
    add_radio_coordinate_corner_debug(ax, extent, img_data.shape, image_origin, cfg)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    apply_colorbar_tick_notation(cbar, transform="linear")
    cbar.set_label(radio_colorbar_label)
    cbar.ax.tick_params(
        labelsize=cfg["tick_fontsize"] - 4,
        colors=cfg.get("colorbar_tick_color", "yellow"),
    )

    title = f"{base_filename}   {freq} MHz  {polar_display}   {time_str}"
    ax.set_title(title, fontsize=cfg["title_fontsize"] - 4, fontweight="bold", pad=20)
    ax.set_xlabel("x (arcsec)", fontsize=cfg["label_fontsize"] - 4)
    ax.set_ylabel("y (arcsec)", fontsize=cfg["label_fontsize"] - 4)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=cfg["tick_fontsize"] - 4,
        colors=cfg.get("tick_color", "yellow"),
    )

    # 添加太阳轮廓
    ax.add_patch(
        patches.Circle(
            (0, 0),
            radius=rsun_obs,
            edgecolor="white",
            facecolor="none",
            linewidth=2,
        )
    )

    plt.tight_layout()
    out_path = os.path.join(pol_output_dir, new_filename)
    plt.savefig(out_path, dpi=cfg["dpi"] - 50, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_multi_band_slot(
    slot_idx: int,
    slot_files: list,
    output_dir: str,
    cfg: dict,
    vmin=None,
    vmax=None,
    *,
    write_sidecar: bool | None = None,
    sequence: int = 1,
    generated_at: datetime.datetime | None = None,
) -> str:
    """
    Process and plot multi-band composite image (all bands in one time slot).

    Note: Output directories have been pre-created by the main process, just concatenate paths here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches

    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)

    all_data = []
    all_headers = []
    all_extents = []
    all_origins = []
    band_info = []
    all_source_files = []
    all_bg_sub_data = []
    all_background_maps = []
    all_background_mask_maps = []
    all_rms_mask_maps = []
    all_bg_diags = []
    all_raw_quality_bad = []
    all_unit_headers = []
    artifact_source_files = []

    # 存储每个波段的对数化数据
    all_log_data = []

    # 检查是否启用左右旋数据加和
    combine_polarizations = cfg.get("combine_polarizations", False)
    polarization = cfg.get("polarization", "RR")
    time_tolerance = cfg.get("time_tolerance_seconds", 1.0)

    # 如果需要保存单独的偏振图像，创建存储路径
    save_individual = cfg.get("save_individual_pols", False)
    individual_outputs = []

    for file_item in slot_files:
        if (
            combine_polarizations
            and polarization == "RR+LL"
            and isinstance(file_item, tuple)
        ):
            # 左右旋数据加和模式：file_item是(RR文件路径, LL文件路径)的元组
            rr_path, ll_path = file_item

            # 读取两个偏振的数据
            rr_data, rr_header = read_fits(rr_path)
            ll_data, ll_header = read_fits(ll_path)
            contributing_headers = [rr_header, ll_header]
            artifact_source_files.extend([rr_path, ll_path])

            # 检查时间对齐
            if not _check_time_alignment(
                rr_header, ll_header, rr_path, ll_path, time_tolerance
            ):
                warnings.warn(
                    f"RR和LL文件时间未对齐: {rr_path} vs {ll_path}", stacklevel=2
                )

            # 数据组合（加权平均或简单相加）
            # TODO: future version should subtract RR and LL backgrounds separately before combination.
            img_data = _combine_polarization_data(rr_data, ll_data, cfg)
            header = rr_header  # 使用RR文件的头文件
            polar_display = "RR+LL"
            source_file_for_diag = f"{rr_path}|{ll_path}"
            freq_for_bg = get_freq_from_header(rr_header) or "Unknown"
            if background_workflow_enabled(cfg) and cfg.get(
                "background_subtract_before_polarization_combine", True
            ):
                rr_sub, rr_background, rr_diag = subtract_radio_background(
                    rr_data,
                    cfg,
                    source_file=rr_path,
                    band=freq_for_bg,
                    polarization="RR",
                )
                ll_sub, ll_background, ll_diag = subtract_radio_background(
                    ll_data,
                    cfg,
                    source_file=ll_path,
                    band=freq_for_bg,
                    polarization="LL",
                )
                img_data_bgsub = _combine_polarization_data(rr_sub, ll_sub, cfg)
                background_map_for_slot = (
                    _combine_polarization_data(rr_background, ll_background, cfg)
                    if rr_background is not None and ll_background is not None
                    else None
                )
                bg_diag_for_slot = dict(rr_diag)
                bg_diag_for_slot.update(
                    {
                        "source_file": source_file_for_diag,
                        "background_mode_used": (
                            f"RR:{rr_diag.get('background_mode_used')};"
                            f"LL:{ll_diag.get('background_mode_used')}"
                        ),
                        "background_file_count": rr_diag.get("background_file_count", 0)
                        + ll_diag.get("background_file_count", 0),
                        "background_scale": (
                            f"RR:{rr_diag.get('background_scale')};"
                            f"LL:{ll_diag.get('background_scale')}"
                        ),
                        "warning": "; ".join(
                            item
                            for item in [
                                rr_diag.get("warning", ""),
                                ll_diag.get("warning", ""),
                            ]
                            if item
                        ),
                    }
                )
            else:
                if background_workflow_enabled(cfg):
                    # This combines first and subtracts later; RR/LL backgrounds may differ.
                    img_data_bgsub, background_map_for_slot, bg_diag_for_slot = (
                        subtract_radio_background(
                            img_data,
                            cfg,
                            source_file=source_file_for_diag,
                            band=freq_for_bg,
                            polarization=polar_display,
                        )
                    )
                else:
                    img_data_bgsub = img_data.copy()
                    background_map_for_slot = None
                    bg_diag_for_slot = _background_disabled_diag(source_file_for_diag)

            # 如果需要保存单独的偏振图像
            if save_individual:
                freq = get_freq_from_header(rr_header) or "Unknown"
                _unused_time_str = get_time_from_header(rr_header)
                base_filename = f"slot_{slot_idx:04d}_freq_{freq}MHz"

                # 保存RR图像
                rr_out = _save_single_pol_image(
                    rr_data,
                    rr_header,
                    output_dir,
                    cfg,
                    vmin,
                    vmax,
                    "RR",
                    base_filename,
                    sequence=sequence,
                    generated_at=generated_at,
                )
                # 保存LL图像
                ll_out = _save_single_pol_image(
                    ll_data,
                    ll_header,
                    output_dir,
                    cfg,
                    vmin,
                    vmax,
                    "LL",
                    base_filename,
                    sequence=sequence,
                    generated_at=generated_at,
                )
                individual_outputs.extend([rr_out, ll_out])

        else:
            # 普通模式：file_item是单个文件路径
            img_data, header = read_fits(file_item)
            contributing_headers = [header]
            artifact_source_files.append(file_item)
            polar_display = get_polar_from_header(header)
            source_file_for_diag = file_item
            if background_workflow_enabled(cfg):
                img_data_bgsub, background_map_for_slot, bg_diag_for_slot = (
                    subtract_radio_background(
                        img_data,
                        cfg,
                        source_file=source_file_for_diag,
                        band=get_freq_from_header(header) or "Unknown",
                        polarization=polar_display,
                    )
                )
            else:
                img_data_bgsub = img_data.copy()
                background_map_for_slot = None
                bg_diag_for_slot = _background_disabled_diag(source_file_for_diag)

        background_map_for_mask = None
        rms_map_for_mask = None
        background_diag_for_mask = {}
        strategy = str(
            cfg.get("radio_background_strategy", "noise_map_only") or "none"
        ).lower()
        if strategy in {"noise_map_only", "local_mesh"} or cfg.get(
            "background_use_for_mask", True
        ):
            background_map_for_mask, rms_map_for_mask, background_diag_for_mask = (
                estimate_background_rms_mesh(img_data, cfg)
            )
        if cfg.get("gaussian_fit_verbose", False):
            print(
                "[Background strategy] "
                f"strategy={strategy}, mask={bool(cfg.get('background_use_for_mask', True))}, "
                f"display={bool(cfg.get('background_use_for_display', False))}, "
                f"fit={bool(cfg.get('background_use_for_fit', False))}"
            )
        if background_diag_for_mask:
            bg_diag_for_slot = dict(bg_diag_for_slot)
            bg_diag_for_slot.update(background_diag_for_mask)

        if (
            cfg.get("background_use_for_display", False)
            and cfg.get("display_input_type", "raw")
            in {"excess", "background_subtracted"}
            and background_map_for_mask is not None
        ):
            img_data_display = img_data - background_map_for_mask
            if cfg.get("background_clip_negative_for_display_only", True):
                img_data_display = np.where(img_data_display < 0, 0.0, img_data_display)
        else:
            img_data_display = img_data

        all_data.append(img_data)
        all_bg_sub_data.append(img_data_bgsub)
        all_background_maps.append(background_map_for_slot)
        all_background_mask_maps.append(background_map_for_mask)
        all_rms_mask_maps.append(rms_map_for_mask)
        all_bg_diags.append(bg_diag_for_slot)
        all_headers.append(header)
        all_unit_headers.append(contributing_headers)
        extent, image_origin = calc_radio_extent_and_origin(header, img_data.shape, cfg)
        all_extents.append(extent)
        all_origins.append(image_origin)
        all_source_files.append(source_file_for_diag)
        all_raw_quality_bad.append(_raw_quality_item_is_bad(source_file_for_diag, cfg))
        band_info.append(
            (
                get_freq_from_header(header) or "Unknown",
                polar_display,
                get_time_from_header(header),
            )
        )

        # 对数化处理：将数据转换为对数坐标，安全处理非正值
        mask = img_data > 0
        display_base_data = img_data_display
        mask = display_base_data > 0
        log_data = np.full_like(display_base_data, np.nan, dtype=np.float64)
        log_data[mask] = np.log10(display_base_data[mask])
        all_log_data.append(log_data)

    n_bands = len(slot_files)
    if cfg["multi_band_layout"] == "auto":
        nrow, ncol = _layout_grid(n_bands)
    else:
        nrow, ncol = cfg["multi_band_layout"]

    # 使用用户配置的子图间距
    wspace = cfg.get("multi_band_wspace", 0.0)
    hspace = cfg.get("multi_band_hspace", 0.0)
    if cfg.get("multi_band_zero_gap", True):
        wspace = 0.0
        hspace = 0.0
    fig_width, fig_height = _auto_multi_band_figure_size(cfg, nrow, ncol, all_extents)
    spectrogram_enabled = _spectrogram_panel_enabled(cfg)
    layout_engine = str(cfg.get("multi_band_layout_engine", "gridspec") or "gridspec")
    if layout_engine == "manual_zero_gap":
        fig = plt.figure(figsize=(fig_width, fig_height))
        axes = _create_manual_radio_axes(
            fig, cfg, nrow, ncol, all_extents, spectrogram_enabled
        )
        if spectrogram_enabled:
            spec_left = float(cfg.get("radio_panel_left", 0.055))
            spec_right = float(cfg.get("radio_panel_right", 0.985))
            spec_bottom = max(0.055, float(cfg.get("radio_grid_bottom", 0.30)) * 0.18)
            spec_top = (
                float(cfg.get("radio_panel_bottom", 0.34))
                - float(cfg.get("spectrogram_hspace", 0.14)) * 0.18
            )
            spec_top = min(max(spec_top, spec_bottom + 0.08), 0.32)
            spectrogram_ax = fig.add_axes(
                [spec_left, spec_bottom, spec_right - spec_left, spec_top - spec_bottom]
            )
        else:
            spectrogram_ax = None
    elif spectrogram_enabled:
        panel_ratio = float(cfg.get("spectrogram_panel_height_ratio", 0.34))
        fig = plt.figure(figsize=(fig_width, fig_height))
        outer_gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[1.0, panel_ratio],
            hspace=float(cfg.get("spectrogram_hspace", 0.08)),
        )
        top_gs = outer_gs[0, 0].subgridspec(nrow, ncol, wspace=wspace, hspace=hspace)
        axes = np.array(
            [
                [fig.add_subplot(top_gs[row, col]) for col in range(ncol)]
                for row in range(nrow)
            ]
        )
        spectrogram_ax = fig.add_subplot(outer_gs[1, 0])
    else:
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(fig_width, fig_height),
            dpi=(
                int(cfg["dpi"])
                if cfg.get("_artifact_fixed_panel_layout", False)
                else None
            ),
        )
        plt.subplots_adjust(
            wspace=wspace,
            hspace=hspace,
            left=float(cfg.get("radio_grid_left", 0.06)),
            right=float(cfg.get("radio_grid_right", 0.98)),
            top=float(cfg.get("radio_grid_top", 0.92)),
            bottom=max(float(cfg.get("radio_grid_bottom", 0.08)), 0.08),
        )
        spectrogram_ax = None

        # 将axes转换为二维数组以便索引
        if nrow == 1 and ncol == 1:
            axes = np.array([[axes]])
        elif nrow == 1:
            axes = axes.reshape(1, -1)
        elif ncol == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(nrow, ncol)

    # 使用预先计算的固定波段颜色范围
    if (
        "fixed_band_vmins" in cfg
        and "fixed_band_vmaxs" in cfg
        and not background_enabled_for_display(cfg)
    ):
        band_vmins = cfg["fixed_band_vmins"]
        band_vmaxs = cfg["fixed_band_vmaxs"]
    else:
        # 如果没有预先计算的范围，则按原有方法计算（兼容性）
        band_vmins, band_vmaxs = _calculate_per_band_ranges(all_log_data, cfg)

    # 计算整体对数范围（用于统一颜色条时）
    all_valid = np.concatenate(
        [d[~np.isnan(d)] for d in all_log_data if len(d[~np.isnan(d)]) > 0]
    )
    if len(all_valid) > 0:
        global_low, global_high = _calculate_range(all_valid, cfg, is_global=True)
    else:
        global_low, global_high = 0, 1
    band_vmins, band_vmaxs = _resolve_multi_band_display_ranges(
        all_log_data, cfg, vmin=vmin, vmax=vmax
    )
    global_low, global_high = band_vmins[0], band_vmaxs[0]

    panel_metadata = []
    artifact_warnings = []
    for idx in range(n_bands):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        freq, polar, band_time = band_info[idx]
        unit_resolution = resolve_colorbar_unit(
            all_unit_headers[idx], cfg.get("radio_colorbar_unit")
        )
        radio_tick_notation = tick_notation_for_transform("log10")
        radio_colorbar_label = colorbar_label(
            unit_resolution,
            transform="log10",
            tick_notation=radio_tick_notation,
        )
        for message in unit_resolution.warnings:
            if message not in artifact_warnings:
                artifact_warnings.append(message)
                warnings.warn(message, stacklevel=2)
        gaussian_cfg = config_for_gaussian_band(cfg, freq)
        gaussian_cfg["_current_radio_image_origin"] = all_origins[idx]
        rsun_obs = all_headers[idx].get("RSUN_OBS", 960.0)

        # 使用对数化数据
        log_data = all_log_data[idx]

        current_cmap = _radio_colormap(cfg)

        im_kwargs = get_imshow_kwargs(
            log_data,
            all_extents[idx],
            cfg,
            cmap=current_cmap,
            aspect="auto",
            origin=all_origins[idx],
        )

        # 为每个波段设置合适的颜色范围
        if cfg.get("use_per_band_colormap", True):
            # 使用每个波段自己的合适范围
            vmin_band, vmax_band = band_vmins[idx], band_vmaxs[idx]
        else:
            # 使用全局范围
            vmin_band, vmax_band = global_low, global_high

        im_kwargs["vmin"] = vmin_band
        im_kwargs["vmax"] = vmax_band

        im = ax.imshow(log_data, **im_kwargs)
        add_radio_coordinate_corner_debug(
            ax, all_extents[idx], log_data.shape, all_origins[idx], cfg
        )
        aspect_mode = cfg.get("multi_band_aspect_mode", "equal_compact")
        if aspect_mode == "fill":
            ax.set_aspect("auto")
        elif aspect_mode in {"equal", "equal_compact"}:
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")
        if cfg.get("save_background_diagnostics", True) and background_workflow_enabled(
            cfg
        ):
            save_background_diagnostics_row(
                _background_diagnostics_row(
                    all_bg_diags[idx],
                    cfg,
                    all_data[idx],
                    all_bg_sub_data[idx],
                    freq,
                    band_time,
                    polar,
                ),
                output_dir,
                cfg,
            )
        if (
            background_workflow_enabled(cfg)
            or cfg.get("save_background_subtracted_image", False)
            or cfg.get("save_estimated_background_map", False)
        ):
            _save_background_products(
                all_bg_sub_data[idx],
                all_background_maps[idx],
                all_extents[idx],
                output_dir,
                f"slot_{slot_idx:04d}_{freq}MHz_{polar}",
                cfg,
                all_origins[idx],
                colorbar_label(unit_resolution, transform="linear"),
                observation_time=_parse_datetime_value(band_time),
                frequency_mhz=freq,
                polarization=polar,
                sequence=(sequence - 1) * max(1, len(all_bg_sub_data)) + idx + 1,
                generated_at=generated_at,
            )
        if cfg.get("enable_gaussian_overlay", False) and not all_raw_quality_bad[idx]:
            multi_fit_result = None
            if gaussian_cfg.get("background_use_for_fit", False):
                fit_base_data = all_bg_sub_data[idx]
                fit_input_type = "background_subtracted"
            else:
                fit_base_data = all_data[idx]
                fit_input_type = "raw"
            if gaussian_cfg.get("gaussian_fit_verbose", False):
                print(
                    "[Gaussian input] using "
                    f"{'background-subtracted radio image' if gaussian_cfg.get('background_use_for_fit', False) else 'raw radio image'} "
                    "with local baseline model"
                )
            if _gaussian_multi_source_enabled(gaussian_cfg):
                multi_fit_result = _fit_multiple_gaussians_on_radio_image(
                    fit_base_data,
                    extent=all_extents[idx],
                    cfg=gaussian_cfg,
                    source_file=all_source_files[idx],
                    background_map=all_background_mask_maps[idx],
                    rms_map=all_rms_mask_maps[idx],
                    fit_input_type=fit_input_type,
                    image_origin=all_origins[idx],
                )
                fit_result = multi_fit_result.primary_result
            else:
                fit_result = fit_elliptical_gaussian_on_radio_image(
                    fit_base_data,
                    extent=all_extents[idx],
                    cfg=gaussian_cfg,
                    source_file=all_source_files[idx],
                    background_map=all_background_mask_maps[idx],
                    rms_map=all_rms_mask_maps[idx],
                    fit_input_type=fit_input_type,
                    image_origin=all_origins[idx],
                )
                _attach_raw_peak_center(
                    fit_result,
                    all_data[idx],
                    all_extents[idx],
                    gaussian_cfg,
                    all_origins[idx],
                )
            if multi_fit_result is not None:
                _overlay_multi_gaussian_fit_on_axis(
                    ax,
                    multi_fit_result,
                    all_extents[idx],
                    fit_base_data.shape,
                    gaussian_cfg,
                )
            elif fit_result is not None:
                overlay_gaussian_fit_on_axis(
                    ax, fit_result, all_extents[idx], fit_base_data.shape, gaussian_cfg
                )
            if cfg.get("save_gaussian_diagnostics", True):
                save_gaussian_diagnostics_row(
                    _gaussian_diagnostics_row(
                        fit_result,
                        gaussian_cfg,
                        freq,
                        band_time,
                        polar,
                        all_bg_diags[idx],
                    ),
                    output_dir,
                    cfg,
                )
                if multi_fit_result is not None:
                    for diagnostic_row in _multi_gaussian_diagnostics_rows(
                        multi_fit_result,
                        gaussian_cfg,
                        freq,
                        band_time,
                        polar,
                        all_bg_diags[idx],
                    ):
                        _save_multi_gaussian_diagnostics_row(
                            diagnostic_row, output_dir, cfg
                        )

        ax.add_patch(
            patches.Circle(
                (0, 0),
                radius=rsun_obs,
                edgecolor="white",
                facecolor="none",
                linewidth=1.5,
            )
        )

        if cfg.get("use_custom_lim", False):
            ax.set_xlim(cfg["custom_xlim"])
            ax.set_ylim(cfg["custom_ylim"])
        else:
            sf = cfg["scale_factor"]
            ax.set_xlim(-rsun_obs * sf, rsun_obs * sf)
            ax.set_ylim(-rsun_obs * sf, rsun_obs * sf)
        if cfg.get("radio_force_exact_xlim_ylim", True):
            if cfg.get("use_custom_lim", False):
                ax.set_xlim(cfg["custom_xlim"])
                ax.set_ylim(cfg["custom_ylim"])
            else:
                sf = cfg["scale_factor"]
                ax.set_xlim(-rsun_obs * sf, rsun_obs * sf)
                ax.set_ylim(-rsun_obs * sf, rsun_obs * sf)
        if cfg.get("radio_remove_axis_margins", True):
            ax.margins(0)
            ax.set_xmargin(0)
            ax.set_ymargin(0)

        # 在子图中标注频率（替代标题）
        freq_text = f"{freq} MHz"
        ax.text(
            0.02,
            0.98,
            freq_text,
            transform=ax.transAxes,
            fontsize=cfg["title_fontsize"] - 6,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.7,
                edgecolor="white",
            ),
        )

        # 坐标轴标签设置：只在最左边一列显示Y轴，最下面一行显示X轴
        if col == 0:  # 最左边一列
            ax.set_ylabel("y (arcsec)", fontsize=cfg["label_fontsize"] - 6)
        else:
            ax.set_ylabel("")
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                labelleft=False,
                colors=cfg.get("tick_color", "yellow"),
            )

        if row == nrow - 1:  # 最下面一行
            ax.set_xlabel("x (arcsec)", fontsize=cfg["label_fontsize"] - 6)
        else:
            ax.set_xlabel("")
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                labelbottom=False,
                colors=cfg.get("tick_color", "yellow"),
            )

        # 调整刻度标签大小
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=cfg["tick_fontsize"] - 8,
            colors=cfg.get("tick_color", "yellow"),
        )

        # 设置坐标轴刻度步长（新增代码）
        hide_inner_ticks = cfg.get("hide_inner_ticks", True)
        x_tick_step = cfg.get("x_tick_step", 200)
        y_tick_step = cfg.get("y_tick_step", 200)
        tick_rotation = cfg.get("tick_label_rotation", 0)

        # 获取当前坐标轴范围
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

        # 计算x轴刻度位置
        if x_tick_step and x_tick_step > 0:
            x_start = math.ceil(current_xlim[0] / x_tick_step) * x_tick_step
            x_end = math.floor(current_xlim[1] / x_tick_step) * x_tick_step
            x_ticks = np.arange(x_start, x_end + x_tick_step / 2, x_tick_step)
            ax.set_xticks(x_ticks)

            # 隐藏内部子图的x轴刻度标签（如果配置要求）
            if hide_inner_ticks and row < nrow - 1:
                ax.set_xticklabels([])

        # 计算y轴刻度位置
        if y_tick_step and y_tick_step > 0:
            y_start = math.ceil(current_ylim[0] / y_tick_step) * y_tick_step
            y_end = math.floor(current_ylim[1] / y_tick_step) * y_tick_step
            y_ticks = np.arange(y_start, y_end + y_tick_step / 2, y_tick_step)
            ax.set_yticks(y_ticks)

            # 隐藏内部子图的y轴刻度标签（如果配置要求）
            if hide_inner_ticks and col > 0:
                ax.set_yticklabels([])

        # 应用刻度标签旋转
        if tick_rotation != 0:
            # x轴刻度标签旋转（只对底部行应用）
            if row == nrow - 1:
                ax.tick_params(
                    axis="x", rotation=tick_rotation, labelrotation=tick_rotation
                )

            # y轴刻度标签旋转（只对最左列应用）
            if col == 0:
                ax.tick_params(
                    axis="y", rotation=tick_rotation, labelrotation=tick_rotation
                )

        # 为每个子图添加嵌入式颜色条，使用用户配置的位置
        _set_compact_radio_ticks(ax, cfg)
        _apply_compact_radio_axis_style(ax, row, col, nrow, ncol, cfg)
        _prune_edge_ticklabels(ax, row, col, nrow, ncol, cfg)

        colorbar_pos = cfg.get("colorbar_position", [0.75, 0.05, 0.22, 0.03])

        # 确保颜色条完全在子图内部
        cax = ax.inset_axes(colorbar_pos)  # [x, y, width, height] 相对于子图内部
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.locator_params(nbins=3)
        apply_colorbar_tick_notation(cbar, transform="log10")
        cbar.set_label(
            radio_colorbar_label,
            fontsize=max(cfg["tick_fontsize"] - 10, 8),
        )
        cbar.ax.tick_params(
            labelsize=cfg["tick_fontsize"] - 10,
            colors=cfg.get("colorbar_tick_color", "yellow"),
        )
        # 添加颜色条刻度旋转（新增代码）
        tick_rotation = cfg.get("tick_label_rotation", 0)
        if tick_rotation != 0:
            cbar.ax.tick_params(axis="x", rotation=tick_rotation)
        # cbar.set_label('log10(I)', fontsize=cfg["tick_fontsize"] - 10, colors='y')
        cbar.ax.locator_params(nbins=3)
        panel_metadata.append(
            {
                "id": f"radio-{idx + 1}",
                "frequency_mhz": (
                    float(freq) if isinstance(freq, (int, float)) else None
                ),
                "polarization": str(polar),
                "transform": "log10",
                "tick_notation": radio_tick_notation,
                "colorbar_label": radio_colorbar_label,
                "unit": unit_resolution.to_dict(),
            }
        )

    # 隐藏多余的子图
    for idx in range(n_bands, nrow * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].axis("off")
        axes[row, col].set_visible(False)

    main_time = band_info[0][2] if band_info else "Unknown"
    current_frame_time = (
        _radio_item_datetime(all_source_files[0], cfg) if all_source_files else None
    )
    if current_frame_time is None:
        current_frame_time = _parse_datetime_value(main_time)

    # 确定显示名称
    if combine_polarizations and polarization == "RR+LL":
        if cfg.get("weighted_average", False):
            rr_weight = cfg.get("rr_weight", 0.5)
            ll_weight = cfg.get("ll_weight", 0.5)
            polar_display = f"RR+LL Combined (weighted: RR={rr_weight}, LL={ll_weight})"
        else:
            polar_display = "RR+LL Combined (sum)"
    elif polarization == "RR":
        polar_display = "Right Circular"
    elif polarization == "LL":
        polar_display = "Left Circular"
    else:
        polar_display = polarization

    # 添加总标题
    fig.suptitle(
        f"Multi-band Radio Synthesis ({polar_display}) - {main_time}",
        fontsize=cfg["title_fontsize"] + 2,
        fontweight="bold",
        y=0.98,
    )

    if spectrogram_ax is not None:
        overlay_spectrogram_panel(spectrogram_ax, cfg, current_frame_time)

    if cfg.get("multi_band_zero_gap", True):
        for radio_ax in axes.ravel():
            if radio_ax.get_visible():
                radio_ax.margins(0)
                radio_ax.set_anchor("C")
    _add_global_radio_axis_labels(fig, axes, cfg, spectrogram_ax=spectrogram_ax)

    # 进一步调整布局
    # 使用tight_layout确保布局紧凑，但保留足够的空间给标题
    # plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Resolve the real save location lazily so unused root-level folders stay absent.
    if any(all_raw_quality_bad):
        overlay_output_dir = _raw_quality_bad_frame_output_dir(
            output_dir,
            cfg,
            _multi_band_output_subdir(cfg),
        )
    else:
        overlay_output_dir = _resolve_multi_band_output_dir(output_dir, cfg)
    suffix_tokens = tuple(
        token for token in _output_suffix(cfg).strip("_").split("_") if token
    )
    output_path = overlay_output_dir / build_scientific_image_filename(
        sequence=sequence,
        start_time=current_frame_time,
        instrument="radio",
        polarization=str(cfg.get("polarization") or polarization),
        product="multi_band_source_map",
        qualifiers=suffix_tokens,
        generated_at=generated_at,
    )

    if cfg["save_plot"]:
        spatial_display = resolve_spatial_display(cfg, transform="log10")
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        should_write_sidecar = (
            bool(cfg.get("write_source_map_sidecar", False))
            if write_sidecar is None
            else bool(write_sidecar)
        )
        artifact_warnings.extend(cfg.get("_spectrogram_unit_warnings", []))
        visible_radio_axes = [
            axes[index // ncol, index % ncol] for index in range(n_bands)
        ]
        save_figure_artifact(
            fig,
            output_path,
            dpi=cfg["dpi"],
            radio_axes=visible_radio_axes,
            panel_metadata=panel_metadata,
            mode="multi_band",
            polarization=str(polarization),
            source_files=list(
                dict.fromkeys(str(path) for path in artifact_source_files)
            ),
            write_sidecar=should_write_sidecar,
            warnings=artifact_warnings,
            display=spatial_display.sidecar_payload(),
            bbox_inches=cfg.get("_artifact_bbox_inches", "tight"),
            pad_inches=float(cfg.get("_artifact_pad_inches", 0.1)),
        )

    if cfg["show_plot"] and cfg.get("_interactive", False):
        plt.show()

    plt.close(fig)

    # 如果保存了单独的偏振图像，打印信息
    if save_individual and individual_outputs:
        print(
            f"Slot {slot_idx}: Saved {len(individual_outputs)} individual polarization images"
        )

    return str(output_path)


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────


def _migrate_config(cfg):
    """Backward compatibility: migrate old use_fixed_cbar configuration to new color_range_mode"""
    if "use_fixed_cbar" in cfg:
        use_fixed_cbar = cfg.pop("use_fixed_cbar")
        if use_fixed_cbar:
            if cfg.get("fixed_vmin") is not None or cfg.get("fixed_vmax") is not None:
                cfg["color_range_mode"] = "fixed"
            else:
                cfg["color_range_mode"] = "global"
        else:
            cfg["color_range_mode"] = "auto"
        print(
            f"Migrated old config: use_fixed_cbar={use_fixed_cbar} -> "
            f"color_range_mode={cfg['color_range_mode']}"
        )
    if "drift_diagnostics_csv" in cfg and "drift_rate_diagnostics_csv" not in cfg:
        cfg["drift_rate_diagnostics_csv"] = cfg["drift_diagnostics_csv"]
    return cfg


def _legacy_check_coordinate_roundtrip():
    shape = (256, 256)
    extent = [-3000.0, 3000.0, -3000.0, 3000.0]
    for x_pix, y_pix in [(0, 0), (127.5, 127.5), (255, 255), (64.25, 200.75)]:
        err = coordinate_roundtrip_error_pixel(x_pix, y_pix, extent, shape)
        if err >= 1e-6 and err >= 0.01:
            raise AssertionError(f"coordinate roundtrip error too large: {err}")
    return True


def _legacy_check_gaussian_center_overlay_consistency():
    cfg = dict(CONFIG)
    cfg.update(
        {
            "fit_use_source_mask": False,
            "gaussian_fit_use_roi": False,
            "gaussian_fit_fallback_to_moment": False,
            "gaussian_fit_max_pixels": 0,
            "fit_min_mask_pixels": 10,
            "fit_snr_threshold": 1.0,
            "fit_peak_fraction_threshold": 0.05,
            "gaussian_quality_requirements": {"require_quality_ok": False},
        }
    )
    shape = (96, 128)
    x0, y0 = 70.25, 34.75
    y, x = np.indices(shape, dtype=np.float64)
    data = 2.0 + 80.0 * np.exp(-0.5 * (((x - x0) / 7.0) ** 2 + ((y - y0) / 5.0) ** 2))
    extent = [-3000.0, 3000.0, -3000.0, 3000.0]
    result = fit_elliptical_gaussian_on_radio_image(data, extent, cfg)
    if result is None:
        raise AssertionError("synthetic gaussian fit failed")
    if math.hypot(result.center_pixel[0] - x0, result.center_pixel[1] - y0) > 0.5:
        raise AssertionError(f"fitted center drifted: {result.center_pixel}")
    x_arcsec, y_arcsec = pixel_to_data_coord(
        result.center_pixel[0], result.center_pixel[1], extent, shape
    )
    x_back, y_back = data_coord_to_pixel(x_arcsec, y_arcsec, extent, shape)
    if (
        math.hypot(x_back - result.center_pixel[0], y_back - result.center_pixel[1])
        > 1e-6
    ):
        raise AssertionError("overlay coordinate conversion is not reversible")
    return True


def _legacy_check_diagnostic_csv_schema():
    import tempfile

    from solar_toolkit.radio.output_paths import plot_output_subdir

    cfg = dict(CONFIG)
    cfg["gaussian_diagnostics_csv"] = "radio_gaussian_fit_diagnostics.csv"
    cfg["multi_band_output_subdir"] = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        row_ok = {name: "" for name in GAUSSIAN_DIAGNOSTIC_FIELDS}
        row_ok.update({"source_file": "ok.fits", "quality_flag": "ok"})
        row_fail = {
            "source_file": "fail.fits",
            "reason": "fit_failed",
            "quality_flag": "fit_failed",
            "fit_input_type": "raw",
        }
        save_gaussian_diagnostics_row(row_ok, tmpdir, cfg)
        save_gaussian_diagnostics_row(row_fail, tmpdir, cfg)
        csv_path = os.path.join(
            tmpdir,
            plot_output_subdir(cfg),
            cfg["gaussian_diagnostics_csv"],
        )
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if list(df.columns) != GAUSSIAN_DIAGNOSTIC_FIELDS or len(df) != 2:
                raise AssertionError("diagnostic CSV schema mismatch")
        except ImportError:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            if len({len(row) for row in rows}) != 1:
                raise AssertionError("diagnostic CSV row lengths differ") from None
    return True


def _legacy_check_drift_manual_endpoint_calculation():
    line = {
        "label": "drift_001",
        "t_start": "2025-01-24T04:48:50",
        "f_start_mhz": 230.0,
        "t_end": "2025-01-24T04:48:56",
        "f_end_mhz": 170.0,
        "color": "white",
    }
    result = calculate_drift_rate_from_line(line)
    if abs(result.drift_rate_mhz_s - (-10.0)) > 1e-9:
        raise AssertionError(f"unexpected drift rate: {result.drift_rate_mhz_s}")
    return True


def _legacy_check_drift_selection_json_roundtrip():
    import tempfile

    cache = SpectrogramCache(
        data=np.zeros((2, 2), dtype=np.float32),
        time_nums=np.array([0.0, 1.0]),
        display_time_nums=(
            mdates.date2num(datetime.datetime(2025, 1, 24, 4, 48, 50)),
            mdates.date2num(datetime.datetime(2025, 1, 24, 4, 49, 0)),
        ),
        time_datetimes=[],
        freq=np.array([100.0, 300.0]),
        title="test",
        cmap="jet",
        vmin=None,
        vmax=None,
        cbar_label="",
        source_file="test.fits",
    )
    lines = [
        {
            "label": "drift_001",
            "t_start": "2025-01-24T04:48:50",
            "f_start_mhz": 230.0,
            "t_end": "2025-01-24T04:48:56",
            "f_end_mhz": 170.0,
            "color": "white",
            "note": "",
        },
        {
            "label": "drift_002",
            "t_start": "2025-01-24T04:48:52",
            "f_start_mhz": 210.0,
            "t_end": "2025-01-24T04:48:58",
            "f_end_mhz": 160.0,
            "color": "cyan",
            "note": "",
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "selection.json")
        save_drift_selection_json(path, lines, cache, CONFIG)
        loaded = load_drift_selection_json(path)
    if len(loaded) != 2:
        raise AssertionError("drift selection JSON line count mismatch")
    if loaded[0]["t_start"] != lines[0]["t_start"]:
        raise AssertionError("drift selection JSON time mismatch")
    if float(loaded[1]["f_end_mhz"]) != 160.0:
        raise AssertionError("drift selection JSON frequency mismatch")
    return True


def _legacy_check_pixel_to_spectrogram_coord_mapping():
    t0 = datetime.datetime(2025, 1, 24, 4, 48, 50)
    t1 = t0 + datetime.timedelta(seconds=10)
    metadata = {
        "axes_bbox_px": {"left": 100, "right": 900, "top": 100, "bottom": 500},
        "x_start_num": mdates.date2num(t0),
        "x_end_num": mdates.date2num(t1),
        "f_min_mhz": 100.0,
        "f_max_mhz": 300.0,
    }
    upper_left = _spectrogram_coord_from_pixel(metadata, 100, 100)
    lower_right = _spectrogram_coord_from_pixel(metadata, 900, 500)
    if abs((_parse_datetime_value(upper_left["time_iso"]) - t0).total_seconds()) > 0.01:
        raise AssertionError("left edge time mapping failed")
    if abs(upper_left["frequency_mhz"] - 300.0) > 1e-9:
        raise AssertionError("top edge frequency mapping failed")
    if (
        abs((_parse_datetime_value(lower_right["time_iso"]) - t1).total_seconds())
        > 0.01
    ):
        raise AssertionError("right edge time mapping failed")
    if abs(lower_right["frequency_mhz"] - 100.0) > 1e-9:
        raise AssertionError("bottom edge frequency mapping failed")
    return True


def _legacy_check_radio_wcs_orientation_positive_cdelt2():
    header = {
        "CRVAL1": 0.0,
        "CRPIX1": 1.0,
        "CDELT1": 1.0,
        "CRVAL2": 0.0,
        "CRPIX2": 1.0,
        "CDELT2": 1.0,
    }
    cfg = {
        "preserve_fits_wcs_orientation": True,
        "radio_image_origin_mode": "auto",
    }
    extent, origin = calc_radio_extent_and_origin(header, (10, 10), cfg)
    if origin != "lower":
        raise AssertionError("preserved FITS orientation should use lower origin")
    _x, y = pixel_to_data_coord(0, 0, extent, (10, 10), origin)
    if abs(y - 0.0) > 1e-9:
        raise AssertionError(f"positive CDELT2 row-0 mapping failed: {y}")
    return True


def _legacy_check_radio_wcs_orientation_negative_cdelt2():
    header = {
        "CRVAL1": 0.0,
        "CRPIX1": 1.0,
        "CDELT1": 1.0,
        "CRVAL2": 0.0,
        "CRPIX2": 1.0,
        "CDELT2": -1.0,
    }
    cfg = {
        "preserve_fits_wcs_orientation": True,
        "radio_image_origin_mode": "auto",
    }
    extent, origin = calc_radio_extent_and_origin(header, (10, 10), cfg)
    _x, y = pixel_to_data_coord(0, 0, extent, (10, 10), origin)
    if abs(y - 0.0) > 1e-9:
        raise AssertionError(f"negative CDELT2 row-0 mapping failed: {y}")
    _x2, y2 = pixel_to_data_coord(0, 1, extent, (10, 10), origin)
    if y2 >= y:
        raise AssertionError("negative CDELT2 should decrease y with row index")
    return True


def _legacy_check_spectrogram_mapping_not_flipped():
    metadata = {
        "axes_bbox_px": {"left": 100, "right": 900, "top": 100, "bottom": 500},
        "x_start_num": mdates.date2num(datetime.datetime(2025, 1, 24, 4, 48, 50)),
        "x_end_num": mdates.date2num(datetime.datetime(2025, 1, 24, 4, 49, 0)),
        "f_min_mhz": 100.0,
        "f_max_mhz": 300.0,
    }
    assert_spectrogram_mapping_not_flipped(metadata)
    return True


def _legacy_check_drift_preview_line_js_present():
    html = _drift_selection_html(
        {
            "fig_width_px": 1000,
            "fig_height_px": 600,
            "axes_bbox_px": {"left": 100, "right": 900, "top": 100, "bottom": 500},
            "x_start_num": 1.0,
            "x_end_num": 2.0,
            "x_start_unix_ms": 0,
            "x_end_unix_ms": 10000,
            "f_min_mhz": 100.0,
            "f_max_mhz": 300.0,
        },
        {"show_preview_line": True, "show_crosshair": True},
    )
    for token in (
        "currentMouse",
        "drawPreviewLine",
        "mouseleave",
        "interactive.show_preview_line",
    ):
        if token not in html:
            raise AssertionError(f"missing JS preview token: {token}")
    return True


def _legacy_check_drift_launch_policy_logic():
    cache = SpectrogramCache(
        data=np.zeros((2, 2), dtype=np.float32),
        time_nums=np.array([0.0, 1.0]),
        display_time_nums=(
            mdates.date2num(datetime.datetime(2025, 1, 24, 4, 48, 50)),
            mdates.date2num(datetime.datetime(2025, 1, 24, 4, 49, 0)),
        ),
        time_datetimes=[],
        freq=np.array([100.0, 300.0]),
        title="test",
        cmap="jet",
        vmin=None,
        vmax=None,
        cbar_label="",
        source_file="test.fits",
    )

    def fake_launch(_cache, _cfg):
        calls.append(1)
        return [
            {
                "label": "drift_001",
                "t_start": "2025-01-24T04:48:50",
                "f_start_mhz": 230.0,
                "t_end": "2025-01-24T04:48:56",
                "f_end_mhz": 170.0,
                "color": "white",
            }
        ]

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = dict(CONFIG)
        cfg.update(
            {
                "enable_drift_rate_overlay": True,
                "drift_rate_mode": "interactive_manual",
                "drift_rate_selection_json": os.path.join(tmpdir, "missing.json"),
                "drift_rate_interactive": {
                    "launch_policy": "cli_only",
                    "print_usage_hint": False,
                },
            }
        )
        calls = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            results = get_or_load_drift_rate_results(
                cache, cfg, launch_func=fake_launch
            )
        if results or calls:
            raise AssertionError("cli_only should not launch without --select-drift")
        if not any("selection JSON" in str(item.message) for item in caught):
            raise AssertionError("missing-selection warning was not emitted")

        cfg["drift_rate_interactive"]["launch_policy"] = "auto_if_missing"
        calls = []
        results = get_or_load_drift_rate_results(cache, cfg, launch_func=fake_launch)
        if len(results) != 1 or len(calls) != 1:
            raise AssertionError("auto_if_missing should launch exactly once")

        existing = os.path.join(tmpdir, "existing.json")
        save_drift_selection_json(existing, [], cache, cfg)
        cfg["drift_rate_selection_json"] = existing
        cfg["drift_rate_interactive"]["launch_policy"] = "always"
        calls = []
        results = get_or_load_drift_rate_results(cache, cfg, launch_func=fake_launch)
        if len(results) != 1 or len(calls) != 1:
            raise AssertionError("always should launch even when JSON exists")
    return True


def _legacy_check_frontend_time_mapping_metadata():
    t0 = datetime.datetime(2025, 1, 24, 4, 48, 50)
    t1 = t0 + datetime.timedelta(seconds=10)
    metadata = {
        "axes_bbox_px": {"left": 100, "right": 900, "top": 100, "bottom": 500},
        "x_start_num": mdates.date2num(t0),
        "x_end_num": mdates.date2num(t1),
        "x_start_unix_ms": int(t0.timestamp() * 1000),
        "x_end_unix_ms": int(t1.timestamp() * 1000),
        "f_min_mhz": 100.0,
        "f_max_mhz": 300.0,
    }
    mid = _spectrogram_coord_from_pixel(metadata, 500, 300)
    mid_dt = _parse_datetime_value(mid["time_iso"])
    if abs((mid_dt - (t0 + datetime.timedelta(seconds=5))).total_seconds()) > 0.01:
        raise AssertionError("midpoint time mapping failed")
    frontend_unix_ms = metadata["x_start_unix_ms"] + 0.5 * (
        metadata["x_end_unix_ms"] - metadata["x_start_unix_ms"]
    )
    if abs(frontend_unix_ms - int((t0.timestamp() + 5) * 1000)) > 1:
        raise AssertionError("frontend unix-ms interpolation failed")
    return True


def _legacy_check_selection_preview_png_size_matches_metadata():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = dict(CONFIG)
        cfg.update(
            {
                "output_dir": tmpdir,
                "drift_rate_selection_preview_png": "preview.png",
                "drift_rate_selection_metadata_json": "preview_meta.json",
            }
        )
        cache = SpectrogramCache(
            data=np.zeros((16, 16), dtype=np.float32),
            time_nums=np.linspace(0, 1, 16),
            display_time_nums=(
                mdates.date2num(datetime.datetime(2025, 1, 24, 4, 48, 50)),
                mdates.date2num(datetime.datetime(2025, 1, 24, 4, 49, 0)),
            ),
            time_datetimes=[],
            freq=np.linspace(100, 300, 16),
            title="test",
            cmap="jet",
            vmin=None,
            vmax=None,
            cbar_label="",
            source_file="test.fits",
        )
        png_path, metadata = render_spectrogram_selection_preview(cache, cfg)
        img = plt.imread(png_path)
        height, width = img.shape[:2]
        if width != metadata["fig_width_px"] or height != metadata["fig_height_px"]:
            raise AssertionError("preview PNG size does not match metadata")
    return True


def run_self_tests():
    tests = [
        _legacy_check_coordinate_roundtrip,
        _legacy_check_gaussian_center_overlay_consistency,
        _legacy_check_diagnostic_csv_schema,
        _legacy_check_drift_manual_endpoint_calculation,
        _legacy_check_drift_selection_json_roundtrip,
        _legacy_check_pixel_to_spectrogram_coord_mapping,
        _legacy_check_radio_wcs_orientation_positive_cdelt2,
        _legacy_check_radio_wcs_orientation_negative_cdelt2,
        _legacy_check_spectrogram_mapping_not_flipped,
        _legacy_check_drift_preview_line_js_present,
        _legacy_check_drift_launch_policy_logic,
        _legacy_check_frontend_time_mapping_metadata,
        _legacy_check_selection_preview_png_size_matches_metadata,
    ]
    for test in tests:
        test()
        print(f"[self-test] {test.__name__}: ok")
    print("[self-test] all checks passed")


def _run_select_drift_workflow(cfg):
    cfg["enable_spectrogram_panel"] = True
    cfg["enable_drift_rate_overlay"] = True
    cfg["_select_drift_now"] = True
    cache = build_spectrogram_cache(cfg)
    if cache is None:
        raise RuntimeError("Could not build spectrogram cache for drift selection")
    render_spectrogram_selection_preview(cache, cfg)
    lines = launch_drift_selection_server(cache, cfg)
    results = [calculate_drift_rate_from_line(line) for line in lines]
    save_drift_rate_diagnostics_once(results, cfg, cache.source_file)
    for result in results:
        print(
            f"{result.label}: {result.f_start_mhz:.1f} -> {result.f_end_mhz:.1f} MHz, "
            f"duration={result.duration_s:.1f} s, "
            f"df/dt={result.drift_rate_mhz_s:.2f} MHz/s"
        )


def _run_source_map_workflow(user_config=None, *, argv=None):
    """
    Main function: process single-band or multi-band radio data according to configuration mode, parallel plotting and saving results.
    """
    global CONFIG, USER_CONFIG
    if user_config is not None:
        USER_CONFIG = dict(user_config or {})
        path_config = load_script_config(
            "radio_source_map_plot_gaussian_overlay",
            DEFAULT_CONFIG,
        )
        CONFIG = build_config(USER_CONFIG, path_config)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--select-drift", action="store_true")
    parser.add_argument("--use-drift-selection")
    parser.add_argument("--drift-port", type=int)
    parser.add_argument("--no-drift-browser", action="store_true")
    parser.add_argument(
        "--drift-launch-policy",
        choices=("cli_only", "auto_if_missing", "always"),
    )
    parser.add_argument("--disable-drift", action="store_true")
    parser.add_argument("--enable-drift", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    cfg = CONFIG
    cfg = _migrate_config(cfg)
    if args.disable_drift:
        cfg["enable_drift_rate_overlay"] = False
        cfg["drift_rate_mode"] = "off"
    if args.enable_drift:
        cfg["enable_drift_rate_overlay"] = True
        if str(cfg.get("drift_rate_mode", "off")).lower() == "off":
            cfg["drift_rate_mode"] = "interactive_manual"
    if args.use_drift_selection:
        cfg["_drift_selection_cli_path"] = args.use_drift_selection
        cfg["enable_drift_rate_overlay"] = True
        cfg["drift_rate_mode"] = "manual_json"
    if args.drift_port is not None:
        cfg.setdefault("drift_rate_interactive", {})["port"] = int(args.drift_port)
    if args.no_drift_browser:
        cfg.setdefault("drift_rate_interactive", {})["auto_open_browser"] = False
    if args.drift_launch_policy:
        cfg.setdefault("drift_rate_interactive", {})[
            "launch_policy"
        ] = args.drift_launch_policy
    if args.self_test or cfg.get("run_self_test", False):
        matplotlib.use("Agg")
        run_self_tests()
        return
    if args.select_drift:
        matplotlib.use("Agg")
        _run_select_drift_workflow(cfg)
        return
    mode = cfg.get("mode", "single_band")
    workspace_selection = _workspace_source_map_selection(cfg)
    workflow = resolve_background_workflow(cfg)
    print(
        f"[Colormap] radio_cmap={cfg.get('radio_cmap', cfg.get('cmap'))}, "
        f"spectrogram_cmap={cfg.get('spectrogram_cmap')}"
    )
    print(
        f"[Background] workflow={workflow}, "
        f"mode={cfg['radio_background_subtraction_mode']}, "
        f"display={background_enabled_for_display(cfg)}, "
        f"fit={background_enabled_for_fit(cfg)}"
    )
    print(
        "[Background strategy] "
        f"strategy={cfg.get('radio_background_strategy', 'noise_map_only')}, "
        f"mask={bool(cfg.get('background_use_for_mask', True))}, "
        f"display={bool(cfg.get('background_use_for_display', False))}, "
        f"fit={bool(cfg.get('background_use_for_fit', False))}"
    )

    # 检查左右旋数据加和配置
    combine_polarizations = cfg.get("combine_polarizations", False)
    polarization = cfg.get("polarization", "RR")

    if combine_polarizations and polarization != "RR+LL":
        print(
            f"Warning: combine_polarizations is True but polarization is '{polarization}'"
        )
        print("Setting polarization to 'RR+LL' for combined mode")
        cfg["polarization"] = "RR+LL"

    if combine_polarizations:
        print("=" * 60)
        print("左右旋数据加和功能已启用")
        print(f"  RR目录后缀: {cfg['rr_dir_suffix']}")
        print(f"  LL目录后缀: {cfg['ll_dir_suffix']}")
        print(f"  时间对齐容差: {cfg.get('time_tolerance_seconds', 1.0)} 秒")

        if cfg.get("weighted_average", False):
            print(
                f"  组合方式: 加权平均 (RR权重={cfg.get('rr_weight', 0.5)}, LL权重={cfg.get('ll_weight', 0.5)})"
            )
        else:
            print("  组合方式: 简单相加")

        if cfg.get("save_individual_pols", False):
            print("  同时保存单独的RR、LL图像: 是")
        else:
            print("  同时保存单独的RR、LL图像: 否")
        print("=" * 60)

    # ── 1. 根据模式决定文件列表 / 时间槽 ──────────────────────
    if mode == "multi_band":
        print("Operation mode: multi-band synthesis")
        slots = _build_multi_band_slots(cfg)
        slot_items = list(enumerate(slots))
        if workspace_selection is not None:
            slot_items = _selected_workspace_slot_items(slots, workspace_selection)
            slots = [slot for _slot_idx, slot in slot_items]
            print(
                "Workspace selection active: plotting only "
                f"{len(slots)} selected source-map slot(s)."
            )
        output_dir = cfg.get("output_dir") or os.path.join(
            cfg["multi_band_root"], "plot"
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # 计算每个波段的固定颜色范围
        if workspace_selection is None and _should_precompute_fixed_band_ranges(cfg):
            fixed_band_vmins, fixed_band_vmaxs = _compute_fixed_band_ranges(cfg)
            cfg["fixed_band_vmins"] = fixed_band_vmins
            cfg["fixed_band_vmaxs"] = fixed_band_vmaxs

        if cfg.get("multi_band_also_save_single", False):
            print(
                "Note: multi_band_also_save_single=True, will also save single-band images"
            )

    else:
        print("Operation mode: single-band")
        single_file = cfg.get("single_file_path")

        if workspace_selection is not None:
            files = _selected_workspace_files(workspace_selection, cfg)
            first_file = files[0]
            output_dir = cfg.get("output_dir") or os.path.join(
                os.path.dirname(first_file), "plot"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(
                "Workspace selection active: processing only "
                f"{len(files)} selected source-map file(s)."
            )

            # 单文件模式不需要预先计算颜色范围
        elif single_file and os.path.isfile(single_file):
            files = [single_file]
            output_dir = cfg.get("output_dir") or os.path.join(
                os.path.dirname(single_file), "plot"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"Single-file mode, processing only: {single_file}")

            # 单文件模式不需要预先计算颜色范围
        else:
            if single_file:
                print(
                    f"[Warning] Specified single file does not exist: {single_file}. Falling back to batch processing mode."
                )

            # 根据是否启用左右旋数据加和来决定数据目录
            if combine_polarizations and polarization == "RR+LL":
                # 左右旋加和模式：需要从RR目录获取文件列表
                data_dir = cfg["data_dir"]
                # 检查是否是RR目录，如果不是则尝试转换为RR目录
                if (
                    cfg["rr_dir_suffix"] not in data_dir
                    and cfg["ll_dir_suffix"] not in data_dir
                ):
                    # 假设数据目录是频率目录，需要添加RR子目录
                    data_dir = os.path.join(data_dir, cfg["rr_dir_suffix"])
                elif cfg["ll_dir_suffix"] in data_dir:
                    # 如果是LL目录，转换为RR目录
                    data_dir = data_dir.replace(
                        cfg["ll_dir_suffix"], cfg["rr_dir_suffix"]
                    )

                if not os.path.exists(data_dir):
                    raise FileNotFoundError(f"找不到RR数据目录: {data_dir}")

                print(
                    f"Single-band mode with RR+LL summation, using RR directory: {data_dir}"
                )
            else:
                data_dir = cfg["data_dir"]
                print(f"Single-band mode, polarization: {polarization}")

            output_dir = cfg.get("output_dir") or os.path.join(data_dir, "plot")
            os.makedirs(output_dir, exist_ok=True)
            files = get_sorted_fits(
                data_dir,
                cfg["start_idx"],
                cfg["end_idx"],
                study_mode=cfg.get("study_mode"),
            )
            print(f"Selected {len(files)} FITS files, output directory: {output_dir}")

            # 为单波段模式计算固定颜色范围
            if len(files) > 0:
                print("计算单波段模式的固定颜色范围...")
                all_band_data = []

                for file_path in tqdm(files, desc="读取文件数据", unit="文件"):
                    try:
                        img_data, header = read_fits(file_path)

                        # 如果是RR+LL模式，需要读取对应的LL文件
                        if combine_polarizations and polarization == "RR+LL":
                            # 获取对应的LL文件路径
                            base_dir = os.path.dirname(os.path.dirname(file_path))
                            file_name = os.path.basename(file_path)
                            ll_path = os.path.join(
                                base_dir, cfg["ll_dir_suffix"], file_name
                            )

                            if os.path.exists(ll_path):
                                ll_data, ll_header = read_fits(ll_path)
                                # 组合数据
                                img_data = _combine_polarization_data(
                                    img_data, ll_data, cfg
                                )

                        # 对数化处理
                        mask = img_data > 0
                        log_data = np.full_like(img_data, np.nan, dtype=np.float64)
                        log_data[mask] = np.log10(img_data[mask])

                        # 收集有效数据
                        valid_data = log_data[~np.isnan(log_data)]
                        if len(valid_data) > 0:
                            all_band_data.extend(valid_data)

                    except Exception as e:
                        warnings.warn(f"读取文件时出错 {file_path}: {e}", stacklevel=2)
                        continue

                if len(all_band_data) > 0:
                    all_band_data_array = np.array(all_band_data)
                    vmin_band, vmax_band = _calculate_range(
                        all_band_data_array, cfg, is_global=False
                    )
                    cfg["fixed_vmin"] = vmin_band
                    cfg["fixed_vmax"] = vmax_band
                    print(
                        f"单波段固定颜色范围: [{vmin_band:.3f}, {vmax_band:.3f}] (基于{len(all_band_data)}个数据点)"
                    )
                else:
                    print("警告: 没有有效数据用于计算颜色范围")

    # ── 1.5 预加载动态频谱（仅一次）──────────────────────────
    if _spectrogram_panel_enabled(cfg):
        radio_items_for_time = slots if mode == "multi_band" else files
        derived_range = _derive_radio_time_range(
            radio_items_for_time,
            margin_seconds=0.0,
            cfg=cfg,
        )
        global _SPECTROGRAM_CACHE
        _prepare_spectrogram_colorbar_label(cfg)
        _SPECTROGRAM_CACHE = build_spectrogram_cache(cfg, derived_range)
        if _SPECTROGRAM_CACHE is None:
            cfg["enable_spectrogram_panel"] = False
            print(
                "[Warning] Spectrogram panel disabled because cache initialization failed."
            )

    # ── 2. 确定运行模式（交互 / 并行） ─────────────────────────
    if cfg["show_plot"]:
        for backend in ("TkAgg", "Qt5Agg", "MacOSX", "WXAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        cfg = {**cfg, "_interactive": True}
        use_parallel = False
        print(
            f"show_plot=True: interactive backend {matplotlib.get_backend()}, single‑process frame‑by‑frame display"
        )
    else:
        matplotlib.use("Agg")
        cfg = {**cfg, "_interactive": False}
        use_parallel = True

    if _spectrogram_panel_enabled(cfg) and use_parallel:
        # The dynamic spectrum is cached in the main process. Keep plotting sequential so the
        # large FITS file is not re-opened by every worker process.
        use_parallel = False
        print(
            "Spectrogram panel enabled: using single-process plotting to reuse the cached spectrum."
        )

    # ── 3. 安全 max_workers ─────────────────────────────────────
    if mode == "multi_band":
        # 对于多波段模式，提取每个slot的第一个文件的第一个元素（如果是元组）
        sample_files = []
        for slot in slots[:5]:
            if slot and len(slot) > 0:
                first_item = slot[0]
                if isinstance(first_item, tuple):
                    sample_files.append(first_item[0])  # 取RR文件
                else:
                    sample_files.append(first_item)
    else:
        sample_files = files[:5]

    safe_workers = _estimate_safe_workers(
        file_list=sample_files,
        requested=cfg.get("max_workers"),
        memory_per_worker_mb=cfg.get("memory_per_worker_mb"),
    )

    # ── 4. 处理色彩范围 ─────────────────────────────────────────
    vmin = vmax = None
    color_range_mode = cfg.get("color_range_mode", "auto")

    if color_range_mode == "auto":
        print("Color range mode: auto adjust per frame")

    elif color_range_mode == "global":
        print("Color range mode: fixed to global min/max")
        all_files = []
        if mode == "multi_band":
            # 对于多波段模式，提取所有文件的第一个元素（如果是元组）
            for slot in slots:
                for item in slot:
                    if isinstance(item, tuple):
                        all_files.append(item[0])  # 取RR文件
                    else:
                        all_files.append(item)
        else:
            all_files = files

        # ★ parallel statistics, reuse safe_workers
        vmin, vmax = compute_global_range(
            all_files, None, None, max_workers=safe_workers
        )

    elif color_range_mode == "fixed":
        fixed_vmin = cfg.get("fixed_vmin")
        fixed_vmax = cfg.get("fixed_vmax")
        if fixed_vmin is not None or fixed_vmax is not None:
            print(
                f"Color range mode: fixed values [{fixed_vmin:.3e}, {fixed_vmax:.3e}]"
            )
            vmin, vmax = fixed_vmin, fixed_vmax
        else:
            print(
                "Warning: color_range_mode='fixed' but fixed_vmin/vmax not set, fallback to auto mode"
            )

    else:
        print(f"Warning: unknown mode '{color_range_mode}', using auto‑adjust mode")

    # ── 5. 预创建单波段输出子目录（multi-band 保存时再创建实际目录）──
    if mode != "multi_band":
        _precreate_single_band_dirs(files, output_dir)

    # ── 6. 绘图（多进程批量 / 单进程交互） ──────────────────────
    t0 = time.time()
    errors = []
    batch_generated_at = datetime.datetime.now(datetime.timezone.utc)

    if mode == "multi_band":
        slot_plan = [
            (sequence, slot_idx, slot)
            for sequence, (slot_idx, slot) in enumerate(slot_items, start=1)
        ]
        if use_parallel and len(slot_items) > 1:
            worker = partial(
                plot_multi_band_slot,
                output_dir=output_dir,
                cfg=cfg,
                vmin=vmin,
                vmax=vmax,
            )
            with ProcessPoolExecutor(max_workers=safe_workers) as executor:
                futures = {
                    executor.submit(
                        worker,
                        slot_idx,
                        slot,
                        sequence=sequence,
                        generated_at=batch_generated_at,
                    ): slot_idx
                    for sequence, slot_idx, slot in slot_plan
                }
                with tqdm(
                    total=len(slot_items),
                    desc="Multi‑band plotting progress",
                    unit="slots",
                ) as pbar:
                    for future in as_completed(futures):
                        slot_idx = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            errors.append((slot_idx, str(e)))
                            tqdm.write(f"[Error] slot {slot_idx}: {e}")
                        finally:
                            pbar.update(1)
        else:
            slot_items = slot_plan
            for sequence, slot_idx, slot in tqdm(
                slot_items, desc="Multi‑band plotting progress", unit="slots"
            ):
                try:
                    plot_multi_band_slot(
                        slot_idx,
                        slot,
                        output_dir,
                        cfg,
                        vmin,
                        vmax,
                        sequence=sequence,
                        generated_at=batch_generated_at,
                    )
                except Exception as e:
                    errors.append((slot_idx, str(e)))
                    tqdm.write(f"[Error] slot {slot_idx}: {e}")

    else:
        file_plan = list(enumerate(files, start=1))
        if use_parallel and len(files) > 1:
            worker = partial(
                plot_single_band, output_dir=output_dir, cfg=cfg, vmin=vmin, vmax=vmax
            )
            with ProcessPoolExecutor(max_workers=safe_workers) as executor:
                futures = {
                    executor.submit(
                        worker,
                        fp,
                        sequence=sequence,
                        generated_at=batch_generated_at,
                    ): fp
                    for sequence, fp in file_plan
                }
                with tqdm(
                    total=len(files), desc="Plotting progress", unit="files"
                ) as pbar:
                    for future in as_completed(futures):
                        fp = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            errors.append((fp, str(e)))
                            tqdm.write(f"[Error] {os.path.basename(fp)}: {e}")
                        finally:
                            pbar.update(1)
        else:
            for sequence, fp in tqdm(file_plan, desc="Plotting progress", unit="files"):
                try:
                    plot_single_band(
                        fp,
                        output_dir,
                        cfg,
                        vmin,
                        vmax,
                        sequence=sequence,
                        generated_at=batch_generated_at,
                    )
                except Exception as e:
                    errors.append((fp, str(e)))
                    tqdm.write(f"[Error] {os.path.basename(fp)}: {e}")

    # ── 7. 汇总 ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    total = len(slots) if mode == "multi_band" else len(files)
    ok = total - len(errors)
    label = "slots" if mode == "multi_band" else "files"
    print(f"\nDone! Success {ok} / total {total} {label}, elapsed {elapsed:.1f} sec")
    if errors:
        print(f"Failed {label} ({len(errors)} items):")
        for item, msg in errors:
            name = item if mode == "multi_band" else os.path.basename(item)
            print(f"  {name}: {msg}")


def run_source_map(user_config: dict, *, argv=None) -> int:
    """Run source-map generation from an explicit event configuration."""

    if not isinstance(user_config, dict):
        raise TypeError("user_config must be a dictionary")
    _run_source_map_workflow(user_config=user_config, argv=argv)
    return 0


def main(user_config=None, *, argv=None) -> int:
    """Run the retained callable contract and return a process status code."""

    _run_source_map_workflow(user_config=user_config, argv=argv)
    return 0


if __name__ == "__main__":
    # On Windows, multiprocessing must start main() inside this guard block
    raise SystemExit(main())
