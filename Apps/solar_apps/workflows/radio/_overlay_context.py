"""Shared overlay configuration, result records, imports, and constants.

This module has no dependency on selection, science, rendering, or execution.
"""

# ruff: noqa: F401, I001

import csv  # noqa: E402


import glob  # noqa: E402


import math  # noqa: E402


import os  # noqa: E402


import re  # noqa: E402


import warnings  # noqa: E402


from dataclasses import dataclass, field  # noqa: E402


from datetime import datetime, timedelta, timezone  # noqa: E402


import astropy.units as u  # noqa: E402


import matplotlib  # noqa: E402


import matplotlib.colors as mcolors  # noqa: E402


import matplotlib.patheffects as mpath_effects  # noqa: E402


import matplotlib.pyplot as plt  # noqa: E402


import numpy as np  # noqa: E402


import sunpy.coordinates  # noqa: E402


import sunpy.map  # noqa: E402


from astropy.coordinates import SkyCoord  # noqa: E402


from astropy.io import fits  # noqa: E402


from matplotlib.lines import Line2D  # noqa: E402


from numpy.typing import NDArray  # noqa: E402


from scipy.interpolate import RegularGridInterpolator, griddata  # noqa: E402


from scipy.ndimage import (  # noqa: E402
    binary_dilation,
    gaussian_filter,
)


from scipy.optimize import curve_fit  # noqa: E402


from solar_toolkit.aia.config import AIA_CONFIG  # noqa: E402


from solar_toolkit.modeling.gaussian import (  # noqa: E402
    elliptical_gaussian_2d as _canonical_elliptical_gaussian_2d,
    true_indices as _canonical_true_indices,
)


from solar_apps.platform.config import apply_config_to_object  # noqa: E402


from solar_apps.workflows.common.image_naming import (
    build_scientific_image_filename,
)  # noqa: E402


from solar_apps.workflows.visualization.video_cli import (
    write_video_from_paths,
)  # noqa: E402


from solar_toolkit.radio.coordinates import (  # noqa: E402
    data_coord_to_pixel as _canonical_data_coord_to_pixel,
    normalize_roi_bounds_arcsec,
    pixel_to_data_coord as _canonical_pixel_to_data_coord,
    unravel_2d_index as _canonical_unravel_2d_index,
)


from solar_toolkit.radio.gaussian import (  # noqa: E402
    GaussianFitResult as _CanonicalGaussianFitResult,
    _gaussian_fit_diag_defaults as _canonical_gaussian_fit_diag_defaults,
    _gaussian_quality_config as _canonical_gaussian_quality_config,
    _limit_fit_pixels as _canonical_limit_fit_pixels,
    _roi_slices_from_mask as _canonical_roi_slices_from_mask,
)


from solar_toolkit.radio.gaussian_background import (  # noqa: E402
    _safe_rms_map as _canonical_safe_rms_map,
    estimate_background_noise as _canonical_estimate_background_noise,
)


from solar_toolkit.radio.gaussian_masks import (  # noqa: E402
    _select_peak_connected_mask as _canonical_select_peak_connected_mask,
)


from solar_toolkit.radio.spectrogram import (  # noqa: E402
    build_spectrogram_cache,
    overlay_spectrogram_panel,
)


from ._overlay_helpers import (
    _parse_millisecond_suffix,
    _slice_file_list,
    _nearest_radio_entry_index,
    _extent_panel_aspect,
    _normalise_band_key,
    _gaussian_diagnostics_row,
    _aia_cutout_extent_arcsec,
    _radio_file_item_label,
    _header_unit_to_arcsec_scale,
)


IntArray = NDArray[np.intp]


_RE_AIA_PATS = [
    re.compile(r"aia\.lev1_euv_12s\.(\d{4}-\d{2}-\d{2}T\d{6}Z)\.\d+\.image_lev1\.fits"),
    re.compile(r"aia\.lev1_euv_12s\.(\d{4}-\d{2}-\d{2}T\d{6}Z)"),
    re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6}Z)"),
    re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"),
    re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"),
    re.compile(r"(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})"),
]


_RE_HMI_PAT = re.compile(r"(\d{8})_(\d{6})")


_RE_RADIO_PAT_YYYYJJJ = re.compile(r"(\d{6,7})_(\d{6})_(\d{1,3})")


_RE_RADIO_PAT_YYYYMMDD = re.compile(r"(\d{8})_(\d{6})")


_RE_AIA_NEW_PAT = re.compile(
    r"aia\.lev1_euv_12s\.(\d{4}-\d{2}-\d{2}T\d{6}Z)\.\d+\.image_lev1\.fits"
)


_RE_HMI_NEW_PAT = re.compile(r"hmi\.M_45s\.(\d{8})_(\d{6})_TAI")


_DATETIME_FMTS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y%m%dT%H%M%SZ",
    "%Y%m%dT%H%M%S",
    "%Y%m%d_%H%M%S",
    "%Y-%m-%dT%H%M%S",
    "%Y-%m-%dT%H%M%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H%M%S.%fZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H%M%S",
    "%Y-%m-%d %H%M%S.%f",
    "%Y%m%dT%H%M%S.%f",
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M%S.%f",
    "%d/%m/%YT%H:%M:%S",
    "%d/%m/%YT%H:%M:%S.%f",
    "%d-%b-%YT%H:%M:%S",
    "%d-%b-%YT%H:%M:%S.%f",
    "%Y%j%H%M%S",
    "%Y%j%H%M%S.%f",
]


@dataclass
class CanvasStyle:
    """
    画布与坐标轴颜色配置
    --------------------
    修改此处即可一键调整图像整体配色风格，不需要改动绘图逻辑。
    """

    # 背景色
    figure_bg: str = "white"  # 整幅图背景
    axes_bg: str = "black"  # 绘图区背景

    # 坐标轴
    tick_color: str = "black"  # 刻度颜色
    spine_color: str = "white"  # 轴边框颜色
    xlabel_color: str = "black"  # X 轴标签颜色
    ylabel_color: str = "black"  # Y 轴标签颜色
    title_color: str = "black"  # 标题颜色

    # 图例
    legend_face: str = "white"  # 图例背景
    legend_text: str = "black"  # 图例文字颜色
    legend_alpha: float = 0.6  # 图例背景透明度

    # 日面边缘
    limb_color: str = "gray"
    limb_lw: float = 1.0
    limb_alpha: float = 0.6

    # HMI 等值线
    hmi_pos_color: str = "red"  # 正极性等值线
    hmi_neg_color: str = "blue"  # 负极性等值线
    hmi_lw: float = 0.8
    hmi_alpha: float = 0.7


@dataclass
class Config:
    """主要配置参数类"""

    # ── 目录配置 ──────────────────────────────────────────────
    radio_base_dir: str = "data/radio"
    aia_base_dir: str = "data/aia/171"
    hmi_base_dir: str = "data/hmi"
    output_dir: str = "outputs/aia_radio_hmi"
    aia_wavelength: str = "171"
    aia_panel_base_dir_template: str | None = None

    # ── 文件处理配置 ───────────────────────────────────────────
    save_figure: bool = True
    dpi: int = 300
    aia_file_start_idx: int = 392
    aia_file_end_idx: int | None = 396
    aia_panel_wavelengths: list[int] | None = None
    aia_panel_ncols: int = 3
    aia_time_threshold_seconds: float = 12.0
    aia_panel_layout_style: str = "separate"
    aia_panel_base_width: float = 4.8
    aia_panel_wspace: float = 0.10
    aia_panel_hspace: float = 0.18
    aia_panel_left: float = 0.055
    aia_panel_right: float = 0.985
    aia_panel_top: float = 0.94
    aia_panel_title_y: float = 0.975
    aia_panel_spectrogram_bottom: float = 0.065
    aia_panel_spectrogram_gap: float = 0.05
    aia_panel_show_per_panel_titles: bool = True
    aia_panel_show_axis_labels: bool = True
    aia_panel_show_tick_labels: bool = True
    aia_panel_global_axis_labels: bool = False
    aia_panel_show_grid: bool = False
    aia_panel_label_mode: str = "none"
    aia_panel_label_x: float = 0.02
    aia_panel_label_y: float = 0.035
    aia_panel_label_y_last_row: float = 0.08
    aia_panel_label_fontsize: float = 8.0
    aia_panel_save_tight: bool = True

    # ── 射电波段配置 ───────────────────────────────────────────
    selected_bands: list[str] = field(
        default_factory=lambda: [
            "149MHz",
            "164MHz",
            "190MHz",
            "205MHz",
            "223MHz",
            "238MHz",
        ]
    )

    # 偏振模式配置
    # "RR": 仅使用右旋圆偏振数据
    # "LL": 仅使用左旋圆偏振数据
    # "RR+LL": 右旋和左旋数据合并
    polarization_mode: str = "RR+LL"

    # ── 左右旋数据加和配置 ─────────────────────────────────────
    combine_polarizations: bool = True  # 是否启用左右旋数据加和功能
    rr_dir_suffix: str = "RR"  # 右旋数据目录后缀
    ll_dir_suffix: str = "LL"  # 左旋数据目录后缀
    weighted_average: bool = False  # 是否使用加权平均（True）或简单相加（False）
    rr_weight: float = 0.5  # 右旋权重（加权平均时使用）
    ll_weight: float = 0.5  # 左旋权重（加权平均时使用）
    # 同时保存功能暂未实现
    save_individual_pols: bool = False  # 是否同时保存单独的RR、LL图像
    time_tolerance_seconds: float = 0.01  # 时间对齐容差（秒）

    radio_time_threshold: int = 6  # 射电与 AIA 时间匹配阈值（秒）
    max_radio_per_band: int = 28
    radio_start_idx: int | None = None
    radio_end_idx: int | None = None
    multi_band_time_tolerance_seconds: float = 0.1

    # AIA radio overlay mode. "gaussian" preserves the historical fit path;
    # "raw" directly maps the radio image onto the AIA cutout grid.
    radio_overlay_mode: str = "gaussian"
    raw_reproject_interpolation_method: str = "linear"

    # Optional spectrogram panel below the AIA/radio overlay.
    enable_spectrogram_panel: bool = False
    spectrogram_file_paths: list[str] | None = None
    spectrogram_file_path: str | None = None
    spectrogram_time_display_mode: str = "auto"
    spectrogram_time_start: str | None = None
    spectrogram_time_end: str | None = None
    spectrogram_time_margin_seconds: float = 30.0
    spectrogram_f_start: float = 80.0
    spectrogram_f_end: float = 340.0
    spectrogram_polarization: str = "sum"
    spectrogram_vmin: float | None = None
    spectrogram_vmax: float | None = None
    spectrogram_use_log10: bool = True
    spectrogram_cmap: str = "jet"
    spectrogram_title: str | None = None
    spectrogram_colorbar_label: str | None = None
    spectrogram_panel_height_ratio: float = 0.34
    spectrogram_hspace: float = 0.08
    spectrogram_draw_colorbar: bool = True
    spectrogram_xtick_format: str = "%H:%M:%S"
    spectrogram_rebin_t_target: int = 1000
    spectrogram_rebin_f_target: int = 700
    spectrogram_chunk_mem_mb: int = 64
    spectrogram_line_color: str = "white"
    spectrogram_line_style: str = "--"
    spectrogram_line_width: float = 1.6
    spectrogram_line_alpha: float = 0.95
    spectrogram_major_tick_seconds: int = 10
    spectrogram_auto_time_locator: bool = True
    spectrogram_max_time_ticks: int = 8
    spectrogram_disable_on_time_mismatch: bool = True
    spectrogram_clip_current_time_line: bool = True
    spectrogram_show_out_of_range_time_note: bool = True
    annotation_fontsize: int = 20

    # Optional drift-rate annotations drawn on the shared spectrogram panel.
    enable_drift_rate_overlay: bool = False
    drift_rate_mode: str = "off"
    drift_rate_interactive: dict = field(default_factory=dict)
    drift_rate_selection_json: str = "spectrogram_drift_rate_manual_selection.json"
    drift_rate_selection_preview_png: str = "drift_rate_selection_preview"
    drift_rate_selection_metadata_json: str = (
        "spectrogram_drift_rate_selection_metadata.json"
    )
    draw_drift_rate_lines: bool = True
    draw_drift_rate_endpoints: bool = True
    draw_drift_rate_label: bool = True
    draw_drift_rate_selected_id: bool = True
    drift_rate_label_format: str = "{label}: df/dt={drift_rate:.2f} MHz/s"
    drift_rate_line_width: float = 2.2
    drift_rate_endpoint_marker: str = "o"
    drift_rate_endpoint_size: float = 30.0
    save_drift_rate_diagnostics: bool = False
    drift_rate_diagnostics_csv: str = "radio_spectrogram_drift_rate_diagnostics.csv"

    # Optional MP4 generation from saved PNG frames.
    make_animation: bool = False
    animation_fps: int = 10
    animation_name: str = "aia_radio_overlay.mp4"
    animation_quality: str = "high"
    animation_output_dir: str | None = None

    # ── 等值线配置 ─────────────────────────────────────────────
    show_radio_contours: bool = False
    contour_levels_peak: list[float] = field(default_factory=lambda: [0.90])
    contour_linewidths: list[float] = field(default_factory=lambda: [2.0])
    contour_alpha: float = 0.90
    contour_smooth_sigma: float = 0
    mark_radio_center: bool = True
    radio_center_marker: str = "x"
    radio_center_size: float = 50
    radio_center_linewidth: float = 1.8
    label_radio_center: bool = False

    # ── 高斯拟合稳健性配置 ─────────────────────────────────────
    enable_gaussian_overlay: bool = True
    draw_low_quality_gaussian_contours: bool = False

    fit_use_source_mask: bool = True
    fit_snr_threshold: float = 5.0
    fit_grow_snr_threshold: float = 3.0
    fit_peak_fraction_threshold: float = 0.40
    fit_grow_peak_fraction_threshold: float = 0.22
    fit_mask_target_min_pixels: int = 18
    fit_mask_target_max_pixels: int = 260
    fit_peak_fraction_threshold_min: float = 0.25
    fit_peak_fraction_threshold_max: float = 0.62
    fit_peak_fraction_threshold_step: float = 0.03
    fit_min_mask_pixels: int = 12
    fit_mask_dilation_pixels: int = 1

    gaussian_fit_use_roi: bool = True
    gaussian_fit_roi_padding_pixels: int = 4
    gaussian_fit_max_pixels: int = 400
    gaussian_fit_normalize_data: bool = True
    gaussian_fit_fallback_to_moment: bool = True
    gaussian_fit_maxfev: int = 8000
    gaussian_fit_verbose: bool = False

    fit_background_model: str = "constant"  # "none", "constant", "plane"
    max_sigma_fraction: float = 0.18
    max_fwhm_arcsec: float = 1800.0
    max_center_peak_distance_arcsec: float = 300.0
    gaussian_max_center_peak_distance_fraction_of_fwhm: float = 0.5

    gaussian_valid_only_for_overlay: bool = True
    gaussian_valid_only_for_trajectory: bool = True
    gaussian_allow_moment_fallback_for_trajectory: bool = False

    background_use_for_mask: bool = True
    background_mesh_size: int = 96
    background_mesh_step: int = 48
    background_sigma_clip: float = 3.0
    background_sigma_clip_iters: int = 3
    background_min_valid_pixels: int = 20
    background_rms_floor: float = 1e-12

    save_gaussian_diagnostics: bool = True
    gaussian_diagnostics_csv: str = "aia_radio_gaussian_fit_diagnostics.csv"

    gaussian_quality_requirements: dict = field(
        default_factory=lambda: {
            "require_quality_ok": True,
            "max_fwhm_arcsec": 1800.0,
            "max_center_peak_distance_arcsec": 300.0,
            "min_snr": 5.0,
            "max_residual_rms_fraction": 0.8,
        }
    )

    gaussian_per_band_params: dict = field(
        default_factory=lambda: {
            "149MHz": {
                "fit_peak_fraction_threshold": 0.38,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.55,
                "fit_mask_target_min_pixels": 18,
                "fit_mask_target_max_pixels": 180,
                "gaussian_fit_roi_padding_pixels": 4,
                "gaussian_fit_max_pixels": 350,
                "max_sigma_fraction": 0.17,
                "fit_background_model": "constant",
                "fit_mask_dilation_pixels": 1,
            },
            "164MHz": {
                "fit_peak_fraction_threshold": 0.38,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.55,
                "fit_mask_target_min_pixels": 18,
                "fit_mask_target_max_pixels": 180,
                "gaussian_fit_roi_padding_pixels": 4,
                "gaussian_fit_max_pixels": 350,
                "max_sigma_fraction": 0.17,
                "fit_background_model": "constant",
                "fit_mask_dilation_pixels": 1,
            },
            "190MHz": {
                "fit_peak_fraction_threshold": 0.40,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.60,
                "fit_mask_target_min_pixels": 18,
                "fit_mask_target_max_pixels": 220,
                "gaussian_fit_roi_padding_pixels": 4,
                "gaussian_fit_max_pixels": 400,
                "max_sigma_fraction": 0.18,
                "fit_background_model": "constant",
                "fit_mask_dilation_pixels": 1,
            },
            "205MHz": {
                "fit_peak_fraction_threshold": 0.40,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.60,
                "fit_mask_target_min_pixels": 18,
                "fit_mask_target_max_pixels": 240,
                "gaussian_fit_roi_padding_pixels": 4,
                "gaussian_fit_max_pixels": 400,
                "max_sigma_fraction": 0.18,
                "fit_background_model": "constant",
                "fit_mask_dilation_pixels": 1,
            },
            "223MHz": {
                "fit_peak_fraction_threshold": 0.42,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.65,
                "fit_mask_target_min_pixels": 16,
                "fit_mask_target_max_pixels": 180,
                "gaussian_fit_roi_padding_pixels": 3,
                "gaussian_fit_max_pixels": 320,
                "max_sigma_fraction": 0.16,
                "fit_background_model": "plane",
                "fit_mask_dilation_pixels": 1,
            },
            "238MHz": {
                "fit_peak_fraction_threshold": 0.42,
                "fit_peak_fraction_threshold_min": 0.25,
                "fit_peak_fraction_threshold_max": 0.65,
                "fit_mask_target_min_pixels": 16,
                "fit_mask_target_max_pixels": 180,
                "gaussian_fit_roi_padding_pixels": 3,
                "gaussian_fit_max_pixels": 320,
                "max_sigma_fraction": 0.16,
                "fit_background_model": "plane",
                "fit_mask_dilation_pixels": 1,
            },
        }
    )

    # ── 显示配置 ───────────────────────────────────────────────
    overlay_hmi: bool = True
    hmi_time_threshold: int = 24  # HMI 与 AIA 时间匹配阈值（小时）
    hmi_threshold_gauss: float = 0.0
    hmi_sigma: int = 2
    hmi_levels_gauss: list[float] = field(default_factory=lambda: [100.0])

    # ── AIA 图像配置 ───────────────────────────────────────────
    aia_vmin: float = 16
    aia_vmax: float = 6666
    aia_cmap: str = "sdoaia171"
    roi_bounds_arcsec: dict | None = None
    roi_bottom_left: list[float] = field(default_factory=lambda: [600, -800])
    roi_top_right: list[float] = field(default_factory=lambda: [1600, 200])

    # ── 画布颜色配置 ───────────────────────────────────────────
    style: CanvasStyle = field(default_factory=CanvasStyle)

    # ── 射电波段颜色配置 ───────────────────────────────────────
    band_colors_dict: dict = field(
        default_factory=lambda: {
            "149.0MHz": ("dodgerblue", "navy"),  # 深蓝系（清晰）
            "164.0MHz": ("orange", "darkorange"),  # 橙色（强对比）
            "190.0MHz": ("crimson", "darkred"),  # 红色（最醒目）
            "205.0MHz": ("mediumorchid", "purple"),  # 紫色（区别红）
            "223.0MHz": ("gold", "goldenrod"),  # 金色（对AIA很好）
            "238.0MHz": ("teal", "darkslategray"),  # 青绿偏暗（避开背景）
        }
    )
    default_colors: list[tuple] = field(
        default_factory=lambda: [
            ("dodgerblue", "navy"),
            ("orange", "darkorange"),
            ("crimson", "darkred"),
            ("mediumorchid", "purple"),
            ("gold", "goldenrod"),
            ("teal", "darkslategray"),
            ("deeppink", "hotpink"),  # 额外增强区分
            ("royalblue", "midnightblue"),  # 更深蓝备选
        ]
    )

    # ── 性能配置 ───────────────────────────────────────────────
    radio_use_float32: bool = True

    # ── 处理选项 ───────────────────────────────────────────────
    debug_mode: bool = True

    # ── 坐标图配置 ─────────────────────────────────────────────
    use_radec_maps: bool = True  # 是否使用赤经赤纬坐标

    def __post_init__(self):
        apply_config_to_object(self, "sdo_aia_radio_hmi_overlay")


_SPECTROGRAM_CONFIG_MAP = {
    "enabled": "enable_spectrogram_panel",
    "file_paths": "spectrogram_file_paths",
    "file_path": "spectrogram_file_path",
    "time_display_mode": "spectrogram_time_display_mode",
    "time_start": "spectrogram_time_start",
    "time_end": "spectrogram_time_end",
    "time_margin_seconds": "spectrogram_time_margin_seconds",
    "f_start": "spectrogram_f_start",
    "f_end": "spectrogram_f_end",
    "polarization": "spectrogram_polarization",
    "vmin": "spectrogram_vmin",
    "vmax": "spectrogram_vmax",
    "use_log10": "spectrogram_use_log10",
    "cmap": "spectrogram_cmap",
    "title": "spectrogram_title",
    "colorbar_label": "spectrogram_colorbar_label",
    "panel_height_ratio": "spectrogram_panel_height_ratio",
    "hspace": "spectrogram_hspace",
    "draw_colorbar": "spectrogram_draw_colorbar",
    "xtick_format": "spectrogram_xtick_format",
    "rebin_t_target": "spectrogram_rebin_t_target",
    "rebin_f_target": "spectrogram_rebin_f_target",
    "chunk_mem_mb": "spectrogram_chunk_mem_mb",
    "line_color": "spectrogram_line_color",
    "line_style": "spectrogram_line_style",
    "line_width": "spectrogram_line_width",
    "line_alpha": "spectrogram_line_alpha",
    "major_tick_seconds": "spectrogram_major_tick_seconds",
    "auto_time_locator": "spectrogram_auto_time_locator",
    "max_time_ticks": "spectrogram_max_time_ticks",
    "disable_on_time_mismatch": "spectrogram_disable_on_time_mismatch",
    "clip_current_time_line": "spectrogram_clip_current_time_line",
    "show_out_of_range_time_note": "spectrogram_show_out_of_range_time_note",
}


_DRIFT_RATE_CONFIG_MAP = {
    "enabled": "enable_drift_rate_overlay",
    "mode": "drift_rate_mode",
    "interactive": "drift_rate_interactive",
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


_ANIMATION_CONFIG_MAP = {
    "make_animation": "make_animation",
    "fps": "animation_fps",
    "animation_fps": "animation_fps",
    "name": "animation_name",
    "animation_name": "animation_name",
    "quality": "animation_quality",
    "animation_quality": "animation_quality",
    "output_dir": "animation_output_dir",
    "animation_output_dir": "animation_output_dir",
}


def _apply_values_to_config(cfg: Config, values: dict | None) -> Config:
    for key, value in (values or {}).items():
        if key == "style" and isinstance(value, dict):
            for style_key, style_value in value.items():
                if hasattr(cfg.style, style_key):
                    setattr(cfg.style, style_key, style_value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _apply_mapped_values_to_config(
    cfg: Config, values: dict | None, key_map: dict[str, str]
) -> Config:
    for key, value in (values or {}).items():
        target = key_map.get(key)
        if target and hasattr(cfg, target):
            setattr(cfg, target, value)
    return cfg


def apply_aia_radio_hmi_user_config(cfg: Config, user_config: dict | None) -> Config:
    """Apply grouped user config to the legacy Config dataclass."""
    if not user_config:
        return cfg
    for section in (
        "paths",
        "aia",
        "hmi",
        "radio",
        "wcs_reproject",
        "gaussian",
        "display",
        "output",
        "runtime",
    ):
        _apply_values_to_config(cfg, user_config.get(section))
    _apply_mapped_values_to_config(
        cfg, user_config.get("spectrogram"), _SPECTROGRAM_CONFIG_MAP
    )
    _apply_mapped_values_to_config(
        cfg, user_config.get("drift_rate"), _DRIFT_RATE_CONFIG_MAP
    )
    _apply_mapped_values_to_config(
        cfg, user_config.get("animation"), _ANIMATION_CONFIG_MAP
    )
    features = user_config.get("features")
    if isinstance(features, dict) and "spectrogram_panel" in features:
        cfg.enable_spectrogram_panel = bool(features["spectrogram_panel"])
    _apply_values_to_config(
        cfg,
        {
            key: value
            for key, value in user_config.items()
            if not isinstance(value, dict)
        },
    )
    return cfg


GaussianFitResult = _CanonicalGaussianFitResult


@dataclass
class GaussianReprojectResult:
    model: np.ndarray
    center_pixel: tuple[float, float]
    center_arcsec: tuple[float, float]
    sigma_pixel: tuple[float, float]
    theta_rad: float
    amplitude: float
    covariance: np.ndarray | None
    quality_flag: str = "ok"
    quality_flag_detail: str = ""
    overlay_valid: bool = True
    trajectory_valid: bool = True
    snr: float | None = None
    residual_rms: float | None = None
    mask_pixel_count: int = 0
    fwhm_major_arcsec: float | None = None
    fwhm_minor_arcsec: float | None = None
    center_peak_distance_arcsec: float | None = None
    source_file: str | None = None
    radio_fit_result: GaussianFitResult | None = None


@dataclass
class RawRadioReprojectResult:
    model: np.ndarray
    peak_pixel: tuple[float, float]
    peak_arcsec: tuple[float, float]
    amplitude: float
    source_file: str | None = None
    overlay_valid: bool = True


_radec_file_cache: dict[tuple[str, str, str], str | None] = {}


GAUSSIAN_DIAGNOSTIC_FIELDS = [
    "source_file",
    "time",
    "band",
    "polarization",
    "quality_flag",
    "quality_flag_detail",
    "center_x_arcsec",
    "center_y_arcsec",
    "center_x_pixel",
    "center_y_pixel",
    "sigma_x_pixel",
    "sigma_y_pixel",
    "fwhm_major_arcsec",
    "fwhm_minor_arcsec",
    "amplitude",
    "snr",
    "residual_rms",
    "mask_pixel_count",
    "fit_peak_fraction_threshold_used",
    "fit_peak_fraction_candidate_counts",
    "background_rms_median",
    "background_level_median",
    "gaussian_fit_method",
    "roi_used",
    "roi_shape",
]
