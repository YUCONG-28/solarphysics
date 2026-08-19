# 模块用途: 从 FITS 数据绘制 CSO 射电动态频谱。
# 主要输入: CSO FITS 文件、偏振通道和降采样配置。
# 主要输出/运行说明: 输出频率-时间谱图，包含内存友好的分块/降采样处理。
"""
Created on Sun Nov 23 00:19:30 2025


"""

"""
CSO Spectrogram Plotting and Analysis Tool

This script provides a comprehensive solution for processing and visualizing
Chinese Solar Radio Telescope (CSO) spectrogram data from FITS files.
It employs memory-efficient techniques to handle large datasets while
maintaining high performance through parallel processing and intelligent
downsampling.

Key Features:
- Memory-mapped I/O for handling large FITS files without loading entire datasets
- Parallel processing of polarization channels (LL and RR)
- Configurable downsampling to balance resolution and performance
- Multiple visualization options: individual polarizations, total intensity, polarization ratio
- Frequency highlighting with customizable markers
- Flexible output options: display, save to file, or both
"""

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import webbrowser  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from functools import wraps  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from solar_apps.platform.config import apply_config_to_object  # noqa: E402
from solar_toolkit.radio import drift_rate as _canonical_drift_rate  # noqa: E402
from solar_toolkit.radio._lazy_spectrogram import (  # noqa: E402
    LazySpectrogram,
    _find_range,
    read_cso_fits,
)

__all__ = [
    "AxisConfigManager",
    "build_parser",
    "ConfigManager",
    "DriftRateResult",
    "DriftSpectrogramView",
    "LazySpectrogram",
    "PlotConfig",
    "calc_bin_sizes",
    "calc_bin_sizes_for_specs",
    "calc_polarization_ratio",
    "calculate_drift_rate_from_line",
    "get_color_limits",
    "get_config_file_paths",
    "get_or_load_drift_rate_results",
    "launch_drift_selection_server",
    "load_drift_selection_json",
    "main",
    "overlay_drift_rate_results",
    "process_and_plot",
    "read_cso_fits",
    "render_spectrogram_selection_preview",
    "save_drift_rate_diagnostics_once",
    "save_drift_selection_json",
    "validate_axis_config",
    "validate_config",
]


# ============================================================
#  ★ CONFIGURATION PARAMETERS - MODIFY ONLY HERE ★
# ============================================================
@dataclass
class PlotConfig:
    """Configuration class for CSO spectrogram plotting parameters."""

    # File path
    # Single-file mode: keep using file_path.
    # Multi-file mode: set file_paths to a list of FITS files ordered or unordered.
    # The program will automatically select the portions overlapping t_start~t_end
    # and concatenate them along the time axis.
    file_path: str = "data/radio/spectrogram.fits"
    # file_paths: list[str] | None = field(default_factory=lambda: [
    #     "data/radio/spectrogram-part-1.fits",
    #     "data/radio/spectrogram-part-2.fits"
    #     ])

    # Time range (UTC)
    t_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 1, 24, 4, 14, 10)
    )
    t_end: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 1, 24, 4, 15, 20)
    )

    # Frequency range (MHz)
    f_start: float = 80
    f_end: float = 340

    # Target number of grid points after downsampling (time / frequency axes)
    # Larger values produce finer plots but are slower; None = no downsampling
    rebin_t_target: int = 10000
    rebin_f_target: int = 10000

    # Peak memory limit per chunk during block reading (MB)
    # Lower values reduce memory pressure further
    chunk_mem_mb: int = 28

    # Maximum number of CPU cores to use (None = auto-detect, 1 = single core)
    max_workers: int | None = None

    # Plot toggles
    plot_ll: bool = False
    plot_rr: bool = False
    plot_sum: bool = True
    plot_ratio: bool = True

    # Color scale configuration - CHOOSE ONE METHOD:
    # Method 1: Percentile-based clipping (automatic)
    use_percentile_clipping: bool = False  # Set to False to use manual limits
    vmin_pct: float = 0.1
    vmax_pct: float = 99.9
    sum_vmin_pct: float = 0.1
    sum_vmax_pct: float = 99.9
    # For ratio, we need symmetric percentiles since data ranges from -1 to 1
    ratio_vmin_pct: float = 1.0  # Use 1st percentile for negative side
    ratio_vmax_pct: float = 99.0  # Use 99th percentile for positive side

    # Method 2: Manual absolute limits (used when use_percentile_clipping = False)
    # Set these to specific values like 0.0 and 10.0
    # Individual polarization limits
    manual_ll_vmin: float | None = 1.8
    manual_ll_vmax: float | None = 5
    manual_rr_vmin: float | None = 1.8
    manual_rr_vmax: float | None = 5
    # Sum and ratio limits
    manual_sum_vmin: float | None = 2.2
    manual_sum_vmax: float | None = 5.6
    manual_ratio_vmin: float | None = -1.0
    manual_ratio_vmax: float | None = 1.0
    ratio_auto_symmetric_scale: bool = True
    ratio_abs_percentile: float = 99.5
    ratio_min_abs_limit: float = 0.03
    ratio_max_abs_limit: float = 1.0

    # Backward compatibility: if individual limits not set, use these
    manual_vmin: float | None = None
    manual_vmax: float | None = None

    # Figure dimensions
    fig_width: float = 12.0
    fig_height_per: float = 3.0  # Height per subplot (inches)

    # Time axis tick intervals (seconds)
    major_tick_interval: int = 10
    minor_tick_interval: int = 2

    # Save path (empty for display only)
    save_path: str = ""
    dpi: int = 300
    show_plot: bool = False
    close_after_save: bool = True

    # List of frequencies to highlight (MHz)
    highlight_freqs: list[float] | None = field(
        default_factory=lambda: [149, 164, 190, 205, 223, 238, 285, 300, 309, 324]
    )
    # [149, 164, 190, 205, 223, 238, 285, 300, 309, 324]

    # 坐标轴显示控制
    show_axis_labels: bool = True  # 是否显示坐标轴标签
    axis_label_rotation: float = 0.0  # 标签旋转角度（度）
    xtick_interval: int | None = None  # X轴刻度间隔（秒），None为自动
    ytick_interval: float | None = None  # Y轴刻度间隔（MHz），None为自动
    xtick_format: str = "%H:%M:%S"  # X轴时间格式
    show_minor_ticks: bool = True  # 是否显示次要刻度

    # 频漂率测量与叠加控制
    # 默认不自动弹出浏览器；运行 `python cso_radio_spectrogram_plot.py --select-drift`
    # 可先打开交互式端点选择前端，保存 JSON 后再用于正式绘图叠加。
    enable_drift_rate_overlay: bool = True
    drift_rate_mode: str = (
        "interactive_manual"  # "off", "interactive_manual", "manual_json"
    )
    drift_rate_selection_json: str = "spectrogram_drift_rate_manual_selection.json"
    drift_rate_selection_preview_png: str = "drift_rate_selection_preview"
    drift_rate_selection_metadata_json: str = (
        "spectrogram_drift_rate_selection_metadata.json"
    )
    export_drift_selection_preview: bool = False
    drift_rate_interactive: dict = field(
        default_factory=lambda: {
            "host": "127.0.0.1",
            "port": 8050,
            "auto_open_browser": True,
            "launch_policy": "always",  # "cli_only", "auto_if_missing", "always"
            "auto_increment_port": True,
            "max_port_tries": 20,
            "block_until_done": True,
            "selection_timeout_seconds": 0,
            "allow_multiple_lines": True,
            "show_crosshair": True,
            "show_live_coordinate": True,
            "show_preview_line": True,
            "line_color_cycle": [
                "white",
                "cyan",
                "lime",
                "magenta",
                "yellow",
                "orange",
            ],
            "print_usage_hint": True,
        }
    )
    draw_drift_rate_lines: bool = True
    draw_drift_rate_endpoints: bool = True
    draw_drift_rate_label: bool = True
    drift_rate_label_format: str = "{label}: df/dt={drift_rate:.2f} MHz/s"
    drift_rate_line_width: float = 2.2
    drift_rate_endpoint_marker: str = "o"
    drift_rate_endpoint_size: float = 30.0
    save_drift_rate_diagnostics: bool = True
    drift_rate_diagnostics_csv: str = "radio_spectrogram_drift_rate_diagnostics.csv"

    def __post_init__(self):
        apply_config_to_object(self, "cso_radio_spectrogram_plot")


DriftRateResult = _canonical_drift_rate.DriftRateResult


@dataclass
class DriftSpectrogramView:
    """Minimal spectrum view used by the drift-rate selector and overlay."""

    data: np.ndarray
    time_nums: np.ndarray
    display_time_nums: tuple[float, float]
    freq: np.ndarray
    title: str
    cmap: str
    vmin: float | None
    vmax: float | None
    cbar_label: str
    source_file: str
    source_files: list[str] = field(default_factory=list)


_DRIFT_RATE_RESULTS_CACHE: dict[tuple[str, str, str], list[DriftRateResult]] = {}
_DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS: set[tuple[str, str]] = set()

DRIFT_RATE_DIAGNOSTIC_FIELDS = [
    "source_file",
    "label",
    "mode",
    "t_start",
    "t_end",
    "f_start_mhz",
    "f_end_mhz",
    "duration_s",
    "bandwidth_mhz",
    "drift_rate_mhz_s",
    "abs_drift_rate_mhz_s",
    "color",
    "quality_flag",
    "warning",
]


# ============================================================
#  UTILITY FUNCTIONS
# ============================================================


def timing_decorator(func):
    """Decorator to measure and print function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"  [{func.__name__}] {time.perf_counter() - t0:.3f} s")
        return result

    return wrapper


def _as_datetime(value, name: str) -> datetime.datetime:
    """Accept datetime objects or ISO datetime strings in PlotConfig."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        return datetime.datetime.fromisoformat(value)
    raise TypeError(f"{name} must be datetime.datetime or ISO datetime string")


def _array_stats(name: str, data: np.ndarray) -> None:
    """Print finite-count and basic statistics for diagnostics."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        print(f"  {name}: finite_count=0, min=nan, max=nan, mean=nan, std=nan")
        return
    print(
        f"  {name}: finite_count={finite.size}, min={np.nanmin(finite):.6g}, "
        f"max={np.nanmax(finite):.6g}, mean={np.nanmean(finite):.6g}, "
        f"std={np.nanstd(finite):.6g}"
    )


def get_config_file_paths(cfg: PlotConfig) -> list[str]:
    """Return FITS paths from cfg.file_paths, with cfg.file_path kept as fallback."""
    file_paths = getattr(cfg, "file_paths", None)
    if file_paths:
        if isinstance(file_paths, (str, os.PathLike)):
            paths = [str(file_paths)]
        else:
            paths = [str(path) for path in file_paths if str(path).strip()]
    else:
        paths = [str(cfg.file_path)] if str(cfg.file_path).strip() else []

    if not paths:
        raise ValueError(
            "No FITS file path is configured. Set cfg.file_path or cfg.file_paths."
        )

    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("FITS file(s) not found:\n" + "\n".join(missing))

    return paths


def _spec_time_bounds(
    spec: "LazySpectrogram",
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return absolute datetime coverage of one LazySpectrogram."""
    finite_time = spec.time[np.isfinite(spec.time)]
    if finite_time.size == 0:
        raise ValueError(f"{spec.polar} has no finite time samples: {spec.source_path}")
    t_min = float(np.nanmin(finite_time))
    t_max = float(np.nanmax(finite_time))
    return (
        spec.dt_base + datetime.timedelta(seconds=t_min),
        spec.dt_base + datetime.timedelta(seconds=t_max),
    )


def _spec_overlaps_time_range(
    spec: "LazySpectrogram",
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> bool:
    """Whether one file has any samples within the requested absolute time range."""
    spec_start, spec_end = _spec_time_bounds(spec)
    return spec_start <= t_end and spec_end >= t_start


def _select_overlapping_specs(
    data_list: list,
    polar_key: str,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> list:
    """Select and time-sort LL/RR spectra that overlap the requested time range."""
    specs = [
        spec
        for spec in data_list
        if polar_key in spec.polar and _spec_overlaps_time_range(spec, t_start, t_end)
    ]
    return sorted(specs, key=lambda spec: _spec_time_bounds(spec)[0])


def _overlap_window(
    spec: "LazySpectrogram",
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> tuple[datetime.datetime, datetime.datetime] | None:
    """Return the part of the requested range covered by one file."""
    spec_start, spec_end = _spec_time_bounds(spec)
    overlap_start = max(t_start, spec_start)
    overlap_end = min(t_end, spec_end)
    if overlap_start >= overlap_end:
        return None
    return overlap_start, overlap_end


def _validate_contiguous_time_coverage(
    specs: list,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
    polar_label: str,
    tolerance_seconds: float = 1.0,
) -> None:
    """Raise an explicit error if the requested range is not covered by the files."""
    intervals = []
    for spec in specs:
        window = _overlap_window(spec, t_start, t_end)
        if window is not None:
            intervals.append((*window, spec.source_path))

    if not intervals:
        raise ValueError(f"No {polar_label} file overlaps the requested time range.")

    intervals.sort(key=lambda item: item[0])
    current_end = t_start
    tol = datetime.timedelta(seconds=float(tolerance_seconds))

    for seg_start, seg_end, source_path in intervals:
        if seg_start > current_end + tol:
            raise ValueError(
                f"{polar_label} files do not fully cover the requested time range. "
                f"Gap: {current_end.isoformat()} -> {seg_start.isoformat()}. "
                f"Next file: {source_path}"
            )
        if seg_end > current_end:
            current_end = seg_end
        if current_end >= t_end - tol:
            return

    raise ValueError(
        f"{polar_label} files end before the requested t_end. "
        f"Covered until {current_end.isoformat()}, requested until {t_end.isoformat()}."
    )


def _estimate_raw_selection_mb(
    specs: list,
    cfg: PlotConfig,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> float:
    """Estimate total selected raw data size for one polarization across files."""
    total_points = 0
    for spec in specs:
        window = _overlap_window(spec, t_start, t_end)
        if window is None:
            continue
        t0, t1 = window
        t0s = (t0 - spec.dt_base).total_seconds()
        t1s = (t1 - spec.dt_base).total_seconds()
        ti0, ti1 = _find_range(spec.time, t0s, t1s)
        fi0, fi1 = _find_range(spec.freq, cfg.f_start, cfg.f_end)
        total_points += max(0, ti1 - ti0 + 1) * max(0, fi1 - fi0 + 1)
    return total_points * 4 / (1024 * 1024)


def get_system_memory_info() -> tuple[float, float, float]:
    """
    Get system memory information.

    Returns:
        Tuple of (total_memory_gb, available_memory_gb, memory_usage_percent)
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        usage_percent = memory.percent
        return total_gb, available_gb, usage_percent
    except ImportError:
        warnings.warn("psutil not installed, cannot get memory info", stacklevel=2)
        return 0.0, 0.0, 0.0


def validate_config(cfg: PlotConfig) -> None:
    """Validate configuration parameters."""
    t_start = _as_datetime(cfg.t_start, "t_start")
    t_end = _as_datetime(cfg.t_end, "t_end")
    if t_start >= t_end:
        raise ValueError(
            f"t_start ({cfg.t_start}) must be earlier than t_end ({cfg.t_end})"
        )

    if cfg.f_start >= cfg.f_end:
        raise ValueError(
            f"f_start ({cfg.f_start}) must be less than f_end ({cfg.f_end})"
        )

    if cfg.rebin_t_target is not None and cfg.rebin_t_target <= 0:
        raise ValueError(f"rebin_t_target must be positive, got {cfg.rebin_t_target}")

    if cfg.rebin_f_target is not None and cfg.rebin_f_target <= 0:
        raise ValueError(f"rebin_f_target must be positive, got {cfg.rebin_f_target}")

    if cfg.chunk_mem_mb <= 0:
        raise ValueError(f"chunk_mem_mb must be positive, got {cfg.chunk_mem_mb}")

    # Check for max_workers attribute (may not exist in older config)
    if hasattr(cfg, "max_workers"):
        if cfg.max_workers is not None and cfg.max_workers <= 0:
            raise ValueError(
                f"max_workers must be positive or None, got {cfg.max_workers}"
            )

    # Check for use_percentile_clipping attribute
    if hasattr(cfg, "use_percentile_clipping"):
        if not cfg.use_percentile_clipping:
            # Check individual polarization limits
            if hasattr(cfg, "manual_ll_vmin") and hasattr(cfg, "manual_ll_vmax"):
                if cfg.manual_ll_vmin is not None and cfg.manual_ll_vmax is not None:
                    if cfg.manual_ll_vmin >= cfg.manual_ll_vmax:
                        raise ValueError(
                            f"manual_ll_vmin ({cfg.manual_ll_vmin}) must be less than manual_ll_vmax ({cfg.manual_ll_vmax})"
                        )

            if hasattr(cfg, "manual_rr_vmin") and hasattr(cfg, "manual_rr_vmax"):
                if cfg.manual_rr_vmin is not None and cfg.manual_rr_vmax is not None:
                    if cfg.manual_rr_vmin >= cfg.manual_rr_vmax:
                        raise ValueError(
                            f"manual_rr_vmin ({cfg.manual_rr_vmin}) must be less than manual_rr_vmax ({cfg.manual_rr_vmax})"
                        )

            # Check backward compatibility limits
            if hasattr(cfg, "manual_vmin") and hasattr(cfg, "manual_vmax"):
                if cfg.manual_vmin is not None and cfg.manual_vmax is not None:
                    if cfg.manual_vmin >= cfg.manual_vmax:
                        raise ValueError(
                            f"manual_vmin ({cfg.manual_vmin}) must be less than manual_vmax ({cfg.manual_vmax})"
                        )

            # Check sum limits
            if hasattr(cfg, "manual_sum_vmin") and hasattr(cfg, "manual_sum_vmax"):
                if cfg.manual_sum_vmin is not None and cfg.manual_sum_vmax is not None:
                    if cfg.manual_sum_vmin >= cfg.manual_sum_vmax:
                        raise ValueError(
                            f"manual_sum_vmin ({cfg.manual_sum_vmin}) must be less than manual_sum_vmax ({cfg.manual_sum_vmax})"
                        )

            # Check ratio limits
            if hasattr(cfg, "manual_ratio_vmin") and hasattr(cfg, "manual_ratio_vmax"):
                if (
                    cfg.manual_ratio_vmin is not None
                    and cfg.manual_ratio_vmax is not None
                ):
                    if cfg.manual_ratio_vmin >= cfg.manual_ratio_vmax:
                        raise ValueError(
                            f"manual_ratio_vmin ({cfg.manual_ratio_vmin}) must be less than manual_ratio_vmax ({cfg.manual_ratio_vmax})"
                        )

    # Warn about potential memory issues
    if (
        hasattr(cfg, "max_workers")
        and cfg.max_workers is not None
        and cfg.max_workers > 2
    ):
        warnings.warn(
            f"max_workers={cfg.max_workers} is set, but only 2 workers are needed for polarization processing",
            stacklevel=2,
        )

    # Check memory configuration
    if cfg.chunk_mem_mb > 500:
        warnings.warn(
            f"chunk_mem_mb={cfg.chunk_mem_mb} MB is quite high. Consider reducing for memory-constrained systems.",
            stacklevel=2,
        )

    # 验证坐标轴配置
    validate_axis_config(cfg)


def validate_axis_config(cfg: PlotConfig) -> None:
    """验证坐标轴配置参数"""
    if not isinstance(cfg.show_axis_labels, bool):
        raise TypeError(
            f"show_axis_labels must be boolean, got {type(cfg.show_axis_labels)}"
        )

    if not isinstance(cfg.axis_label_rotation, (int, float)):
        raise TypeError(
            f"axis_label_rotation must be numeric, got {type(cfg.axis_label_rotation)}"
        )

    if not (-90 <= cfg.axis_label_rotation <= 90):
        warnings.warn(
            f"axis_label_rotation should be between -90 and 90 degrees, got {cfg.axis_label_rotation}",
            stacklevel=2,
        )
        cfg.axis_label_rotation = np.clip(cfg.axis_label_rotation, -90, 90)

    if cfg.xtick_interval is not None and cfg.xtick_interval <= 0:
        raise ValueError(f"xtick_interval must be positive, got {cfg.xtick_interval}")

    if cfg.ytick_interval is not None and cfg.ytick_interval <= 0:
        raise ValueError(f"ytick_interval must be positive, got {cfg.ytick_interval}")

    if not isinstance(cfg.show_minor_ticks, bool):
        raise TypeError(
            f"show_minor_ticks must be boolean, got {type(cfg.show_minor_ticks)}"
        )


class AxisConfigManager:
    """坐标轴配置管理器，负责处理和优化坐标轴显示"""

    @staticmethod
    def calculate_optimal_ticks(
        data_range: float, num_points: int = None, max_ticks: int = 10
    ) -> float:
        """
        计算最优刻度间隔

        Args:
            data_range: 数据范围
            num_points: 数据点数（可选）
            max_ticks: 最大刻度数

        Returns:
            建议的刻度间隔
        """
        if data_range <= 0:
            return 1.0

        # 基于范围计算基础间隔
        base_intervals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        estimated_ticks = max_ticks

        # 如果提供点数，尝试基于点数调整
        if num_points is not None and num_points > 0:
            points_per_tick = num_points / max_ticks
            if points_per_tick < 1:
                estimated_ticks = min(max_ticks, num_points)

        ideal_interval = data_range / estimated_ticks

        # 找到最接近的理想刻度间隔
        for base in base_intervals:
            for multiplier in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                interval = base * multiplier
                if interval >= ideal_interval:
                    return interval

        return ideal_interval

    @staticmethod
    def configure_time_axis(
        ax, cfg: PlotConfig, time_values: list[datetime.datetime]
    ) -> None:
        """配置时间轴"""
        if not cfg.show_axis_labels:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            # 设置主要刻度定位器
            if cfg.xtick_interval is not None:
                major_locator = mdates.SecondLocator(interval=cfg.xtick_interval)
            else:
                # 自动计算：基于时间范围计算最优间隔
                time_range = (time_values[-1] - time_values[0]).total_seconds()
                optimal_interval = AxisConfigManager.calculate_optimal_ticks(
                    time_range, len(time_values), max_ticks=15
                )
                major_locator = mdates.SecondLocator(
                    interval=max(1, int(optimal_interval))
                )

            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(cfg.xtick_format))

            # 设置次要刻度
            if cfg.show_minor_ticks:
                if cfg.xtick_interval is not None:
                    minor_interval = max(1, cfg.xtick_interval // 5)
                else:
                    minor_interval = max(1, int(major_locator._interval // 5))
                ax.xaxis.set_minor_locator(
                    mdates.SecondLocator(interval=minor_interval)
                )

            # 设置标签旋转
            if cfg.axis_label_rotation != 0:
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=cfg.axis_label_rotation,
                    ha="right" if cfg.axis_label_rotation > 0 else "left",
                )

    @staticmethod
    def configure_frequency_axis(ax, cfg: PlotConfig, freq_values: np.ndarray) -> None:
        """配置频率轴"""
        f_min = float(np.nanmin(freq_values))
        f_max = float(np.nanmax(freq_values))
        ax.set_ylim(f_min, f_max)
        if not cfg.show_axis_labels:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        else:
            # 设置主要刻度
            if cfg.ytick_interval is not None:
                yticks = np.arange(
                    np.ceil(f_min / cfg.ytick_interval) * cfg.ytick_interval,
                    f_max,
                    cfg.ytick_interval,
                )
                ax.set_yticks(yticks)
            else:
                # 自动计算刻度
                freq_range = f_max - f_min
                optimal_interval = AxisConfigManager.calculate_optimal_ticks(
                    freq_range, len(freq_values), max_ticks=10
                )
                yticks = np.arange(
                    np.ceil(f_min / optimal_interval) * optimal_interval,
                    f_max,
                    optimal_interval,
                )
                ax.set_yticks(yticks)

            # 设置次要刻度
            if cfg.show_minor_ticks:
                from matplotlib.ticker import AutoMinorLocator

                ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            # 设置标签旋转
            if cfg.axis_label_rotation != 0:
                plt.setp(
                    ax.get_yticklabels(), rotation=cfg.axis_label_rotation, va="center"
                )


# ============================================================
#  HELPER FUNCTIONS
# ============================================================


def calc_bin_sizes(spec: LazySpectrogram, cfg: PlotConfig):
    """Calculate t_bin / f_bin based on actual slice range and target point count."""
    t_start = _as_datetime(cfg.t_start, "t_start")
    t_end = _as_datetime(cfg.t_end, "t_end")
    t1s = (t_start - spec.dt_base).total_seconds()
    t2s = (t_end - spec.dt_base).total_seconds()
    ti0, ti1 = _find_range(spec.time, t1s, t2s)
    fi0, fi1 = _find_range(spec.freq, cfg.f_start, cfg.f_end)
    n_t = ti1 - ti0 + 1
    n_f = fi1 - fi0 + 1
    t_bin = max(1, n_t // cfg.rebin_t_target) if cfg.rebin_t_target else 1
    f_bin = max(1, n_f // cfg.rebin_f_target) if cfg.rebin_f_target else 1
    return t_bin, f_bin


def calc_bin_sizes_for_specs(
    specs: list,
    cfg: PlotConfig,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> tuple[int, int]:
    """Calculate common t_bin/f_bin for a requested range spanning one or more files."""
    total_t = 0
    min_t = None
    min_f = None

    for spec in specs:
        window = _overlap_window(spec, t_start, t_end)
        if window is None:
            continue
        t0, t1 = window
        t0s = (t0 - spec.dt_base).total_seconds()
        t1s = (t1 - spec.dt_base).total_seconds()
        ti0, ti1 = _find_range(spec.time, t0s, t1s)
        fi0, fi1 = _find_range(spec.freq, cfg.f_start, cfg.f_end)
        n_t = ti1 - ti0 + 1
        n_f = fi1 - fi0 + 1
        total_t += n_t
        min_t = n_t if min_t is None else min(min_t, n_t)
        min_f = n_f if min_f is None else min(min_f, n_f)

    if total_t <= 0 or min_t is None or min_f is None or min_f <= 0:
        raise ValueError(
            "Requested time/frequency range has no valid samples in the selected files."
        )

    t_bin = max(1, total_t // cfg.rebin_t_target) if cfg.rebin_t_target else 1
    f_bin = max(1, min_f // cfg.rebin_f_target) if cfg.rebin_f_target else 1

    # A short edge segment must not be fully trimmed away by a larger global bin.
    t_bin = min(t_bin, max(1, min_t))
    f_bin = min(f_bin, max(1, min_f))
    return t_bin, f_bin


def _read_polarization_segments_rebinned(
    specs: list,
    cfg: PlotConfig,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
    t_bin: int,
    f_bin: int,
    common_base: datetime.datetime,
    polar_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read all overlapping file segments for one polarization and concatenate in time."""
    z_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None
    last_time = -np.inf

    for spec in specs:
        window = _overlap_window(spec, t_start, t_end)
        if window is None:
            continue
        seg_start, seg_end = window
        print(
            f"  [{polar_label}] using {os.path.basename(spec.source_path)}: "
            f"{seg_start.isoformat()} -> {seg_end.isoformat()}"
        )
        z_seg, tt_seg, freq_seg = spec.read_slice_rebinned(
            t1=seg_start,
            t2=seg_end,
            f1=cfg.f_start,
            f2=cfg.f_end,
            t_bin=t_bin,
            f_bin=f_bin,
            chunk_mem_mb=cfg.chunk_mem_mb,
        )

        # Convert each file's time seconds to seconds relative to one common base.
        base_offset = (spec.dt_base - common_base).total_seconds()
        tt_common = tt_seg.astype(np.float64) + base_offset

        # If two FITS files overlap, keep the earlier file and remove duplicate/backward columns.
        keep = tt_common > last_time
        if not np.any(keep):
            print(
                f"  [{polar_label}] skipped fully overlapping segment: {spec.source_path}"
            )
            continue
        if not np.all(keep):
            removed = int(np.count_nonzero(~keep))
            print(f"  [{polar_label}] removed {removed} overlapping time column(s).")
            z_seg = z_seg[:, keep]
            tt_common = tt_common[keep]

        if freq_ref is None:
            freq_ref = freq_seg
        elif freq_ref.shape != freq_seg.shape or not np.allclose(
            freq_ref, freq_seg, rtol=0.0, atol=1e-3
        ):
            raise ValueError(
                "Frequency axes are inconsistent between FITS files. "
                "Please regrid/interpolate frequency first or use files from the same CSO mode.\n"
                f"Reference: {freq_ref[0]:.6f}~{freq_ref[-1]:.6f} MHz, "
                f"current: {freq_seg[0]:.6f}~{freq_seg[-1]:.6f} MHz, "
                f"file={spec.source_path}"
            )

        z_parts.append(z_seg)
        t_parts.append(tt_common)
        last_time = float(tt_common[-1])

    if not z_parts or freq_ref is None:
        raise ValueError(
            f"No usable {polar_label} segment found in the requested range."
        )

    z_all = np.concatenate(z_parts, axis=1)
    t_all = np.concatenate(t_parts)

    if t_all.size >= 3:
        dt = np.diff(t_all)
        median_dt = float(np.nanmedian(dt))
        if median_dt > 0 and float(np.nanmax(dt)) > 2.5 * median_dt:
            warnings.warn(
                f"Detected a possible time gap in merged {polar_label} data: "
                f"median dt={median_dt:.6g}s, max dt={float(np.nanmax(dt)):.6g}s. "
                "imshow uses a continuous time extent, so inspect the file boundary if the plot looks stretched.",
                stacklevel=2,
            )

    return z_all, t_all, freq_ref


def calc_polarization_ratio(
    Z_r: np.ndarray, Z_l: np.ndarray, eps: float = 1e-30
) -> np.ndarray:
    """
    Calculate polarization ratio (R-L)/(R+L) with safe division.

    Note:
        The result ranges from -1 to 1:
        - Positive values indicate R > L (right-handed dominance)
        - Negative values indicate L > R (left-handed dominance)
        - Zero indicates equal intensity (unpolarized)

    Color mapping:
        - 'bwr' colormap: blue for negative, white for 0, red for positive
        - This ensures correct visual representation of polarization direction
    """
    denom = Z_r + Z_l
    valid = (
        np.isfinite(Z_r) & np.isfinite(Z_l) & np.isfinite(denom) & (np.abs(denom) > eps)
    )
    ratio = np.full_like(Z_r, np.nan, dtype=np.float32)
    ratio[valid] = np.clip(
        (Z_r[valid] - Z_l[valid]) / denom[valid],
        -1.0,
        1.0,
    ).astype(np.float32)
    return ratio


def _safe_log10(arr: np.ndarray) -> np.ndarray:
    """Compute base-10 logarithm safely, handling non-positive values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.where(arr > 0, arr, np.nan))


def get_color_limits(
    data: np.ndarray, cfg: PlotConfig, plot_type: str = "ll"
) -> tuple[float, float]:
    """
    Get color scale limits based on configuration.

    Args:
        data: Input data array
        cfg: Plot configuration
        plot_type: Type of plot - "ll", "rr", "sum", or "ratio"

    Returns:
        Tuple of (vmin, vmax)
    """
    # Check if use_percentile_clipping attribute exists and is enabled
    if hasattr(cfg, "use_percentile_clipping") and not cfg.use_percentile_clipping:
        # Manual limits mode
        vmin = None
        vmax = None

        if plot_type == "ll":
            # Try individual LL limits first
            if hasattr(cfg, "manual_ll_vmin") and cfg.manual_ll_vmin is not None:
                vmin = cfg.manual_ll_vmin
            elif hasattr(cfg, "manual_vmin") and cfg.manual_vmin is not None:
                vmin = cfg.manual_vmin

            if hasattr(cfg, "manual_ll_vmax") and cfg.manual_ll_vmax is not None:
                vmax = cfg.manual_ll_vmax
            elif hasattr(cfg, "manual_vmax") and cfg.manual_vmax is not None:
                vmax = cfg.manual_vmax

        elif plot_type == "rr":
            # Try individual RR limits first
            if hasattr(cfg, "manual_rr_vmin") and cfg.manual_rr_vmin is not None:
                vmin = cfg.manual_rr_vmin
            elif hasattr(cfg, "manual_vmin") and cfg.manual_vmin is not None:
                vmin = cfg.manual_vmin

            if hasattr(cfg, "manual_rr_vmax") and cfg.manual_rr_vmax is not None:
                vmax = cfg.manual_rr_vmax
            elif hasattr(cfg, "manual_vmax") and cfg.manual_vmax is not None:
                vmax = cfg.manual_vmax

        elif plot_type == "sum":
            if hasattr(cfg, "manual_sum_vmin"):
                vmin = cfg.manual_sum_vmin
            if hasattr(cfg, "manual_sum_vmax"):
                vmax = cfg.manual_sum_vmax

        elif plot_type == "ratio":
            if hasattr(cfg, "manual_ratio_vmin"):
                vmin = cfg.manual_ratio_vmin
            if hasattr(cfg, "manual_ratio_vmax"):
                vmax = cfg.manual_ratio_vmax
            # ✅ FIX: Enforce symmetric color scale so that 0 always maps to
            # white in the 'bwr' colormap, regardless of user-supplied limits.
            if vmin is not None and vmax is not None:
                max_abs = max(abs(vmin), abs(vmax))
                vmin = -max_abs
                vmax = max_abs

        # If manual limits are not set, fall back to data range
        if vmin is None or vmax is None:
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                vmin = vmin if vmin is not None else np.nanmin(valid_data)
                vmax = vmax if vmax is not None else np.nanmax(valid_data)
            else:
                vmin, vmax = 0.0, 1.0
        return float(vmin), float(vmax)
    else:
        # Percentile-based clipping mode (default)
        if plot_type == "sum":
            vmin = np.nanpercentile(data, cfg.sum_vmin_pct)
            vmax = np.nanpercentile(data, cfg.sum_vmax_pct)
        elif plot_type == "ratio":
            vmin = np.nanpercentile(data, cfg.ratio_vmin_pct)
            vmax = np.nanpercentile(data, cfg.ratio_vmax_pct)
        else:
            # For LL and RR, use the same percentiles
            vmin = np.nanpercentile(data, cfg.vmin_pct)
            vmax = np.nanpercentile(data, cfg.vmax_pct)
        return float(vmin), float(vmax)


def _datetime_list_from_seconds(
    dt_base: datetime.datetime, seconds_array: np.ndarray
) -> list[datetime.datetime]:
    return [
        dt_base + datetime.timedelta(seconds=float(seconds))
        for seconds in seconds_array
    ]


def _prepare_imshow_extent(
    data: np.ndarray,
    time_seconds: np.ndarray,
    freq: np.ndarray,
    dt_base: datetime.datetime,
) -> tuple[np.ndarray, list[float], list[datetime.datetime], np.ndarray, bool]:
    """Prepare image data, datetime x extent, and ascending y extent for imshow."""
    freq_flipped = False
    if freq.size >= 2 and float(freq[0]) > float(freq[-1]):
        data = data[::-1, :].copy()
        freq = freq[::-1].copy()
        freq_flipped = True

    f_min = float(np.nanmin(freq))
    f_max = float(np.nanmax(freq))
    dt_list = _datetime_list_from_seconds(dt_base, time_seconds)
    x_start_num = mdates.date2num(dt_list[0])
    x_end_num = mdates.date2num(dt_list[-1])
    extent = [x_start_num, x_end_num, f_min, f_max]
    return data, extent, dt_list, freq, freq_flipped


def _configure_datetime_axis(
    ax, cfg: PlotConfig, x_start_num: float, x_end_num: float
) -> None:
    ax.xaxis_date()
    if cfg.xtick_interval is not None:
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=cfg.xtick_interval))
    else:
        span_seconds = max((x_end_num - x_start_num) * 86400.0, 1.0)
        interval = AxisConfigManager.calculate_optimal_ticks(span_seconds, max_ticks=15)
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(interval))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(cfg.xtick_format))
    ax.set_xlim(x_start_num, x_end_num)


def _get_ratio_norm_and_limits(ratio: np.ndarray, cfg: PlotConfig):
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        raise ValueError("Polarization ratio contains no finite values")

    if cfg.ratio_auto_symmetric_scale:
        raw_abs = float(np.nanpercentile(np.abs(finite), cfg.ratio_abs_percentile))
        vmax_abs = float(
            np.clip(raw_abs, cfg.ratio_min_abs_limit, cfg.ratio_max_abs_limit)
        )
        if raw_abs < cfg.ratio_min_abs_limit:
            print(
                "  Ratio is physically close to zero; using enhanced symmetric "
                f"display range +/-{vmax_abs:.4f}."
            )
    else:
        vmin, vmax = get_color_limits(ratio, cfg, "ratio")
        vmax_abs = max(abs(vmin), abs(vmax))

    vmin = -vmax_abs
    vmax = vmax_abs
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), vmin, vmax


def _resolve_output_path(
    cfg: PlotConfig,
    t_start_dt: datetime.datetime,
    t_end_dt: datetime.datetime,
    items: list[dict],
) -> str:
    raw = (cfg.save_path or "").strip()
    if not raw:
        return ""

    tags = [item.get("plot_type", "") for item in items if item.get("plot_type")]
    plot_tags = "_".join(tags) if tags else "spectrogram"
    scale_tag = "auto" if cfg.use_percentile_clipping else "manual"
    from solar_apps.workflows.common.image_naming import build_scientific_image_filename

    auto_name = build_scientific_image_filename(
        sequence=1,
        start_time=t_start_dt,
        end_time=t_end_dt,
        instrument="cso",
        product="dynamic_spectrum",
        qualifiers=(
            plot_tags,
            f"{float(cfg.f_start):g}_to_{float(cfg.f_end):g}mhz",
            scale_tag,
        ),
        generated_at=datetime.datetime.now(datetime.timezone.utc),
    )

    img_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff"}
    ext = os.path.splitext(raw)[1].lower()
    if ext in img_exts:
        os.makedirs(os.path.dirname(raw) or ".", exist_ok=True)
        return raw
    if os.path.isdir(raw):
        return os.path.join(raw, auto_name)

    os.makedirs(raw, exist_ok=True)
    return os.path.join(raw, auto_name)


# ============================================================
#  DRIFT-RATE SELECTION / MEASUREMENT / OVERLAY
# ============================================================


def _cfg_get(cfg, key: str, default=None):
    """Read a value from either a dataclass-style config or a dict-style config."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg, key: str, value) -> None:
    """Set a value on either a dataclass-style config or a dict-style config."""
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _drift_interactive_cfg(cfg) -> dict:
    return dict(_cfg_get(cfg, "drift_rate_interactive", {}) or {})


def _date_num_to_datetime(value: float) -> datetime.datetime:
    """Convert Matplotlib date number to naive UTC datetime."""
    dt = mdates.num2date(float(value))
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt


def _parse_datetime_value(value) -> datetime.datetime | None:
    """Parse common FITS/config/browser datetime strings into naive datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time())
    text = str(value).strip()
    if not text or text.lower() == "none" or text == "Unknown":
        return None
    text = text.replace("Z", "").replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H%M%S.%f",
        "%Y-%m-%d %H%M%S",
        "%Y%m%d %H%M%S.%f",
        "%Y%m%d %H%M%S",
        "%Y%m%dT%H%M%S.%f",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


def _datetime_to_unix_ms_utc(dt_value: datetime.datetime) -> int:
    """Treat naive datetimes as UTC for stable browser/Python time mapping."""
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=datetime.timezone.utc)
    return int(dt_value.timestamp() * 1000)


def _parse_drift_datetime(value, fallback_num=None) -> datetime.datetime:
    if fallback_num is not None:
        try:
            return _date_num_to_datetime(float(fallback_num))
        except Exception:
            pass
    parsed = _parse_datetime_value(value)
    if parsed is None:
        raise ValueError(f"Cannot parse drift-rate datetime: {value!r}")
    return parsed


def calculate_drift_rate_from_line(line: dict) -> DriftRateResult:
    """Calculate df/dt with the historical CSO endpoint semantics."""
    return _canonical_drift_rate._calculate_drift_rate_from_line(
        line,
        profile=_canonical_drift_rate._CSO_DRIFT_RATE_PROFILE,
        t_start=_parse_drift_datetime(line.get("t_start"), line.get("t_start_num")),
        t_end=_parse_drift_datetime(line.get("t_end"), line.get("t_end_num")),
    )


def _mark_drift_range_warnings(
    results: list[DriftRateResult], cache: DriftSpectrogramView
) -> list[DriftRateResult]:
    if not results:
        return results
    x0, x1 = cache.display_time_nums
    t_min = _date_num_to_datetime(min(x0, x1))
    t_max = _date_num_to_datetime(max(x0, x1))
    f_min = float(np.nanmin(cache.freq))
    f_max = float(np.nanmax(cache.freq))
    for result in results:
        warnings_list = []
        if (
            result.t_start < t_min
            or result.t_start > t_max
            or result.t_end < t_min
            or result.t_end > t_max
        ):
            warnings_list.append("time_out_of_range")
        if (
            result.f_start_mhz < f_min
            or result.f_start_mhz > f_max
            or result.f_end_mhz < f_min
            or result.f_end_mhz > f_max
        ):
            warnings_list.append("frequency_out_of_range")
        if result.duration_s < 0:
            warnings_list.append("negative_duration")
        if warnings_list:
            result.warning = ";".join(warnings_list)
            result.quality_flag = "warning"
    return results


def _drift_output_base_dir(cfg) -> str:
    raw = str(_cfg_get(cfg, "save_path", "") or "").strip()
    if not raw:
        return os.getcwd()
    img_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff"}
    ext = os.path.splitext(raw)[1].lower()
    if ext in img_exts:
        return os.path.dirname(raw) or "."
    return raw


def _drift_output_path(cfg, key: str) -> str:
    path = str(_cfg_get(cfg, key, "") or "").strip()
    if not path:
        path = key
    if os.path.isabs(path):
        return path
    base = _drift_output_base_dir(cfg)
    return os.path.join(base, path)


def save_drift_selection_json(
    path: str, lines: list[dict], cache: DriftSpectrogramView, cfg
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "source_file": cache.source_file,
        "source_files": cache.source_files,
        "time_start": _date_num_to_datetime(cache.display_time_nums[0]).isoformat(),
        "time_end": _date_num_to_datetime(cache.display_time_nums[1]).isoformat(),
        "f_min_mhz": float(np.nanmin(cache.freq)),
        "f_max_mhz": float(np.nanmax(cache.freq)),
        "lines": list(lines or []),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_drift_selection_payload(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"lines": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported drift-rate selection JSON format: {path}")
    payload.setdefault("lines", [])
    return payload


def load_drift_selection_json(path: str) -> list[dict]:
    return list(_load_drift_selection_payload(path).get("lines", []) or [])


def _spectrogram_coord_from_pixel(metadata: dict, x_px: float, y_px: float) -> dict:
    """Map preview-image pixel coordinates back to spectrum time/frequency coordinates."""
    bbox = metadata["axes_bbox_px"]
    if not (
        bbox["left"] <= x_px <= bbox["right"] and bbox["top"] <= y_px <= bbox["bottom"]
    ):
        raise ValueError("Pixel is outside the spectrogram axes")
    xf = (x_px - bbox["left"]) / (bbox["right"] - bbox["left"])
    yf = (y_px - bbox["top"]) / (bbox["bottom"] - bbox["top"])
    xnum = metadata["x_start_num"] + xf * (
        metadata["x_end_num"] - metadata["x_start_num"]
    )
    freq_val = metadata["f_max_mhz"] - yf * (
        metadata["f_max_mhz"] - metadata["f_min_mhz"]
    )
    return {
        "time_num": float(xnum),
        "time_iso": _date_num_to_datetime(xnum).isoformat(timespec="milliseconds"),
        "frequency_mhz": float(freq_val),
    }


def render_spectrogram_selection_preview(
    cache: DriftSpectrogramView, cfg
) -> tuple[str, dict]:
    """Render a stable PNG preview used by the browser-based endpoint selector."""
    metadata_path = _drift_output_path(cfg, "drift_rate_selection_metadata_json")

    if not _cfg_get(cfg, "show_plot", False):
        plt.switch_backend("Agg")

    fig_width = float(_cfg_get(cfg, "fig_width", 12.0))
    fig_height = max(3.2, float(_cfg_get(cfg, "fig_height_per", 3.0)) * 1.25)
    dpi = int(_cfg_get(cfg, "dpi", 300))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    x0, x1 = cache.display_time_nums
    from solar_apps.workflows.common.image_naming import (
        configured_scientific_image_path,
    )

    configured_name = _cfg_get(
        cfg,
        "drift_rate_selection_preview_png",
        "drift_rate_selection_preview",
    )
    preview_path = str(
        configured_scientific_image_path(
            configured_name,
            output_dir=_drift_output_base_dir(cfg),
            sequence=1,
            start_time=_date_num_to_datetime(x0),
            end_time=_date_num_to_datetime(x1),
            instrument="cso",
            product=str(configured_name or "drift_rate_selection_preview"),
            generated_at=datetime.datetime.now(datetime.timezone.utc),
        )
    )
    os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    f_min = float(np.nanmin(cache.freq))
    f_max = float(np.nanmax(cache.freq))
    im = ax.imshow(
        cache.data,
        extent=[x0, x1, f_min, f_max],
        origin="lower",
        aspect="auto",
        cmap=cache.cmap,
        vmin=cache.vmin,
        vmax=cache.vmax,
    )
    ax.set_title(cache.title)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Frequency (MHz)")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(_cfg_get(cfg, "xtick_format", "%H:%M:%S"))
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(f_min, f_max)
    fig.colorbar(im, ax=ax, pad=0.01, label=cache.cbar_label)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.canvas.draw()

    bbox = ax.get_window_extent()
    fig_width_px = int(round(fig.bbox.width))
    fig_height_px = int(round(fig.bbox.height))
    axes_bbox_px = {
        "left": float(bbox.x0),
        "right": float(bbox.x1),
        "top": float(fig_height_px - bbox.y1),
        "bottom": float(fig_height_px - bbox.y0),
    }

    dt0 = _date_num_to_datetime(x0)
    dt1 = _date_num_to_datetime(x1)
    metadata = {
        "fig_width_px": fig_width_px,
        "fig_height_px": fig_height_px,
        "axes_bbox_px": axes_bbox_px,
        "x_start_num": float(x0),
        "x_end_num": float(x1),
        "x_start_unix_ms": _datetime_to_unix_ms_utc(dt0),
        "x_end_unix_ms": _datetime_to_unix_ms_utc(dt1),
        "f_min_mhz": f_min,
        "f_max_mhz": f_max,
        "source_file": cache.source_file,
    }

    fig.savefig(preview_path, dpi=dpi)
    plt.close(fig)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return preview_path, metadata


def _drift_selection_html(metadata: dict, interactive: dict) -> str:
    metadata_json = json.dumps(metadata)
    interactive_json = json.dumps(interactive)
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CSO drift-rate endpoint selection</title>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: #202124; color: #f1f3f4; }
.bar { display: flex; gap: 8px; align-items: center; padding: 10px 14px; background: #111; position: sticky; top: 0; z-index: 3; }
button { padding: 7px 10px; border: 1px solid #555; background: #2d2f31; color: white; cursor: pointer; border-radius: 4px; }
button:hover { background: #3a3d40; }
#status { margin-left: auto; color: #d7e3fc; white-space: nowrap; }
#wrap { position: relative; display: inline-block; margin: 14px; }
#spec { display: block; max-width: calc(100vw - 28px); height: auto; }
#overlay { position: absolute; left: 0; top: 0; pointer-events: auto; }
#hint { padding: 0 14px 12px; color: #c9d1d9; line-height: 1.5; }
</style>
</head>
<body>
<div class="bar">
  <button id="undo">Undo last point</button>
  <button id="delete">Delete last line</button>
  <button id="clear">Clear all</button>
  <button id="check">Check mapping</button>
  <button id="save">Save</button>
  <button id="finish">Save & Continue</button>
  <span id="status">Move over the spectrum</span>
</div>
<div id="wrap">
  <img id="spec" src="/preview.png" alt="spectrogram">
  <canvas id="overlay"></canvas>
</div>
<div id="hint">
  Click two points for each drift-rate line. The first point is the start point and the second point is the end point.
  A preview line follows the mouse before the end point is fixed. Top = high frequency, bottom = low frequency.
</div>
<script>
const metadata = __METADATA_JSON__;
const interactive = __INTERACTIVE_JSON__;
const img = document.getElementById('spec');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
let points = [];
let lines = [];
let currentMouse = null;
const colors = interactive.line_color_cycle || ['white','cyan','lime','yellow','magenta','orange'];

function scaleInfo() {
  return {sx: img.clientWidth / metadata.fig_width_px, sy: img.clientHeight / metadata.fig_height_px};
}
function resizeCanvas() {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
  draw();
}
function eventPixel(ev) {
  const rect = img.getBoundingClientRect();
  const s = scaleInfo();
  return {x: (ev.clientX - rect.left) / s.sx, y: (ev.clientY - rect.top) / s.sy};
}
function inAxes(p) {
  const b = metadata.axes_bbox_px;
  return p.x >= b.left && p.x <= b.right && p.y >= b.top && p.y <= b.bottom;
}
function mapCoord(p) {
  const b = metadata.axes_bbox_px;
  const xf = (p.x - b.left) / (b.right - b.left);
  const yf = (p.y - b.top) / (b.bottom - b.top);
  const xnum = metadata.x_start_num + xf * (metadata.x_end_num - metadata.x_start_num);
  const f = metadata.f_max_mhz - yf * (metadata.f_max_mhz - metadata.f_min_mhz);
  const unix_ms = metadata.x_start_unix_ms + xf * (metadata.x_end_unix_ms - metadata.x_start_unix_ms);
  const iso = new Date(unix_ms).toISOString().replace('Z','');
  return {time_num: xnum, time_iso: iso, frequency_mhz: f};
}
function fmtTime(iso) {
  const t = iso.split('T')[1] || iso.split(' ')[1] || iso;
  return t.substring(0, 12);
}
function drift(a, b) {
  const dt = (b.time_num - a.time_num) * 86400.0;
  return (b.frequency_mhz - a.frequency_mhz) / dt;
}
function drawPoint(p, color) {
  const s = scaleInfo();
  ctx.beginPath();
  ctx.arc(p.x * s.sx, p.y * s.sy, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#000';
  ctx.stroke();
}
function drawLineObj(line, idx) {
  const s = scaleInfo();
  const a = line._p1, b = line._p2;
  ctx.strokeStyle = line.color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(a.x * s.sx, a.y * s.sy);
  ctx.lineTo(b.x * s.sx, b.y * s.sy);
  ctx.stroke();
  drawPoint(a, line.color);
  drawPoint(b, line.color);
  const mx = (a.x + b.x) * 0.5 * s.sx + 8;
  const my = (a.y + b.y) * 0.5 * s.sy - 8 - idx * 4;
  ctx.font = '13px Arial';
  const text = `${line.label} df/dt=${line.rate.toFixed(2)} MHz/s`;
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(mx - 4, my - 15, w, 19);
  ctx.fillStyle = line.color;
  ctx.fillText(text, mx, my);
}
function drawPreviewLine() {
  if (!interactive.show_preview_line) return;
  if (points.length !== 1 || !currentMouse || !inAxes(currentMouse)) return;
  const s = scaleInfo();
  const a = points[0];
  const b = currentMouse;
  const color = colors[lines.length % colors.length];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.0;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(a.x * s.sx, a.y * s.sy);
  ctx.lineTo(b.x * s.sx, b.y * s.sy);
  ctx.stroke();
  ctx.setLineDash([]);
  drawPoint(a, color);
  drawPoint(b, color);
  const ca = a.coord;
  const cb = mapCoord(b);
  const dt = (cb.time_num - ca.time_num) * 86400.0;
  if (Math.abs(dt) > 1e-9) {
    const rate = (cb.frequency_mhz - ca.frequency_mhz) / dt;
    const mx = (a.x + b.x) * 0.5 * s.sx + 8;
    const my = (a.y + b.y) * 0.5 * s.sy - 8;
    const text = `preview df/dt=${rate.toFixed(2)} MHz/s`;
    ctx.font = '13px Arial';
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(mx - 4, my - 15, w, 19);
    ctx.fillStyle = color;
    ctx.fillText(text, mx, my);
  }
  ctx.restore();
}
function drawCrosshair(p) {
  if (!interactive.show_crosshair || !p || !inAxes(p)) return;
  const s = scaleInfo();
  const b = metadata.axes_bbox_px;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(b.left * s.sx, p.y * s.sy);
  ctx.lineTo(b.right * s.sx, p.y * s.sy);
  ctx.moveTo(p.x * s.sx, b.top * s.sy);
  ctx.lineTo(p.x * s.sx, b.bottom * s.sy);
  ctx.stroke();
  ctx.restore();
}
function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  lines.forEach(drawLineObj);
  points.forEach(p => drawPoint(p, colors[lines.length % colors.length]));
  drawPreviewLine();
  drawCrosshair(currentMouse);
}
function addPoint(p) {
  const coord = mapCoord(p);
  p.coord = coord;
  points.push(p);
  if (points.length === 1) {
    statusEl.textContent = 'Start point fixed. Move mouse to preview line; click again to set end point.';
  }
  if (points.length === 2) {
    if (!interactive.allow_multiple_lines) lines = [];
    const idx = lines.length + 1;
    const color = colors[(idx - 1) % colors.length];
    const rate = drift(points[0].coord, points[1].coord);
    const label = 'drift_' + String(idx).padStart(3, '0');
    lines.push({
      label, color, rate,
      t_start: points[0].coord.time_iso,
      t_start_num: points[0].coord.time_num,
      f_start_mhz: points[0].coord.frequency_mhz,
      t_end: points[1].coord.time_iso,
      t_end_num: points[1].coord.time_num,
      f_end_mhz: points[1].coord.frequency_mhz,
      mode: 'manual_endpoint',
      note: '',
      _p1: points[0],
      _p2: points[1]
    });
    points = [];
    statusEl.textContent = `${label} saved. Move to the next start point or Save & Continue.`;
  }
  draw();
}
img.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
canvas.addEventListener('mousemove', ev => {
  const p = eventPixel(ev);
  if (!inAxes(p)) {
    currentMouse = null;
    statusEl.textContent = 'outside axes';
    draw();
    return;
  }
  currentMouse = p;
  const c = mapCoord(p);
  statusEl.textContent = `Time: ${fmtTime(c.time_iso)}   Frequency: ${c.frequency_mhz.toFixed(1)} MHz`;
  draw();
});
canvas.addEventListener('mouseleave', () => { currentMouse = null; draw(); });
canvas.addEventListener('click', ev => {
  const p = eventPixel(ev);
  if (!inAxes(p)) {
    statusEl.textContent = 'Click ignored: outside axes';
    return;
  }
  currentMouse = p;
  addPoint(p);
});
document.getElementById('undo').onclick = () => { if (points.length > 0) points.pop(); currentMouse = null; draw(); };
document.getElementById('delete').onclick = () => { lines.pop(); draw(); };
document.getElementById('clear').onclick = () => { points = []; lines = []; currentMouse = null; draw(); };
document.getElementById('check').onclick = () => {
  const b = metadata.axes_bbox_px;
  const top = mapCoord({x:b.left, y:b.top});
  const bottom = mapCoord({x:b.left, y:b.bottom});
  statusEl.textContent = `Mapping check: top=${top.frequency_mhz.toFixed(1)} MHz, bottom=${bottom.frequency_mhz.toFixed(1)} MHz`;
};
async function post(path) {
  const payload = lines.map(l => ({
    label: l.label, color: l.color, mode: l.mode,
    t_start: l.t_start, t_start_num: l.t_start_num, f_start_mhz: l.f_start_mhz,
    t_end: l.t_end, t_end_num: l.t_end_num, f_end_mhz: l.f_end_mhz,
    note: l.note || ''
  }));
  const resp = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({lines: payload})});
  statusEl.textContent = await resp.text();
}
document.getElementById('save').onclick = () => post('/api/save');
document.getElementById('finish').onclick = () => post('/api/finish');
resizeCanvas();
</script>
</body>
</html>"""
    return html.replace("__METADATA_JSON__", metadata_json).replace(
        "__INTERACTIVE_JSON__", interactive_json
    )


def launch_drift_selection_server(cache: DriftSpectrogramView, cfg) -> list[dict]:
    preview_path, metadata = render_spectrogram_selection_preview(cache, cfg)
    selection_path = _drift_output_path(cfg, "drift_rate_selection_json")
    os.makedirs(os.path.dirname(selection_path) or ".", exist_ok=True)
    interactive = _drift_interactive_cfg(cfg)
    host = str(interactive.get("host", "127.0.0.1"))
    requested_port = int(interactive.get("port", 8050))
    auto_increment = bool(interactive.get("auto_increment_port", True))
    max_tries = max(1, int(interactive.get("max_port_tries", 20) or 20))
    done_event = threading.Event()
    state = {"lines": []}

    class DriftSelectionHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, status, content, content_type="text/plain; charset=utf-8"):
            body = content.encode("utf-8") if isinstance(content, str) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(
                    200,
                    _drift_selection_html(metadata, interactive),
                    "text/html; charset=utf-8",
                )
            elif path == "/preview.png":
                with open(preview_path, "rb") as handle:
                    self._send(200, handle.read(), "image/png")
            elif path == "/metadata.json":
                self._send(200, json.dumps(metadata), "application/json")
            else:
                self._send(404, "not found")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            lines = payload.get("lines", [])
            state["lines"] = lines
            save_drift_selection_json(selection_path, lines, cache, cfg)
            if path == "/api/finish":
                done_event.set()
                self._send(200, f"Saved {len(lines)} line(s). You can close this tab.")
            elif path == "/api/save":
                self._send(200, f"Saved {len(lines)} line(s).")
            else:
                self._send(404, "not found")

    server = None
    last_error = None
    for candidate_port in range(requested_port, requested_port + max_tries):
        try:
            server = ThreadingHTTPServer((host, candidate_port), DriftSelectionHandler)
            port = candidate_port
            break
        except OSError as exc:
            last_error = exc
            if not auto_increment:
                break
    if server is None:
        end_port = requested_port + max_tries - 1
        raise OSError(
            f"Cannot start drift selection server. Ports {requested_port}-{end_port} are unavailable."
        ) from last_error

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{port}"
    print("=" * 70)
    print("[Drift selection] Interactive endpoint selector is running")
    print(f"[Drift selection] URL: {url}")
    print(f"[Drift selection] Preview PNG: {preview_path}")
    print(
        f"[Drift selection] Metadata JSON: {_drift_output_path(cfg, 'drift_rate_selection_metadata_json')}"
    )
    print(f"[Drift selection] Selection JSON: {selection_path}")
    print("[Drift selection] Click two points for each drift-rate line.")
    print("[Drift selection] Click 'Save & Continue' to return to Python.")
    print("=" * 70)
    if interactive.get("auto_open_browser", True):
        opened = webbrowser.open(url)
        if not opened:
            print(f"[Drift selection] Browser did not open; copy this URL: {url}")

    try:
        if interactive.get("block_until_done", True):
            timeout = float(interactive.get("selection_timeout_seconds", 0) or 0)
            finished = done_event.wait(timeout if timeout > 0 else None)
            if not finished:
                raise TimeoutError("Drift-rate selection timed out")
    finally:
        server.shutdown()
        thread.join(timeout=5)

    if not state["lines"] and os.path.exists(selection_path):
        state["lines"] = load_drift_selection_json(selection_path)
    return list(state["lines"] or [])


def get_or_load_drift_rate_results(
    cache: DriftSpectrogramView, cfg, launch_func=None
) -> list[DriftRateResult]:
    if not _cfg_get(cfg, "enable_drift_rate_overlay", False):
        return []
    mode = str(_cfg_get(cfg, "drift_rate_mode", "off") or "off").lower()
    if mode == "off":
        return []
    launch_func = launch_func or launch_drift_selection_server
    selection_path = _drift_output_path(cfg, "drift_rate_selection_json")
    cli_path = _cfg_get(cfg, "_drift_selection_cli_path", None)
    if cli_path:
        selection_path = str(cli_path)
    interactive = _drift_interactive_cfg(cfg)
    launch_policy = str(interactive.get("launch_policy", "cli_only") or "cli_only")
    cache_key = (mode, os.path.abspath(selection_path), launch_policy)
    if cache_key in _DRIFT_RATE_RESULTS_CACHE:
        return _DRIFT_RATE_RESULTS_CACHE[cache_key]

    selection_exists = os.path.exists(selection_path)
    export_preview = bool(_cfg_get(cfg, "export_drift_selection_preview", False))

    if mode == "interactive_manual":
        select_now = bool(_cfg_get(cfg, "_select_drift_now", False))
        if select_now or (launch_policy == "always" and not export_preview):
            lines = launch_func(cache, cfg)
        elif (
            launch_policy == "auto_if_missing"
            and not selection_exists
            and not export_preview
        ):
            print(
                "[Drift selection] selection JSON not found; starting interactive selector..."
            )
            lines = launch_func(cache, cfg)
        elif selection_exists:
            payload = _load_drift_selection_payload(selection_path)
            source_file = payload.get("source_file")
            if source_file and os.path.abspath(str(source_file)) != os.path.abspath(
                cache.source_file
            ):
                warnings.warn(
                    "Drift-rate selection source_file differs from current spectrogram data.",
                    stacklevel=2,
                )
            lines = list(payload.get("lines", []) or [])
        else:
            if interactive.get("print_usage_hint", True):
                print(
                    "[Drift selection] No selection JSON found. Run:\n"
                    "  python cso_radio_spectrogram_plot.py --select-drift --drift-port 8050\n"
                    "or set cfg.drift_rate_interactive['launch_policy'] = 'auto_if_missing'."
                )
            return []
    elif mode == "manual_json":
        if not selection_exists:
            warnings.warn(
                f"No drift-rate selection JSON found for manual_json mode: {selection_path}",
                stacklevel=2,
            )
            return []
        lines = load_drift_selection_json(selection_path)
    else:
        warnings.warn(
            f"Unsupported drift_rate_mode={mode!r}; drift overlay disabled.",
            stacklevel=2,
        )
        return []

    results = [calculate_drift_rate_from_line(line) for line in lines]
    results = _mark_drift_range_warnings(results, cache)
    _DRIFT_RATE_RESULTS_CACHE[cache_key] = results
    return results


def overlay_drift_rate_results(ax, results: list[DriftRateResult], cfg) -> None:
    if not results:
        return
    color_cycle = _drift_interactive_cfg(cfg).get(
        "line_color_cycle", ["white", "cyan", "lime", "yellow", "magenta", "orange"]
    )
    line_width = float(_cfg_get(cfg, "drift_rate_line_width", 2.2))
    endpoint_marker = _cfg_get(cfg, "drift_rate_endpoint_marker", "o")
    endpoint_size = float(_cfg_get(cfg, "drift_rate_endpoint_size", 30.0))
    for idx, result in enumerate(results):
        color = result.color or color_cycle[idx % len(color_cycle)]
        x1 = mdates.date2num(result.t_start)
        x2 = mdates.date2num(result.t_end)
        if _cfg_get(cfg, "draw_drift_rate_lines", True):
            ax.plot(
                [x1, x2],
                [result.f_start_mhz, result.f_end_mhz],
                color=color,
                linewidth=line_width,
                alpha=0.95,
                clip_on=True,
                zorder=4,
            )
        if _cfg_get(cfg, "draw_drift_rate_endpoints", True):
            ax.scatter(
                [x1, x2],
                [result.f_start_mhz, result.f_end_mhz],
                marker=endpoint_marker,
                s=endpoint_size,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                clip_on=True,
                zorder=5,
            )
        if _cfg_get(cfg, "draw_drift_rate_label", True):
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (result.f_start_mhz + result.f_end_mhz)
            label = _cfg_get(
                cfg, "drift_rate_label_format", "{label}: df/dt={drift_rate:.2f} MHz/s"
            ).format(
                label=result.label,
                drift_rate=result.drift_rate_mhz_s,
                abs_drift_rate=result.abs_drift_rate_mhz_s,
            )
            if result.warning:
                label = f"{label} ({result.warning})"
            ax.annotate(
                label,
                xy=(xm, ym),
                xytext=(8, 8 + 5 * (idx % 3)),
                textcoords="offset points",
                color=color,
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
                clip_on=True,
                zorder=6,
            )


def save_drift_rate_diagnostics_once(
    results: list[DriftRateResult], cfg, source_file: str
) -> None:
    if not results or not _cfg_get(cfg, "save_drift_rate_diagnostics", True):
        return
    csv_path = _drift_output_path(cfg, "drift_rate_diagnostics_csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    key = (os.path.abspath(csv_path), str(source_file))
    if key in _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS:
        return
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DRIFT_RATE_DIAGNOSTIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_file": source_file,
                    "label": result.label,
                    "mode": result.mode,
                    "t_start": result.t_start.isoformat(timespec="milliseconds"),
                    "t_end": result.t_end.isoformat(timespec="milliseconds"),
                    "f_start_mhz": result.f_start_mhz,
                    "f_end_mhz": result.f_end_mhz,
                    "duration_s": result.duration_s,
                    "bandwidth_mhz": result.bandwidth_mhz,
                    "drift_rate_mhz_s": result.drift_rate_mhz_s,
                    "abs_drift_rate_mhz_s": result.abs_drift_rate_mhz_s,
                    "color": result.color,
                    "quality_flag": result.quality_flag,
                    "warning": result.warning,
                }
            )
    _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS.add(key)
    print(f"[Drift selection] Diagnostics CSV: {os.path.abspath(csv_path)}")


def _build_drift_spectrogram_view(
    cfg: PlotConfig,
    items: list[dict],
    display_data_list: list[np.ndarray],
    extent_list: list[list[float]],
    freq_out: np.ndarray,
) -> DriftSpectrogramView | None:
    """Use the plotted SUM spectrum, when available, as the measurement background."""
    if not items or not display_data_list or not extent_list:
        return None

    preferred = None
    for index, item in enumerate(items):
        if item.get("plot_type") == "sum":
            preferred = index
            break
    if preferred is None:
        for index, item in enumerate(items):
            if item.get("plot_type") != "ratio":
                preferred = index
                break
    if preferred is None:
        preferred = 0

    item = items[preferred]
    extent = extent_list[preferred]
    data = display_data_list[preferred]
    source_files = []
    try:
        source_files = get_config_file_paths(cfg)
    except Exception:
        source_files = []
    source_file = (
        ";".join(source_files)
        if source_files
        else str(_cfg_get(cfg, "file_path", "unknown"))
    )

    return DriftSpectrogramView(
        data=data,
        time_nums=np.linspace(float(extent[0]), float(extent[1]), data.shape[1]),
        display_time_nums=(float(extent[0]), float(extent[1])),
        freq=np.asarray(freq_out, dtype=float),
        title=str(item.get("title", "CSO dynamic spectrum")),
        cmap=str(item.get("cmap", "jet")),
        vmin=item.get("vmin"),
        vmax=item.get("vmax"),
        cbar_label=str(item.get("cbar_label", "")),
        source_file=source_file,
        source_files=source_files,
    )


def optimize_workers(
    cfg: PlotConfig, data_size_mb: float, chunk_mem_mb: int
) -> tuple[int, float]:
    """
    Optimize number of workers based on available memory, data size, and chunk memory.

    Args:
        cfg: Plot configuration
        data_size_mb: Estimated data size in MB (per polarization)
        chunk_mem_mb: Memory limit per chunk

    Returns:
        Tuple of (optimal_workers, estimated_peak_memory_mb)
    """
    # Maximum workers needed for polarization processing (LL and RR)
    max_needed_workers = 2

    # Get user's worker preference
    user_workers = max_needed_workers
    if hasattr(cfg, "max_workers") and cfg.max_workers is not None:
        user_workers = min(cfg.max_workers, max_needed_workers)

    # Calculate memory requirements more accurately
    # For sequential processing (optimal_workers = 1):
    #   Peak memory = chunk_mem_mb + data_size_mb (one polarization at a time)
    # For parallel processing (optimal_workers = 2):
    #   Peak memory = 2 * chunk_mem_mb + 2 * data_size_mb (both polarizations simultaneously)
    # However, data_size_mb is already per polarization, so we need to adjust

    # Try to get available system memory
    try:
        import psutil

        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB

        # Calculate memory for different worker counts
        memory_for_1_worker = chunk_mem_mb + data_size_mb  # Sequential
        memory_for_2_workers = 2 * (chunk_mem_mb + data_size_mb)  # Parallel

        # Determine optimal workers based on memory constraints
        if user_workers == 1:
            # User explicitly wants 1 worker
            optimal_workers = 1
            estimated_peak_memory = memory_for_1_worker
        else:
            # Check if we can afford parallel processing
            if memory_for_2_workers <= available_memory * 0.8:  # 80% safety margin
                optimal_workers = 2
                estimated_peak_memory = memory_for_2_workers
            elif memory_for_1_worker <= available_memory * 0.8:
                # Can't do parallel, but can do sequential
                optimal_workers = 1
                estimated_peak_memory = memory_for_1_worker
                print(
                    "  Note: Insufficient memory for parallel processing. Switching to sequential."
                )
            else:
                # Even sequential processing exceeds memory limits
                raise MemoryError(
                    f"Insufficient memory. Required: {memory_for_1_worker:.1f} MB, "
                    f"Available: {available_memory:.1f} MB. "
                    f"Try reducing chunk_mem_mb or data range."
                )

        # Additional safety check
        if estimated_peak_memory > available_memory * 0.9:
            warnings.warn(
                f"Estimated peak memory ({estimated_peak_memory:.1f} MB) exceeds 90% of "
                f"available memory ({available_memory:.1f} MB). Consider reducing settings.",
                stacklevel=2,
            )

    except ImportError:
        warnings.warn("psutil not installed, using conservative defaults", stacklevel=2)
        # Conservative default: assume limited memory
        optimal_workers = 1
        estimated_peak_memory = chunk_mem_mb + data_size_mb

    # Ensure at least 1 worker
    optimal_workers = max(1, optimal_workers)

    return optimal_workers, estimated_peak_memory


# ============================================================
#  MAIN PROCESSING AND PLOTTING
# ============================================================


@timing_decorator
def process_and_plot(cfg: PlotConfig, data_list: list):
    """Main processing pipeline: read data, compute derived quantities, and generate plots."""
    validate_config(cfg)
    t_start_dt = _as_datetime(cfg.t_start, "t_start")
    t_end_dt = _as_datetime(cfg.t_end, "t_end")

    cso_l_specs = _select_overlapping_specs(data_list, "LL", t_start_dt, t_end_dt)
    cso_r_specs = _select_overlapping_specs(data_list, "RR", t_start_dt, t_end_dt)
    if not cso_l_specs or not cso_r_specs:
        available = []
        for spec in data_list:
            spec_start, spec_end = _spec_time_bounds(spec)
            available.append(
                f"{spec.polar}: {spec_start.isoformat()} -> {spec_end.isoformat()}  "
                f"{spec.source_path}"
            )
        raise ValueError(
            "Complete LL and RR data not found for the requested time range.\n"
            "Available file coverage:\n" + "\n".join(available)
        )

    _validate_contiguous_time_coverage(cso_l_specs, t_start_dt, t_end_dt, "LL")
    _validate_contiguous_time_coverage(cso_r_specs, t_start_dt, t_end_dt, "RR")

    common_base = min(spec.dt_base for spec in [*cso_l_specs, *cso_r_specs])
    t_bin, f_bin = calc_bin_sizes_for_specs(cso_l_specs, cfg, t_start_dt, t_end_dt)

    estimated_data_size_mb = _estimate_raw_selection_mb(
        cso_l_specs, cfg, t_start_dt, t_end_dt
    )
    optimal_workers, estimated_peak_memory = optimize_workers(
        cfg, estimated_data_size_mb, cfg.chunk_mem_mb
    )

    print("Selected FITS coverage:")
    for spec in [*cso_l_specs, *cso_r_specs]:
        spec_start, spec_end = _spec_time_bounds(spec)
        print(
            f"  - {spec.polar}: {spec_start.isoformat()} -> {spec_end.isoformat()} | "
            f"{os.path.basename(spec.source_path)}"
        )
    print(f"Common time base: {common_base.isoformat()}")
    print(f"Common bin size: t_bin={t_bin}, f_bin={f_bin}")

    print("Memory configuration:")
    print(f"  - Chunk memory limit: {cfg.chunk_mem_mb} MB per worker")
    print(f"  - Estimated data size: {estimated_data_size_mb:.1f} MB")
    print(f"  - Optimal workers: {optimal_workers}")
    print(f"  - Estimated peak memory: {estimated_peak_memory:.1f} MB")

    if estimated_peak_memory > 2000:
        print(
            f"Warning: Estimated peak memory ({estimated_peak_memory:.1f} MB) is high."
        )
        print("   Consider reducing chunk_mem_mb or max_workers.")

    print("Block reading + downsampling + multi-file time concatenation...")

    if optimal_workers == 1:
        print("Processing sequentially with 1 worker for memory control...")
        print("  Processing LL polarization...")
        Z_l, tt, freq = _read_polarization_segments_rebinned(
            cso_l_specs,
            cfg,
            t_start_dt,
            t_end_dt,
            t_bin,
            f_bin,
            common_base,
            "LL",
        )
        import gc

        gc.collect()

        print("  Processing RR polarization...")
        Z_r, tt_r, freq_r = _read_polarization_segments_rebinned(
            cso_r_specs,
            cfg,
            t_start_dt,
            t_end_dt,
            t_bin,
            f_bin,
            common_base,
            "RR",
        )
    else:
        print(f"Processing in parallel with {optimal_workers} workers...")
        with ThreadPoolExecutor(max_workers=optimal_workers) as exe:
            future_to_polar = {
                exe.submit(
                    _read_polarization_segments_rebinned,
                    cso_l_specs,
                    cfg,
                    t_start_dt,
                    t_end_dt,
                    t_bin,
                    f_bin,
                    common_base,
                    "LL",
                ): "LL",
                exe.submit(
                    _read_polarization_segments_rebinned,
                    cso_r_specs,
                    cfg,
                    t_start_dt,
                    t_end_dt,
                    t_bin,
                    f_bin,
                    common_base,
                    "RR",
                ): "RR",
            }
            results_dict = {}
            for future in as_completed(future_to_polar):
                polar_key = future_to_polar[future]
                results_dict[polar_key] = future.result()

            Z_l, tt, freq = results_dict["LL"]
            Z_r, tt_r, freq_r = results_dict["RR"]

    if Z_l.shape != Z_r.shape:
        raise ValueError(
            f"LL/RR shape mismatch after merge: LL={Z_l.shape}, RR={Z_r.shape}"
        )
    if tt.shape != tt_r.shape or not np.allclose(tt, tt_r, rtol=0.0, atol=1e-6):
        raise ValueError("LL/RR time axes are inconsistent after multi-file merging.")
    if freq.shape != freq_r.shape or not np.allclose(freq, freq_r, rtol=0.0, atol=1e-3):
        raise ValueError(
            "LL/RR frequency axes are inconsistent after multi-file merging."
        )

    Z_sum = Z_l + Z_r
    ratio = calc_polarization_ratio(Z_r, Z_l)

    dt_list = _datetime_list_from_seconds(common_base, tt)
    print("Axis diagnostics:")
    print(f"  common dt_base: {common_base.isoformat()}")
    print(f"  tt[0], tt[-1]: {float(tt[0]):.6f}, {float(tt[-1]):.6f}")
    print(f"  first displayed datetime: {dt_list[0].isoformat()}")
    print(f"  last displayed datetime: {dt_list[-1].isoformat()}")
    print(f"  freq[0], freq[-1]: {float(freq[0]):.6f}, {float(freq[-1]):.6f}")

    print("Data diagnostics:")
    _array_stats("LL", Z_l)
    _array_stats("RR", Z_r)
    _array_stats("Z_sum", Z_sum)
    _array_stats("ratio", ratio)

    items = []
    date_str = t_start_dt.strftime("%Y-%m-%d")

    def create_plot_item(data, title, cmap, cbar_label, plot_type="ll"):
        vmin, vmax = get_color_limits(data, cfg, plot_type)
        return dict(
            data=data,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
            plot_type=plot_type,
        )

    if cfg.plot_ll:
        Z_log = _safe_log10(Z_l)
        items.append(
            create_plot_item(
                Z_log,
                f"CSO/CBSm LL {date_str}",
                "jet",
                r"log$_{10}$ Brightness Temp (K)",
                plot_type="ll",
            )
        )

    if cfg.plot_rr:
        Z_log = _safe_log10(Z_r)
        items.append(
            create_plot_item(
                Z_log,
                f"CSO/CBSm RR {date_str}",
                "jet",
                r"log$_{10}$ Brightness Temp (K)",
                plot_type="rr",
            )
        )

    if cfg.plot_sum:
        Z_log = _safe_log10(Z_sum)
        items.append(
            create_plot_item(
                Z_log,
                f"CSO/CBSm LL+RR {date_str}",
                "jet",
                r"log$_{10}$ Brightness Temp (K)",
                plot_type="sum",
            )
        )

    if cfg.plot_ratio:
        items.append(
            dict(
                data=ratio,
                title="CSO/CBSm Polarization Ratio (R-L)/(R+L)",
                cmap="bwr",
                vmin=cfg.manual_ratio_vmin,
                vmax=cfg.manual_ratio_vmax,
                cbar_label="Polarization Ratio (R-L)/(R+L)",
                plot_type="ratio",
            )
        )

    if not items:
        print("No plot items selected, exiting")
        return

    display_data_list = []
    extent_list = []
    freq_out = None
    freq_flipped = False
    for item in items:
        data_out, extent, dt_list_i, freq_out_i, flipped = _prepare_imshow_extent(
            item["data"], tt, freq, common_base
        )
        display_data_list.append(data_out)
        extent_list.append(extent)
        dt_list = dt_list_i
        freq_out = freq_out_i
        freq_flipped = freq_flipped or flipped

    x_start_num, x_end_num = extent_list[0][0], extent_list[0][1]
    f_min, f_max = extent_list[0][2], extent_list[0][3]
    print(f"  frequency flipped: {freq_flipped}")

    drift_results: list[DriftRateResult] = []
    drift_cache = _build_drift_spectrogram_view(
        cfg, items, display_data_list, extent_list, freq_out
    )
    if drift_cache is not None:
        if _cfg_get(cfg, "export_drift_selection_preview", False):
            preview_path, _metadata = render_spectrogram_selection_preview(
                drift_cache, cfg
            )
            print(f"[Drift selection] Preview PNG: {preview_path}")
            print(
                "[Drift selection] Metadata JSON: "
                f"{_drift_output_path(cfg, 'drift_rate_selection_metadata_json')}"
            )
        drift_results = get_or_load_drift_rate_results(drift_cache, cfg)
        if drift_results:
            save_drift_rate_diagnostics_once(
                drift_results, cfg, drift_cache.source_file
            )
            print("Drift-rate measurements:")
            for result in drift_results:
                print(
                    f"  - {result.label}: "
                    f"{result.f_start_mhz:.2f}->{result.f_end_mhz:.2f} MHz, "
                    f"duration={result.duration_s:.3f} s, "
                    f"df/dt={result.drift_rate_mhz_s:.4f} MHz/s"
                )

    if not cfg.show_plot:
        plt.switch_backend("Agg")

    n = len(items)
    fig, axs = plt.subplots(
        n,
        1,
        figsize=(cfg.fig_width, cfg.fig_height_per * n),
        sharex=True,
        sharey=True,
        constrained_layout=True,  # Better layout management
    )
    fig.subplots_adjust(hspace=0.15)

    if n == 1:
        axs = [axs]

    for idx, (ax, item, display_data, extent) in enumerate(
        zip(axs, items, display_data_list, extent_list, strict=False)
    ):
        if item["plot_type"] == "ratio":
            norm, vmin, vmax = _get_ratio_norm_and_limits(item["data"], cfg)
            print(f"  ratio display limits: vmin={vmin:.6f}, vmax={vmax:.6f}")
            im = ax.imshow(
                display_data,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=item["cmap"],
                norm=norm,
            )
        else:
            im = ax.imshow(
                display_data,
                extent=extent,
                origin="lower",
                aspect="auto",
                cmap=item["cmap"],
                vmin=item["vmin"],
                vmax=item["vmax"],
            )
        ax.set_title(item["title"], fontsize=12)

        ax.set_ylim(f_min, f_max)
        AxisConfigManager.configure_frequency_axis(ax, cfg, freq_out)

        if idx == len(items) - 1:
            _configure_datetime_axis(ax, cfg, x_start_num, x_end_num)
            if cfg.show_axis_labels:
                ax.set_xlabel("Time (UT)", fontsize=10)
        else:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter(cfg.xtick_format))
            ax.set_xlim(x_start_num, x_end_num)
            ax.tick_params(labelbottom=False)

        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(item["cbar_label"], fontsize=9)

        if cfg.highlight_freqs is not None:
            for freq_val in cfg.highlight_freqs:
                if f_min <= freq_val <= f_max:
                    ax.axhline(
                        y=freq_val,
                        color="red",
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.6,
                    )
                    # 添加文本标签（只在有标签显示时才添加）
                    if cfg.show_axis_labels:
                        x_min = mdates.date2num(dt_list[0])
                        x_max = mdates.date2num(dt_list[-1])
                        x_pos = x_min + 0.01 * (x_max - x_min)
                        ax.text(
                            x_pos,
                            freq_val + 0.01 * (cfg.f_end - cfg.f_start),
                            f"{freq_val} MHz",
                            color="red",
                            fontsize=4,
                            verticalalignment="bottom",
                            horizontalalignment="left",
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor="g", alpha=0.3
                            ),
                        )

        if drift_results:
            overlay_drift_rate_results(ax, drift_results, cfg)

    save_path = _resolve_output_path(cfg, t_start_dt, t_end_dt, items)
    if save_path:
        fig.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight")
        if not os.path.exists(save_path):
            raise RuntimeError(f"Save failed: file not found at {save_path}")
        if os.path.getsize(save_path) == 0:
            raise RuntimeError(f"Save failed: file is empty at {save_path}")
        print(f"Final save path: {os.path.abspath(save_path)}")

    if cfg.show_plot:
        plt.show()
    if cfg.close_after_save:
        plt.close(fig)


class ConfigManager:
    """配置管理器，集中处理所有配置逻辑"""

    @staticmethod
    def create_default_config() -> PlotConfig:
        """创建默认配置"""
        return PlotConfig()

    @staticmethod
    def update_config_from_dict(cfg: PlotConfig, config_dict: dict) -> PlotConfig:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @staticmethod
    def validate_all(cfg: PlotConfig) -> tuple[bool, list[str]]:
        """验证所有配置，返回(是否有效, 错误信息列表)"""
        errors = []

        try:
            validate_config(cfg)
        except Exception as e:
            errors.append(f"基础配置错误: {e}")

        try:
            validate_axis_config(cfg)
        except Exception as e:
            errors.append(f"坐标轴配置错误: {e}")

        return len(errors) == 0, errors


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible CSO workflow parser."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--file-path", help="CSO FITS file to process.")
    parser.add_argument("--time-start", help="UTC spectrum start time.")
    parser.add_argument("--time-end", help="UTC spectrum end time.")
    parser.add_argument("--frequency-start", type=float, help="Lower frequency in MHz.")
    parser.add_argument("--frequency-end", type=float, help="Upper frequency in MHz.")
    parser.add_argument(
        "--output-dir",
        help="Folder for the spectrum, selection, diagnostics, and metadata outputs.",
    )
    parser.add_argument("--rebin-time", type=int, help="Target time-axis samples.")
    parser.add_argument(
        "--rebin-frequency", type=int, help="Target frequency-axis samples."
    )
    parser.add_argument("--max-workers", type=int, help="Maximum worker count.")
    parser.add_argument("--plot-ll", action="store_true", help="Include LL intensity.")
    parser.add_argument("--plot-rr", action="store_true", help="Include RR intensity.")
    parser.add_argument(
        "--no-plot-sum", action="store_true", help="Exclude total intensity."
    )
    parser.add_argument(
        "--no-plot-ratio", action="store_true", help="Exclude polarization ratio."
    )
    parser.add_argument(
        "--select-drift",
        action="store_true",
        help="Open browser endpoint selector for manual drift-rate measurement.",
    )
    parser.add_argument(
        "--use-drift-selection", help="Use an existing drift-rate selection JSON file."
    )
    parser.add_argument(
        "--export-drift-preview",
        action="store_true",
        help="Export the drift-selection preview PNG and metadata without opening a server.",
    )
    parser.add_argument(
        "--drift-port", type=int, help="Port for the drift-rate selection web UI."
    )
    parser.add_argument(
        "--no-drift-browser",
        action="store_true",
        help="Do not automatically open browser for drift selection.",
    )
    parser.add_argument(
        "--drift-launch-policy",
        choices=("cli_only", "auto_if_missing", "always"),
        help="When to launch the interactive drift selector.",
    )
    parser.add_argument(
        "--disable-drift",
        action="store_true",
        help="Disable drift-rate overlay and diagnostics.",
    )
    parser.add_argument(
        "--enable-drift",
        action="store_true",
        help="Enable drift-rate overlay and diagnostics.",
    )
    return parser


def _apply_cli_overrides(cfg: PlotConfig, args: argparse.Namespace) -> None:
    """Apply only explicit CLI overrides to a CSO plot configuration."""
    if args.file_path:
        cfg.file_path = str(args.file_path)
    if args.time_start:
        parsed = _parse_datetime_value(args.time_start)
        if parsed is None:
            raise ValueError(f"Invalid --time-start: {args.time_start!r}")
        cfg.t_start = parsed
    if args.time_end:
        parsed = _parse_datetime_value(args.time_end)
        if parsed is None:
            raise ValueError(f"Invalid --time-end: {args.time_end!r}")
        cfg.t_end = parsed
    if args.frequency_start is not None:
        cfg.f_start = float(args.frequency_start)
    if args.frequency_end is not None:
        cfg.f_end = float(args.frequency_end)
    if args.output_dir:
        cfg.save_path = str(args.output_dir)
    if args.rebin_time is not None:
        cfg.rebin_t_target = int(args.rebin_time)
    if args.rebin_frequency is not None:
        cfg.rebin_f_target = int(args.rebin_frequency)
    if args.max_workers is not None:
        cfg.max_workers = int(args.max_workers)
    if args.plot_ll:
        cfg.plot_ll = True
    if args.plot_rr:
        cfg.plot_rr = True
    if args.no_plot_sum:
        cfg.plot_sum = False
    if args.no_plot_ratio:
        cfg.plot_ratio = False
    if args.export_drift_preview:
        cfg.export_drift_selection_preview = True
    if args.disable_drift:
        cfg.enable_drift_rate_overlay = False
        cfg.drift_rate_mode = "off"
    if args.enable_drift:
        cfg.enable_drift_rate_overlay = True
        if str(cfg.drift_rate_mode).lower() == "off":
            cfg.drift_rate_mode = "interactive_manual"
    if args.use_drift_selection:
        cfg._drift_selection_cli_path = args.use_drift_selection
        cfg.enable_drift_rate_overlay = True
        cfg.drift_rate_mode = "manual_json"
    if args.drift_port is not None:
        cfg.drift_rate_interactive["port"] = int(args.drift_port)
    if args.no_drift_browser:
        cfg.drift_rate_interactive["auto_open_browser"] = False
    if args.drift_launch_policy:
        cfg.drift_rate_interactive["launch_policy"] = args.drift_launch_policy
    if args.select_drift:
        cfg.enable_drift_rate_overlay = True
        cfg.drift_rate_mode = "interactive_manual"
        cfg._select_drift_now = True


def _print_config_summary(cfg: PlotConfig, file_paths: list[str]) -> None:
    """Print the historical workflow configuration summary."""
    print("=" * 60)
    print("CSO Spectrogram Plotting Tool")
    print("=" * 60)
    print("Files:")
    for file_path in file_paths:
        print(f"  - {os.path.basename(file_path)}")
    print(f"Time range: {cfg.t_start} to {cfg.t_end}")
    print(f"Frequency range: {cfg.f_start} to {cfg.f_end} MHz")
    print(
        "Color scale method: "
        f"{'Manual limits' if not cfg.use_percentile_clipping else 'Percentile clipping'}"
    )
    if not cfg.use_percentile_clipping:
        ll_vmin = (
            cfg.manual_ll_vmin
            if hasattr(cfg, "manual_ll_vmin") and cfg.manual_ll_vmin is not None
            else cfg.manual_vmin
        )
        ll_vmax = (
            cfg.manual_ll_vmax
            if hasattr(cfg, "manual_ll_vmax") and cfg.manual_ll_vmax is not None
            else cfg.manual_vmax
        )
        rr_vmin = (
            cfg.manual_rr_vmin
            if hasattr(cfg, "manual_rr_vmin") and cfg.manual_rr_vmin is not None
            else cfg.manual_vmin
        )
        rr_vmax = (
            cfg.manual_rr_vmax
            if hasattr(cfg, "manual_rr_vmax") and cfg.manual_rr_vmax is not None
            else cfg.manual_vmax
        )
        print(f"  Manual limits - LL: [{ll_vmin}, {ll_vmax}]")
        print(f"  Manual limits - RR: [{rr_vmin}, {rr_vmax}]")
        print(f"  Manual limits - Sum: [{cfg.manual_sum_vmin}, {cfg.manual_sum_vmax}]")
        print(
            "  Manual limits - Ratio: "
            f"[{cfg.manual_ratio_vmin}, {cfg.manual_ratio_vmax}]"
        )
    else:
        print(f"  Percentile clipping - LL/RR: [{cfg.vmin_pct}%, {cfg.vmax_pct}%]")
        print(
            "  Percentile clipping - Sum: "
            f"[{cfg.sum_vmin_pct}%, {cfg.sum_vmax_pct}%]"
        )
        print(
            "  Percentile clipping - Ratio: "
            f"[{cfg.ratio_vmin_pct}%, {cfg.ratio_vmax_pct}%]"
        )

    print("Axis configuration:")
    print(f"  - Show labels: {cfg.show_axis_labels}")
    print(f"  - Label rotation: {cfg.axis_label_rotation} degrees")
    print(f"  - X tick interval: {cfg.xtick_interval or 'auto'}")
    print(f"  - Y tick interval: {cfg.ytick_interval or 'auto'}")
    print(f"  - X tick format: {cfg.xtick_format}")
    print(f"  - Show minor ticks: {cfg.show_minor_ticks}")

    total_gb, available_gb, usage_percent = get_system_memory_info()
    if total_gb > 0:
        print(
            f"System memory: {total_gb:.1f} GB total, "
            f"{available_gb:.1f} GB available ({usage_percent:.1f}% used)"
        )
    print("Memory configuration:")
    print(f"  - Chunk memory: {cfg.chunk_mem_mb} MB")
    print("Drift-rate configuration:")
    print(f"  - Enabled: {cfg.enable_drift_rate_overlay}")
    print(f"  - Mode: {cfg.drift_rate_mode}")
    print(f"  - Selection JSON: {_drift_output_path(cfg, 'drift_rate_selection_json')}")
    print(f"  - Launch policy: {cfg.drift_rate_interactive.get('launch_policy')}")
    max_workers = cfg.max_workers if hasattr(cfg, "max_workers") else None
    print(
        f"  - Max workers: {max_workers if max_workers is not None else 'Auto-detect'}"
    )
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Run the CSO dynamic-spectrogram workflow."""
    cfg = PlotConfig()
    args, _unknown = build_parser().parse_known_args(argv)
    _apply_cli_overrides(cfg, args)

    is_valid, errors = ConfigManager.validate_all(cfg)
    if not is_valid:
        print("Configuration errors:", "\n".join(errors))
        return 1

    file_paths = get_config_file_paths(cfg)
    _print_config_summary(cfg, file_paths)
    t0 = time.perf_counter()
    hdus = []
    data_list = []
    try:
        for file_path in file_paths:
            data_part, hdu = read_cso_fits(file_path)
            data_list.extend(data_part)
            hdus.append(hdu)
        process_and_plot(cfg, data_list)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        for hdu in hdus:
            hdu.close()

    print(f"\nTotal execution time: {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
