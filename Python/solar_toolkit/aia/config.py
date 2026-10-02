"""AIA configuration objects.

English: Lightweight configuration defaults for AIA EUV processing. Scientific
defaults stay behavior-compatible with the historical script unless a real-data
comparison approves a change.

中文：AIA EUV 处理的轻量配置默认值。除非经过真实观测数据对比确认，否则
科学默认参数保持与历史脚本兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AIA_CONFIG", "AIAConfig", "DIFF_CONFIG"]

AIA_CONFIG: dict = {
    94: {"cmap": "sdoaia94", "vmin": 0.6, "vmax": 420},
    131: {"cmap": "sdoaia131", "vmin": 0.9, "vmax": 810},
    171: {"cmap": "sdoaia171", "vmin": 16.0, "vmax": 2800},
    193: {"cmap": "sdoaia193", "vmin": 49.0, "vmax": 4200},
    211: {"cmap": "sdoaia211", "vmin": 22.0, "vmax": 2800},
    304: {"cmap": "sdoaia304", "vmin": 1.2, "vmax": 1400},
    335: {"cmap": "sdoaia335", "vmin": 0.5, "vmax": 700},
    1600: {"cmap": "sdoaia1600", "vmin": 5.0, "vmax": 3500},
}

# Original AIA EUV images use LogNorm because the intensity range is large.
# Difference images contain signed values, so they must use a linear Normalize
# with symmetric vmin/vmax around zero. SunPy examples often show AIA 193
# differences with about +/-200, but practical science plots should adjust the
# range by event, ROI, and wavelength. The default below is therefore an auto
# symmetric percentile of abs(diff_data), with DIFF_CONFIG as a fallback or
# explicit config mode. Historical fixed values such as +/-777 or +/-888 are
# intentionally not the default because they can hide weak structure or saturate
# strong events.
DIFF_CONFIG: dict = {
    94: {"cmap": "RdBu_r", "vlim": 120.0},
    131: {"cmap": "RdBu_r", "vlim": 200.0},
    171: {"cmap": "RdBu_r", "vlim": 250.0},
    193: {"cmap": "RdBu_r", "vlim": 250.0},
    211: {"cmap": "RdBu_r", "vlim": 220.0},
    304: {"cmap": "RdBu_r", "vlim": 300.0},
    335: {"cmap": "RdBu_r", "vlim": 120.0},
    1600: {"cmap": "RdBu_r", "vlim": 600.0},
}


def _normalize_wave_float_dict(value, name: str) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            wave = int(k)
            normalized[wave] = float(v)
        return normalized
    if isinstance(value, (list, tuple)):
        normalized = {}
        for item in value:
            if isinstance(item, str):
                if ":" not in item:
                    raise ValueError(
                        f"{name} item must use WAVE:VALUE format, got {item!r}"
                    )
                wave_str, val_str = item.split(":", 1)
                normalized[int(wave_str)] = float(val_str)
            else:
                raise ValueError(f"{name} must be dict or list of WAVE:VALUE strings.")
        return normalized
    raise ValueError(f"{name} must be dict, list, tuple, or None.")


@dataclass
class AIAConfig:
    # User-facing defaults. Keep these values behavior-compatible with the
    # historical processor unless a real-data comparison approves a change.
    root_dir: str = "data/aia"
    year: str | None = None
    date: str | None = None

    # Supported modes: single, mosaic, and test. use_test_mode previews one
    # selected FITS file even when another mode is configured.
    mode: str = "mosaic"
    use_test_mode: bool = False

    # test_file has highest priority. If it is empty, test mode sorts files
    # under data_path/test_wave by time and selects the zero-based test_index.
    test_file: str | None = None
    test_wave: int = 131
    test_index: int = 0

    data_path: str | None = None
    output_dir: str | None = None
    single_band_output_subdir: str = "plot"
    mosaic_output_subdir: str = "multi_band"
    mosaic_difference_output_subdir: str = "multi_band_difference"
    mosaic_original_plus_difference_output_subdir: str = (
        "multi_band_original_plus_difference"
    )

    start_idx: int = 0
    end_idx: int | None = None

    # ROI in arcsec helioprojective coordinates: (xmin, xmax, ymin, ymax).
    # Test mode is the safest way to tune it interactively.
    roi_bounds: tuple[float, float, float, float] | None = None

    # None means use AIA_CONFIG. Override only for temporary plot tuning.
    user_vmin: float | None = None
    user_vmax: float | None = None
    user_cmap: str | None = None

    base_fig_width: float = 8.0
    dpi: int = 300
    show_limb: bool = False
    show_grid: bool = True
    show_colorbar: bool = False

    # Test mode automatically disables saving and displays the preview.
    save_image: bool = True
    show_image: bool = False
    use_band_subdirs: bool = True
    max_workers: int | None = None

    draw_original: bool = False
    multi_band_composite: bool = False
    multi_band_wavelengths: tuple[int, ...] | None = (
        94,
        131,
        171,
        193,
        211,
        304,
    )
    multi_band_merge_axes: bool = True
    multi_band_also_save_single: bool = False

    # Mosaic columns per row. None uses an automatic near-square layout.
    mosaic_ncols: int | None = 3
    # True forces wspace/hspace=0 for seamless mosaic panels.
    mosaic_seamless: bool = True
    # Show coordinate ticks and labels only on the outside mosaic edges.
    mosaic_show_outer_axes: bool = True
    # Outside-edge coordinate label font sizes.
    mosaic_ticklabel_fontsize: int = 7
    mosaic_axislabel_fontsize: int = 9
    # Hide internal panel axes to avoid duplicate labels in seamless mosaics.
    mosaic_hide_inner_axes: bool = True
    # Hide trailing empty panels when the grid is not completely filled.
    mosaic_hide_empty_panels: bool = True
    # Manual axes layout prevents GridSpec/WCSAxes decorations from expanding
    # gaps between mosaic panels.
    mosaic_manual_layout: bool = True
    # Keep margin only around the whole mosaic, with no gaps between panels.
    mosaic_left: float = 0.055
    mosaic_right: float = 0.995
    mosaic_bottom: float = 0.055
    mosaic_top: float = 0.935
    mosaic_top_no_title: float = 0.995
    mosaic_title_y: float = 0.975
    # Default avoids tight bbox so saving does not reintroduce white borders.
    mosaic_save_tight: bool = False
    mosaic_pad_inches: float = 0.0
    # Do not stretch science images by default; fill axes only for display use.
    mosaic_force_fill_axes: bool = False
    # Print figure/panel ratios when debugging mosaic white-space issues.
    mosaic_debug_layout: bool = False
    # Reduce outside tick labels so neighboring panels do not duplicate labels.
    mosaic_reduce_tick_overlap: bool = True
    mosaic_max_ticks_per_axis: int = 3
    mosaic_hide_boundary_ticklabels: bool = True
    mosaic_x_tick_strategy: str = "all_bottom"
    mosaic_y_tick_strategy: str = "all_left"
    # Keep only one X/Y axis label for the whole figure.
    mosaic_outer_axislabel_once: bool = True
    # Clean mode: hide all panel tick numbers and add global outer axis labels.
    mosaic_global_outer_axes: bool = True
    # Lower-left wavelength/time label positions inside each panel.
    mosaic_panel_label_x: float = 0.02
    mosaic_panel_label_y: float = 0.035
    mosaic_panel_label_y_last_row: float = 0.08
    # Each mosaic job can open many FITS files; cap workers by default to
    # reduce memory pressure.
    mosaic_max_workers: int | None = 12
    mosaic_max_slots: int | None = None
    mosaic_difference_inline: bool = True

    # ===== Difference image options =====
    draw_difference: bool = True
    difference_method: str = "running"
    difference_output_mode: str = "auto"
    difference_wavelengths: tuple[int, ...] | None = (
        94,
        131,
        171,
        193,
        211,
        304,
    )
    difference_base_index: int | None = None
    difference_output_subdir: str = "difference"
    difference_norm_mode: str = "auto"
    difference_percentile: float = 99.5
    difference_vmin: float | None = None
    difference_vmax: float | None = None
    difference_cmap: str = "RdBu_r"
    difference_cmap_mode: str = "diverging"
    warn_band_difference_cmap: bool = False
    difference_save_reference: bool = False
    difference_show_colorbar: bool = False
    difference_derotate: bool = False
    difference_vmin_by_wave: dict = field(default_factory=dict)
    difference_vmax_by_wave: dict = field(default_factory=dict)
    difference_vlim_by_wave: dict = field(default_factory=dict)

    multi_band_wspace: float = 0.06
    multi_band_hspace: float = 0.06
    figure_pad_inches: float = 0.15
    figure_suptitle_fontsize: float = 34
    single_map_title_fontsize: float = 13

    def __post_init__(self):
        self.difference_vmin_by_wave = _normalize_wave_float_dict(
            self.difference_vmin_by_wave, "difference_vmin_by_wave"
        )
        self.difference_vmax_by_wave = _normalize_wave_float_dict(
            self.difference_vmax_by_wave, "difference_vmax_by_wave"
        )
        self.difference_vlim_by_wave = _normalize_wave_float_dict(
            self.difference_vlim_by_wave, "difference_vlim_by_wave"
        )
        if self.mode not in ("single", "mosaic", "test"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.data_path is None and self.year and self.date:
            self.data_path = str(
                Path(self.root_dir) / self.year / self.date / "SDO" / "AIA"
            )
        elif self.data_path is not None:
            self.data_path = str(Path(self.data_path))
        if self.test_file is not None:
            self.test_file = str(Path(self.test_file))
        if self.output_dir is None:
            self.output_dir = self.data_path
        if self.multi_band_wavelengths is not None:
            self.multi_band_wavelengths = tuple(
                int(w) for w in self.multi_band_wavelengths
            )
        if self.difference_method not in ("base", "running"):
            raise ValueError(f"Invalid difference_method: {self.difference_method}")
        if self.difference_output_mode not in ("auto", "mosaic", "single", "both"):
            raise ValueError(
                f"Invalid difference_output_mode: {self.difference_output_mode}"
            )
        if self.difference_norm_mode not in ("auto", "fixed", "config"):
            raise ValueError(
                f"Invalid difference_norm_mode: {self.difference_norm_mode}"
            )
        if self.difference_cmap_mode not in ("band", "diverging", "custom"):
            raise ValueError(
                f"Invalid difference_cmap_mode: {self.difference_cmap_mode}"
            )
        if self.difference_wavelengths is not None:
            self.difference_wavelengths = tuple(
                int(w) for w in self.difference_wavelengths
            )
        if not self.draw_original and not self.draw_difference:
            raise ValueError(
                "Nothing to draw: at least one of draw_original or "
                "draw_difference must be True."
            )
        if self.draw_difference and self.difference_percentile <= 0:
            raise ValueError("difference_percentile must be positive.")
