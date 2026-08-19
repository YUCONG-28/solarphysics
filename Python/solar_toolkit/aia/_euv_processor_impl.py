"""Heavy rendering and worker execution for the SDO/AIA EUV processor.

This private module owns the SunPy/Astropy map operations, plotting, and batch
workers.  Shared configuration, FITS selection, difference-limit, and mosaic
layout helpers live in their semantic AIA modules; runtime dispatch and the CLI
live in :mod:`solar_toolkit.aia.processor` and :mod:`solar_toolkit.aia.cli`.
"""

import datetime as dt
import gc
import multiprocessing
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpath_effects
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import propagate_with_solar_surface
from tqdm import tqdm

from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
    format_utc_filename_time,
)

from . import config as _config_helpers
from . import difference as _difference_helpers
from . import io as _io_helpers
from . import mosaic as _mosaic_helpers

AIA_CONFIG = _config_helpers.AIA_CONFIG
DIFF_CONFIG = _config_helpers.DIFF_CONFIG
AIAConfig = _config_helpers.AIAConfig

_diff_config_vlim = _difference_helpers.diff_config_vlim
_resolve_fixed_difference_limits_for_wave = (
    _difference_helpers.resolve_fixed_difference_limits_for_wave
)

_build_multi_band_slots = _io_helpers.build_multi_band_slots
_discover_wavelength_dirs = _io_helpers.discover_wavelength_dirs
_parse_timestr = _io_helpers.parse_timestr
_resolve_files = _io_helpers.resolve_files
_resolve_single_files = _io_helpers.resolve_single_files
_resolve_test_file = _io_helpers.resolve_test_file
_slice_band_files = _io_helpers.slice_band_files
_sorted_fits_for_band = _io_helpers.sorted_fits_for_band

_auto_mosaic_ncols = _mosaic_helpers.auto_mosaic_ncols
_layout_grid = _mosaic_helpers.layout_grid
_layout_mosaic_grid = _mosaic_helpers.layout_mosaic_grid
_mosaic_slot_wavelengths = _mosaic_helpers.mosaic_slot_wavelengths
_ordered_unique = _mosaic_helpers.ordered_unique

# Canonical configuration and validation live in solar_toolkit.aia.config.


@dataclass
class PanelData:
    cutout_map: sunpy.map.GenericMap
    wave_val: int
    iso_time: str
    date_ymd: str
    cmap: str
    norm: mcolors.Normalize
    observation_time: dt.datetime | None = None
    panel_kind: str = "original"
    panel_label: str | None = None
    is_difference: bool = False


# Path and FITS selection helpers are imported from solar_toolkit.aia.io.


# ==============================================================================
# Plotting Core
# ==============================================================================
def _resolve_display_params(
    current_map: sunpy.map.GenericMap,
    user_cmap: str | None,
    user_vmin: float | None,
    user_vmax: float | None,
) -> tuple[str, mcolors.Normalize]:
    wave_val = int(current_map.wavelength.value)
    config = AIA_CONFIG.get(wave_val)
    sunpy_norm = current_map.plot_settings["norm"]
    sunpy_cmap = current_map.plot_settings["cmap"]

    if config is None:
        warnings.warn(
            f"AIA wavelength {wave_val} Å is not in AIA_CONFIG; using SunPy "
            "default plot_settings unless CLI overrides are provided.",
            RuntimeWarning,
            stacklevel=2,
        )
        final_cmap = user_cmap or sunpy_cmap
        final_vmin = user_vmin if user_vmin is not None else sunpy_norm.vmin
        final_vmax = user_vmax if user_vmax is not None else sunpy_norm.vmax
    else:
        final_cmap = user_cmap or config["cmap"]
        final_vmin = user_vmin if user_vmin is not None else config["vmin"]
        final_vmax = user_vmax if user_vmax is not None else config["vmax"]

    if not (final_vmin and final_vmax and final_vmin > 0 and final_vmax > final_vmin):
        warnings.warn(
            f"Invalid LogNorm limits for AIA {wave_val} Å; falling back to "
            "vmin=1.0, vmax=10000.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        final_vmin, final_vmax = 1.0, 1e4

    return final_cmap, mcolors.LogNorm(vmin=final_vmin, vmax=final_vmax)


# Difference-limit helpers are imported from solar_toolkit.aia.difference.


def _resolve_difference_params(
    diff_data: np.ndarray,
    wave_val: int,
    cfg: AIAConfig,
) -> tuple[str, mcolors.Normalize]:
    diff_config = DIFF_CONFIG.get(wave_val, {})
    if cfg.difference_cmap_mode == "band":
        band_config = AIA_CONFIG.get(wave_val)
        if band_config is None:
            raise ValueError(
                f"AIA {wave_val}: missing AIA_CONFIG entry; cannot force "
                "band colormap."
            )
        if cfg.warn_band_difference_cmap:
            warnings.warn(
                "difference_cmap_mode='band' uses the AIA sequential band colormap. "
                "For signed difference maps, difference_cmap_mode='diverging' with "
                "RdBu_r is often clearer for positive/negative contrast.",
                RuntimeWarning,
                stacklevel=2,
            )
        cmap = band_config["cmap"]
    elif cfg.difference_cmap_mode == "diverging":
        cmap = diff_config.get("cmap") or "RdBu_r"
    elif cfg.difference_cmap_mode == "custom":
        if not cfg.difference_cmap:
            raise ValueError("difference_cmap_mode='custom' requires difference_cmap.")
        cmap = cfg.difference_cmap
    else:
        raise ValueError(f"Invalid difference_cmap_mode: {cfg.difference_cmap_mode}")

    def _difference_norm(vmin: float, vmax: float) -> mcolors.Normalize:
        if not vmin < vmax:
            raise ValueError("difference_vmin must be smaller than difference_vmax.")
        if vmin < 0 < vmax:
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        warnings.warn(
            "Difference limits do not span zero; using linear Normalize "
            "instead of TwoSlopeNorm.",
            RuntimeWarning,
            stacklevel=2,
        )
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    fixed_limits = _resolve_fixed_difference_limits_for_wave(wave_val, cfg)
    if fixed_limits is not None:
        vmin, vmax = fixed_limits
        return cmap, _difference_norm(vmin, vmax)

    if cfg.difference_norm_mode == "config":
        vlim = _diff_config_vlim(wave_val)
        return cmap, _difference_norm(-vlim, vlim)

    finite = np.asarray(diff_data[np.isfinite(diff_data)])
    if finite.size:
        abs_finite = np.abs(finite)
        vlim = float(np.nanpercentile(abs_finite, cfg.difference_percentile))
    else:
        vlim = np.nan
    if not np.isfinite(vlim) or vlim <= 0:
        vlim = _diff_config_vlim(wave_val)
    if cfg.mosaic_debug_layout:
        print(
            f"AIA {wave_val} auto difference limits: "
            f"percentile={cfg.difference_percentile}, "
            f"vmin={-vlim:.3g}, vmax={vlim:.3g}"
        )
    return cmap, _difference_norm(-vlim, vlim)


def _load_exposure_normalized_map(path: Path) -> sunpy.map.GenericMap:
    current_map = sunpy.map.Map(path)
    exp_time = current_map.exposure_time.to_value(u.s)
    if exp_time <= 0:
        raise ValueError(f"{path.name}: abnormal exposure time ({exp_time}s)")

    normalized_data = current_map.data / exp_time
    meta = current_map.meta.copy()
    meta["bunit"] = "DN / s"
    return sunpy.map.Map(normalized_data, meta)


def _cutout_roi(
    aia_map: sunpy.map.GenericMap,
    cfg: AIAConfig,
) -> sunpy.map.GenericMap:
    tx1, tx2, ty1, ty2 = cfg.roi_bounds
    frame = aia_map.coordinate_frame
    bl = SkyCoord(Tx=tx1 * u.arcsec, Ty=ty1 * u.arcsec, frame=frame)
    tr = SkyCoord(Tx=tx2 * u.arcsec, Ty=ty2 * u.arcsec, frame=frame)
    return aia_map.submap(bl, top_right=tr)


def _load_normalized_cutout(
    path: Path,
    cfg: AIAConfig,
) -> sunpy.map.GenericMap:
    full_map = _load_exposure_normalized_map(path)
    return _cutout_roi(full_map, cfg)


def _make_difference_map(
    current_map: sunpy.map.GenericMap,
    reference_map: sunpy.map.GenericMap | None,
    cfg: AIAConfig,
    wave: int | None = None,
) -> sunpy.map.GenericMap:
    meta = current_map.meta.copy()

    if reference_map is None:
        diff_quantity = current_map.quantity - current_map.quantity
    else:
        if reference_map.data.shape != current_map.data.shape:
            raise ValueError(
                f"shape mismatch current={current_map.data.shape}, "
                f"reference={reference_map.data.shape}"
            )
        diff_quantity = current_map.quantity - reference_map.quantity

    meta["bunit"] = diff_quantity.unit.to_string()
    diff_data = diff_quantity.value
    nan_fraction = np.count_nonzero(~np.isfinite(diff_data)) / diff_data.size
    if nan_fraction > 0.05:
        warnings.warn(
            f"Difference map contains {nan_fraction:.1%} NaN pixels. "
            "If this occurs in running difference, set difference_derotate=False; "
            "if using derotation, reproject full map before ROI cutout.",
            RuntimeWarning,
            stacklevel=2,
        )

    if cfg.mosaic_debug_layout:
        finite = diff_data[np.isfinite(diff_data)]
        if finite.size:
            wave_label = f"AIA {wave}" if wave is not None else "AIA"
            print(
                f"{wave_label} diff stats: "
                f"min={np.nanmin(diff_data):.3g}, "
                f"max={np.nanmax(diff_data):.3g}, "
                f"p1={np.nanpercentile(finite, 1):.3g}, "
                f"p99={np.nanpercentile(finite, 99):.3g}, "
                f"nan_fraction={nan_fraction:.3%}"
            )

    diff_data = np.nan_to_num(diff_data, nan=0.0, posinf=0.0, neginf=0.0)
    return sunpy.map.Map(diff_data, meta)


def _load_difference_map_from_paths(
    current_path: Path,
    reference_path: Path | None,
    wave: int,
    cfg: AIAConfig,
) -> sunpy.map.GenericMap:
    current_full = None
    reference_full = None
    reference_aligned_full = None

    try:
        current_full = _load_exposure_normalized_map(current_path)

        wave_val = int(current_full.wavelength.value)
        if wave_val != wave:
            raise ValueError(
                f"{current_path.name}: FITS wavelength {wave_val} does not "
                f"match expected band {wave}"
            )

        current_cutout = _cutout_roi(current_full, cfg)

        if reference_path is None:
            reference_cutout = None
        else:
            reference_full = _load_exposure_normalized_map(reference_path)
            ref_wave = int(reference_full.wavelength.value)
            if ref_wave != wave:
                raise ValueError(
                    f"{reference_path.name}: FITS wavelength {ref_wave} does "
                    f"not match expected band {wave}"
                )

            if cfg.difference_derotate:
                # Reproject the full map first. Reprojecting an already-cut ROI
                # can discard pixels that should rotate into the current ROI.
                with propagate_with_solar_surface():
                    reference_aligned_full = reference_full.reproject_to(
                        current_full.wcs
                    )
                reference_cutout = _cutout_roi(reference_aligned_full, cfg)
            else:
                reference_cutout = _cutout_roi(reference_full, cfg)

        return _make_difference_map(
            current_cutout,
            reference_cutout,
            cfg,
            wave=wave,
        )

    finally:
        del current_full, reference_full, reference_aligned_full
        gc.collect()


def _plot_difference_map(
    diff_map: sunpy.map.GenericMap,
    wave_val: int,
    title: str,
    save_path: Path,
    cfg: AIAConfig,
    prev_or_base_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    tx1, tx2, ty1, ty2 = cfg.roi_bounds
    dx = abs(tx2 - tx1)
    dy = abs(ty2 - ty1)
    aspect_ratio = dy / dx if dx != 0 else 1.0
    fig_width = cfg.base_fig_width
    fig_height = fig_width * aspect_ratio
    cmap, norm = _resolve_difference_params(diff_map.data, wave_val, cfg)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    try:
        ax = fig.add_subplot(projection=diff_map)
        ax.set_facecolor("white")
        im = diff_map.plot(axes=ax, cmap=cmap, norm=norm, annotate=False)

        if cfg.show_limb:
            diff_map.draw_limb(axes=ax, color="black", linewidth=0.8, alpha=0.6)

        if cfg.show_grid:
            overlay = diff_map.draw_grid(
                axes=ax,
                color="black",
                linewidth=0.3,
                alpha=0.3,
                linestyle="--",
                annotate=False,
            )
            _silence_heliographic_overlay(overlay)
            _purge_stonyhurst_text_artists(ax)

        if cfg.difference_show_colorbar or cfg.show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
            cbar.set_label("Difference intensity (DN/s)", fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        lon, lat = ax.coords
        lon.set_axislabel("Helioprojective Longitude (Solar-X)", fontsize=10)
        lat.set_axislabel("Helioprojective Latitude (Solar-Y)", fontsize=10)
        lon.set_ticks(direction="in")
        lat.set_ticks(direction="in")
        lon.set_ticks_position("tb")
        lat.set_ticks_position("lr")

        if prev_or_base_label:
            title = f"{title}\n{prev_or_base_label}"
        ax.set_title(title, fontsize=cfg.single_map_title_fontsize, pad=22)
        fig.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.11)

        if cfg.save_image:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path,
                dpi=cfg.dpi,
                bbox_inches="tight",
                facecolor="white",
                pad_inches=cfg.figure_pad_inches,
            )

        if cfg.show_image:
            plt.show()

    finally:
        plt.close(fig)
        gc.collect()


# Mosaic grid helpers are imported from solar_toolkit.aia.mosaic.


def _compute_mosaic_axes_rects(
    nrow: int,
    ncol: int,
    cfg: AIAConfig,
    has_title: bool,
) -> list[tuple[float, float, float, float]]:
    if cfg.mosaic_show_outer_axes:
        left = cfg.mosaic_left
        bottom = cfg.mosaic_bottom
    else:
        left = cfg.mosaic_left
        bottom = cfg.mosaic_bottom

    right = cfg.mosaic_right
    top = cfg.mosaic_top if has_title else cfg.mosaic_top_no_title

    usable_w = right - left
    usable_h = top - bottom
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError(
            "Invalid mosaic margins: usable width/height must be positive."
        )

    panel_w = usable_w / ncol
    panel_h = usable_h / nrow
    rects: list[tuple[float, float, float, float]] = []
    for idx in range(nrow * ncol):
        row, col = divmod(idx, ncol)
        x0 = left + col * panel_w
        y0 = top - (row + 1) * panel_h
        rects.append((x0, y0, panel_w, panel_h))
    return rects


def _compute_mosaic_figure_size(
    nrow: int,
    ncol: int,
    aspect_ratio: float,
    cfg: AIAConfig,
    has_title: bool,
) -> tuple[float, float]:
    fig_width = cfg.base_fig_width * ncol

    left = cfg.mosaic_left
    right = cfg.mosaic_right
    bottom = cfg.mosaic_bottom
    top = cfg.mosaic_top if has_title else cfg.mosaic_top_no_title

    usable_w = right - left
    usable_h = top - bottom
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError(
            "Invalid mosaic margins: usable width/height must be positive."
        )

    fig_height = fig_width * aspect_ratio * (nrow / ncol) * (usable_w / usable_h)
    if fig_height <= 0:
        raise ValueError("Computed mosaic figure height must be positive.")

    return fig_width, max(fig_height, 2.0)


def _debug_mosaic_layout(
    nrow: int,
    ncol: int,
    fig_width: float,
    fig_height: float,
    aspect_ratio: float,
    cfg: AIAConfig,
    has_title: bool,
) -> None:
    if not cfg.mosaic_debug_layout:
        return

    left = cfg.mosaic_left
    right = cfg.mosaic_right
    bottom = cfg.mosaic_bottom
    top = cfg.mosaic_top if has_title else cfg.mosaic_top_no_title
    usable_w = right - left
    usable_h = top - bottom
    physical_panel_w = fig_width * usable_w / ncol
    physical_panel_h = fig_height * usable_h / nrow
    physical_panel_aspect = physical_panel_h / physical_panel_w

    print("Mosaic layout debug:")
    print(f"nrow={nrow}, ncol={ncol}")
    print(f"fig_width={fig_width:.3f}, fig_height={fig_height:.3f}")
    print(f"roi_aspect={aspect_ratio:.6f}")
    print(f"usable_w={usable_w:.6f}, usable_h={usable_h:.6f}")
    print(f"physical_panel_w={physical_panel_w:.6f}")
    print(f"physical_panel_h={physical_panel_h:.6f}")
    print(f"physical_panel_aspect={physical_panel_aspect:.6f}")


def _finalize_panel_aspect(ax, aspect_ratio: float, cfg: AIAConfig) -> None:
    if cfg.mosaic_force_fill_axes:
        try:
            ax.set_aspect("auto")
        except Exception:
            pass
        return

    try:
        ax.set_box_aspect(aspect_ratio)
    except Exception:
        pass
    try:
        ax.set_aspect("equal", adjustable="box", anchor="C")
    except Exception:
        pass


def _obs_time_isot_label(aia_map: sunpy.map.GenericMap, fallback_path: Path) -> str:
    try:
        return str(aia_map.date.isot)
    except Exception:
        time_str = _parse_timestr(fallback_path).strip()
        return time_str[:-1] if time_str.endswith("Z") else time_str


def _obs_date_ymd(
    aia_map: sunpy.map.GenericMap, fallback_path: Path | None = None
) -> str:
    try:
        dt = aia_map.date.to_datetime()
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    except Exception:
        if fallback_path is not None:
            match = re.search(r"\d{4}-\d{2}-\d{2}", _parse_timestr(fallback_path))
            if match:
                return match.group(0)
        return ""


def _observation_datetime(aia_map) -> dt.datetime | None:
    try:
        value = aia_map.date.to_datetime(timezone=dt.UTC)
    except TypeError:
        try:
            value = aia_map.date.to_datetime()
        except Exception:
            return None
    except Exception:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _validated_filename_time(
    value,
    *,
    generated_at: dt.datetime,
) -> tuple[object, str]:
    try:
        format_utc_filename_time(value)
    except (TypeError, ValueError):
        return generated_at, "generated"
    return value, "observation"


def _validated_filename_interval(
    first,
    second,
    *,
    generated_at: dt.datetime,
) -> tuple[object, object | None, str]:
    """Return a chronological observation interval or one batch fallback."""

    try:
        first_text = format_utc_filename_time(first)
        second_text = format_utc_filename_time(second)
    except (TypeError, ValueError):
        return generated_at, None, "generated"
    if first_text <= second_text:
        return first, second, "observation"
    return second, first, "observation"


def _hide_wcs_frame_for_seamless(ax) -> None:
    lon, lat = ax.coords
    lon.set_ticks_visible(False)
    lat.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticklabel_visible(False)
    lon.set_axislabel("")
    lat.set_axislabel("")
    ax.set_frame_on(False)


def _configure_mosaic_axes(
    ax, row: int, col: int, nrow: int, ncol: int, cfg: AIAConfig
) -> None:
    if cfg.mosaic_global_outer_axes or not cfg.mosaic_show_outer_axes:
        _hide_wcs_frame_for_seamless(ax)
        return

    lon, lat = ax.coords
    is_last_row = row == nrow - 1
    is_first_col = col == 0

    try:
        lon.set_ticks(direction="in")
        lat.set_ticks(direction="in")
        lon.set_ticks_position("b")
        lon.set_ticklabel_position("b")
        lat.set_ticks_position("l")
        lat.set_ticklabel_position("l")
        lon.set_ticklabel(
            size=cfg.mosaic_ticklabel_fontsize,
            exclude_overlapping=True,
        )
        lat.set_ticklabel(
            size=cfg.mosaic_ticklabel_fontsize,
            exclude_overlapping=True,
        )
    except (TypeError, AttributeError):
        try:
            lon.set_ticklabel(size=cfg.mosaic_ticklabel_fontsize)
            lat.set_ticklabel(size=cfg.mosaic_ticklabel_fontsize)
        except (TypeError, AttributeError):
            pass

    if cfg.mosaic_x_tick_strategy == "all_bottom":
        show_lon = is_last_row
    elif cfg.mosaic_x_tick_strategy == "first_bottom_only":
        show_lon = is_last_row and col == 0
    elif cfg.mosaic_x_tick_strategy == "alternating_bottom":
        show_lon = is_last_row and (col % 2 == 0)
    else:
        raise ValueError("Invalid mosaic_x_tick_strategy")

    if cfg.mosaic_y_tick_strategy == "all_left":
        show_lat = is_first_col
    elif cfg.mosaic_y_tick_strategy == "first_left_only":
        show_lat = is_first_col and row == nrow - 1
    elif cfg.mosaic_y_tick_strategy == "alternating_left":
        show_lat = is_first_col and (row % 2 == 0)
    else:
        raise ValueError("Invalid mosaic_y_tick_strategy")

    if not cfg.mosaic_hide_inner_axes:
        show_lon = show_lon or is_last_row
        show_lat = show_lat or is_first_col

    lon.set_ticks_visible(show_lon)
    lon.set_ticklabel_visible(show_lon)
    lat.set_ticks_visible(show_lat)
    lat.set_ticklabel_visible(show_lat)

    if cfg.mosaic_reduce_tick_overlap:
        try:
            if show_lon:
                lon.set_ticks(number=cfg.mosaic_max_ticks_per_axis)
            if show_lat:
                lat.set_ticks(number=cfg.mosaic_max_ticks_per_axis)
        except Exception:
            pass

    lon.set_axislabel("")
    lat.set_axislabel("")

    ax.set_frame_on(True)


def _suppress_mosaic_boundary_ticklabels(
    ax, row: int, col: int, nrow: int, ncol: int, cfg: AIAConfig
) -> None:
    if not cfg.mosaic_hide_boundary_ticklabels:
        return

    # WCSAxes tick labels are version-dependent; rely on safer tick-count and
    # exclude-overlap controls, and only attempt best-effort private cleanup.
    try:
        lon, lat = ax.coords
        if row == nrow - 1 and col not in (0, ncol - 1):
            lon.set_ticklabel(exclude_overlapping=True)
        if col == 0 and row not in (0, nrow - 1):
            lat.set_ticklabel(exclude_overlapping=True)
    except Exception:
        pass


def _add_global_mosaic_axislabels(fig, cfg: AIAConfig) -> None:
    fig.text(
        0.5,
        0.015,
        "Helioprojective Longitude (Solar-X)",
        ha="center",
        va="center",
        fontsize=cfg.mosaic_axislabel_fontsize,
    )
    fig.text(
        0.015,
        0.5,
        "Helioprojective Latitude (Solar-Y)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=cfg.mosaic_axislabel_fontsize,
    )


def _silence_heliographic_overlay(overlay) -> None:
    if overlay is None:
        return
    try:
        lon, lat = overlay[0], overlay[1]
        lon.set_axislabel("")
        lat.set_axislabel("")
        lon.set_ticklabel_visible(False)
        lat.set_ticklabel_visible(False)
        lon.set_ticks_visible(False)
        lat.set_ticks_visible(False)
    except (TypeError, KeyError, IndexError, AttributeError):
        pass


def _purge_stonyhurst_text_artists(ax) -> None:
    for text in ax.texts:
        label = text.get_text().lower()
        if "stonyhurst" in label or "carrington" in label:
            text.set_visible(False)


def _process_single_worker(
    file_path: Path,
    cfg: AIAConfig,
    sequence: int = 1,
    generated_at: dt.datetime | None = None,
) -> tuple[bool, str]:
    current_map = None
    raw_cutout = None
    cutout_map = None
    fig = None
    plt = None

    try:
        if not cfg.show_image:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        current_map = sunpy.map.Map(file_path)
        _unused_wave_val = int(current_map.wavelength.value)
        exp_time = current_map.exposure_time.to_value(u.s)

        if exp_time <= 0:
            return False, f"{file_path.name}: abnormal exposure time ({exp_time}s)"

        tx1, tx2, ty1, ty2 = cfg.roi_bounds
        dx = abs(tx2 - tx1)
        dy = abs(ty2 - ty1)
        aspect_ratio = dy / dx if dx != 0 else 1.0
        fig_width = cfg.base_fig_width
        fig_height = fig_width * aspect_ratio

        with propagate_with_solar_surface():
            frame = current_map.coordinate_frame
            bl = SkyCoord(Tx=tx1 * u.arcsec, Ty=ty1 * u.arcsec, frame=frame)
            tr = SkyCoord(Tx=tx2 * u.arcsec, Ty=ty2 * u.arcsec, frame=frame)
            raw_cutout = current_map.submap(bl, top_right=tr)

        normalized_data = raw_cutout.data / exp_time
        cutout_map = sunpy.map.Map(normalized_data, raw_cutout.meta)

        final_cmap, final_norm = _resolve_display_params(
            current_map, cfg.user_cmap, cfg.user_vmin, cfg.user_vmax
        )
        time_str = _parse_timestr(file_path)

        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
        ax = fig.add_subplot(projection=cutout_map)
        ax.set_facecolor("white")

        im = cutout_map.plot(axes=ax, cmap=final_cmap, norm=final_norm, annotate=False)

        if cfg.show_limb:
            current_map.draw_limb(axes=ax, color="black", linewidth=0.8, alpha=0.6)

        if cfg.show_grid:
            overlay = cutout_map.draw_grid(
                axes=ax,
                color="black",
                linewidth=0.3,
                alpha=0.3,
                linestyle="--",
                annotate=False,
            )
            _silence_heliographic_overlay(overlay)
            _purge_stonyhurst_text_artists(ax)

        if cfg.show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity (DN/s)", fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        lon, lat = ax.coords
        lon.set_axislabel("Helioprojective Longitude (Solar-X)", fontsize=10)
        lat.set_axislabel("Helioprojective Latitude (Solar-Y)", fontsize=10)
        lon.set_ticks(direction="in")
        lat.set_ticks(direction="in")
        lon.set_ticks_position("tb")
        lat.set_ticks_position("lr")
        ax.set_title(f"{time_str}", fontsize=cfg.single_map_title_fontsize, pad=22)
        fig.subplots_adjust(left=0.13, right=0.95, top=0.93, bottom=0.11)

        if cfg.save_image:
            output_root = Path(cfg.output_dir or cfg.data_path)
            output_band = (
                output_root / str(_unused_wave_val)
                if cfg.use_band_subdirs
                else output_root
            )
            save_dir = output_band / cfg.single_band_output_subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            batch_time = generated_at or dt.datetime.now(dt.UTC)
            observation_time = _observation_datetime(current_map)
            filename_time, time_source = _validated_filename_time(
                observation_time,
                generated_at=batch_time,
            )
            save_path = save_dir / build_image_filename(
                ImageFilenameSpec(
                    sequence=sequence,
                    start_time=filename_time,
                    instrument="aia",
                    channel=f"{_unused_wave_val}a",
                    product="intensity",
                    time_source=time_source,
                )
            )
            fig.savefig(
                save_path,
                dpi=cfg.dpi,
                bbox_inches="tight",
                facecolor="white",
                pad_inches=cfg.figure_pad_inches,
            )

        if cfg.show_image:
            plt.show()

        return True, ""

    except Exception as exc:
        return False, f"{file_path.name} -> {exc}"

    finally:
        if fig is not None:
            plt.close(fig)
        del current_map, raw_cutout, cutout_map
        gc.collect()


def _load_aia_cutout_panel(path: Path, expected_wave: int, cfg: AIAConfig) -> PanelData:
    current_map = None
    raw_cutout = None
    normalized_data = None

    try:
        current_map = sunpy.map.Map(path)
        wave_val = int(current_map.wavelength.value)
        if wave_val != expected_wave:
            raise ValueError(
                f"{path.name}: FITS wavelength {wave_val} does not match "
                f"expected band {expected_wave}"
            )

        exp_time = current_map.exposure_time.to_value(u.s)
        if exp_time <= 0:
            raise ValueError(f"{path.name}: abnormal exposure time ({exp_time}s)")

        tx1, tx2, ty1, ty2 = cfg.roi_bounds
        with propagate_with_solar_surface():
            frame = current_map.coordinate_frame
            bl = SkyCoord(Tx=tx1 * u.arcsec, Ty=ty1 * u.arcsec, frame=frame)
            tr = SkyCoord(Tx=tx2 * u.arcsec, Ty=ty2 * u.arcsec, frame=frame)
            raw_cutout = current_map.submap(bl, top_right=tr)

        normalized_data = raw_cutout.data / exp_time
        cutout_map = sunpy.map.Map(normalized_data, raw_cutout.meta)
        final_cmap, final_norm = _resolve_display_params(
            current_map, cfg.user_cmap, cfg.user_vmin, cfg.user_vmax
        )

        return PanelData(
            cutout_map=cutout_map,
            wave_val=wave_val,
            iso_time=_obs_time_isot_label(current_map, path),
            date_ymd=_obs_date_ymd(current_map, path),
            cmap=final_cmap,
            norm=final_norm,
            observation_time=_observation_datetime(current_map),
            panel_kind="original",
            panel_label=f"{_obs_time_isot_label(current_map, path)} AIA {wave_val} original",
            is_difference=False,
        )

    finally:
        del current_map, raw_cutout, normalized_data
        gc.collect()


def _load_difference_cutout_panel(
    current_path: Path,
    reference_path: Path | None,
    wave: int,
    cfg: AIAConfig,
    method_label: str,
) -> PanelData:
    diff_map = None

    try:
        diff_map = _load_difference_map_from_paths(
            current_path,
            reference_path,
            wave,
            cfg,
        )
        cmap, norm = _resolve_difference_params(diff_map.data, wave, cfg)
        current_time = _obs_time_isot_label(diff_map, current_path)
        if reference_path is None:
            if method_label == "base":
                relation = "reference frame, zero difference"
            else:
                relation = "reference frame, no previous frame"
        elif method_label == "running":
            relation = "current - previous"
        else:
            relation = "current - base"
        panel_label = f"{current_time} AIA {wave} {method_label} diff\n{relation}"

        return PanelData(
            cutout_map=diff_map,
            wave_val=wave,
            iso_time=current_time,
            date_ymd=_obs_date_ymd(diff_map, current_path),
            cmap=cmap,
            norm=norm,
            observation_time=_observation_datetime(diff_map),
            panel_kind="difference",
            panel_label=panel_label,
            is_difference=True,
        )

    except Exception as exc:
        reference_msg = str(reference_path) if reference_path is not None else "None"
        raise RuntimeError(
            f"AIA {wave} {method_label} difference failed; "
            f"current file={current_path}; reference/base file={reference_msg}; "
            f"{exc}"
        ) from exc

    finally:
        del diff_map
        gc.collect()


def _draw_aia_panel(fig, ax, panel: PanelData, cfg: AIAConfig):
    im = panel.cutout_map.plot(
        axes=ax,
        cmap=panel.cmap,
        norm=panel.norm,
        annotate=False,
    )

    if cfg.show_limb:
        panel.cutout_map.draw_limb(
            axes=ax,
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

    if cfg.show_grid:
        overlay = panel.cutout_map.draw_grid(
            axes=ax,
            color="black",
            linewidth=0.3,
            alpha=0.3,
            linestyle="--",
            annotate=False,
        )
        _silence_heliographic_overlay(overlay)
        _purge_stonyhurst_text_artists(ax)

    if cfg.show_colorbar:
        cbar_label = "Difference intensity (DN/s)" if panel.is_difference else "DN/s"
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(
            cbar_label, fontsize=8
        )

    return im


def _add_panel_label(
    ax,
    iso_time: str,
    wave_val: int,
    row: int,
    nrow: int,
    cfg: AIAConfig,
    panel_label: str | None = None,
) -> None:
    label_y = (
        cfg.mosaic_panel_label_y_last_row
        if (
            (cfg.mosaic_show_outer_axes or cfg.mosaic_global_outer_axes)
            and row == nrow - 1
        )
        else cfg.mosaic_panel_label_y
    )
    ax.text(
        cfg.mosaic_panel_label_x,
        label_y,
        panel_label or f"{iso_time} AIA {wave_val}",
        transform=ax.transAxes,
        fontsize=11 if panel_label and "\n" in panel_label else 13,
        va="bottom",
        ha="left",
        color="white",
        path_effects=[
            mpath_effects.withStroke(
                linewidth=2.2,
                foreground="black",
                alpha=0.65,
            )
        ],
    )


def _save_mosaic_figure(fig, save_path: Path, cfg: AIAConfig) -> None:
    if cfg.mosaic_save_tight:
        bbox_inches = "tight"
        pad_inches = cfg.mosaic_pad_inches
    else:
        bbox_inches = None
        pad_inches = 0.0

    fig.savefig(
        save_path,
        dpi=cfg.dpi,
        bbox_inches=bbox_inches,
        facecolor="white",
        pad_inches=pad_inches,
    )


# Mosaic wavelength ordering is imported from solar_toolkit.aia.mosaic.


def _mosaic_save_prefix(cfg: AIAConfig) -> str:
    if cfg.mosaic_difference_inline and cfg.draw_difference:
        if cfg.draw_original:
            return "multi_original_plus_diff"
        return "multi_diff_only"
    return "multi"


def _mosaic_save_dir(cfg: AIAConfig) -> Path:
    output_root = Path(cfg.output_dir or cfg.data_path)

    if cfg.draw_difference and cfg.mosaic_difference_inline and not cfg.draw_original:
        return output_root / cfg.mosaic_difference_output_subdir

    if cfg.draw_difference and cfg.mosaic_difference_inline and cfg.draw_original:
        return output_root / cfg.mosaic_original_plus_difference_output_subdir

    return output_root / cfg.mosaic_output_subdir


def _base_difference_reference_path(wave: int, cfg: AIAConfig) -> Path:
    files = _sorted_fits_for_band(Path(cfg.data_path), wave, cfg.use_band_subdirs)
    if cfg.difference_base_index is None:
        sliced_files = _slice_band_files(files, cfg.start_idx, cfg.end_idx)
        if not sliced_files:
            raise ValueError(f"AIA {wave}: no selected files for base difference.")
        return sliced_files[0]
    if cfg.difference_base_index < 0 or cfg.difference_base_index >= len(files):
        raise ValueError(
            f"AIA {wave}: difference_base_index={cfg.difference_base_index} "
            f"is out of range for {len(files)} files."
        )
    return files[cfg.difference_base_index]


def _process_multi_band_worker(
    slot_idx: int,
    paths: tuple[Path, ...],
    wavelengths: tuple[int, ...],
    cfg: AIAConfig,
    previous_paths: tuple[Path, ...] | None = None,
    generated_at: dt.datetime | None = None,
) -> tuple[bool, str]:
    fig = None
    panels: list[PanelData] = []
    plt = None

    try:
        if not cfg.show_image:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        if len(paths) != len(wavelengths):
            return False, "Internal error: paths and wavelengths lengths differ."

        tx1, tx2, ty1, ty2 = cfg.roi_bounds
        dx = abs(tx2 - tx1)
        dy = abs(ty2 - ty1)
        aspect_ratio = dy / dx if dx != 0 else 1.0

        wave_to_current_path = dict(zip(wavelengths, paths, strict=False))
        wave_to_previous_path = (
            dict(zip(wavelengths, previous_paths, strict=False))
            if previous_paths is not None
            else {}
        )

        if cfg.draw_original:
            original_waves = cfg.multi_band_wavelengths or wavelengths
            for expected_wave in original_waves:
                path = wave_to_current_path.get(expected_wave)
                if path is None:
                    raise ValueError(
                        f"Missing current slot path for AIA {expected_wave}."
                    )
                try:
                    panels.append(_load_aia_cutout_panel(path, expected_wave, cfg))
                except Exception as exc:
                    raise RuntimeError(
                        f"wave={expected_wave}, slot_idx={slot_idx}, "
                        f"current file={path}, previous/base file=None: {exc}"
                    ) from exc

        if cfg.draw_difference and cfg.mosaic_difference_inline:
            diff_waves = (
                cfg.difference_wavelengths or cfg.multi_band_wavelengths or wavelengths
            )
            for wave in diff_waves:
                current_path = wave_to_current_path.get(wave)
                if current_path is None:
                    raise ValueError(f"Missing current slot path for AIA {wave}.")

                if cfg.difference_method == "base":
                    reference_path = _base_difference_reference_path(wave, cfg)
                    if (
                        current_path == reference_path
                        and not cfg.difference_save_reference
                    ):
                        continue
                    if current_path == reference_path:
                        reference_path = None
                else:
                    reference_path = wave_to_previous_path.get(wave)
                    if reference_path is None and not cfg.difference_save_reference:
                        if slot_idx == 0:
                            continue
                        raise ValueError(f"Missing previous slot path for AIA {wave}.")

                try:
                    panels.append(
                        _load_difference_cutout_panel(
                            current_path,
                            reference_path,
                            wave,
                            cfg,
                            cfg.difference_method,
                        )
                    )
                except Exception as exc:
                    reference_msg = (
                        str(reference_path) if reference_path is not None else "None"
                    )
                    raise RuntimeError(
                        f"wave={wave}, slot_idx={slot_idx}, "
                        f"current file={current_path}, "
                        f"previous/base file={reference_msg}: {exc}"
                    ) from exc

        if not panels:
            return (
                True,
                f"multi-band slot {slot_idx}: no panels selected; skipped.",
            )

        n_panels = len(panels)
        nrow, ncol = _layout_mosaic_grid(n_panels, cfg.mosaic_ncols)
        wspace = 0.0 if cfg.mosaic_seamless else cfg.multi_band_wspace
        hspace = 0.0 if cfg.mosaic_seamless else cfg.multi_band_hspace
        date_ymd = panels[0].date_ymd if panels else ""
        fig_width, fig_height = _compute_mosaic_figure_size(
            nrow=nrow,
            ncol=ncol,
            aspect_ratio=aspect_ratio,
            cfg=cfg,
            has_title=bool(date_ymd),
        )
        _debug_mosaic_layout(
            nrow=nrow,
            ncol=ncol,
            fig_width=fig_width,
            fig_height=fig_height,
            aspect_ratio=aspect_ratio,
            cfg=cfg,
            has_title=bool(date_ymd),
        )

        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

        if date_ymd:
            fig.suptitle(
                date_ymd,
                fontsize=cfg.figure_suptitle_fontsize,
                y=cfg.mosaic_title_y,
                fontweight="medium",
            )

        rects = None
        gs = None
        if cfg.mosaic_manual_layout:
            rects = _compute_mosaic_axes_rects(
                nrow,
                ncol,
                cfg,
                has_title=bool(date_ymd),
            )
        else:
            gs = fig.add_gridspec(
                nrow,
                ncol,
                figure=fig,
                wspace=wspace,
                hspace=hspace,
            )

        for idx in range(n_panels):
            row, col = divmod(idx, ncol)
            panel = panels[idx]
            if cfg.mosaic_manual_layout:
                ax = fig.add_axes(rects[idx], projection=panel.cutout_map)
            else:
                ax = fig.add_subplot(gs[row, col], projection=panel.cutout_map)
            ax.set_facecolor("white")
            _draw_aia_panel(fig, ax, panel, cfg)
            _finalize_panel_aspect(ax, aspect_ratio, cfg)

            if cfg.mosaic_global_outer_axes:
                _hide_wcs_frame_for_seamless(ax)
            elif cfg.mosaic_show_outer_axes:
                _configure_mosaic_axes(ax, row, col, nrow, ncol, cfg)
                _suppress_mosaic_boundary_ticklabels(ax, row, col, nrow, ncol, cfg)
            else:
                _hide_wcs_frame_for_seamless(ax)

            _add_panel_label(
                ax,
                panel.iso_time,
                panel.wave_val,
                row,
                nrow,
                cfg,
                panel.panel_label,
            )

        for idx in range(n_panels, nrow * ncol):
            row, col = divmod(idx, ncol)
            if not cfg.mosaic_hide_empty_panels:
                if cfg.mosaic_manual_layout:
                    ax_empty = fig.add_axes(rects[idx])
                else:
                    ax_empty = fig.add_subplot(gs[row, col])
                ax_empty.set_xticks([])
                ax_empty.set_yticks([])
                ax_empty.set_facecolor("white")

        if cfg.mosaic_global_outer_axes or (
            cfg.mosaic_show_outer_axes and cfg.mosaic_outer_axislabel_once
        ):
            _add_global_mosaic_axislabels(fig, cfg)

        if not cfg.mosaic_manual_layout:
            if cfg.mosaic_show_outer_axes:
                left, bottom = cfg.mosaic_left, cfg.mosaic_bottom
            else:
                left, bottom = cfg.mosaic_left, cfg.mosaic_bottom
            top = cfg.mosaic_top if date_ymd else cfg.mosaic_top_no_title
            fig.subplots_adjust(
                left=left,
                right=cfg.mosaic_right,
                bottom=bottom,
                top=top,
                wspace=wspace,
                hspace=hspace,
            )

        if cfg.save_image:
            save_dir = _mosaic_save_dir(cfg)
            save_dir.mkdir(parents=True, exist_ok=True)
            prefix = _mosaic_save_prefix(cfg)
            observation_times = sorted(
                panel.observation_time
                for panel in panels
                if panel.observation_time is not None
            )
            if observation_times:
                filename_start = observation_times[0]
                filename_end = observation_times[-1]
                time_source = "observation"
            else:
                filename_start = generated_at or dt.datetime.now(dt.UTC)
                filename_end = None
                time_source = "generated"
            save_path = save_dir / build_image_filename(
                ImageFilenameSpec(
                    sequence=slot_idx + 1,
                    start_time=filename_start,
                    end_time=filename_end,
                    instrument="aia",
                    product="mosaic",
                    qualifiers=prefix,
                    time_source=time_source,
                )
            )
            _save_mosaic_figure(fig, save_path, cfg)

        if cfg.show_image:
            plt.show()

        return True, ""

    except Exception as exc:
        return False, f"multi-band slot {slot_idx} -> {exc}"

    finally:
        if fig is not None:
            plt.close(fig)
        del panels
        gc.collect()


def _difference_save_dir(data_path: Path, wave: int, cfg: AIAConfig) -> Path:
    output_root = Path(cfg.output_dir) if cfg.output_dir else data_path
    method_dir = f"{cfg.difference_method}_difference"
    if cfg.use_band_subdirs:
        return output_root / str(wave) / cfg.difference_output_subdir / method_dir
    return output_root / cfg.difference_output_subdir / str(wave) / method_dir


# Difference image meaning:
# base difference: I_diff(t) = I(t) - I(t_base). It highlights enhancement,
# dimming, EUV waves, jets, and accumulated evolution relative to a reference
# frame, but is more sensitive to solar rotation and long-term background drift.
# running difference: I_diff(t) = I(t) - I(t - delta_t). It highlights
# short-timescale motion, fronts, and fast propagating structures, but adjacent
# positive/negative edges represent the rate of change rather than an absolute
# brightness enhancement.
def _process_difference_band_worker(
    wave: int,
    cfg: AIAConfig,
    generated_at: dt.datetime | None = None,
) -> tuple[bool, str]:
    if not cfg.show_image:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data_path = Path(cfg.data_path)
    generated_at = generated_at or dt.datetime.now(dt.UTC)
    success_count = 0
    error_messages: list[str] = []

    try:
        files = _sorted_fits_for_band(data_path, wave, cfg.use_band_subdirs)
        sliced_files = _slice_band_files(files, cfg.start_idx, cfg.end_idx)
        if len(sliced_files) < 2:
            return (
                False,
                f"AIA {wave}: Difference processing requires at least 2 FITS "
                f"files; found {len(sliced_files)} in selected range.",
            )

        save_dir = _difference_save_dir(data_path, wave, cfg)

        if cfg.difference_method == "base":
            if cfg.difference_base_index is None:
                base_path = sliced_files[0]
            else:
                if cfg.difference_base_index < 0 or cfg.difference_base_index >= len(
                    files
                ):
                    return (
                        False,
                        f"AIA {wave}: difference_base_index={cfg.difference_base_index} "
                        f"is out of range for {len(files)} files.",
                    )
                base_path = files[cfg.difference_base_index]

            base_time = _parse_timestr(base_path)

            export_plan = [
                current_file
                for current_file in sliced_files
                if current_file != base_path or cfg.difference_save_reference
            ]
            for sequence, current_file in enumerate(export_plan, start=1):
                diff_map = None
                try:
                    reference_path = None if current_file == base_path else base_path
                    diff_map = _load_difference_map_from_paths(
                        current_file,
                        reference_path,
                        wave,
                        cfg,
                    )
                    current_time = _parse_timestr(current_file)
                    if reference_path is None:
                        filename_start, time_source = _validated_filename_time(
                            _observation_datetime(diff_map) or current_time,
                            generated_at=generated_at,
                        )
                        filename_end = None
                    else:
                        filename_start, filename_end, time_source = (
                            _validated_filename_interval(
                                base_time,
                                _observation_datetime(diff_map) or current_time,
                                generated_at=generated_at,
                            )
                        )
                    save_path = save_dir / build_image_filename(
                        ImageFilenameSpec(
                            sequence=sequence,
                            start_time=filename_start,
                            end_time=filename_end,
                            instrument="aia",
                            channel=f"{wave}a",
                            product="difference",
                            qualifiers="base",
                            time_source=time_source,
                        )
                    )
                    label = (
                        "reference frame, zero difference"
                        if reference_path is None
                        else f"{current_time} - base {base_time}"
                    )
                    _plot_difference_map(
                        diff_map,
                        wave,
                        f"{current_time} AIA {wave} base difference",
                        save_path,
                        cfg,
                        prev_or_base_label=label,
                    )
                    success_count += 1
                except Exception as exc:
                    error_messages.append(
                        f"wave={wave}, current file={current_file}, "
                        f"previous/base file={base_path}: {exc}"
                    )
                    plt.close("all")
                finally:
                    del diff_map
                    gc.collect()

        else:
            if cfg.difference_save_reference:
                diff_map = None
                try:
                    diff_map = _load_difference_map_from_paths(
                        sliced_files[0],
                        None,
                        wave,
                        cfg,
                    )
                    first_time = _parse_timestr(sliced_files[0])
                    filename_time, time_source = _validated_filename_time(
                        _observation_datetime(diff_map) or first_time,
                        generated_at=generated_at,
                    )
                    save_path = save_dir / build_image_filename(
                        ImageFilenameSpec(
                            sequence=1,
                            start_time=filename_time,
                            instrument="aia",
                            channel=f"{wave}a",
                            product="difference",
                            qualifiers=("running", "reference"),
                            time_source=time_source,
                        )
                    )
                    _plot_difference_map(
                        diff_map,
                        wave,
                        f"{first_time} AIA {wave} running difference",
                        save_path,
                        cfg,
                        prev_or_base_label="reference frame, no previous frame",
                    )
                    success_count += 1
                except Exception as exc:
                    error_messages.append(
                        f"wave={wave}, current file={sliced_files[0]}, "
                        f"previous/base file=None: {exc}"
                    )
                    plt.close("all")
                finally:
                    del diff_map
                    gc.collect()

            running_start = 2 if cfg.difference_save_reference else 1
            for sequence, i in enumerate(
                range(1, len(sliced_files)),
                start=running_start,
            ):
                prev_file = sliced_files[i - 1]
                current_file = sliced_files[i]
                diff_map = None
                try:
                    diff_map = _load_difference_map_from_paths(
                        current_file,
                        prev_file,
                        wave,
                        cfg,
                    )
                    current_time = _parse_timestr(current_file)
                    prev_time = _parse_timestr(prev_file)
                    filename_start, filename_end, time_source = (
                        _validated_filename_interval(
                            prev_time,
                            _observation_datetime(diff_map) or current_time,
                            generated_at=generated_at,
                        )
                    )
                    save_path = save_dir / build_image_filename(
                        ImageFilenameSpec(
                            sequence=sequence,
                            start_time=filename_start,
                            end_time=filename_end,
                            instrument="aia",
                            channel=f"{wave}a",
                            product="difference",
                            qualifiers="running",
                            time_source=time_source,
                        )
                    )
                    _plot_difference_map(
                        diff_map,
                        wave,
                        f"{current_time} AIA {wave} running difference",
                        save_path,
                        cfg,
                        prev_or_base_label=f"{current_time} - {prev_time}",
                    )
                    success_count += 1
                except Exception as exc:
                    error_messages.append(
                        f"wave={wave}, current file={current_file}, "
                        f"previous/base file={prev_file}: {exc}"
                    )
                    plt.close("all")
                finally:
                    del diff_map
                    gc.collect()

    except Exception as exc:
        plt.close("all")
        gc.collect()
        return False, f"AIA {wave}: {exc}"

    if success_count == 0:
        detail = "; ".join(error_messages[:3])
        return False, f"AIA {wave}: no difference frames saved. {detail}"

    if error_messages:
        return (
            True,
            f"AIA {wave}: saved {success_count} difference frames; "
            f"skipped {len(error_messages)} frames. First error: {error_messages[0]}",
        )
    return True, f"AIA {wave}: saved {success_count} difference frames."


# ==============================================================================
# Batch Processing
# ==============================================================================
def _worker_count(cfg: AIAConfig) -> int:
    # Each worker holds full-resolution AIA images; cap the implicit default so
    # memory usage stays predictable on high-core machines.
    if cfg.max_workers is not None:
        return cfg.max_workers
    return max(1, min(multiprocessing.cpu_count() - 1, 8))


def _mosaic_worker_count(cfg: AIAConfig) -> int:
    if cfg.max_workers is not None:
        return cfg.max_workers
    if cfg.mosaic_max_workers is not None:
        return max(1, cfg.mosaic_max_workers)
    return max(1, min(2, multiprocessing.cpu_count() - 1))


def _run_single_batch(cfg: AIAConfig) -> None:
    selected_files = _resolve_single_files(cfg)
    if not selected_files:
        raise ValueError("No FITS files selected for single-band processing.")

    start_time = time.time()
    success_count = 0
    error_count = 0
    workers = _worker_count(cfg)
    batch_generated_at = dt.datetime.now(dt.UTC)

    print(f"Single-band mode: {len(selected_files)} files")
    print(
        "Starting serial processing ..."
        if workers == 1
        else f"Starting multiprocessing, allocated cores: {workers} ..."
    )

    if workers == 1:
        results = (
            _process_single_worker(file_path, cfg, sequence, batch_generated_at)
            for sequence, file_path in enumerate(selected_files, start=1)
        )
        iterator = tqdm(
            results,
            total=len(selected_files),
            desc="Processing",
            unit="file",
        )
        for success, msg in iterator:
            if success:
                success_count += 1
            else:
                error_count += 1
                tqdm.write(f"\n  [Failed] {msg}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_single_worker,
                    file_path,
                    cfg,
                    sequence,
                    batch_generated_at,
                ): file_path
                for sequence, file_path in enumerate(selected_files, start=1)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(selected_files),
                desc="Processing",
                unit="file",
            ):
                success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"\n  [Failed] {msg}")

    elapsed = time.time() - start_time
    print(
        f"\nSingle-band processing completed! Success: {success_count}, "
        f"Failed: {error_count}, Total time: {elapsed:.2f} seconds"
    )


def _run_mosaic_batch(cfg: AIAConfig) -> None:
    if not cfg.use_band_subdirs:
        raise ValueError(
            "Multi-band mosaic requires wavelength subdirectories "
            "(use_band_subdirs=True)."
        )

    waves = _mosaic_slot_wavelengths(cfg)

    slots = _build_multi_band_slots(cfg, waves)
    if not slots:
        raise ValueError("No time slots selected for multi-band mosaic processing.")
    if cfg.mosaic_max_slots is not None:
        slots = slots[: cfg.mosaic_max_slots]
        print(f"Limiting mosaic slots to first {len(slots)} for memory-safe preview.")

    start_time = time.time()
    success_count = 0
    error_count = 0
    workers = _mosaic_worker_count(cfg)
    batch_generated_at = dt.datetime.now(dt.UTC)

    print(f"Multi-band mosaic mode: slot wavelengths {waves}")
    if cfg.mosaic_difference_inline:
        print("Mosaic inline difference panels: enabled")
    print(
        f"Total {len(slots)} time slots; each slot contains {len(waves)} "
        "time-sorted band files."
    )
    print(f"Mosaic memory-safe workers: {workers}")

    if workers == 1:
        results = (
            _process_multi_band_worker(
                idx,
                slots[idx],
                waves,
                cfg,
                slots[idx - 1] if idx > 0 else None,
                batch_generated_at,
            )
            for idx in range(len(slots))
        )
        iterator = tqdm(
            results,
            total=len(slots),
            desc="Multi-band",
            unit="slot",
        )
        for success, msg in iterator:
            if success:
                success_count += 1
            else:
                error_count += 1
                tqdm.write(f"\n  [Failed] {msg}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_multi_band_worker,
                    idx,
                    slots[idx],
                    waves,
                    cfg,
                    slots[idx - 1] if idx > 0 else None,
                    batch_generated_at,
                ): idx
                for idx in range(len(slots))
            }
            for future in tqdm(
                as_completed(futures),
                total=len(slots),
                desc="Multi-band",
                unit="slot",
            ):
                success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"\n  [Failed] {msg}")

    elapsed = time.time() - start_time
    print(
        f"\nMulti-band mosaic completed! Success: {success_count}, "
        f"Failed: {error_count}, Total time: {elapsed:.2f} seconds"
    )

    if cfg.multi_band_also_save_single and cfg.draw_original:
        print("\n--- Exporting single-band images as requested ---")
        _run_single_batch(cfg)


def _run_difference_batch(cfg: AIAConfig) -> None:
    if not cfg.draw_difference:
        return

    data_path = Path(cfg.data_path)
    if cfg.difference_wavelengths is not None:
        waves = cfg.difference_wavelengths
    elif cfg.multi_band_wavelengths is not None:
        waves = cfg.multi_band_wavelengths
    else:
        waves = _discover_wavelength_dirs(data_path)

    if not waves:
        raise ValueError("No wavelengths selected for difference processing.")

    workers = cfg.max_workers or min(
        len(waves), max(1, multiprocessing.cpu_count() - 1)
    )
    print("\n--- Difference mode enabled ---")
    print(f"Difference method: {cfg.difference_method}")
    print(f"Difference wavelengths: {waves}")
    print(f"Difference norm mode: {cfg.difference_norm_mode}")
    if cfg.difference_norm_mode == "fixed":
        print(
            "Difference fixed limits: "
            f"vmin={cfg.difference_vmin}, vmax={cfg.difference_vmax}"
        )
    else:
        print(f"Difference percentile: {cfg.difference_percentile}")
    print(f"Difference workers: {workers}")

    start_time = time.time()
    success_count = 0
    error_count = 0
    batch_generated_at = dt.datetime.now(dt.UTC)

    if workers == 1:
        results = (
            _process_difference_band_worker(wave, cfg, batch_generated_at)
            for wave in waves
        )
        iterator = tqdm(
            results,
            total=len(waves),
            desc="Difference",
            unit="band",
        )
        for success, msg in iterator:
            if success:
                success_count += 1
                if msg:
                    tqdm.write(f"  [OK] {msg}")
            else:
                error_count += 1
                tqdm.write(f"\n  [Failed] {msg}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_difference_band_worker,
                    wave,
                    cfg,
                    batch_generated_at,
                ): wave
                for wave in waves
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Difference",
                unit="band",
            ):
                success, msg = future.result()
                if success:
                    success_count += 1
                    if msg:
                        tqdm.write(f"  [OK] {msg}")
                else:
                    error_count += 1
                    tqdm.write(f"\n  [Failed] {msg}")

    elapsed = time.time() - start_time
    print(
        f"\nDifference processing completed! Success bands: {success_count}, "
        f"Failed bands: {error_count}, Total time: {elapsed:.2f} seconds"
    )


def _run_test_mode(cfg: AIAConfig) -> None:
    test_path = _resolve_test_file(cfg)

    cfg.save_image = False
    cfg.show_image = True
    cfg.multi_band_composite = False
    cfg.max_workers = 1

    print("Test mode: previewing one AIA FITS file only.")
    print("No image will be saved.")
    print(f"Selected file: {test_path}")
    print(f"ROI: {cfg.roi_bounds}")
    print(
        "Display override: "
        f"cmap={cfg.user_cmap}, vmin={cfg.user_vmin}, vmax={cfg.user_vmax}"
    )
    print(
        f"Grid={cfg.show_grid}, Limb={cfg.show_limb}, " f"Colorbar={cfg.show_colorbar}"
    )

    success, msg = _process_single_worker(test_path, cfg)
    if not success:
        raise RuntimeError(msg)
