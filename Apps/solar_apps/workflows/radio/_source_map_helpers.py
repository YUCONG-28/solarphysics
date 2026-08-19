"""Private leaf helpers extracted from source_map_workflow.

Kept in a sibling module so source_map_workflow.py stays the only
public entry point while the monolith shrinks. No behavior change.
"""

from __future__ import annotations


def _deep_update_dict(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update_dict(result[key], value)
        else:
            result[key] = value
    return result


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


def get_radio_image_origin(header, cfg) -> str:
    mode = str(cfg.get("radio_image_origin_mode", "auto") or "auto").lower()
    if mode in {"upper", "lower"}:
        return mode
    if cfg.get("preserve_fits_wcs_orientation", True):
        return "lower"
    return "upper"


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


def _require_study_mode(value: object) -> str:
    mode = str(value or "").strip().casefold()
    if mode not in {"exploratory", "confirmatory"}:
        raise ValueError(
            "radio study_mode must be explicit: exploratory or confirmatory"
        )
    return mode


def _raw_quality_filter_enabled(cfg: dict) -> bool:
    return bool(cfg.get("enable_raw_quality_filter", False))


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


def _multi_band_output_subdir(cfg: dict) -> str:
    polarization = cfg.get("polarization", "RR")
    subdir_template = cfg.get("multi_band_output_subdir", "multi_band_{polar}")
    return str(subdir_template).format(polar=polarization)


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


__all__ = [
    "_deep_update_dict",
    "resolve_background_workflow",
    "_gaussian_multi_source_enabled",
    "_background_disabled_diag",
    "get_radio_image_origin",
    "_radio_background_mode",
    "_safe_background_median_size",
    "_gaussian_diagnostics_row",
    "get_time_from_header",
    "get_freq_from_header",
    "get_polar_from_header",
    "_require_study_mode",
    "_raw_quality_filter_enabled",
    "_nearest_time_entry_index",
    "_build_slots_by_position",
    "_candidate_slot_index",
    "_get_radio_display_range",
    "_apply_fixed_single_band_artifact_layout",
    "_apply_compact_radio_axis_style",
    "_prune_edge_ticklabels",
    "_add_global_radio_axis_labels",
    "_multi_band_output_subdir",
    "_should_precompute_fixed_band_ranges",
    "_combine_polarization_data",
    "_migrate_config",
]
