"""Overlay orchestration and historical compatibility namespace.

Configuration lives in context; input pairing, science adapters, and rendering
have separate modules. This namespace keeps existing imports and replacement
hooks available to the public facade. Run caches remain owned here.
"""

# ruff: noqa: F401, I001

from ._overlay_context import (  # noqa: F401
    AIA_CONFIG,
    CanvasStyle,
    Config,
    GAUSSIAN_DIAGNOSTIC_FIELDS,
    GaussianFitResult,
    GaussianReprojectResult,
    IntArray,
    Line2D,
    NDArray,
    RawRadioReprojectResult,
    RegularGridInterpolator,
    SkyCoord,
    _ANIMATION_CONFIG_MAP,
    _CanonicalGaussianFitResult,
    _DATETIME_FMTS,
    _DRIFT_RATE_CONFIG_MAP,
    _RE_AIA_NEW_PAT,
    _RE_AIA_PATS,
    _RE_HMI_NEW_PAT,
    _RE_HMI_PAT,
    _RE_RADIO_PAT_YYYYJJJ,
    _RE_RADIO_PAT_YYYYMMDD,
    _SPECTROGRAM_CONFIG_MAP,
    _aia_cutout_extent_arcsec,
    _apply_mapped_values_to_config,
    _apply_values_to_config,
    _canonical_data_coord_to_pixel,
    _canonical_elliptical_gaussian_2d,
    _canonical_estimate_background_noise,
    _canonical_gaussian_fit_diag_defaults,
    _canonical_gaussian_quality_config,
    _canonical_limit_fit_pixels,
    _canonical_pixel_to_data_coord,
    _canonical_roi_slices_from_mask,
    _canonical_safe_rms_map,
    _canonical_select_peak_connected_mask,
    _canonical_true_indices,
    _canonical_unravel_2d_index,
    _extent_panel_aspect,
    _gaussian_diagnostics_row,
    _header_unit_to_arcsec_scale,
    _nearest_radio_entry_index,
    _normalise_band_key,
    _parse_millisecond_suffix,
    _radec_file_cache,
    _radio_file_item_label,
    _slice_file_list,
    apply_aia_radio_hmi_user_config,
    apply_config_to_object,
    binary_dilation,
    build_scientific_image_filename,
    build_spectrogram_cache,
    csv,
    curve_fit,
    dataclass,
    datetime,
    field,
    fits,
    gaussian_filter,
    glob,
    griddata,
    math,
    matplotlib,
    mcolors,
    mpath_effects,
    normalize_roi_bounds_arcsec,
    np,
    os,
    overlay_spectrogram_panel,
    plt,
    re,
    sunpy,
    timedelta,
    timezone,
    u,
    warnings,
    write_video_from_paths,
)

from ._overlay_selection import (  # noqa: F401
    _aia_panel_dir,
    _aia_panel_wavelengths,
    _build_common_radio_slots_from_entries,
    _fits_files_in_dir,
    _match_rr_ll_by_time,
    _multi_wave_overlay_enabled,
    _nearest_file_by_time,
    _parse_flexible_datetime,
    _parse_time_from_filename,
    _radio_time_key_from_item,
    _slot_reference_time,
    build_matched_pairs,
    build_multi_wave_matched_pairs,
    build_radio_time_slots_for_overlay,
    parse_aia_time_from_filename,
    parse_hmi_time_from_filename,
    parse_radio_time_from_filename,
)

from ._overlay_science import (  # noqa: F401
    _aia_pixel_to_arcsec,
    _attach_gaussian_fit_metadata,
    _build_radio_pixel_to_aia_pixel_mapper,
    _combine_polarization_data,
    _find_radec_file,
    _fit_failure_warning,
    _gaussian_fit_diag_defaults,
    _gaussian_quality_config,
    _get_padded_aia_map,
    _legacy_check_gaussian_fit_synthetic_source,
    _limit_fit_pixels,
    _load_fits_2d,
    _mesh_values_to_map,
    _radio_overlay_mode,
    _robust_median_mad,
    _roi_slices_from_mask,
    _safe_rms_map,
    _select_peak_connected_mask,
    _set_gaussian_failure_diag,
    _sigma_clip_values,
    _true_indices,
    _unravel_2d_index,
    _update_gaussian_quality,
    _weighted_moment_initial_guess,
    compute_contour_levels,
    config_for_gaussian_band,
    coordinate_roundtrip_error_pixel,
    create_source_mask,
    data_coord_to_pixel,
    elliptical_gaussian_2d,
    elliptical_gaussian_2d_with_constant_bg,
    elliptical_gaussian_2d_with_plane_bg,
    estimate_background_noise,
    estimate_background_rms_mesh,
    extract_radio_2d_data,
    fit_elliptical_gaussian,
    fit_elliptical_gaussian_on_radio_image,
    gaussian_only_from_popt,
    get_solar_position,
    pixel_to_data_coord,
    reproject_radio_for_overlay,
    reproject_radio_via_gaussian_fit,
    reproject_raw_radio_to_aia,
    save_gaussian_diagnostics_row,
    smooth_for_contour,
)

from ._overlay_render import (  # noqa: F401
    _add_multi_wave_panel_label,
    _aia_display_config_for_wave,
    _apply_multi_wave_row_axis_labels,
    _band_color_lookup_label,
    _build_selected_band_legend_elements,
    _configure_plotting,
    _create_mosaic_multi_wave_figure,
    _draw_aia_base_panel,
    _draw_direct_raw_header_contours,
    _draw_radio_contours_on_axis,
    _draw_raw_radio_contours_on_axis,
    _format_aia_panel_time,
    _load_overlay_radio_data,
    _multi_wave_figure_title,
    _multi_wave_mosaic_enabled,
    _multi_wave_panel_title,
    _raw_header_arcsec_axes,
    _roi_panel_aspect,
    _selected_band_legend_label,
    _valid_panel_aspect,
    create_multi_wave_figure,
    get_band_color,
    process_aia_group,
    process_hmi_for_overlay,
    process_multi_wave_aia_group,
)


__all__ = [
    "CanvasStyle",
    "Config",
    "GaussianFitResult",
    "GaussianReprojectResult",
    "RawRadioReprojectResult",
    "apply_aia_radio_hmi_user_config",
    "build_matched_pairs",
    "build_multi_wave_matched_pairs",
    "build_radio_time_slots_for_overlay",
    "compute_contour_levels",
    "create_multi_wave_figure",
    "extract_radio_2d_data",
    "fit_elliptical_gaussian_on_radio_image",
    "get_band_color",
    "main",
    "parse_aia_time_from_filename",
    "parse_hmi_time_from_filename",
    "parse_radio_time_from_filename",
    "process_aia_group",
    "process_hmi_for_overlay",
    "process_multi_wave_aia_group",
    "reproject_radio_for_overlay",
    "reproject_radio_via_gaussian_fit",
    "reproject_raw_radio_to_aia",
    "run_overlay_workflow",
]


def _spectrogram_config_dict(cfg: Config) -> dict:
    spectrogram_cfg = dict(vars(cfg))
    spectrogram_cfg["tick_color"] = cfg.style.tick_color
    spectrogram_cfg.setdefault("title_fontsize", 20)
    spectrogram_cfg.setdefault("label_fontsize", 18)
    spectrogram_cfg.setdefault("tick_fontsize", 14)
    return spectrogram_cfg


def _matched_radio_time_range(matched_pairs) -> tuple[datetime, datetime] | None:
    times = []
    for _aia_file, _hmi_file, sub_tasks in matched_pairs:
        for _sub_index, single_slice_bands in sub_tasks:
            for file_list in single_slice_bands.values():
                for _file_item, _polarization, radio_time in file_list:
                    if radio_time is not None:
                        times.append(radio_time)
    if not times:
        return None
    return min(times), max(times)


def _build_aia_spectrogram_cache(cfg: Config, matched_pairs):
    if not cfg.enable_spectrogram_panel:
        return None
    try:
        return build_spectrogram_cache(
            _spectrogram_config_dict(cfg),
            radio_time_range=_matched_radio_time_range(matched_pairs),
        )
    except Exception as exc:
        print(f"  [Spectrogram] failed to build cache: {exc}")
        return None


def _write_animation_from_frames(frame_paths: list[str], cfg: Config) -> str | None:
    if not cfg.make_animation or not frame_paths:
        return None
    output_dir = cfg.animation_output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.animation_name)
    quality = (
        cfg.animation_quality if cfg.animation_quality in {"high", "low"} else "high"
    )
    try:
        ok = write_video_from_paths(
            frame_paths,
            output_path,
            fps=int(cfg.animation_fps),
            quality=quality,
        )
    except Exception as exc:
        print(f"  [Animation] failed to write MP4: {exc}")
        return None
    if ok:
        print(f"  [Animation] saved MP4: {output_path}")
        return output_path
    print("  [Animation] MP4 generation failed")
    return None


def run_overlay_workflow(user_config=None, *, argv=None):
    """Generate overlay products and return the saved frame paths."""

    # Keep rendering style local to this run, including failed input selection.
    with matplotlib.rc_context():
        _configure_plotting()
        del argv  # Reserved for a stable CLI-compatible workflow signature.
        cfg = Config()
        cfg = apply_aia_radio_hmi_user_config(cfg, user_config)
        os.makedirs(cfg.output_dir, exist_ok=True)
        color_cache = []

        # 构建匹配对
        matched = (
            build_multi_wave_matched_pairs(cfg)
            if _multi_wave_overlay_enabled(cfg)
            else build_matched_pairs(cfg)
        )
        spectrogram_cache = _build_aia_spectrogram_cache(cfg, matched)
        frame_paths: list[str] = []
        batch_generated_at = datetime.now(timezone.utc)
        next_sequence = 1
        print(f"共构建 {len(matched)} 个 AIA 任务")

        # 串行处理（或可用线程池，但可能导致 FITS 读取冲突）
        if _multi_wave_overlay_enabled(cfg):
            for i, (aia_files_by_wave, hmi_file, sub_tasks) in enumerate(matched):
                frame_paths.extend(
                    process_multi_wave_aia_group(
                        aia_files_by_wave,
                        hmi_file,
                        sub_tasks,
                        i + 1,
                        len(matched),
                        cfg,
                        color_cache,
                        spectrogram_cache=spectrogram_cache,
                        sequence_start=next_sequence,
                        generated_at=batch_generated_at,
                    )
                )
                next_sequence += len(sub_tasks)
        else:
            for i, (aia_file, hmi_file, sub_tasks) in enumerate(matched):
                frame_paths.extend(
                    process_aia_group(
                        aia_file,
                        hmi_file,
                        sub_tasks,
                        i + 1,
                        len(matched),
                        cfg,
                        color_cache,
                        spectrogram_cache=spectrogram_cache,
                        sequence_start=next_sequence,
                        generated_at=batch_generated_at,
                    )
                )
                next_sequence += len(sub_tasks)

        _write_animation_from_frames(frame_paths, cfg)
        return frame_paths


_run_overlay_workflow = run_overlay_workflow


def main(user_config=None, *, argv=None) -> int:
    """Run the packaged overlay workflow and return a process status code."""

    _run_overlay_workflow(user_config=user_config, argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
