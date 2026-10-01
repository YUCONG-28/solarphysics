"""Overlay figure layout, drawing, legends, and image persistence.

Functions retain the historical core dependency hooks; scientific operations
and product ordering are unchanged.
"""

# ruff: noqa: F401, I001

from ._overlay_context import (
    AIA_CONFIG,
    Config,
    GaussianReprojectResult,
    Line2D,
    _aia_cutout_extent_arcsec,
    _extent_panel_aspect,
    _header_unit_to_arcsec_scale,
    _radio_file_item_label,
    build_scientific_image_filename,
    datetime,
    fits,
    gaussian_filter,
    math,
    matplotlib,
    mcolors,
    mpath_effects,
    np,
    os,
    overlay_spectrogram_panel,
    plt,
    re,
    sunpy,
    timezone,
    u,
)


def _workflow():
    """Resolve historical core replacement hooks only when a function runs."""

    from . import _overlay_workflow_core

    return _overlay_workflow_core


def _configure_plotting() -> None:
    """Apply the historical non-interactive plotting policy at run time."""

    matplotlib.use("Agg", force=True)
    plt.rcParams["font.family"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "SimHei",
        "Microsoft YaHei",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _multi_wave_mosaic_enabled(cfg: Config) -> bool:
    return str(getattr(cfg, "aia_panel_layout_style", "separate") or "").lower() in {
        "mosaic",
        "multi_band",
    }


def _valid_panel_aspect(panel_aspect_ratio: float | None) -> float:
    try:
        aspect = float(panel_aspect_ratio)
    except TypeError, ValueError:
        aspect = 1.0
    if not np.isfinite(aspect) or aspect <= 0:
        return 1.0
    return aspect


def _roi_panel_aspect(cfg: Config) -> float:
    try:
        left, bottom = [float(v) for v in cfg.roi_bottom_left]
        right, top = [float(v) for v in cfg.roi_top_right]
    except Exception:
        return 1.0
    width = abs(right - left)
    height = abs(top - bottom)
    return height / width if width > 0 else 1.0


def _create_mosaic_multi_wave_figure(
    cfg: Config,
    wavelengths: list[int],
    nrows: int,
    ncols: int,
    panel_aspect_ratio: float | None,
):
    aspect = _workflow()._valid_panel_aspect(
        panel_aspect_ratio or _workflow()._roi_panel_aspect(cfg)
    )
    panel_ratio = max(float(cfg.spectrogram_panel_height_ratio), 0.15)
    spectrogram_enabled = bool(cfg.enable_spectrogram_panel)

    panel_width_in = max(float(getattr(cfg, "aia_panel_base_width", 4.8)), 2.5)
    panel_height_in = panel_width_in * aspect
    left_margin_in = 0.85
    right_margin_in = 0.35
    top_margin_in = 0.70
    bottom_margin_in = 0.70
    spectrum_gap_in = max(float(getattr(cfg, "aia_panel_spectrogram_gap", 0.05)), 0.0)
    spectrum_height_in = panel_height_in * panel_ratio if spectrogram_enabled else 0.0
    if spectrogram_enabled:
        spectrum_gap_in = max(spectrum_gap_in, 0.45)

    mosaic_width_in = panel_width_in * ncols
    mosaic_height_in = panel_height_in * nrows
    fig_width = mosaic_width_in + left_margin_in + right_margin_in
    fig_height = (
        top_margin_in
        + mosaic_height_in
        + (spectrum_gap_in + spectrum_height_in if spectrogram_enabled else 0.0)
        + bottom_margin_in
    )
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=cfg.style.figure_bg)

    left = left_margin_in / fig_width
    right = 1.0 - right_margin_in / fig_width
    mosaic_top = 1.0 - top_margin_in / fig_height
    mosaic_height = mosaic_height_in / fig_height
    mosaic_bottom = mosaic_top - mosaic_height
    panel_w = (right - left) / ncols
    panel_h = mosaic_height / nrows

    axes_by_wave = {}
    for index, wavelength in enumerate(wavelengths):
        row, col = divmod(index, ncols)
        axes_by_wave[int(wavelength)] = fig.add_axes(
            [
                left + col * panel_w,
                mosaic_top - (row + 1) * panel_h,
                panel_w,
                panel_h,
            ]
        )

    spectrogram_ax = None
    if spectrogram_enabled:
        spectrum_bottom = bottom_margin_in / fig_height
        spectrogram_ax = fig.add_axes(
            [left, spectrum_bottom, right - left, spectrum_height_in / fig_height]
        )

    if getattr(cfg, "aia_panel_global_axis_labels", False):
        fig.text(
            (left + right) / 2.0,
            max(mosaic_bottom - 0.022, 0.02),
            "Helioprojective Longitude (Solar-X)",
            ha="center",
            va="center",
            fontsize=9,
        )
        fig.text(
            max(left - 0.035, 0.01),
            mosaic_bottom + mosaic_height / 2.0,
            "Helioprojective Latitude (Solar-Y)",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=9,
        )

    return fig, axes_by_wave, spectrogram_ax


def create_multi_wave_figure(
    cfg: Config,
    wavelengths: list[int],
    panel_aspect_ratio: float | None = None,
):
    """Create a 2-D AIA panel grid plus an optional shared spectrogram axis."""
    n_panels = max(len(wavelengths), 1)
    ncols = max(1, min(int(getattr(cfg, "aia_panel_ncols", 3) or 3), n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    if _workflow()._multi_wave_mosaic_enabled(cfg):
        return _workflow()._create_mosaic_multi_wave_figure(
            cfg, wavelengths, nrows, ncols, panel_aspect_ratio
        )

    spectrogram_enabled = bool(cfg.enable_spectrogram_panel)
    panel_ratio = max(float(cfg.spectrogram_panel_height_ratio), 0.15)
    height_ratios = [1.0] * nrows + ([panel_ratio] if spectrogram_enabled else [])
    fig_height = 3.6 * nrows + (2.2 if spectrogram_enabled else 0.2)
    fig = plt.figure(figsize=(4.4 * ncols, fig_height))
    gs = fig.add_gridspec(
        nrows + (1 if spectrogram_enabled else 0),
        ncols,
        height_ratios=height_ratios,
        hspace=float(getattr(cfg, "aia_panel_hspace", 0.18)),
        wspace=float(getattr(cfg, "aia_panel_wspace", 0.10)),
    )
    axes_by_wave = {}
    for index, wavelength in enumerate(wavelengths):
        row, col = divmod(index, ncols)
        axes_by_wave[int(wavelength)] = fig.add_subplot(gs[row, col])
    spectrogram_ax = fig.add_subplot(gs[nrows, :]) if spectrogram_enabled else None
    return fig, axes_by_wave, spectrogram_ax


def _aia_display_config_for_wave(wavelength: int) -> tuple[str, float, float]:
    wave_cfg = AIA_CONFIG.get(int(wavelength), {})
    return (
        str(wave_cfg.get("cmap", f"sdoaia{int(wavelength)}")),
        float(wave_cfg.get("vmin", 1.0)),
        float(wave_cfg.get("vmax", 1000.0)),
    )


def _draw_aia_base_panel(ax, aia_cutout, wavelength: int, cfg: Config) -> list[float]:
    extent_arcsec = _workflow()._aia_cutout_extent_arcsec(aia_cutout)
    cmap_name, vmin, vmax = _workflow()._aia_display_config_for_wave(wavelength)
    my_cmap = plt.get_cmap(cmap_name).copy()
    my_cmap.set_bad(color="black")
    ax.imshow(
        aia_cutout.data,
        cmap=my_cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        origin="lower",
        extent=extent_arcsec,
    )
    ax.set_facecolor("black")
    ax.set_xlim([extent_arcsec[0], extent_arcsec[1]])
    ax.set_ylim([extent_arcsec[2], extent_arcsec[3]])
    ax.set_aspect("equal", adjustable="box")
    if _workflow()._multi_wave_mosaic_enabled(cfg):
        show_ticks = bool(getattr(cfg, "aia_panel_show_tick_labels", True))
        ax.tick_params(
            colors=cfg.style.tick_color,
            labelsize=7,
            length=2 if show_ticks else 0,
            labelbottom=show_ticks,
            labelleft=show_ticks,
        )
        if not show_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_xlabel("" if not cfg.aia_panel_show_axis_labels else "Solar X (arcsec)")
        ax.set_ylabel("" if not cfg.aia_panel_show_axis_labels else "Solar Y (arcsec)")
        if cfg.aia_panel_show_grid:
            ax.grid(color="white", alpha=0.22, linewidth=0.35, linestyle="-")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.4)
    else:
        ax.tick_params(colors=cfg.style.tick_color, labelsize=7)
        ax.set_xlabel("Solar X (arcsec)", fontsize=8)
        ax.set_ylabel("Solar Y (arcsec)", fontsize=8)
    try:
        rsun_pix = aia_cutout.rsun_obs.to(u.arcsec).value
        circle = plt.Circle(
            (aia_cutout.center.Tx.value, aia_cutout.center.Ty.value),
            rsun_pix,
            fill=False,
            color=cfg.style.limb_color,
            lw=cfg.style.limb_lw,
            alpha=cfg.style.limb_alpha,
        )
        ax.add_patch(circle)
    except Exception:
        pass
    return extent_arcsec


def _apply_multi_wave_row_axis_labels(
    ax,
    row: int,
    nrows: int,
    cfg: Config,
    col: int | None = None,
) -> None:
    is_last_row = row == nrows - 1
    show_tick_labels = bool(getattr(cfg, "aia_panel_show_tick_labels", True))
    ax.tick_params(labelbottom=is_last_row and show_tick_labels)
    if _workflow()._multi_wave_mosaic_enabled(cfg) and col is not None:
        ax.tick_params(labelleft=col == 0 and show_tick_labels)
    if not bool(getattr(cfg, "aia_panel_show_axis_labels", True)):
        ax.set_xlabel("")
        ax.set_ylabel("")
        return
    ax.set_xlabel("Solar X (arcsec)" if is_last_row else "")


def _format_aia_panel_time(aia_cutout) -> str:
    obs_time = getattr(aia_cutout, "date", None)
    if obs_time is None:
        return ""
    for attr in ("isot", "iso"):
        value = getattr(obs_time, attr, None)
        if value:
            text = str(value)
            return text.replace(" ", "T")
    if isinstance(obs_time, datetime):
        return obs_time.isoformat(timespec="milliseconds")
    return str(obs_time).replace(" ", "T")


def _add_multi_wave_panel_label(
    ax,
    aia_cutout,
    wavelength: int,
    row: int,
    nrows: int,
    cfg: Config,
) -> None:
    if str(getattr(cfg, "aia_panel_label_mode", "none") or "").lower() != "inside":
        return
    label_y = (
        cfg.aia_panel_label_y_last_row if row == nrows - 1 else cfg.aia_panel_label_y
    )
    time_label = _workflow()._format_aia_panel_time(aia_cutout)
    label = f"{time_label} AIA {wavelength}" if time_label else f"AIA {wavelength}"
    ax.text(
        cfg.aia_panel_label_x,
        label_y,
        label,
        transform=ax.transAxes,
        fontsize=cfg.aia_panel_label_fontsize,
        va="bottom",
        ha="left",
        color="white",
        path_effects=[
            mpath_effects.withStroke(linewidth=1.8, foreground="black", alpha=0.7)
        ],
    )


def _multi_wave_figure_title(
    first_radio_time: datetime | None,
    aia_cutouts: dict[int, object],
    wavelengths: list[int],
) -> str:
    if first_radio_time is not None:
        return first_radio_time.strftime("%Y-%m-%d")
    for wavelength in wavelengths:
        cutout = aia_cutouts.get(wavelength)
        label = _workflow()._format_aia_panel_time(cutout) if cutout is not None else ""
        if len(label) >= 10:
            return label[:10]
    return ""


def _load_overlay_radio_data(file_item, polarization: str, cfg: Config):
    if (
        cfg.combine_polarizations
        and polarization == "RR+LL"
        and isinstance(file_item, tuple)
    ):
        rr_path, ll_path = file_item
        rr_data, ra_map, dec_map, rr_header, _ = _workflow().extract_radio_2d_data(
            rr_path, cfg.radio_use_float32, cfg
        )
        ll_data, _, _, _, _ = _workflow().extract_radio_2d_data(
            ll_path, cfg.radio_use_float32, cfg
        )
        if rr_data is None or ll_data is None:
            return None, None, None, None
        return (
            _workflow()._combine_polarization_data(rr_data, ll_data, cfg),
            ra_map,
            dec_map,
            rr_header,
        )
    radio_data, ra_map, dec_map, radio_header, _ = _workflow().extract_radio_2d_data(
        file_item, cfg.radio_use_float32, cfg
    )
    return radio_data, ra_map, dec_map, radio_header


def _raw_header_arcsec_axes(
    radio_shape: tuple[int, int], radio_header: fits.Header | None
) -> tuple[np.ndarray, np.ndarray] | None:
    if radio_header is None or len(radio_shape) != 2:
        return None
    required = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2")
    if any(key not in radio_header for key in required):
        return None
    try:
        crpix1 = float(radio_header["CRPIX1"])
        crpix2 = float(radio_header["CRPIX2"])
        crval1 = float(radio_header["CRVAL1"])
        crval2 = float(radio_header["CRVAL2"])
        cdelt1 = float(radio_header["CDELT1"])
        cdelt2 = float(radio_header["CDELT2"])
    except TypeError, ValueError:
        return None
    values = (crpix1, crpix2, crval1, crval2, cdelt1, cdelt2)
    if not all(np.isfinite(value) for value in values):
        return None

    ny, nx = radio_shape
    x_scale = _workflow()._header_unit_to_arcsec_scale(
        radio_header.get("CUNIT1", "arcsec")
    )
    y_scale = _workflow()._header_unit_to_arcsec_scale(
        radio_header.get("CUNIT2", "arcsec")
    )
    x_axis = (crval1 + (np.arange(nx, dtype=float) + 1.0 - crpix1) * cdelt1) * x_scale
    y_axis = (crval2 + (np.arange(ny, dtype=float) + 1.0 - crpix2) * cdelt2) * y_scale
    if not (np.all(np.isfinite(x_axis)) and np.all(np.isfinite(y_axis))):
        return None
    return x_axis, y_axis


def _draw_direct_raw_header_contours(
    ax,
    radio_data: np.ndarray,
    radio_header: fits.Header | None,
    cfg: Config,
    color_main: str,
) -> bool:
    if _workflow()._radio_overlay_mode(cfg) != "raw" or bool(
        getattr(cfg, "use_radec_maps", True)
    ):
        return False
    radio_values = np.asarray(radio_data, dtype=np.float64)
    if radio_values.ndim != 2 or not np.isfinite(radio_values).any():
        return False
    axes = _workflow()._raw_header_arcsec_axes(radio_values.shape, radio_header)
    if axes is None:
        return False
    if not cfg.show_radio_contours:
        return True

    x_axis, y_axis = axes
    contour_data = radio_values
    if cfg.contour_smooth_sigma > 0:
        contour_data = _workflow().smooth_for_contour(
            contour_data, cfg.contour_smooth_sigma
        )
    levels = _workflow().compute_contour_levels(contour_data, cfg)
    if levels:
        ax.contour(
            x_axis,
            y_axis,
            contour_data,
            levels=levels,
            colors=[color_main],
            linewidths=cfg.contour_linewidths,
            alpha=cfg.contour_alpha,
            origin="lower",
        )
    return True


def _draw_radio_contours_on_axis(
    ax,
    aia_cutout,
    extent_arcsec: list[float],
    single_slice_bands: dict,
    cfg: Config,
    color_cache: list,
) -> datetime | None:
    first_radio_time = None

    def _band_freq(item):
        m = re.search(r"(\d+\.?\d*)MHz", item[0])
        return float(m.group(1)) if m else 0.0

    for band_label, file_list in sorted(single_slice_bands.items(), key=_band_freq):
        band_idx = (
            cfg.selected_bands.index(band_label)
            if band_label in cfg.selected_bands
            else 0
        )
        search_bl = (
            band_label if "." in band_label else band_label.replace("MHz", ".0MHz")
        )
        color_main, _ = _workflow().get_band_color(
            search_bl, band_idx, cfg, color_cache
        )
        for file_item, polarization, radio_time in file_list:
            if first_radio_time is None and radio_time:
                first_radio_time = radio_time
            radio_data, ra_map, dec_map, radio_header = (
                _workflow()._load_overlay_radio_data(file_item, polarization, cfg)
            )
            if radio_data is None:
                continue
            if _workflow()._draw_direct_raw_header_contours(
                ax, radio_data, radio_header, cfg, color_main
            ):
                continue
            result = _workflow().reproject_radio_for_overlay(
                radio_data,
                ra_map,
                dec_map,
                aia_cutout,
                cfg,
                radio_header,
                source_file=_workflow()._radio_file_item_label(file_item),
                band_label=band_label,
                polarization=polarization,
                radio_time=radio_time,
            )
            if result is None:
                continue
            if (
                not getattr(result, "overlay_valid", True)
                and not cfg.draw_low_quality_gaussian_contours
            ):
                continue

            if cfg.show_radio_contours:
                contour_data = result.model
                if cfg.contour_smooth_sigma > 0:
                    contour_data = _workflow().smooth_for_contour(
                        contour_data, cfg.contour_smooth_sigma
                    )
                levels = _workflow().compute_contour_levels(contour_data, cfg)
                if levels:
                    ax.contour(
                        contour_data,
                        levels=levels,
                        extent=extent_arcsec,
                        colors=[color_main],
                        linewidths=cfg.contour_linewidths,
                        alpha=cfg.contour_alpha,
                        origin="lower",
                    )

            if cfg.mark_radio_center and isinstance(result, GaussianReprojectResult):
                center_x, center_y = result.center_arcsec
                ax.scatter(
                    [center_x],
                    [center_y],
                    marker=cfg.radio_center_marker,
                    s=cfg.radio_center_size,
                    color=color_main,
                    linewidths=cfg.radio_center_linewidth,
                )
                if cfg.label_radio_center:
                    ax.annotate(
                        "Gaussian-fit center",
                        xy=(center_x, center_y),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=color_main,
                        va="center",
                        ha="left",
                    )
    return first_radio_time


def _draw_raw_radio_contours_on_axis(
    ax,
    aia_cutout,
    extent_arcsec: list[float],
    single_slice_bands: dict,
    cfg: Config,
    color_cache: list,
) -> datetime | None:
    return _workflow()._draw_radio_contours_on_axis(
        ax, aia_cutout, extent_arcsec, single_slice_bands, cfg, color_cache
    )


def _multi_wave_panel_title(
    wavelength: int,
    cfg: Config,
    hmi_file: str | None,
    radio_time: datetime | None,
) -> str:
    radio_label = (
        "Gaussian Radio"
        if _workflow()._radio_overlay_mode(cfg) == "gaussian"
        else "Raw Radio"
    )
    layers = [f"AIA {wavelength}"]
    if hmi_file and cfg.overlay_hmi:
        layers.append("HMI")
    layers.append(radio_label)
    title_time = radio_time.strftime("%H:%M:%S UT") if radio_time else "Unknown time"
    return f"{' + '.join(layers)}\n{title_time}"


def process_multi_wave_aia_group(
    aia_files_by_wave: dict[int, str],
    hmi_file: str | None,
    sub_tasks: list[tuple[int, dict]],
    task_index: int,
    total_tasks: int,
    cfg: Config,
    color_cache: list,
    spectrogram_cache=None,
    *,
    sequence_start: int = 1,
    generated_at: datetime | None = None,
):
    """Draw one or more radio-first frames as six AIA wave panels plus spectrum."""
    wavelengths = _workflow()._aia_panel_wavelengths(cfg)
    saved_paths: list[str] = []
    aia_cutouts = {}
    for wavelength in wavelengths:
        aia_file = aia_files_by_wave.get(wavelength)
        if not aia_file:
            continue
        try:
            aia_cutouts[wavelength] = _workflow()._get_padded_aia_map(
                _workflow().sunpy.map.Map(aia_file), cfg
            )
        except Exception as exc:
            print(f"  [AIA {wavelength}] read failed: {exc}")
            return saved_paths
    if len(aia_cutouts) != len(wavelengths):
        return saved_paths

    first_extent = _workflow()._aia_cutout_extent_arcsec(aia_cutouts[wavelengths[0]])
    panel_aspect_ratio = _workflow()._extent_panel_aspect(first_extent)
    ncols = max(1, min(int(getattr(cfg, "aia_panel_ncols", 3) or 3), len(wavelengths)))
    nrows = int(math.ceil(len(wavelengths) / ncols))

    generated_at = generated_at or datetime.now(timezone.utc)
    for local_index, (sub_index, single_slice_bands) in enumerate(sub_tasks):
        print(
            f"multi-wave frame {task_index}/{total_tasks}, "
            f"sequence {sub_index + 1}/{len(sub_tasks)}",
            flush=True,
        )
        fig, axes_by_wave, spectrogram_ax = _workflow().create_multi_wave_figure(
            cfg, wavelengths, panel_aspect_ratio=panel_aspect_ratio
        )
        first_radio_time = None
        for panel_index, wavelength in enumerate(wavelengths):
            row, col = divmod(panel_index, ncols)
            ax = axes_by_wave[wavelength]
            aia_cutout = aia_cutouts[wavelength]
            extent_arcsec = _workflow()._draw_aia_base_panel(
                ax, aia_cutout, wavelength, cfg
            )
            _workflow()._apply_multi_wave_row_axis_labels(ax, row, nrows, cfg, col=col)
            if _workflow()._multi_wave_mosaic_enabled(cfg):
                _workflow()._add_multi_wave_panel_label(
                    ax, aia_cutout, wavelength, row, nrows, cfg
                )
            if hmi_file and cfg.overlay_hmi:
                try:
                    _workflow().process_hmi_for_overlay(
                        hmi_file, aia_cutout.wcs, cfg, ax
                    )
                except Exception as exc:
                    print(f"  [HMI] overlay failed on AIA {wavelength}: {exc}")
            panel_radio_time = _workflow()._draw_radio_contours_on_axis(
                ax, aia_cutout, extent_arcsec, single_slice_bands, cfg, color_cache
            )
            if first_radio_time is None and panel_radio_time is not None:
                first_radio_time = panel_radio_time
            if (
                not _workflow()._multi_wave_mosaic_enabled(cfg)
                or cfg.aia_panel_show_per_panel_titles
            ):
                ax.set_title(
                    _workflow()._multi_wave_panel_title(
                        wavelength,
                        cfg,
                        hmi_file=hmi_file,
                        radio_time=first_radio_time,
                    ),
                    fontsize=9,
                )

        if _workflow()._multi_wave_mosaic_enabled(cfg):
            figure_title = _workflow()._multi_wave_figure_title(
                first_radio_time, aia_cutouts, wavelengths
            )
            if figure_title:
                fig.suptitle(
                    figure_title,
                    fontsize=20,
                    y=float(getattr(cfg, "aia_panel_title_y", 0.975)),
                    fontweight="medium",
                )

        legend_elements = _workflow()._build_selected_band_legend_elements(
            cfg, color_cache
        )
        if legend_elements and wavelengths:
            axes_by_wave[wavelengths[0]].legend(
                handles=legend_elements,
                loc="upper right",
                facecolor=cfg.style.legend_face,
                edgecolor="none",
                labelcolor=cfg.style.legend_text,
                framealpha=cfg.style.legend_alpha,
                fontsize=7,
            )

        if spectrogram_ax is not None:
            spectrogram_cfg = _workflow()._spectrogram_config_dict(cfg)
            if spectrogram_cache is None:
                spectrogram_cfg["enable_spectrogram_panel"] = False
            _workflow().overlay_spectrogram_panel(
                spectrogram_ax,
                spectrogram_cfg,
                first_radio_time,
                cache=spectrogram_cache,
            )

        if cfg.save_figure:
            polarization = cfg.polarization_mode
            out_name = _workflow().build_scientific_image_filename(
                sequence=sequence_start + local_index,
                start_time=first_radio_time,
                instrument="aia_hmi_radio" if hmi_file else "aia_radio",
                polarization=polarization,
                product="multi_wavelength_source_overlay",
                qualifiers=("spectrogram",) if spectrogram_ax is not None else (),
                generated_at=generated_at,
            )
            saved_path = os.path.join(cfg.output_dir, out_name)
            save_tight = not _workflow()._multi_wave_mosaic_enabled(cfg) or bool(
                getattr(cfg, "aia_panel_save_tight", True)
            )
            plt.savefig(
                saved_path,
                dpi=cfg.dpi,
                bbox_inches="tight" if save_tight else None,
                pad_inches=0.0 if not save_tight else 0.1,
                facecolor=cfg.style.figure_bg,
            )
            print(f"  saved multi-wave image: {saved_path}")
            saved_paths.append(saved_path)
        plt.close(fig)
    return saved_paths


def process_aia_group(
    aia_file: str,
    hmi_file: str | None,
    sub_tasks: list[tuple[int, dict]],
    task_index: int,
    total_tasks: int,
    cfg: Config,
    color_cache: list,
    spectrogram_cache=None,
    *,
    sequence_start: int = 1,
    generated_at: datetime | None = None,
):
    """
    处理单个 AIA 文件及其对应的所有时间切片绘图任务。

    参数
    ----------
    aia_file   : str
        AIA FITS 文件完整路径。
    hmi_file   : Optional[str]
        HMI FITS 文件完整路径，若无则传入 None。
    sub_tasks  : List[Tuple[int, Dict]]
        时间切片列表，每个元素为 (索引, {波段: [(文件路径, 偏振, 时间), ...]}).
    task_index : int
        当前任务序号（从 1 开始，用于打印进度）。
    total_tasks: int
        任务总数。
    cfg        : Config
        全局配置对象。
    color_cache: List
        颜色缓存列表，用于图例颜色一致性。

    功能描述
    ----------
    - 加载 AIA 图像并裁剪/扩充到用户设定 ROI。
    - 遍历每个时间切片，对各个射电波段执行：数据提取、高斯拟合、坐标重投影、平滑及等值线绘制。
    - 若启用 HMI，将其重投影到 AIA 坐标系并绘制磁图等值线。
    - 添加图例、标题，保存图像到输出目录。
    """
    print(f"\n处理 AIA 文件 [{task_index}/{total_tasks}]: {os.path.basename(aia_file)}")

    saved_paths: list[str] = []

    try:
        aia_map = _workflow().sunpy.map.Map(aia_file)
    except Exception as e:
        print(f"  读取 AIA 失败: {e}")
        return saved_paths

    aia_cutout = _workflow()._get_padded_aia_map(aia_map, cfg)
    aia_data = aia_cutout.data
    extent_arcsec = [
        aia_cutout.bottom_left_coord.Tx.value,
        aia_cutout.top_right_coord.Tx.value,
        aia_cutout.bottom_left_coord.Ty.value,
        aia_cutout.top_right_coord.Ty.value,
    ]

    # 【核心：遍历时间切片，每一帧生成一张图】
    generated_at = generated_at or datetime.now(timezone.utc)
    for local_index, (sub_index, single_slice_bands) in enumerate(sub_tasks):
        print(f"  -> 绘制序列帧 {sub_index + 1}/{len(sub_tasks)}")

        spectrogram_ax = None
        if cfg.enable_spectrogram_panel:
            panel_ratio = max(float(cfg.spectrogram_panel_height_ratio), 0.15)
            fig = plt.figure(figsize=(10, 10 * (1.0 + panel_ratio)))
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[1.0, panel_ratio],
                hspace=float(cfg.spectrogram_hspace),
            )
            ax = fig.add_subplot(gs[0, 0])
            spectrogram_ax = fig.add_subplot(gs[1, 0])
        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        # --- 1. 绘制 AIA 底图 ---
        # 提取当前的 colormap，并强制将无数据的 NaN 区域（即扩充的深空画布）渲染为纯黑
        my_cmap = plt.get_cmap(cfg.aia_cmap).copy()
        my_cmap.set_bad(color="black")

        # 【核心修复】：加入 norm=mcolors.LogNorm(...)，使用对数缩放！
        ax.imshow(
            aia_data,
            cmap=my_cmap,
            norm=mcolors.LogNorm(vmin=cfg.aia_vmin, vmax=cfg.aia_vmax),
            origin="lower",
            extent=extent_arcsec,
        )

        # 同时也把坐标轴的背景底色设为黑，作为双重保险
        ax.set_facecolor("black")

        ax.set_xlabel("Solar X (arcsec)")
        ax.set_ylabel("Solar Y (arcsec)")
        ax.tick_params(colors=cfg.style.tick_color)

        ax.set_xlim([extent_arcsec[0], extent_arcsec[1]])
        ax.set_ylim([extent_arcsec[2], extent_arcsec[3]])

        rsun_pix = aia_cutout.rsun_obs.to(u.arcsec).value
        circle = plt.Circle(
            (aia_cutout.center.Tx.value, aia_cutout.center.Ty.value),
            rsun_pix,
            fill=False,
            color=cfg.style.limb_color,
            lw=cfg.style.limb_lw,
            alpha=cfg.style.limb_alpha,
        )
        ax.add_patch(circle)

        # --- 2. 处理并叠加 HMI ---
        if hmi_file and cfg.overlay_hmi:
            try:
                _workflow().process_hmi_for_overlay(hmi_file, aia_cutout.wcs, cfg, ax)
            except Exception as e:
                print(f"  处理 HMI 失败: {e}")

        legend_elements = _workflow()._build_selected_band_legend_elements(
            cfg, color_cache
        )
        first_radio_time = None

        # --- 3. 提取并遍历当前切片的波段数据 ---
        def _band_freq(item):
            m = re.search(r"(\d+\.?\d*)MHz", item[0])
            return float(m.group(1)) if m else 0.0

        sorted_bands = sorted(single_slice_bands.items(), key=_band_freq)

        for band_label, file_list in sorted_bands:
            band_idx = (
                cfg.selected_bands.index(band_label)
                if band_label in cfg.selected_bands
                else 0
            )
            search_bl = (
                band_label if "." in band_label else band_label.replace("MHz", ".0MHz")
            )
            color_main, _ = _workflow().get_band_color(
                search_bl, band_idx, cfg, color_cache
            )

            # 解析数据结构：(文件路径, 偏振模式, 射电时间)
            for file_item, polarization, radio_time in file_list:
                if first_radio_time is None and radio_time:
                    first_radio_time = radio_time

                # 对应匹配对中的元组和字符串解包
                if (
                    cfg.combine_polarizations
                    and polarization == "RR+LL"
                    and isinstance(file_item, tuple)
                ):
                    rr_path, ll_path = file_item
                    (
                        rr_data,
                        ra_map,
                        dec_map,
                        rr_header,
                        _,
                    ) = _workflow().extract_radio_2d_data(
                        rr_path, cfg.radio_use_float32, cfg
                    )
                    ll_data, _, _, _, _ = _workflow().extract_radio_2d_data(
                        ll_path, cfg.radio_use_float32, cfg
                    )
                    if rr_data is None or ll_data is None:
                        continue
                    radio_data = _workflow()._combine_polarization_data(
                        rr_data, ll_data, cfg
                    )
                    radio_header2 = rr_header
                else:
                    # 单偏振模式时，file_item 就是单文件的路径（字符串）
                    (
                        radio_data,
                        ra_map,
                        dec_map,
                        radio_header2,
                        _,
                    ) = _workflow().extract_radio_2d_data(
                        file_item, cfg.radio_use_float32, cfg
                    )
                    if radio_data is None:
                        continue

                # 重新投影
                if isinstance(file_item, tuple):
                    source_file_for_fit = f"{file_item[0]}|{file_item[1]}"
                else:
                    source_file_for_fit = str(file_item)

                fit_result = _workflow().reproject_radio_for_overlay(
                    radio_data,
                    ra_map,
                    dec_map,
                    aia_cutout,
                    cfg,
                    radio_header2,
                    source_file=source_file_for_fit,
                    band_label=band_label,
                    polarization=polarization,
                    radio_time=radio_time,
                )
                if fit_result is None:
                    continue
                if (
                    not fit_result.overlay_valid
                    and not cfg.draw_low_quality_gaussian_contours
                ):
                    continue
                model_data = fit_result.model

                if cfg.show_radio_contours:
                    contour_data = model_data
                    if cfg.contour_smooth_sigma > 0:
                        contour_data = _workflow().smooth_for_contour(
                            contour_data, cfg.contour_smooth_sigma
                        )

                    levels = _workflow().compute_contour_levels(contour_data, cfg)
                    if not levels:
                        continue

                    # 绘制射电等值线
                    ax.contour(
                        contour_data,
                        levels=levels,
                        extent=extent_arcsec,
                        colors=[color_main],
                        linewidths=cfg.contour_linewidths,
                        alpha=cfg.contour_alpha,
                        origin="lower",
                    )

                if cfg.mark_radio_center and isinstance(
                    fit_result, GaussianReprojectResult
                ):
                    center_x, center_y = fit_result.center_arcsec
                    ax.scatter(
                        [center_x],
                        [center_y],
                        marker=cfg.radio_center_marker,
                        s=cfg.radio_center_size,
                        color=color_main,
                        linewidths=cfg.radio_center_linewidth,
                    )
                    if cfg.label_radio_center:
                        ax.annotate(
                            "Gaussian-fit center",
                            xy=(center_x, center_y),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=color_main,
                            va="center",
                            ha="left",
                        )

        # --- 4. 标题、图例与保存 ---
        if cfg.combine_polarizations and cfg.polarization_mode == "RR+LL":
            polar_display = (
                "RR+LL (sum)"
                if not cfg.weighted_average
                else f"RR+LL (w={cfg.rr_weight}:{cfg.ll_weight})"
            )
        else:
            polar_display = cfg.polarization_mode

        title_time = (
            first_radio_time.strftime("%Y-%m-%d %H:%M:%S") + " UT"
            if first_radio_time
            else "Unknown Time"
        )
        ax.set_title(
            f"AIA 171 Å + Radio ({polar_display}) + HMI\n{title_time}",
            color=cfg.style.title_color,
        )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                facecolor=cfg.style.legend_face,
                edgecolor="none",
                labelcolor=cfg.style.legend_text,
                framealpha=cfg.style.legend_alpha,
            )

        if spectrogram_ax is not None:
            spectrogram_cfg = _workflow()._spectrogram_config_dict(cfg)
            if spectrogram_cache is None:
                spectrogram_cfg["enable_spectrogram_panel"] = False
            _workflow().overlay_spectrogram_panel(
                spectrogram_ax,
                spectrogram_cfg,
                first_radio_time,
                cache=spectrogram_cache,
            )

        if cfg.save_figure:
            polarization = cfg.polarization_mode
            qualifiers = []
            if cfg.combine_polarizations and cfg.weighted_average:
                qualifiers.append(f"weighted_{cfg.rr_weight:g}_{cfg.ll_weight:g}")
            if spectrogram_ax is not None:
                qualifiers.append("spectrogram")

            out_name = _workflow().build_scientific_image_filename(
                sequence=sequence_start + local_index,
                start_time=first_radio_time
                or _workflow().parse_aia_time_from_filename(aia_file),
                instrument="aia_hmi_radio" if hmi_file else "aia_radio",
                channel="171a",
                polarization=polarization,
                product="source_overlay",
                qualifiers=qualifiers,
                generated_at=generated_at,
            )
            saved_path = os.path.join(cfg.output_dir, out_name)

            plt.savefig(
                saved_path,
                dpi=cfg.dpi,
                bbox_inches="tight",
                facecolor=cfg.style.figure_bg,
            )
            print(f"  保存图像: {saved_path}")

        if cfg.save_figure:
            saved_paths.append(saved_path)

        plt.close(fig)

    return saved_paths


def process_hmi_for_overlay(hmi_file: str, target_wcs, cfg: Config, ax):
    """读取 HMI 数据，投影到 AIA 坐标系并绘制磁图等值线"""
    try:
        hmi_map = _workflow().sunpy.map.Map(hmi_file)
    except Exception as e:
        print(f"  读取 HMI 失败: {e}")
        return

    # 重投影到 AIA 网格
    hmi_reprojected = hmi_map.reproject_to(target_wcs)
    hmi_data = hmi_reprojected.data

    # 平滑
    if cfg.hmi_sigma > 0:
        hmi_data = _workflow().gaussian_filter(hmi_data, sigma=cfg.hmi_sigma)

    # 正负水平
    pos_data = np.where(hmi_data > cfg.hmi_threshold_gauss, hmi_data, 0)
    neg_data = np.where(hmi_data < -cfg.hmi_threshold_gauss, -hmi_data, 0)

    extent = [
        target_wcs.pixel_to_world(0, 0).Tx.value,
        target_wcs.pixel_to_world(target_wcs.array_shape[1], 0).Tx.value,
        target_wcs.pixel_to_world(0, 0).Ty.value,
        target_wcs.pixel_to_world(0, target_wcs.array_shape[0]).Ty.value,
    ]
    if cfg.hmi_levels_gauss:
        # 使用指定级别
        pos_levels = cfg.hmi_levels_gauss
        neg_levels = cfg.hmi_levels_gauss
    else:
        # 自动：最大最小值的百分比
        max_val = np.max(pos_data)
        min_val = np.max(neg_data)
        pos_levels = [max_val * 0.5]
        neg_levels = [min_val * 0.5]

    ax.contour(
        pos_data,
        levels=pos_levels,
        extent=extent,
        colors=cfg.style.hmi_pos_color,
        linewidths=cfg.style.hmi_lw,
        alpha=cfg.style.hmi_alpha,
        origin="lower",
    )
    ax.contour(
        neg_data,
        levels=neg_levels,
        extent=extent,
        colors=cfg.style.hmi_neg_color,
        linewidths=cfg.style.hmi_lw,
        alpha=cfg.style.hmi_alpha,
        origin="lower",
    )


def get_band_color(
    band_label: str, band_idx: int, cfg: Config, color_cache: list | None = None
) -> tuple[str, str]:
    """获取波段主颜色和填充颜色"""
    if cfg.band_colors_dict and band_label in cfg.band_colors_dict:
        return cfg.band_colors_dict[band_label]
    idx = band_idx % len(cfg.default_colors)
    return cfg.default_colors[idx]


def _build_selected_band_legend_elements(
    cfg: Config, color_cache: list | None = None
) -> list[Line2D]:
    """Build legend handles for every user-selected band, independent of fits."""
    legend_elements = []
    for band_idx, band_label in enumerate(cfg.selected_bands):
        color_label = _workflow()._band_color_lookup_label(band_label, cfg)
        color_main, _ = _workflow().get_band_color(
            color_label, band_idx, cfg, color_cache
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color_main,
                lw=2,
                label=_workflow()._selected_band_legend_label(cfg, band_label),
            )
        )
    return legend_elements


def _band_color_lookup_label(band_label: str, cfg: Config) -> str:
    if cfg.band_colors_dict and band_label in cfg.band_colors_dict:
        return band_label
    if "." in band_label:
        return band_label
    return band_label.replace("MHz", ".0MHz")


def _selected_band_legend_label(cfg: Config, band_label: str) -> str:
    if cfg.combine_polarizations and cfg.polarization_mode == "RR+LL":
        return (
            f"{band_label} (RR+LL sum)"
            if not cfg.weighted_average
            else f"{band_label} (RR+LL)"
        )
    return f"{band_label} ({cfg.polarization_mode})"
