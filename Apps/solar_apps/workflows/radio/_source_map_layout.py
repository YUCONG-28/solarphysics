"""Source-map canvas geometry, shared axes and scientific colorbar layout."""

from __future__ import annotations

import math
import numpy as np


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


def _layout_grid(n: int):
    """Automatically calculate subplot layout"""
    if n <= 0:
        return 1, 1
    ncol = max(1, math.ceil(math.sqrt(n)))
    nrow = max(1, math.ceil(n / ncol))
    return nrow, ncol


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
