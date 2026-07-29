"""Synchronized scientific animations for event/control RMHD runs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import JetConfig, MHDConfig
from ..physics.jet import JetResult, reconnection_flux_rate
from ..physics.radio import RadioResult
from ..physics.rmhd import MHDResult
from .animations import (
    AnimationFormat,
    _canvas_rgb,
    _transcode_delivery,
    _write,
    _write_lossless_master,
    require_mp4_backend,
    validate_animation_formats,
)

SCIENTIFIC_STEMS = (
    "causal_chain",
    "reconnection_topology",
    "bidirectional_outflow",
    "radio_event_control",
)

NAVY = "#0B2545"
TEAL = "#2A7F8E"
ORANGE = "#D97706"
RED = "#C9302C"
GREY = "#5B6573"
LIGHT_GREY = "#D6DCE3"


def _frame_indices(result: MHDResult, render_profile: str) -> np.ndarray:
    count = len(result.times)
    target = count if render_profile == "scientific-4k" else min(count, 120)
    return np.linspace(0, count - 1, target, dtype=int)


def _figure(
    render_profile: str,
    rows: int = 2,
    columns: int = 2,
) -> tuple[plt.Figure, np.ndarray]:
    if render_profile == "scientific-4k":
        figsize, dpi = (12.8, 7.2), 300
    else:
        figsize, dpi = (9.6, 5.4), 100
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
        squeeze=False,
    )
    return fig, axes


def _extent(result: MHDResult) -> tuple[float, float, float, float]:
    return (
        float(result.grid.x.min()),
        float(result.grid.x.max()),
        float(result.grid.y.min()),
        float(result.grid.y.max()),
    )


def _control_index(event: MHDResult, control: MHDResult, index: int) -> int:
    return int(np.argmin(np.abs(control.times - event.times[index])))


def _paired_fields(
    event: MHDResult,
    control: MHDResult,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control_index = _control_index(event, control, index)
    event_fields = event.snapshot_fields(index)
    control_fields = control.snapshot_fields(control_index)
    delta_current = event_fields[4] - control_fields[4]
    return delta_current, event_fields[2], event_fields[3], event_fields[4]


def _limits(
    event: MHDResult,
    control: MHDResult,
) -> tuple[float, float]:
    sample = np.linspace(0, len(event.times) - 1, min(25, len(event.times)), dtype=int)
    delta_values: list[float] = []
    velocity_values: list[float] = []
    for index in sample:
        delta_current, velocity_x, _, _ = _paired_fields(event, control, int(index))
        delta_values.append(float(np.quantile(np.abs(delta_current), 0.995)))
        velocity_values.append(float(np.quantile(np.abs(velocity_x), 0.995)))
    return max(delta_values + [1.0e-8]), max(velocity_values + [1.0e-8])


def _xo_points(
    result: MHDResult,
    config: MHDConfig,
    index: int,
) -> tuple[float, float, float]:
    center = config.sheet_center_fraction * config.ly
    row = int(np.argmin(np.abs(result.grid.y - center)))
    line = result.psi[index, row]
    x_index = int(np.argmin(line))
    o_index = int(np.argmax(line))
    return (
        float(result.grid.x[x_index]),
        float(result.grid.x[o_index]),
        float(result.grid.y[row]),
    )


def _remove_contours(contour) -> None:
    for collection in getattr(contour, "collections", ()):
        collection.remove()


def _radio_cursor_time(event: MHDResult, radio: RadioResult, index: int) -> float:
    fraction = float(index) / max(len(event.times) - 1, 1)
    return float(radio.times_s[0] + fraction * np.ptp(radio.times_s))


def _activity_panel(
    axis: plt.Axes,
    event: MHDResult,
    jet: JetResult,
    jet_config: JetConfig,
) -> tuple[plt.Line2D, plt.Line2D]:
    axis.plot(event.times, jet.jet_activity, color=TEAL, lw=1.8, label="jet activity")
    axis.plot(
        event.times,
        jet.reconnection_activity,
        color=ORANGE,
        lw=1.8,
        label=r"$|d(\psi_O-\psi_X)/dt|$ activity",
    )
    axis.axhline(
        jet_config.jet_threshold,
        color=TEAL,
        lw=0.8,
        ls="--",
        alpha=0.7,
    )
    axis.axhline(
        jet_config.reconnection_threshold,
        color=ORANGE,
        lw=0.8,
        ls=":",
        alpha=0.8,
    )
    cursor = axis.axvline(event.times[0], color=RED, lw=1.2)
    point = axis.plot([event.times[0]], [jet.jet_activity[0]], "o", color=RED, ms=4)[0]
    axis.set(
        xlim=(event.times[0], event.times[-1]),
        ylim=(-0.03, 1.05),
        xlabel="Normalized MHD time",
        ylabel="Normalized activity",
        title="Jet and reconnection gates",
    )
    axis.legend(loc="upper left", fontsize=7, frameon=False)
    axis.grid(True, color=LIGHT_GREY, lw=0.5, alpha=0.65)
    return cursor, point


def iter_causal_chain(
    event: MHDResult,
    control: MHDResult,
    event_jet: JetResult,
    event_radio: RadioResult,
    mhd_config: MHDConfig,
    jet_config: JetConfig,
    render_profile: str,
) -> Iterable[np.ndarray]:
    indices = _frame_indices(event, render_profile)
    current_limit, velocity_limit = _limits(event, control)
    delta_current, velocity_x, _, _ = _paired_fields(event, control, int(indices[0]))
    extent = _extent(event)
    psi_levels = np.linspace(float(event.psi.min()), float(event.psi.max()), 11)
    fig, axes = _figure(render_profile)
    current_axis, velocity_axis, activity_axis, radio_axis = axes.ravel()
    current_image = current_axis.imshow(
        delta_current,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-current_limit,
        vmax=current_limit,
    )
    current_contour = current_axis.contour(
        event.grid.x_mesh,
        event.grid.y_mesh,
        event.psi[int(indices[0])],
        levels=psi_levels,
        colors="black",
        linewidths=0.30,
        alpha=0.55,
    )
    xpoint, opoint, sheet_y = _xo_points(event, mhd_config, int(indices[0]))
    x_marker = current_axis.plot(xpoint, sheet_y, "x", color=ORANGE, ms=6, mew=1.3)[0]
    o_marker = current_axis.plot(
        opoint,
        sheet_y,
        "o",
        mfc="none",
        mec=NAVY,
        ms=6,
        mew=1.0,
    )[0]
    current_axis.set(xlabel="x", ylabel="y")
    fig.colorbar(current_image, ax=current_axis, label=r"$j_z(event)-j_z(control)$")

    velocity_image = velocity_axis.imshow(
        velocity_x,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-velocity_limit,
        vmax=velocity_limit,
    )
    velocity_contour = velocity_axis.contour(
        event.grid.x_mesh,
        event.grid.y_mesh,
        event.psi[int(indices[0])],
        levels=psi_levels,
        colors="black",
        linewidths=0.30,
        alpha=0.50,
    )
    velocity_axis.set(xlabel="x", ylabel="y")
    fig.colorbar(velocity_image, ax=velocity_axis, label=r"Signed outflow $v_x$")
    activity_cursor, activity_point = _activity_panel(
        activity_axis,
        event,
        event_jet,
        jet_config,
    )

    radio_axis.pcolormesh(
        event_radio.times_s,
        event_radio.frequencies_mhz,
        event_radio.intensity,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    radio_axis.plot(
        event_radio.times_s,
        event_radio.ridge_frequency_mhz,
        color="#67E8F9",
        lw=1.2,
        label="Type III ridge",
    )
    if event_radio.spike_catalog.size:
        radio_axis.scatter(
            event_radio.spike_catalog[:, 0],
            event_radio.spike_catalog[:, 1],
            facecolors="none",
            edgecolors="white",
            linewidths=0.7,
            s=18,
            label="strict topping spikes",
        )
    radio_cursor = radio_axis.axvline(event_radio.times_s[0], color="white", lw=1.0)
    radio_axis.set(
        xlim=(event_radio.times_s[0], event_radio.times_s[-1]),
        ylim=(event_radio.frequencies_mhz.min(), event_radio.frequencies_mhz.max()),
        xlabel="Radio-proxy time (s)",
        ylabel="Frequency (MHz)",
        title="Synthetic dynamic spectrum",
    )
    radio_axis.legend(loc="upper right", fontsize=7, frameon=False)

    for index in indices:
        index = int(index)
        time_value = float(event.times[index])
        delta_current, velocity_x, _, _ = _paired_fields(event, control, index)
        current_image.set_data(delta_current)
        velocity_image.set_data(velocity_x)
        _remove_contours(current_contour)
        current_contour = current_axis.contour(
            event.grid.x_mesh,
            event.grid.y_mesh,
            event.psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.30,
            alpha=0.55,
        )
        _remove_contours(velocity_contour)
        velocity_contour = velocity_axis.contour(
            event.grid.x_mesh,
            event.grid.y_mesh,
            event.psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.30,
            alpha=0.50,
        )
        xpoint, opoint, sheet_y = _xo_points(event, mhd_config, index)
        x_marker.set_data([xpoint], [sheet_y])
        o_marker.set_data([opoint], [sheet_y])
        current_axis.set_title(f"Tearing current relative to control  |  t={time_value:.2f}")
        velocity_axis.set_title(f"Bidirectional reconnection outflow  |  t={time_value:.2f}")
        activity_cursor.set_xdata([time_value, time_value])
        activity_point.set_data([time_value], [event_jet.jet_activity[index]])
        radio_time = _radio_cursor_time(event, event_radio, index)
        radio_cursor.set_xdata([radio_time, radio_time])
        fig.suptitle(
            "Reduced-MHD to Type III proxy causal chain",
            color=NAVY,
            fontsize=12,
        )
        yield _canvas_rgb(fig)
    plt.close(fig)


def iter_reconnection_topology(
    event: MHDResult,
    control: MHDResult,
    mhd_config: MHDConfig,
    render_profile: str,
) -> Iterable[np.ndarray]:
    indices = _frame_indices(event, render_profile)
    current_limit, _ = _limits(event, control)
    extent = _extent(event)
    psi_levels = np.linspace(float(event.psi.min()), float(event.psi.max()), 13)
    delta_current, _, _, _ = _paired_fields(event, control, int(indices[0]))
    rate = reconnection_flux_rate(event)
    fig, axes = _figure(render_profile, 1, 2)
    field_axis, diagnostic_axis = axes.ravel()
    image = field_axis.imshow(
        delta_current,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-current_limit,
        vmax=current_limit,
    )
    contour = field_axis.contour(
        event.grid.x_mesh,
        event.grid.y_mesh,
        event.psi[int(indices[0])],
        levels=psi_levels,
        colors="black",
        linewidths=0.35,
        alpha=0.65,
    )
    xpoint, opoint, sheet_y = _xo_points(event, mhd_config, int(indices[0]))
    x_marker = field_axis.plot(xpoint, sheet_y, "x", color=ORANGE, ms=8, mew=1.4)[0]
    o_marker = field_axis.plot(
        opoint,
        sheet_y,
        "o",
        mfc="none",
        mec=NAVY,
        ms=7,
        mew=1.2,
    )[0]
    field_axis.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=field_axis, label=r"$\delta j_z$")
    diagnostic_axis.plot(
        event.times,
        event.flux_difference,
        color=NAVY,
        lw=1.8,
        label=r"$|\psi_O-\psi_X|$",
    )
    diagnostic_axis.plot(
        event.times,
        event.island_width_proxy,
        color=TEAL,
        lw=1.8,
        label="island-width proxy",
    )
    rate_scaled = rate / max(float(np.max(rate)), 1.0e-15)
    diagnostic_axis.plot(
        event.times,
        rate_scaled,
        color=ORANGE,
        lw=1.8,
        label=r"$R_\psi/\max R_\psi$",
    )
    cursor = diagnostic_axis.axvline(event.times[0], color=RED, lw=1.2)
    diagnostic_axis.set(
        xlim=(event.times[0], event.times[-1]),
        xlabel="Normalized MHD time",
        ylabel="Diagnostic value",
        title="Topology and reconnection diagnostics",
    )
    diagnostic_axis.legend(loc="best", fontsize=8, frameon=False)
    diagnostic_axis.grid(True, color=LIGHT_GREY, lw=0.5)
    for index in indices:
        index = int(index)
        delta_current, _, _, _ = _paired_fields(event, control, index)
        image.set_data(delta_current)
        _remove_contours(contour)
        contour = field_axis.contour(
            event.grid.x_mesh,
            event.grid.y_mesh,
            event.psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.35,
            alpha=0.65,
        )
        xpoint, opoint, sheet_y = _xo_points(event, mhd_config, index)
        x_marker.set_data([xpoint], [sheet_y])
        o_marker.set_data([opoint], [sheet_y])
        time_value = float(event.times[index])
        cursor.set_xdata([time_value, time_value])
        field_axis.set_title(f"Event-control current and flux topology  |  t={time_value:.2f}")
        fig.suptitle("Tearing topology and localized reconnection", color=NAVY)
        yield _canvas_rgb(fig)
    plt.close(fig)


def iter_bidirectional_outflow(
    event: MHDResult,
    event_jet: JetResult,
    mhd_config: MHDConfig,
    jet_config: JetConfig,
    render_profile: str,
) -> Iterable[np.ndarray]:
    indices = _frame_indices(event, render_profile)
    extent = _extent(event)
    velocity_limit = max(float(np.quantile(np.abs(event_jet.absolute_speed), 0.995)), 1e-8)
    _, _, velocity_x, _, _, _ = event.snapshot_fields(int(indices[0]))
    psi_levels = np.linspace(float(event.psi.min()), float(event.psi.max()), 13)
    fig, axes = _figure(render_profile, 1, 2)
    field_axis, diagnostic_axis = axes.ravel()
    image = field_axis.imshow(
        velocity_x,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-velocity_limit,
        vmax=velocity_limit,
    )
    contour = field_axis.contour(
        event.grid.x_mesh,
        event.grid.y_mesh,
        event.psi[int(indices[0])],
        levels=psi_levels,
        colors="black",
        linewidths=0.35,
        alpha=0.65,
    )
    field_axis.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=field_axis, label=r"Signed outflow $v_x$")
    diagnostic_axis.plot(
        event.times,
        event_jet.positive_speed,
        color=RED,
        lw=1.8,
        label=r"$v_x>0$ quantile",
    )
    diagnostic_axis.plot(
        event.times,
        -event_jet.negative_speed,
        color=NAVY,
        lw=1.8,
        label=r"$v_x<0$ quantile",
    )
    diagnostic_axis.plot(
        event.times,
        event_jet.bidirectional_speed,
        color=TEAL,
        lw=1.8,
        label="bidirectional minimum",
    )
    if event_jet.onset_time_normalized is not None:
        diagnostic_axis.axvline(
            event_jet.onset_time_normalized,
            color=ORANGE,
            ls="--",
            lw=1.0,
            label="sustained onset",
        )
    peak_index = int(np.argmax(event_jet.bidirectional_speed))
    diagnostic_axis.axvline(
        event.times[peak_index],
        color=GREY,
        ls=":",
        lw=1.0,
        label="peak outflow",
    )
    cursor = diagnostic_axis.axvline(event.times[0], color=RED, lw=1.2)
    diagnostic_axis.set(
        xlim=(event.times[0], event.times[-1]),
        xlabel="Normalized MHD time",
        ylabel="Signed sheet-localized velocity",
        title="Bidirectional outflow diagnostics",
    )
    diagnostic_axis.legend(loc="best", fontsize=8, frameon=False)
    diagnostic_axis.grid(True, color=LIGHT_GREY, lw=0.5)
    for index in indices:
        index = int(index)
        _, _, velocity_x, _, _, _ = event.snapshot_fields(index)
        image.set_data(velocity_x)
        _remove_contours(contour)
        contour = field_axis.contour(
            event.grid.x_mesh,
            event.grid.y_mesh,
            event.psi[index],
            levels=psi_levels,
            colors="black",
            linewidths=0.35,
            alpha=0.65,
        )
        time_value = float(event.times[index])
        cursor.set_xdata([time_value, time_value])
        field_axis.set_title(f"Signed reconnection outflow  |  t={time_value:.2f}")
        fig.suptitle(
            "Where and when bidirectional reconnection jets form",
            color=NAVY,
        )
        yield _canvas_rgb(fig)
    plt.close(fig)


def iter_radio_event_control(
    event: MHDResult,
    event_jet: JetResult,
    control_jet: JetResult,
    event_radio: RadioResult,
    control_radio: RadioResult,
    jet_config: JetConfig,
    render_profile: str,
) -> Iterable[np.ndarray]:
    indices = _frame_indices(event, render_profile)
    fig, axes = _figure(render_profile)
    event_axis, control_axis, gate_axis, mapping_axis = axes.ravel()
    for axis, radio, title in (
        (event_axis, event_radio, "Event: strict jet + reconnection gate"),
        (control_axis, control_radio, "Control: zero perturbation"),
    ):
        axis.pcolormesh(
            radio.times_s,
            radio.frequencies_mhz,
            radio.intensity,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axis.plot(
            radio.times_s,
            radio.ridge_frequency_mhz,
            color="#67E8F9",
            lw=1.2,
        )
        if radio.spike_catalog.size:
            axis.scatter(
                radio.spike_catalog[:, 0],
                radio.spike_catalog[:, 1],
                facecolors="none",
                edgecolors="white",
                linewidths=0.7,
                s=20,
            )
        axis.set(
            xlim=(radio.times_s[0], radio.times_s[-1]),
            ylim=(radio.frequencies_mhz.min(), radio.frequencies_mhz.max()),
            xlabel="Time (s)",
            ylabel="Frequency (MHz)",
            title=title,
        )
    event_cursor = event_axis.axvline(event_radio.times_s[0], color="white", lw=1.0)
    control_cursor = control_axis.axvline(
        control_radio.times_s[0],
        color="white",
        lw=1.0,
    )
    gate_axis.plot(
        event_radio.times_s,
        event_radio.jet_activity,
        color=TEAL,
        lw=1.7,
        label="event jet",
    )
    gate_axis.plot(
        event_radio.times_s,
        event_radio.conditioned_reconnection_activity,
        color=ORANGE,
        lw=1.7,
        label="event reconnection",
    )
    gate_axis.plot(
        control_radio.times_s,
        control_radio.jet_activity,
        color=GREY,
        lw=1.0,
        ls="--",
        label="control jet",
    )
    gate_axis.axhline(jet_config.jet_threshold, color=TEAL, ls="--", lw=0.8)
    gate_axis.axhline(
        jet_config.reconnection_threshold,
        color=ORANGE,
        ls=":",
        lw=0.8,
    )
    gate_cursor = gate_axis.axvline(event_radio.times_s[0], color=RED, lw=1.2)
    gate_axis.set(
        xlim=(event_radio.times_s[0], event_radio.times_s[-1]),
        ylim=(-0.03, 1.05),
        xlabel="Time (s)",
        ylabel="Normalized activity",
        title="Conditioned event gates",
    )
    gate_axis.legend(loc="upper right", fontsize=7, frameon=False)
    gate_axis.grid(True, color=LIGHT_GREY, lw=0.5)
    mapping_axis.plot(
        event_radio.beam_height_mm,
        event_radio.ridge_frequency_mhz,
        color=NAVY,
        lw=1.8,
    )
    map_point = mapping_axis.plot(
        [event_radio.beam_height_mm[0]],
        [event_radio.ridge_frequency_mhz[0]],
        "o",
        color=ORANGE,
        ms=6,
    )[0]
    mapping_axis.set(
        xlabel="Electron-beam height (Mm)",
        ylabel="Plasma-frequency proxy (MHz)",
        title="Exponential density mapping",
    )
    mapping_axis.grid(True, color=LIGHT_GREY, lw=0.5)
    for index in indices:
        index = int(index)
        radio_time = _radio_cursor_time(event, event_radio, index)
        radio_index = int(np.argmin(np.abs(event_radio.times_s - radio_time)))
        event_cursor.set_xdata([radio_time, radio_time])
        control_cursor.set_xdata([radio_time, radio_time])
        gate_cursor.set_xdata([radio_time, radio_time])
        map_point.set_data(
            [event_radio.beam_height_mm[radio_index]],
            [event_radio.ridge_frequency_mhz[radio_index]],
        )
        fig.suptitle(
            "Spike-topping event versus no-event control"
            f"  |  event spikes={len(event_radio.spike_catalog)}, "
            f"control spikes={len(control_radio.spike_catalog)}",
            color=NAVY,
        )
        yield _canvas_rgb(fig)
    plt.close(fig)


def _gif_from_master(master: Path, output: Path) -> None:
    subprocess.run(
        [
            require_mp4_backend(),
            "-y",
            "-i",
            str(master),
            "-vf",
            "fps=10,scale=960:540:flags=lanczos",
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def save_scientific_animations(
    event: MHDResult,
    control: MHDResult,
    event_jet: JetResult,
    control_jet: JetResult,
    event_radio: RadioResult,
    control_radio: RadioResult,
    mhd_config: MHDConfig,
    jet_config: JetConfig,
    animations_dir: Path,
    formats: tuple[AnimationFormat, ...],
    *,
    render_profile: str,
) -> list[Path]:
    """Render the four traceable scientific event/control animations."""

    validate_animation_formats(formats)
    animations_dir.mkdir(parents=True, exist_ok=True)
    renderers: dict[str, Callable[[str], Iterable[np.ndarray]]] = {
        "causal_chain": lambda profile: iter_causal_chain(
            event,
            control,
            event_jet,
            event_radio,
            mhd_config,
            jet_config,
            profile,
        ),
        "reconnection_topology": lambda profile: iter_reconnection_topology(
            event,
            control,
            mhd_config,
            profile,
        ),
        "bidirectional_outflow": lambda profile: iter_bidirectional_outflow(
            event,
            event_jet,
            mhd_config,
            jet_config,
            profile,
        ),
        "radio_event_control": lambda profile: iter_radio_event_control(
            event,
            event_jet,
            control_jet,
            event_radio,
            control_radio,
            jet_config,
            profile,
        ),
    }
    paths: list[Path] = []
    encoders: dict[str, str] = {}
    for stem, renderer in renderers.items():
        if render_profile == "scientific-4k" and "mp4" in formats:
            master = animations_dir / f"{stem}_master_ffv1.mkv"
            delivery = animations_dir / f"{stem}.mp4"
            _write_lossless_master(master, renderer(render_profile), fps=30)
            encoders[stem] = _transcode_delivery(master, delivery)
            paths.extend((master, delivery))
            if "gif" in formats:
                preview = animations_dir / f"{stem}.gif"
                _gif_from_master(master, preview)
                paths.append(preview)
        else:
            profile = "scientific-preview"
            for animation_format in formats:
                path = animations_dir / f"{stem}.{animation_format}"
                _write(path, list(renderer(profile)), fps=10)
                paths.append(path)
    report = animations_dir / "media_encoding.json"
    report.write_text(
        json.dumps(
            {
                "schema": "scientific-media-v1",
                "stems": list(SCIENTIFIC_STEMS),
                "fps": 30 if render_profile == "scientific-4k" else 10,
                "frames": len(_frame_indices(event, render_profile)),
                "delivery_encoders": encoders,
                "nvenc_fallback_policy": "libx264-crf17",
                "interpolation": "none",
                "color_scales": "fixed over the complete sequence",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(report)
    return paths
