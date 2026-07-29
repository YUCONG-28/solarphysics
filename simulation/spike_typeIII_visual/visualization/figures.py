"""Evidence-first static figures for the simulation and slide deck."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ..config import JetConfig, MHDConfig
from ..physics.fields import MHDFieldSeries
from ..physics.jet import JetResult
from ..physics.radio import RadioResult
from ..physics.rmhd import MHDResult

NAVY = "#0B2545"
TEAL = "#2A7F8E"
ORANGE = "#D97706"
RED = "#C9302C"
MUTED = "#4B5563"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


_RENDER_PROFILE = "legacy"


def _save(fig: plt.Figure, path: Path) -> None:
    dpi = 300 if _RENDER_PROFILE == "presentation-4k" else 125
    fig.savefig(path, dpi=dpi, bbox_inches=None, facecolor="white")
    if _RENDER_PROFILE == "presentation-4k":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
        fig.savefig(path.with_suffix(".svg"), bbox_inches=None, facecolor="white")
    plt.close(fig)


def _figure() -> tuple[plt.Figure, plt.Axes]:
    fig, axis = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    return fig, axis


def save_harris_field(
    result: MHDResult | MHDFieldSeries,
    config: MHDConfig,
    path: Path,
) -> None:
    magnetic_x, magnetic_y, _, _, current_map, _ = result.snapshot_fields(0)
    x_index = int(np.argmin(np.abs(result.grid.x)))

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    field_axis = fig.add_subplot(grid[0, 0])
    profile_axis = fig.add_subplot(grid[0, 1])
    color = np.hypot(magnetic_x, magnetic_y)
    field_axis.streamplot(
        result.grid.x,
        result.grid.y,
        magnetic_x,
        magnetic_y,
        color=color,
        cmap="Blues",
        density=1.7,
        linewidth=1.2,
        arrowsize=1.0,
    )
    sheet_center = config.sheet_center_fraction * config.ly
    for center in (-sheet_center, sheet_center):
        field_axis.axhspan(
            center - config.sheet_half_width,
            center + config.sheet_half_width,
            color=RED,
            alpha=0.10,
        )
        field_axis.axhline(center, color=RED, linewidth=1.0)
    field_axis.set(
        title="Periodic double-Harris initial field",
        xlabel="x (normalized)",
        ylabel="y (normalized)",
        xlim=(result.grid.x.min(), result.grid.x.max()),
        ylim=(result.grid.y.min(), result.grid.y.max()),
    )
    profile_axis.plot(
        magnetic_x[:, x_index],
        result.grid.y,
        color=NAVY,
        linewidth=2.4,
        label=r"$B_x$",
    )
    profile_axis.plot(
        current_map[:, x_index],
        result.grid.y,
        color=RED,
        linewidth=2.1,
        label=r"$J_z$",
    )
    profile_axis.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    profile_axis.grid(True)
    profile_axis.set(
        title="Two field reversals and current layers",
        xlabel="Normalized amplitude",
        ylabel="y / L",
    )
    profile_axis.legend(frameon=False)
    fig.suptitle(
        "The simulated initial condition is periodic and contains two sheets",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_tearing_structure(result: MHDResult | MHDFieldSeries, path: Path) -> None:
    indices = [0, len(result.times) // 2, len(result.times) - 1]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 7.2), constrained_layout=True, sharex=True, sharey=True
    )
    for axis, index in zip(axes, indices, strict=True):
        _, _, _, _, current, _ = result.snapshot_fields(index)
        limit = max(float(np.percentile(np.abs(current), 98.0)), 1.0)
        axis.imshow(
            current,
            extent=(
                result.grid.x.min(),
                result.grid.x.max(),
                result.grid.y.min(),
                result.grid.y.max(),
            ),
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            alpha=0.78,
        )
        axis.contour(
            result.grid.x_mesh,
            result.grid.y_mesh,
            result.psi[index],
            levels=18,
            colors="#111827",
            linewidths=0.55,
            alpha=0.75,
        )
        axis.set_title(f"t = {result.times[index]:.2f}")
        axis.set_xlabel("x (normalized)")
    axes[0].set_ylabel("y (normalized)")
    fig.suptitle(
        "Tearing perturbation reorganizes the current sheets into magnetic islands",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_current_density(result: MHDResult | MHDFieldSeries, path: Path) -> None:
    index = len(result.times) - 1
    _, _, _, _, current, _ = result.snapshot_fields(index)
    limit = float(np.percentile(np.abs(current), 99.5))
    upper_sheet_index = int(np.argmin(np.abs(result.grid.y - np.pi / 2.0)))

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    map_axis = fig.add_subplot(grid[0, 0])
    cut_axis = fig.add_subplot(grid[0, 1])
    image = map_axis.imshow(
        current,
        extent=(
            result.grid.x.min(),
            result.grid.x.max(),
            result.grid.y.min(),
            result.grid.y.max(),
        ),
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    map_axis.contour(
        result.grid.x_mesh,
        result.grid.y_mesh,
        result.psi[index],
        levels=16,
        colors="black",
        linewidths=0.45,
        alpha=0.55,
    )
    map_axis.set(
        title=f"Current density at t = {result.times[index]:.2f}",
        xlabel="x (normalized)",
        ylabel="y (normalized)",
    )
    fig.colorbar(image, ax=map_axis, shrink=0.82, label=r"$J_z$")
    cut_axis.plot(
        result.grid.x,
        current[upper_sheet_index],
        color=RED,
        linewidth=2.0,
    )
    cut_axis.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    cut_axis.grid(True)
    cut_axis.set(
        title="Upper-sheet current cut",
        xlabel="x (normalized)",
        ylabel=r"$J_z$",
    )
    fig.suptitle(
        "Tearing concentrates current into localized X/O-point structure",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_jet_structure(
    result: MHDResult | MHDFieldSeries,
    jet: JetResult,
    path: Path,
) -> None:
    index = int(np.argmax(result.max_speed))
    _, _, velocity_x, velocity_y, current, _ = result.snapshot_fields(index)
    speed = np.hypot(velocity_x, velocity_y)
    stride = max(1, result.grid.x.size // 24)

    fig, axis = _figure()
    image = axis.imshow(
        speed,
        extent=(
            result.grid.x.min(),
            result.grid.x.max(),
            result.grid.y.min(),
            result.grid.y.max(),
        ),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axis.quiver(
        result.grid.x_mesh[::stride, ::stride],
        result.grid.y_mesh[::stride, ::stride],
        velocity_x[::stride, ::stride],
        velocity_y[::stride, ::stride],
        color="white",
        alpha=0.82,
        scale=3.3,
        width=0.0024,
    )
    axis.contour(
        result.grid.x_mesh,
        result.grid.y_mesh,
        current,
        levels=[-4.0, 4.0],
        colors=[TEAL, RED],
        linewidths=1.0,
        alpha=0.85,
    )
    axis.contour(
        result.grid.x_mesh,
        result.grid.y_mesh,
        jet.sheet_mask.astype(float),
        levels=[0.5],
        colors=["#FDE68A"],
        linewidths=1.2,
    )
    axis.set(
        title=f"Velocity magnitude and direction at t = {result.times[index]:.2f}",
        xlabel="x (normalized)",
        ylabel="y (normalized)",
    )
    fig.colorbar(image, ax=axis, shrink=0.82, label="|v| (normalized)")
    fig.suptitle(
        "Opposite-sign sheet-localized outflows are tested with a robust jet proxy",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_jet_diagnostics(
    result: MHDResult | MHDFieldSeries,
    jet: JetResult,
    config: JetConfig,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0].plot(
        result.times,
        jet.positive_speed,
        color=TEAL,
        linewidth=2.2,
        label=r"$V_J^+$",
    )
    axes[0].plot(
        result.times,
        jet.negative_speed,
        color=ORANGE,
        linewidth=2.2,
        label=r"$V_J^-$",
    )
    axes[0].plot(
        result.times,
        jet.bidirectional_speed,
        color=NAVY,
        linewidth=2.4,
        label=r"$\min(V_J^+,V_J^-)$",
    )
    axes[0].set(
        title=f"Sheet-localized p={config.velocity_quantile:.2f} quantiles",
        xlabel="Normalized MHD time",
        ylabel="Normalized speed",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        result.times,
        jet.jet_activity,
        color=TEAL,
        linewidth=2.4,
        label=r"$q_J$",
    )
    axes[1].plot(
        result.times,
        jet.reconnection_activity,
        color=RED,
        linewidth=2.4,
        label=r"$q_R$",
    )
    axes[1].axhline(
        config.jet_threshold,
        color=NAVY,
        linestyle="--",
        linewidth=1.2,
        label="threshold",
    )
    if jet.onset_time_normalized is not None:
        axes[1].axvline(
            jet.onset_time_normalized,
            color=ORANGE,
            linestyle=":",
            linewidth=2.0,
            label=r"$\tau_J$",
        )
    axes[1].set(
        title="Normalized jet and reconnection activity",
        xlabel="Normalized MHD time",
        ylabel="Activity",
        ylim=(-0.03, 1.05),
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True)
    fig.suptitle(
        "Jet onset requires a sustained, bidirectional sheet-localized signal",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_activity_timeline(
    radio: RadioResult,
    config: JetConfig,
    path: Path,
) -> None:
    fig, axis = _figure()
    axis.plot(
        radio.times_s,
        radio.jet_activity,
        color=TEAL,
        linewidth=2.5,
        label=r"$q_J(t)$",
    )
    axis.plot(
        radio.times_s,
        radio.conditioned_reconnection_activity,
        color=RED,
        linewidth=2.5,
        label=r"$q_R(t)$",
    )
    axis.axhline(
        config.jet_threshold,
        color=NAVY,
        linestyle="--",
        linewidth=1.2,
        label="0.6 threshold",
    )
    if radio.spike_catalog.size:
        for center_time in radio.spike_catalog[:, 0]:
            axis.axvline(center_time, color=ORANGE, alpha=0.38, linewidth=1.0)
        axis.scatter(
            radio.spike_catalog[:, 0],
            np.full(len(radio.spike_catalog), 1.02),
            marker="v",
            color=ORANGE,
            s=38,
            label="spike centers",
        )
    axis.set(
        title="Compressed onset mapping used only for event conditioning",
        xlabel="Radio-proxy time (s)",
        ylabel="Activity",
        xlim=(radio.times_s.min(), radio.times_s.max()),
        ylim=(-0.03, 1.08),
    )
    axis.grid(True)
    axis.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Every conditioned spike must satisfy both jet and reconnection thresholds",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_electron_propagation(radio: RadioResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0].plot(
        radio.times_s,
        radio.beam_height_mm,
        color=NAVY,
        linewidth=2.6,
    )
    axes[0].fill_between(
        radio.times_s,
        0.0,
        radio.beam_height_mm,
        color=TEAL,
        alpha=0.12,
    )
    axes[0].grid(True)
    axes[0].set(
        title=f"Kinematic beam: v = 0.20 c, γ = {radio.beam_gamma:.4f}",
        xlabel="Time (s)",
        ylabel="Height above injection site (Mm)",
    )
    axes[1].plot(
        radio.times_s,
        radio.injection_activity,
        color=RED,
        linewidth=2.4,
    )
    axes[1].fill_between(
        radio.times_s,
        0.0,
        radio.injection_activity,
        color=ORANGE,
        alpha=0.18,
    )
    axes[1].grid(True)
    axes[1].set(
        title="MHD reconnection proxy modulates electron injection",
        xlabel="Radio-proxy time (s)",
        ylabel="Normalized injection activity",
        ylim=(-0.02, 1.05),
    )
    fig.suptitle(
        "Electron propagation remains a kinematic proxy, not a test-particle solution",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_dynamic_spectrum(radio: RadioResult, path: Path) -> None:
    fig, axis = _figure()
    image = axis.pcolormesh(
        radio.times_s,
        radio.frequencies_mhz,
        radio.intensity,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axis.plot(
        radio.times_s,
        radio.ridge_frequency_mhz,
        color="#67E8F9",
        linewidth=1.8,
        label="Density-model plasma-frequency ridge",
    )
    if radio.spike_catalog.size:
        axis.scatter(
            radio.spike_catalog[:, 0],
            radio.spike_catalog[:, 1],
            facecolors="none",
            edgecolors="#FDE68A",
            s=44,
            linewidths=1.0,
            label="Imposed spike centers",
        )
    axis.set(
        title="Synthetic dynamic spectrum",
        xlabel="Time (s)",
        ylabel="Frequency (MHz)",
        xlim=(radio.times_s.min(), radio.times_s.max()),
        ylim=(radio.frequencies_mhz.min(), radio.frequencies_mhz.max()),
    )
    axis.legend(loc="upper right", framealpha=0.82)
    fig.colorbar(image, ax=axis, shrink=0.82, label="Normalized intensity")
    fig.suptitle(
        "An exponential coronal density model produces a Type III-like downward drift",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def save_mhd_diagnostics(
    result: MHDResult | MHDFieldSeries,
    config: MHDConfig,
    path: Path,
) -> None:
    total_energy = (
        result.total_energy
        if isinstance(result, MHDFieldSeries)
        else result.magnetic_energy + result.kinetic_energy
    )
    drift_percent = 100.0 * (total_energy[-1] - total_energy[0]) / total_energy[0]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0, 0].plot(result.times, result.magnetic_energy, label="Magnetic", color=NAVY)
    axes[0, 0].plot(result.times, result.kinetic_energy, label="Kinetic", color=ORANGE)
    axes[0, 0].plot(result.times, total_energy, label="Total", color=TEAL)
    axes[0, 0].set(title=f"Energy diagnostics (drift {drift_percent:+.2f}%)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(result.times, result.max_current, color=RED)
    axes[0, 1].set(title="Peak |Jz|")
    axes[1, 0].plot(result.times, result.max_speed, color=TEAL)
    axes[1, 0].set(title="Peak flow speed")
    axes[1, 1].plot(result.times, result.reconnection_proxy, color=ORANGE)
    reconnection_title = (
        r"$|d(\psi_O-\psi_X)/dt|$"
        if isinstance(result, MHDFieldSeries)
        else r"Reduced-MHD reconnection proxy"
    )
    axes[1, 1].set(title=reconnection_title)
    for axis in axes.flat:
        axis.grid(True)
        axis.set_xlabel("Normalized time")
    fig.suptitle(
        (
            "Athena full-MHD"
            if isinstance(result, MHDFieldSeries)
            else "Reduced-MHD"
        )
        + f" run: {config.nx}×{config.ny}, η=ν={config.resistivity:g}",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, path)


def _keyframe_indices(length: int) -> list[int]:
    return [0, max(1, length // 2), length - 1]


def save_keyframe_strips(
    result: MHDResult | MHDFieldSeries,
    radio: RadioResult,
    figures_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    indices = _keyframe_indices(len(result.times))

    tearing_path = figures_dir / "tearing_keyframes.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for axis, index in zip(axes, indices, strict=True):
        _, _, _, _, current, _ = result.snapshot_fields(index)
        axis.imshow(current, origin="lower", aspect="auto", cmap="RdBu_r")
        axis.contour(result.psi[index], levels=14, colors="black", linewidths=0.45)
        axis.set_title(f"t={result.times[index]:.2f}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("tearing.gif — magnetic-island evolution", color=NAVY, fontsize=18)
    _save(fig, tearing_path)
    paths.append(tearing_path)

    jet_path = figures_dir / "jet_keyframes.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for axis, index in zip(axes, indices, strict=True):
        _, _, vx, vy, _, _ = result.snapshot_fields(index)
        speed = np.hypot(vx, vy)
        axis.imshow(speed, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(f"t={result.times[index]:.2f}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("jet.gif — bidirectional flow growth", color=NAVY, fontsize=18)
    _save(fig, jet_path)
    paths.append(jet_path)

    electron_path = figures_dir / "electron_beam_keyframes.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), constrained_layout=True)
    radio_indices = _keyframe_indices(len(radio.times_s))
    for axis, index in zip(axes, radio_indices, strict=True):
        axis.plot(
            radio.times_s[: index + 1],
            radio.beam_height_mm[: index + 1],
            color=NAVY,
            linewidth=2.4,
        )
        axis.scatter(
            [radio.times_s[index]],
            [radio.beam_height_mm[index]],
            color=ORANGE,
            s=60,
            zorder=3,
        )
        axis.set(
            title=f"t={radio.times_s[index]:.2f} s",
            xlim=(0.0, radio.times_s[-1]),
            ylim=(0.0, radio.beam_height_mm[-1] * 1.05),
        )
        axis.grid(True)
    fig.suptitle(
        "electron_beam.gif — kinematic height-time propagation",
        color=NAVY,
        fontsize=18,
    )
    _save(fig, electron_path)
    paths.append(electron_path)

    spectrum_path = figures_dir / "typeIII_keyframes.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for axis, index in zip(axes, radio_indices, strict=True):
        axis.pcolormesh(
            radio.times_s[: index + 1],
            radio.frequencies_mhz,
            radio.intensity[:, : index + 1],
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        axis.set(
            title=f"t={radio.times_s[index]:.2f} s",
            xlim=(0.0, radio.times_s[-1]),
            ylim=(radio.frequencies_mhz.min(), radio.frequencies_mhz.max()),
        )
    fig.suptitle(
        "typeIII.gif — progressive synthetic dynamic spectrum",
        color=NAVY,
        fontsize=18,
    )
    _save(fig, spectrum_path)
    paths.append(spectrum_path)
    return paths


def save_static_figures(
    result: MHDResult | MHDFieldSeries,
    jet: JetResult,
    radio: RadioResult,
    config: MHDConfig,
    jet_config: JetConfig,
    figures_dir: Path,
    *,
    render_profile: str = "legacy",
) -> list[Path]:
    """Write all static figures and keyframe strips."""

    global _RENDER_PROFILE
    if render_profile not in {"legacy", "preview", "presentation-4k"}:
        raise ValueError("Unknown render profile.")
    _RENDER_PROFILE = render_profile
    _apply_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    writers = [
        (
            lambda path: save_harris_field(result, config, path),
            figures_dir / "harris_field.png",
        ),
        (
            lambda path: save_tearing_structure(result, path),
            figures_dir / "tearing_structure.png",
        ),
        (
            lambda path: save_current_density(result, path),
            figures_dir / "current_density.png",
        ),
        (
            lambda path: save_jet_structure(result, jet, path),
            figures_dir / "jet_structure.png",
        ),
        (
            lambda path: save_jet_diagnostics(
                result,
                jet,
                jet_config,
                path,
            ),
            figures_dir / "jet_diagnostics.png",
        ),
        (
            lambda path: save_activity_timeline(radio, jet_config, path),
            figures_dir / "activity_timeline.png",
        ),
        (
            lambda path: save_electron_propagation(radio, path),
            figures_dir / "electron_propagation.png",
        ),
        (
            lambda path: save_dynamic_spectrum(radio, path),
            figures_dir / "typeIII_dynamic_spectrum.png",
        ),
        (
            lambda path: save_mhd_diagnostics(result, config, path),
            figures_dir / "mhd_diagnostics.png",
        ),
    ]
    paths: list[Path] = []
    for writer, path in writers:
        writer(path)
        paths.append(path)
        if render_profile == "presentation-4k":
            paths.extend((path.with_suffix(".pdf"), path.with_suffix(".svg")))
    paths.extend(save_keyframe_strips(result, radio, figures_dir))
    if render_profile == "presentation-4k":
        for path in list(paths):
            if path.suffix == ".png" and path.with_suffix(".pdf") not in paths:
                paths.extend((path.with_suffix(".pdf"), path.with_suffix(".svg")))
    return paths
