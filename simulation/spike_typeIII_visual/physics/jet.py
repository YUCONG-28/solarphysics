"""Sheet-localized bidirectional reconnection-outflow diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import JetConfig, MHDConfig
from .fields import MHDFieldSeries
from .rmhd import MHDResult, SpectralGrid

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class JetResult:
    """Time series used to test and condition jet-associated radio spikes."""

    sheet_mask: BoolArray
    positive_speed: FloatArray
    negative_speed: FloatArray
    bidirectional_speed: FloatArray
    absolute_speed: FloatArray
    global_jet_speed: FloatArray
    global_jet_activity: FloatArray
    jet_activity: FloatArray
    reconnection_activity: FloatArray
    onset_index: int | None
    onset_time_normalized: float | None


def sheet_region_mask(
    grid: SpectralGrid,
    mhd_config: MHDConfig,
    jet_config: JetConfig,
    *,
    sheet_centers: tuple[float, ...] | None = None,
    x_points: tuple[float, ...] | None = None,
    outflow_half_window: float | None = None,
    sheet_normal: str = "y",
    outflow_direction: str = "x",
    periodic_outflow: bool = True,
) -> BoolArray:
    """Return geometry-configured sheet/X-point diagnostic windows."""

    centers = sheet_centers or (
        -mhd_config.sheet_center_fraction * mhd_config.ly,
        mhd_config.sheet_center_fraction * mhd_config.ly,
    )
    half_width = jet_config.sheet_half_width_factor * mhd_config.sheet_half_width
    normal_mesh = grid.y_mesh if sheet_normal == "y" else grid.x_mesh
    outflow_mesh = grid.x_mesh if outflow_direction == "x" else grid.y_mesh
    normal_distance = np.min(
        np.asarray([np.abs(normal_mesh - center) for center in centers]),
        axis=0,
    )
    points = x_points or (-np.pi, np.pi)
    x_half_width = (
        outflow_half_window
        if outflow_half_window is not None
        else jet_config.xpoint_half_window_fraction * mhd_config.lx
    )
    outflow_extent = mhd_config.lx if outflow_direction == "x" else mhd_config.ly
    if periodic_outflow:
        point_distances = [
            np.abs((outflow_mesh - point + 0.5 * outflow_extent) % outflow_extent
                   - 0.5 * outflow_extent)
            for point in points
        ]
    else:
        point_distances = [np.abs(outflow_mesh - point) for point in points]
    if not point_distances:
        raise ValueError("At least one diagnostic X-point coordinate is required.")
    outflow_distance = np.min(np.asarray(point_distances), axis=0)
    return np.asarray(
        (normal_distance <= half_width) & (outflow_distance <= x_half_width),
        dtype=bool,
    )


def normalize_activity(values: FloatArray) -> FloatArray:
    """Normalize relative to the initial value without relaxing thresholds."""

    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span <= 1.0e-15:
        return np.zeros_like(values)
    return np.clip((values - values[0]) / span, 0.0, 1.0)


def reconnection_flux_rate(
    mhd_result: MHDResult | MHDFieldSeries,
) -> FloatArray:
    """Return the standard topology-based reconnection-rate magnitude.

    Reduced-MHD runs use ``|d(psi_O-psi_X)/dt|``.  Older or imported data
    without a usable flux-difference series retain the historical proxy as a
    compatibility fallback, but formal event runs always provide the topology
    diagnostic.
    """

    times = np.asarray(mhd_result.times, dtype=float)
    flux = np.asarray(mhd_result.flux_difference, dtype=float)
    if (
        flux.shape == times.shape
        and len(times) >= 3
        and np.all(np.isfinite(flux))
        and np.ptp(times) > 0.0
    ):
        return np.abs(np.gradient(flux, times, edge_order=2))
    return np.abs(np.asarray(mhd_result.reconnection_proxy, dtype=float))


def find_sustained_onset(
    activity: FloatArray,
    threshold: float,
    consecutive: int,
) -> int | None:
    """Return the first index beginning a sustained threshold crossing."""

    above = np.asarray(activity, dtype=float) >= threshold
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1.")
    if above.size < consecutive:
        return None
    run = np.convolve(
        above.astype(np.int8),
        np.ones(consecutive, dtype=np.int8),
        mode="valid",
    )
    matches = np.flatnonzero(run == consecutive)
    return int(matches[0]) if matches.size else None


def _signed_quantile(values: FloatArray, positive: bool, quantile: float) -> float:
    selected = values[values > 0.0] if positive else -values[values < 0.0]
    if selected.size == 0:
        return 0.0
    return float(np.quantile(selected, quantile))


def diagnose_jet(
    mhd_result: MHDResult | MHDFieldSeries,
    mhd_config: MHDConfig,
    jet_config: JetConfig,
) -> JetResult:
    """Measure robust, sheet-localized opposite-sign x-directed outflows."""

    geometry = getattr(mhd_result, "geometry", None)
    mask = sheet_region_mask(
        mhd_result.grid,
        mhd_config,
        jet_config,
        sheet_centers=(
            None if geometry is None else geometry.sheet_centers_y
        ),
        x_points=None if geometry is None else geometry.x_points,
        outflow_half_window=(
            None if geometry is None else geometry.outflow_half_window
        ),
        sheet_normal="y" if geometry is None else geometry.sheet_normal,
        outflow_direction="x" if geometry is None else geometry.outflow_direction,
        periodic_outflow=(
            geometry is None or geometry.kind == "double_harris_periodic"
        ),
    )
    positive: list[float] = []
    negative: list[float] = []
    absolute: list[float] = []
    global_speed: list[float] = []
    for index in range(len(mhd_result.times)):
        _, _, velocity_x, velocity_y, _, _ = mhd_result.snapshot_fields(index)
        local_direction = (
            "x" if geometry is None else geometry.outflow_direction
        )
        local_velocity = velocity_x if local_direction == "x" else velocity_y
        sheet_velocity = local_velocity[mask]
        positive.append(
            _signed_quantile(
                sheet_velocity,
                positive=True,
                quantile=jet_config.velocity_quantile,
            )
        )
        negative.append(
            _signed_quantile(
                sheet_velocity,
                positive=False,
                quantile=jet_config.velocity_quantile,
            )
        )
        absolute.append(
            float(np.quantile(np.abs(sheet_velocity), jet_config.velocity_quantile))
        )
        if geometry is not None and geometry.kind == "open_solar_jet":
            vertical_velocity = (
                velocity_y
                if geometry.vertical_direction == "y"
                else velocity_x
            )
            vertical_mesh = (
                mhd_result.grid.y_mesh
                if geometry.vertical_direction == "y"
                else mhd_result.grid.x_mesh
            )
            transverse_mesh = (
                mhd_result.grid.x_mesh
                if geometry.vertical_direction == "y"
                else mhd_result.grid.y_mesh
            )
            launch = min(geometry.sheet_centers_y)
            plume_mask = (
                (vertical_mesh >= launch)
                & (np.abs(transverse_mesh - geometry.x_points[0])
                   <= 2.0 * geometry.outflow_half_window)
            )
            upward = vertical_velocity[plume_mask]
            upward = upward[upward > 0.0]
            global_speed.append(
                0.0
                if upward.size == 0
                else float(np.quantile(upward, jet_config.velocity_quantile))
            )
        else:
            global_speed.append(absolute[-1])

    positive_array = np.asarray(positive, dtype=float)
    negative_array = np.asarray(negative, dtype=float)
    absolute_array = np.asarray(absolute, dtype=float)
    global_speed_array = np.asarray(global_speed, dtype=float)
    bidirectional = np.minimum(positive_array, negative_array)
    local_activity = normalize_activity(bidirectional)
    global_activity = normalize_activity(global_speed_array)
    jet_activity = (
        np.minimum(local_activity, global_activity)
        if geometry is not None and geometry.kind == "open_solar_jet"
        else local_activity
    )
    reconnection_activity = normalize_activity(reconnection_flux_rate(mhd_result))
    onset_index = find_sustained_onset(
        jet_activity,
        jet_config.jet_threshold,
        jet_config.consecutive_snapshots,
    )
    onset_time = None if onset_index is None else float(mhd_result.times[onset_index])
    return JetResult(
        sheet_mask=mask,
        positive_speed=positive_array,
        negative_speed=negative_array,
        bidirectional_speed=bidirectional,
        absolute_speed=absolute_array,
        global_jet_speed=global_speed_array,
        global_jet_activity=global_activity,
        jet_activity=jet_activity,
        reconnection_activity=reconnection_activity,
        onset_index=onset_index,
        onset_time_normalized=onset_time,
    )


def map_active_interval_to_radio_time(
    mhd_times: FloatArray,
    values: FloatArray,
    onset_time_normalized: float | None,
    radio_times_s: FloatArray,
    onset_start_s: float,
    onset_end_s: float,
) -> FloatArray:
    """Compress the post-onset MHD interval into the radio onset window."""

    radio_times_s = np.asarray(radio_times_s, dtype=float)
    mapped = np.zeros_like(radio_times_s)
    if onset_time_normalized is None:
        return mapped
    tau_end = float(mhd_times[-1])
    if tau_end <= onset_time_normalized:
        return mapped
    inside = (radio_times_s >= onset_start_s) & (radio_times_s <= onset_end_s)
    tau = onset_time_normalized + (
        (radio_times_s[inside] - onset_start_s)
        / (onset_end_s - onset_start_s)
        * (tau_end - onset_time_normalized)
    )
    mapped[inside] = np.interp(tau, mhd_times, values)
    return mapped
