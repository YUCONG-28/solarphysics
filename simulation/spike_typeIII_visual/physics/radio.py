"""Kinematic electron-beam and phenomenological radio-emission proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import JetConfig, RadioConfig, TimeCalibrationConfig
from .fields import MHDFieldSeries
from .jet import JetResult, map_active_interval_to_radio_time, normalize_activity
from .rmhd import MHDResult

FloatArray = NDArray[np.float64]

SPEED_OF_LIGHT_M_S = 299_792_458.0
VACUUM_PERMEABILITY_H_M = 4.0e-7 * np.pi
PROTON_MASS_KG = 1.67262192595e-27


@dataclass(frozen=True)
class RadioResult:
    """Synthetic dynamic spectrum and its deterministic ingredients."""

    times_s: FloatArray
    frequencies_mhz: FloatArray
    intensity: FloatArray
    ridge_frequency_mhz: FloatArray
    beam_height_mm: FloatArray
    injection_activity: FloatArray
    jet_activity: FloatArray
    conditioned_reconnection_activity: FloatArray
    spike_catalog: FloatArray
    base_density_cm3: float
    beam_gamma: float
    event_status: str
    jet_coincidence_fraction: float | None
    jet_spike_lag_s: FloatArray
    topping_margin_mhz: FloatArray
    jet_onset_radio_s: float | None


def plasma_frequency_hz(electron_density_cm3: FloatArray | float) -> FloatArray:
    """Electron plasma frequency for density in cm^-3."""

    return 8_980.0 * np.sqrt(np.asarray(electron_density_cm3, dtype=float))


def electron_beam_kinematics(
    times_s: FloatArray,
    speed_fraction_c: float,
) -> tuple[FloatArray, float]:
    """Return beam height in Mm and Lorentz factor for constant speed."""

    if not 0.0 < speed_fraction_c < 1.0:
        raise ValueError("speed_fraction_c must be in (0, 1).")
    speed_m_s = speed_fraction_c * SPEED_OF_LIGHT_M_S
    height_mm = speed_m_s * np.asarray(times_s, dtype=float) / 1.0e6
    gamma = 1.0 / np.sqrt(1.0 - speed_fraction_c**2)
    return height_mm, float(gamma)


def exponential_coronal_density_cm3(
    height_mm: FloatArray,
    base_density_cm3: float,
    scale_height_mm: float,
) -> FloatArray:
    """Evaluate an exponential coronal electron-density proxy."""

    if base_density_cm3 <= 0.0:
        raise ValueError("base_density_cm3 must be positive.")
    if scale_height_mm <= 0.0:
        raise ValueError("scale_height_mm must be positive.")
    return base_density_cm3 * np.exp(
        -np.asarray(height_mm, dtype=float) / scale_height_mm
    )


def typeiii_ridge_frequency_mhz(
    height_mm: FloatArray,
    base_density_cm3: float,
    scale_height_mm: float,
) -> FloatArray:
    """Map beam height to a fundamental plasma-frequency Type III ridge."""

    density_cm3 = exponential_coronal_density_cm3(
        height_mm,
        base_density_cm3,
        scale_height_mm,
    )
    return plasma_frequency_hz(density_cm3) / 1.0e6


def gaussian_spike_pulse(
    frequency_mesh_mhz: FloatArray,
    time_mesh_s: FloatArray,
    center_frequency_mhz: float,
    center_time_s: float,
    sigma_frequency_mhz: float,
    sigma_time_s: float,
    amplitude: float,
) -> FloatArray:
    """Return one phenomenological narrowband time-frequency spike."""

    return amplitude * np.exp(
        -0.5 * ((time_mesh_s - center_time_s) / sigma_time_s) ** 2
        - 0.5 * ((frequency_mesh_mhz - center_frequency_mhz) / sigma_frequency_mhz) ** 2
    )


def _normalized_activity(
    result: MHDResult | MHDFieldSeries,
    target_times: FloatArray,
) -> FloatArray:
    normalized = normalize_activity(result.reconnection_proxy)
    scaled_time = target_times / target_times[-1] * result.times[-1]
    return np.interp(scaled_time, result.times, normalized)


def alfven_time_seconds(config: TimeCalibrationConfig) -> float:
    """Return L0/vA using explicit SI-converted magnetic and density scales."""

    if config.mode != "alfven":
        raise ValueError("Alfvén time is defined only for mode='alfven'.")
    assert config.length_scale_mm is not None
    assert config.magnetic_field_gauss is not None
    assert config.electron_density_cm3 is not None
    length_m = config.length_scale_mm * 1.0e6
    magnetic_t = config.magnetic_field_gauss * 1.0e-4
    density_m3 = config.electron_density_cm3 * 1.0e6
    mass_density = (
        config.mean_mass_per_electron * PROTON_MASS_KG * density_m3
    )
    alfven_speed = magnetic_t / np.sqrt(VACUUM_PERMEABILITY_H_M * mass_density)
    return float(length_m / alfven_speed)


def _conditioned_activity(
    mhd_times: FloatArray,
    values: FloatArray,
    onset_time_normalized: float | None,
    radio_times_s: FloatArray,
    onset_start_s: float,
    onset_end_s: float,
    calibration: TimeCalibrationConfig,
) -> tuple[FloatArray, float | None]:
    if onset_time_normalized is None:
        return np.zeros_like(radio_times_s), None
    if calibration.mode in {"proxy", "event"}:
        mapped = map_active_interval_to_radio_time(
            mhd_times,
            values,
            onset_time_normalized,
            radio_times_s,
            onset_start_s,
            onset_end_s,
        )
        return mapped, onset_start_s
    scale_s = alfven_time_seconds(calibration)
    physical_times = np.asarray(mhd_times, dtype=float) * scale_s
    mapped = np.interp(
        radio_times_s,
        physical_times,
        values,
        left=0.0,
        right=0.0,
    )
    inside = (radio_times_s >= onset_start_s) & (radio_times_s <= onset_end_s)
    return np.where(inside, mapped, 0.0), onset_time_normalized * scale_s


def sample_jet_conditioned_times(
    radio_times_s: FloatArray,
    jet_activity: FloatArray,
    reconnection_activity: FloatArray,
    spike_count: int,
    rng: np.random.Generator,
    jet_config: JetConfig,
) -> FloatArray:
    """Sample unique, activity-weighted times from valid onset intervals."""

    if spike_count <= 0:
        return np.empty(0, dtype=float)
    dense_times = np.linspace(
        float(radio_times_s[0]),
        float(radio_times_s[-1]),
        max(8192, 64 * spike_count),
    )
    dense_jet = np.interp(dense_times, radio_times_s, jet_activity)
    dense_reconnection = np.interp(
        dense_times,
        radio_times_s,
        reconnection_activity,
    )
    valid = (dense_jet >= jet_config.jet_threshold) & (
        dense_reconnection >= jet_config.reconnection_threshold
    )
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return np.empty(0, dtype=float)
    weights = dense_jet[candidates] * dense_reconnection[candidates]
    weights /= float(np.sum(weights))
    count = min(spike_count, int(candidates.size))
    selected = rng.choice(candidates, size=count, replace=False, p=weights)
    return np.sort(dense_times[selected])


def synthesize_radio_proxy(
    mhd_result: MHDResult | MHDFieldSeries,
    config: RadioConfig,
    seed: int,
    *,
    jet_result: JetResult | None = None,
    jet_config: JetConfig | None = None,
    spike_coupling: str = "uniform",
    time_calibration: TimeCalibrationConfig | None = None,
) -> RadioResult:
    """Create a deterministic Type III ridge plus onset spike components.

    The ridge follows an exponential coronal density model.  Spike components
    are imposed Gaussian time-frequency pulses and are not predicted by the
    reduced-MHD equations.
    """

    if spike_coupling not in {"jet", "uniform"}:
        raise ValueError("spike_coupling must be 'jet' or 'uniform'.")
    if spike_coupling == "jet" and (jet_result is None or jet_config is None):
        raise ValueError("jet coupling requires jet_result and jet_config.")
    calibration = time_calibration or TimeCalibrationConfig()

    # Keep the phenomenological event catalogue invariant when the background
    # time-frequency grid changes.  The spawned streams are part of the
    # reproducibility contract and are recorded as ``seed-sequence-v1``.
    background_seed, spike_seed = np.random.SeedSequence(seed).spawn(2)
    background_rng = np.random.default_rng(background_seed)
    spike_rng = np.random.default_rng(spike_seed)
    times = np.linspace(0.0, config.duration_s, config.time_samples)
    frequencies = np.linspace(
        config.min_frequency_mhz,
        config.max_frequency_mhz,
        config.frequency_samples,
    )
    height_mm, gamma = electron_beam_kinematics(
        times,
        config.beam_speed_fraction_c,
    )
    base_density = (config.start_frequency_mhz * 1.0e6 / 8_980.0) ** 2
    ridge_frequency = typeiii_ridge_frequency_mhz(
        height_mm,
        base_density,
        config.density_scale_height_mm,
    )
    activity = _normalized_activity(mhd_result, times)
    onset_max = min(
        config.spike_onset_cap_s,
        config.duration_s * config.spike_onset_fraction,
    )
    jet_activity = np.zeros_like(times)
    conditioned_reconnection = np.zeros_like(times)
    jet_onset_radio_s: float | None = None
    if jet_result is not None:
        jet_activity, jet_onset_radio_s = _conditioned_activity(
            mhd_result.times,
            jet_result.jet_activity,
            jet_result.onset_time_normalized,
            times,
            config.spike_onset_start_s,
            onset_max,
            calibration,
        )
        conditioned_reconnection, _ = _conditioned_activity(
            mhd_result.times,
            jet_result.reconnection_activity,
            jet_result.onset_time_normalized,
            times,
            config.spike_onset_start_s,
            onset_max,
            calibration,
        )

    frequency_mesh, time_mesh = np.meshgrid(frequencies, times, indexing="ij")
    ridge_width_mhz = 5.0 + 2.0 * activity
    ridge_amplitude = 0.35 + 0.65 * activity
    ridge = ridge_amplitude[None, :] * np.exp(
        -0.5
        * ((frequency_mesh - ridge_frequency[None, :]) / ridge_width_mhz[None, :]) ** 2
    )
    background = 0.035 + 0.025 * background_rng.standard_normal(ridge.shape)
    intensity = np.clip(background + ridge, 0.0, None)

    if spike_coupling == "jet":
        assert jet_config is not None
        center_times = sample_jet_conditioned_times(
            times,
            jet_activity,
            conditioned_reconnection,
            config.spike_count,
            spike_rng,
            jet_config,
        )
    else:
        center_times = np.sort(
            spike_rng.uniform(
                config.spike_onset_start_s,
                onset_max,
                size=config.spike_count,
            )
        )

    spike_rows: list[list[float]] = []
    for center_time_value in center_times:
        center_time = float(center_time_value)
        center_height_mm = (
            config.beam_speed_fraction_c
            * SPEED_OF_LIGHT_M_S
            * center_time
            / 1.0e6
        )
        ridge_at_center = float(
            typeiii_ridge_frequency_mhz(
                np.asarray([center_height_mm]),
                float(base_density),
                config.density_scale_height_mm,
            )[0]
        )
        available_offset_max = min(
            config.spike_frequency_offset_max_mhz,
            config.max_frequency_mhz - ridge_at_center,
        )
        if available_offset_max <= config.spike_frequency_offset_min_mhz:
            raise ValueError(
                "Frequency band leaves no room for a strictly positive "
                "Spike-Topping offset at the selected onset time."
            )
        frequency_offset = float(
            spike_rng.uniform(
                config.spike_frequency_offset_min_mhz,
                available_offset_max,
            )
        )
        center_frequency = ridge_at_center + frequency_offset
        sigma_time = float(spike_rng.uniform(0.018, 0.055))
        sigma_frequency = float(spike_rng.uniform(2.0, 7.0))
        amplitude = float(spike_rng.uniform(0.45, 0.95))
        pulse = gaussian_spike_pulse(
            frequency_mesh,
            time_mesh,
            center_frequency,
            center_time,
            sigma_frequency,
            sigma_time,
            amplitude,
        )
        intensity += pulse
        spike_rows.append(
            [
                center_time,
                center_frequency,
                sigma_time,
                sigma_frequency,
                amplitude,
            ]
        )

    intensity -= float(np.min(intensity))
    intensity /= max(float(np.max(intensity)), 1e-15)
    spike_catalog = np.asarray(spike_rows, dtype=float).reshape((-1, 5))
    topping_margin = (
        np.empty(0, dtype=float)
        if not spike_rows
        else spike_catalog[:, 1]
        - typeiii_ridge_frequency_mhz(
            (
                config.beam_speed_fraction_c
                * SPEED_OF_LIGHT_M_S
                * spike_catalog[:, 0]
                / 1.0e6
            ),
            float(base_density),
            config.density_scale_height_mm,
        )
    )
    if not spike_rows:
        event_status = "no_event"
        coincidence: float | None = None
        jet_lag = np.empty(0, dtype=float)
    else:
        event_status = "events"
        if jet_config is None:
            coincidence = None
        else:
            spike_jet = np.interp(spike_catalog[:, 0], times, jet_activity)
            spike_reconnection = np.interp(
                spike_catalog[:, 0],
                times,
                conditioned_reconnection,
            )
            coincidence = float(
                np.mean(
                    (spike_jet >= jet_config.jet_threshold)
                    & (spike_reconnection >= jet_config.reconnection_threshold)
                )
            )
        jet_lag = (
            np.full(len(spike_catalog), np.nan, dtype=float)
            if jet_onset_radio_s is None
            else spike_catalog[:, 0] - jet_onset_radio_s
        )
    return RadioResult(
        times_s=times,
        frequencies_mhz=frequencies,
        intensity=intensity,
        ridge_frequency_mhz=ridge_frequency,
        beam_height_mm=height_mm,
        injection_activity=activity,
        jet_activity=jet_activity,
        conditioned_reconnection_activity=conditioned_reconnection,
        spike_catalog=spike_catalog,
        base_density_cm3=float(base_density),
        beam_gamma=float(gamma),
        event_status=event_status,
        jet_coincidence_fraction=coincidence,
        jet_spike_lag_s=jet_lag,
        topping_margin_mhz=topping_margin,
        jet_onset_radio_s=jet_onset_radio_s,
    )
