"""Newkirk density-model extrapolation for radio Gaussian and drift diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .io import _sample_float, truthy

__all__ = [
    "attach_newkirk_height_to_gaussian",
    "effective_density_factor",
    "extrapolate_drift_line_with_newkirk",
    "newkirk_assumption_label",
    "newkirk_density_cm3",
    "newkirk_height_from_frequency_mhz",
    "newkirk_radius_from_density",
    "newkirk_radius_from_frequency_mhz",
    "newkirk_speed_from_drift_rate",
    "plasma_density_from_frequency_mhz",
]

RSUN_KM = 695_700.0
SPEED_OF_LIGHT_KM_S = 299_792.458
NEWKIRK_BASE_DENSITY_CM3 = 4.2e4
NEWKIRK_EXPONENT = 4.32
PLASMA_FREQ_COEFF_KHZ = 8.98


def plasma_density_from_frequency_mhz(freq_mhz, harmonic=1):
    """Return plasma density in cm^-3 for observed MHz frequencies.

    Invalid frequency samples produce NaN, preserving the input shape. The
    emission harmonic is a finite positive scalar configuration value.
    """
    freq = np.asarray(freq_mhz, dtype=np.float64)
    harmonic = _positive_parameter(harmonic, "harmonic")
    with np.errstate(over="ignore", invalid="ignore"):
        density = (1000.0 * freq / (PLASMA_FREQ_COEFF_KHZ * harmonic)) ** 2
    density = np.where(
        (freq > 0) & np.isfinite(freq) & np.isfinite(density), density, np.nan
    )
    return _scalar_or_array(density)


def newkirk_density_cm3(r_rsun, multiplier=1.0):
    """Return Newkirk density in cm^-3 at heliocentric radii in solar radii.

    Nonpositive/nonfinite radii and numerical overflow yield NaN. The density
    multiplier must be finite and positive. Inputs are never modified.
    """
    r = np.asarray(r_rsun, dtype=np.float64)
    multiplier = _positive_parameter(multiplier, "multiplier")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        density = multiplier * NEWKIRK_BASE_DENSITY_CM3 * 10.0 ** (NEWKIRK_EXPONENT / r)
    density = np.where((r > 0) & np.isfinite(r) & np.isfinite(density), density, np.nan)
    return _scalar_or_array(density)


def newkirk_radius_from_density(ne_cm3, multiplier=1.0):
    """Return radius in solar radii, or NaN outside the model's invertible domain."""
    ne = np.asarray(ne_cm3, dtype=np.float64)
    multiplier = _positive_parameter(multiplier, "multiplier")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.log10(ne / (multiplier * NEWKIRK_BASE_DENSITY_CM3))
        radius = NEWKIRK_EXPONENT / denom
    radius = np.where(
        (ne > multiplier * NEWKIRK_BASE_DENSITY_CM3)
        & np.isfinite(ne)
        & np.isfinite(radius),
        radius,
        np.nan,
    )
    return _scalar_or_array(radius)


def newkirk_radius_from_frequency_mhz(freq_mhz, multiplier=1.0, harmonic=1):
    """Infer heliocentric solar radii; invalid MHz samples remain NaN."""
    density = plasma_density_from_frequency_mhz(freq_mhz, harmonic=harmonic)
    return newkirk_radius_from_density(density, multiplier=multiplier)


def newkirk_height_from_frequency_mhz(freq_mhz, multiplier=1.0, harmonic=1):
    """Infer height above the photosphere in solar radii (radius minus one)."""
    radius = np.asarray(
        newkirk_radius_from_frequency_mhz(
            freq_mhz, multiplier=multiplier, harmonic=harmonic
        ),
        dtype=np.float64,
    )
    return _scalar_or_array(radius - 1.0)


def newkirk_speed_from_drift_rate(freq_mhz, drift_rate_mhz_s, r_rsun):
    """Return radial drift and km/s, with an invalid-sample reason when needed."""
    freq = _sample_float(freq_mhz)
    drift = _sample_float(drift_rate_mhz_s)
    radius = _sample_float(r_rsun)
    if freq <= 0 or radius <= 0 or not np.isfinite([freq, drift, radius]).all():
        dr_dt = np.nan
    else:
        dr_dt = -(drift / freq) * radius**2 / (2.16 * math.log(10.0))
    return {
        "frequency_mhz": freq,
        "drift_rate_mhz_s": drift,
        "r_rsun": radius,
        "dr_dt_rsun_s": float(dr_dt) if np.isfinite(dr_dt) else np.nan,
        "speed_km_s": float(dr_dt * RSUN_KM) if np.isfinite(dr_dt) else np.nan,
        "quality_flag": "ok" if np.isfinite(dr_dt) else "invalid_physical_input",
        "reason": (
            ""
            if np.isfinite(dr_dt)
            else "frequency_and_radius_must_be_positive_and_all_values_finite"
        ),
    }


def effective_density_factor(multiplier, harmonic):
    """Return the Newkirk density/emission degeneracy factor N*s^2."""
    return (
        _positive_parameter(multiplier, "multiplier")
        * _positive_parameter(harmonic, "harmonic") ** 2
    )


def newkirk_assumption_label(multiplier, harmonic):
    """Return a compact physical label for a density and emission assumption."""
    factor = effective_density_factor(multiplier, harmonic)
    return (
        f"{float(multiplier):g}x Newkirk, H={float(harmonic):g}, " f"N*s^2={factor:g}"
    )


def attach_newkirk_height_to_gaussian(
    df,
    multiplier,
    harmonic,
):
    """Attach model columns to explicitly valid Gaussian diagnostics.

    Quality-rejected fits are excluded. Accepted fits with invalid frequencies
    remain as rows with NaN model quantities and ``newkirk_invalid_reason``.
    The input table is copied; density/emission assumptions must be finite and
    positive even when the table contains no valid rows.
    """
    multiplier = _positive_parameter(multiplier, "multiplier")
    harmonic = _positive_parameter(harmonic, "harmonic")
    data = _valid_gaussian_rows(pd.DataFrame(df).copy())
    if data.empty:
        return data

    freq = pd.to_numeric(data["freq"], errors="coerce")
    r_model = np.asarray(
        newkirk_radius_from_frequency_mhz(
            freq.to_numpy(dtype=float), multiplier=multiplier, harmonic=harmonic
        ),
        dtype=float,
    )
    data["newkirk_multiplier"] = float(multiplier)
    data["newkirk_harmonic"] = (
        int(harmonic) if float(harmonic).is_integer() else float(harmonic)
    )
    data["density_multiplier"] = float(multiplier)
    data["emission_harmonic"] = float(harmonic)
    data["effective_density_factor"] = effective_density_factor(multiplier, harmonic)
    data["newkirk_assumption_label"] = newkirk_assumption_label(multiplier, harmonic)
    data["newkirk_density_cm3"] = plasma_density_from_frequency_mhz(
        freq.to_numpy(dtype=float), harmonic=harmonic
    )
    data["newkirk_r_rsun"] = r_model
    data["newkirk_height_rsun"] = r_model - 1.0
    invalid_frequency = ~np.isfinite(freq) | (freq <= 0)
    data["newkirk_quality_flag"] = np.where(
        np.isfinite(r_model), "ok", "invalid_model_input"
    )
    data["newkirk_invalid_reason"] = np.where(
        invalid_frequency,
        "invalid_frequency",
        np.where(np.isfinite(r_model), "", "outside_model_domain"),
    )
    return data.reset_index(drop=True)


def extrapolate_drift_line_with_newkirk(drift_row, multiplier, harmonic):
    """Return midpoint model diagnostics without mutating the input drift row.

    Invalid observations retain their row and receive NaN derived values plus
    a model-quality reason. Missing duration is accepted for legacy tables;
    a supplied duration must be finite and nonzero.
    """
    multiplier = _positive_parameter(multiplier, "multiplier")
    harmonic = _positive_parameter(harmonic, "harmonic")
    row = dict(drift_row)
    f_start = _sample_float(row.get("f_start_mhz", np.nan))
    f_end = _sample_float(row.get("f_end_mhz", np.nan))
    drift = _sample_float(row.get("drift_rate_mhz_s", np.nan))
    duration = _sample_float(row.get("duration_s"))
    reason = ""
    if not np.isfinite([f_start, f_end]).all() or min(f_start, f_end) <= 0:
        reason = "invalid_frequency"
    elif not np.isfinite(drift):
        reason = "invalid_drift_rate"
    elif "duration_s" in row and (not np.isfinite(duration) or duration == 0):
        reason = "invalid_duration"
    elif "quality_flag" in row and str(row["quality_flag"]).strip().lower() != "ok":
        reason = "invalid_input_quality"
    mid_freq = float((f_start + f_end) / 2.0) if not reason else np.nan
    radius = float(
        newkirk_radius_from_frequency_mhz(
            mid_freq, multiplier=multiplier, harmonic=harmonic
        )
    )
    speed = newkirk_speed_from_drift_rate(mid_freq, drift, radius)
    if not reason and not np.isfinite(radius):
        reason = "outside_model_domain"
    speed_km_s = speed["speed_km_s"]
    speed_c = speed_km_s / SPEED_OF_LIGHT_KM_S if np.isfinite(speed_km_s) else np.nan
    out = dict(row)
    out.update(
        {
            "newkirk_multiplier": float(multiplier),
            "newkirk_harmonic": (
                int(harmonic) if float(harmonic).is_integer() else float(harmonic)
            ),
            "density_multiplier": float(multiplier),
            "emission_harmonic": float(harmonic),
            "effective_density_factor": effective_density_factor(multiplier, harmonic),
            "newkirk_assumption_label": newkirk_assumption_label(multiplier, harmonic),
            "mid_frequency_mhz": mid_freq,
            "newkirk_r_rsun": radius,
            "newkirk_height_rsun": radius - 1.0 if np.isfinite(radius) else np.nan,
            "dr_dt_rsun_s": speed["dr_dt_rsun_s"],
            "speed_km_s": speed_km_s,
            "v_over_c": speed_c,
            "newkirk_speed_km_s": speed_km_s,
            "newkirk_speed_c": speed_c,
            "newkirk_quality_flag": "invalid_model_input" if reason else "ok",
            "newkirk_invalid_reason": reason,
        }
    )
    return out


def _valid_gaussian_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"quality_flag", "overlay_valid", "trajectory_valid"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Gaussian diagnostics missing required columns: {sorted(missing)}"
        )
    mask = (
        df["quality_flag"].astype(str).str.strip().str.lower().eq("ok")
        & df["overlay_valid"].map(truthy)
        & df["trajectory_valid"].map(truthy)
    )
    return df.loc[mask].copy()


def _truthy(value) -> bool:
    """Compatibility alias to the shared quality-flag parser."""
    return truthy(value)


def _positive_parameter(value, name):
    """Validate scalar model configuration separately from observed samples."""
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive scalar") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return number


def _scalar_or_array(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return value
