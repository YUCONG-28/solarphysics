"""Instrument-aware optically thin AIA forward model utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AIAResponse:
    """Reviewed temperature response sampled on a log10-temperature grid."""

    channel: str
    log10_temperature_k: FloatArray
    response_dn_cm5_s_pixel: FloatArray
    calibration_id: str

    def __post_init__(self) -> None:
        if self.channel not in {"94", "131", "171", "193", "211", "304", "335"}:
            raise ValueError("Unsupported AIA channel.")
        if self.log10_temperature_k.shape != self.response_dn_cm5_s_pixel.shape:
            raise ValueError("Temperature and response arrays must have equal shape.")
        if len(self.log10_temperature_k) < 2:
            raise ValueError("AIA response requires at least two samples.")
        if not np.all(np.diff(self.log10_temperature_k) > 0.0):
            raise ValueError("AIA response temperatures must strictly increase.")
        if np.any(self.response_dn_cm5_s_pixel < 0.0):
            raise ValueError("AIA response must be non-negative.")
        if not self.calibration_id:
            raise ValueError("A calibration identifier is required.")


def synthesize_aia_intensity(
    electron_density_cm3: FloatArray,
    temperature_k: FloatArray,
    *,
    los_depth_cm: float,
    response: AIAResponse,
    exposure_s: float = 1.0,
    psf_sigma_pixels: float = 0.0,
) -> FloatArray:
    """Evaluate ``ne**2 * D_LOS * R(T)`` and optionally apply a Gaussian PSF.

    The supplied response must already correspond to the event date and desired
    calibration.  This function deliberately does not invent a response curve.
    """

    density = np.asarray(electron_density_cm3, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    if density.shape != temperature.shape:
        raise ValueError("Density and temperature must have equal shape.")
    if los_depth_cm <= 0.0 or exposure_s <= 0.0:
        raise ValueError("LOS depth and exposure must be positive.")
    if psf_sigma_pixels < 0.0:
        raise ValueError("PSF width cannot be negative.")
    if np.any(density < 0.0) or np.any(temperature <= 0.0):
        raise ValueError("Density must be non-negative and temperature positive.")
    response_values = np.interp(
        np.log10(temperature),
        response.log10_temperature_k,
        response.response_dn_cm5_s_pixel,
        left=0.0,
        right=0.0,
    )
    intensity = density**2 * los_depth_cm * response_values * exposure_s
    if psf_sigma_pixels > 0.0:
        intensity = gaussian_filter(intensity, psf_sigma_pixels, mode="nearest")
    return np.asarray(intensity, dtype=float)
