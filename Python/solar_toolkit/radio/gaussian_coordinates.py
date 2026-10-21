"""WCS calibration of native-pixel Gaussian fits, without refitting the data."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from solar_toolkit.map.coordinates import hpc_pixel_coordinates, hpc_world_to_pixel

__all__ = ["attach_wcs_geometry", "wcs_fwhm_curve"]


def attach_wcs_geometry(result, header, *, raw_peak_pixel=None):
    """Attach exact HPC centers and local angular widths to a pixel fit.

    Coordinates are in arcseconds and zero-based pixel centers. Widths use the
    WCS Jacobian at the fitted center; they describe a local ellipse, not a
    globally Gaussian sky model. The fitted pixel model and input header are
    unchanged. The result's coordinate annotations are updated in place.
    """
    result.coordinate_header = deepcopy(header)
    x0, y0 = result.center_pixel
    result.center_arcsec = tuple(
        float(value) for value in hpc_pixel_coordinates(header, x0, y0)
    )
    initial = getattr(result, "initial_center_pixel_tuple", None)
    if initial is not None:
        result.initial_center_arcsec = tuple(
            float(value) for value in hpc_pixel_coordinates(header, *initial)
        )
    if raw_peak_pixel is not None:
        result.raw_center_pixel = tuple(raw_peak_pixel)
        result.raw_center_arcsec = tuple(
            float(value) for value in hpc_pixel_coordinates(header, *raw_peak_pixel)
        )
    # A symmetric, small pixel step avoids assuming that projected axes are
    # aligned or that their scale is constant across a large field of view.
    step = 1e-3
    wx, wy = hpc_pixel_coordinates(
        header, [x0 + step, x0 - step, x0, x0], [y0, y0, y0 + step, y0 - step]
    )
    jacobian = np.array(
        [[wx[0] - wx[1], wx[2] - wx[3]], [wy[0] - wy[1], wy[2] - wy[3]]]
    ) / (2 * step)
    theta = result.theta_rad
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    covariance = rotation @ np.diag(np.square(result.sigma_pixel)) @ rotation.T
    angular_covariance = jacobian @ covariance @ jacobian.T
    if not np.isfinite(angular_covariance).all():
        raise ValueError("Gaussian center has no finite WCS angular covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(angular_covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError("Gaussian WCS transform must have nonzero angular scales")
    order = np.argsort(eigenvalues)[::-1]
    result.wcs_fwhm_arcsec = tuple(
        2 * np.sqrt(2 * np.log(2)) * np.sqrt(eigenvalues[order])
    )
    result.fwhm_width_arcsec, result.fwhm_height_arcsec = result.wcs_fwhm_arcsec
    result.fwhm_major_arcsec, result.fwhm_minor_arcsec = result.wcs_fwhm_arcsec
    major = eigenvectors[:, order[0]]
    result.theta_world_rad = float(np.arctan2(major[1], major[0]))
    result.width_coordinate_model = "local_wcs_jacobian"
    back_x, back_y = hpc_world_to_pixel(header, *result.center_arcsec)
    result.coordinate_roundtrip_error_pixel = float(np.hypot(back_x - x0, back_y - y0))
    return result


def wcs_fwhm_curve(result, *, samples=181):
    """Map the native half-maximum ellipse to HPC, including WCS distortion."""
    angle = np.linspace(0, 2 * np.pi, samples)
    sx, sy = result.sigma_pixel
    radius = np.sqrt(2 * np.log(2))
    x, y = radius * sx * np.cos(angle), radius * sy * np.sin(angle)
    c, s = np.cos(result.theta_rad), np.sin(result.theta_rad)
    return hpc_pixel_coordinates(
        result.coordinate_header,
        result.center_pixel[0] + c * x - s * y,
        result.center_pixel[1] + s * x + c * y,
    )
