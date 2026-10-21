"""Native-pixel single-source fits and transparent radio-frame time matching.

This module does not select an event ROI, identify a radio emission class, or
apply an absolute pointing correction. Fit covariance is a formal, independent
pixel least-squares diagnostic; radio-beam correlation, calibration, and source
model uncertainty are not included. All angular results use the supplied WCS.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.optimize import least_squares

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d

__all__ = ["fit_radio_plane", "evaluate_radio_model", "nearest_time_matches"]

_VERSION = "native_radio_gaussian_v1"
_FWHM = 2 * math.sqrt(2 * math.log(2))
_PARAMETERS = [
    "amplitude",
    "center_x_pixel",
    "center_y_pixel",
    "sigma_x_pixel",
    "sigma_y_pixel",
    "theta_rad",
    "background_constant",
    "background_slope_x",
    "background_slope_y",
]


def _empty_result(roi, background):
    saved_roi = []
    try:
        values = list(roi)
    except TypeError:
        values = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        saved_roi.append(number if np.isfinite(number) else None)
    return {
        "algorithm_version": _VERSION,
        "background_model": background,
        "roi_xyxy": saved_roi,
        "pixel_origin": 0,
        "roi_stop_exclusive": True,
        "fit_success": False,
        "quality_status": "failed",
        "quality_reasons": [],
        "center_x_pixel": None,
        "center_y_pixel": None,
        "center_hpc_x_arcsec": None,
        "center_hpc_y_arcsec": None,
        "sigma_x_pixel": None,
        "sigma_y_pixel": None,
        "theta_rad": None,
        "fwhm_major_arcsec": None,
        "fwhm_minor_arcsec": None,
        "fwhm_x_pixel": None,
        "fwhm_y_pixel": None,
        "amplitude": None,
        "residual_rms": None,
        "residual_rms_over_amplitude": None,
        "formal_covariance": None,
        "formal_center_sigma_x_pixel": None,
        "formal_center_sigma_y_pixel": None,
        "formal_covariance_caveat": "Independent-pixel formal fit covariance only; "
        "not full localization uncertainty or a co-location significance.",
        "finite_fraction": 0.0,
        "n_finite_pixels": 0,
        "nfev": 0,
        "parameters": None,
        "significant_peak_count": 0,
        "model_adequacy": "not_assessed",
        "wcs_roundtrip_error_pixel": None,
    }


def _model(x, y, parameters, origin, background):
    values = elliptical_gaussian_2d((x, y), *parameters[:6]) + parameters[6]
    if background == "plane":
        values = (
            values + parameters[7] * (x - origin[0]) + parameters[8] * (y - origin[1])
        )
    return values


def _model_jacobian(x, y, p, origin, background):
    """Analytic Jacobian avoids repeated model evaluations during batch fits."""
    amplitude, cx, cy, sx, sy, theta = p[:6]
    cosine, sine = np.cos(theta), np.sin(theta)
    xr = cosine * (x - cx) + sine * (y - cy)
    yr = -sine * (x - cx) + cosine * (y - cy)
    exponential = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    source = amplitude * exponential
    columns = [
        exponential,
        source * (xr * cosine / sx**2 - yr * sine / sy**2),
        source * (xr * sine / sx**2 + yr * cosine / sy**2),
        source * xr**2 / sx**3,
        source * yr**2 / sy**3,
        source * xr * yr * (1 / sy**2 - 1 / sx**2),
        np.ones_like(x),
    ]
    if background == "plane":
        columns += [x - origin[0], y - origin[1]]
    return np.column_stack(columns)


def evaluate_radio_model(shape, fit_result):
    """Evaluate a saved successful fit over the full native image grid.

    Returned values do not replace missing input pixels. For a residual image,
    subtract this model from the original array so its NaNs remain missing.
    """
    if not fit_result.get("fit_success") or fit_result.get("parameters") is None:
        raise ValueError("A successful fit with saved parameters is required")
    y, x = np.indices(shape, dtype=float)
    return _model(
        x,
        y,
        np.asarray(fit_result["parameters"], dtype=float),
        fit_result["background_origin_pixel"],
        fit_result["background_model"],
    )


def _hpc_at(wcs, x, y):
    world = wcs.all_pix2world(np.array([[x, y]], dtype=float), 0)[0]
    units = wcs.world_axis_units
    lon = (world[0] * u.Unit(units[0])).to_value(u.deg)
    lat = (world[1] * u.Unit(units[1])).to_value(u.deg)
    return np.array([((lon + 180) % 360 - 180) * 3600, lat * 3600])


def _attach_wcs(result, header, parameters):
    if not str(header.get("CTYPE1", "")).startswith("HPLN") or not str(
        header.get("CTYPE2", "")
    ).startswith("HPLT"):
        raise ValueError("Explicit HPLN/HPLT WCS is required; no RA/Dec guessing")
    required = ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2"]
    if any(key not in header for key in required):
        raise ValueError("Incomplete native WCS metadata")
    if not (
        ("CDELT1" in header and "CDELT2" in header)
        or ("CD1_1" in header and "CD2_2" in header)
    ):
        raise ValueError("Missing native WCS scale")
    wcs = WCS(header, naxis=2)
    x, y = parameters[1:3]
    center = _hpc_at(wcs, x, y)
    if not np.all(np.isfinite(center)):
        raise ValueError("Nonfinite projected center")
    # Use WCS forward/backward values rather than a solar-radius or extent scale.
    world = wcs.all_pix2world([[x, y]], 0)
    back = wcs.all_world2pix(world, 0)[0]
    error = float(np.linalg.norm(back - [x, y]))
    if not np.isfinite(error) or error >= 0.1:
        raise ValueError("WCS roundtrip exceeds 0.1 native pixel")
    step = 0.01
    jacobian = np.column_stack(
        [
            (_hpc_at(wcs, x + step, y) - _hpc_at(wcs, x - step, y)) / (2 * step),
            (_hpc_at(wcs, x, y + step) - _hpc_at(wcs, x, y - step)) / (2 * step),
        ]
    )
    sx, sy, theta = parameters[3:6]
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pixel_shape = rotation @ np.diag([sx**2, sy**2]) @ rotation.T
    angular_shape = jacobian @ pixel_shape @ jacobian.T
    eigenvalues = np.linalg.eigvalsh(angular_shape)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Invalid local WCS scale")
    result.update(
        center_hpc_x_arcsec=float(center[0]),
        center_hpc_y_arcsec=float(center[1]),
        wcs_roundtrip_error_pixel=error,
        fwhm_major_arcsec=float(_FWHM * np.sqrt(eigenvalues[-1])),
        fwhm_minor_arcsec=float(_FWHM * np.sqrt(eigenvalues[0])),
        local_wcs_jacobian_arcsec_per_pixel=jacobian.tolist(),
        width_status="Apparent width using local WCS; no beam deconvolution",
    )


def fit_radio_plane(data, header, roi_xyxy, background="constant", *, max_nfev=250):
    """Fit a rotated Gaussian plus fixed-choice background in native pixels.

    ``roi_xyxy`` is an integer half-open rectangle ``(x0, y0, x1, y1)``.
    Invalid inputs and convergence failures return diagnostic records. Numerical
    success is separate from quality flags, and a multi-peak image never starts
    an implicit multi-component fit. Equal pixel weights are used throughout.
    Background slopes are relative to the saved ROI center, not solar center.
    """
    result = _empty_result(roi_xyxy, background)
    if background not in {"constant", "plane"}:
        result["quality_reasons"].append("unsupported_background")
        return result
    try:
        array = np.asarray(data, dtype=float)
        roi = np.asarray(roi_xyxy, dtype=float)
    except (ValueError, TypeError, OverflowError):
        result["quality_reasons"].append("expected_numeric_image_and_roi")
        return result
    if array.ndim != 2 or roi.shape != (4,):
        result["quality_reasons"].append("expected_2d_array_and_four_roi_bounds")
        return result
    if any(not np.isfinite(v) or int(v) != v for v in roi):
        result["quality_reasons"].append("roi_bounds_must_be_finite_integers")
        return result
    x0, y0, x1, y1 = map(int, roi)
    if not (0 <= x0 < x1 <= array.shape[1] and 0 <= y0 < y1 <= array.shape[0]):
        result["quality_reasons"].append("roi_outside_image_or_empty")
        return result
    work = array[y0:y1, x0:x1]
    finite = np.isfinite(work)
    n = int(finite.sum())
    result.update(n_finite_pixels=n, finite_fraction=float(finite.mean()))
    npar = 7 if background == "constant" else 9
    if n < max(npar * 3, 25):
        result["quality_reasons"].append("insufficient_finite_pixels")
        return result
    yy, xx = np.indices(work.shape, dtype=float)
    xx += x0
    yy += y0
    origin = np.array([(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2])
    result["background_origin_pixel"] = origin.tolist()
    edge = finite & ((xx == x0) | (xx == x1 - 1) | (yy == y0) | (yy == y1 - 1))
    b0 = float(np.median(work[edge])) if edge.any() else float(np.median(work[finite]))
    noise = 1.4826 * float(np.median(np.abs(work[edge] - b0))) if edge.any() else 0.0
    peak = float(np.max(work[finite]))
    scale = peak - b0
    if not np.isfinite(scale) or scale <= max(abs(peak), 1.0) * 1e-12:
        result["quality_reasons"].append("no_positive_source_contrast")
        return result
    weights = np.where(finite, np.maximum(work - b0 - 0.2 * scale, 0), 0)
    total = weights.sum()
    cx, cy = float(np.sum(weights * xx) / total), float(np.sum(weights * yy) / total)
    dx, dy = xx - cx, yy - cy
    moment = (
        np.array(
            [
                [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
                [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
            ]
        )
        / total
    )
    eig, vec = np.linalg.eigh(moment)
    max_sigma = float(max(work.shape))
    sx, sy = np.clip(np.sqrt(np.maximum(eig[::-1], 1)), 0.5, 0.8 * max_sigma)
    theta = math.atan2(vec[1, -1], vec[0, -1])
    # Amplitudes/backgrounds normalized by source contrast for conditioning.
    p0 = [1.0, cx, cy, sx, sy, theta, b0 / scale]
    lower = [0, x0, y0, 0.3, 0.3, -2 * np.pi, -np.inf]
    upper = [np.inf, x1 - 1, y1 - 1, max_sigma, max_sigma, 2 * np.pi, np.inf]
    if background == "plane":
        p0 += [0.0, 0.0]
        lower += [-np.inf, -np.inf]
        upper += [np.inf, np.inf]
    x, y, target = xx[finite], yy[finite], work[finite] / scale
    try:
        fitted = least_squares(
            lambda p: _model(x, y, p, origin, background) - target,
            p0,
            bounds=(lower, upper),
            x_scale="jac",
            jac=lambda p: _model_jacobian(x, y, p, origin, background),
            max_nfev=int(max_nfev),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["quality_reasons"].append("optimization_error")
        result["failure_detail"] = str(exc)
        return result
    result.update(nfev=int(fitted.nfev), optimizer_message=str(fitted.message))
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        result["quality_reasons"].append("optimizer_not_converged")
        return result
    parameters = fitted.x.copy()
    scale_vector = np.ones(npar)
    scale_vector[[0, *range(6, npar)]] = scale
    parameters *= scale_vector
    residual = fitted.fun * scale
    rms = float(np.sqrt(np.mean(residual**2)))
    result.update(
        fit_success=True,
        center_x_pixel=float(parameters[1]),
        center_y_pixel=float(parameters[2]),
        sigma_x_pixel=float(parameters[3]),
        sigma_y_pixel=float(parameters[4]),
        theta_rad=float(parameters[5]),
        fwhm_x_pixel=float(_FWHM * parameters[3]),
        fwhm_y_pixel=float(_FWHM * parameters[4]),
        amplitude=float(parameters[0]),
        parameters=parameters.tolist(),
        parameter_names=_PARAMETERS[:npar],
        residual_rms=rms,
        residual_rms_over_amplitude=rms / parameters[0],
        residual_max_abs_over_amplitude=float(np.max(np.abs(residual)) / parameters[0]),
        border_noise_mad=float(noise),
        border_snr=float(parameters[0] / noise) if noise > 0 else None,
    )
    rank = int(np.linalg.matrix_rank(fitted.jac))
    result["jacobian_rank"] = rank
    if rank == npar:
        covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac) * float(
            np.sum(fitted.fun**2) / (n - npar)
        )
        covariance *= np.outer(scale_vector, scale_vector)
        if np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) >= 0):
            result["formal_covariance"] = covariance.tolist()
            result["formal_center_sigma_x_pixel"] = float(np.sqrt(covariance[1, 1]))
            result["formal_center_sigma_y_pixel"] = float(np.sqrt(covariance[2, 2]))
    else:
        result["quality_reasons"].append("rank_deficient_formal_covariance")
    # Mask-normalized smoothing detects separated bright maxima without filling
    # missing pixels into the fit. These are candidate peaks, not source IDs.
    smooth_support = gaussian_filter(finite.astype(float), 1.0)
    smooth = gaussian_filter(np.where(finite, work, 0.0), 1.0) / np.maximum(
        smooth_support, 1e-12
    )
    distance = max(2, int(round(min(parameters[3:5]) / 1.5)))
    maxima = (
        finite
        & (smooth_support > 0.8)
        & (smooth == maximum_filter(smooth, size=2 * distance + 1))
        & (smooth > b0 + max(0.35 * parameters[0], 5 * noise))
    )
    components, count = label(maxima)
    result["significant_peak_count"] = int(count)
    peak_positions = []
    for number in range(1, count + 1):
        py, px = np.unravel_index(
            np.argmax(np.where(components == number, smooth, -np.inf)), work.shape
        )
        peak_positions.append([int(px + x0), int(py + y0)])
    result["candidate_peak_pixels"] = peak_positions
    if count > 1:
        result["quality_reasons"].append("multiple_bright_peaks_detected")
    if (
        result["residual_rms_over_amplitude"] > 0.15
        or result["residual_max_abs_over_amplitude"] > 0.5
    ):
        result["quality_reasons"].append("large_single_gaussian_residual")
    sx, sy, theta = parameters[3:6]
    extent_x = 2 * np.hypot(sx * np.cos(theta), sy * np.sin(theta))
    extent_y = 2 * np.hypot(sx * np.sin(theta), sy * np.cos(theta))
    if (
        parameters[1] - extent_x < x0
        or parameters[1] + extent_x > x1 - 1
        or parameters[2] - extent_y < y0
        or parameters[2] + extent_y > y1 - 1
    ):
        result["quality_reasons"].append("two_sigma_source_reaches_roi_boundary")
    if result["finite_fraction"] < 0.8:
        result["quality_reasons"].append("low_valid_pixel_coverage")
    if result["border_snr"] is not None and result["border_snr"] < 5:
        result["quality_reasons"].append("low_border_snr")
    try:
        _attach_wcs(result, header, parameters)
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        result["quality_reasons"].append("invalid_or_unsupported_hpc_wcs")
        result["wcs_failure_detail"] = str(exc)
    result["model_adequacy"] = (
        "diagnostic_concern"
        if any(
            reason in result["quality_reasons"]
            for reason in (
                "multiple_bright_peaks_detected",
                "large_single_gaussian_residual",
            )
        )
        else "no_flag_from_heuristics"
    )
    result["quality_status"] = "review" if result["quality_reasons"] else "ok"
    return result


def nearest_time_matches(query_times, reference_times, tolerance_sec):
    """Match numeric timestamps while retaining misses, ties and frame reuse.

    Times must be in the same scale and seconds convention. No clock offset is
    inferred. Tied nearest frames are marked ambiguous with all candidate
    indices; their lowest original index is retained for reproducible inspection.
    A timestamp match does not prove overlapping integration windows.
    """
    tolerance = float(tolerance_sec)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_sec must be finite and nonnegative")
    queries = np.asarray(query_times, dtype=float)
    references = np.asarray(reference_times, dtype=float)
    if queries.ndim != 1 or references.ndim != 1:
        raise ValueError("Times must be one-dimensional")
    valid_indices = np.flatnonzero(np.isfinite(references))
    rows = []
    for i, time in enumerate(queries):
        row = {
            "query_index": int(i),
            "reference_index": None,
            "time_delta_sec": None,
            "status": "unmatched",
            "candidate_indices": [],
            "reference_reused": False,
            "reference_use_count": 0,
            "tolerance_sec": tolerance,
        }
        if not np.isfinite(time):
            row["status"] = "invalid_query_time"
        elif len(valid_indices):
            distances = np.abs(references[valid_indices] - time)
            best = float(distances.min())
            # Only a floating-point representation allowance, not clock error.
            epsilon = max(1e-9, 2 * float(np.spacing(abs(time))))
            if best <= tolerance + epsilon:
                choices = valid_indices[np.abs(distances - best) <= epsilon]
                j = int(choices[0])
                row.update(
                    reference_index=j,
                    time_delta_sec=float(references[j] - time),
                    status="matched" if len(choices) == 1 else "ambiguous_nearest",
                    candidate_indices=[int(k) for k in choices],
                )
        rows.append(row)
    uses = Counter(
        row["reference_index"] for row in rows if row["reference_index"] is not None
    )
    for row in rows:
        count = uses.get(row["reference_index"], 0)
        row.update(reference_reused=count > 1, reference_use_count=int(count))
    return rows
