"""Shared pure helpers for rotated 2D Gaussian source fitting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

IntArray = NDArray[np.intp]

__all__ = [
    "elliptical_gaussian_2d",
    "fit_elliptical_gaussian",
    "initial_guess_from_peak",
    "true_indices",
    "unravel_2d_index",
]


def elliptical_gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, theta):
    """Evaluate a rotated Gaussian without a background or beam correction.

    ``xy`` contains NumPy x/y coordinate arrays with broadcast-compatible
    shapes, usually matching 2-D grids. ``x0``/``y0`` and the nonzero widths
    ``sigma_x``/``sigma_y`` use those coordinate units; ``theta`` is radians
    in the rotation convention defined by this model. ``A`` is the peak in
    the caller's intensity unit. The result has the broadcast coordinate shape.

    This evaluator does not validate physical parameter choices, mask missing
    values or fit data. NumPy broadcasting and nonfinite arithmetic apply;
    inputs are not modified. The fitting wrapper imposes its own positive-width
    and amplitude bounds.
    """

    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_shift = x - x0
    y_shift = y - y0
    x_rot = cos_t * x_shift + sin_t * y_shift
    y_rot = -sin_t * x_shift + cos_t * y_shift
    exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    return A * np.exp(exponent)


def unravel_2d_index(
    flat_index: int | np.integer, shape: tuple[int, ...]
) -> tuple[int, int]:
    """Return integer 2D coordinates for a flattened index."""

    coords = np.asarray(np.unravel_index(int(flat_index), shape), dtype=np.intp)
    return int(coords[0]), int(coords[1])


def true_indices(mask: np.ndarray) -> IntArray:
    """Return one-dimensional indices where ``mask`` is true."""

    mask_bool = np.asarray(mask, dtype=np.bool_)
    return np.asarray(np.nonzero(mask_bool)[0], dtype=np.intp)


def initial_guess_from_peak(data, x, y):
    """Build the legacy peak/FWHM guess on explicit image-coordinate axes.

    ``data`` is a finite 2-D image of shape ``(len(y), len(x))`` and x/y are
    nondegenerate 1-D coordinate arrays. The returned tuple is
    ``(A, x0, y0, sigma_x, sigma_y, theta)``: the global maximum sets amplitude
    and center, half-maximum row/column spans divided by 2.355 estimate widths,
    and the initial angle is zero radians. A span/10 fallback is used where a
    half-maximum span cannot be measured. Absolute spans support either axis
    direction without reordering coordinates or data.

    This is an unweighted single-source heuristic, without NaN/mask handling,
    background fitting or parameter validation. Callers must select suitable
    image data; it is not a source-detection or uncertainty estimate.
    """

    max_y, max_x = unravel_2d_index(int(np.argmax(data)), data.shape)
    init_x0 = x[max_x]
    init_y0 = y[max_y]
    init_A = np.max(data)

    half_max = init_A / 2.0
    row_max = data[max_y, :]
    indices = true_indices(row_max >= half_max)
    if len(indices) > 1:
        init_sigma_x = abs(x[indices[-1]] - x[indices[0]]) / 2.355
    else:
        init_sigma_x = abs(x[-1] - x[0]) / 10.0

    col_max = data[:, max_x]
    indices_y = true_indices(col_max >= half_max)
    if len(indices_y) > 1:
        init_sigma_y = abs(y[indices_y[-1]] - y[indices_y[0]]) / 2.355
    else:
        init_sigma_y = abs(y[-1] - y[0]) / 10.0

    return (init_A, init_x0, init_y0, init_sigma_x, init_sigma_y, 0.0)


def fit_elliptical_gaussian(data, x, y, initial_guess=None):
    """Fit an unweighted single Gaussian on caller-supplied coordinate axes.

    ``data`` is a finite image of shape ``(len(y), len(x))``; x/y are 1-D axes
    in the desired coordinate units, and amplitude uses the image's intensity
    unit. An explicit ``initial_guess`` has order
    ``(A, x0, y0, sigma_x, sigma_y, theta)`` with angle in radians. Otherwise
    :func:`initial_guess_from_peak` supplies positive widths for ascending or
    descending axes, preserving data and coordinate ordering.

    Returns ``(parameters, covariance)`` from SciPy ``curve_fit``: six fitted
    parameters in the order above and their 6-by-6 formal covariance matrix.
    Existing bounds are retained: nonnegative amplitude, widths at least
    ``1e-3`` in coordinate units, unbounded centers and angle in
    ``[-pi/2, pi/2]``. The model has no background term or beam deconvolution;
    covariance does not include calibration or correlated-pixel uncertainty.

    This compatibility fitter does not mask missing samples or validate image
    shape separately. NumPy/SciPy input errors and convergence failures
    propagate. Inputs are not modified, and no quality acceptance is implied.
    """

    X, Y = np.meshgrid(x, y)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    data_flat = data.ravel()

    if initial_guess is None:
        initial_guess = initial_guess_from_peak(data, x, y)

    bounds = (
        [0, -np.inf, -np.inf, 1e-3, 1e-3, -np.pi / 2],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi / 2],
    )

    popt, pcov = curve_fit(
        elliptical_gaussian_2d,
        (x_flat, y_flat),
        data_flat,
        p0=initial_guess,
        bounds=bounds,
        maxfev=5000,
    )
    return popt, pcov


# Backward-compatible private names used by older scripts/tests.
_unravel_2d_index = unravel_2d_index
_true_indices = true_indices
