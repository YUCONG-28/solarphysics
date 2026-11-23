# Gaussian source fitting

Purpose: state the fitting model, its initialization, its quality control, and the
difference between an observed apparent source size and an intrinsic one, so that a
quoted center or width carries its conditions with it.

Reader: whoever fits a radio source, overlays a fitted center on an image, or
compares a fitted height with a model height. Vocabulary is in
[terminology](terminology.md); reader and display behavior in
[radio comparison](radio_comparison.md); units and masks in
[foundations](foundations.md).

This note states method conditions. It cites no literature catalog: a personal
reference list is not public source material.

## 1. Purpose and reader

The implementation reuses one rotated elliptical Gaussian across the radio
workflows. This page is the reference for what that model assumes, which
diagnostics decide whether a fit may be quoted, and what its covariance does not
cover.

## 2. Scope

Covers the single-source model, background alternatives, initialization, quality
control, centroid validation, and beam deconvolution.

Does not cover absolute flux calibration, time matching, or the physical
interpretation of a derived height.

## 3. Definitions and notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `I(x, y)` | Observed intensity in the ROI | image units |
| `A` | Peak amplitude above background | image units |
| `x0, y0` | Fitted source center | arcsec |
| `σx, σy` | Gaussian widths along the rotated axes | arcsec |
| `θ` | Position angle of the major axis | rad |
| `B, B0, Bx, By` | Constant or planar background terms | image units |
| `σ_noise` | Robust pixel-noise estimate | image units |
| `FWHM_observed` | Full width at half maximum of the fitted source | arcsec |
| `FWHM_beam` | Instrumental beam width | arcsec |
| `FWHM_intrinsic` | Beam-deconvolved width | arcsec |

## 4. Method

The default single-source model is an elliptical Gaussian with a constant
background:

```math
I(x,y) = A\exp\!\left[-\tfrac{1}{2}Q(x,y)\right] + B,
\qquad
Q(x,y) = \left(\frac{x'}{\sigma_x}\right)^{2}
       + \left(\frac{y'}{\sigma_y}\right)^{2}
\tag{1}
```

```math
\begin{aligned}
x' &= (x - x_0)\cos\theta + (y - y_0)\sin\theta \\
y' &= -(x - x_0)\sin\theta + (y - y_0)\cos\theta
\end{aligned}
\tag{2}
```

For an ROI with a clear gradient, the background becomes a tilted plane:

```math
I(x,y) = G(x,y) + B_0 + B_x x + B_y y
\tag{3}
```

with `G` the Gaussian term of Eq. (1) [implemented]. A multi-source model is used
only when the ROI is demonstrably multi-peaked and the interpretation supports it.

Initial values:

| Parameter | Initial value |
| --- | --- |
| Amplitude `A` | ROI peak minus edge background |
| Position `x0, y0` | Peak pixel or intensity-weighted centroid |
| Widths `σx, σy` | Half-maximum region |
| Background `B` | ROI-edge median |
| Rotation `θ` | Zero, or a second-moment estimate |

Constraints: positive amplitude and widths, a centroid inside the ROI, and
rejection of any solution in which the background dominates the fitted source.

Noise for the residual diagnostics uses the median absolute deviation when the
selected noise region supports it:

```math
\sigma_{\mathrm{noise}} \approx 1.4826\,\mathrm{MAD}
\tag{4}
```

## 5. Assumptions and validity

- The source is unresolved or moderately resolved inside the ROI, so one elliptical
  Gaussian plus background describes it.
- The noise region is source-free and large enough for the MAD estimator to be
  meaningful; the region used is recorded.
- Parameter covariance describes pixel-noise propagation only.
- A deconvolved width is quoted only when the beam is known and measured, and the
  correction is applied axis by axis.
- Weak, extended, or off-center sources are sensitive to the background choice; a
  centroid shift between background alternatives is a diagnostic, not noise.

## 6. Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| Background dominates the source | Rejection rule | Fit rejected | Parameters and residuals |
| Centroid on the ROI edge | Flag | Frame excluded from trajectory fits | ROI bounds and center |
| Multi-peak ROI | Flag | No automatic multi-source rescue | Peak list |
| Implausibly large FWHM | Flag | Size not quoted | Fitted widths |
| Fitted and contour centers disagree | `delta_r` | Review of background, ROI, threshold | Both centers |
| Beam unknown | Report level | Only apparent size quoted | Beam provenance |
| Structured residuals | Flag | Model inadequate | Residual image |

The fitted center and an independent contour or intensity-weighted center are
recorded in parallel:

```math
\Delta r = \sqrt{(x_G - x_C)^2 + (y_G - y_C)^2}
\tag{5}
```

With a reliable beam, an approximate axis-by-axis deconvolution is

```math
\mathrm{FWHM}_{\mathrm{intrinsic}}^{2}
\approx \mathrm{FWHM}_{\mathrm{observed}}^{2}
- \mathrm{FWHM}_{\mathrm{beam}}^{2}
\tag{6}
```

Without a beam, the result is an **observed apparent source size**. Scattering,
refraction, and magnetic-field geometry can shift, broaden, elongate, or split an
apparent source even after instrumental beam treatment [inferred]. A covariance
that covers pixel noise only is never quoted as a localization error.

## 7. Verification

A fit is quotable when its status, SNR, reduced chi-square, residual RMS, noise
estimate with region, and parameter covariance are recorded, and every flag has
been read. Preserving the original ROI, the model image, and the residual image is
what makes a later re-check possible.

## 8. Code anchors

| Behavior | Implementation |
| --- | --- |
| Model and Jacobian | `Python/solar_toolkit/radio/gaussian_models.py` |
| Background and noise estimation | `Python/solar_toolkit/radio/gaussian_background.py` |
| Fit driver | `Python/solar_toolkit/radio/gaussian_fit.py` |
| Diagnostics and flags | `Python/solar_toolkit/radio/gaussian_diagnostics.py` |
| ROI masks | `Python/solar_toolkit/radio/gaussian_masks.py` |
| Display overlay | `Python/solar_toolkit/radio/gaussian.py` |
