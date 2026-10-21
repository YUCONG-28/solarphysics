# Scientific data contracts

The library builds on Astropy and SunPy while keeping instrument-specific
selection, calibration and physical assumptions explicit. The reference
implementations are [Astropy 8.0.1](https://github.com/astropy/astropy/tree/v8.0.1)
and [SunPy 8.0.1](https://github.com/sunpy/sunpy/tree/v8.0.1). Calling an upstream
operation does not imply that every mask, uncertainty or header field is
propagated automatically.

## Time and validity

Scientific durations use continuous TAI nanoseconds or Astropy `TimeDelta`,
not subtraction of UTC `datetime` values. Equivalent UTC and TAI inputs must
produce the same result, including across leap seconds. Observation inputs
require explicit dates and times; missing dates never become the current
wall-clock time. Invalid calendar fields are rejected instead of rolling into
the next minute.

`TimeStamp` preserves nanosecond precision. Python `datetime` compatibility
views retain their documented microsecond precision and reject leap seconds,
which that type cannot represent. They are not used for scientific durations.
Numeric epochs require an explicit format. Supported compact calendar integers
are parsed as calendar fields, never as floating-point epochs.

The strict observation boundary is `normalize_time`. Legacy `parse_time` and
date-only radio configuration retain their midnight compatibility convention;
that convention must not supply a missing observation date or clock. Explicit
short-date overlay formats remain compatibility inputs, without truncating
fractional seconds or ignoring trailing text.

An absent optional quality column remains compatible with old tables. A
present but missing, empty or unknown value is not evidence of validity.
Coverage planning can reuse observations and keeps real offsets; equal
timestamps alone do not identify duplicate observations.

## Units, masks and precision

APIs with unit-bearing parameter names retain their numeric return types and
documented units. Equivalent Astropy `Quantity` inputs are converted explicitly
using [`to_value`](https://docs.astropy.org/en/stable/units/quantity.html);
incompatible dimensions raise an error. Bare numbers keep the units stated by
the API. Units are not inferred from magnitude.

Explicit masks, masked arrays and nonfinite samples are excluded from
scientific statistics. Missing scientific values remain missing, including
when an array is fully masked. Display-only filling must be requested
explicitly. Array conversions do not modify the source data or mask.

Integer detector arrays are promoted before polarization arithmetic. Block
averaging accumulates in float64; lower output precision requires an explicit
request. CSO ratio output precision remains unchanged. Empty derivatives stay
empty; a finite single sample has zero derivative and a missing single sample
remains NaN. Forward differences of one sample remain empty.

## Maps, uncertainty and coverage

Astropy WCS supplies coordinate transforms, including CD/PC matrices, CROTA,
axis order, angular units and projections. A rectangular extent cannot replace
a rotated or projected coordinate grid. Use a WCS-aware display or explicitly
reproject such data before using a rectangular image display.

`read_aia_background` resamples through the full inverse WCS to a regular
helioprojective display grid. Its bilinear interpolation preserves missing
coverage and is not a flux-conserving measurement. The returned one-dimensional
axes describe that resampled grid. Radio Gaussian fits remain native pixel
models: an explicit `coordinate_header` supplies world centers, local angular
widths and mapped contour curves without changing the fitted pixel model.
Local angular widths use the WCS Jacobian at the center; they do not describe
a globally Gaussian sky source over a distorted projection.

AIA exposure normalization retains the acquisition exposure and mask, updates
the data unit and propagates standard deviation, variance or inverse variance
through [CCDData arithmetic](https://docs.astropy.org/en/stable/nddata/ccddata.html).
Already normalized rates are not divided again. Uncertainty is not invented
when none was supplied.

`align_maps_to_reference` returns a map by default; `return_footprint=True`
returns the map and coverage. Acquisition unit, exposure and observation time
are kept separate from the target WCS, observer and coordinate reference time.
[SunPy reprojection](https://docs.sunpy.org/en/stable/generated/api/sunpy.map.GenericMap.html#sunpy.map.GenericMap.reproject_to)
does not preserve non-WCS metadata automatically. If uncertainty cannot be
propagated, `uncertainty_policy="raise"` rejects the operation; explicit
`"drop"` records its removal.

Radio scattered interpolation uses `fallback=None` by default. Explicit
`fallback="nearest"` enables historical rendering behavior, recording the
method, reason and footprint. Extrapolated values are not observed coverage
and must be excluded from downstream measurements.

## FITS and dependency verification

`read_fits_data_header` returns a detached array and copied header without
keeping a hidden file mapping alive. For large images, use an explicit
`astropy.io.fits.open` context and section reads.

The library supports Python 3.12 and later; Apps retain Python 3.14. The
[minimum constraints](../environment/library-minimum-py312.txt) exercise direct
runtime floors on Python 3.12. Optional, development and transitive dependencies
are resolved separately. This is a compatibility test, not a sealed environment;
exact artifact replay follows the [environment manual](../environment/README.md).

See the [API map](../Python/docs/FUNCTION_MAP.md) for entry points and the
[migration guide](migration.md) for changed behavior.
