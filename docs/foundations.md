# Scientific data contracts

## Purpose and reader

Purpose: fix what a value returned by `solar_toolkit` means — its unit, its
mask, its uncertainty and its coordinate frame — so a caller can separate a
measurement from a convenience value.

Reader: a caller deciding whether a returned value may be used as a measurement,
and the reviewer checking that a documentation change kept the contract.

The library builds on Astropy and SunPy while keeping instrument-specific
selection, calibration and physical assumptions explicit. The reference
implementations are [Astropy 8.0.1](https://github.com/astropy/astropy/tree/v8.0.1)
and [SunPy 8.0.1](https://github.com/sunpy/sunpy/tree/v8.0.1). Calling an upstream
operation does not imply that every mask, uncertainty or header field is
propagated automatically.

The one claim these contracts do not support: that a call which returns is
therefore a calibrated measurement. Every negative boundary below is as binding
as the positive statement beside it; the tables in
[Failure modes and uncertainty](#failure-modes-and-uncertainty) name what is
retained when one is hit.

Vocabulary: [terminology](terminology.md). File structure:
[writing-standard](writing-standard.md).

## Scope

In scope, one subsection of [Method](#method) each: time scales and validity,
units and masks, precision of derived values, maps, uncertainty and coverage.

Deliberately out of scope:

- Instrument-specific selection, calibration and physical assumptions; the
  library keeps them explicit and they remain the caller's.
- Astropy and SunPy behavior beyond what a cited call documents.
- A sealed environment. The dependency floors in [Verification](#verification)
  are a compatibility test; exact artifact replay follows the
  [environment manual](../environment/README.md).

## Definitions and notation

| Symbol or field | Unit, frame | Meaning |
| --- | --- | --- |
| TAI | continuous scale, no leap second | time scale scientific durations are expressed in |
| `TimeStamp` | nanosecond resolution | exact observation stamp; the strict input boundary is `normalize_time` |
| `datetime` view | microsecond resolution | compatibility view; rejects leap seconds and is not used for durations |
| validity mask | boolean per pixel | explicit mask, masked array or nonfinite sample excluded from statistics |
| count rate | the unit carried by the data unit | already normalized rate; not divided by exposure again |
| HPLN/HPLT | helioprojective WCS axes, in the angular unit of the supplied WCS | axes of the regular display grid returned by `read_aia_background` |
| `coordinate_header` | WCS header | supplies world centers and local angular widths to a native-pixel Gaussian fit |
| `uncertainty_policy` | flag: `raise` or `drop` | reject an unpropagatable uncertainty, or record its removal |
| `fallback` | flag: `None` or `nearest` | radio scattered-interpolation behavior |
| `return_footprint` | flag | return the coverage together with the map |

## Method

### Time and validity

Scientific durations use continuous TAI nanoseconds or Astropy `TimeDelta`, not
subtraction of UTC `datetime` values.

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

Coverage planning can reuse observations and keeps real offsets.

### Units, masks and precision

APIs with unit-bearing parameter names retain their numeric return types and
documented units. Equivalent Astropy `Quantity` inputs are converted explicitly
using [`to_value`](https://docs.astropy.org/en/stable/units/quantity.html);
incompatible dimensions raise an error. Bare numbers keep the units stated by
the API.

Explicit masks, masked arrays and nonfinite samples are excluded from scientific
statistics. Missing scientific values remain missing, including when an array is
fully masked. Display-only filling must be requested explicitly. Array
conversions do not modify the source data or mask.

Integer detector arrays are promoted before polarization arithmetic. Block
averaging accumulates in float64; lower output precision requires an explicit
request. CSO ratio output precision remains unchanged. Empty derivatives stay
empty; a finite single sample has zero derivative and a missing single sample
remains NaN. Forward differences of one sample remain empty.

### Maps, uncertainty and coverage

Astropy WCS supplies coordinate transforms, including CD/PC matrices, CROTA,
axis order, angular units and projections.

`read_aia_background` resamples through the full inverse WCS to a regular
helioprojective display grid. The returned one-dimensional axes describe that
resampled grid. Radio Gaussian fits remain native pixel models: an explicit
`coordinate_header` supplies world centers, local angular widths and mapped
contour curves without changing the fitted pixel model.

AIA exposure normalization retains the acquisition exposure and mask, updates
the data unit and propagates standard deviation, variance or inverse variance
through [CCDData arithmetic](https://docs.astropy.org/en/stable/nddata/ccddata.html).

`align_maps_to_reference` returns a map by default; `return_footprint=True`
returns the map and coverage. Acquisition unit, exposure and observation time
are kept separate from the target WCS, observer and coordinate reference time.
If uncertainty cannot be propagated, `uncertainty_policy="raise"` rejects the
operation; explicit `"drop"` records its removal.

Radio scattered interpolation uses `fallback=None` by default. Explicit
`fallback="nearest"` enables historical rendering behavior, recording the
method, reason and footprint.

## Assumptions and validity

Each statement below bounds the method above; the failure table that follows
gives the trigger and the detection for it.

- Equivalent UTC and TAI inputs must produce the same result, including across
  leap seconds.
- Observation inputs require explicit dates and times; missing dates never
  become the current wall-clock time. Invalid calendar fields are rejected
  instead of rolling into the next minute.
- An absent optional quality column remains compatible with old tables. A
  present but missing, empty or unknown value is not evidence of validity.
- Equal timestamps alone do not identify duplicate observations.
- Units are not inferred from magnitude.
- A rectangular extent cannot replace a rotated or projected coordinate grid.
  Use a WCS-aware display or explicitly reproject such data before using a
  rectangular image display.
- Bilinear interpolation in `read_aia_background` preserves missing coverage and
  is not a flux-conserving measurement.
- Local angular widths use the WCS Jacobian at the center; they do not describe
  a globally Gaussian sky source over a distorted projection.
- Already normalized rates are not divided again. Uncertainty is not invented
  when none was supplied.
- [SunPy reprojection](https://docs.sunpy.org/en/stable/generated/api/sunpy.map.GenericMap.html#sunpy.map.GenericMap.reproject_to)
  does not preserve non-WCS metadata automatically.
- Extrapolated values are not observed coverage and must be excluded from
  downstream measurements.

## Failure modes and uncertainty

| Trigger | Detection | Consequence | Retained record |
| --- | --- | --- | --- |
| UTC `datetime` subtraction used for a duration | the `datetime` view rejects a leap second with `ValueError` | the result is not a scientific duration | the exact `TimeStamp` remains the observation |
| Observation input without an explicit date, or with invalid calendar fields | parsing is rejected rather than rolled into the next minute | no wall-clock time is substituted | the rejected input stays the caller's to correct |
| Scientific value missing, including a fully masked array | the mask, not the data, is inspected | the value stays missing | the mask, unchanged by array conversions |
| Uncertainty cannot be propagated through reprojection | `uncertainty_policy="raise"` rejects the operation | the map is not returned with an invented uncertainty | `uncertainty_policy="drop"` records the removal |
| Interpolation needs a value outside the observed radio footprint | `fallback="nearest"` chosen explicitly | the value is not observed coverage | recorded method, reason and footprint |
| Rotated or projected grid displayed as a rectangular extent | WCS carries CD/PC, CROTA or a projection | the extent does not describe the grid | the WCS metadata |

## Verification

`read_fits_data_header` returns a detached array and copied header without
keeping a hidden file mapping alive. For large images, use an explicit
`astropy.io.fits.open` context and section reads.

The library supports Python 3.12 and later; Apps retain Python 3.14. The
[minimum constraints](../environment/library-minimum-py312.txt) exercise direct
runtime floors on Python 3.12. Optional, development and transitive dependencies
are resolved separately. This is a compatibility test, not a sealed environment;
exact artifact replay follows the [environment manual](../environment/README.md).

A change is verified by exercising those floors and by reading one FITS product
back through an explicit section read.

## Code anchors

| Step | Module or entry point |
| --- | --- |
| Time normalization and exact stamps | `normalize_time`, `TimeStamp`; legacy `parse_time` retained |
| Unit conversion | Astropy `Quantity` via `to_value` |
| AIA display background | `read_aia_background` |
| Map alignment and coverage | `align_maps_to_reference` |
| FITS array and header read | `read_fits_data_header` |

See the [API map](../Python/docs/FUNCTION_MAP.md) for entry points and the
[migration guide](migration.md) for changed behavior.
