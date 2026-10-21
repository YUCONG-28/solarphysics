# Migration guide

This guide describes current repository compatibility boundaries. The
[changelog](../Python/CHANGELOG.md) records historical changes. Version numbers
remain synchronized between the two distributions; documentation cleanup alone
does not create a release.

## Documentation and metadata

- [README.md](../README.md) is the canonical English home;
  [README.zh-CN.md](../README.zh-CN.md) is the Chinese counterpart. Existing
  [README.en.md](../README.en.md) links and section anchors remain available.
  Previous Chinese homepage section anchors also remain on the English home.
- [CONTRIBUTING.md](../CONTRIBUTING.md) is the canonical English contribution
  guide, with [Chinese guidance](../CONTRIBUTING.zh-CN.md) and an
  [English compatibility entry](../CONTRIBUTING.en.md).
- Current package and citation metadata identify Y. Li, Shandong University.
  Historical copyright and third-party attribution remain intact. The root
  [citation entry](../CITATION.cff) and [license scope](../LICENSING.md) make the
  repository-level terms explicit; existing GPL/MPL component terms remain.

## Installations and imports

The library now requires Python 3.12 or later and the Astropy 8 / SunPy 8 /
NumPy 2 generation. Python 3.10 and 3.11 installations must upgrade their
Miniforge environment before installing this checkout. Apps continue to use
Python 3.14. The minimum-dependency CI job verifies direct runtime floors
separately from the current-dependency matrix.

## Scientific foundations

See [scientific data contracts](foundations.md) for units, masks, uncertainty
and WCS semantics. Public imports and numeric return types remain available.
Science durations use exact TAI differences; malformed timestamps no longer
pass through a permissive secondary parser. Explicitly missing quality values
are invalid, while tables without optional quality columns remain supported.

Masked values no longer contribute to peaks or statistics. Equivalent units
are converted; incompatible dimensions raise errors. Integer polarization
arithmetic no longer wraps before division. Block means now default to float64;
callers requiring a smaller representation can explicitly request a dtype.

Reprojection no longer silently substitutes nearest-neighbor interpolation.
Select `fallback="nearest"` explicitly for historical rendering and inspect
the returned footprint before interpreting values as measurements. Map
reprojection rejects unpropagated uncertainty by default; an explicit
`uncertainty_policy="drop"` records its removal. Exposure normalization
propagates supported uncertainties. FITS reads return detached arrays.

AIA display backgrounds now use a regular angular grid sampled through the
complete WCS. Display pixels may therefore differ from a simple stride through
the native array. Missing coverage remains NaN. ROI/center transforms require
explicit helioprojective axes; `legacy_linear=True` is an explicit historical
coordinate approximation where supported. Gaussian fitting accepts an optional
`coordinate_header`; native fitted parameters remain unchanged while world
centers and local widths follow that WCS. Precomputed brightness-temperature
loading preserves missing values by default; use `fill_missing=0` only for
display compatibility.

## Package installation boundaries

Create the named Miniforge environment before using a fresh checkout; see the
[library setup](../README.md#start-with-the-library). The scientific library
keeps the `solar-physics-toolkit` distribution name, `solar_toolkit` import
namespace and Python 3.12 minimum. Apps requires Python 3.14 and editable source
installs of both `Python` and `Apps`; Apps wheel checks validate contents rather
than a standalone deployment. Scientific-library wheel installation is checked
independently of Apps. Source builds require setuptools 77 or later for SPDX
license metadata.

The source specifications resolve version ranges. For exact platform replay,
follow the [environment manual](../environment/README.md); source-binding
refresh is not a new artifact capture or runtime replay. Runtime state,
observations and outputs stay under ignored `Local/` or private external roots.

Scientific modules remain independent of Apps. Use public imports listed in
the [API map](../Python/docs/FUNCTION_MAP.md); applications and workflow entry
points use `Apps/run.sh` or `Apps/run.ps1` with explicit inputs. No built-in
research event is supplied.

## Scientific input validation

The following stricter input handling can affect existing analyses:

- `map.create_aia_submap` now uses all four caller bounds in the existing order
  `(xmin, xmax, ymin, ymax)` in arcseconds. Bounds must be finite and strictly
  increase on each axis. Explicit `legacy_roi=True` reproduces the historical
  horizontal-strip behavior; it is a compatibility choice, not the default.
- `map.normalize_aia_exposure` requires explicit DN, count or photon units,
  optionally per pixel. Existing rates are returned unchanged. Integrated
  values require finite positive exposure in seconds and are converted to
  rates, preserving the original exposure metadata and copying the mask.
  Missing/unsupported units and invalid exposure raise `ValueError`.
  AIA single-frame, mosaic and difference entry points use this shared policy,
  so rate images are not divided by exposure again.
- `aia.background.find_nearest_aia` requires finite nonnegative scalar time
  tolerance, including for empty tables. NaN, infinity and booleans raise
  `ValueError`; zero accepts an exact time match only.
- `aia.io.build_multi_band_slots` retains its legacy index-based composition:
  it sorts each band's files, truncates to the shortest list and groups by
  position. A slot is not evidence of simultaneous observations. Use
  `time.build_sequence_match_plan` with caller-selected tolerances when matching
  by observation time is required.
- `map.get_map_obs_time` returns microsecond-resolution naive UTC while
  converting the original time scale, including TAI and FITS `TIMESYS`.
  Date-only metadata needs `TIME-OBS` or `TIME_OBS`; absent observation metadata
  raises `KeyError`, and incomplete or invalid timestamps raise `ValueError`.
  Image filename times likewise convert explicitly labeled TAI to UTC.
- `cme.running_difference` requires identical array shapes and returns floating
  differences. Unsigned input is converted before subtraction, preserving
  negative changes; implicit shape broadcasting is rejected.
- HMI magnetic processing excludes explicit masks as well as nonfinite pixels,
  preserves missing pixels and copies the output mask without changing inputs.
  PFSS global boundaries reject masked cells even when their stored values are
  finite; a mask with no flagged cells remains valid.
- `xray_dem.processing.calculate_derivative` retains `median-gradient`,
  `gradient` and `forward` modes. Flux must be one-dimensional; a supplied time
  axis must contain equally many valid, strictly increasing timestamps rather
  than unlabelled numbers. Explicit spacing must be a finite positive scalar.
  Invalid methods, axes or spacing raise `ValueError` even for short arrays;
  relative nanosecond times preserve fine cadence at large epochs.
- Automatic Gaussian width guesses use absolute coordinate spacing, allowing
  decreasing axes without rearranging the coordinate grids or data.
- `timeseries.derivative_series` requires a one-dimensional series and finite
  positive scalar spacing. Empty input returns an empty array; a finite single
  sample returns zero while a missing single sample remains NaN.
- Frequency configuration bounds must be finite and positive. Saved YAML
  configuration uses safe sequences that can be read by the existing safe loader.
  The compatibility `SolarDataConfig` no longer supplies machine paths or an
  event ROI: `data_dir`, `output_dir` and `roi_bounds` default to `None` and must
  be supplied explicitly before a workflow uses them.
- `xray_dem.hxi.load_hxi_lightcurve` converts caller-supplied epoch and interval
  bounds to UTC, rejects reversed intervals and requires finite one-dimensional
  time offsets with matching four-channel counts. Its historical epoch and
  energy labels remain compatibility assumptions; the reader does not infer
  instrument clock calibration from FITS metadata.
- Generic manifest links resolve to their selected input files even when the
  input and link roots have different depths. Archive wavelength quantities are
  converted to angstroms before band selection, including nanometre inputs.
- `radio.io.truthy` accepts explicit true values, numeric `1` and recognized
  true strings. Missing values, NaN and unknown strings are false.
- A supplied but empty trajectory `quality_flag` is `unknown`, rather than
  `ok`. Older tables without that column retain their compatibility behavior.
  Timezone-aware timestamps are converted to UTC before trajectory processing.
  The shared radio calendar parser accepts exact compact integers and strings
  consistently: `20000101044829509` represents
  `2000-01-01T04:48:29.509` UTC, rather than Unix nanoseconds. CSV center-table
  loaders preserve time columns as text before inference, including columns
  with missing values. Floating-point timestamps are invalid because their
  digits may already be rounded; they and other invalid times are excluded
  from normalized trajectories even with `valid_only=False`. Use ISO text or
  exact compact text, and store spreadsheet timestamps as text. Numeric Unix
  epochs require explicit conversion by the caller.
  `radio.reprojection.nearest_time_index` also normalizes naive UTC, offset-aware
  and explicitly scaled times before comparing integer nanoseconds; ties still
  select the first candidate and tolerances remain inclusive.
- Newkirk harmonic and multiplier settings must be finite and positive or
  raise `ValueError`. Non-positive or non-finite observation frequencies
  produce NaN heights. Quality-accepted Gaussian samples with invalid frequency
  retain their rows with `newkirk_quality_flag` and `newkirk_invalid_reason`;
  quality-rejected fits are still excluded. Drift-line extrapolation retains
  invalid values with reasons, and drift-rate estimation marks these samples
  `quality_flag=invalid_frequency`.
  The drift-table builder validates model settings even for empty or fully
  quality-rejected input tables.
- The HMI overlay CLI requires explicit `--roi-bounds` or a caller-owned private
  configuration. The library's overlay and magnetogram workflow functions also
  require an explicit ROI before file access. Their order is
  `(xmin, ymin, xmax, ymax)` in arcseconds, which differs from
  `create_aia_submap`. Exported historical ROI constants remain available for
  deliberate compatibility calls.

These flags describe software validity conditions. They do not replace
instrument calibration or scientific assessment of a result.

## Output filenames

The [image-naming API](../Python/README.md#scientific-image-filenames) defines a
contract for workflows that adopt it. Explicit caller filenames remain valid.
Retained exporters may keep their documented naming, including the AIA/radio
composite stem and collision suffixes. Do not infer that every directory-only
export uses the image-naming contract or rename historical results implicitly.

## Maintenance commands

`tools quick save`, `push` and `update` now reject `main`, detached HEAD and an existing
staged change before work begins. A failed save preserves the original index.
`tools quick push` reuses an existing pull request for the branch when present.
Radio help lists its catalog without initializing runtime state.

`tools release prepare` is a preview by default; execution changes only the two
version files and changelog on a clean feature branch. The old `release run`
spelling is an alias for that preparation step and no longer publishes.
After its pull request is merged and CI passes, a maintainer invokes
`tools release publish --version <X.Y.Z> --commit <full-merged-commit-sha>`
as a separate preview, adding `--execute` only to publish the selected target.
See the [maintenance workflow](../WORKFLOW_README.md) for the full procedure.
