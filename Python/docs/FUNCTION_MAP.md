# Public function map

Purpose: name every public entry point of the library partition, the input
contract each one honours, and the compatibility imports a reorganization keeps
available.

Reader: anyone looking up which module owns an operation, and the reviewer who
must decide whether a new entry point belongs in this table. Shared vocabulary
is in [terminology](../../docs/terminology.md); the document classes in
[the writing standard](../../docs/writing-standard.md).

The [scientific data contracts](../../docs/foundations.md) define shared time,
unit, mask, WCS and uncertainty behavior. Numerical and coordinate operations
never use plotting fallbacks to manufacture valid observations.

## Purpose and boundary

| Owned here | Belongs elsewhere |
| --- | --- |
| The public module inventory and the responsibility of each area | Scientific method and validity: [scientific data contracts](../../docs/foundations.md) |
| The input and output contract of each listed entry point | Command syntax, interfaces and frontends: [Apps guide](../../Apps/README.md) |
| Compatibility imports and their canonical replacements | Partitions, runtime layout and allowed roots: [ARCHITECTURE](../../ARCHITECTURE.md) |

Scientific configuration objects accept explicit caller inputs. Real event
parameters remain private. Application servers, CLIs and workflow orchestration
are owned by `Apps/solar_apps`, with dependency direction
`solar_apps -> solar_toolkit`. Imports do not start an interface, download data
or discover workstation events.

## Layout

| Area | Public modules | Responsibility |
| --- | --- | --- |
| Foundation | `solar_toolkit.time`, `io`, `data`, `map`, `timeseries` | Parsing, discovery, explicit I/O, coordinate-neutral array/table helpers |
| Image time matching | `time.precision`, `time.sequence`, `radio.time_matching` | Exact UTC/TAI clocks, fixed chronological hierarchy, logical FITS planes and strict polarization pairing |
| Modeling | `solar_toolkit.modeling.gaussian`, `radio.source_geometry` | Reusable Gaussian models and source geometry |
| PFSS | `solar_toolkit.modeling.pfss` | Global-field computation and result-bundle validation; solving and tracing need optional backend dependencies |
| NLFFF | `solar_toolkit.modeling.nlfff` | SHARP vector boundary checks and regular local grid reads |
| AIA/HMI | `solar_toolkit.aia`, `hmi` | FITS selection, normalization, differences, mosaics and magnetogram calculations |
| CME/network | `solar_toolkit.cme`, `net` | Explicit file processing, queries and downloads |
| X-ray | `solar_toolkit.xray_dem.hxi`, `processing`, `sxr`, `brightness_temperature` | Reusable readers, AIA reference geometry, precomputed brightness-temperature display and numerical processing |
| Radio core | `solar_toolkit.radio` | FITS metadata, Gaussian fitting, spectra, ROI statistics, trajectories and diagnostics |
| Radio split modules | `radio.reprojection`, `radio.cso_processing`, `radio.physical_diagnostics` | Reprojection, CSO array processing and pure physical diagnostics |
| DART/CSO | `radio.dart_spectrogram`, `radio.cso`, `radio.cso_extract`, `radio.cso_window` | Spectrum readers, format helpers and bounded numerical processing |
| MUSER | `radio.muser`, `radio.muser_comparison`, `radio.muser_localization`, `radio.raw_comparison` | Explicit image/spectrum readers, time matching, native-pixel fits and comparison helpers |
| Spectrum display | `radio.spectrum_display` | Relative backgrounds and optional missing-column compaction with original UTC mappings |
| Quality | `radio.quality_science`, `radio.quality_ml`, `radio.quality_autoencoder` | Rule evidence and optional learned diagnostics; ML and Torch dependencies are separate profiles |
| Visualization | `solar_toolkit.visualization` | Reusable plotting, frame, media and deterministic image-filename helpers |

### Compose explicit inputs

Choose a reader, normalize its time/coordinate conventions, then pass its
arrays or tables into a numerical helper. Plotting and writing outputs are
separate caller decisions; importing a module does not start a workflow.

| Operation | Input contract | Output and failure behavior |
| --- | --- | --- |
| `time.normalize_time` | A complete timestamp, explicit time scale where applicable, or a numeric value with explicit format | Source text/scale, true UTC and continuous integer TAI nanoseconds. Incomplete observation times raise `ValueError`; legacy `datetime` wrappers remain microsecond-limited. |
| `time.build_sequence_match_plan` | Independent `TimeStream` sequences; optional original-UTC `time_start`/`time_end`, tolerances and explicit offsets | A fixed full-coverage hierarchy ordered by increasing filtered frame count. The largest stream provides output UTC/row count; references never go backwards. Bounds are half-open; over-tolerance selections remain flagged. See [time matching](../../docs/time_matching.md). |
| `radio.time_matching.index_radio_frames` / `pair_radio_frames` / `materialize_radio_frame` | Explicit FITS files and logical frequency/Stokes selection; optional pairing window and tolerance | Exact file/HDU/plane references and UTC. Strict RR/LL members are filtered before one-to-one pairing; changed input versions are rejected when materialized. |
| `radio.muser_comparison.nearest_index` | One-dimensional finite, nondecreasing UTC Unix seconds; finite target in seconds; finite nonnegative tolerance in seconds | Original index or `None` if empty/outside tolerance. Invalid inputs raise `ValueError`; no implicit sorting. Ties retain the earlier candidate. |
| `xray_dem.sxr.load_sxr_data` | User-owned CSV/TXT or NetCDF path; optional start and/or end in UTC (offset-aware inputs are converted) | CSV/TXT returns a DataFrame with normalized `obs_time`; NetCDF returns a loaded Dataset detached from its file. Bounds are inclusive; omitted bounds are open. Reversed/invalid bounds raise `ValueError`. |
| `timeseries.normalize_time_column` / `crop_time_range` | A DataFrame and explicit time-column name; parseable timestamps | A copied table with UTC-normalized times or selected rows, preserving row order. Naive timestamps are interpreted as UTC. |
| `modeling.gaussian.elliptical_gaussian_2d` | NumPy coordinate arrays with broadcast-compatible shapes; center/widths in coordinate units; rotation in radians | Model values on the broadcast grid; amplitude uses caller-selected intensity units. No background term, beam correction or parameter validation is applied. |
| `map.create_aia_submap` | Four finite arcsecond bounds `(xmin, xmax, ymin, ymax)`, strictly increasing on both axes | A full rectangular submap by default. Explicit `legacy_roi=True` reproduces the historical horizontal strip. |
| `map.normalize_aia_exposure` | Explicit DN/count/photon integrated units or rates, optionally per pixel; integrated inputs need finite positive exposure in seconds | Integrated values become rates with original exposure metadata retained and mask copied. Already-normalized rates are unchanged; invalid units/exposure raise `ValueError`. |
| `map.get_map_obs_time` | A map or FITS header with complete observation time; date-only headers need `TIME-OBS` or `TIME_OBS` | Microsecond-resolution naive UTC, converting TAI/`TIMESYS`. Missing metadata raises `KeyError`; incomplete/invalid times raise `ValueError`. |
| `cme.running_difference` | Identically shaped arrays, optionally masked or unit-bearing | Floating subtraction in the current image's unit (or the previous image's unit if only it declares one), preserving negative changes and NaN masks. Shape mismatch and incompatible units raise rather than broadcasting or stripping dimensions. |
| `xray_dem.processing.calculate_derivative` | One-dimensional flux; optional equally sized, strictly increasing timestamp axis; optional finite positive scalar spacing in seconds | `median-gradient`, `gradient` or `forward` differences. Naive times mean UTC; invalid methods/axes/spacing raise `ValueError`. |
| `radio.muser.read_muser_spectrum` | A caller-selected FITS file with explicit `PWR_I`, frequency extension in MHz and UTC Julian-day times | A `MuserSpectrum` with frequency-by-time data, sorted frequency coordinates, `Time` values and the declared unit. Axis/shape mismatches raise `ValueError`; no XX/YY-to-circular-polarization conversion. |
| `radio.muser.read_muser_images` | A combined FITS image with explicit Stokes I, frequency and beam extensions | Frequency, copied 2D image, celestial header and beam for each frequency plane. Unsupported Stokes/shape declarations raise `ValueError`. |
| `io.scan_fits` / `io.resolve_fits_path` / `io.materialize_fits` | Local FITS image or table inputs, including gzip; optional private temporary directory for random access | Discovery prefers the plain file when both representations exist. Explicit paths take priority. The context manager yields a seekable local FITS path and cleans up temporary files on exit. |
| `radio.cso_extract.extract_native_polarization` | Floating linear-SFU RCP/LCP data, explicit UTC interval, MHz bounds and frequency-bin width | Per native time and frequency bin: paired sums, count, mean total intensity in SFU and polarization in percent. Returns integer TAI nanoseconds and coverage metadata. Conflicting records at the same instant raise an error. |
| `radio.cso_extract.extract_spectrum` | Explicit FITS paths, inclusive UTC Unix-second/MHz bounds, quantized/native selection and binning options | Frequency-by-time spectrum arrays, sample counts and metadata; empty bins remain NaN. Identical timestamps retain the first source column. Floating SFU products use RCP+LCP. Eight-bit products allow RCP or RCP+LCP and retain encoded-intensity units, with no SFU conversion. |
| `radio.cso_window.read_cso_total_window` | Explicit CSO FITS paths with UTC, RCP/LCP and linear SFU declarations; Unix-second/MHz bounds and bin widths | Legacy full-width bins may extend past requested upper bounds. Time/frequency bin centers, a finite-sample RCP+LCP mean and metadata; empty bins remain NaN. Path-ordered overlapping files raise `ValueError` and require caller-selected deduplication. |
| `radio.spectrum_display.relative_background_db` | A frequency-by-time array, increasing Unix-second times, an optional maximum bracketing gap and background percentile | Relative dB intensities and provenance. Optional short-gap estimation uses linear intensity; original UTC stays unchanged. |
| `radio.spectrum_display.compact_spectrum_time` | Spectrum arrays with increasing `time_edges_unix` and an explicit target UTC Unix second | Display array/edges, retained original indices, cursor and metadata. All-empty columns are omitted only for display. Missing/outside cursor times raise unless explicitly allowed, in which case the cursor is omitted. |
| `radio.cso_comparison.read_cso_comparison` | Explicit linear-SFU RCP/LCP FITS files, half-open Unix-second bounds, frequency bands and processing budget | R+L spectrum and native-sample light curves with finite-sample counts. Empty bins stay NaN; overlapping files are rejected. Chunk sizes do not change numerical results. |
| `radio.raw_comparison.prepare_muser` / `prepare_aia` / `prepare_dart` | Explicit image paths, target UTC Unix seconds, tolerances, frequency and polarization selection | Native image/header and actual source-time metadata, or a diagnostic NO DATA result. Missing times are not inferred from filenames; complete circular-polarization pairs are required when requested. |
| `radio.muser_localization.fit_radio_plane` | Native 2D image, explicit HPLN/HPLT WCS and half-open pixel ROI | Gaussian/background parameters, projected center and separate fit/quality diagnostics. Apparent widths are not beam-deconvolved; formal covariance does not include calibration uncertainty. |
| `xray_dem.brightness_temperature.AiaMapGeometry` / `load_brightness_temperature` / `brightness_temperature_extent` | Caller-selected AIA reference HDU with explicit helioprojective WCS, precomputed 2D Kelvin array and explicit pixel-center bounds | WCS bounds in arcseconds, UTC, a copied MK array preserving missing values, and pixel-edge extent. `fill_missing` explicitly enables display filling. These helpers do not perform DEM inversion. |
| `radio.fieldline_association.fieldline_band_segments` | Carrington Cartesian field line, PFSS outer radius, frequency band, harmonic and either a Newkirk multiplier or sampled Gauss magnitudes | Conditional band intervals with 3D points, frequencies and provenance back to the original edges. |
| `modeling.nlfff.read_sharp_cea_boundary` / `read_nlfff_grid` | Three co-registered SHARP CEA components or an explicit local-volume NPZ | Validated Gauss data, boundary validity mask or grid spacing in Mm. See [magnetic models](../../docs/magnetic_models.md) for coordinate and file conventions. |
| `map.coordinates.hpc_pixel_coordinates` / `hpc_pixel_grid` / `hpc_world_to_pixel` | Explicit HPLN/HPLT or SOLX/SOLY FITS axes; zero-based pixel centers or arcsecond coordinates | Astropy WCS transforms including angular units and axis order. `hpc_world_bounds` gives a display bounding box, not a pixel-to-world mapping. |
| `map.align_maps_to_reference` | Source map and target WCS; `return_footprint=False`, `uncertainty_policy="raise"` | New map, optionally with coverage. Source acquisition metadata and target coordinate reference metadata are kept distinct. Unsupported uncertainty propagation raises unless explicit `"drop"` is selected. |
| `radio.gaussian.fit_elliptical_gaussian_on_radio_image` | Native 2D pixel array; optional `coordinate_header` for helioprojective WCS | Pixel fit retained; WCS center, local angular FWHM and exact mapped half-maximum display curve when a header is supplied. Missing or masked pixels do not enter the fit. |
| `hmi.process_hmi_magnetic_field` | Magnetogram, nonnegative scalar magnetic threshold and Gaussian sigma in pixels | A new map excluding nonfinite and explicitly masked samples, with missing pixels and a copied mask retained. Input data/metadata are unchanged. Missing BUNIT assumes Gauss on output only; finite unmasked inputs retain the original Gaussian result. |
| `hmi.run_overlay_workflow` / `run_magnetogram_workflow` | Caller-selected paths and explicit ROI `(xmin, ymin, xmax, ymax)` in arcseconds | Rendered products; absent ROI fails before file access. This ROI order differs from `map.create_aia_submap`. |

The [radio composition](../examples/radio/README.md) connects a synthetic FITS
reader to time matching and missing-column handling. The
[SXR composition](../examples/sxr/README.md) connects a reader to smoothing,
time derivatives and plotting; flux is in W m^-2 and its derivative in
W m^-2 s^-1. These examples exercise software contracts, not observation
calibration or feature identification.

## Dependency directions

A listed module is imported by the application layer, never the reverse:

```text
solar_apps -> solar_toolkit
```

The forbidden direction is `solar_toolkit -> solar_apps`. Nothing in this
table starts an interface, downloads data or discovers a workstation event, and
no application workflow is listed here.

## Run and verify

This page is reference material: its rows are contracts, not steps. The
environment requirement and the commands that exercise them have one home.
Install and verify the library with Miniforge `solarphysics_env_latest` and
Python 3.12 or later, following [the library guide](../README.md#install) and
[its verification commands](../README.md#verify); the library-specific rules
are in [the Python contribution guide](../CONTRIBUTING.md).

Optional dependency profiles are listed in the [library guide](../README.md#install).
PFSS backend and input requirements are documented in its
[module guide](../solar_toolkit/modeling/pfss/README.md).
See the [migration guide](../../docs/migration.md#scientific-input-validation)
for stricter scientific input handling and explicit compatibility choices.

## Rules that must not be violated

1. A module is imported as `solar_toolkit.<module>`; only public entry points
   appear in the tables above.
2. Every listed entry point takes its inputs explicitly: no filename-inferred
   times, no implicit sorting and no hidden download.
3. Numerical and coordinate operations never use a plotting fallback to
   manufacture a valid observation.
4. Documented input failures raise; a listed `ValueError` or `KeyError` is not
   converted into a silent default.
5. A compatibility import stays importable and keeps warning with
   `SolarToolkitDeprecationWarning`; documentation cleanup never drops it.
6. Scientific configuration objects accept caller inputs. Real event parameters
   and machine paths never appear in this table.

### Compatibility imports

The following imports remain available and warn with
`SolarToolkitDeprecationWarning`. New code uses the replacement API;
documentation and structural cleanup preserve the compatibility imports.

| Compatibility import | Canonical replacement |
| --- | --- |
| `solar_toolkit.coordinates` | `solar_toolkit.map.coordinates` |
| `solar_toolkit.cso` | `solar_toolkit.radio.cso` |
| `solar_toolkit.gaussian` | `solar_toolkit.modeling.gaussian` |
| `solar_toolkit.solar_analysis_utils` | The corresponding `time`, `io`, `map`, `hmi`, or `visualization` API |

## Related documents

- [Python/README.md](../README.md) — library boundary, install and verification.
- [Python/CONTRIBUTING.md](../CONTRIBUTING.md) — library test, API and distribution rules.
- [Python/solar_toolkit/modeling/pfss/README.md](../solar_toolkit/modeling/pfss/README.md) — PFSS backend and input requirements.
- [Apps/README.md](../../Apps/README.md) — the application layer that composes these modules.
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — partition map and dependency directions.
- [docs/migration.md](../../docs/migration.md#scientific-input-validation) — stricter scientific input handling.
