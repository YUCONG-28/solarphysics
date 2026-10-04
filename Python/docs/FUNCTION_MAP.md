# Public function map

| Area | Public modules | Responsibility |
| --- | --- | --- |
| Foundation | `solar_toolkit.time`, `io`, `data`, `map`, `timeseries` | Parsing, discovery, explicit I/O, coordinate-neutral array/table helpers |
| Modeling | `solar_toolkit.modeling.gaussian`, `radio.source_geometry` | Reusable Gaussian models and source geometry |
| PFSS | `solar_toolkit.modeling.pfss` | Global-field computation and result-bundle validation; solving and tracing need optional backend dependencies |
| AIA/HMI | `solar_toolkit.aia`, `hmi` | FITS selection, normalization, differences, mosaics and magnetogram calculations |
| CME/network | `solar_toolkit.cme`, `net` | Explicit file processing, queries and downloads |
| X-ray | `solar_toolkit.xray_dem.hxi`, `processing`, `sxr` | Reusable readers and numerical processing |
| Radio core | `solar_toolkit.radio` | FITS metadata, Gaussian fitting, spectra, ROI statistics, trajectories and diagnostics |
| Radio split modules | `radio.reprojection`, `radio.cso_processing`, `radio.physical_diagnostics` | Reprojection, CSO array processing and pure physical diagnostics |
| DART/CSO | `radio.dart_spectrogram`, `radio.cso`, `radio.cso_extract`, `radio.cso_window` | Spectrum readers, format helpers and bounded numerical processing |
| MUSER | `radio.muser`, `radio.muser_comparison` | Explicit image/spectrum readers, time matching and comparison helpers |
| Spectrum display | `radio.spectrum_display` | Relative backgrounds and optional missing-column compaction with original UTC mappings |
| Quality | `radio.quality_science`, `radio.quality_ml`, `radio.quality_autoencoder` | Rule evidence and optional learned diagnostics; ML and Torch dependencies are separate profiles |
| Visualization | `solar_toolkit.visualization` | Reusable plotting, frame, media and deterministic image-filename helpers |

Scientific configuration objects accept explicit caller inputs. Real event
parameters remain private. Application servers, CLIs and workflow orchestration
are owned by `Apps/solar_apps`, with dependency direction
`solar_apps -> solar_toolkit`. Imports do not start an interface, download data
or discover workstation events.

## Compose explicit inputs

Choose a reader, normalize its time/coordinate conventions, then pass its
arrays or tables into a numerical helper. Plotting and writing outputs are
separate caller decisions; importing a module does not start a workflow.

| Operation | Input contract | Output and failure behavior |
| --- | --- | --- |
| `radio.muser_comparison.nearest_index` | One-dimensional finite, nondecreasing UTC Unix seconds; finite target in seconds; finite nonnegative tolerance in seconds | Original index or `None` if empty/outside tolerance. Invalid inputs raise `ValueError`; no implicit sorting. Ties retain the earlier candidate. |
| `xray_dem.sxr.load_sxr_data` | User-owned CSV/TXT or NetCDF path; optional start and/or end in UTC (offset-aware inputs are converted) | CSV/TXT returns a DataFrame with normalized `obs_time`; NetCDF returns a loaded Dataset detached from its file. Bounds are inclusive; omitted bounds are open. Reversed/invalid bounds raise `ValueError`. |
| `timeseries.normalize_time_column` / `crop_time_range` | A DataFrame and explicit time-column name; parseable timestamps | A copied table with UTC-normalized times or selected rows, preserving row order. Naive timestamps are interpreted as UTC. |
| `modeling.gaussian.elliptical_gaussian_2d` | Equal-shaped coordinate grids; center/widths in grid units; rotation in radians | Model values on the grid; amplitude uses caller-selected intensity units. |
| `radio.muser.read_muser_spectrum` | A caller-selected FITS file with explicit `PWR_I`, frequency extension in MHz and UTC Julian-day times | A `MuserSpectrum` with frequency-by-time data, sorted frequency coordinates, `Time` values and the declared unit. Axis/shape mismatches raise `ValueError`; no XX/YY-to-circular-polarization conversion. |
| `radio.muser.read_muser_images` | A combined FITS image with explicit Stokes I, frequency and beam extensions | Frequency, copied 2D image, celestial header and beam for each frequency plane. Unsupported Stokes/shape declarations raise `ValueError`. |
| `radio.cso_extract.extract_spectrum` | Explicit FITS paths, UTC Unix-second interval, frequency bounds in MHz, quantized/native selection and binning options | Frequency-by-time spectrum arrays and metadata; empty bins remain NaN. Floating SFU products use RCP+LCP. Eight-bit products allow RCP or RCP+LCP and retain encoded-intensity units, with no SFU conversion. |
| `radio.cso_window.read_cso_total_window` | Explicit CSO FITS paths with UTC, RCP/LCP and linear SFU declarations; Unix-second/MHz bounds and bin widths | Time/frequency bin centers, a finite-sample RCP+LCP mean and metadata; empty bins remain NaN. Overlapping files raise `ValueError` and require caller-selected deduplication. |
| `radio.spectrum_display.relative_background_db` | A frequency-by-time array, increasing Unix-second times, an optional maximum bracketing gap and background percentile | Relative dB intensities and provenance. Optional short-gap estimation uses linear intensity; original UTC stays unchanged. |
| `radio.spectrum_display.compact_spectrum_time` | Spectrum arrays with increasing `time_edges_unix` and an explicit target UTC Unix second | Display array/edges, retained original indices, cursor and metadata. All-empty columns are omitted only for display. Missing/outside cursor times raise unless explicitly allowed, in which case the cursor is omitted. |

The [radio composition](../examples/radio/README.md) connects a synthetic FITS
reader to time matching and missing-column handling. The
[SXR composition](../examples/sxr/README.md) connects a reader to smoothing,
time derivatives and plotting; flux is in W m^-2 and its derivative in
W m^-2 s^-1. These examples exercise software contracts, not observation
calibration or feature identification.

Optional dependency profiles are listed in the [library guide](../README.md#install).
PFSS backend and input requirements are documented in its
[module guide](../solar_toolkit/modeling/pfss/README.md).

## Compatibility imports

The following imports remain available and warn with
`SolarToolkitDeprecationWarning`. New code uses the replacement API;
documentation and structural cleanup preserve the compatibility imports.

| Compatibility import | Canonical replacement |
| --- | --- |
| `solar_toolkit.coordinates` | `solar_toolkit.map.coordinates` |
| `solar_toolkit.cso` | `solar_toolkit.radio.cso` |
| `solar_toolkit.gaussian` | `solar_toolkit.modeling.gaussian` |
| `solar_toolkit.solar_analysis_utils` | The corresponding `time`, `io`, `map`, `hmi`, or `visualization` API |

See [public API examples](../examples/public_api/README.md) for explicit inputs
and [the library guide](../README.md) for installation and verification.
