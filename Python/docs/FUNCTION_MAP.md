# Public function map

| Area | Public modules | Responsibility |
| --- | --- | --- |
| Foundation | `solar_toolkit.time`, `io`, `data`, `map`, `timeseries` | Parsing, discovery, explicit I/O, coordinate-neutral array/table helpers |
| Modeling | `solar_toolkit.modeling.gaussian`, `radio.source_geometry` | Reusable Gaussian models and source geometry |
| PFSS | `solar_toolkit.modeling.pfss` | Global-field computation and verified result bundles; solving and tracing need optional backend dependencies |
| AIA/HMI | `solar_toolkit.aia`, `hmi` | FITS selection, normalization, differences, mosaics and magnetogram calculations |
| CME/network | `solar_toolkit.cme`, `net` | Explicit file processing, queries and downloads |
| X-ray | `solar_toolkit.xray_dem.hxi`, `processing`, `sxr` | Reusable readers and numerical processing |
| Radio core | `solar_toolkit.radio` | FITS metadata, Gaussian fitting, spectra, ROI statistics, trajectories and diagnostics |
| Radio split modules | `radio.reprojection`, `radio.cso_processing`, `radio.physical_diagnostics` | Reprojection, CSO array processing and pure physical diagnostics |
| DART/CSO | `radio.dart_spectrogram`, `radio.cso` | Spectrum readers and bounded numerical processing |
| Quality | `radio.quality_science`, `radio.quality_ml`, `radio.quality_autoencoder` | Rule evidence and optional learned diagnostics; ML and Torch dependencies are separate profiles |

Application servers, event recipes, CLIs and workflow orchestration are owned by
the public `Apps/solar_apps` namespace and must depend on this public layer, never the
reverse.

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
| `radio.cso_extract.extract_spectrum` | Explicit FITS paths, UTC Unix-second interval, frequency bounds in MHz, and binning options | Spectrum arrays and extraction metadata; caller chooses polarization and subsequent processing. |

The [radio composition](../examples/radio/README.md) connects a synthetic FITS
reader to time matching and missing-column handling. The
[SXR composition](../examples/sxr/README.md) connects a reader to smoothing,
time derivatives and plotting; flux is in W m^-2 and its derivative in
W m^-2 s^-1. These examples exercise software contracts, not observation
calibration or feature identification.

## Compatibility imports

The following imports remain available during 0.x and warn with
`SolarToolkitDeprecationWarning`. New code uses the replacement API; removal is
planned for 1.0.0, not for a structural cleanup.

| Compatibility import | Canonical replacement |
| --- | --- |
| `solar_toolkit.coordinates` | `solar_toolkit.map.coordinates` |
| `solar_toolkit.cso` | `solar_toolkit.radio.cso` |
| `solar_toolkit.gaussian` | `solar_toolkit.modeling.gaussian` |
| `solar_toolkit.solar_analysis_utils` | The corresponding `time`, `io`, `map`, `hmi`, or `visualization` API |

See [public API examples](../examples/public_api/README.md) for explicit inputs
and [the library guide](../README.md) for installation and verification.
