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
