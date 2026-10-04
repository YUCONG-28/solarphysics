# Solar Physics Toolkit

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

`solar-physics-toolkit` is the reusable scientific-library partition of the
[`solarphysics`](https://github.com/YUCONG-28/solarphysics) repository. The
distribution name remains `solar-physics-toolkit`, the import namespace remains
`solar_toolkit`, and this partition is version 0.3.0.

The repository is a collection of composable modules. Select APIs for your own
inputs and scientific conditions; examples are small demonstrations of composition.

## Public boundary

The wheel contains reusable base, time, I/O, data, map, time-series, modeling,
visualization, AIA, HMI, CME, network, X-ray and radio computation modules.
The [function map](docs/FUNCTION_MAP.md) describes their roles, selected input
contracts and compatibility imports.

Explicit scientific configuration types, such as `RadioEventConfig`, are part
of the library. Real event parameters and machine paths are supplied by the
caller and remain outside public source. Application CLIs, GUI/Web servers,
browser launchers, static assets and workflow orchestration live in
[`../Apps`](../Apps/README.md), with dependencies flowing
`solar_apps -> solar_toolkit`. Private configuration, state and outputs belong
under ignored `../Local` or a user-selected private directory.
The library installs no console scripts and discovers no workstation events.

## Install

The editable commands below use dependency ranges. For a confirmatory run,
recreate and verify a sealed target-platform environment as described in the
[environment lock manual](../environment/README.md), then install this checkout
with `--no-deps --no-build-isolation`.

All shell commands in this guide run from the unified repository root.
On macOS/Linux:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pip install -e ./Python
```

On Windows:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Python
```

[`pyproject.toml`](pyproject.toml) is the canonical dependency definition.
Choose an optional profile only when the APIs you use need it:

| Extra | Purpose |
| --- | --- |
| `full` | Optional NetCDF, media, image, download and other science dependencies |
| `quality-ml` | Supervised radio-quality diagnostics with scikit-learn and joblib |
| `quality-autoencoder` | Optional CPU image-feature extraction with Torch |
| `dev` | Contributor test, formatting and build tools |

For example, add `[full]` to the editable path when reading NetCDF files:

```powershell
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[full]"
```

Radio quality APIs keep rule results, human labels and learned predictions
separate. `quality_science` supplies signed-asinh statistics, morphology and
cross-frame pre-screening without a fitted model. `quality_ml` accepts explicit
human `good` / `degraded` / `bad` labels, uses observation-batch splits, a
calibrated histogram gradient-boosting classifier, metadata OOD checks and
hash-verified model bundles, and leaves
human labels and final bad-frame lists unchanged. `quality_autoencoder` accepts
only human `good` images and returns reconstruction and latent-distance
features; it has no bad-frame classification API.

`solarphysics_env_latest` is the default environment for current development
and validation. The retained `solarphysics_env` environment is the formal
compatibility fallback and can be selected explicitly with Miniforge
`conda run -n solarphysics_env`; it is not updated or activated by default.

## Compose modules

Choose a reader, normalize time and coordinate conventions, then pass its
arrays or tables to a numerical helper. Library use supplies paths and
scientific configuration explicitly:

```python
from datetime import datetime, timedelta, timezone

from solar_toolkit.radio.config import RadioEventConfig
from solar_toolkit.radio.reprojection import nearest_time_index

event = RadioEventConfig.from_mapping(
    {"user": {"data": {"multi_band_freqs": [149.0, 164.0]}}}
)
target = datetime(2000, 1, 1, tzinfo=timezone.utc)
index = nearest_time_index(
    target,
    [target - timedelta(seconds=1), target + timedelta(seconds=2)],
)
```

| Example | Inputs and role |
| --- | --- |
| [Public API](examples/public_api/README.md) | Deterministic time matching and Gaussian arrays; no observations or generated files |
| [Radio](examples/radio/README.md) | Synthetic FITS reading, UTC matching and missing-column display |
| [SXR](examples/sxr/README.md) | A synthetic or caller-supplied table, smoothing, derivatives and plotting |
| [STEREO EUVI](examples/stereo/README.md) | Caller-selected FITS maps and an optional ROI |

Examples show software composition. Users choose calibration, scientific
conditions and result validation for their own data. Importing a library
module does not download data, start an interface or execute an analysis.

## Scientific image filenames

Automatically named scientific images use this deterministic contract:

```text
NNNN_START[-END]_[generated_]INSTRUMENT_[CHANNEL]_[POLARIZATION]_PRODUCT_[QUALIFIER].ext
```

`NNNN` is a four-digit sequence assigned in observation-time and declared
product order before parallel work starts. Times are UTC
`YYYYMMDDTHHMMSSZ`, truncated to whole seconds; interval figures use their
earliest and latest valid observation times, and difference figures use the
reference-to-current interval. If observation time is unavailable, a workflow
captures one UTC time at batch start and includes `generated` in every fallback
name from that batch.

Feature tokens are lowercase ASCII in instrument, channel/frequency,
polarization/Stokes, product, then processing-qualifier order. Examples include
`171a`, `223p5mhz`, `lcp`, `rcp`, `lcp_plus_rcp`, and `stokes_v_over_i`.
`solar_toolkit.visualization.image_naming` exposes `ImageFilenameSpec`,
`format_utc_filename_time()`, and `build_image_filename()` for the same
contract. Existing callers that pass a complete output filename keep that name;
directory-only automatic workflows generate a contract name. Legacy filename
constants and `format_time_for_filename()` remain importable for compatibility,
but automatic workflows do not use the legacy constants.

Examples:

```text
0001_20000101T000000Z_aia_171a_intensity.png
0002_20000101T000001Z_radio_223mhz_lcp_source_map.png
0001_20000101T000000Z-20000101T000100Z_dart_stokes_i_v_over_i_dynamic_spectrum.png
```

## Verify

Follow the repository [contributor setup](../CONTRIBUTING.en.md) to install the
development tools. Library-specific checks are described in
[Python contribution guidance](CONTRIBUTING.md). From the repository root:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip check
& $Conda run -n solarphysics_env_latest python -m compileall -q Python\solar_toolkit Python\tests Python\examples
& $Conda run -n solarphysics_env_latest python -m ruff check Python\solar_toolkit Python\tests Python\examples
& $Conda run -n solarphysics_env_latest python -m pytest Python\tests --basetemp "Local/tmp/pytest-library-<run>"
& $Conda run -n solarphysics_env_latest python tools/build_distributions.py --project Python --output-dir "Local/outputs/checks/library-<run>"
```

Replace `<run>` with a unique run identifier. Build output must be a new or
empty private directory. The build tool inspects the wheel and source archive,
rebuilds the wheel outside the checkout, and compares package and license
contents. CI also installs the rebuilt wheel in an independent Miniforge
environment and exercises the existing synthetic examples while blocking
application imports. These checks do not publish a release.

For macOS/Linux, use `"$MINIFORGE_CONDA" run -n solarphysics_env_latest` in
place of `& $Conda run -n solarphysics_env_latest` and use `/` in source paths.

`solar_toolkit/` holds the installable package, `tests/` holds data-independent
checks, and `examples/` holds the small compositions above. See the repository
[architecture](../ARCHITECTURE.md) for application and runtime ownership.

## License and citation

This Python partition is covered by [`LICENSE`](LICENSE). The repository root
does not impose that license on other source partitions. Citation
metadata is in [`CITATION.cff`](CITATION.cff).
