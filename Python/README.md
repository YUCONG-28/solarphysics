# Solar Physics Toolkit

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

`solar-physics-toolkit` is the reusable scientific-library partition of the
[`solarphysics`](https://github.com/YUCONG-28/solarphysics) repository. The
distribution name remains `solar-physics-toolkit`, the import namespace remains
`solar_toolkit`, and this partition is version 0.3.0.

## Public boundary

The wheel contains reusable base, time, I/O, data, map, time-series, modeling,
AIA, HMI, CME, network, X-ray and radio computation modules. Radio computation
includes the explicit modules:

- `solar_toolkit.radio.reprojection`
- `solar_toolkit.radio.cso_processing`
- `solar_toolkit.radio.dart_spectrogram`
- `solar_toolkit.radio.physical_diagnostics`
- `solar_toolkit.radio.quality_autoencoder`
- `solar_toolkit.radio.quality_science`
- `solar_toolkit.radio.quality_ml`

Event configuration, machine-specific path configuration, CLIs, GUI/Web
servers, browser launchers, static assets and cross-domain orchestration are
intentionally not distributed in the library wheel. Their tracked source lives
in `../Apps` and depends on this package in the direction
`solar_apps -> solar_toolkit`; private runtime state remains under ignored
`../Local`.
There are no public console scripts in version 0.3.0.

## Install

The editable commands below use dependency ranges. For a confirmatory run,
recreate and verify a sealed target-platform environment as described in the
[environment lock manual](../environment/README.md), then install this checkout
with `--no-deps --no-build-isolation`.

From the unified repository root:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Python
```

The radio-quality pipeline deliberately keeps automatic rules, human truth and
machine-learning predictions separate. `quality_science` provides signed-asinh
statistics, morphology, cross-frame consistency and conservative
`good_candidate` / `bad_candidate` / `uncertain` pre-screening without a fitted
model. Supervised training is optional and accepts only explicit human
`good` / `degraded` / `bad` labels:

```powershell
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-ml]"
```

`quality_ml` uses observation-batch splits, an independently calibrated
histogram gradient-boosting classifier, metadata OOD checks and hash-verified
model bundles. It never changes a rule result, human label or final bad-frame
list; model publication remains an explicit Local-application action.

After enough human-confirmed good frames exist, the separate
`quality_autoencoder` extra can train a small CPU convolutional autoencoder:

```powershell
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-autoencoder]"
```

It accepts only human `good` images and emits reconstruction and latent-distance
features for the calibrated tree model. It has no bad-frame classification API.

`solarphysics_env_latest` is the default environment for current development
and validation. The retained `solarphysics_env` environment is the formal
compatibility fallback and can be selected explicitly with Miniforge
`conda run -n solarphysics_env`; it is not updated or activated by default.
That environment retains known Conda ownership warnings and lacks newer optional
packages including FITSIO and PyQtGraph, so workflows that need those packages
must use `solarphysics_env_latest`.

Library use always supplies paths and event configuration explicitly:

```python
from solar_toolkit.radio.config import RadioEventConfig
from solar_toolkit.radio.reprojection import nearest_time_index
from datetime import datetime, timedelta

event = RadioEventConfig.from_mapping(
    {"user": {"data": {"multi_band_freqs": [149.0, 164.0]}}}
)
target = datetime.fromisoformat("2025-01-24T04:48:30")
index = nearest_time_index(
    target,
    [target - timedelta(seconds=1), target + timedelta(seconds=2)],
)
```

See [`examples/public_api`](examples/public_api) for source-tree examples.

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
0001_20250124T044830Z_aia_171a_intensity.png
0002_20250124T044831Z_radio_223mhz_lcp_source_map.png
0001_20250124T044800Z-20250124T045000Z_dart_stokes_i_v_over_i_dynamic_spectrum.png
```

## Verify

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip check
& $Conda run -n solarphysics_env_latest python -c "import ssl; ssl.create_default_context(); import sunpy.map; from sunpy.net import Fido"
& $Conda run -n solarphysics_env_latest python -m compileall -q solar_toolkit tests examples\public_api
& $Conda run -n solarphysics_env_latest python -m ruff check solar_toolkit tests examples\public_api
& $Conda run -n solarphysics_env_latest python -m pytest tests
& $Conda run -n solarphysics_env_latest python -m build --wheel --no-isolation --outdir dist .
```

## License and citation

This Python partition is covered by [`LICENSE`](LICENSE). The repository root
does not impose that license on the separate Paper evidence layer. Citation
metadata is in [`CITATION.cff`](CITATION.cff).
