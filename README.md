# Solar Physics Toolkit

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

Composable Python tools for multi-wavelength solar-physics analysis.
`solar_toolkit` provides readers, time matching, image processing, geometry,
radio analysis and plotting.
Choose modules and combine them around your own inputs, parameters and
scientific question.

A small set of examples and application adapters demonstrates composition.
Users supply observations, event parameters, processing conditions and
validation for their own tasks.

| Capability | Scope |
| --- | --- |
| Data and time | Explicit FITS/table I/O, UTC/TAI handling and chronological sequence matching |
| Solar images | AIA/HMI normalization, masks, map geometry and CME differences |
| Radio analysis | Spectrum readers, source fitting, reprojection, ROI statistics and trajectories |
| X-ray time series | SXR readers, smoothing, derivatives and precomputed brightness-temperature display |
| Models | Gaussian models; [global PFSS](Python/solar_toolkit/modeling/pfss/README.md) requires separately installed backends and a valid global radial boundary |
| Quality diagnostics | Rule evidence; learned diagnostics require explicit labels and separate [optional profiles](Python/README.md#install) |

The [API map](Python/docs/FUNCTION_MAP.md) describes inputs, units and failure
behavior. Users choose calibration and physical-model assumptions for their data.

<a id="从基础模块开始"></a>

## Start with the library

The scientific library supports Python 3.10 and later. Repository development
uses Python 3.14 in Miniforge’s `solarphysics_env_latest` environment. For a
fresh clone, replace `<miniforge-root>` with your Miniforge installation
directory and run from the repository root.

macOS/Linux:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" create -n solarphysics_env_latest -c conda-forge python=3.14 pip
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pip install -e ./Python
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
```

Windows PowerShell:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda create -n solarphysics_env_latest -c conda-forge python=3.14 pip
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Python
& $Conda run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
```

If the environment already exists, inspect its Python version and skip
`create`. The example needs no observations and prints:

```text
aia.lev1_euv_12s.2000-01-01T000012Z.171.image_lev1.fits
```

The library works without installing Apps. A direct nearest-time lookup returns
the selected input index:

```python
from datetime import datetime, timedelta, timezone

from solar_toolkit.radio.reprojection import nearest_time_index

target = datetime(2000, 1, 1, tzinfo=timezone.utc)
samples = [target - timedelta(seconds=1), target + timedelta(seconds=2)]
print(nearest_time_index(target, samples))  # 0
```

| Task | Entry point |
| --- | --- |
| Find, install and verify scientific modules | [Library guide](Python/README.md) |
| Find public interfaces and compatibility entry points | [API map](Python/docs/FUNCTION_MAP.md) |
| Start with small deterministic examples | [Public API examples](Python/examples/public_api/README.md) |
| Compose spectrum reading, time matching and display | [Radio example](Python/examples/radio/README.md) |
| Read user-supplied EUVI images | [STEREO example](Python/examples/stereo/README.md) |
| Compose SXR reading and curve plotting | [SXR example](Python/examples/sxr/README.md) |

Generated data and outputs belong in the repository's `Local/` directory or a
user-selected directory outside the repository. Synthetic inputs illustrate
interfaces and check software behavior; they are not observational evidence.

<a id="应用展示"></a>

## Application demonstration

`Apps` provides interfaces and workflow composition around the scientific
modules. It requires Python 3.14 and editable installs of both source partitions;
see the [application guide](Apps/README.md) for installation and setup. Start the
desktop demonstration with:

```bash
./Apps/run.sh frontend app-v1
```

On Windows, use `Apps\run.ps1 frontend app-v1`. Supply explicit inputs,
configuration and allowed paths. Installation does not include real task
parameters or observations.

<a id="仓库结构"></a>

## Repository layout

| Directory | Contents |
| --- | --- |
| `Python/` | Independently installable scientific library, tests and small examples |
| `Apps/` | Application demonstrations, interface adapters, workflows and tests |
| `tools/` | General maintenance and collection-manifest utilities |
| `environment/` | Platform-specific dependency locks and replay instructions |
| `docs/` | Technical guide index, time-matching and radio-comparison guides |
| `Local/` | Ignored private configuration, data, state and outputs |

Public source contains code and necessary technical documentation. Research
notes, literature selections, progress, real event parameters, observations and
results belong in `Local/` or outside the repository.

Dependencies flow from `solar_apps` to `solar_toolkit`. Importing the library
does not launch interfaces, download data or run analyses. See the
[architecture](ARCHITECTURE.md), [contribution guide](CONTRIBUTING.md) and
[documentation index](docs/README.md).

<a id="贡献与维护"></a>

## Contributing and maintenance

Read the [contribution guide](CONTRIBUTING.md) before making changes.
Issues and pull requests should use minimal synthetic inputs and describe
expected behavior, actual behavior and the relevant environment. Project
metadata defines installation and test dependencies; the
[environment guide](environment/README.md) explains platform-lock scope.

The [development workflow](WORKFLOW_README.md) covers branches, commits and
separate release stages. Existing users can consult the
[migration guide](docs/migration.md) and [changelog](Python/CHANGELOG.md).

<a id="许可与引用"></a>

## Licensing and citation

Self-owned root files, `tools/`, `docs/` and `environment/` use the root
[MIT license](LICENSE). The scientific library retains [Python/LICENSE](Python/LICENSE).
Apps includes MIT source, GPL-3.0-only App 1.0 source and an MPL-2.0 media asset;
see the [component license scope](LICENSING.md) before redistributing.

Citation metadata for Y. Li, Shandong University, is available in the root
[CITATION.cff](CITATION.cff); the library also provides
[partition citation metadata](Python/CITATION.cff). Cite the version or commit
you actually used. These files do not assign a DOI.
