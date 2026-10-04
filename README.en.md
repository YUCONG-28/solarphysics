# Solar Physics Toolkit

[中文](README.md) | [English](README.en.md)

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

A composable foundation for solar physics code. `solar_toolkit` provides data
readers, time matching, image processing, geometry, radio analysis and plotting.
Choose modules and combine them around your own inputs, parameters and
scientific question.

A small set of examples and application adapters demonstrates composition.
Users supply observations, event parameters, processing conditions and
validation for their own tasks.

## Start with the library

The scientific library supports Python 3.10 and later. Repository development
uses the Miniforge `solarphysics_env_latest` environment. Install from the
repository root:

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pip install -e ./Python
```

On Windows, use `<miniforge-root>\Scripts\conda.exe` for the same command.
The library can be used without installing the application layer:

```python
from datetime import datetime, timedelta, timezone

from solar_toolkit.radio.reprojection import nearest_time_index

target = datetime(2000, 1, 1, tzinfo=timezone.utc)
samples = [target - timedelta(seconds=1), target + timedelta(seconds=2)]
index = nearest_time_index(target, samples)
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

## Application demonstration

`Apps` provides interfaces and workflow composition around the scientific
modules. It requires Python 3.14; see the [application guide](Apps/README.md)
for installation and setup. Start the desktop demonstration with:

```bash
./Apps/run.sh frontend app-v1
```

On Windows, use `Apps\run.ps1 frontend app-v1`. Supply explicit inputs,
configuration and allowed paths. Installation does not include real task
parameters or observations.

## Repository layout

| Directory | Contents |
| --- | --- |
| `Python/` | Independently installable scientific library, tests and small examples |
| `Apps/` | Application demonstrations, interface adapters, workflows and tests |
| `tools/` | General maintenance and collection-manifest utilities |
| `environment/` | Platform-specific dependency locks and replay instructions |
| `docs/` | Interface, architecture and development navigation |
| `Local/` | Ignored private configuration, data, state and outputs |

Public source contains code and necessary technical documentation. Research
notes, literature selections, progress, real event parameters, observations and
results belong in `Local/` or outside the repository.

Dependencies flow from `solar_apps` to `solar_toolkit`. Importing the library
does not launch interfaces, download data or run analyses. See the
[architecture](ARCHITECTURE.md), [contribution guide](CONTRIBUTING.en.md) and
[documentation index](docs/README.md).

## Contributing and maintenance

Read the [contribution guide](CONTRIBUTING.en.md) before making changes.
Issues and pull requests should use minimal synthetic inputs and describe
expected behavior, actual behavior and the relevant environment. Project
metadata defines installation and test dependencies; the
[environment guide](environment/README.md) explains platform-lock scope.

The [development workflow](WORKFLOW_README.md) covers branches, commits and
separately invoked release tooling.

## Licensing and citation

The scientific library uses the [MIT license](Python/LICENSE). Application
code and resources retain their component licenses; see
[Apps/LICENSE](Apps/LICENSE), the
[desktop application license](Apps/solar_apps/frontends/app_v1/LICENSE.md) and
[media notice](Apps/solar_apps/ui/media/NOTICE.txt). There is no single MIT
license covering the entire repository.

Scientific-library citation metadata is in [CITATION.cff](Python/CITATION.cff).
