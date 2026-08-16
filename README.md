# Solar Physics Toolkit

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

`solarphysics` is a research-oriented workspace built around the reusable
`solar-physics-toolkit` Python library. The distribution installs the
`solar_toolkit` import namespace and provides focused building blocks for
solar-observation analysis.

> This repository is not an out-of-the-box data product. Scientific choices,
> event configuration, and local data paths remain explicit so analyses can be
> reviewed and reproduced.

## Install the library

The commands in this section resolve the current dependency ranges and are
appropriate for exploratory development. They are not an exact environment
lock. Formal runs must use a sealed, replay-tested platform lock; the current
gate and maintenance procedure are documented in the
[environment lock manual](environment/README.md).

From the repository root, install the reusable package in the primary
Miniforge environment:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[dev]"
```

`solarphysics_env` is retained as an explicit compatibility environment. The
workspace does not automatically fall back to another Python installation.

## Library capabilities

The toolkit provides focused components for observation discovery, FITS and
map processing, coordinates, time series, radio-source analysis, X-ray/DEM
workflows, visualization, and media export. Scientific choices and local data
locations remain explicit rather than being hidden in package defaults.

The public package covers:

- observation discovery, download helpers, and FITS I/O;
- solar maps, coordinates, time matching, and time-series utilities;
- radio imaging, source fitting, trajectories, spectrograms, and frame quality;
- AIA, HMI, CME, X-ray, and differential-emission-measure workflows;
- scientific visualization, deterministic image naming, and media export.

The library contains no GUI, Web server, event-specific path, or application
runtime state. See the [Python package reference](Python/README.md),
[quickstart](Python/docs/quickstart.md), and
[package organization](Python/CODE_ORGANIZATION_MANIFEST.md) for the detailed
API and dependency boundaries.

## Minimal example

```python
from solar_toolkit.time import extract_time_from_filename, nearest_by_time

observations = [
    (name, extract_time_from_filename(name))
    for name in (
        "aia.lev1_euv_12s.2024-01-10T062925Z.171.image_lev1.fits",
        "aia.lev1_euv_12s.2024-01-10T062937Z.171.image_lev1.fits",
    )
]
nearest = nearest_by_time(
    "2024-01-10T06:29:33Z",
    observations,
    key=lambda item: item[1],
    max_diff_seconds=12,
)
print(nearest[0] if nearest else "no match")
```

## Applications and research evidence

The versioned [`Apps`](Apps) partition contains the Miniforge-launched desktop,
Web, and Streamlit applications. It depends on the reusable library while
keeping workflow orchestration and user-interface code outside the package.

After following the [Apps manual](Apps/README.md), launch the stable native
Solar Physics App 1.0 from the repository root with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1
```

App 1.0 provides ten English PyQt6 interfaces for the Workbench, radio
workflows, image viewing and composition, time synchronization, projects,
presets, batch processing, and recoverable subprocess tasks. The
`app-v1-preview` command remains a one-release compatibility alias to the same
implementation. Earlier Flask, Streamlit, and PySide6 entry points remain
available as deprecated compatibility surfaces.

The [integration plan](Apps/docs/app-v1-integration-plan.md),
[capability matrix](Apps/docs/app-v1-capability-matrix.md), and
[interface-equivalence audit](Apps/docs/app-v1-equivalence-audit.md) document
the application boundary and release gates.

The [`Paper`](Paper) partition is the static literature-evidence layer.
Catalog retrieval and validation live under
[`tools/literature`](tools/literature). These components are kept separate from
both the Python library and application orchestration. Private configuration,
state, workspaces, and generated outputs remain under the ignored `Local/`
runtime tree.

See the [repository architecture](ARCHITECTURE.md) for the complete source,
dependency, data, and runtime boundaries.

## Development

For the standard save, branch, test, push, PR, merge, and cleanup workflow,
see the [project workflow guide](WORKFLOW_README.md). The guide is written in
Chinese and includes both macOS and Windows commands.

Run the complete application test suite with the primary Miniforge
interpreter:

```powershell
& $Conda run -n solarphysics_env_latest python -m pytest .\Apps\tests -q
```

Run public-package checks in the primary Miniforge environment:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip check
& $Conda run -n solarphysics_env_latest python -m compileall -q Python/solar_toolkit Python/tests
& $Conda run -n solarphysics_env_latest python -m ruff check Python/solar_toolkit Python/tests
& $Conda run -n solarphysics_env_latest python -m pytest Python/tests
```

## License and citation

The Python library and applications use the MIT License. Bundled third-party
assets retain their own notices. Citation metadata for the reusable library is
provided in [`Python/CITATION.cff`](Python/CITATION.cff).
