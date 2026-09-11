# Solar Physics App 1.0

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

Solar Physics App 1.0 is the primary deliverable of this repository: a native
PyQt6 desktop application that orchestrates reusable solar-observation
workflows for AIA/HMI, radio, X-ray/DEM, and media processing. The repository
also contains the supporting `solar_toolkit` Python library, literature
evidence, and maintenance tooling.

## Quick start

### macOS / Linux

```bash
./Apps/run.sh frontend app-v1
```

### Windows

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1
```

First-time setup initializes the private runtime and creates the fail-closed
path configuration:

```bash
./Apps/run.sh admin init
# then edit Local/configs/paths.local.yaml and add apps.allowed_roots
```

For a machine-local `solarphysics` command that works from any directory, see
[Apps/README.md](Apps/README.md). This shortcut is not part of the repository.

## App 1.0 at a glance

The native application ships twelve PyQt6 modules:

`Workbench`, `Data Download`, `Radio Workspace`, `Image Viewer`,
`Image Composer`, `Bad Frame Review`, `Source Map`, `DART Spectrogram`,
`ROI Light Curve`, `Radio Composite`, `Source Trajectory`, `Global PFSS`.

Launch one module directly:

```bash
./Apps/run.sh frontend app-v1 --module image-composer
```

The public command hierarchy is `frontend`, `workflow`, `admin`, and `tools`.
The complete application manual is [Apps/README.md](Apps/README.md).

## Current capabilities

- **STEREO EUVI:** Workbench produces single images, overviews and ROI movies.
  Optional official SolarSoft/IDL `secchi-prep` calibration saves verified FITS
  and provenance; the default `legacy` mode is not full SECCHI calibration.
- **Global PFSS:** inspect verified result bundles, magnetic field lines and
  conditional radio-source diagnostics in the native viewer. Computation uses
  a global radial-field boundary and separately installed optional dependencies;
  the viewer does not launch remote jobs or require the solver.
- **Radio diagnostics:** preserve precise observation times and distinguish
  projected heights from model-dependent three-dimensional constraints.
  Conditional associations do not establish a unique source location.

See the [application guide](Apps/README.md) and
[PFSS scientific contract](Python/solar_toolkit/modeling/pfss/README.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `Apps/` | App 1.0, legacy frontends, workflows, and tests — the main deliverable |
| `Python/` | Reusable `solar_toolkit` scientific library; no GUI or application state |
| `Paper/` | Static literature evidence and publication metadata |
| `tools/` | Literature retrieval/validation and repository maintenance tools |
| `environment/` | Sealed environment-lock and replay documentation |
| `Local/` | Ignored private runtime: configuration, state, workspaces, outputs, logs |

## Supporting code

- [Documentation index](docs/README.md) — guides, examples, and maintenance boundaries.
- [STEREO EUVI example](Python/examples/stereo/README.md) — local FITS input and reproducible ROI plots.
- [`Python/README.md`](Python/README.md) — library boundary, install, and verification.
- [`Paper/README.md`](Paper/README.md) — literature catalog and update rules.
- [`tools/literature/README.md`](tools/literature/README.md) — catalog retrieval and validation.
- [`environment/README.md`](environment/README.md) — environment lock and fresh-replay procedure.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — source, dependency, data, and runtime boundaries.
- [`WORKFLOW_README.md`](WORKFLOW_README.md) — save, branch, test, push, and release workflow.

## Development

Run all App tests with the primary Miniforge environment:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pytest Apps\tests -q
```

On macOS, use `Apps/run.sh` or the environment interpreter directly. Library
checks live under `Python/tests`; see [`Python/README.md`](Python/README.md).

Offline app health matrix:

```bash
./Apps/run.sh tools health --output Local/tmp/apps-health.json
```

## License and citation

MIT License. Citation metadata for the reusable library is in
[`Python/CITATION.cff`](Python/CITATION.cff).
