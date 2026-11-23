# Contributing

Purpose: state the loop for changing the scientific library, what a delivery must
carry, and how to recover when a check fails.

Reader: a contributor changing `solar_toolkit/` or `tests/`. Environment setup,
style checks, public-source review and the Git workflow have one home, the
repository [contributor guide](../CONTRIBUTING.md); this page adds only the
library-specific steps.

Reusable science APIs live in `solar_toolkit/`; data-independent library checks
live in `tests/`. All commands below run from the repository root through
Miniforge `solarphysics_env_latest`.

## 1. Daily loop

Work API-first: change the module, change or add the data-independent check that
covers it, run that check, then broaden when shared behavior changed.

### Library tests

Use generated arrays, tables or tiny synthetic FITS/NetCDF fixtures. Tests must
not depend on external observations, workstation paths or an application
installation. Run the tests for the changed APIs first, then broaden when
shared behavior changes. For example:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pytest Python/tests/test_muser.py Python/tests/test_xray_dem_utils.py --basetemp "Local/tmp/pytest-library-<run>"
```

Replace `<run>` with a unique run identifier. On macOS/Linux, use the
`<miniforge-root>/bin/conda` executable with the same `run -n` arguments.
Install optional dependencies before validating their code paths: CI installs
`xarray` and `netCDF4` explicitly so the NetCDF cases run. A skipped optional
test is not evidence that its numerical or file-handling behavior passed.

## 2. Submit a reproducible change

### Scientific APIs

- Keep readers, numerical calculations and plotting helpers independently
  reusable. Application orchestration belongs in `../Apps/`.
- Require explicit inputs and document array shapes, units, UTC conventions,
  missing-data behavior and algorithm conditions.
- Preserve public exports and compatibility imports when reorganizing code;
  the [function map](docs/FUNCTION_MAP.md) lists the maintained entry points.
- Keep imports free of data downloads, interface startup and analysis execution.
- Select optional dependencies for the affected API. `full` supplies NetCDF,
  media and other optional science helpers; the two quality extras remain
  separate from `full`. See [installation profiles](README.md#install).

Examples demonstrate a small number of module compositions with synthetic or
caller-selected inputs. Keep generated files in an explicitly selected private
directory and clear notebook outputs before committing. Scientific calibration
and feature identification remain caller responsibilities.

### Distribution checks

Editable-source tests and installed-package checks answer different questions.
Use the [library verification commands](README.md#verify) to inspect the wheel
and source archive, rebuild outside the checkout, and compare package and
license contents. Choose a new or empty output directory under ignored `Local`
or outside the repository.

CI installs the rebuilt wheel into an independent Miniforge environment
without Apps and runs the four existing deterministic API/synthetic examples
through `tools/installed_library_smoke.py`. The check blocks application imports
and verifies that loaded library modules come from the installed environment.
Builds and validation outputs are private artifacts; these commands do not
publish releases.

## 3. Conflict and rollback

- A reorganization that collides with an existing import keeps the compatibility
  import. Documentation and structural cleanup never drop one.
- A check that fails for a missing optional dependency is a failed check, not a
  rollback signal: install the profile for the affected API and rerun.
- A distribution check refuses an output directory that is not new or empty.
  Select a new path under ignored `Local` instead of clearing an existing one.
- A change that cannot keep a public export is out of scope here; stop and raise
  it rather than rewriting the contract inside a cleanup.

## 4. Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| Collection fails on a missing optional import | The affected profile is not installed | Install `full`, `quality-ml` or `quality-autoencoder` for the API under change, then rerun |
| A test reports as skipped | An optional dependency is absent | Rerun with the dependency installed; the skip is not evidence that the behavior passed |
| A test needs an observation file or a workstation path | The fixture is not synthetic | Replace it with generated arrays, a table, or a tiny synthetic FITS/NetCDF fixture |
| The distribution check rejects the output directory | The directory already holds a build | Choose a new or empty directory under ignored `Local` or outside the repository |
| A consumer import is gone after a reorganization | The compatibility import was dropped | Restore it; the replacements are listed in the [function map](docs/FUNCTION_MAP.md) |
| Commands resolve an unexpected interpreter | The command did not run through Miniforge `solarphysics_env_latest` | Rerun through `conda run -n solarphysics_env_latest`; no system-Python fallback |

## 5. Pre-delivery checklist

- [ ] Focused tests for the changed APIs pass through `solarphysics_env_latest`,
      then the broader suite when shared behavior changed.
- [ ] Public exports and compatibility imports are unchanged or restored.
- [ ] Optional dependencies are installed for every code path touched; no skipped
      test is counted as passing.
- [ ] Distribution checks, when the delivery touches packaging, ran into a new or
      empty private directory.
- [ ] Generated files and notebook outputs are not committed.

## 6. Related documents

- [README.md](README.md) — library boundary, install and verification.
- [docs/FUNCTION_MAP.md](docs/FUNCTION_MAP.md) — public modules and their input and output contracts.
- [CHANGELOG.md](CHANGELOG.md) — the dated record of library changes.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — environment setup, style checks, public-source review and the Git workflow.
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — partitions and dependency directions.
- [../WORKFLOW_README.md](../WORKFLOW_README.md) — commit, review and release procedure.
