# Contributing

Reusable science APIs live in `solar_toolkit/`; data-independent library checks
live in `tests/`. The repository [contributor guide](../CONTRIBUTING.en.md) owns
environment setup, style checks, public-source review and the Git workflow.
All commands below run from the repository root through Miniforge
`solarphysics_env_latest`.

## Scientific APIs

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

## Library tests

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

Examples demonstrate a small number of module compositions with synthetic or
caller-selected inputs. Keep generated files in an explicitly selected private
directory and clear notebook outputs before committing. Scientific calibration
and feature identification remain caller responsibilities.

## Distribution checks

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
