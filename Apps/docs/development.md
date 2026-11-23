# Apps development

Purpose: the application partition's development loop — prepare the environment,
run the checks, read the health report, and deliver a change.

Reader: anyone changing code, tests or documentation under `Apps`. Installation is
owned by the [Apps guide](../README.md#install); structure, formats and the
dependency rules by [Apps architecture](architecture.md).

## Daily loop

### Environment

Follow the [installation guide](../README.md#install) to create the primary
Miniforge environment and install both source partitions. Run commands from the
repository root. Install the development extras in that same environment:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[dev,quality-ml]"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Apps[dev]"
```

```bash
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m pip install -e "./Python[dev,quality-ml]"
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m pip install -e "./Apps[dev]"
```

Use `solarphysics_env` only for an explicitly selected compatibility comparison.
The launchers select the interpreter; subprocesses inherit it. Exploratory
installs resolve ranges. See the [environment guide](../../environment/README.md)
for platform-lock validation, artifact replay and source-binding maintenance.
Do not describe a source-hash refresh as a new environment capture or replay.

### Checks

Compile, lint, format-check and test the affected area before broader checks.
Use a unique private pytest temporary directory for each run.

Windows:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
$PytestTemp = Join-Path ".\Local\tmp" ("pytest-apps-" + [guid]::NewGuid().ToString("N"))
& $Conda run -n solarphysics_env_latest python -m compileall -q Apps/solar_apps Apps/tests Apps/examples
& $Conda run -n solarphysics_env_latest python -m ruff check Apps/solar_apps Apps/tests Apps/examples
& $Conda run -n solarphysics_env_latest python -m black --check Apps/solar_apps Apps/tests Apps/examples
& $Conda run -n solarphysics_env_latest python -m pytest Apps/tests --basetemp $PytestTemp
```

macOS:

```bash
pytest_tmp="Local/tmp/pytest-apps-$(date +%s)-$$"
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m compileall -q Apps/solar_apps Apps/tests Apps/examples
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m ruff check Apps/solar_apps Apps/tests Apps/examples
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m black --check Apps/solar_apps Apps/tests Apps/examples
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m pytest Apps/tests --basetemp "$pytest_tmp"
```

Documentation changes should check relative links and the documentation/privacy
contracts. Changes to shared platform, UI, CLI or workflows also require the
Apps suite. Scientific behavior changes require the relevant library checks.
Use deterministic synthetic inputs; keep raw verification output private.

Inspect supported commands with the public launchers:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1 --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 workflow radio --help
```

```bash
./Apps/run.sh frontend app-v1 --help
./Apps/run.sh workflow radio --help
```

For UI changes, exercise the affected native page or supervised local server.
Verify Light, Dark and Auto while changing the operating-system color scheme;
verify Dark Dimmed on native pages that support it. Stop every supervised server
and confirm that its port is closed when finished.

### Health reports

The offline health matrix checks catalog-driven frontend entries, native pages
and offline smokes. Legacy Flask/Streamlit servers are reported as `not_run` with
a structured reason. Use `--include-legacy-servers` only when local loopback
ports are available. The report is written atomically; overall failure returns
a non-zero exit code. CI uses the default offline matrix.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 tools health --output .\Local\tmp\apps-health.json
```

```bash
./Apps/run.sh tools health --output Local/tmp/apps-health.json
```

## Submit a reproducible change

The optional `tools quick` and `tools release` commands expose repository
maintenance operations. Inspect their help and use the
[development workflow](../../WORKFLOW_README.md) for save, update and release
procedures. `release prepare` creates version changes for a normal pull
request; `release publish` validates and publishes an explicitly selected merged
commit as a separate stage. Both stages preview their actions by default.

Hold a change to these boundaries:

- Preserve dependency direction, public imports, stable CLI IDs and compatible
  artifact formats. See [Apps architecture](architecture.md).
- Keep reusable calculations in `solar_toolkit`; interfaces and composed
  workflow orchestration belong in `solar_apps`.
- Keep private scientific configuration separate from machine path
  authorization. Do not add observation defaults.
- Add focused tests for shared behavior and retain meaningful assertions.
- Keep observations, personal paths, state, workspaces, logs, verification
  receipts and generated research products under `Local/` or an external
  private runtime root.
- Keep UI theme state out of scientific sidecars and cache signatures.
- Keep third-party assets with their license and notice files. Required package
  resources are explicitly listed in `Apps/pyproject.toml`; update that list
  and the distribution boundary checks together when a resource changes.

Apps uses editable installs of both source partitions from the checkout.
Distribution builds check source archives, package resources and license
contents; they do not establish a standalone Apps wheel deployment. Library
wheel installation is tested separately.

The public-source policy, package-content checks and contribution workflow are
shared with the repository. Do not publish raw operation history or execution
receipts in documentation.

## Conflict and rollback

The `tools release` stages are read-only unless `--execute` is supplied
(`Apps/solar_apps/cli/release.py`), so every stage is inspected before it acts:

- A version change: run the stage, read the preview, and repeat the same command
  with `--execute` only when the preview matches the intention. A preview that is
  abandoned leaves the prepared change unexecuted, and `prepare --execute`
  requires a clean feature branch.
- An existing environment: inspect it first, and use `env update` in place of
  `env create` from the [installation guide](../README.md#install) only when you
  intend to update that environment.
- A failed health run: the report is written atomically, so a non-zero exit code
  does not leave a partially written report. Fix the reported entry and rerun the
  matrix.
- Two concurrent test runs: each run needs its own unique private base-temporary
  directory, so a collision means the directory was reused rather than a defect
  in the change.

## Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| `solar_apps` or `solar_toolkit` cannot be imported | One source partition is missing from the selected environment | Install both partitions in editable mode in `solarphysics_env_latest`. |
| A compile, Ruff or Black check fails | The affected files were not formatted, or a syntax error is present | Run the same command on the affected paths and fix what it reports. |
| A test run collides with another run | Both runs share one base-temporary directory | Rerun with a unique private directory. |
| The health matrix reports legacy servers | The default matrix is offline | Add `--include-legacy-servers` only when local loopback ports are available. |
| `tools health` exits non-zero | At least one entry failed | Read the JSON report at the `--output` path and fix the reported entry. |
| A port stays occupied after a UI check | A supervised server was not stopped | Stop every supervised server and confirm the port is closed. |
| A documentation contract fails | A renamed heading or moved link broke a reference | Update every inbound link and anchor in the same change. |

## Pre-delivery checklist

- The changed area compiles, lints, passes the format check, and passes its
  focused tests; shared platform, UI, CLI or workflow changes also ran the Apps
  suite.
- Every test run used its own unique writable base-temporary directory.
- A documentation change passed the relative-link and documentation/privacy
  contracts.
- A UI change was exercised in Light, Dark and Auto across an operating-system
  color-scheme change, and in Dark Dimmed where the page supports it.
- Every supervised server is stopped and its port is closed.
- Raw verification output stayed private; no operation history or execution
  receipt entered the documentation.
- Third-party assets keep their license and notice files, and the
  `Apps/pyproject.toml` resource list matches the distribution boundary checks.

## Related documents

- [Apps guide](../README.md) — installation, command hierarchy and runtime boundary.
- [Apps architecture](architecture.md) — interface ownership, formats and dependency rules.
- [Environment guide](../../environment/README.md) — platform locks and exact replay.
- [Development workflow](../../WORKFLOW_README.md) — save, update and release procedure.
- [ARCHITECTURE.md](../../ARCHITECTURE.md) — partitions and the source/runtime boundary.
- [Library development](../../Python/CONTRIBUTING.md) — checks for the library partition.
