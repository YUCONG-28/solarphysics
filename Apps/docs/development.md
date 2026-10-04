# Apps development

## Environment

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

## Checks

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

## Health reports

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

## Maintenance and contributions

The optional `tools quick` and `tools release` commands expose repository
maintenance operations. Inspect their help and use the
[development workflow](../../WORKFLOW_README.md) for save, update and release
procedures. A help command does not authorize a subsequent commit, push or
release.

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

The public-source policy, package-content checks and contribution workflow are
shared with the repository. Do not publish raw operation history or execution
receipts in documentation.
