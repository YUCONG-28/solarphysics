# Apps development

## Environment

Use Miniforge and the primary application environment. The launcher is the
source of truth for interpreter selection.

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda env update -n solarphysics_env_latest -f .\Apps\environment.miniforge.yml
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-ml]"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Apps[dev]"
```

```bash
/Users/<user>/miniforge3/bin/conda env update -n solarphysics_env_latest -f Apps/environment.miniforge.yml
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Python[quality-ml]"
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Apps[dev]"
```

Use `solarphysics_env` only by passing it explicitly to `Apps/run.ps1` or
`Apps/run.sh` for a
compatibility comparison. No other environment is supported by the Apps CLI.

## Checks

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m compileall -q Apps/solar_apps Apps/tests
& $Conda run -n solarphysics_env_latest python -m ruff check Apps/solar_apps Apps/tests
& $Conda run -n solarphysics_env_latest python -m pytest Apps/tests --basetemp .\Local\tmp\pytest-apps
```

The same checks run unchanged through `conda run` on macOS. Use
`./Apps/run.sh frontend <id> --help` for macOS frontend smoke tests.

Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File Apps/run.ps1
frontend <id> --help` for each frontend and exercise the
affected UI through a supervised local server. Verify explicit Light and Dark,
then Auto while changing the emulated operating-system color scheme. Stop the
server and confirm its port is closed.

## Health matrix

Run the offline App health matrix from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 tools health --output .\Local\tmp\apps-health.json
```

```bash
./Apps/run.sh tools health --output Local/tmp/apps-health.json
```

The default matrix covers every catalog-driven formal frontend entry, every App
1.0 page, and the offline smokes. Legacy Flask/Streamlit servers are reported
as `not_run` with a machine-readable reason; pass `--include-legacy-servers`
only when loopback ports are explicitly allowed. The report is atomic
structured JSON, and the command exits non-zero when the overall status is
`fail`. CI runs the default matrix without `--include-legacy-servers`.

## Contributions

- Preserve the package dependency direction and stable CLI IDs.
- Add a focused unit or contract test for shared behavior.
- Do not commit files from `Local/`, data-year directories, or `overview/`.
- Do not place personal paths, credentials, user state, logs, screenshots, or
  generated research products in examples or documentation.
- Keep UI theme state out of scientific sidecars and cache signatures.
- Keep third-party assets with their license and notice files.
