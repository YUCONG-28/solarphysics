# Contributing

[中文](CONTRIBUTING.md) | [English](CONTRIBUTING.en.md) | [Home](README.en.md)

This repository maintains composable scientific modules, a small set of
synthetic examples and application adapters. Scientific calculations belong in
`Python/solar_toolkit`, interfaces and orchestration in `Apps/solar_apps`, and
general maintenance utilities in `tools`. Follow the applicable `AGENTS.md`,
preserve public interfaces and retain component licenses.

## Prepare the environment

Run all commands below from the repository root. Use Miniforge's
`solarphysics_env_latest`; reserve `solarphysics_env` for explicitly requested
compatibility checks. Install the library development dependencies:

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pip install -e './Python[dev]'
```

The equivalent Windows PowerShell command is:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[dev]"
```

See [library development](Python/CONTRIBUTING.md) for optional dependencies
and scientific interface requirements. Application development requires Python
3.14; install its dependencies using the
[Apps development guide](Apps/docs/development.md). Platform locks and replay
are described in the [environment guide](environment/README.md).

## Submit a reproducible change

1. Branch from the latest remote `main`, preserve existing uncommitted work,
   and stage only files relevant to the change.
2. Provide minimal synthetic inputs with shapes, units, UTC conventions and
   expected behavior.
3. Run relevant tests and broaden validation when shared behavior changes.
   State which checks ran and their limitations.
4. Inspect actual staged content before committing and opening a pull request.
   Merging, deleting branches and publishing releases each require explicit authorization.

Run the basic checks from the repository root, with a temporary directory
dedicated to this run:

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pytest Python/tests --basetemp Local/tmp/pytest-library-check
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m ruff check --config Python/pyproject.toml Python/solar_toolkit Python/tests
<miniforge-root>/bin/conda run -n solarphysics_env_latest python tools/public_source_policy.py --staged
```

See [Apps development](Apps/docs/development.md) for application checks and
[library development](Python/CONTRIBUTING.md) for distribution builds and
independent installation checks. The [maintenance workflow](WORKFLOW_README.md)
covers branches, quick saves and maintenance tools.

## Public content boundary

Commits, issues and pull requests should contain general code, synthetic
reproduction material and necessary technical explanations. Keep real event
presets, observations, research progress, results, personal paths, credentials
and raw execution logs in `Local/` or outside the repository. Runtime outputs
must also use those locations; symlinks must not redirect writes into public
source directories.

Public configuration types may remain in the library; callers supply their
scientific parameters explicitly. Preserve attribution and licenses when
reusing implementations. Document current behavior and distinguish historical
check records from validation of the current change.
