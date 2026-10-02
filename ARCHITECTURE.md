# Repository architecture

The repository has public code partitions and a private runtime
partition. Observation-year directories and `overview/` are local data, not
source partitions.

```text
solarphysics/
|-- Python/             reusable solar_toolkit library
|-- Apps/               application interfaces and workflow orchestration
|-- tools/              general maintenance utilities
`-- Local/              ignored configuration, state, workspaces and outputs
```

## Dependency rules

Application dependencies flow in one direction:

```text
Apps/solar_apps/frontends
    |---> solar_apps/ui ------> solar_apps/platform
    |---> solar_apps/workflows -> solar_apps/platform
    `---> Python/solar_toolkit

Python/solar_toolkit -X-> solar_apps
solar_apps/platform  -X-> frontends or workflows
```

- `frontends` adapt user actions to shared UI, platform, workflow, and library
  APIs.
- `ui` owns the visual system and framework-specific adapters; it may depend
  only on platform services within `solar_apps`.
- `workflows` orchestrate scientific library functions and platform services;
  they do not import concrete frontends.
- `platform` owns runtime layout, configuration, state, allowed roots, native
  path selection, and subprocess policy without importing a workflow or UI.
- `solar_toolkit` remains reusable and never imports `solar_apps`.

## Source and runtime boundary

`Apps/` is public, reviewable source. It contains no observation data, personal
paths, saved UI state, logs, model artifacts, generated figures, or migration
evidence. The only committed configuration is the fail-closed example at
`Apps/configs/examples/paths.example.yaml`.

`Local/` is ignored in full. `Apps/run.ps1 admin init` creates its runtime
layout:

```text
Local/
|-- configs/paths.local.yaml
|-- state/
|-- workspaces/
|-- outputs/
|-- logs/
`-- tmp/
```

Allowed roots are revalidated whenever they cross an application boundary.
Remembered paths are convenience state, never authorization. Scientific data
and year-based observation folders stay outside Git.

## Runtime boundary

Application execution is restricted to Miniforge. The primary environment is
`solarphysics_env_latest`; `solarphysics_env` is an explicit compatibility
environment. The launcher rejects other environments and does not fall back to
a virtual environment or system Python. Child processes inherit the resolved
Miniforge interpreter.

## Composable library and examples

The scientific library is the primary reusable surface. Examples and application
adapters compose its APIs with explicit inputs and parameters. They do not
provide built-in research events. Public source contains no research records,
real event configurations, observation data or generated scientific results.
Private scientific configuration is supplied by the caller; machine path
authorization is handled separately by the application platform.
