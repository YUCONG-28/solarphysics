# Repository architecture

Purpose: own the partition map, the permitted dependency directions, and the rule
set that separates reviewed public source from private runtime state.

Reader: anyone adding a module, a configuration file, a path constant, or a
documentation page. Shared vocabulary is in
[docs/terminology.md](docs/terminology.md); the document classes in
[docs/writing-standard.md](docs/writing-standard.md); the contribution loop in
[CONTRIBUTING.md](CONTRIBUTING.md).

This page owns no scientific method, no command syntax, and no installation
recipe.

## 1. Purpose and boundary

The repository has public code partitions and a private runtime partition.
Scientific observations and research records belong in private storage, outside
the public source partitions.

| Owned here | Belongs elsewhere |
| --- | --- |
| The partition map and the dependency diagram | Command syntax and frontend behaviour: [Apps/README.md](Apps/README.md) |
| The source/runtime boundary and the `Local/` layout | Scientific contracts and validity: [docs/foundations.md](docs/foundations.md) |
| Allowed-root and interpreter rules | Library API surface: [Python/README.md](Python/README.md) |
| The public/private publication boundary | Lock files and replay evidence: [environment/README.md](environment/README.md) |

## 2. Layout

```text
solarphysics/
|-- Python/             reusable solar_toolkit library
|-- Apps/               application interfaces and workflow orchestration
|-- tools/              general maintenance utilities
|-- docs/               technical guide navigation and methods
|-- environment/        target-platform artifact locks and replay guidance
`-- Local/              ignored configuration, state, workspaces and outputs
```

`Apps/run.ps1 admin init` creates the ignored runtime layout:

```text
Local/
|-- configs/paths.local.yaml
|-- state/
|-- workspaces/
|-- observations/
|-- outputs/
|-- logs/
`-- tmp/
```

## 3. Dependency directions

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

The forbidden directions are exactly the two `-X->` arrows: the library must not
import the application layer, and the platform layer must not import a frontend or
a workflow.

## 4. Run and verify

Runtime files are resolved by `Apps/run.ps1 admin init` (or `Apps/run.sh` on
macOS); there is no separate verification command at this level. The rules below
are asserted by `Apps/tests/contracts/test_repository_policy.py` and
`Apps/tests/contracts/test_docs.py`; the application commands that exercise them
are documented once in [Apps/README.md](Apps/README.md).

## 5. Rules that must not be violated

1. `Apps/` is public, reviewable source: it contains no observation data, personal
   paths, saved UI state, logs, model artifacts, generated figures, or migration
   evidence.
2. The only committed machine-path configuration is the fail-closed example at
   `Apps/configs/examples/paths.example.yaml`.
3. `Local/` stays ignored in full; every runtime file is created below it.
4. Allowed roots are revalidated whenever they cross an application boundary.
   Remembered paths are convenience state, never authorization.
5. Runtime outputs must resolve under `Local/` or outside the repository; allowed
   roots never authorize writes into public source.
6. Scientific data and year-based observation folders stay outside Git.
7. Application execution is restricted to Miniforge. `solarphysics_env_latest` is
   the primary environment and `solarphysics_env` the explicit compatibility
   environment. The launcher rejects other environments and never falls back to a
   virtual environment or system Python; child processes inherit the resolved
   Miniforge interpreter.
8. The scientific library is the primary reusable surface. Examples and
   application adapters compose its APIs with explicit inputs and parameters and
   provide no built-in research events.
9. Public source contains no research records, real event configurations,
   observation data, or generated scientific results. Private scientific
   configuration is supplied by the caller; machine path authorization is handled
   separately by the application platform.

## 6. Related documents

- [README.md](README.md) — library entry point and capability overview.
- [Apps/README.md](Apps/README.md) — application commands, frontends, themes, privacy policy.
- [Python/README.md](Python/README.md) — library boundary, install, and verification.
- [docs/README.md](docs/README.md) — technical guide index and methods.
- [environment/README.md](environment/README.md) — platform locks and the exact-environment gate.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how a change is prepared, verified and submitted.
