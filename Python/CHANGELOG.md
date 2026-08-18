# Changelog

## 0.3.0

- Moved all CLI, Web/GUI, browser, event configuration and orchestration code
  to the ignored local `solar_apps` application repository.
- Removed public console scripts and implicit workstation path/event discovery.
- Added pure radio computation modules for reprojection, CSO processing and
  physical diagnostics.
- Re-rooted the package under the unified `solarphysics` monorepository.

## Unreleased

- Unified the repository version into a single source (`_version.py`) read by
  both the `solar-physics-toolkit` and `solarphysics-apps` distributions.
- Added the `tools release` command (version bump, changelog rewrite, tag,
  push, and GitHub release; dry-run by default).
- Added the `tools quick` command family (`check`/`save`/`push`/`update`) for
  fast verify, commit, push, and Pull-Request workflow.
- Updated contributor and architecture documentation for `solarphysics_env`
  setup, standalone `pre-commit` usage, verification environment variables, and
  public-surface documentation sync rules.

## 0.2.0

- Reorganized standalone workflows under `scripts/` by scientific task and
  moved reusable implementations into `solar_toolkit/`.
- Added package-owned commands: `solar-aia`, `solar-radio`,
  `solar-image-viewer`, and `solar-webapp`.
- Expanded `solar-radio` into eight package-owned subcommands: `centers`,
  `pipeline`, `source-map`, `overlay`, `quicklook`, `raw-quality`,
  `roi-lightcurve`, and `trajectory`.
- Added the integrated `/radio` workspace in the local webapp for explicit
  Preview, Run, and Cancel actions over selected radio modules.
- Added path-only local configuration support through `solar_toolkit.path_config`,
  `configs/paths.example.yaml`, and ignored `configs/paths.local.yaml`.
- Split data-independent tests into `tests/` and local-data examples into
  `examples/`.
- Added GitHub Actions lightweight CI, citation metadata, contribution guidance,
  pre-commit/Gitleaks checks, and architecture documentation.
