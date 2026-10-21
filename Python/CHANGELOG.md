# Changelog

## Unreleased

- Made the English home and contribution guide canonical, added full Chinese
  counterparts, and retained the previous English paths and section anchors.
- Added fresh Miniforge setup, deterministic installation output, a connected
  documentation index and migration guidance. Clarified editable Apps installs
  and the scope of scientific image naming.
- Added repository citation metadata and MIT terms for self-owned root,
  maintenance, documentation and environment files while preserving component
  GPL/MPL licenses and historical attribution. Current author metadata is Y. Li,
  Shandong University.
- Corrected AIA ROI bounds, exposure-unit handling and TAI-to-UTC conversion;
  retained explicit legacy ROI selection. Preserved negative unsigned-input differences
  and HMI masks, rejected masked PFSS boundaries, and tightened derivative axes
  and spacing while preserving derivative modes and decreasing Gaussian axes.
- Hardened radio quality, UTC and finite-input handling. Invalid Newkirk
  frequencies retain missing values and reasons where row preservation applies;
  HMI overlays require a caller-selected ROI. See the migration guide for details.
- Routed AIA single-image, mosaic and differencing inputs through the shared
  exposure normalizer. Multi-band index slots remain explicitly unsynchronized.
- Corrected generic manifest links, safe YAML configuration roundtrips,
  finite frequency/derivative validation and unit-equivalent archive wavelengths.
  Malformed download-index structures are ignored rather than causing crashes.
- Unified nearest-time lookup across UTC, timezone-aware and TAI inputs, and
  validate Newkirk model settings even when no observations are accepted.
- Normalized HXI epoch/interval times and validated four-channel input shape;
  removed implicit machine paths and event ROI from the legacy config adapter.
- Made quick saves reject protected branch/index states before mutation and
  reuse existing pull requests. Help and invalid commands leave runtime state
  uninitialized.
- Split release preparation from publication: `release prepare` (also `run`)
  changes only version files and changelog for a pull request; `release publish`
  validates an explicit merged commit and its CI before tag/Release publication.
  Both stages preview by default.

## 0.3.0

- Moved CLI, Web/GUI, browser and orchestration code out of the reusable library
  into `solar_apps`. Scientific configuration types remain public; real event
  parameters and machine paths are caller-supplied private inputs.
- Removed public console scripts and implicit workstation path/event discovery.
- Added pure radio computation modules for reprojection, CSO processing and
  physical diagnostics.
- Re-rooted the package under the unified `solarphysics` monorepository.

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
