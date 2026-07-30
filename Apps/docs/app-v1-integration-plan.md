# Solar Physics App 1.0 Integration Plan

This document is the repository UTF-8 version of the approved phased
integration plan. It adapts the original conceptual `core/`, `gui/`, `data/`,
and `output/` layers to the repository's existing boundaries:

- reusable scientific calculations remain in `Python/solar_toolkit`;
- application orchestration and interfaces remain in `Apps/solar_apps`;
- configuration, state, workspaces, outputs, logs, indexes, and temporary files
  remain in the ignored `Local/` runtime tree;
- observation data is never copied into public source.

The released implementation lives under `solar_apps/frontends/app_v1/`. It
uses an English PyQt6 interface and does not import PySide6 or PyQt5 in the same
process. The previous `app-v1-preview` command is a compatibility alias to the
same implementation; it does not create a second code path.

## Phase gates

### Phase 0: Baseline and contracts

Validate the current in-progress frontends, record the capability matrix, add
the isolated GPLv3 experiment, declare PyQt6, and define versioned module,
request, result, artifact, project, timeline, and runtime-path contracts.

### Phase 1: PyQt6 shell

Add the `app-v1-preview` entry point, main window, navigation, plot surface,
parameter panel, task queue, logs, outputs, themes, cancellation, and explicit
placeholder pages for the original ten target interfaces.

### Phase 2: Scientific integration

- 2A: AIA read/plot, HMI overlay, and Image Viewer.
- 2B: Bad Frame Review, Source Map, Gaussian fitting, ROI Light Curve, and
  Radio Composite.
- 2C: DART spectrogram, drift rate, Newkirk, Source Trajectory, and existing
  DEM adapters.

Each batch reuses existing scientific code and requires a separate approval
gate after automated fixtures and a user-selected real-data check.

### Phase 3: Time synchronization

Add a private SQLite time index, UTC matching, tolerances, manual offsets,
missing/duplicate handling, and current-time broadcast across time-aware pages.

### Phase 4: Free image composition

Rewrite the Image Composer interface in PyQt6 while reusing its existing data,
matching, and export logic. Preserve `.fic.json` schema 1 import compatibility
and add synchronized multi-panel, layer, alignment, grid, scaling, and
high-resolution export behavior.

### Phase 5: Research workflow and release

Add `.spapp.json` projects, presets, redraw, batch queues, video, atomic output,
recovery, and the original ten-interface parity audit. Promote the implementation to a
stable `app-v1` frontend only after the gate passes. Retain existing frontends
as deprecated compatibility surfaces; deletion is outside this plan.

## Confirmation and verification

Real data loading, scientific runs, exports, and cross-workflow transfers must
show a confirmation summary. Browsing generated previews and editing
parameters do not require repeated confirmation.

Every phase runs import smoke tests, compilation, formatting/lint checks,
focused tests, the complete Apps suite, supervised Windows launch/stop checks,
and appropriate Auto/Light/Dark verification. Work stops after each phase and
after each Phase 2 batch until the user authorizes the next gate.

## Locked contracts and runtime locations

The public App 1.0 contracts are `ModuleDescriptor`, `RunRequest`, `RunResult`,
`ArtifactManifestV1`, `TimelineSource`, `SyncSelection`, and
`AppV1ProjectV1`. The Data Download extension adds `ObservationQueryV1`,
`RemoteObservationV1`, and `ObservationCollectionV1`. Version 1 contracts
exchange JSON-compatible values and UTC
timestamps. Input references may contain allowed-root locators and checksums;
they never embed original observation bytes.

App 1.0 extends the existing `RuntimeLayout` only:

- outputs: `Local/outputs/app_v1/<project>/<run>/<module>/`;
- project files: `Local/workspaces/app_v1/*.spapp.json`;
- time index: `Local/state/app_v1/time_index.sqlite3`;
- logs and temporary files: the corresponding existing `Local/logs/` and
  `Local/tmp/` roots, each below `app_v1/`.
- downloaded observations: `Local/observations/<mission>/...`; generated
  search sets, receipts, and manifests remain below
  `Local/outputs/app_v1/.../data-download/`.

Each completed module run may contain `images/`, `data/`, `media/`, and
`manifest.json`. The manifest records the request parameters, UTC range, input
references or checksums, software versions, products, and terminal status.
Raw observations, private absolute paths, and generated runtime state do not
belong in public source or test fixtures.

## Locked implementation assumptions

- Windows is the primary platform and `solarphysics_env_latest` is the primary
  interpreter.
- Visible App 1.0 interface text is English.
- PyQt6 is the only Qt binding loaded by the App 1.0 process. Legacy PyQt5 and
  PySide6 interfaces continue in separate processes.
- The App v1 subtree is GPL-3.0-only. Existing repository licensing
  outside that subtree is unchanged.
- "Rewrite" refers to interface and orchestration code. Scientific
  calculations continue to use `Python/solar_toolkit` and existing workflows.
- The order is strictly 0, 1, 2A, 2B, 2C, 3, 4, and 5. After Phase 1, the user
  authorized automatic progression whenever the preceding gate passes.
