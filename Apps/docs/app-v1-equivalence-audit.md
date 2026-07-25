# App 1.0 Interface Equivalence Audit

This audit maps each of the ten approved App 1.0 interfaces to its retained
scientific implementation and native PyQt6 surface. "Equivalent" means that
the user can reach the established workflow, pass the same scientific inputs
and parameters, and retain its outputs and metadata. It does not mean that
application chrome or framework-specific layout is pixel-identical.

| App 1.0 interface | Retained implementation | Native surface and parity result |
| --- | --- | --- |
| Workbench | Existing workflow launchers and artifact contracts | Equivalent as the application home, module navigation, project context, task queue, logs, outputs, and recovery surface. It does not duplicate science. |
| Radio Workspace | Existing radio frontends and workflows | Equivalent as grouped radio navigation, time broadcast, and task aggregation. Calculations remain in their dedicated modules. |
| Image Viewer | Image Viewer, AIA workflow, and HMI overlay workflow | Equivalent adapter for image discovery, AIA rendering, HMI overlay, confirmation, and process-isolated execution. |
| Image Composer | Existing schema 1 model, matching, and export logic | Equivalent PyQt6 composer with drag/move, overlap, z-order, opacity, alignment, grid, aspect-preserving sizing, multi-panel layout, UTC selection, high-resolution PNG, and sequence video. Existing `.fic.json` schema 1 projects remain importable. |
| Bad Frame Review | Radio Bad Frame Review index and server | Equivalent adapter for review-mode selection and supervised launch, including explicit all-scanned-frame browsing. |
| Source Map | Radio Source Map workflow and artifact contracts | Equivalent adapter for confirmed preparation, map rendering, one-region ROI selection, Gaussian products, and manifest-compatible outputs. |
| DART Spectrogram | DART/CSO workflow and canonical drift tools | Equivalent adapter for spectrogram loading, bounded selection, narrow-band extraction, drift-rate calculation, and Newkirk diagnostics. |
| ROI Light Curve | Existing ROI Light Curve frontend | Equivalent adapter for supervised, confirmed one-ROI analysis while retaining multi-region import selection. |
| Radio Composite Figure | Existing composite application and sequence exporter | Equivalent adapter for confirmed multi-frequency composite images, frame packages, video, and recoverable supervised execution. |
| Source Trajectory | Existing Source Trajectory and DEM workflows | Equivalent adapter for trajectory inspection plus existing DEM/radio products; no new DEM inversion or GPU algorithm is introduced. |

## Cross-interface release contracts

- App v1 runs PyQt6 in a dedicated process; PySide6 and PyQt5 are rejected if
  already loaded.
- All long-running work uses the shared sequential `QProcess` queue. Tasks can
  be cancelled, a confirmed batch preserves input order, the last terminal
  task can be redrawn, and failed tasks can be queued again after the cause is
  corrected.
- Every real load, calculation, export, and cross-module transfer retains the
  input/parameter/output/workload confirmation gate.
- UTC is stored and exchanged through one coordinator and a rebuildable
  private SQLite index.
- `.spapp.json` schema 1 projects contain modules, parameters, time-sync
  configuration, window layout, and safe relative `manifest.json` references.
  They never embed observation bytes.
- Parameter presets are versioned, module-scoped, and atomic. Project writes,
  established workflow manifests, composer images, and composer videos use
  atomic replacement or staging contracts.
- Auto, Light, and Dark affect application chrome only; scientific arrays,
  WCS, normalization, metadata, and exports are theme-invariant.

The stable launcher is `Apps/run.ps1 frontend app-v1`. The
`app-v1-preview` launcher remains a compatibility alias for one release cycle.
Legacy Flask, Streamlit, and PySide6 launchers are marked deprecated but remain
available and continue to run in separate processes.
