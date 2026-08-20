# App 1.0 Interface Equivalence Audit

This audit maps each of the eleven approved App 1.0 interfaces to its retained
scientific implementation and native PyQt6 surface. "Equivalent" means that
the user can reach the established workflow, pass the same scientific inputs
and parameters, and retain its outputs and metadata. It does not mean that
application chrome or framework-specific layout is pixel-identical.

| App 1.0 interface | Retained implementation | Native surface and parity result |
| --- | --- | --- |
| Workbench | Existing workflow launchers and artifact contracts | Equivalent as the application home, module navigation, project context, task queue, logs, outputs, and recovery surface. It does not duplicate science. |
| Data Download | JSOC, VSO, NOAA SUVI, and SOAR provider clients in `solar_toolkit.net` | Native two-stage search, unchecked preview, explicit selection, atomic download, cancellation, retry, SHA-256 receipt, and typed Workbench/DAG functions. Scientific observations remain under `Local/observations`; generated task records remain under App 1.0 outputs. |
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

- App v1 runs PyQt6 in a dedicated process; foreign Qt bindings are rejected if
  already loaded.
- All long-running work uses supervised `QProcess` workers. The application
  task queue remains FIFO; a typed workflow uses 1–4 FIFO process lanes for
  same-level nodes. A failed node blocks only its descendants, while
  independent branches continue. Tasks and flows can be cancelled or retried.
- Every real load, calculation, export, and cross-module transfer retains the
  input/parameter/output/workload confirmation gate.
- Normal App 1.0 operations never start Flask, Streamlit, a browser, or a
  second Qt binding. Pages with a deprecated predecessor expose it only through
  `More` → `Open legacy interface`, with a separate confirmation and no
  automatic fallback. Data Download and Image Composer are native-only.
- Workers may emit `APP_V1_EVENT` schema 1 records for progress, logs,
  previews, artifacts, and terminal results. Legacy line output remains
  accepted while retained scientific commands migrate to the structured
  protocol.
- UTC is stored and exchanged through one coordinator and a rebuildable
  private SQLite index.
- `.spapp.json` schema 1 projects contain modules, parameters, time-sync
  configuration, window layout, and safe relative `manifest.json` references.
  They never embed observation bytes. The active workflow is referenced by
  `layout.active_flow_id`.
- `.spflow.json` schema 1 stores function/variant IDs, all typed parameters,
  connections, concurrency, disabled/group state, and visual layout. It does
  not embed observations or reuse historical calculations implicitly.
- Qt forms, argv/config construction, confirmations, migration, presets, and
  workflow validation share the same parameter/function catalog. Free-form
  additional arguments are not accepted; unknown legacy argv is reported and
  blocks execution until corrected.
- Basic plotting, path selection, ROI history/import/export, playback,
  artifact preview, export, and parameter-form behavior each have one shared
  App 1.0 implementation. Plot styling is data in `PlotSpec`, not a renderer
  variant. Only choices that change numerical/scientific meaning are exposed
  as scientific variants.
- Parameter presets are versioned, module-scoped, and atomic. Project writes,
  established workflow manifests, composer images, and composer videos use
  atomic replacement or staging contracts.
- Auto, Light, Dark, and Dark Dimmed affect application chrome only;
  scientific arrays, WCS, normalization, metadata, and exports are
  theme-invariant. Auto follows only the system Light/Dark setting.

The stable launcher is `Apps/run.ps1 frontend app-v1`. The
`app-v1-preview` launcher remains a compatibility alias for one release cycle.
Legacy Flask and Streamlit launchers remain deprecated compatibility surfaces.
The Image Composer compatibility launcher selects the native App 1.0 page.
