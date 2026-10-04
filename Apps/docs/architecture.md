# Apps architecture

`solar_apps` adapts the reusable `solar_toolkit` library into interfaces and
composed workflows. The [repository architecture](../../ARCHITECTURE.md) defines
the dependency direction and public-source boundary.

```text
solar_apps/
|-- cli/          command routing and compatibility aliases
|-- platform/     runtime layout, configuration, state, paths, processes
|-- ui/           theme and Web, Streamlit, Qt, and media adapters
|-- frontends/    native demonstration and retained compatibility interfaces
`-- workflows/    AIA, HMI, radio, PFSS, visualization, data, net and X-ray flows
```

Frontends translate user input into workflow or library calls. Workflows
coordinate operations while reusable calculations remain in `solar_toolkit`.
Platform services are domain-neutral; UI services handle framework-specific
interaction. The library remains independently importable.

## Interface ownership

The native App 1.0 pages adapt existing implementations:

| Page | Owning APIs and adapter responsibility |
| --- | --- |
| Workbench | Workflow launchers, navigation, project context and task aggregation |
| Data Download | `solar_toolkit.net` provider clients; search, explicit selection, cancellation, retry and checksum receipts |
| Radio Workspace | Existing radio workflows; grouped navigation and time broadcast |
| Image Viewer | Image Viewer, AIA and HMI workflows; image discovery, rendering and supervised execution |
| Image Composer | Composer model, frame matching and export; native canvas and `.fic.json` persistence |
| Bad Frame Review | Radio quality/review APIs; review selection and all-scanned-frame browsing |
| Source Map | Source-map workflow; preparation, display, ROI selection, Gaussian products and manifests |
| DART Spectrogram | DART/CSO and drift APIs; bounded spectra, narrow bands and diagnostics |
| ROI Light Curve | Library extraction APIs; multi-region import and confirmed one-ROI analysis |
| Radio Composite Figure | Existing composite and sequence exporters; multi-frequency images and media |
| Source Trajectory | Trajectory and DEM/radio workflows; parameter selection and product inspection |
| Global PFSS | `solar_toolkit.modeling.pfss`; bundle verification, filtering, field-line display and export |

Interface availability describes how code is reached. Scientific equivalence
requires validation of the selected inputs, parameters and outputs. PFSS solver
execution and optional backends are separate from viewer availability.

## Native execution and shared services

The native interface runs PyQt6 in a dedicated process and rejects an already
loaded foreign Qt binding. Normal actions use supervised `QProcess` workers.
The application queue is FIFO; typed workflows use 1–4 process lanes for
same-level nodes. A failed node blocks its descendants while independent
branches may continue. Tasks and flows support cancellation and retry.

`ParameterSpec` and `FunctionSpec` provide one catalog for forms, validation,
argv/config construction, confirmations, presets, migration and typed artifact
ports. Unknown legacy arguments block execution and remain visible in the
migration report. `VariantFamilySpec` represents choices that change scientific
meaning; style choices remain data in `PlotSpec`.

Shared services cover path fields, image canvases, ROI import/export, playback,
artifact preview and export. `ScientificPlotRenderer` uses Matplotlib Agg in
workers. Real loads, calculations, exports and cross-module transfers retain
input/parameter/output/workload confirmation.

Workers may emit `APP_V1_EVENT` schema-1 records for progress, logs, previews,
artifacts and terminal results. Legacy line output remains accepted. UTC is
exchanged through one coordinator and a rebuildable private SQLite index.

## Private persistence and formats

`RuntimeLayout` resolves runtime directories under `Local/` or an explicitly
configured external root. Directory validation precedes creation and rejects
resolved destinations in public source. Machine path authorization and private
scientific configuration are separate inputs. Production code must not inject
paths into `sys.path` or discover roots using fixed parent indexes.

| Format | Contract |
| --- | --- |
| `paths.local.yaml` | Machine-local allowed roots; `apps.runtime_layout_version: 2` identifies the configuration layout. |
| `StateStore` JSON | Versioned, allow-listed latest fields; no scientific arrays or operation timeline. |
| `.fic.json`, schema 1 | Image Composer project; existing projects remain importable. |
| `.spapp.json`, schema 1 | Modules, parameters, time synchronization, window layout and safe relative manifest references; `layout.active_flow_id` identifies the active workflow. |
| `.spflow.json`, schema 1 | Function/variant IDs, typed parameters, connections, concurrency, disabled/group state and visual layout. |
| Parameter presets | Versioned, module-scoped files written atomically. |

Project and flow files reference observations without embedding their bytes or
implicitly reusing earlier calculations. Configuration-layout versions are
independent of frontend state schemas. Project writes, workflow manifests and
composer exports use atomic replacement or staging contracts. Downloads stay in
private observations directories; task manifests and receipts stay in private
outputs directories.

`RecentPathMemory` tries the current field, then matching field/operation/dialog,
operation, frontend and global entries, then an allowed root. Files remember
their parent; directory selectors remember the selected directory. Memory is
re-resolved against current allowed roots and never grants access.

## Display contracts

Auto, Light, Dark and the native Dark Dimmed theme change application chrome.
Scientific arrays, WCS, normalization, sidecars, exports and scientific cache
signatures are theme-invariant. Auto follows the system Light/Dark setting.

`SpatialRadioDisplay` defines colormap, invalid-value color, linear/log10
transform, percentile/fixed range, range scope, units, field of view and
preview/export profile. It applies to spatial source maps, Gaussian main
panels, pipeline map products, ROI references and RR/LL comparisons.

Effective display values resolve from scientific constraints, explicit current
CLI/UI overrides, private scientific configuration or saved settings, then
Source Map defaults. The optional schema-1 sidecar `display` object records the
effective values. Scientific display settings participate in preview/export
cache signatures. Spectra, residuals, light curves, trajectories and
multi-instrument overlays retain their own display and analysis contracts.

## Compatibility entry points

Use canonical IDs in new scripts. Retained aliases dispatch to the same
implementation:

| Retained form | Canonical form |
| --- | --- |
| `frontend app-v1-preview` | `frontend app-v1` |
| `webapp` | `frontend workbench` |
| `image_viewer` | `frontend image-viewer` |
| `image_composer` | `frontend image-composer` |
| `bad_frame_review` | `frontend bad-frame-review` |
| `bad_frame_ml` | `tools bad-frame-ml` |
| `aia ...` | `workflow aia ...` |
| `radio source-map-app ...` | `frontend source-map ...` |
| `radio dart-spectrogram ...` | `frontend dart-spectrogram ...` |
| `radio roi-lightcurve ...` | `frontend roi-lightcurve ...` |
| `radio source-trajectory-app ...` | `frontend source-trajectory ...` |
| Other `radio ...` commands | `workflow radio ...` |

The Image Composer compatibility launcher selects the native page. Deprecated
Flask/Streamlit interfaces remain explicit compatibility surfaces. Native pages
with a predecessor expose it through **More → Open legacy interface**, with a
separate confirmation and no automatic fallback. Data Download and Image
Composer are native-only.

## Workflow maintenance boundaries

Keep public workflow modules as compatibility entry points. Internal modules
divide application responsibilities:

| Entry point | Internal responsibilities |
| --- | --- |
| `workflows.radio.source_map_workflow` | `_source_map_selection` verifies inputs; `_source_map_ranges` resolves display limits; `_source_map_layout` owns axes and colorbar geometry. The core owns execution and the shared spectrogram cache. |
| `workflows.radio.pipeline_workflow` | `_pipeline_products` builds diagnostic tables and provenance; `_pipeline_plotting` presents products. Scientific dependencies load when needed. |
| `workflows.radio.overlay_workflow` | `_overlay_context` holds configuration; `_overlay_selection` pairs inputs; `_overlay_science` adapts library operations; `_overlay_render` renders outputs. The core coordinates the stages. |
| `frontends.app_v1.phase4_page` | `phase4_canvas` handles canvas controls; `phase4_project` owns persistence and dirty state; `phase4_export` handles output controls and confirmation. |
| `frontends.radio.roi_lightcurve.roi_lightcurve_app` | `_roi_app_files`, `_roi_app_regions`, `_roi_app_analysis`, `_roi_app_display`, `_roi_app_page` and `_roi_app_state` separate file planning, regions, analysis, display, page construction and state. |

Validation commands and contribution rules are in
[Apps development](development.md).
