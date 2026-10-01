# Apps architecture

`solar_apps` is an application layer over the reusable `solar_toolkit` library.
It is organized by responsibility rather than by launch script:

```text
solar_apps/
|-- cli/          command routing and compatibility aliases
|-- platform/     runtime layout, configuration, state, paths, processes
|-- ui/           theme and Web, Streamlit, Qt, and media adapters
|-- frontends/    App 1.0 and retained compatibility interfaces
`-- workflows/    AIA, HMI, radio, visualization, data, net, and X-ray flows
```

Frontend modules translate user intent into workflow calls. Workflow modules
coordinate scientific operations but leave reusable calculations in
`solar_toolkit`. Platform services are domain-neutral. The UI layer shares one
semantic design system without changing scientific normalization or exports.

## Radio workflow maintenance

Keep the public workflow modules as compatibility entry points. Their internal
modules divide application responsibilities without replacing scientific code:

| Entry point | Internal responsibilities |
| --- | --- |
| `workflows.radio.source_map_workflow` | `_source_map_selection` discovers and verifies inputs; `_source_map_ranges` resolves display limits; `_source_map_layout` owns axes and colorbar geometry. The core owns execution and the shared spectrogram cache. |
| `workflows.radio.pipeline_workflow` | `_pipeline_products` builds diagnostic tables and provenance; `_pipeline_plotting` presents the resulting products. Scientific dependencies load when the operation needs them. |
| `workflows.radio.overlay_workflow` | `_overlay_context` holds configuration; `_overlay_selection` pairs inputs; `_overlay_science` adapts scientific operations; `_overlay_render` renders panels and outputs. The core coordinates the stages. |
| `frontends.app_v1.phase4_page` | `phase4_canvas` handles canvas and layer controls; `phase4_project` handles schema-1 persistence and dirty state; `phase4_export` handles output controls and confirmation. The page retains Qt signals. |
| `frontends.radio.roi_lightcurve.roi_lightcurve_app` | `_roi_app_files` plans inputs and owns the reference-image LRU; `_roi_app_regions` imports and selects regions; `_roi_app_analysis` manages analysis and session caches; `_roi_app_display` presents and exports results; `_roi_app_page` builds the Streamlit page; `_roi_app_state` holds settings, models and session initialization. The application adapter retains its grid and crop caches. |

Run focused tests for the affected responsibility before the complete Apps and
library checks. Compare structural changes with the corrected scientific
baseline using synthetic observations, decoded images, tables and sidecars.
Keep execution evidence below `Local/`; historical acceptance records do not
replace verification of the current change.

Runtime files are resolved through `RuntimeLayout` and live below the ignored
repository-level `Local/` directory. Production code must not inject paths into
`sys.path` or infer repository roots with fixed parent indexes.

Private configuration layout version 2 is declared by
`apps.runtime_layout_version`; migrations retain only explicitly supported
settings. This version is independent of the per-frontend `StateStore` schema.

See the repository-level [architecture](../../ARCHITECTURE.md) for dependency
and privacy rules.
