# App 1.0 Capability Matrix

Every capability in the approved plan has an existing anchor, a delivered
App 1.0 surface, and an owning phase. "Adapter" means a PyQt6 application
surface over existing scientific code, not a second implementation of the
calculation. All rows are assigned and implemented; release evidence is
summarized in the equivalence audit.

| Capability | Current anchor | Delivered App 1.0 work | Phase |
| --- | --- | --- | --- |
| Data/config/output boundaries | `RuntimeLayout`, allowed roots, state stores | App-specific versioned contracts | 0 |
| Primer themes and responsive chrome | Per-frontend theme helpers | Fusion-based Auto, Light, Dark, and Dark Dimmed tokens; split navigation/context layout and scrollable pages | 1 |
| Run confirmation | `QMessageBox` and frontend-specific prompts | Shared scrollable, copyable Module/Input/Parameters/Output/Workload dialog with no endpoint field | 1 |
| Structured worker updates | Legacy `PROGRESS`/`LOG` lines | `APP_V1_EVENT` schema 1 progress, log, preview, artifact, and result events | 1 |
| Typed business parameters | Frontend forms and CLI parsers | One `ParameterSpec` source for Qt forms, validation, argv/config construction, confirmation, presets, and migration reports | 1 |
| Atomic capability catalog | Workbench and radio action registries | Shared `FunctionSpec` catalog with typed artifact ports and page-template aliases | 1 |
| Visual workflow composition | Fixed page orchestration | Acyclic `.spflow.json` schema 1 editor with drag/drop, typed connections, grouping, disabling, duplication, undo/redo, validation, and parameter search | 1 |
| Scientific choices | Workflow-specific algorithm switches | `VariantFamilySpec` only for numerically distinct methods, including weighted versus geometric threshold centroids | 1 |
| Shared plotting | Per-page Matplotlib helpers | One worker-only Matplotlib Agg `ScientificPlotRenderer` driven by versioned `PlotSpec`; style remains editable data | 1 |
| Shared interaction/export | Per-page path, ROI, playback, preview, and export helpers | One allowed-root path field, image canvas, ROI controller, playback controller, artifact browser, export service, and schema form | 1 |
| Unified Workbench | Flask Workbench | Native Qt dashboard and task summary | 1 |
| Radio Workspace | Flask Radio Workspace | Native Qt grouped navigation | 1 |
| Task, log, parameter, output panels | Existing per-frontend controls | Shared process-backed Qt shell | 1 |
| AIA read and plot | AIA workflow and `solar_toolkit` | Qt adapter and confirmation | 2A |
| HMI overlay | HMI overlay workflow | Qt adapter and confirmation | 2A |
| Image sequence viewing | Image Viewer | Native Qt parity page | 2A |
| Custom local data input | Allowed-root path services | Versioned module input adapters | 2A |
| Radio frame quality | Bad Frame Review | Native Qt parity page | 2B |
| Radio source read/map | Source Map workflow | Native Qt parity page | 2B |
| Gaussian fitting and centers | Radio fitting workflows | Qt controls and artifact contract | 2B |
| ROI flux/light curves | ROI Light Curve | Native Qt parity page | 2B |
| Multi-frequency composite | Radio Composite | Native Qt parity page | 2B |
| Radio spectrogram | DART/CSO workflows | Native bounded spectrum and narrow-band rendering with in-page artifact preview | 2C |
| Drift rate | Canonical drift-rate workflow | Qt selection and result adapter | 2C |
| Newkirk height estimate | Existing diagnostics/quicklook | Qt parameter/result adapter | 2C |
| Source trajectory | Source Trajectory | Native Qt parity page | 2C |
| Existing DEM workflows | X-ray/DEM workflow modules | Qt adapters; no new inversion algorithm | 2C |
| Time index and synchronization | Per-workflow nearest-time helpers | Shared private SQLite index and broadcast | 3 |
| Free image composition | PySide6 Image Composer | PyQt6 rewrite with `.fic.json` import | 4 |
| Layer/grid/alignment/high-resolution output | Image Composer model/export | Qt parity and visual QA | 4 |
| Project and parameter save | State stores and `.fic.json` | Versioned `.spapp.json` project | 5 |
| One-click redraw and batch processing | Individual workflow CLIs | Confirmed process queue plus 1–4 lane DAG execution, branch-safe failure propagation, cancellation, and retry | 5 |
| Video generation | Existing media and composite exporters | Shared project-aware orchestration | 5 |
| Stable App 1.0 release | Nine compatibility launchers and ten interfaces | Normal operations stay native; legacy launch is an explicit per-page More action | 5 |

GPU acceleration, new DEM algorithms, AI event detection/classification, a
remote Web service, and multi-user collaboration remain App 2.0 scope.
