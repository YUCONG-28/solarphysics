# Solar Physics Applications

`Apps` is the public application and workflow layer for the reusable
`solar_toolkit` library in `../Python`. It contains browser, Streamlit, and Qt
frontends, shared UI/platform services, and reproducible workflow orchestration.
Machine configuration, remembered fields, recent paths, logs, outputs, and
scientific data belong in the ignored `../Local` runtime tree.

## Requirements

- Windows with PowerShell, or Apple Silicon macOS 15+
- Miniforge
- the primary environment `solarphysics_env_latest`
- local observation and output directories selected by the user

`solarphysics_env` is supported only as an explicitly selected compatibility
environment. The launcher rejects `solarphysics_backup`, arbitrary Conda
environments, virtual environments, and system Python.

## Install

This installation path resolves version ranges and is intended for exploratory
development and compatibility testing. It must not be cited as an exact or
frozen environment. See the repository
[environment lock manual](../environment/README.md) for the fail-closed lock,
wheel-hash, and fresh-replay procedure.

Create or update the Miniforge environment and install both source partitions:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda env update -n solarphysics_env_latest -f .\Apps\environment.miniforge.yml
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-ml]"
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Apps
```

On macOS:

```bash
/Users/<user>/miniforge3/bin/conda env update -n solarphysics_env_latest -f Apps/environment.miniforge.yml
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Python[quality-ml]"
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Apps"
```

Initialize the private runtime tree:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 admin init
```

```bash
./Apps/run.sh admin init
```

This creates `Local/configs`, `state`, `workspaces`, `outputs`, `logs`, and
`tmp`, copies the single fail-closed configuration template to
`Local/configs/paths.local.yaml`, and writes private `Local/run.ps1` and
`Local/run.sh` forwarders for local launch habits. Add one or more absolute data/output
directories to `apps.allowed_roots`; an empty list fails closed.

If Miniforge is not in its usual location, pass `-MiniforgeRoot
"<miniforge-root>"` or set `SOLAR_MINIFORGE_ROOT`. The selected environment may
be made explicit:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 -EnvironmentName solarphysics_env_latest frontend workbench --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 -EnvironmentName solarphysics_env frontend workbench --help
```

The equivalent macOS option is
`./Apps/run.sh --environment-name solarphysics_env frontend workbench --help`.
Pass `--miniforge-root` when Miniforge is outside the default user directory.

Every Flask server, Streamlit process, Qt application, Workbench job, native
path dialog, and media worker inherits the interpreter resolved by the launcher.
There is no fallback to another Python installation.

## macOS quick start checklist

Run every command from the repository root.

- [ ] Confirm Miniforge is installed and the primary environment is available:

  ```bash
  /Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python --version
  ```

- [ ] On first use, or after dependency changes, create or update the
  environment and install both source partitions:

  ```bash
  /Users/<user>/miniforge3/bin/conda env update -n solarphysics_env_latest -f Apps/environment.miniforge.yml
  /Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Python[quality-ml]"
  /Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Apps"
  ```

- [ ] Initialize the private runtime once:

  ```bash
  ./Apps/run.sh admin init
  ```

- [ ] Edit `Local/configs/paths.local.yaml` and replace the empty
  `apps.allowed_roots` list with absolute data and output directories:

  ```yaml
  apps:
    runtime_layout_version: 2
    allowed_roots:
      - /absolute/path/to/data
      - /absolute/path/to/output
  ```

  Keep this file private. For a one-off launch, pass one absolute directory
  with `--allowed-roots /absolute/path/to/data` instead.

- [ ] Inspect a frontend's options, then launch it by removing `--help`:

  ```bash
  ./Apps/run.sh frontend <frontend-id> --help
  ./Apps/run.sh frontend <frontend-id>
  ```

  For example, launch Source Map with a one-off filesystem boundary:

  ```bash
  ./Apps/run.sh frontend source-map --allowed-roots /absolute/path/to/data
  ```

- [ ] If Miniforge is outside its default user location, add
  `--miniforge-root /absolute/path/to/miniforge3` before `frontend`.
- [ ] Stop a browser or Streamlit frontend with `Ctrl+C` in its supervising
  terminal. Close a Qt frontend through its application window.

## Applications and interfaces

Solar Physics App 1.0 is the stable native application. It contains eleven
English PyQt6 interfaces behind one process-isolated entry point:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1
```

On macOS or Linux, run the same native application from the repository root:

```bash
./Apps/run.sh frontend app-v1
```

Open the native Image Composer directly with
`./Apps/run.sh frontend app-v1 --module image-composer`. Existing automation
may continue to use `frontend image-composer --project <file.fic.json>`; that
compatibility route selects the same App 1.0 page. Composer startup remains
fail-closed unless allowed roots are configured privately or supplied with
`--allowed-roots` as an OS path-separated list.

The `app-v1-preview` frontend ID is a compatibility alias to that same module
for one release cycle. It is not a separate implementation.

Normal App 1.0 actions stay inside the PyQt6 window and use supervised
headless workers. They do not start Flask, Streamlit, or a browser.
Every page can open its predefined typed workflow with **Edit workflow**.
Atomic functions can be dragged into a visual DAG, connected through typed
artifact ports, and edited through the shared Common/Advanced schema form.
Flows are saved privately as
`Local/workspaces/app_v1/flows/<flow-id>.spflow.json` (schema 1); observation
data is referenced, never embedded. Unknown legacy arguments block execution
and are retained in a migration report.

The native **Data Download** page provides one search-preview-select-download
entry point for SDO/AIA and HMI, STEREO-A/B EUVI, SOHO/LASCO C1/C2/C3,
GOES-16/18 SUVI, and Solar Orbiter/EUI. Downloaded science data is archived
below `Local/observations/`; task manifests, search results, and receipts stay
below `Local/outputs/app_v1/.../data-download/`.

Use a page's **More → Open legacy interface** action only when an explicit
compatibility fallback is needed. The application chrome supports Auto,
Light, Dark, and Dark Dimmed GitHub Primer-inspired themes; scientific images
and exports are not recolored by the UI theme.

The earlier frontends remain launchable but are deprecated compatibility
surfaces:

| Frontend ID | Interface | Framework | Start command |
| --- | --- | --- | --- |
| `workbench` | General Workbench (`/`) and Radio Workspace (`/radio`) | Flask (deprecated) | `... -File .\Apps\run.ps1 frontend workbench` |
| `image-viewer` | Image sequence viewer and media export | Flask (deprecated) | `... -File .\Apps\run.ps1 frontend image-viewer` |
| `image-composer` | Free image composer compatibility route | App 1.0 / PyQt6 | `... -File .\Apps\run.ps1 frontend image-composer` |
| `bad-frame-review` | Radio bad-frame review | Flask (deprecated) | `... -File .\Apps\run.ps1 frontend bad-frame-review` |
| `source-map` | Radio source-map preparation and ROI annotation | Flask (deprecated) | `... -File .\Apps\run.ps1 frontend source-map` |
| `dart-spectrogram` | DART spectrogram analysis | Streamlit (deprecated) | `... -File .\Apps\run.ps1 frontend dart-spectrogram` |
| `roi-lightcurve` | Multi-region import and one-ROI light-curve analysis | Streamlit (deprecated) | `... -File .\Apps\run.ps1 frontend roi-lightcurve` |
| `radio-composite` | Multi-frequency Source Map ROI composite images and videos | Streamlit (deprecated) | `... -File .\Apps\run.ps1 frontend radio-composite` |
| `aia-radio-composite` | AIA/Radio Gaussian, multi-frequency ROI curve, and DART/CSO composite | Streamlit (deprecated) | `... -File .\Apps\run.ps1 frontend aia-radio-composite` |
| `source-trajectory` | Radio-source trajectory inspection | Streamlit (deprecated) | `... -File .\Apps\run.ps1 frontend source-trajectory` |

Replace `...` in the table with `powershell.exe -NoProfile -ExecutionPolicy
Bypass`. Append `--help` to a command before launching it. Browser-opening, host, port,
and allowed-root options are frontend-specific and are listed by that help.
Servers bind to loopback by default.

## Command model

The public command hierarchy is:

```text
Apps/run.ps1 frontend <frontend-id> [arguments]
Apps/run.ps1 workflow <domain> <command> [arguments]
Apps/run.ps1 admin <command> [arguments]
Apps/run.ps1 tools <command> [arguments]
Apps/run.sh frontend <frontend-id> [arguments]
Apps/run.sh workflow <domain> <command> [arguments]
Apps/run.sh admin <command> [arguments]
Apps/run.sh tools <command> [arguments]
```

Workflow domains are `aia`, `radio`, `hmi`, `net`, `data`, `visualization`, and
`xray-dem`. The `tools` group also carries repository maintenance commands:
`release` (version bump, tag, and GitHub release) and `quick`
(check/save/push/update). Discover commands without importing heavy optional
modules:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 workflow radio --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 workflow aia --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 tools bad-frame-ml --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 tools release --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 tools quick --help
```

The `release` and `quick` commands are documented in the
[project workflow guide](../WORKFLOW_README.md) sections 11 and 10.

Compatibility aliases preserve established command semantics while scripts
move to stable IDs:

| Legacy form | Canonical form |
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
| other `radio ...` commands | `workflow radio ...` |

Aliases are compatibility adapters, not separate implementations. New scripts
and documentation should use canonical commands.

## Theme system

All supported interfaces use the Radio Source Map semantic design system. Each
frontend offers:

- `Auto` (default), which follows operating-system color-scheme changes;
- `Light`, an explicit light palette;
- `Dark`, an explicit dark palette.

The theme controls surfaces, text, borders, controls, focus, status, and
framework-native chrome. It does not invert images and does not change data,
scientific colormaps, normalization, color limits, sidecars, cache keys, or
exported figures. A **Reset UI State** action restores the default theme and
clears that frontend's saved UI state.

## State and recent-path memory

`StateStore` writes versioned JSON atomically under `Local/state`. Frontends
save only allow-listed current field values; a corrupt or incompatible file is
ignored safely. State is a latest snapshot, never an operation log: it contains
no action timeline, run history, job identifier, upload content, generated
result, or scientific data.

The private configuration currently uses `apps.runtime_layout_version: 2`.
`admin init` upgrades older local layouts with an exact migration allow-list;
this runtime-layout version is separate from each `StateStore` JSON schema.

`RecentPathMemory` stores only the most recent usable directory for a path
field. A native dialog starts from the first valid choice in this order:

1. the current field value;
2. the same frontend, operation, field, and dialog type;
3. the same frontend and operation;
4. the same frontend;
5. the global entry for that dialog type;
6. the first configured allowed root.

Files remember their parent directory; folder selectors remember the folder.
Save As remembers only its directory and never silently overwrites an existing
file. Single-value fields replace their value. Multi-path fields append and
deduplicate with native Windows/macOS path semantics. Cancelled or failed dialogs leave
manual input untouched.

Remembered paths are re-resolved and revalidated against the current allowed
roots before use. Missing paths, changed roots, malformed state, or symbolic
link escapes are ignored. Memory never grants filesystem access.

## Allowed roots and local configuration

`Apps/configs/examples/paths.example.yaml` is the only committed application
configuration template. Its allowed-root list is empty by design. Keep the
machine copy in `Local/configs/paths.local.yaml`; never commit real paths.

Allowed-root precedence is:

1. a frontend or workflow's explicit `--allowed-roots` argument;
2. `SOLAR_APPS_ALLOWED_ROOTS`;
3. `Local/configs/paths.local.yaml`.

The complete repository, its ancestors, and drive roots are rejected. Input
files and directories must exist. An output target must have a real allowed
parent. Browser path requests use loopback-only, temporary-token-protected
endpoints, and returned paths are revalidated in the parent process.

## Spatial radio display contract

Spatial radio-map panels share `SpatialRadioDisplay`. The contract defines:

- colormap and bad-value color;
- linear or base-10 logarithmic transform;
- automatic percentile or fixed range;
- range scope, lower/upper percentiles, `vmin`, and `vmax`;
- display-unit override;
- field of view;
- preview or export render profile.

It applies to Source Map, Workbench source-map and Gaussian main panels, the
pipeline source-map stage, ROI reference maps, and RR/LL comparison map panels.
Effective values resolve in this order:

1. immutable scientific workflow constraints;
2. explicit CLI or UI overrides for the current operation;
3. event configuration or saved settings;
4. Source Map display defaults.

The optional `display` object in a schema-1 Source Map sidecar records effective
display settings while remaining backward compatible with older sidecars.
Display settings participate in scientific preview/export cache signatures;
the UI theme does not.

This contract does not replace domain-specific science. Bad-frame previews,
Gaussian residuals, dynamic spectra, DART panels, light curves, trajectories,
and multi-instrument overlays retain their own normalization, colormaps, WCS,
contours, and analysis parameters.

## Frontend notes

### Workbench and Radio Workspace

Workbench exposes registered application workflows through stable workflow IDs
rather than repository script paths. The root interface covers general tools;
`/radio` provides grouped radio discovery, inspection, fitting, plotting, and
export actions. Active catalog entries must be importable. Historical entries
are presented as non-executable reference material.

### Image Viewer

Image Viewer discovers allowed image sequences, previews them in the browser,
and exports supported media without changing the source images. Its settings,
theme, and last directories persist independently from other frontends.

### Image Composer

The Qt composer arranges multiple image folders on a canvas and matches frames
by time or relative index. Project content is saved explicitly as `.fic.json`;
general UI state is stored separately. Output file selection always uses Save
As behavior and explicit overwrite confirmation.

### Source Map, ROI Light Curve, and Radio Composite

Source Map prepares single- or synchronized multi-band images and matching
coordinate sidecars, then supports rectangle/lasso ROI annotation. ROI Light
Curve may import a multi-region ROI JSON, but stages and analyzes exactly one
selected ROI after the existing confirmation step. Upload priority and
allowed-root checks are unchanged; failed imports preserve current state.

Radio Composite keeps the complete operation in one Streamlit page. Select one
or more Source Map frequencies, render a reference time, and confirm one
rectangle or lasso ROI in HPLN/HPLT coordinates. The same ROI can then be
previewed across every selected frequency and observation time with frame,
previous/next, and play/pause controls before export.

The static reference and every sequence frame retain the three-row layout. The
middle panel is the current frequency's native-sample ROI `raw_sum` curve; the
lower panel is the selected DART narrowband source Stokes I dB intensity. Both
curves share UTC and use the same thin marker for the current Source Map time;
neither curve is interpolated or resampled. Sequence export uses real radio
frames with an adjustable stride and creates one FFprobe-validated MP4 per
frequency, the matching PNG frames, per-frequency manifests and curve CSVs,
shared DART/ROI metadata, and a `radio-composite-v2` ZIP package.
Each frequency freezes its scientific canvas and all three panel bounds from
the first valid frame, so titles and tick labels cannot resize later frames.
`Generate MP4 video` and `Save every composite frame as PNG` are independent
options and both remain enabled by default. MP4/WebM frame streams use PyAV
when available and retain the existing FFmpeg fallback.

AIA Radio Composite uses a radio-source ROI rather than an AIA-pixel ROI. It
can render one or more selected AIA wavelengths as an ordered panel grid and
overlay Gaussian fits from one or more selected radio frequencies on every
panel; the fitted centers and contours are independently switchable, and the
contour percentage is user-controlled. No FWHM ellipse or contour is drawn
unless the contour control is enabled. It automatically matches every selected
ROI imaging frequency to a CSO or DART narrow band with a user-controlled
bandwidth. Dual-axis flux curves can be combined or rendered as one aligned row
per frequency. The dynamic video uses the first selected radio frequency's real
observation times, rematches and redraws every AIA/radio top frame, and moves
the aligned UTC markers across every flux row and the spectrum. The PNG, CSV
exports, metadata, and video retain the shared UTC axis without resampling or
interpolation.

When Miniforge is already activated and the prompt is in the `Python`
directory, use the exact relative launcher form:

```powershell
# (solarphysics_env_latest) <repo>\Python>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ..\Apps\run.ps1 frontend radio-composite
```

### Bad Frame, DART, and Source Trajectory

The bad-frame reviewer preserves automatic assessment, human labels, and model
predictions as separate evidence. DART keeps logarithmic dB, relative-linear,
and Stokes-ratio semantics explicit. Source Trajectory preserves its scientific
frame, band, polarization, and matching controls. Their visual styling is
shared, but their scientific display contracts are not replaced by the spatial
radio-map contract.

## Privacy and publication policy

Only source, tests, synthetic fixtures, documentation, the empty configuration
template, and required license notices belong in `Apps`. The following remain
ignored and must not be published:

- `Local/` and migration backups;
- observation-year directories and `overview/`;
- personal or machine-specific paths and UI state;
- logs, run histories, job records, caches, workspaces, models, and outputs;
- FITS, spreadsheets, screenshots, videos, databases, and other real data;
- credentials, cookies, tokens, authentication files, and private email;
- historical inventory/manifests and legacy source or test trees.

The Apps distribution contains separately scoped components: general Apps
source is licensed under [MIT](LICENSE), the App 1.0 source under
[GPL-3.0-only](solar_apps/frontends/app_v1/LICENSE.md), and the bundled
Mediabunny asset under MPL-2.0. Complete license texts and the Mediabunny notice
are included in the distribution. PyQt6 remains separately licensed by its
publisher.

## Verification

Use the primary Miniforge environment for compilation, lint, tests, and help
smokes:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m compileall -q Apps/solar_apps Apps/tests
& $Conda run -n solarphysics_env_latest python -m ruff check Apps/solar_apps Apps/tests
& $Conda run -n solarphysics_env_latest python -m pytest Apps/tests --basetemp .\Local\tmp\pytest-apps
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend workbench --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend source-map --help
```

CI performs the same Apps checks in Windows and macOS arm64 Miniforge
environments and also
enforces repository boundaries, package contents, documentation privacy, and
the public Python package contract.

## Troubleshooting

**The launcher cannot find Miniforge**

Pass `-MiniforgeRoot "<miniforge-root>"` on Windows,
`--miniforge-root "<miniforge-root>"` on macOS, or set
`SOLAR_MINIFORGE_ROOT`. Do not
work around the error with system Python.

**The environment is rejected**

Use `solarphysics_env_latest`, or explicitly select `solarphysics_env` for a
compatibility check. Other names are intentionally unsupported.

**A path selector is disabled or opens in a fallback directory**

Run `admin init`, add existing absolute directories to the private allowed-root
configuration, and confirm the remembered directory is still inside those
roots. Manual input is preserved when a native dialog cannot start.

**Saved state is ignored**

State with an invalid schema, namespace, JSON value, or path is discarded
safely. Use Reset UI State to recreate a clean frontend snapshot.

**A browser port remains occupied**

Stop the supervised launcher process before restarting. Preview servers should
not be left running after validation.

**An optional scientific import fails**

Confirm that both `Python` and `Apps` are installed editable in the selected
Miniforge environment and inspect that command's `--help` for optional package
requirements.
