# Solar Physics Applications

`Apps` contains application adapters and workflow orchestration for the reusable
[`solar_toolkit`](../Python/README.md) library. Use the library to compose your
own analysis; the interfaces demonstrate ways to connect those APIs. Supply
scientific inputs and parameters explicitly, and keep observations, configuration,
state and generated products in private storage.

## Install

Apps require Python 3.14, Miniforge, and `solarphysics_env_latest`. The public
launchers are `Apps/run.ps1` on Windows and `Apps/run.sh` on macOS. The
`solarphysics_env` environment is available only for an explicitly selected
compatibility check. The launchers do not fall back to system Python or a
virtual environment.

Run the following commands from the repository root after replacing
`<miniforge-root>` with your Miniforge installation directory.

Windows:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda env update -n solarphysics_env_latest -f .\Apps\environment.miniforge.yml
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-ml]"
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Apps
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 admin init
```

macOS:

```bash
"<miniforge-root>/bin/conda" env update -n solarphysics_env_latest -f Apps/environment.miniforge.yml
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m pip install -e "./Python[quality-ml]"
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python -m pip install -e ./Apps
./Apps/run.sh admin init
```

These commands resolve dependency ranges for development. For hash-verified
platform dependencies, follow the [environment guide](../environment/README.md).
An exploratory installation is not an exact environment replay.

If Miniforge is outside its default location, set `SOLAR_MINIFORGE_ROOT`, or
pass `-MiniforgeRoot "<miniforge-root>"` to the Windows launcher or
`--miniforge-root "<miniforge-root>"` to the macOS launcher before the command
group. Use `-EnvironmentName solarphysics_env` or
`--environment-name solarphysics_env` only for a compatibility check. Child
processes inherit the selected Miniforge interpreter.

## Private paths and scientific configuration

`admin init` creates the ignored runtime directories, copies
[`paths.example.yaml`](configs/examples/paths.example.yaml) to
`Local/configs/paths.local.yaml`, and creates private launcher forwarders.
The template has an empty `apps.allowed_roots` list. Add the absolute data and
output directories that this machine should allow:

```yaml
apps:
  runtime_layout_version: 2
  allowed_roots:
    - /absolute/path/to/data
    - /absolute/path/to/output
```

Use native absolute paths on Windows. Allowed roots resolve in this order:
explicit `--allowed-roots`, `SOLAR_APPS_ALLOWED_ROOTS`, then the private YAML
file. Multiple roots use the operating system's path separator: `;` on Windows
and `:` on macOS. The repository root, its ancestors and drive roots are
rejected. Paths and remembered directories are revalidated before access.

Keep runtime files under `Local/` or outside the repository. Set
`SOLAR_APPS_LOCAL_ROOT` to select an external private runtime directory.
`RuntimeLayout` and the [synthetic examples](examples/README.md) validate resolved
destinations before creating files, including symbolic links and each example's
derived output files.

Path authorization does not provide scientific configuration. Observation files,
UTC ranges, regions and analysis choices must come from the caller. Radio
workflow entry points accept a private JSON object with `--config-file`, an
importable caller-owned module with `--config`, or an explicit mapping supplied
by an application adapter. JSON is data-only; a module executes trusted Python
configuration code. These are separate from `paths.local.yaml`.

Inspect the chosen command's help before preparing its configuration:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 workflow radio source-map --help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 workflow radio source-map --config-file "<private-config-dir>\radio.json"
```

```bash
./Apps/run.sh workflow radio source-map --help
./Apps/run.sh workflow radio source-map --config-file "<private-config-dir>/radio.json"
```

A JSON file may contain a `user` section and the model sections required by the
chosen workflow. There are no built-in observation presets; absent scientific
configuration raises an explicit error. Other domains expose their own input
options rather than a universal configuration format.

## Commands and interfaces

The command hierarchy is:

```text
Apps/run.ps1 frontend <frontend-id> [arguments]
Apps/run.ps1 workflow <domain> [command] [arguments]
Apps/run.ps1 admin <command> [arguments]
Apps/run.ps1 tools <command> [arguments]
```

Use `./Apps/run.sh` with the same arguments on macOS. Workflow domains are
`aia`, `radio`, `hmi`, `net`, `data`, `visualization`, `xray-dem` and `pfss`.
Run a group or command with `--help` to inspect its options.

The native demonstration interface starts with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1
```

```bash
./Apps/run.sh frontend app-v1
```

It adapts existing workflows through typed forms and supervised workers. Open a
particular page with `--module`, for example
`frontend app-v1 --module image-composer`. The `app-v1-preview` ID is a
compatibility alias to the same implementation. Retained interfaces are listed
below; inspect their help for allowed-root, host, port and browser options.

| Command | Interface |
| --- | --- |
| `frontend workbench` | Workbench and Radio Workspace, Flask |
| `frontend image-viewer` | Sequence viewer and media export, Flask |
| `frontend image-composer` | Native Image Composer compatibility route |
| `frontend bad-frame-review` | Radio frame review, Flask |
| `frontend source-map` | Source Map preparation and ROI annotation, Flask |
| `frontend dart-spectrogram` | DART spectrum controls, Streamlit |
| `frontend roi-lightcurve` | Radio ROI light curves, Streamlit |
| `frontend radio-composite` | Multi-frequency composite display, Streamlit |
| `frontend aia-radio-composite` | AIA/radio/ROI/spectrum display, Streamlit |
| `frontend source-trajectory` | Radio trajectory inspection, Streamlit |

The browser interfaces are deprecated compatibility surfaces. Browser servers
bind to loopback by default; stop the supervising process with `Ctrl+C` when
finished. Close native interfaces through their application window. Current
interface ownership, formats and compatibility aliases are documented in
[Apps architecture](docs/architecture.md).

STEREO EUVI plotting uses `python-preprocess` by default. It preserves header
pointing and is not full instrument calibration. `legacy` remains accepted for
compatibility. The optional `secchi-prep` mode requires an explicitly configured
SolarSoft installation and IDL executable; it stops if calibration is unavailable
or fails. Those external dependencies are installed separately.

The PFSS viewer opens with `frontend app-v1 --module pfss`. Computing a result
uses `workflow pfss` and separate optional dependencies. A visible-disk LOS
magnetogram is not a global radial boundary; follow the
[PFSS scientific contract](../Python/solar_toolkit/modeling/pfss/README.md).
For composite input formats and controls, see the
[AIA/radio guide](../docs/aia_radio_composite.md).

## State, display and troubleshooting

`StateStore` persists allow-listed current fields as versioned JSON. Corrupt or
incompatible state is ignored. `RecentPathMemory` restores a usable directory
only after checking current allowed roots. **Reset UI State** clears the saved
frontend snapshot. Neither store is a research record or operation history.

Auto, Light and Dark themes affect interface chrome. The native app also offers
Dark Dimmed. Scientific arrays, colormaps, normalization and exports retain their
own settings. Spatial radio panels use `SpatialRadioDisplay`; its effective
settings and cache signatures are described in the architecture guide.

| Symptom | Check |
| --- | --- |
| Miniforge cannot be found | Set `SOLAR_MINIFORGE_ROOT` or pass the launcher's Miniforge-root option. |
| A path control is disabled | Configure allowed roots and confirm that the requested path exists inside them. |
| Scientific configuration is missing | Supply the selected workflow's explicit inputs or private configuration. |
| An optional import fails | Install both source partitions in the selected environment and inspect the command's dependency requirements. |
| A browser port remains occupied | Stop the previous supervising process before restarting. |

## Development and licenses

Build, checks, health reports and maintenance commands are documented in
[Apps development](docs/development.md). Only code, tests, synthetic examples,
technical documentation and required notices belong in the public tree; keep
observations, machine paths, state, logs, workspaces and generated products
private.

General Apps source is [MIT](LICENSE); App 1.0 source is
[GPL-3.0-only](solar_apps/frontends/app_v1/LICENSE.md). The bundled Mediabunny
asset is MPL-2.0 and includes its license and notice. PyQt6 is separately licensed
by its publisher. Preserve these component notices when redistributing code.
