# Synthetic examples

Purpose: run two deterministic examples that compose application services, and
state where their private output may be written.

Reader: anyone checking a fresh installation, or adding an example to this
directory.

These two examples compose application services with deterministic inputs and
require no observation files or GUI. Install both source partitions as described
in the [Apps guide](../README.md#install), then run from the repository root using
the primary Miniforge environment.

## Daily loop

Windows:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python .\Apps\examples\synthetic_radio_display.py
& $Conda run -n solarphysics_env_latest python .\Apps\examples\synthetic_state_and_paths.py
```

macOS:

```bash
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python Apps/examples/synthetic_radio_display.py
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python Apps/examples/synthetic_state_and_paths.py
```

### Spatial radio display

`synthetic_radio_display.py` builds a deterministic two-source NumPy array,
applies `SpatialRadioDisplay`, and writes a PNG and schema-1 JSON sidecar. It
shows colormap, invalid-value color, transform, percentile, field-of-view and
cache-signature behavior. Both the PNG and its derived JSON destination are
validated before either is written.

Choose another private PNG destination with `--output`:

```powershell
& $Conda run -n solarphysics_env_latest python .\Apps\examples\synthetic_radio_display.py --output .\Local\outputs\examples\custom\radio.png
```

```bash
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python Apps/examples/synthetic_radio_display.py --output Local/outputs/examples/custom/radio.png
```

### State and recent paths

`synthetic_state_and_paths.py` saves current synthetic UI fields with
`StateStore`, remembers one directory with `RecentPathMemory`, and creates fresh
readers to check restart recovery. It validates `synthetic-input`,
`ui_state.json`, `recent_paths.json` and `summary.json` before creating any of
them. Saved content contains only the latest values, without operation history,
logs or scientific results.

Choose another private directory with `--output-dir`:

```powershell
& $Conda run -n solarphysics_env_latest python .\Apps\examples\synthetic_state_and_paths.py --output-dir .\Local\outputs\examples\custom\state
```

```bash
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python Apps/examples/synthetic_state_and_paths.py --output-dir Local/outputs/examples/custom/state
```

## Private output destinations

The examples create their private directories as needed. Defaults write below
`Local/outputs/examples/`; explicit destinations must resolve inside `Local/` or
outside the repository. The repository root and public source directories are
rejected.

Each module exposes `run_demo(...)` and `main(argv=None) -> int`; importing it
creates no files. A reproducible run is one launcher invocation with an explicit
destination. An absolute path outside the repository is also supported.

## Conflict and rollback

Before creating runtime directories or writing any artifact, each example
validates every actual target, including symbolic links and derived files. A
rejected destination therefore stops the run before the first file exists, and
recovering means choosing another private destination and repeating the same
launcher invocation. Nothing in the public tree needs to be undone.

## Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| `conda` cannot be found | `<miniforge-root>` was left as the placeholder | Replace it with the Miniforge installation directory. |
| The example module cannot be imported | Only one source partition is installed | Install both partitions as described in the [Apps guide](../README.md#install). |
| A destination is rejected | The path is the repository root, a public source directory, or resolves outside the private runtime | Choose a path under `Local/` or outside the repository. |
| A script path is not found | The command was not run from the repository root | Run it from the repository root. |

## Pre-delivery checklist

- Both examples ran in `solarphysics_env_latest`, not in a system or virtual
  environment.
- Every output stayed below `Local/` or outside the repository.
- No observation file, saved UI state, log, or generated product was added to the
  public tree.

## Related documents

- [Apps guide](../README.md) — installation, command hierarchy and runtime boundary.
- [Apps development](../docs/development.md) — environment, checks and the health matrix.
- [Apps architecture](../docs/architecture.md) — platform services and display contracts.
