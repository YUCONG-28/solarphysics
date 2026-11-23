# AIA Radio Composite frontend

Purpose: own the retained Streamlit adapter that composes existing AIA, radio,
ROI and spectrum APIs into one shared-UTC display, and state which capabilities
stay in `solar_toolkit`.

Reader: anyone running or changing `aia-radio-composite`. Input formats,
controls and output files are owned by the
[AIA/radio usage guide](../../../../../docs/aia_radio_composite.md); the selection
rules by the [time matching guide](../../../../../docs/time_matching.md).

`aia-radio-composite` is a retained Streamlit adapter that composes existing
AIA, radio, ROI and spectrum APIs. It displays AIA panels with radio Gaussian
overlays, radio-source ROI light curves and a DART/CSO spectrum on a shared UTC
axis. After filtering the image UTC window, streams form an ascending-count
hierarchy: the smallest is the root, and the largest supplies every output frame
and its observed UTC. Preview and video share exact references from a fixed
full-coverage monotone plan. Ancestors may repeat, empty streams remain explicit,
and scientific samples are not interpolated. Time tolerance flags a difference
instead of discarding an output row.

## 1. Purpose and boundary

| Owned here | Belongs elsewhere |
| --- | --- |
| Streamlit controls, UI state and allowed-root path selection (`app.py`) | Gaussian fitting, AIA and radio FITS readers, radio ROI masks, DART and CSO readers: `solar_toolkit` |
| Request validation, adapter orchestration and exports (`application.py`) | Parent-child nearest selection, full-coverage constraints and accumulated time differences: [time matching guide](../../../../../docs/time_matching.md) |
| Logical image streams and the fixed preview/video plan (`sequence.py`) | Machine path authorization: the `solar_apps` platform services described in the [Apps architecture](../../../../docs/architecture.md) |
| Plotly interaction and static composition from in-memory values (`rendering/`) | Input formats, controls and output files: [AIA/radio usage guide](../../../../../docs/aia_radio_composite.md) |
| Request, result, time-alignment and normalized spectrum contracts (`models/`) | Process launch and Miniforge selection: `Apps/run.sh` or `Apps/run.ps1` |

This package does not implement Gaussian fitting, AIA or Radio FITS readers,
Radio ROI masks, DART readers, or CSO readers. Those capabilities remain in:

- `solar_toolkit.aia`
- `solar_toolkit.radio.gaussian`
- `solar_toolkit.radio.roi_lightcurve`
- `solar_toolkit.radio.dart_spectrogram`
- `solar_toolkit.radio.cso`

## 2. Layout

| Module | Responsibility |
| --- | --- |
| `cli.py` | Managed Miniforge process launch |
| `app.py` | Streamlit controls, UI state and allowed-root path selection |
| `application.py` | Request validation, adapter orchestration and exports |
| `sequence.py` | Logical image streams and fixed plans from `solar_toolkit.time` |
| `models/` | Request, result, time-alignment and normalized spectrum contracts |
| `adapters/` | Calls to existing library and workflow APIs |
| `rendering/` | Plotly interaction and static composition from in-memory values |

## 3. Dependency directions

```text
frontends/radio/aia_radio_composite/app.py
    |---> application.py ------> models/ and adapters/
    |---> rendering/ ----------> in-memory values only

Python/solar_toolkit         -X-> solar_apps
frontends/radio/aia_radio_composite -X-> FITS files opened from `rendering/`
```

Rendering must not open FITS files. The application depends on models, adapters
and rendering; `solar_toolkit` remains independent of this frontend.

## 4. Run and verify

The adapter runs in the Miniforge `solarphysics_env_latest` environment with
Python 3.14, after both source partitions are installed as described in the
[Apps guide](../../../../README.md#install). Start it from the repository root
with `./Apps/run.sh frontend aia-radio-composite` on macOS, or the same command
through `Apps/run.ps1` on Windows, then supply the data and output directories as
arguments.

The launcher help lists the accepted arguments, and `--help` prints them without
starting a server. Private path authorization is separate from scientific
inputs: allowed roots decide which directories may be read, and the AIA
wavelengths, radio frequencies, UTC windows and ROI parameters remain caller
supplied. A run is verified by inspecting the private products below the
configured output directory; the file set and columns are listed in the
[AIA/radio usage guide](../../../../../docs/aia_radio_composite.md).

## 5. Rules that must not be violated

1. Rendering composes from in-memory values and never opens a FITS file.
2. Scientific reads, fitting, masks and extraction stay in `solar_toolkit`; this
   package calls them through `adapters/` and does not reimplement them.
3. `solar_toolkit` never imports this package or any other `solar_apps` frontend.
4. Private path authorization is separate from scientific inputs. UI state,
   generated figures, metadata and logs stay under `Local/` or an explicitly
   configured external private runtime directory.
5. The UI persists only declared primitive controls and confirmed arcsec ROI.
6. Observation data and generated scientific products do not belong in this
   package.

## 6. Related documents

- [Apps guide](../../../../README.md) — installation, the command hierarchy and the runtime boundary.
- [AIA/radio usage guide](../../../../../docs/aia_radio_composite.md) — input formats, controls and output files.
- [Time matching guide](../../../../../docs/time_matching.md) — defines parent-child nearest selection, full-coverage constraints and accumulated output-relative time differences.
- [Apps architecture](../../../../docs/architecture.md) — interface ownership, compatibility entry points and display contracts.
