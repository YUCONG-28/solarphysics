# AIA Radio Composite frontend

`aia-radio-composite` is a retained Streamlit adapter that composes existing
AIA, radio, ROI and spectrum APIs. It displays AIA panels with radio Gaussian
overlays, radio-source ROI light curves and a DART/CSO spectrum on a shared UTC
axis. After filtering the image UTC window, streams form an ascending-count
hierarchy: the smallest is the root, and the largest supplies every output frame
and its observed UTC. Preview and video share exact references from a fixed
full-coverage monotone plan. Ancestors may repeat, empty streams remain explicit,
and scientific samples are not interpolated. Time tolerance flags a difference
instead of discarding an output row.

## Package boundaries

| Module | Responsibility |
| --- | --- |
| `cli.py` | Managed Miniforge process launch |
| `app.py` | Streamlit controls, UI state and allowed-root path selection |
| `application.py` | Request validation, adapter orchestration and exports |
| `sequence.py` | Logical image streams and fixed plans from `solar_toolkit.time` |
| `models/` | Request, result, time-alignment and normalized spectrum contracts |
| `adapters/` | Calls to existing library and workflow APIs |
| `rendering/` | Plotly interaction and static composition from in-memory values |

Rendering must not open FITS files. The application depends on models, adapters
and rendering; `solar_toolkit` remains independent of this frontend.

## Scientific ownership

This package does not implement Gaussian fitting, AIA or Radio FITS readers,
Radio ROI masks, DART readers, or CSO readers. Those capabilities remain in:

- `solar_toolkit.aia`
- `solar_toolkit.radio.gaussian`
- `solar_toolkit.radio.roi_lightcurve`
- `solar_toolkit.radio.dart_spectrogram`
- `solar_toolkit.radio.cso`

Private path authorization is separate from scientific inputs. UI state,
generated figures, metadata and logs stay under `Local/` or an explicitly
configured external private runtime directory. The UI persists only declared
primitive controls and confirmed arcsec ROI. Observation data and generated
scientific products do not belong in this package.

See the [Apps guide](../../../../README.md) for installation and the
[AIA/radio usage guide](../../../../../docs/aia_radio_composite.md) for input
formats, controls and outputs. The [time matching guide](../../../../../docs/time_matching.md)
defines parent-child nearest selection, full-coverage constraints and accumulated
output-relative time differences.
