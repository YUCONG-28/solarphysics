# AIA Radio Composite frontend

This package provides the `aia-radio-composite` local web frontend for a
three-panel solar-research product:

1. One or more AIA wavelength panels with time-matched, quality-controlled
   radio Gaussian overlays.
2. Multi-frequency intensity curves from an ROI drawn directly on the
   time-matched raw radio FITS image with the established ROI-lightcurve
   selector (the AIA composite PNG is preview-only).
3. A DART or CSO dynamic spectrum aligned to the same UTC interval.

The UI exposes Gaussian-center/contour controls, raw-radio display
percentiles, independent flux and spectrum UTC extraction windows, a shared
UTC display range with a reference-time marker, and synchronized MP4 export.
The first selected radio frequency supplies the video's real observation
timeline. Every frame rematches all selected radio frequencies and AIA
wavelengths, redraws the top grid, and moves both lower UTC markers. Incomplete
matches are skipped and scientific samples are never interpolated.

## Package boundaries

```text
cli.py
  managed process launch only

application.py
  UI state, path policy, adapter orchestration, and exports

models/schema.py
  validated request, result, and normalized spectrum contracts

adapters/
  calls existing solar_toolkit and solar_apps workflow APIs

rendering/
  Plotly interaction and static three-panel composition
```

Dependencies flow from `application` to `models`, `adapters`, and `rendering`.
Adapters may call `solar_toolkit` or established `solar_apps.workflows` APIs.
Rendering consumes normalized in-memory values and must not open FITS files.
`solar_toolkit` never imports this frontend.

## Scientific ownership

This package does not implement Gaussian fitting, AIA or Radio FITS readers,
Radio ROI masks, DART readers, or CSO readers. Those capabilities remain in:

- `solar_toolkit.aia`
- `solar_toolkit.radio.gaussian`
- `solar_toolkit.radio.roi_lightcurve`
- `solar_toolkit.radio.dart_spectrogram`
- `solar_toolkit.radio.cso`

Runtime paths, UI state, generated figures, metadata, and logs belong under the
ignored `Local` tree or an explicitly allowed user output directory. No
observation data or generated scientific products belong in this package.
The Streamlit UI uses the shared native path dialog and persists only its
declared primitive controls and confirmed arcsec ROI under `Local/state`.

The installation, data-format, operation, and output guide is documented in
`docs/aia_radio_composite.md`.
