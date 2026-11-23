# STEREO EUVI example

Purpose: show how caller-selected EUVI maps are cropped and displayed, without
downloading observations or selecting an event.

Reader: whoever has a local EUVI FITS directory and a private output directory.

`euvi_plot.ipynb` reads caller-selected FITS maps and optionally crops a supplied
helioprojective ROI. It does not download observations or select an event.

## What it demonstrates

- Reading caller-selected FITS maps through the library.
- An optional helioprojective ROI crop supplied by the caller.
- Display settings taken from the input map.

## How to run

Set `EUVI_DATA_DIR` to the input directory and `EUVI_OUTPUT_DIR` to a private
output directory under `Local/` or outside the repository before starting
Jupyter in the supported Miniforge environment.
The library `dev` profile supplies Jupyter. After
[installing it](../../README.md#install), start from the repository root:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
export EUVI_DATA_DIR="<private-input-directory>"
export EUVI_OUTPUT_DIR="<private-output-directory>"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m jupyter notebook Python/examples/stereo/euvi_plot.ipynb
```

On Windows, set the two variables with `$env:EUVI_DATA_DIR` and
`$env:EUVI_OUTPUT_DIR`, and use the Miniforge PowerShell invocation in the
library guide. Use a kernel whose interpreter belongs to
`solarphysics_env_latest`.

## Expected output

`EUVI_ROI_ARCSEC` optionally accepts `[left, bottom, right, top]` in arcseconds;
when omitted the complete input map is plotted. Intensity settings come from the
input map. Review your ROI and calibration conditions for your own data.

Restart the kernel and run every cell to check reproducibility. Missing inputs
fail explicitly. Clear all outputs and execution counts before committing.

## Related documents

- [Python/README.md](../../README.md) — library boundary, install and verification.
- [Python/docs/FUNCTION_MAP.md](../../docs/FUNCTION_MAP.md) — map and WCS input contracts.
- [Python/CONTRIBUTING.md](../../CONTRIBUTING.md) — the library change loop and its checks.
- [Public API](../public_api/README.md) — deterministic smoke checks with no generated files.
- [Radio](../radio/README.md) — synthetic FITS reading, UTC matching and display compaction.
