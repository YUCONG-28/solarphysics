# STEREO EUVI example

`euvi_plot.ipynb` reads caller-selected FITS maps and optionally crops a supplied
helioprojective ROI. It does not download observations or select an event.

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

`EUVI_ROI_ARCSEC` optionally accepts `[left, bottom, right, top]` in arcseconds;
when omitted the complete input map is plotted. Intensity settings come from the
input map. Review your ROI and calibration conditions for your own data.

Restart the kernel and run every cell to check reproducibility. Missing inputs
fail explicitly. Clear all outputs and execution counts before committing.
