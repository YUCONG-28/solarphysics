# STEREO EUVI example

[`euvi_plot.ipynb`](euvi_plot.ipynb) crops local EUVI FITS maps in helioprojective
coordinates and saves PNG figures. It uses SunPy without downloading observations.

1. Use the Miniforge `solarphysics_env_latest` environment.
2. Open the notebook from within this repository. Set `EUVI_DATA_DIR` before
   starting Jupyter, or edit `euvi_dir` in the input cell. The default is
   `Local/observations/stereo-a/euvi/20250124/171` relative to the repository root.
3. Review the ROI bounds and intensity limits for your observation, then run all
   cells in order. Missing directories and empty FITS selections fail explicitly.
4. Inspect the figures in `Local/outputs/examples/stereo/euvi-roi`. Rerunning
   replaces PNG files with the same input stem; source FITS files are untouched.

For a reproducibility check, restart the kernel and run all cells. Confirm that
one figure is produced per FITS file and that the crop covers the intended region.
The original parameters target the 2025-01-24 example and need review for other
events. Clear outputs and execution counts before committing the notebook.

Return to the [library guide](../../README.md) or [maintenance index](../../../docs/README.md).
