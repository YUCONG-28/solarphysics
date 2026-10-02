# STEREO EUVI example

`euvi_plot.ipynb` reads caller-selected FITS maps and optionally crops a supplied
helioprojective ROI. It does not download observations or select an event.

Set `EUVI_DATA_DIR` to the input directory and `EUVI_OUTPUT_DIR` to a private
output directory before starting Jupyter in the supported Miniforge environment.
`EUVI_ROI_ARCSEC` optionally accepts `[left, bottom, right, top]` in arcseconds;
when omitted the complete input map is plotted. Intensity settings come from the
input map. Review your ROI and calibration conditions for your own data.

Restart the kernel and run every cell to check reproducibility. Missing inputs
fail explicitly. Clear all outputs and execution counts before committing.
