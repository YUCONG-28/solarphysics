# SXR composition example

This small example combines the public SXR loader, smoothing and derivative
helpers with a plot. No input file means a deterministic synthetic CSV is
created in the selected private output directory. It makes no downloads.

After [installing the library](../../README.md#install), run from the repository
root through Miniforge:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/sxr/sxr_example.py --output-dir "Local/outputs/examples/sxr-<run>"
```

On Windows, use the Miniforge PowerShell invocation in the library guide.
Replace `<run>` with a unique identifier and choose a new output directory for
each synthetic run. Outputs may also go to a private directory outside the
repository. To use your own input,
add `--input <data-file>` and optionally `--start-time`, `--end-time`, or both.
Each bound is inclusive; a missing bound leaves that side open. Offset-aware
times are converted to UTC and naive times are interpreted as UTC. The plotted
window must retain at least five samples for the smoothing step.
CSV inputs require `time`, `xrsa_flux` and `xrsb_flux`; flux is expressed in
W m^-2. NetCDF inputs need xarray and a compatible reader backend; the `full`
profile includes xarray and netCDF4. The example does
not perform instrument calibration or infer a physical relationship from the
derivative.
