# SXR composition example

This small example combines the public SXR loader, smoothing and derivative
helpers with a plot. No input file means a deterministic synthetic CSV is
created in the selected private output directory. It makes no downloads.

From the repository root, in the supported Miniforge environment:

```bash
python Python/examples/sxr/sxr_example.py --output-dir Local/outputs/examples/sxr
```

Choose a new output directory for each synthetic run. To use your own input,
add `--input <data-file>` and optionally both `--start-time` and `--end-time`.
CSV inputs require `time`, `xrsa_flux` and `xrsb_flux`; flux is expressed in
W m^-2. NetCDF inputs require the optional xarray dependency. The example does
not perform instrument calibration or infer a physical relationship from the
derivative.
