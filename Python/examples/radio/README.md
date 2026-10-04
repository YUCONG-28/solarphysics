# Synthetic radio composition

This small example composes bounded CSO FITS extraction, UTC matching and
optional missing-column display compaction. It creates its own inputs with seed
0 at the neutral `2000-01-01` epoch; no observations or application are required.

After [installing the library](../../README.md#install), run from the repository
root through Miniforge:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/radio/synthetic_radio.py --output-dir "Local/outputs/examples/radio-<run>"
```

On Windows, use the Miniforge PowerShell invocation in the library guide.
Replace `<run>` with a unique run identifier.

The output directory must be explicitly selected and new. Use the ignored
`Local` directory or a private directory outside the repository. The example
writes a tiny synthetic FITS file, display arrays and a JSON record.

`relative_background_db` keeps the original UTC axis. The separate
`compact_spectrum_time` display transform omits entirely missing columns and
returns their original indices and a UTC cursor mapping. Choose these functions
independently when composing an analysis.
