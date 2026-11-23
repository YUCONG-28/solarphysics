# Synthetic radio composition

Purpose: show one bounded radio composition end to end, from a synthetic FITS
cube to a matched UTC frame and a compacted display spectrum.

Reader: whoever wants a working radio example without observations or the
applications.

This small example composes bounded CSO FITS extraction, UTC matching and
optional missing-column display compaction. It creates its own inputs with seed
0 at the neutral `2000-01-01` epoch; no observations or application are required.

## What it demonstrates

- Bounded CSO FITS extraction through `solar_toolkit.radio`.
- UTC matching across synthetic frames.
- `relative_background_db` and `compact_spectrum_time` as separate display
  transforms.

## How to run

After [installing the library](../../README.md#install), run from the repository
root through Miniforge:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/radio/synthetic_radio.py --output-dir "Local/outputs/examples/radio-<run>"
```

On Windows, use the Miniforge PowerShell invocation in the library guide.
Replace `<run>` with a unique run identifier.

## Expected output

The output directory must be explicitly selected and new. Use the ignored
`Local` directory or a private directory outside the repository. The example
writes a tiny synthetic FITS file, display arrays and a JSON record. It also
writes `native_polarization.npz` with one column per native observation time
and checks the percent polarization against the channel sums of the input cube.

`relative_background_db` keeps the original UTC axis. The separate
`compact_spectrum_time` display transform omits entirely missing columns and
returns their original indices and a UTC cursor mapping. Choose these functions
independently when composing an analysis.

## Related documents

- [Python/README.md](../../README.md) — library boundary, install and verification.
- [Python/docs/FUNCTION_MAP.md](../../docs/FUNCTION_MAP.md) — radio input and output contracts.
- [Python/CONTRIBUTING.md](../../CONTRIBUTING.md) — the library change loop and its checks.
- [Public API](../public_api/README.md) — deterministic smoke checks with no generated files.
- [SXR](../sxr/README.md) — smoothing, derivatives and plotting over a caller-supplied table.
