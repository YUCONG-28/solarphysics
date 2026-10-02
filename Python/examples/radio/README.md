# Synthetic radio composition

This small example composes bounded CSO FITS extraction, UTC matching and
optional missing-column display compaction. It creates its own inputs with seed
0 at the neutral `2000-01-01` epoch; no observations or application are required.

Run from `Python` after installing the toolkit:

```sh
python examples/radio/synthetic_radio.py --output-dir ../Local/radio-example
```

The output directory must be explicitly selected and new. Use the ignored
`Local` directory or a private directory outside the repository. The example
writes a tiny synthetic FITS file, display arrays and a JSON record.

`relative_background_db` keeps the original UTC axis. The separate
`compact_spectrum_time` display transform omits entirely missing columns and
returns their original indices and a UTC cursor mapping. Choose these functions
independently when composing an analysis.
