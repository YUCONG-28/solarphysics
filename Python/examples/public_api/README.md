# Public API Examples / 公共 API 示例

Purpose: show the smallest deterministic compositions of the public API, so an
install can be checked without any observation file.

Reader: whoever has just installed the library and wants a fast, repeatable
smoke check.

These examples are deterministic, need no external observations, and can be
used as small installation smoke checks.

这些示例具有确定性，不依赖外部观测数据，可用于验证安装和公共 API。

## What it demonstrates

- `time_matching_example.py` uses `solar_toolkit.time`.
- `sequence_matching_example.py` builds a fixed hierarchy with `solar_toolkit.time`.
- `gaussian_model_example.py` uses `solar_toolkit.modeling.gaussian`.
- `scientific_contracts_example.py` verifies leap-second durations, nanosecond
  intervals, equivalent frequency units, millisecond spacing and masked samples.
- `native_radio_fit_example.py` fits a synthetic source with explicit pixel ROI
  and helioprojective WCS, then checks its native-pixel center and model.
- `magnetic_models_example.py` generates synthetic PFSS, SHARP and local
  grid files in a new output directory. It checks field-line band clipping
  and the magnetic input readers.

## How to run

After [installing the library](../../README.md#install), run from the repository
root through Miniforge:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/sequence_matching_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/gaussian_model_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/native_radio_fit_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/scientific_contracts_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/magnetic_models_example.py --output-dir Local/magnetic-demo
```

On Windows, use the Miniforge PowerShell invocation in the library guide.

## Expected output

The time example takes an explicit target UTC and a tolerance in seconds,
and returns a filename or `None`. The Gaussian example returns a deterministic
`(size, size)` array from coordinate grids, widths in grid units and an angle
in radians. The magnetic example writes synthetic inputs. The other examples
work in memory.

The sequence example filters a half-open UTC window, orders synthetic streams
by increasing frame count, and prints four output rows. The smallest stream
provides the root; the largest provides output UTC. Each layer covers all of
its parent's frames using monotone minimum-cost matching. Ancestors may repeat,
and their times can differ from output UTC; tolerances flag those differences
without discarding observations. Use the plan's exact references for both
preview and export instead of finding nearest frames again while rendering.

For small multi-module compositions, reuse the existing
[radio](../radio/README.md) and [SXR](../sxr/README.md) examples. Their output
directory must be explicitly provided outside public source or under `Local/`.

## Related documents

- [Python/README.md](../../README.md) — library boundary, install and verification.
- [Python/docs/FUNCTION_MAP.md](../../docs/FUNCTION_MAP.md) — input and output contracts.
- [Python/CONTRIBUTING.md](../../CONTRIBUTING.md) — the library change loop and its checks.
- [radio](../radio/README.md) — synthetic FITS reading, UTC matching and display compaction.
- [SXR](../sxr/README.md) — smoothing, derivatives and plotting over a caller-supplied table.
