# Public API Examples / 公共 API 示例

These examples are deterministic, need no external observations, and can be
used as small installation smoke checks.

这些示例具有确定性，不依赖外部观测数据，可用于验证安装和公共 API。

- `time_matching_example.py` uses `solar_toolkit.time`.
- `gaussian_model_example.py` uses `solar_toolkit.modeling.gaussian`.

After [installing the library](../../README.md#install), run from the repository
root through Miniforge:

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/gaussian_model_example.py
```

On Windows, use the Miniforge PowerShell invocation in the library guide.

The time example takes an explicit target UTC and a tolerance in seconds,
and returns a filename or `None`. The Gaussian example returns a deterministic
`(size, size)` array from coordinate grids, widths in grid units and an angle
in radians. Neither example reads an observation, writes generated files or
starts an application.

For small multi-module compositions, reuse the existing
[radio](../radio/README.md) and [SXR](../sxr/README.md) examples. Their output
directory must be explicitly provided outside public source or under `Local/`.
See the [function map](../../docs/FUNCTION_MAP.md) for input and output contracts.
