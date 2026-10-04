# Contributing

Thank you for helping improve this solar-physics toolkit. Reusable helpers live
in `solar_toolkit/`, tracked application workflows live in `../Apps/`, and
data-independent library checks live in `tests/`.

## Environment

The project is developed with Miniforge and the `solarphysics_env_latest` environment:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".[dev]"
```

The retained `solarphysics_env` environment is the formal backup. Select it
explicitly only for compatibility comparison or fallback; current development
commands continue to target `solarphysics_env_latest`.

The `.[dev]` extra is the minimum contributor install for tests and style
checks. Use `.[dev,full]` when validating broader science workflows that need
the optional NetCDF, image, radio, or download helpers.

Use the same Miniforge executable with `conda run` from non-activated shells;
do not fall back to a system interpreter.

## Before Committing

Run the lightweight checks that do not require local FITS/NetCDF data:

```powershell
& $Conda run -n solarphysics_env_latest python -m compileall -q solar_toolkit tests examples
& $Conda run -n solarphysics_env_latest python -m pytest -q tests
& $Conda run -n solarphysics_env_latest python -m pre_commit run --all-files
```

`pre-commit` runs whitespace, end-of-file, YAML, large-file, Gitleaks, Ruff, and
Black checks, so style, formatting, and obvious data-policy problems are caught
before code reaches shared branches.

The public-source hook inspects actual staged Git blobs, including changes
hidden by different working-tree contents. Deletions are allowed. CI inspects
the committed tree with the same rules; neither check replaces a review of
scientific context or the separate secret scan. Start pre-commit from the
selected Miniforge environment so its local hook uses that interpreter.

Build validation is separate from editable-source tests. From the repository
root, use `tools/build_distributions.py --project Python --output-dir` with a
new private directory through the selected Miniforge interpreter. It checks
the wheel and source archive, rebuilds the wheel outside the checkout, and
checks its contents again. CI installs that rebuilt wheel into a fresh
Miniforge environment without Apps, then runs the four existing synthetic/API
examples through `tools/installed_library_smoke.py`. These builds are local
validation artifacts and are not uploaded as releases.

## Code Style

- Use Black as the Python formatter.
- Use Ruff for linting, import sorting, and safe automatic fixes.
- Use Pylance in VSCode for type analysis, navigation, and completions.
- Keep autopep8, yapf, flake8, and pylint as manual fallback tools only; do not
  configure them as the default formatter or linter for this project.

## Data and Paths

- Do not commit FITS, NetCDF, generated plots, videos, Excel files, caches, or local data products.
- Keep machine paths in ignored `../Local/configs/paths.local.yaml`.
- Start from the fail-closed template in `../Apps/configs/examples`.

## Workflow Changes

- Keep Apps workflows runnable through `Apps/run.ps1` and `Apps/run.sh`.
- Prefer adding reusable helpers to `solar_toolkit/` when multiple workflows need them.
- Examples that require local data belong in `examples/`; tests in `tests/` must be data-independent.
