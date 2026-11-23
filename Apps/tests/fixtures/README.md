# Test fixtures

Purpose: hold the small, deterministic synthetic inputs that an Apps test may
load, and keep everything that is not one of those inputs out of the public tree.

Reader: anyone adding a file below `Apps/tests/fixtures`.

## Boundary

Only small, deterministic synthetic fixtures belong here. Observation files,
generated products, saved UI state, logs, and machine-specific paths must stay
in the ignored `Local/` runtime tree.

## Rules

- A fixture is read as committed: no observation file, network access, or GUI is
  needed to use it.
- A fixture is small enough to be reviewed in the change that adds it, and
  arrives together with the test that reads it.
- A test that can generate its input at run time does not add a committed copy.
- No fixture records a machine path, a personal path, or a run timestamp.

## Related documents

- [../../README.md](../../README.md) — application partition overview.
- [../../docs/development.md](../../docs/development.md) — how an Apps test run is prepared.
- [../../../ARCHITECTURE.md](../../../ARCHITECTURE.md) — the source/runtime boundary.
