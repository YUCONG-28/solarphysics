# Apps working agreement

Purpose: the application partition's public-source boundary, its runtime
environment rule, and the verification expectation for an Apps change.

Reader: anyone editing `Apps`, including an agent delegated to work there. The
repository-wide rules in `../AGENTS.md` still apply; dependency directions and the
source/runtime boundary are owned by `../ARCHITECTURE.md`.

Precedence: this file is more specific than `../AGENTS.md` where the two differ,
and a direct instruction from the user wins over both.

These instructions apply to the public application partition.

## Boundaries

- Keep reusable scientific calculations in `../Python/solar_toolkit`.
- Keep application code under `solar_apps` and preserve the dependency rules
  documented in `../ARCHITECTURE.md`.
- Keep machine configuration, UI state, workspaces, outputs, logs, and tests'
  generated files under the ignored `../Local` runtime tree.
- Never add observation data, personal paths, operation history, credentials,
  migration evidence, or generated scientific products to `Apps`.
- Keep `configs/examples/paths.example.yaml` as the only public configuration
  template and keep it fail-closed.

## Runtime and commands

- Use `run.ps1` on Windows and `run.sh` on macOS as the public entry points.
- Use `solarphysics_env_latest` by default. Use `solarphysics_env` only for an
  explicit compatibility check.
- Do not add a virtual-environment, system-Python, or arbitrary-interpreter
  fallback, and never select `solarphysics_backup`.
- Ensure subprocesses inherit the interpreter selected by the launcher.

## Verification

- Compile changed Python files, run focused tests, and then run the Apps test
  suite when shared platform, UI, CLI, or workflow behavior changes.
- Use a unique writable pytest base-temporary directory.
- For frontend changes, verify Light, Dark, and Auto modes and shut down every
  supervised preview server before finishing.
- Do not update snapshots or generated artifacts merely to hide a regression.

## Related documents

- [README.md](README.md) — installation, command hierarchy and frontend behavior.
- [docs/architecture.md](docs/architecture.md) — interface ownership, formats and dependency directions.
- [docs/development.md](docs/development.md) — environment, checks and delivery steps.
- [tests/fixtures/README.md](tests/fixtures/README.md) — what may be committed as a test fixture.
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — repository partitions and the forbidden dependency directions.
- [../WORKFLOW_README.md](../WORKFLOW_README.md) — commit, review and release procedure.
