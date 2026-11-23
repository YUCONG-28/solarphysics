# Code repository instructions

Purpose: the public source boundary, the environment rule, and the verification
expectation for every change in this repository.

Reader: any agent or contributor working in this checkout. Partition-level
`Python/AGENTS.md` and `Apps/AGENTS.md` are more specific and win where they
disagree; a direct instruction from the user wins over all of them.


This repository contains composable scientific modules, a small number of
examples, and application adapters. Keep scientific calculations in `Python`,
interfaces and workflow orchestration in `Apps`, and general maintenance
utilities in `tools`.

## Public source boundary

- Publish only reusable code, tests, synthetic examples and necessary technical
  documentation. Do not add research progress, personal literature catalogs,
  real event presets, observations, result reports, operation logs or screenshots
  from research work.
- Require explicit scientific inputs. Keep machine configuration, state,
  workspaces, generated files and validation receipts in ignored `Local` or an
  explicitly configured private runtime root.
- Keep scientific-library imports independent of `solar_apps` and free of
  data downloads, interface startup and analysis execution.
- Preserve public APIs and original licensing when moving code. Event-specific
  configuration belongs to the user, outside the public package.

## Environment and verification

- Use the Miniforge `solarphysics_env_latest` environment for development.
  Use `solarphysics_env` only for an explicitly requested compatibility check.
- Do not use system Python or recreate unsupported environments.
- Read the applicable `Python/AGENTS.md` or `Apps/AGENTS.md` before editing.
- Run relevant tests, then broader suites when shared behavior changes.
  Use distinct private temporary directories for independent test runs.
- Review actual staged contents before committing. Never include private
  migration records or raw test output in public commits or PR descriptions.
- Do not merge or delete branches without explicit user confirmation.

## Related documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — partitions, dependency directions, source/runtime boundary.
- [Python/AGENTS.md](Python/AGENTS.md) — library partition rules.
- [Apps/AGENTS.md](Apps/AGENTS.md) — application partition rules.
- [WORKFLOW_README.md](WORKFLOW_README.md) — commit, review and release procedure.
- [docs/writing-standard.md](docs/writing-standard.md) — documentation classes and claim discipline.
