# Project Instructions

Purpose: the library partition's environment rule, its API discipline, and the
verification expectation for a change under `Python/`.

Reader: anyone editing `Python/`, including an agent delegated to work there. The
repository-wide rules in `../AGENTS.md` still apply; dependency directions and
the source/runtime boundary are owned by `../ARCHITECTURE.md`.

Precedence: this file is more specific than `../AGENTS.md` where the two differ,
and a direct instruction from the user wins over both.

These instructions apply to this Python project and its subdirectories.

## Python environment

- Always run Python through the Miniforge `solarphysics_env_latest` environment.
- Resolve Miniforge explicitly in PowerShell:

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
```

- Keep `solarphysics_env` as the formal compatibility environment. Switch to it
  only through explicit Miniforge `conda run -n solarphysics_env` when
  a comparison or compatibility fallback is required; it is not the default.

- Run commands through Miniforge so Conda supplies the environment DLL paths:

```powershell
& $Conda run -n solarphysics_env_latest python -m pytest
& $Conda run -n solarphysics_env_latest python -m pip
& $Conda run -n solarphysics_env_latest python script.py
```

- Do not fall back to a virtual environment, system Python, or bare
  `python`/`pip`/`pytest` command.

## Codex agent behavior

Use these behavioral guidelines when writing, reviewing, or refactoring code in
this repository. Project-specific instructions above take priority if any
guideline conflicts with them.

- Think before coding: state assumptions explicitly, surface tradeoffs, and ask
  for clarification when the task has multiple plausible interpretations.
- Simplicity first: prefer the smallest implementation that solves the stated
  problem, without speculative features or unnecessary abstractions.
- Surgical changes: touch only what the task requires, match the surrounding
  style, and avoid unrelated refactors or formatting churn.
- Goal-driven execution: define concrete success criteria, verify them with the
  appropriate checks, and do not claim completion without fresh evidence.

## Branch completion workflow

- If a branch is created while working on this project, do not silently leave it behind at the end of the task.
- Before finishing, ask the user whether they want to merge the branch into `main` and delete the branch.
- Do not merge into `main` or delete the branch without the user's explicit confirmation.
- If the user confirms, switch to `main`, merge the work branch, then delete the completed local branch.
- If a remote branch was pushed, ask for confirmation before deleting the remote branch too.

## Radio and verification workflow

- Prefer `solar_toolkit.radio` for new reusable radio-processing imports.
- Do not add old internal `scripts.*` compatibility modules. Public application
  aliases belong in the `Apps/run.ps1` CLI router.
- For radio work, optimize for denoising and clear radio-source visibility on
  AIA imagery, not only for visually prettier Gaussian overlays.
- Keep literature/formula notes mapped to the actual code path and clearly
  separate implemented behavior from future work.
- For focused radio changes, prefer targeted tests first, such as relevant
  `tests\test_radio_*.py` files, then broaden only when shared behavior changed.
- If a broader suite fails, isolate whether the failure is in the changed path
  before treating it as a task blocker.

## Cross-repository and presentation routing

- Use Codex, not a Work deliverable, as the completion surface for code edits,
  tests, compatibility checks, and Git state in this repository.
- Keep research records and personal literature catalogs outside public source.
  Describe methods through their implemented API contracts and general conditions.
- Use `academic-paper-director` for manuscript or citation-sensitive text and
  `ppt-style-director` for research-deck style. Create or edit a real `.pptx`
  with `presentation-skill` or `Presentations`, followed by render-based QA.
- Do not expose raw observations, large generated outputs, local paths, or
  private research artifacts through Calendar or Sites. Publish only a
  separately reviewed export after an explicit request.

## Related documents

- [README.md](README.md) — library boundary, install and verification.
- [CONTRIBUTING.md](CONTRIBUTING.md) — the library change loop and its checks.
- [docs/FUNCTION_MAP.md](docs/FUNCTION_MAP.md) — public modules and their input and output contracts.
- [CHANGELOG.md](CHANGELOG.md) — the dated record of library changes.
- [../AGENTS.md](../AGENTS.md) — repository-wide source boundary and environment rule.
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — partitions and the forbidden dependency directions.
- [../docs/writing-standard.md](../docs/writing-standard.md) — document classes and claim discipline.
