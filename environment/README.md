# Reproducible environment locks

`Apps/environment.miniforge.yml` and the two `pyproject.toml` files are source
specifications, not lock files. They contain version ranges and therefore ask
Conda and pip to solve again. That is suitable for compatibility testing, but
it is not sufficient evidence for a confirmatory research run.

## Current gate

The exact-environment gate is **green** for `osx-arm64-py314`. Its committed
lock seals 18 Conda artifacts and 206 pip wheels, and a fresh environment has
passed hash-required installation, runtime verification, `pip check`, and the
complete Python and Apps test suites. Run `tools/environment_lock.py check
--require-artifact-hashes` to read the current combined lock digest.

A run may claim this exact environment only when it installs the sealed lock
with `--require-hashes` and produces a successful detached replay receipt.
Source-specification installs remain suitable for exploration, but a run must not describe that environment as exact or frozen.
An earlier candidate failed replay because it
combined incompatible PyQt6 and PySide6 runtimes; it has been superseded by the
single-PyQt6 lock and is not release evidence.

The first supported lock target is `osx-arm64-py314`. Windows and Linux jobs
remain compatibility checks until independently generated and replay-tested
locks for those platforms are committed. A lock from one platform must never
be relabeled for another platform.

## Evidence model

Each completed `environment/locks/<target>/` directory contains:

- `conda-explicit.txt`: exact conda-forge artifact URLs with SHA-256 fragments;
- `pip-pins.txt`: exact versions observed in the validated candidate;
- `pip-hashed.txt`: exact versions plus the selected wheel SHA-256 values;
- `pip-artifacts.json`: wheel filenames, sizes, versions, and SHA-256 values;
- `lock-receipt.json`: target, Python version, source-specification hashes,
  component hashes, and one combined `environment_lock_sha256`.

`pip-pins.txt` alone is only an intermediate version lock. It becomes a sealed
artifact lock only after `seal-pip` has authenticated one compatible wheel for
every pin. Strict checks reject an unsealed directory.

The lock never contains a user path, environment prefix, editable-source URL,
credential, or package-cache location. The two repository distributions are
installed from the reviewed checkout only after all third-party artifacts.

## Create or refresh a lock

Lock maintenance is a deliberate online maintenance operation. It is not part
of a scientific run. Use a new disposable target-platform environment; do not
capture a long-lived personal environment merely because it has a familiar
name.

From the repository root, create the candidate from the source specifications
and install the declared profiles:

```bash
CONDA="<miniforge-root>/bin/conda"
TARGET_ENV="solarphysics_lock_candidate"
export SOLAR_MINIFORGE_ROOT="<miniforge-root>"
export SOLAR_APPS_ALLOW_LOCK_CANDIDATE=1
"$CONDA" env create -n "$TARGET_ENV" -f Apps/environment.miniforge.yml
"$CONDA" run -n "$TARGET_ENV" python -m pip install -e "./Python[dev,quality-ml]"
"$CONDA" run -n "$TARGET_ENV" python -m pip install -e "./Apps[dev]"
"$CONDA" run -n "$TARGET_ENV" python -m pip check
```

The lock-candidate opt-in is accepted only for the exact disposable environment
name above and still requires an explicitly verified Miniforge root. Normal
application launches continue to accept only the primary or formal standby
environment.

Those commands resolve the current ranges and are not yet reproducible. Run
the complete relevant tests before observing the candidate. Then preview and
write the installed-version and Conda-artifact lock:

```bash
"$CONDA" run -n "$TARGET_ENV" python tools/environment_lock.py capture \
  --conda "$CONDA" \
  --environment "$TARGET_ENV" \
  --target osx-arm64-py314
"$CONDA" run -n "$TARGET_ENV" python tools/environment_lock.py capture \
  --conda "$CONDA" \
  --environment "$TARGET_ENV" \
  --target osx-arm64-py314 \
  --apply
```

`capture` is read-only with respect to the environment. It runs `pip check`,
checks every direct requirement in the selected profiles, rejects missing or
incompatible packages, and excludes the two editable local distributions. It
does not install, upgrade, download, or solve anything.

Download exactly one compatible wheel for every emitted pin into a temporary
directory outside the repository, then preview and seal the pip artifacts:

```bash
WHEELHOUSE="<temporary-wheelhouse>"
"$CONDA" run -n "$TARGET_ENV" python -m pip download \
  --only-binary=:all: \
  --no-deps \
  --requirement environment/locks/osx-arm64-py314/pip-pins.txt \
  --dest "$WHEELHOUSE"
"$CONDA" run -n "$TARGET_ENV" python tools/environment_lock.py seal-pip \
  --target osx-arm64-py314 \
  --wheelhouse "$WHEELHOUSE" \
  --conda "$CONDA" \
  --environment "$TARGET_ENV"
"$CONDA" run -n "$TARGET_ENV" python tools/environment_lock.py seal-pip \
  --target osx-arm64-py314 \
  --wheelhouse "$WHEELHOUSE" \
  --conda "$CONDA" \
  --environment "$TARGET_ENV" \
  --apply
"$CONDA" run -n "$TARGET_ENV" python tools/environment_lock.py check \
  --require-artifact-hashes
```

Never hand-edit hashes or invent a result for a package that was not present.
If any step fails, leave the exact-environment gate red, correct the candidate
or source specification, and start the capture again.

## Recreate from a sealed lock

The following is the fresh-environment path after the target directory exists
and the strict check passes:

```bash
CONDA="<miniforge-root>/bin/conda"
REPLAY_ENV="solarphysics_replay"
export SOLAR_MINIFORGE_ROOT="<miniforge-root>"
export SOLAR_APPS_ALLOW_LOCK_REPLAY=1
"$CONDA" create -n "$REPLAY_ENV" \
  --file environment/locks/osx-arm64-py314/conda-explicit.txt
"$CONDA" run -n "$REPLAY_ENV" python -m pip install \
  --only-binary=:all: \
  --require-hashes \
  --requirement environment/locks/osx-arm64-py314/pip-hashed.txt
"$CONDA" run -n "$REPLAY_ENV" python -m pip install \
  --no-deps \
  --no-build-isolation \
  --editable ./Python \
  --editable ./Apps
"$CONDA" run -n "$REPLAY_ENV" python -m pip check
"$CONDA" run -n "$REPLAY_ENV" python tools/environment_lock.py verify-runtime \
  --conda "$CONDA" \
  --environment "$REPLAY_ENV" \
  --target osx-arm64-py314 \
  --receipt "<run-directory>/environment-replay.json"
"$CONDA" run -n "$REPLAY_ENV" python tools/environment_lock.py verify-runtime \
  --conda "$CONDA" \
  --environment "$REPLAY_ENV" \
  --target osx-arm64-py314 \
  --receipt "<run-directory>/environment-replay.json" \
  --apply
```

Run the complete package and Apps test suites in that replay environment. Only
the combined `environment_lock_sha256` and the detached SHA-256 of the generated
replay receipt may be bound into a formal run manifest. The receipt proves the
sealed Conda/pip versions, the selected editable checkout, and `pip check`; it
does not reconstruct an installed wheel archive from `site-packages`, so the
fresh install command must retain `--require-hashes`. Editing a source
environment file makes `check` fail until the lock is regenerated and replayed.
