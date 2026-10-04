# Reproducible environment locks

Use the Miniforge `solarphysics_env_latest` environment for normal development.
`Apps/environment.miniforge.yml` and the two `pyproject.toml` files are source
specifications with version ranges. Conda and pip resolve those ranges again
when installing; this supports compatibility testing, but a run must not describe that environment as exact or frozen.

## Validation boundary

An exact-environment claim requires a sealed lock for the target platform,
hash-required installation, a successful detached runtime-verification receipt,
and the applicable tests in that environment. A lock from one platform must
never be relabeled for another platform.

`tools/environment_lock.py check --require-artifact-hashes` validates committed
artifact hashes and their source bindings and reports the combined lock digest.
This offline check does not install the environment or prove a successful replay.
Likewise, CI that resolves source specifications establishes compatibility,
not exact-environment evidence. Validation receipts and test output remain in
private run directories, outside the committed locks.

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
application launches use `solarphysics_env_latest`; the standby environment
`solarphysics_env` is for explicitly requested compatibility checks.

Those commands resolve the current ranges and are not yet an exact replay. Run
the complete relevant tests before capturing the candidate. The examples below
use `osx-arm64-py314`; select the matching target on every command when maintaining
a different platform. Then preview and write the installed-version and
Conda-artifact lock:

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
If any step fails, do not treat the incomplete lock as validated. Correct the
candidate or source specification, then capture and seal again.

## Rebind unchanged artifacts to compatible source requirements

For compatible edits to the two `pyproject.toml` files or the lock tool,
`refresh-sources` can update source hash bindings while preserving artifact
hashes and capture metadata. It validates all selected locks before writing.
The default is a preview; use `--apply` only after reviewing it:

```bash
"$CONDA" run -n solarphysics_env_latest python tools/environment_lock.py refresh-sources
"$CONDA" run -n solarphysics_env_latest python tools/environment_lock.py refresh-sources --apply
"$CONDA" run -n solarphysics_env_latest python tools/environment_lock.py check \
  --require-artifact-hashes
```

This records no new environment capture or replay. A changed
`Apps/environment.miniforge.yml` is rejected and requires the capture workflow
above. Incompatible requirements also require a new candidate and sealed lock.

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

Run the complete package and Apps test suites in that replay environment. The
combined `environment_lock_sha256` and the detached SHA-256 of the generated
replay receipt identify the verification used by a run manifest. The receipt checks the
sealed Conda/pip versions, the selected editable checkout, and `pip check`; it
does not reconstruct an installed wheel archive from `site-packages`, so the
fresh install command must retain `--require-hashes`. Keep the receipt in a
private run directory and generate it against the current combined lock digest;
source rebinding changes that digest even when the artifacts remain unchanged.
