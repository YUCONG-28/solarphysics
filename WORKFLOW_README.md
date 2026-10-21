# Repository maintenance workflow

Start with the [contribution guide](CONTRIBUTING.md) or its
[Chinese counterpart](CONTRIBUTING.zh-CN.md). This guide covers branches,
selective commits, pull requests and the separate release stages. Run commands
from the repository root and replace example arguments with your own values.

## Start a change

Inspect the working tree before switching branches. With a clean checkout:

```bash
git status --short
git switch main
git pull --ff-only origin main
git switch -c docs/example-change
```

Use a descriptive branch name for each change. Preserve existing local work;
use a separate checkout when a branch switch would disturb it. A synchronized
folder does not replace Git history; keep `.git` out of two-way file sync.

## Check and commit selected files

Run the tests relevant to the change, then broaden validation when shared
behavior changes. The [library](Python/CONTRIBUTING.md) and
[Apps](Apps/docs/development.md) guides describe their checks. Development uses
Miniforge `solarphysics_env_latest`; each test or build run needs its own private
output directory.

```bash
git diff
git add -- <path-one> <path-two>
git diff --cached
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python tools/public_source_policy.py --staged
git commit -m "docs: clarify module composition"
```

On Windows, use `<miniforge-root>\Scripts\conda.exe` with the same `run -n`
arguments. The public-source check reads staged Git objects, including deletions
and unusual filenames. Review the complete staged diff and credential checks
as well. Keep observations, event presets, research records, personal paths,
credentials and raw run logs outside public source, as described in the
[public content boundary](CONTRIBUTING.md#public-content-boundary).

## Review and merge

Push the feature branch and open a pull request targeting `main`:

```bash
git push -u origin HEAD
```

Describe the concrete problem, resulting behavior, compatibility effects and
validation limits. Maintainers merge after review and the required CI checks
pass for the current commit. Avoid force-pushing shared history. To incorporate
new work from `main` into a feature branch:

```bash
git fetch origin
git merge origin/main
```

Resolve conflicts deliberately and rerun affected checks. Once the pull request
is merged, update a clean local `main` with `git pull --ff-only origin main`.
Delete a completed local branch with `git branch -d <branch>` only after checking
that its work is merged. Remote branch deletion is a separate repository action.

To unstage a file while keeping its contents:

```bash
git restore --staged -- <path>
```

Do not discard unreviewed work to make a checkout appear clean. If a credential
has entered a commit, rotate it and address the history; deleting the current
file does not remove older copies.

## Quick maintenance commands

These commands use the installed Apps maintenance entry point. On Windows,
replace `./Apps/run.sh` with `Apps\run.ps1`.

| Command | Behavior |
| --- | --- |
| `tools quick check` | Run library `pip check`, compilation and Ruff checks; this supplements the test suites. |
| `tools quick save -m "<message>" -- <paths>` | Stage only the named paths, inspect public-source policy and commit the selected change. |
| `tools quick push` | Push the feature branch and create or reuse its pull request when `gh` is available. |
| `tools quick update -m "<message>" -- <paths>` | Run `check`, `save` and `push` in sequence, stopping on failure. |

`save`, `push` and `update` reject `main`, detached HEAD and an existing staged change
before running their work. This protects a previously prepared index; a failed
save leaves the user's original staging intact. Inspect your diff before using
these non-interactive helpers. They do not run `git add .` or force-push.
Help and invalid command handling do not initialize the application runtime.

## Prepare a version pull request

A normal source or documentation pull request needs no version bump. When a
version change is needed, use a clean feature branch. Preview the change first:

```bash
./Apps/run.sh tools release check
./Apps/run.sh tools release prepare --bump patch --note "<release-note>"
```

`check` and the default `prepare` are read-only and do not fetch remote refs.
`check` uses the cached `origin/main`; `prepare` also inspects remote tags
through a read-only lookup and requires a readable `origin` remote.
`--bump` accepts `patch`, `minor` or `major`. After reviewing the preview:

```bash
./Apps/run.sh tools release prepare --bump patch --note "<release-note>" --execute
```

Execution updates only `Python/solar_toolkit/_version.py`,
`Apps/solar_apps/_version.py` and `Python/CHANGELOG.md`. It does not commit,
create a tag, push or create a GitHub Release. Review and validate those files,
then use the normal save/push pull-request workflow. The old
`tools release run --bump ... --note ...` spelling remains an alias for
`prepare`, with the same limited behavior.

## Publish an accepted version

Publication is a distinct maintainer action after the version pull request is
merged. Use a clean checkout of the full merged commit SHA; a detached release
checkout is supported. Supply the version and that SHA explicitly:

```bash
./Apps/run.sh tools release publish --version <X.Y.Z> --commit <full-merged-commit-sha>
```

The default previews publication after checking that `main` contains the commit,
that the committed versions and changelog match, that the commit belongs to an
accepted pull request merged into `main`, that its required CI checks passed,
and that strict environment-lock and existing-tag checks succeed. These checks
use the selected commit; passing checks for another commit are insufficient.
The publisher requires GitHub access through `gh` for remote verification.

After reviewing that result, add `--execute` to create and push the version tag
and create the GitHub Release. It does not edit source or make a release commit.
A tag or Release already pointing at the same commit can be reused on retry;
a conflicting target is rejected. This documentation does not itself publish
any version.
