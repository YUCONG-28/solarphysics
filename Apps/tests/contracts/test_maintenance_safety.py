"""Real temporary indexes and mocked remotes for maintenance failure paths."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from solar_apps.cli import quick_update as quick
from solar_apps.cli import release

SHA = "a" * 40
OTHER_SHA = "b" * 40


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Example")
    git(tmp_path, "config", "user.email", "example@example.invalid")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "disabled-hooks"))
    (tmp_path / "base.py").write_text("value = 0\n")
    git(tmp_path, "add", "base.py")
    git(tmp_path, "commit", "-qm", "baseline")
    git(tmp_path, "switch", "-qc", "feature/test")
    return tmp_path


@pytest.mark.parametrize("branch", ["main", "detached"])
def test_quick_save_and_update_reject_before_any_write(repository, monkeypatch, branch):
    if branch == "main":
        git(repository, "switch", "-q", "main")
    else:
        git(repository, "switch", "-q", "--detach")
    (repository / "selected.py").write_text("value = 1\n")
    head = git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(
        quick, "_python_checks", lambda *_: pytest.fail("checks should not run")
    )
    assert quick._quick_save(repository, message="selected", paths=["selected.py"]) == 1
    assert (
        quick._quick_update(repository, message="selected", paths=["selected.py"]) == 1
    )
    assert git(repository, "rev-parse", "HEAD") == head
    assert git(repository, "diff", "--cached", "--name-only") == ""


@pytest.mark.parametrize("existing", ["selected.py", "unrelated.py"])
def test_existing_index_is_preserved_exactly(repository, existing):
    for name in ("selected.py", "unrelated.py"):
        (repository / name).write_text("value = 1\n")
    git(repository, "add", existing)
    (repository / existing).write_text("value = 2\n")
    before = git(repository, "diff", "--cached", "--binary")
    assert quick._quick_save(repository, message="selected", paths=["selected.py"]) == 1
    assert git(repository, "diff", "--cached", "--binary") == before
    assert (repository / existing).read_text() == "value = 2\n"


@pytest.mark.parametrize("failure", ["guard", "commit"])
def test_failed_save_only_unstages_selected_files(repository, monkeypatch, failure):
    (repository / "selected.py").write_text("value = 1\n")
    (repository / "unrelated.py").write_text("value = 2\n")
    head = git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(
        quick,
        "_staged_has_private_content",
        lambda _: ["blocked"] if failure == "guard" else [],
    )
    original = quick._git

    def execute(root, *args, **kwargs):
        if args[0] == "commit":
            return subprocess.CompletedProcess(args, 1)
        return original(root, *args, **kwargs)

    monkeypatch.setattr(quick, "_git", execute)
    assert quick._quick_save(repository, message="selected", paths=["selected.py"]) == 1
    assert git(repository, "rev-parse", "HEAD") == head
    assert git(repository, "diff", "--cached", "--name-only") == ""
    assert (repository / "selected.py").read_text() == "value = 1\n"
    assert (repository / "unrelated.py").read_text() == "value = 2\n"


def test_save_commits_only_explicit_paths(repository, monkeypatch):
    (repository / "selected.py").write_text("value = 1\n")
    (repository / "unrelated.py").write_text("value = 2\n")
    monkeypatch.setattr(quick, "_staged_has_private_content", lambda _: [])
    assert quick._quick_save(repository, message="selected", paths=["selected.py"]) == 0
    assert (
        git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        == "selected.py"
    )
    assert (repository / "unrelated.py").exists()


def test_push_reuses_an_existing_pull_request(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        quick,
        "_git",
        lambda root, *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "" if args[0] == "diff" else "feature/test\n"
        ),
    )
    monkeypatch.setattr("shutil.which", lambda _: "gh")

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps([{"url": "https://example.invalid/pr/1"}])
        )

    monkeypatch.setattr(quick.subprocess, "run", run)
    assert quick._quick_push(tmp_path) == 0
    assert len(calls) == 1 and calls[0][1:3] == ["pr", "list"]


def test_push_refuses_existing_staging_without_touching_it(repository, monkeypatch):
    (repository / "selected.py").write_text("value = 1\n")
    git(repository, "add", "selected.py")
    before = git(repository, "diff", "--cached", "--binary")
    original = quick._git

    def execute(root, *args, **kwargs):
        assert args[0] != "push", "No remote mutation is permitted with staged changes"
        return original(root, *args, **kwargs)

    monkeypatch.setattr(quick, "_git", execute)
    assert quick._quick_push(repository) == 1
    assert git(repository, "diff", "--cached", "--binary") == before


def test_source_inspector_launch_failure_cleans_only_this_staging(
    repository, monkeypatch
):
    (repository / "selected.py").write_text("value = 1\n")
    monkeypatch.setattr(
        quick,
        "_staged_has_private_content",
        lambda _: (_ for _ in ()).throw(OSError("inspector could not start")),
    )
    with pytest.raises(OSError, match="inspector"):
        quick._quick_save(repository, message="selected", paths=["selected.py"])
    assert git(repository, "diff", "--cached", "--name-only") == ""
    assert (repository / "selected.py").read_text() == "value = 1\n"


@pytest.fixture
def version_tree(tmp_path, monkeypatch):
    for name in (release.TOOLKIT_VERSION, release.APPS_VERSION):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text('__version__ = "0.3.0"\n')
    (tmp_path / release.CHANGELOG).write_text(
        "# Changelog\n\n## Unreleased\n\n- Added a reusable module.\n\n## 0.3.0\n\n- Before.\n"
    )
    monkeypatch.setattr(release, "_check_clean", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(release, "_strict_locks", lambda _: None)
    monkeypatch.setattr(release, "_tag_targets", lambda *_: (None, None))
    return tmp_path


def version_contents(root):
    return {
        name: (root / name).read_bytes()
        for name in (release.TOOLKIT_VERSION, release.APPS_VERSION, release.CHANGELOG)
    }


def test_prepare_preview_and_legacy_run_never_write_or_publish(
    version_tree, monkeypatch
):
    commands = []

    def execute(root, *args, **kwargs):
        commands.append(args)
        assert args == ("rev-parse", "--abbrev-ref", "HEAD")
        return "main\n"

    monkeypatch.setattr(release, "_git", execute)
    before = version_contents(version_tree)
    assert (
        release._prepare_release(
            version_tree, bump="patch", note="Release", execute=False
        )
        == 0
    )
    assert (
        release._run_release(version_tree, bump="patch", note="Release", execute=False)
        == 0
    )
    assert version_contents(version_tree) == before
    assert len(commands) == 2


def test_prepare_execute_requires_feature_branch_and_only_edits_review_files(
    version_tree, monkeypatch
):
    monkeypatch.setattr(release, "_git", lambda *_args, **_kwargs: "main\n")
    before = version_contents(version_tree)
    with pytest.raises(RuntimeError, match="feature branch"):
        release._prepare_release(
            version_tree, bump="patch", note="Release", execute=True
        )
    assert version_contents(version_tree) == before
    monkeypatch.setattr(release, "_git", lambda *_args, **_kwargs: "release/version\n")
    assert (
        release._run_release(version_tree, bump="patch", note="Release", execute=True)
        == 0
    )
    assert release._current_version(version_tree) == "0.3.1"
    assert "## 0.3.1 - " in (version_tree / release.CHANGELOG).read_text()


@pytest.mark.parametrize("failure", ["tag", "locks"])
def test_prepare_conflict_or_lock_failure_never_changes_files(
    version_tree, monkeypatch, failure
):
    monkeypatch.setattr(release, "_git", lambda *_args, **_kwargs: "release/version\n")
    if failure == "tag":
        monkeypatch.setattr(release, "_tag_targets", lambda *_: (None, SHA))
    else:
        monkeypatch.setattr(
            release,
            "_strict_locks",
            lambda _: (_ for _ in ()).throw(RuntimeError("invalid locks")),
        )
    before = version_contents(version_tree)
    with pytest.raises(RuntimeError):
        release._prepare_release(
            version_tree, bump="patch", note="Release", execute=True
        )
    assert version_contents(version_tree) == before


@pytest.fixture
def publication(tmp_path, monkeypatch):
    state = {
        "local": None,
        "remote": None,
        "existing": None,
        "writes": [],
        "release_calls": 0,
        "pr": True,
        "checks": "success",
        "compare": "identical",
        "push_failures": 0,
        "release_failures": 0,
    }
    monkeypatch.setattr(release, "_current_version", lambda _: "0.3.1")
    monkeypatch.setattr(release, "_strict_locks", lambda _: None)
    monkeypatch.setattr(release, "_which", lambda _: "gh")
    monkeypatch.setattr(
        release, "_tag_targets", lambda *_: (state["local"], state["remote"])
    )

    def execute(root, *args, **kwargs):
        if args == ("rev-parse", "HEAD"):
            return SHA
        if args == ("status", "--short"):
            return ""
        if args[0] == "show":
            return "# Changelog\n\n## 0.3.1 - 2000-01-01\n\n- Reviewed changes.\n"
        state["writes"].append(args)
        if args[0] == "tag":
            state["local"] = SHA
        elif args[0] == "push":
            if state["push_failures"]:
                state["push_failures"] -= 1
                raise RuntimeError("simulated push failure")
            state["remote"] = SHA
        else:
            pytest.fail(f"Unexpected git write: {args}")
        return ""

    def github(root, endpoint, **kwargs):
        if "/compare/" in endpoint:
            return {"status": state["compare"]}
        if "/pulls?" in endpoint:
            return (
                [
                    {
                        "merged_at": "2000-01-01",
                        "base": {"ref": "main"},
                        "merge_commit_sha": SHA,
                    }
                ]
                if state["pr"]
                else []
            )
        if "/check-runs?" in endpoint:
            checks = [
                {
                    "id": i,
                    "name": name,
                    "status": "completed",
                    "conclusion": state["checks"],
                }
                for i, name in enumerate(release.REQUIRED_CHECKS)
            ]
            return {"check_runs": checks, "total_count": len(checks)}
        if "/releases/tags/" in endpoint:
            return state["existing"]
        pytest.fail(endpoint)

    def publish(command, **kwargs):
        assert command[:3] == ["gh", "release", "create"]
        assert "--verify-tag" in command and SHA in command
        state["release_calls"] += 1
        if state["release_failures"]:
            state["release_failures"] -= 1
            raise RuntimeError("simulated Release failure")
        state["existing"] = {
            "tag_name": "v0.3.1",
            "draft": False,
            "html_url": "https://example.invalid/release",
        }

    monkeypatch.setattr(release, "_git", execute)
    monkeypatch.setattr(release, "_github_json", github)
    monkeypatch.setattr(release, "_run", publish)
    return tmp_path, state


def publish(publication, execute=True):
    root, _ = publication
    return release._publish_release(root, version="0.3.1", commit=SHA, execute=execute)


def test_publish_preview_is_read_only(publication):
    assert publish(publication, execute=False) == 0
    assert publication[1]["writes"] == []
    assert publication[1]["release_calls"] == 0


@pytest.mark.parametrize(
    "failure", ["pr", "checks", "main", "local-tag", "remote-tag", "draft"]
)
def test_publish_rejects_before_any_tag_write(publication, failure):
    _, state = publication
    if failure == "pr":
        state["pr"] = False
    elif failure == "checks":
        state["checks"] = "failure"
    elif failure == "main":
        state["compare"] = "behind"
    elif failure == "local-tag":
        state["local"] = OTHER_SHA
    elif failure == "remote-tag":
        state["remote"] = OTHER_SHA
    else:
        state["existing"] = {"tag_name": "v0.3.1", "draft": True}
    with pytest.raises(RuntimeError):
        publish(publication)
    assert state["writes"] == [] and state["release_calls"] == 0


@pytest.mark.parametrize("failure", ["push_failures", "release_failures"])
def test_publish_retries_partial_failures_without_retagging(publication, failure):
    _, state = publication
    state[failure] = 1
    with pytest.raises(RuntimeError, match="simulated"):
        publish(publication)
    assert publish(publication) == 0
    assert sum(args[0] == "tag" for args in state["writes"]) == 1
    assert all(args[:3] != ("push", "origin", "main") for args in state["writes"])
    writes = list(state["writes"])
    assert publish(publication) == 0
    assert state["writes"] == writes


def test_publish_lock_failure_precedes_tag_operations(publication, monkeypatch):
    monkeypatch.setattr(
        release,
        "_strict_locks",
        lambda _: (_ for _ in ()).throw(RuntimeError("invalid locks")),
    )
    with pytest.raises(RuntimeError, match="locks"):
        publish(publication)
    assert publication[1]["writes"] == []


def test_github_404_is_distinct_from_authentication_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "_which", lambda _: "gh")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "request failed (HTTP 401)"
        ),
    )
    with pytest.raises(RuntimeError, match="authentication"):
        release._github_json(tmp_path, "releases/tags/v0.3.1", missing_ok=True)
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "request failed (HTTP 404)"
        ),
    )
    assert (
        release._github_json(tmp_path, "releases/tags/v0.3.1", missing_ok=True) is None
    )


def test_ci_requires_offline_strict_lock_validation():
    repo = Path(__file__).resolve().parents[3]
    assert (
        "python ../tools/environment_lock.py check --require-artifact-hashes"
        in (repo / ".github/workflows/ci.yml").read_text()
    )
