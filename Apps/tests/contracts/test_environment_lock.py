"""Fail-closed contracts for platform-specific environment evidence."""

from __future__ import annotations

import subprocess
import sys
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_ROOT = REPO_ROOT / "environment" / "locks"
LOCK_TOOL = REPO_ROOT / "tools" / "environment_lock.py"
LOCK_MANUAL = REPO_ROOT / "environment" / "README.md"


@pytest.fixture(scope="module")
def lock_module():
    spec = importlib.util.spec_from_file_location("public_environment_lock", LOCK_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_environment_gate_matches_committed_lock_state() -> None:
    targets = sorted(path for path in LOCK_ROOT.glob("*") if path.is_dir())
    completed = subprocess.run(
        [
            sys.executable,
            str(LOCK_TOOL),
            "check",
            "--require-artifact-hashes",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if targets:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode == 2
        assert "no committed platform locks found" in completed.stderr
        normalized_manual = " ".join(LOCK_MANUAL.read_text(encoding="utf-8").split())
        assert "exact-environment gate is therefore **red**" in normalized_manual


def test_lock_workflow_preserves_artifact_and_source_boundaries() -> None:
    source = LOCK_TOOL.read_text(encoding="utf-8")
    manual = LOCK_MANUAL.read_text(encoding="utf-8")

    assert '"--explicit", "--sha256"' in source
    assert '"python", "-m", "pip", "check"' in source
    assert "pip_artifacts_sha256" in source
    assert "--require-hashes" in manual
    assert "--no-build-isolation" in manual
    assert "must not describe that environment as exact" in manual
    assert "/Users/" not in manual
    assert "C:\\Users\\" not in manual


def test_capture_requires_explicit_apply_and_has_no_install_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, str(LOCK_TOOL), "capture", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--apply" in completed.stdout

    capture_source = (
        LOCK_TOOL.read_text(encoding="utf-8")
        .split("def _capture", maxsplit=1)[1]
        .split("def _check_target", maxsplit=1)[0]
    )
    assert '"install"' not in capture_source
    assert '"download"' not in capture_source


def test_lock_profiles_include_local_build_backends(lock_module) -> None:
    requirements = {
        str(requirement) for requirement in lock_module._source_requirements()
    }

    assert "setuptools>=77" in requirements
    assert "wheel" in requirements or "wheel>=0.45" in requirements


def test_fresh_capture_invalidates_an_older_pip_seal(lock_module, tmp_path) -> None:
    for name in ("pip-hashed.txt", "pip-artifacts.json"):
        (tmp_path / name).write_text("stale", encoding="utf-8")
    unrelated = tmp_path / "conda-explicit.txt"
    unrelated.write_text("keep", encoding="utf-8")

    lock_module._invalidate_pip_seal(tmp_path)

    assert not (tmp_path / "pip-hashed.txt").exists()
    assert not (tmp_path / "pip-artifacts.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_pep660_editables_accept_only_this_reviewed_checkout(lock_module) -> None:
    payload = (
        "["
        f'{{"name":"solar-physics-toolkit","editable_project_location":'
        f'"{(REPO_ROOT / "Python").as_posix()}"}},'
        f'{{"name":"solarphysics-apps","editable_project_location":'
        f'"{(REPO_ROOT / "Apps").as_posix()}"}}'
        "]"
    )
    observed = lock_module._validate_editable_projects(
        payload, require_current_checkout=True
    )
    assert set(observed) == {"solar-physics-toolkit", "solarphysics-apps"}

    with pytest.raises(lock_module.LockError, match="exactly"):
        lock_module._validate_editable_projects(
            payload[:-1]
            + ',{"name":"third-party","editable_project_location":"/tmp/third"}]',
            require_current_checkout=True,
        )


def test_freeze_parser_ignores_only_known_local_editable_paths(lock_module) -> None:
    freeze = "\n".join(
        (
            f"-e {(REPO_ROOT / 'Python').as_posix()}",
            f"-e file://{(REPO_ROOT / 'Apps').as_posix()}",
            "numpy==2.5.1",
        )
    )
    assert lock_module._freeze_to_pins(freeze, set()) == {"numpy": "2.5.1"}

    with pytest.raises(lock_module.LockError, match="unrecognized editable"):
        lock_module._freeze_to_pins("-e /tmp/third-party", set())


@pytest.fixture
def synthetic_locks(lock_module, monkeypatch, tmp_path):
    """Build complete, independently synthetic sealed locks without a runtime."""
    monkeypatch.setattr(lock_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lock_module, "LOCK_ROOT", tmp_path / "environment" / "locks")
    project = """
[build-system]
requires = ["setuptools>=77", "wheel>=0.45"]
[project]
requires-python = ">=3.14"
dependencies = ["numpy>=1"]
[project.optional-dependencies]
dev = ["ruff>=0.15.12,<0.17"]
quality-ml = []
"""
    for relative, text in (
        ("Apps/pyproject.toml", project),
        ("Python/pyproject.toml", project),
        ("Apps/environment.miniforge.yml", "name: synthetic\n"),
        ("tools/environment_lock.py", "# synthetic source\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    pins = {"numpy": "1.0", "ruff": "0.15.12", "setuptools": "77.0", "wheel": "0.45"}
    targets = ("linux-64-py314", "osx-arm64-py314")
    for target in targets:
        platform = target.rsplit("-py", maxsplit=1)[0]
        directory = lock_module.LOCK_ROOT / target
        directory.mkdir(parents=True)
        explicit = (
            f"# platform: {platform}\n@EXPLICIT\n"
            f"https://conda.anaconda.org/conda-forge/{platform}/"
            f"python-3.14.0-synthetic_0.conda#{'0' * 64}\n"
        ).encode()
        artifacts = [
            {
                "name": name,
                "version": version,
                "filename": f"{name}-{version}-py3-none-any.whl",
                "sha256": lock_module._sha256_bytes(name.encode()),
                "size": 100,
            }
            for name, version in pins.items()
        ]
        files = {
            "conda-explicit.txt": explicit,
            "pip-pins.txt": lock_module._render_pins(pins, target),
            "pip-hashed.txt": (
                "\n".join(
                    f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"
                    for item in artifacts
                )
                + "\n"
            ).encode(),
            "pip-artifacts.json": lock_module._canonical_json(
                {
                    "schema": "solarphysics-pip-artifacts/v1",
                    "target": target,
                    "platform": platform,
                    "python_full_version": "3.14.0",
                    "artifacts": artifacts,
                }
            ),
        }
        for name, content in files.items():
            (directory / name).write_bytes(content)
        marker = {
            "implementation_name": "cpython",
            "implementation_version": "3.14.0",
            "os_name": "posix",
            "platform_machine": "x86_64" if platform == "linux-64" else "arm64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux" if platform == "linux-64" else "Darwin",
            "python_full_version": "3.14.0",
            "python_version": "3.14",
            "sys_platform": "linux" if platform == "linux-64" else "darwin",
        }
        receipt = {
            "schema": lock_module.SCHEMA,
            "target": target,
            "platform": platform,
            "python_full_version": "3.14.0",
            "implementation": "CPython",
            "marker_environment": marker,
            "profiles": {
                relative: list(extras)
                for relative, extras in sorted(lock_module.SOURCE_PROFILES.items())
            },
            "source_files": lock_module._source_hashes(),
            "lock_files": {
                name: lock_module._sha256_bytes(content)
                for name, content in files.items()
            },
            "conda_artifacts_exact": True,
            "conda_artifacts_sha256": True,
            "pip_versions_exact": True,
            "pip_artifacts_sha256": True,
            "capture_mode": "read-only-observation-of-installed-environment",
            "local_source_install": "editable-no-deps-no-build-isolation",
        }
        receipt["environment_lock_sha256"] = lock_module._environment_lock_sha(receipt)
        (directory / "lock-receipt.json").write_bytes(
            lock_module._canonical_json(receipt)
        )
    with (tmp_path / "Apps" / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write("\n# independent synthetic source change\n")
    monkeypatch.setattr(
        lock_module,
        "_run",
        lambda _command: pytest.fail(
            "source refresh must not inspect or replay a runtime"
        ),
    )
    return lock_module, targets


def _lock_snapshot(module) -> dict[str, bytes]:
    return {
        path.relative_to(module.LOCK_ROOT).as_posix(): path.read_bytes()
        for path in module.LOCK_ROOT.rglob("*")
        if path.is_file()
    }


def test_refresh_sources_preview_preserves_all_evidence(
    synthetic_locks, capsys
) -> None:
    module, targets = synthetic_locks
    before = _lock_snapshot(module)

    assert module.main(["refresh-sources"]) == 0

    assert _lock_snapshot(module) == before
    output = capsys.readouterr().out
    assert "preview only; add --apply" in output
    assert "no environment capture or replay performed" in output
    for target in targets:
        assert f"would refresh {target}:" in output
        with pytest.raises(module.LockError, match="stale"):
            module._check_target(target)


def test_refresh_sources_apply_changes_only_source_binding(
    synthetic_locks, monkeypatch
) -> None:
    module, targets = synthetic_locks
    before = _lock_snapshot(module)

    assert module.main(["refresh-sources", "--apply"]) == 0

    after = _lock_snapshot(module)
    for target in targets:
        receipt_name = f"{target}/lock-receipt.json"
        old = json.loads(before[receipt_name])
        new = module._check_target(target)
        assert {key for key in new if old[key] != new[key]} == {
            "source_files",
            "environment_lock_sha256",
        }
        assert new["source_files"] == module._source_hashes()
        assert new["environment_lock_sha256"] == module._environment_lock_sha(new)
    assert {
        name: content
        for name, content in before.items()
        if not name.endswith("lock-receipt.json")
    } == {
        name: content
        for name, content in after.items()
        if not name.endswith("lock-receipt.json")
    }
    monkeypatch.setattr(
        module,
        "_atomic_write",
        lambda *_args: pytest.fail("an unchanged source binding must not be rewritten"),
    )
    assert module.main(["refresh-sources", "--apply"]) == 0


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("first_target_matches_conda", [False, True])
def test_refresh_sources_rejects_conda_source_changes_without_any_writes(
    synthetic_locks, capsys, apply, first_target_matches_conda
) -> None:
    module, targets = synthetic_locks
    conda_source = "Apps/environment.miniforge.yml"
    source_path = module.REPO_ROOT / conda_source
    source_path.write_text(
        "name: synthetic\ndependencies:\n  - python=3.99\n", encoding="utf-8"
    )
    if first_target_matches_conda:
        # A valid earlier target can queue an update before a later one fails.
        path = module.LOCK_ROOT / targets[0] / "lock-receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["source_files"][conda_source] = module._sha256_path(source_path)
        receipt["environment_lock_sha256"] = module._environment_lock_sha(receipt)
        path.write_bytes(module._canonical_json(receipt))
    before = _lock_snapshot(module)

    arguments = ["refresh-sources", "--apply"] if apply else ["refresh-sources"]
    assert module.main(arguments) == 2

    output = capsys.readouterr()
    assert "Conda source specification changed" in output.err
    assert "use capture" in output.err
    assert not output.out
    assert _lock_snapshot(module) == before


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("artifact_checksum", "lock checksum mismatch"),
        ("pip_metadata", "does not match version pin"),
        ("combined_checksum", "combined environment lock SHA mismatch"),
        ("source_map", "invalid source hash map"),
        ("profiles", "dependency profiles disagree"),
        ("marker", "marker environment Python versions disagree"),
        ("platform", "Conda platform and receipt disagree"),
        ("implementation", "unsupported Python implementation"),
        ("capture_mode", "unsupported capture mode"),
        ("local_source_install", "unsupported local source installation mode"),
        ("constraints", "source constraints reject locked versions"),
    ),
)
def test_refresh_sources_validates_every_target_before_writing(
    synthetic_locks, capsys, failure, message
) -> None:
    module, targets = synthetic_locks
    directory = module.LOCK_ROOT / targets[-1]
    receipt_path = directory / "lock-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    if failure == "artifact_checksum":
        with (directory / "pip-pins.txt").open("a", encoding="utf-8") as stream:
            stream.write("# changed artifact\n")
    elif failure == "pip_metadata":
        artifact_path = directory / "pip-artifacts.json"
        artifacts = json.loads(artifact_path.read_bytes())
        artifacts["artifacts"][0]["version"] = "2.0"
        artifact_path.write_bytes(module._canonical_json(artifacts))
        receipt["lock_files"][artifact_path.name] = module._sha256_path(artifact_path)
    elif failure == "source_map":
        receipt["source_files"]["tools/environment_lock.py"] = "invalid"
    elif failure == "profiles":
        receipt["profiles"]["Apps/pyproject.toml"] = []
    elif failure == "marker":
        receipt["marker_environment"]["implementation_version"] = "3.13.0"
    elif failure == "platform":
        receipt["platform"] = "linux-64"
    elif failure == "constraints":
        project_path = module.REPO_ROOT / "Python" / "pyproject.toml"
        project_path.write_text(
            project_path.read_text(encoding="utf-8").replace("numpy>=1", "numpy>=2"),
            encoding="utf-8",
        )
    else:
        receipt[failure] = "invalid"
    if failure == "combined_checksum":
        receipt["environment_lock_sha256"] = "0" * 64
    else:
        receipt["environment_lock_sha256"] = module._environment_lock_sha(receipt)
    receipt_path.write_bytes(module._canonical_json(receipt))
    before = _lock_snapshot(module)

    assert module.main(["refresh-sources", "--apply"]) == 2

    assert message in capsys.readouterr().err
    assert _lock_snapshot(module) == before


def test_refresh_sources_can_select_one_target(synthetic_locks) -> None:
    module, targets = synthetic_locks
    other_receipt = module.LOCK_ROOT / targets[-1] / "lock-receipt.json"
    other_receipt.write_text("invalid", encoding="utf-8")
    before = other_receipt.read_bytes()

    assert module.main(["refresh-sources", "--target", targets[0], "--apply"]) == 0

    assert module._check_target(targets[0])["source_files"] == module._source_hashes()
    assert other_receipt.read_bytes() == before
