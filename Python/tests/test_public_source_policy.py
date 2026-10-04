"""Synthetic Git/index and archive publication-boundary regressions."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

POLICY = Path(__file__).resolve().parents[2] / "tools" / "public_source_policy.py"
spec = importlib.util.spec_from_file_location("public_policy", POLICY)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize(
    "module",
    [
        "solar_apps",
        "PyQt5.QtCore",
        "PyQt6",
        "PySide2",
        "PySide6",
        "flask",
        "Flask",
        "streamlit",
    ],
)
def test_installed_smoke_rejects_application_imports(module):
    smoke_spec = importlib.util.spec_from_file_location(
        "installed_smoke", POLICY.with_name("installed_library_smoke.py")
    )
    smoke = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke)
    with pytest.raises(AssertionError, match="application import"):
        smoke.NoApplicationImports().find_spec(module)


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Synthetic Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    return tmp_path


def test_staged_content_is_inspected_instead_of_worktree(repo):
    path = repo / "source.py"
    private = "/" + "home/fictional-user/data/"
    path.write_text(f"ROOT = {private!r}\n")
    git(repo, "add", "source.py")
    path.write_text("ROOT = None\n")
    assert "personal absolute path" in policy.check_git(repo, staged=True)[0]
    git(repo, "add", "source.py")
    path.write_text(f"ROOT = {private!r}\n")
    assert policy.check_git(repo, staged=True) == []


@pytest.mark.parametrize("dirname", ["Local", "LOCAL", "local-migration-backup"])
def test_casefolded_runtime_names_are_blocked(repo, dirname):
    folder = repo / dirname
    folder.mkdir()
    (folder / "result.py").write_text("VALUE = 1\n")
    git(repo, "add", ".")
    assert "private/generated directory" in policy.check_git(repo, staged=True)[0]


def test_deletions_are_allowed_and_names_with_spaces_are_unambiguous(repo):
    path = repo / "sample result.csv"
    path.write_text("synthetic,value\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Synthetic fixture")
    assert len(policy.check_git(repo)) == 1
    git(repo, "rm", "sample result.csv")
    assert policy.check_git(repo, staged=True) == []


def test_nul_records_preserve_newline_and_tab_names(repo):
    # Windows cannot create these names; build the index with Git plumbing.
    oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"synthetic",
        check=True,
        capture_output=True,
    ).stdout.strip()
    record = b"100644 " + oid + b"\todd\nname\t.csv\0"
    subprocess.run(
        ["git", "update-index", "-z", "--index-info"],
        cwd=repo,
        input=record,
        check=True,
    )
    offenders = policy.check_git(repo, staged=True)
    assert len(offenders) == 1
    assert "odd\\nname\\t.csv" in offenders[0]


def test_git_failure_is_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="inspection failed"):
        policy.check_git(tmp_path, staged=True)


def test_configuration_exceptions_do_not_bypass_content_checks():
    name = "Apps/configs/examples/paths.example.yaml"
    assert policy.check_blob(name, b"allowed_roots: []\n") == []
    assert policy.check_blob("Apps/configs/session.json", b"{}")
    assert policy.check_blob(name, ("root: /" + "home/fictional/data/").encode())
    assert policy.check_blob("demo.ipynb", b'{"cells":[{"outputs":[{}]}]}')


def test_wheel_namespace_and_resources_are_checked(tmp_path):
    wheel = tmp_path / "sample.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("solar_toolkit/__init__.py", '"""Example."""')
        archive.writestr("solar_toolkit/result.csv", "synthetic")
        archive.writestr("solar_apps/__init__.py", "")
    errors = policy.check_archive(wheel, "Python")
    assert any("unapproved package resource" in error for error in errors)
    assert any("unexpected wheel namespace" in error for error in errors)


def test_approved_application_assets_and_licenses_are_required(tmp_path):
    wheel = tmp_path / "sample.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("solar_apps/__init__.py", "")
        for name in policy.APP_ASSETS:
            archive.writestr("solar_apps/" + name, "synthetic")
        archive.writestr(
            "solarphysics_apps-0.dist-info/licenses/LICENSES/GPL-3.0-only.txt",
            "license",
        )
        archive.writestr(
            "solarphysics_apps-0.dist-info/METADATA",
            "License-Expression: MIT AND GPL-3.0-only AND MPL-2.0\n",
        )
    assert policy.check_archive(wheel, "Apps") == []


def test_source_archive_rejects_links_and_traversal(tmp_path):
    import io
    import tarfile

    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("package/../outside.py")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        policy.check_archive(archive, "Python")


@pytest.mark.parametrize(
    "name",
    ["C:/solar_toolkit/__init__.py", "C:relative.py", "../escape/", "/absolute/"],
)
def test_archives_reject_cross_platform_unsafe_paths(tmp_path, name):
    import io
    import tarfile

    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        policy.check_archive(archive, "Python")


def test_archives_cannot_hide_content_with_duplicate_members(tmp_path):
    import io
    import tarfile

    wheel = tmp_path / "sample.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("solar_toolkit/__init__.py", "first")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("solar_toolkit/__init__.py", "second")
    with pytest.raises(ValueError, match="duplicate"):
        policy.check_archive(wheel, "Python")
    source = tmp_path / "sample.tar.gz"
    for roots, reason in [
        (["root", "root"], "duplicate"),
        (["first", "second"], "multiple"),
    ]:
        with tarfile.open(source, "w:gz") as archive:
            for root in roots:
                member = tarfile.TarInfo(root + "/solar_toolkit/__init__.py")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError, match=reason):
            policy.check_archive(source, "Python")
