"""Synthetic output-boundary checks for the radio composition example."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


def _snapshot(root: Path) -> dict:
    contents = {}
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            contents[relative] = ("symlink", str(path.readlink()))
        elif path.is_file():
            contents[relative] = ("file", path.read_bytes())
        else:
            contents[relative] = ("directory", None)
    return contents


@pytest.fixture
def example(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "Python").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "examples/radio/synthetic_radio.py"
    namespace = runpy.run_path(str(source))
    monkeypatch.setitem(
        namespace["run_example"].__globals__,
        "__file__",
        str(repository / "Python/examples/radio/synthetic_radio.py"),
    )
    return repository, namespace


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform.")


def test_radio_example_rejects_local_symlink_into_public_source(example, tmp_path):
    repository, namespace = example
    protected = repository / "Python/protected.py"
    protected.write_bytes(b"synthetic protected source\n")
    _symlink(repository / "Local", repository / "Python", directory=True)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="Output must be outside"):
        namespace["run_example"](repository / "Local/new_output")

    assert _snapshot(tmp_path) == before


def test_radio_example_main_writes_fresh_local_output(example):
    repository, namespace = example
    output = repository / "Local/new_output"

    assert namespace["main"](["--output-dir", str(output)]) == 0

    assert {path.name for path in output.iterdir()} == {
        "synthetic.fits",
        "display_arrays.npz",
        "summary.json",
    }
    assert all(path.stat().st_size > 0 for path in output.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert summary["seed"] == 0
    assert summary["epoch_utc"] == "2000-01-01T00:00:00Z"


def test_radio_example_rejects_existing_directory_with_child_symlink(example, tmp_path):
    repository, namespace = example
    output = repository / "Local/existing_output"
    output.mkdir(parents=True)
    protected = repository / "Python/protected.py"
    protected.write_bytes(b"synthetic protected source\n")
    _symlink(output / "summary.json", protected)
    before = _snapshot(tmp_path)

    with pytest.raises(FileExistsError):
        namespace["run_example"](output)

    assert _snapshot(tmp_path) == before
