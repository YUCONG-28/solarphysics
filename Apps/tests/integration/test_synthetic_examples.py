from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from solar_apps.platform.layout import RuntimeLayout

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "Apps" / "examples"


def _load_example(name: str) -> ModuleType:
    path = EXAMPLES_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"apps_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(tmp_path: Path) -> RuntimeLayout:
    root = tmp_path / "workspace"
    (root / "Apps").mkdir(parents=True)
    (root / "Python").mkdir()
    return RuntimeLayout.discover(root, environ={}).ensure()


def test_examples_are_import_safe_and_write_only_private_outputs(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    radio = _load_example("synthetic_radio_display")
    state = _load_example("synthetic_state_and_paths")

    assert list(layout.outputs_dir.rglob("*")) == []

    radio_result = radio.run_demo(layout=layout, size=48)
    state_result = state.run_demo(layout=layout)

    assert radio_result["image"].is_file()
    assert radio_result["sidecar"].is_file()
    assert state_result["summary"].is_file()
    assert radio_result["image"].is_relative_to(layout.outputs_dir)
    assert state_result["summary"].is_relative_to(layout.outputs_dir)
    assert list(layout.apps_root.rglob("*")) == []


@pytest.mark.parametrize(
    ("example_name", "argument_name", "filename"),
    (
        ("synthetic_radio_display", "output", "blocked.png"),
        ("synthetic_state_and_paths", "output_dir", "blocked"),
    ),
)
def test_examples_reject_outputs_inside_apps(
    tmp_path: Path,
    example_name: str,
    argument_name: str,
    filename: str,
) -> None:
    layout = _layout(tmp_path)
    module = _load_example(example_name)

    with pytest.raises(ValueError, match="inside Local/ or outside the repository"):
        module.run_demo(layout=layout, **{argument_name: layout.apps_root / filename})


@pytest.mark.parametrize(
    ("example_name", "argument_name", "filename"),
    (
        ("synthetic_radio_display", "output", "blocked.png"),
        ("synthetic_state_and_paths", "output_dir", "blocked"),
    ),
)
@pytest.mark.parametrize(
    "partition", ("Apps", "Python", "tools", "docs", "results", "")
)
def test_examples_validate_output_before_creating_runtime_directories(
    tmp_path: Path, example_name: str, argument_name: str, filename: str, partition: str
) -> None:
    repo = tmp_path / "workspace"
    layout = RuntimeLayout.discover(repo, environ={})
    module = _load_example(example_name)
    destination = repo / partition / filename if partition else repo
    with pytest.raises(ValueError, match="inside Local/"):
        module.run_demo(layout=layout, **{argument_name: destination})
    assert not repo.exists()


@pytest.mark.parametrize(
    ("example_name", "argument_name", "filename"),
    (
        ("synthetic_radio_display", "output", "display.png"),
        ("synthetic_state_and_paths", "output_dir", "state"),
    ),
)
def test_examples_reject_external_symlink_into_public_source(
    tmp_path: Path, example_name: str, argument_name: str, filename: str
) -> None:
    repo = tmp_path / "workspace"
    source = repo / "Python"
    source.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    layout = RuntimeLayout.discover(repo, environ={})
    with pytest.raises(ValueError, match="inside Local/"):
        _load_example(example_name).run_demo(
            layout=layout, **{argument_name: alias / filename}
        )
    assert not layout.local_root.exists()
    assert list(source.iterdir()) == []


@pytest.mark.parametrize(
    ("example_name", "argument_name", "filename", "artifact"),
    (
        ("synthetic_radio_display", "output", "display.png", "image"),
        ("synthetic_state_and_paths", "output_dir", "state", "summary"),
    ),
)
def test_examples_accept_explicit_external_outputs(
    tmp_path: Path, example_name: str, argument_name: str, filename: str, artifact: str
) -> None:
    layout = _layout(tmp_path)
    destination = tmp_path / "private" / filename
    result = _load_example(example_name).run_demo(
        layout=layout, **{argument_name: destination}
    )
    assert result[artifact].is_file()
    assert result[artifact].is_relative_to(tmp_path / "private")


def _filesystem_snapshot(root: Path) -> dict[str, tuple[str, bytes | str | None]]:
    snapshot: dict[str, tuple[str, bytes | str | None]] = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[name] = ("symlink", str(path.readlink()))
        elif path.is_file():
            snapshot[name] = ("file", path.read_bytes())
        else:
            snapshot[name] = ("directory", None)
    return snapshot


@pytest.mark.parametrize(
    ("example_name", "argument_name", "destination_name", "target_name"),
    (
        ("synthetic_radio_display", "output", "image.png", "image.png"),
        ("synthetic_radio_display", "output", "image.png", "image.json"),
        ("synthetic_state_and_paths", "output_dir", "state", "synthetic-input"),
        ("synthetic_state_and_paths", "output_dir", "state", "ui_state.json"),
        ("synthetic_state_and_paths", "output_dir", "state", "recent_paths.json"),
        ("synthetic_state_and_paths", "output_dir", "state", "summary.json"),
    ),
)
def test_examples_validate_every_write_target_before_any_filesystem_change(
    tmp_path: Path,
    example_name: str,
    argument_name: str,
    destination_name: str,
    target_name: str,
) -> None:
    repo = tmp_path / "workspace"
    source = repo / "Python"
    source.mkdir(parents=True)
    destination = tmp_path / "private" / destination_name
    parent = destination.parent if argument_name == "output" else destination
    parent.mkdir(parents=True)
    alias = parent / target_name
    directory_target = target_name == "synthetic-input"
    source_target = source / target_name
    if directory_target:
        source_target.mkdir()
        (source_target / "protected.txt").write_bytes(b"synthetic sentinel\n")
    else:
        source_target.write_bytes(b"synthetic sentinel\n")
    try:
        alias.symlink_to(source_target, target_is_directory=directory_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")
    layout = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "runtime")}
    )
    example = _load_example(example_name)
    before = _filesystem_snapshot(tmp_path)

    with pytest.raises(ValueError, match="inside Local/"):
        example.run_demo(layout=layout, **{argument_name: destination})

    assert _filesystem_snapshot(tmp_path) == before
    assert not layout.local_root.exists()
