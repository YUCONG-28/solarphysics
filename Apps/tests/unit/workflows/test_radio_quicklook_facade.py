"""Contracts for the Apps radio quicklook facade."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from solar_apps.workflows.radio import quicklook as apps_quicklook
from solar_apps.workflows.radio.configs import DEFAULT_CONFIG_NAME
from solar_toolkit.radio import quicklook as python_quicklook

APPS_ROOT = Path(__file__).resolve().parents[3]
QUICKLOOK_PATH = APPS_ROOT / "solar_apps" / "workflows" / "radio" / "quicklook.py"


def test_facade_re_exports_python_science_identities() -> None:
    for name in (
        "filter_valid_gaussian_centers",
        "plot_gaussian_center_trajectory",
        "build_quicklook_summary",
    ):
        assert getattr(apps_quicklook, name) is getattr(python_quicklook, name)
    assert apps_quicklook.VALID_CENTERS_NAME == python_quicklook.VALID_CENTERS_NAME
    assert apps_quicklook.HEIGHT_ROWS_NAME == python_quicklook.HEIGHT_ROWS_NAME
    assert apps_quicklook.HEIGHT_PLOT_NAME == python_quicklook.HEIGHT_PLOT_NAME
    assert apps_quicklook.TRAJECTORY_PLOT_NAME == python_quicklook.TRAJECTORY_PLOT_NAME
    assert (
        apps_quicklook.DEFAULT_ANALYSIS_SUBDIR
        == python_quicklook.DEFAULT_ANALYSIS_SUBDIR
    )


def test_facade_preserves_public_api() -> None:
    assert apps_quicklook.__all__ == [
        "build_parser",
        "build_quicklook_config",
        "build_quicklook_summary",
        "filter_valid_gaussian_centers",
        "plot_gaussian_center_trajectory",
        "resolve_gaussian_csv",
        "run_gaussian_newkirk_quicklook",
        "main",
    ]


def test_apps_default_config_name_is_preserved() -> None:
    assert apps_quicklook.DEFAULT_CONFIG_NAME == DEFAULT_CONFIG_NAME
    assert apps_quicklook.build_parser().get_default("config") == DEFAULT_CONFIG_NAME


@pytest.mark.parametrize(
    ("wrapper_name", "python_name", "call"),
    [
        (
            "build_quicklook_config",
            "build_quicklook_config",
            lambda wrapper: wrapper(),
        ),
        (
            "resolve_gaussian_csv",
            "resolve_gaussian_csv",
            lambda wrapper: wrapper(gaussian_csv=None),
        ),
        (
            "run_gaussian_newkirk_quicklook",
            "run_gaussian_newkirk_quicklook",
            lambda wrapper: wrapper(gaussian_csv=None, output_dir="out"),
        ),
    ],
)
def test_wrappers_forward_with_apps_default(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    python_name: str,
    call,
) -> None:
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        python_quicklook,
        python_name,
        lambda *args, **kwargs: captured.append((args, kwargs)) or "result",
    )

    result = call(getattr(apps_quicklook, wrapper_name))

    assert result == "result"
    assert captured
    assert captured[-1][1]["config_name"] == DEFAULT_CONFIG_NAME


def test_cli_parser_exposes_quicklook_arguments() -> None:
    parser = apps_quicklook.build_parser()
    parsed = parser.parse_args(
        ["--gaussian-csv", "diag.csv", "--config", "custom", "--output-dir", "out"]
    )
    assert parsed.gaussian_csv == "diag.csv"
    assert parsed.config == "custom"
    assert parsed.output_dir == "out"

    defaults = parser.parse_args([])
    assert defaults.gaussian_csv is None
    assert defaults.config == DEFAULT_CONFIG_NAME
    assert defaults.output_dir == "quicklook_outputs"


def test_main_forwards_arguments_and_prints_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(*, gaussian_csv, config_name, output_dir):
        calls.append(
            {
                "gaussian_csv": gaussian_csv,
                "config_name": config_name,
                "output_dir": output_dir,
            }
        )
        return {"input_csv": "resolved.csv"}

    monkeypatch.setattr(apps_quicklook, "run_gaussian_newkirk_quicklook", fake_run)
    assert (
        apps_quicklook.main(
            ["--gaussian-csv", "diag.csv", "--config", "custom", "--output-dir", "out"]
        )
        == 0
    )
    assert calls == [
        {"gaussian_csv": "diag.csv", "config_name": "custom", "output_dir": "out"}
    ]
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "Quicklook input: resolved.csv"
    assert lines[1].startswith("Quicklook output: ")


def test_facade_source_does_not_reimplement_science() -> None:
    source = QUICKLOOK_PATH.read_text(encoding="utf-8")
    forbidden_markers = [
        "import pandas",
        "matplotlib",
        "plt.",
        "pd.",
        "build_gaussian_newkirk_height_table",
        "plot_event_gaussian_newkirk_height_comparison",
        "def plot_gaussian_center_trajectory",
        "def build_quicklook_summary",
        "def filter_valid_gaussian_centers",
        "_safe_range",
        "fig, ax",
    ]
    for marker in forbidden_markers:
        assert marker not in source, f"facade re-implements science: {marker!r}"

    tree = ast.parse(source, filename=str(QUICKLOOK_PATH))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert defined <= {
        "build_parser",
        "main",
        "build_quicklook_config",
        "resolve_gaussian_csv",
        "run_gaussian_newkirk_quicklook",
    }
    assert "from solar_toolkit.radio import quicklook as _quicklook_impl" in source


def test_radio_dispatcher_quicklook_entrypoint_unchanged() -> None:
    from solar_apps.workflows.radio.dispatcher import _COMMANDS

    assert _COMMANDS["quicklook"] == ("solar_apps.workflows.radio.quicklook", "main")
