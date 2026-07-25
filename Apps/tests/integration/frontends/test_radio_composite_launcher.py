"""Tests for the managed Radio Composite Figure launcher."""

from __future__ import annotations

import socket
from types import SimpleNamespace

from solar_apps.frontends.radio.composite_figure import (
    composite_figure_launcher as launcher,
)
from solar_apps.platform.processes import selected_python_executable


def test_build_streamlit_command_forwards_public_directories() -> None:
    args = launcher.build_parser().parse_args(
        [
            "--radio-dir",
            "D:/radio",
            "--dart-dir",
            "D:/dart",
            "--output-dir",
            "D:/outputs",
            "--allowed-roots",
            "D:/radio;D:/dart;D:/outputs",
            "--no-browser",
        ]
    )

    command = launcher.build_streamlit_command(args, port=8765)

    assert command[:4] == [str(selected_python_executable()), "-m", "streamlit", "run"]
    assert "composite_figure_app.py" in command[4]
    assert command[command.index("--server.port") + 1] == "8765"
    assert command[command.index("--radio-dir") + 1] == "D:/radio"
    assert command[command.index("--dart-dir") + 1] == "D:/dart"
    assert command[command.index("--output-dir") + 1] == "D:/outputs"


def test_pick_port_uses_another_port_when_preferred_is_occupied() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        preferred = int(occupied.getsockname()[1])
        selected = launcher._pick_port(preferred)

    assert selected > 0
    assert selected != preferred


def test_dry_run_does_not_start_streamlit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8765)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run started Streamlit")
        ),
    )

    assert launcher.main(["--dry-run", "--no-browser"]) == 0
    output = capsys.readouterr().out
    assert "streamlit run" in output
    assert "composite_figure_app.py" in output


def test_intentional_idle_shutdown_returns_success(monkeypatch) -> None:
    class Process:
        returncode = 1

    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8765)
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *_args, **_kwargs: Process()
    )
    monkeypatch.setattr(
        launcher, "_wait_with_auto_stop", lambda *_args, **_kwargs: True
    )

    assert launcher.main(["--no-browser"]) == 0


def test_browser_connection_check_tolerates_empty_netstat_output(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=None),
    )

    assert launcher._has_browser_connection(8765) is False
