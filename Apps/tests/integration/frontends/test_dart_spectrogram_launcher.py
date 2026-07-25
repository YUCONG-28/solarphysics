"""Tests for the managed DART Streamlit launcher and dispatcher entry."""

from __future__ import annotations

import socket
from types import SimpleNamespace

from solar_apps.frontends.radio.dart_spectrogram import (
    dart_spectrogram_launcher as launcher,
)
from solar_apps.platform.processes import selected_python_executable
from solar_apps.workflows.radio.dispatcher import _COMMANDS
from solar_apps.workflows.radio.dispatcher import (
    build_parser as build_dispatcher_parser,
)


def test_build_streamlit_command_forwards_local_directories() -> None:
    args = launcher.build_parser().parse_args(
        [
            "--input-dir",
            "D:/data/20250124Spec",
            "--output-dir",
            "D:/outputs",
            "--allowed-roots",
            "D:/data;D:/outputs",
            "--no-browser",
        ]
    )
    command = launcher.build_streamlit_command(args, port=8765)

    assert command[:4] == [str(selected_python_executable()), "-m", "streamlit", "run"]
    assert "dart_spectrogram_app.py" in command[4]
    assert command[command.index("--server.port") + 1] == "8765"
    assert command[command.index("--input-dir") + 1] == "D:/data/20250124Spec"
    assert command[command.index("--output-dir") + 1] == "D:/outputs"
    assert command[command.index("--allowed-roots") + 1] == "D:/data;D:/outputs"


def test_pick_port_uses_another_port_when_preferred_is_occupied() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        preferred = int(occupied.getsockname()[1])
        selected = launcher._pick_port(preferred)

    assert selected > 0
    assert selected != preferred


def test_main_opens_browser_and_uses_auto_stop(monkeypatch) -> None:
    process = SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8765)
    launched: list[tuple[list[str], dict]] = []

    def fake_popen(command, **kwargs):
        launched.append((command, kwargs))
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    waited: list[tuple[object, int, float]] = []
    monkeypatch.setattr(
        launcher,
        "_wait_with_auto_stop",
        lambda child, *, port, idle_seconds: waited.append((child, port, idle_seconds)),
    )

    result = launcher.main(["--auto-stop-idle-sec", "12"])

    assert result == 0
    assert opened == ["http://127.0.0.1:8765"]
    assert waited == [(process, 8765, 12.0)]
    assert launched[0][0][:3] == [str(selected_python_executable()), "-m", "streamlit"]
    assert launched[0][1]["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in launched[0][1]["env"]


def test_intentional_idle_shutdown_returns_success(monkeypatch) -> None:
    process = SimpleNamespace(returncode=1)
    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8765)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        launcher, "_wait_with_auto_stop", lambda *_args, **_kwargs: True
    )

    assert launcher.main(["--no-browser"]) == 0


def test_localized_netstat_output_is_decoded_safely(monkeypatch) -> None:
    observed = {}

    def fake_run(*_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher._has_browser_connection(8765) is False
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"


def test_workflow_dispatcher_does_not_register_frontends() -> None:
    assert "dart-spectrogram" not in _COMMANDS
    assert "roi-lightcurve" not in _COMMANDS
    assert "source-map-app" not in _COMMANDS
    assert build_dispatcher_parser().parse_args(["source-map"]).command == "source-map"
