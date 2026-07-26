"""Managed launcher tests for the AIA radio composite frontend."""

from __future__ import annotations

from pathlib import Path

from solar_apps.cli.router import FRONTEND_TARGETS
from solar_apps.frontends.radio.aia_radio_composite import cli
from solar_apps.platform.processes import selected_python_executable


def test_router_registers_public_frontend_id() -> None:
    assert FRONTEND_TARGETS["aia-radio-composite"] == (
        "solar_apps.frontends.radio.aia_radio_composite.cli"
    )


def test_streamlit_command_forwards_data_and_output_paths() -> None:
    args = cli.build_parser().parse_args(
        [
            "--aia-dir",
            "D:/aia",
            "--radio-dir",
            "D:/radio",
            "--spectrum-path",
            "D:/dart",
            "--output-dir",
            "D:/output",
            "--allowed-roots",
            "D:/aia;D:/radio;D:/dart;D:/output",
            "--no-browser",
        ]
    )

    command = cli.build_streamlit_command(args, port=8511)

    assert command[:4] == [
        str(selected_python_executable()),
        "-m",
        "streamlit",
        "run",
    ]
    assert Path(command[4]).parts[-2:] == ("aia_radio_composite", "app.py")
    assert command[command.index("--server.port") + 1] == "8511"
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--aia-dir") + 1] == "D:/aia"
    assert command[command.index("--radio-dir") + 1] == "D:/radio"
    assert command[command.index("--spectrum-path") + 1] == "D:/dart"
    assert command[command.index("--output-dir") + 1] == "D:/output"
    assert command[command.index("--allowed-roots") + 1] == (
        "D:/aia;D:/radio;D:/dart;D:/output"
    )


def test_dry_run_prints_command_without_starting_process(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_pick_port", lambda port: 8511)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run started Streamlit")
        ),
    )

    assert cli.main(["--dry-run", "--no-browser"]) == 0
    output = capsys.readouterr().out
    assert "streamlit run" in output
    assert "aia_radio_composite/app.py" in output.replace("\\", "/")
