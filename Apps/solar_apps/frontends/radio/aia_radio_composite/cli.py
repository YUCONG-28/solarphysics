"""Managed Miniforge launcher for the AIA radio composite Streamlit app."""

from __future__ import annotations

import argparse
import socket
import subprocess
import webbrowser
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.processes import (
    miniforge_subprocess_environment,
    python_module_command,
)

from . import FRONTEND_ID

__all__ = ["FRONTEND_ID", "build_parser", "build_streamlit_command", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the managed frontend launcher parser."""

    parser = argparse.ArgumentParser(
        prog="solar-apps frontend aia-radio-composite",
        description="Launch the AIA/radio/ROI/spectrum composite app.",
    )
    parser.add_argument("--aia-dir", default=None)
    parser.add_argument("--radio-dir", default=None)
    parser.add_argument("--spectrum-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to the app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8511,
        help="Preferred Streamlit port; a free port is selected if occupied.",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--browser", dest="browser", action="store_true")
    browser.add_argument("--no-browser", dest="browser", action="store_false")
    parser.set_defaults(browser=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved Streamlit command without starting it.",
    )
    return parser


def build_streamlit_command(
    args: argparse.Namespace,
    *,
    port: int,
) -> list[str]:
    """Return the managed Streamlit subprocess command."""

    app_script = Path(__file__).with_name("app.py")
    command = python_module_command(
        "streamlit",
        [
            "run",
            str(app_script),
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--",
        ],
    )
    _append_option(command, "--aia-dir", args.aia_dir)
    _append_option(command, "--radio-dir", args.radio_dir)
    _append_option(command, "--spectrum-path", args.spectrum_path)
    _append_option(command, "--output-dir", args.output_dir)
    _append_option(command, "--allowed-roots", args.allowed_roots)
    return command


def main(argv: list[str] | None = None) -> int:
    """Launch the app through the selected workspace Miniforge runtime."""

    args = build_parser().parse_args(argv)
    port = _pick_port(int(args.port))
    command = build_streamlit_command(args, port=port)
    if args.dry_run:
        print(" ".join(command))
        return 0
    process = subprocess.Popen(
        command,
        cwd=RuntimeLayout.discover().repo_root,
        env=miniforge_subprocess_environment(),
    )
    url = f"http://127.0.0.1:{port}"
    print(f"AIA Radio Composite app: {url}")
    if args.browser:
        webbrowser.open(url)
    try:
        process.wait()
    except KeyboardInterrupt:
        _terminate_process(process)
    return int(process.returncode or 0)


def _append_option(command: list[str], flag: str, value: object) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _pick_port(preferred_port: int) -> int:
    if preferred_port > 0 and _port_is_free(preferred_port):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", int(port))) != 0


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
