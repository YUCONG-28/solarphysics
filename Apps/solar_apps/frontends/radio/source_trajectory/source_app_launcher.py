"""Managed launcher for the radio-source Streamlit app.

The Streamlit app itself intentionally stays focused on rendering. This wrapper
owns process lifetime, so local runs can stop after the browser tab disconnects.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import time
import webbrowser
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.processes import (
    miniforge_subprocess_environment,
    python_module_command,
)

from solar_toolkit.radio.trajectory import FRAME_MODE_LABELS
from solar_apps.workflows.visualization.radio_source_trajectory import (
    FACET_BY_OPTIONS,
    PLOT_LAYOUTS,
)

__all__ = ["build_parser", "build_streamlit_command", "main"]

THEME_MODES = ("light", "dark", "auto")
SCREEN_FIT_MODES = ("auto", "landscape", "portrait")


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the managed Streamlit launcher."""

    parser = argparse.ArgumentParser(
        prog="solar-apps frontend source-trajectory",
        description="Launch the radio-source Streamlit app as a managed process.",
    )
    parser.add_argument("--centers", default=None, help="Center CSV/XLSX path.")
    parser.add_argument("--time-start", default=None, help="Inclusive time start.")
    parser.add_argument("--time-end", default=None, help="Inclusive time end.")
    parser.add_argument("--aia-dir", default=None, help="AIA FITS folder.")
    parser.add_argument("--aia-pattern", default=None, help="AIA FITS glob pattern.")
    parser.add_argument(
        "--frame-mode",
        choices=list(FRAME_MODE_LABELS),
        default=None,
        help="Default trajectory display mode.",
    )
    parser.add_argument("--tail-n", type=int, default=None, help="Tail length.")
    parser.add_argument(
        "--plot-layout",
        choices=PLOT_LAYOUTS,
        default=None,
        help="Default trajectory layout.",
    )
    parser.add_argument(
        "--facet-by",
        choices=FACET_BY_OPTIONS,
        default=None,
        help="Default faceting dimension.",
    )
    parser.add_argument(
        "--settings-file", default=None, help="Local JSON settings file."
    )
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to the app.",
    )
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="Ignore saved settings for this run.",
    )
    parser.add_argument(
        "--theme-mode",
        choices=THEME_MODES,
        default=None,
        help="Default app theme mode.",
    )
    parser.add_argument(
        "--screen-fit",
        choices=SCREEN_FIT_MODES,
        default=None,
        help="Default screen layout mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Preferred Streamlit port; another free port is used if occupied.",
    )
    auto_stop = parser.add_mutually_exclusive_group()
    auto_stop.add_argument(
        "--auto-stop",
        dest="auto_stop",
        action="store_true",
        help="Stop Streamlit after browser connections go idle.",
    )
    auto_stop.add_argument(
        "--no-auto-stop",
        dest="auto_stop",
        action="store_false",
        help="Keep Streamlit running until interrupted.",
    )
    parser.set_defaults(auto_stop=True)
    parser.add_argument(
        "--auto-stop-idle-sec",
        type=float,
        default=60.0,
        help="Idle seconds without browser connections before stopping.",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument(
        "--browser",
        dest="browser",
        action="store_true",
        help="Open the app URL in the default browser.",
    )
    browser.add_argument(
        "--no-browser",
        dest="browser",
        action="store_false",
        help="Print the app URL without opening a browser.",
    )
    parser.set_defaults(browser=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Streamlit command without launching it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch Streamlit and optionally stop it after the UI goes idle."""

    args = build_parser().parse_args(argv)
    port = _pick_port(int(args.port))
    url = f"http://127.0.0.1:{port}"
    command = build_streamlit_command(args, port=port)
    if args.dry_run:
        print(" ".join(command))
        return 0

    process = subprocess.Popen(
        command,
        cwd=RuntimeLayout.discover().repo_root,
        env=miniforge_subprocess_environment(),
    )
    print(f"Radio source trajectory app: {url}")
    if args.browser:
        webbrowser.open(url)

    stopped_for_idle = False
    try:
        if args.auto_stop:
            stopped_for_idle = _wait_with_auto_stop(
                process, port=port, idle_seconds=args.auto_stop_idle_sec
            )
        else:
            process.wait()
    except KeyboardInterrupt:
        _terminate_process(process)
        return 0
    if stopped_for_idle:
        return 0
    return int(process.returncode or 0)


def build_streamlit_command(args: argparse.Namespace, *, port: int) -> list[str]:
    """Return the Streamlit subprocess command for tests and dry runs."""

    app_script = Path(__file__).with_name("source_app.py")
    command = python_module_command(
        "streamlit",
        [
            "run",
            str(app_script),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--",
        ],
    )
    _append_option(command, "--centers", args.centers)
    _append_option(command, "--time-start", args.time_start)
    _append_option(command, "--time-end", args.time_end)
    _append_option(command, "--aia-dir", args.aia_dir)
    _append_option(command, "--aia-pattern", args.aia_pattern)
    _append_option(command, "--frame-mode", args.frame_mode)
    _append_option(command, "--tail-n", args.tail_n)
    _append_option(command, "--plot-layout", args.plot_layout)
    _append_option(command, "--facet-by", args.facet_by)
    _append_option(command, "--settings-file", args.settings_file)
    _append_option(command, "--allowed-roots", args.allowed_roots)
    if args.reset_settings:
        command.append("--reset-settings")
    _append_option(command, "--theme-mode", args.theme_mode)
    _append_option(command, "--screen-fit", args.screen_fit)
    return command


def _append_option(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
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


def _wait_with_auto_stop(
    process: subprocess.Popen,
    *,
    port: int,
    idle_seconds: float,
) -> bool:
    last_seen = time.monotonic()
    grace = max(5.0, float(idle_seconds))
    while process.poll() is None:
        if _has_browser_connection(port):
            last_seen = time.monotonic()
        elapsed = time.monotonic() - last_seen
        if elapsed >= max(1.0, float(idle_seconds)) and elapsed >= grace:
            _terminate_process(process)
            return True
        time.sleep(1.0)
    return False


def _has_browser_connection(port: int) -> bool:
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    marker_v4 = f":{int(port)} "
    marker_v6 = f":{int(port)}]"
    for line in (result.stdout or "").splitlines():
        if "ESTABLISHED" not in line.upper():
            continue
        if marker_v4 in line or marker_v6 in line:
            return True
    return False


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
