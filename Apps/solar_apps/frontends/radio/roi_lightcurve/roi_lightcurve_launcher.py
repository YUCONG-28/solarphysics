"""Managed launcher for the radio ROI light-curve Streamlit app."""

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

from solar_toolkit.radio.centers import POL_LCP, POL_RCP, POL_SUM

__all__ = ["build_parser", "build_streamlit_command", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the managed Streamlit launcher."""

    parser = argparse.ArgumentParser(
        prog="solar-apps frontend roi-lightcurve",
        description="Launch the radio ROI light-curve Streamlit app as a managed process.",
    )
    parser.add_argument("--radio-dir", default=None, help="Default radio FITS folder.")
    parser.add_argument("--pattern", default=None, help="Default FITS glob pattern.")
    recursive = parser.add_mutually_exclusive_group()
    recursive.add_argument("--recursive", dest="recursive", action="store_true")
    recursive.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=None)
    parser.add_argument(
        "--time-start", default=None, help="Default inclusive time start."
    )
    parser.add_argument("--time-end", default=None, help="Default inclusive time end.")
    parser.add_argument("--output-dir", default=None, help="Default output folder.")
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to the app.",
    )
    parser.add_argument(
        "--pair-time-tolerance-sec",
        type=float,
        default=None,
        help="Default LCP/RCP pairing tolerance in seconds.",
    )
    parser.add_argument(
        "--polarization",
        choices=[POL_SUM, POL_LCP, POL_RCP, "all"],
        default=None,
        help="Default polarization mode.",
    )
    parser.add_argument(
        "--metric",
        choices=["raw_sum", "raw_mean", "raw_peak"],
        default=None,
        help="Default plotted metric.",
    )
    parser.add_argument(
        "--settings-file", default=None, help="Local JSON settings file."
    )
    parser.add_argument("--reset-settings", action="store_true")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Preferred Streamlit port; another free port is used if occupied.",
    )
    auto_stop = parser.add_mutually_exclusive_group()
    auto_stop.add_argument("--auto-stop", dest="auto_stop", action="store_true")
    auto_stop.add_argument("--no-auto-stop", dest="auto_stop", action="store_false")
    parser.set_defaults(auto_stop=True)
    parser.add_argument(
        "--auto-stop-idle-sec",
        type=float,
        default=60.0,
        help="Idle seconds without browser connections before stopping.",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--browser", dest="browser", action="store_true")
    browser.add_argument("--no-browser", dest="browser", action="store_false")
    parser.set_defaults(browser=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the Streamlit command."
    )
    return parser


def build_streamlit_command(args: argparse.Namespace, *, port: int) -> list[str]:
    """Return the Streamlit subprocess command for tests and dry runs."""

    app_script = Path(__file__).with_name("roi_lightcurve_app.py")
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
    _append_option(command, "--radio-dir", args.radio_dir)
    _append_option(command, "--pattern", args.pattern)
    if args.recursive is True:
        command.append("--recursive")
    elif args.recursive is False:
        command.append("--no-recursive")
    _append_option(command, "--time-start", args.time_start)
    _append_option(command, "--time-end", args.time_end)
    _append_option(command, "--output-dir", args.output_dir)
    _append_option(command, "--allowed-roots", args.allowed_roots)
    _append_option(command, "--pair-time-tolerance-sec", args.pair_time_tolerance_sec)
    _append_option(command, "--polarization", args.polarization)
    _append_option(command, "--metric", args.metric)
    _append_option(command, "--settings-file", args.settings_file)
    if args.reset_settings:
        command.append("--reset-settings")
    return command


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
    print(f"Radio ROI light-curve app: {url}")
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
