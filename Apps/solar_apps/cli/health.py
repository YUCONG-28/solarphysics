"""Directory-driven App health matrix for the solarphysics application tree.

The health matrix derives every check from the two public catalogues:

* solar_apps.frontends.catalog.FRONTENDS (all public frontends)
* solar_apps.frontends.app_v1.catalog.MODULES (all App 1.0 pages)

No ID list is maintained in this module or in its tests.  All runtime
directories and caches live under a system temporary directory (or an
injected local_root); the repository Local tree is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from solar_apps.frontends.app_v1.catalog import MODULES
from solar_apps.frontends.catalog import FRONTENDS, FrontendSpec

SCHEMA_VERSION = "1.0"
LEGACY_TOOLKITS = frozenset({"flask", "streamlit"})
SMOKE_PREFIX = "APP_V1_SMOKE "
PAGE_PREFIX = "APP_V1_HEALTH_PAGE "
HEALTH_PREFIX = "solar-apps-health-"
DEFAULT_SUBPROCESS_TIMEOUT_S = 60.0
PAGE_TIMEOUT_S = 90.0
LEGACY_PROBE_TIMEOUT_S = 25.0
LEGACY_IDLE_SECONDS = 3600

_PAGE_SCRIPT = r"""
import json
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from solar_apps.frontends.app_v1.window import AppV1MainWindow
from solar_apps.platform.layout import RuntimeLayout

local = Path(sys.argv[1])
module = sys.argv[2]
repo_root = Path(sys.argv[3])
allowed = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None

application = QApplication(["solar-apps-health-page"])
layout = RuntimeLayout.discover(
    repo_root, environ={"SOLAR_APPS_LOCAL_ROOT": str(local)}
).ensure()
composer_roots = (Path(allowed),) if allowed else None
window = AppV1MainWindow(
    layout,
    initial_module=module,
    composer_allowed_roots=composer_roots,
)
window.select_module(module)
application.processEvents()
application.processEvents()
window.close()
application.processEvents()
result = {
    "selected_module": module,
    "registered_modules": list(window.registered_module_ids),
    "registered_count": len(window.registered_module_ids),
    "foreign_qt_loaded": any(
        name == "PyQt5"
        or name.startswith("PyQt5.")
        or name == "PySide6"
        or name.startswith("PySide6.")
        for name in sys.modules
    ),
    "running_qprocess_count": window.task_controller.running_process_count,
    "process_running": window.task_controller.process_running,
}
print("APP_V1_HEALTH_PAGE " + json.dumps(result, sort_keys=True), flush=True)
"""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One machine-readable health check outcome."""

    id: str
    category: str
    status: str
    detail: str
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status,
            "detail": self.detail,
            "duration_seconds": self.duration_seconds,
        }


def now_utc() -> str:
    """Return the current UTC time as an ISO-8601 Z string."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def aggregate_status(checks: Sequence[CheckResult]) -> str:
    """Fail the matrix when any required check fails; not_run never fails."""

    return "fail" if any(check.status == "fail" for check in checks) else "pass"


def build_report(
    checks: Sequence[CheckResult],
    started_at_utc: str,
    finished_at_utc: str,
) -> dict[str, object]:
    """Build the versioned health report envelope."""

    return {
        "schema_version": SCHEMA_VERSION,
        "overall_status": aggregate_status(checks),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "checks": [check.to_dict() for check in checks],
    }


def atomic_write_json(
    path: str | os.PathLike[str], payload: Mapping[str, object]
) -> None:
    """Write JSON atomically through a same-directory temp file and rename."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def check_ids() -> tuple[str, ...]:
    """Return the complete matrix check IDs derived from the catalogues."""

    ids: list[str] = ["catalog-frontends", "catalog-app-v1-modules"]
    for frontend in FRONTENDS:
        ids.append(f"entry-{frontend.id}")
    for frontend in FRONTENDS:
        ids.append(f"help-{frontend.id}")
    for module in MODULES:
        ids.append(f"page-{module.module_id}")
    ids.extend(("smoke-basic", "smoke-cancel", "smoke-project"))
    for frontend in FRONTENDS:
        if frontend.toolkit in LEGACY_TOOLKITS:
            ids.append(f"legacy-{frontend.id}")
    return tuple(ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Apps/run.sh tools health",
        description=(
            "Run the offline App health matrix and write an atomic JSON report. "
            "All runtime directories and caches use system temporary space."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="JSON-PATH",
        help="Destination for the atomic JSON health report.",
    )
    parser.add_argument(
        "--include-legacy-servers",
        action="store_true",
        help=(
            "Controlled startup and HTTP probe for flask/streamlit entries on "
            "random 127.0.0.1 ports. Servers that cannot start without data "
            "stay not_run with a machine-readable reason."
        ),
    )
    return parser


def _subprocess_environment(
    local_root: Path,
    cache_root: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "SOLAR_APPS_LOCAL_ROOT": str(local_root),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "SUNPY_CONFIGDIR": str(cache_root / "sunpy"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _ensure_roots(local_root: Path, cache_root: Path) -> None:
    (local_root / "allowed").mkdir(parents=True, exist_ok=True)
    for directory in (
        cache_root / "matplotlib",
        cache_root / "sunpy",
        cache_root / "xdg",
        local_root / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _tail(path: Path, lines: int = 8, limit: int = 400) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    tail = "\n".join(text.splitlines()[-lines:])
    return tail[-limit:]


def _measure(
    check_id: str, category: str, fn: Callable[[], tuple[str, str]]
) -> CheckResult:
    started = time.monotonic()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - every check must become a JSON entry
        status, detail = "fail", f"unexpected error: {type(exc).__name__}: {exc}"
    if status not in {"pass", "fail", "not_run"}:
        invalid = status
        status = "fail"
        detail = f"invalid status {invalid!r} from check {check_id!r}"
    return CheckResult(
        check_id,
        category,
        status,
        detail,
        round(time.monotonic() - started, 3),
    )


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=dict(env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def _tail_text(text: str, limit: int = 400) -> str:
    return "\n".join(text.splitlines()[-8:])[-limit:]


def _catalog_checks() -> list[CheckResult]:
    def frontends() -> tuple[str, str]:
        detail = f"frontends={len(FRONTENDS)} ids={','.join(f.id for f in FRONTENDS)}"
        return ("pass" if FRONTENDS else "fail"), detail

    def modules() -> tuple[str, str]:
        detail = f"modules={len(MODULES)} ids={','.join(m.module_id for m in MODULES)}"
        return ("pass" if MODULES else "fail"), detail

    return [
        _measure("catalog-frontends", "catalog", frontends),
        _measure("catalog-app-v1-modules", "catalog", modules),
    ]


def _run_entry_check(
    spec: FrontendSpec,
    *,
    apps_root: Path,
    python: str,
    env: Mapping[str, str],
) -> CheckResult:
    def run() -> tuple[str, str]:
        script = (
            "import importlib, sys; "
            "m = importlib.import_module(sys.argv[1]); "
            "assert callable(m.main)"
        )
        completed = _run(
            [python, "-c", script, spec.entry_module],
            cwd=apps_root,
            env=env,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT_S,
        )
        if completed.returncode != 0:
            return (
                "fail",
                f"module={spec.entry_module} import/callable check failed "
                f"rc={completed.returncode}: {_tail_text(completed.stderr)}",
            )
        return "pass", f"module={spec.entry_module} main=callable"

    return _measure(f"entry-{spec.id}", "frontends.entry", run)


def _run_help_check(
    spec: FrontendSpec,
    *,
    apps_root: Path,
    python: str,
    env: Mapping[str, str],
) -> tuple[CheckResult, str]:
    captured: dict[str, str] = {}

    def run() -> tuple[str, str]:
        completed = _run(
            [python, "-m", spec.entry_module, "--help"],
            cwd=apps_root,
            env=env,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT_S,
        )
        captured["text"] = completed.stdout if completed.returncode == 0 else ""
        if completed.returncode != 0:
            return (
                "fail",
                f"module={spec.entry_module} --help failed "
                f"rc={completed.returncode}: {_tail_text(completed.stderr)}",
            )
        return "pass", f"module={spec.entry_module} --help rc=0"

    check = _measure(f"help-{spec.id}", "frontends.help", run)
    return check, captured.get("text", "")


def _run_page_check(
    module_id: str,
    *,
    apps_root: Path,
    local_root: Path,
    allowed_root: Path,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[CheckResult, dict[str, object] | None]:
    python = python or sys.executable
    cache_root = local_root / "cache"
    _ensure_roots(local_root, cache_root)
    environment = _subprocess_environment(local_root, cache_root, base=env)
    allowed = str(allowed_root) if module_id == "image-composer" else ""

    def run() -> tuple[str, str]:
        completed = _run(
            [
                python,
                "-c",
                _PAGE_SCRIPT,
                str(local_root),
                module_id,
                str(apps_root.parent),
                allowed,
            ],
            cwd=apps_root,
            env=environment,
            timeout=PAGE_TIMEOUT_S,
        )
        if completed.returncode != 0:
            return (
                "fail",
                f"module={module_id} offscreen page process exited "
                f"rc={completed.returncode}: {_tail_text(completed.stderr)}",
            )
        line = next(
            (
                item
                for item in completed.stdout.splitlines()
                if item.startswith(PAGE_PREFIX)
            ),
            None,
        )
        if line is None:
            return "fail", f"module={module_id} produced no {PAGE_PREFIX} line"
        payload = json.loads(line.removeprefix(PAGE_PREFIX))
        if payload.get("selected_module") != module_id:
            return "fail", f"module={module_id} selection mismatch: {payload}"
        if payload.get("registered_count") != len(MODULES):
            return "fail", f"module={module_id} registered count mismatch: {payload}"
        if payload.get("foreign_qt_loaded") is not False:
            return "fail", f"module={module_id} foreign Qt loaded: {payload}"
        if payload.get("running_qprocess_count") != 0:
            return "fail", f"module={module_id} residual QProcess: {payload}"
        if payload.get("process_running") is not False:
            return "fail", f"module={module_id} task process still running: {payload}"
        return (
            "pass",
            f"module={module_id} pages={payload.get('registered_count')} "
            "foreign_qt=false qprocess=0",
        )

    check = _measure(f"page-{module_id}", "app_v1.pages", run)
    payload: dict[str, object] | None = None
    if check.status == "pass":
        payload = {
            "selected_module": module_id,
            "registered_count": len(MODULES),
            "foreign_qt_loaded": False,
            "running_qprocess_count": 0,
            "process_running": False,
        }
    return check, payload


def _validate_smoke(mode: str, payload: Mapping[str, object]) -> tuple[bool, str]:
    if payload.get("foreign_qt_loaded") is not False:
        return (
            False,
            f"foreign Qt loaded: {payload.get('forbidden_frontend_modules')!r}",
        )
    if payload.get("process_running") is not False:
        return False, "task process still running"
    if payload.get("running_qprocess_count") != 0:
        return False, f"residual QProcess count={payload.get('running_qprocess_count')}"
    if mode == "basic":
        themes = payload.get("themes")
        if not isinstance(themes, Mapping) or set(themes) != {
            "auto",
            "light",
            "dark",
            "dark_dimmed",
        }:
            return False, f"theme coverage missing: {themes!r}"
        registered = set(payload.get("registered_modules") or ())
        expected = {module.module_id for module in MODULES}
        if registered != expected:
            return (
                False,
                f"registered modules mismatch: {sorted(registered ^ expected)}",
            )
        return True, "themes=auto,light,dark,dark_dimmed pages=all qprocess=0"
    if mode == "cancel":
        if payload.get("task_status") != "cancelled":
            return False, f"task_status={payload.get('task_status')!r}"
        return True, "task_status=cancelled qprocess=0"
    if mode == "project":
        if payload.get("restored_project_id") != "smoke-project":
            return False, f"restored_project_id={payload.get('restored_project_id')!r}"
        if payload.get("restored_parameter") != 2.5:
            return False, f"restored_parameter={payload.get('restored_parameter')!r}"
        return True, "project_id=smoke-project parameter=2.5 qprocess=0"
    raise ValueError(f"Unknown smoke mode: {mode}")


def _run_smoke_check(
    mode: str,
    *,
    apps_root: Path,
    local_root: Path,
    python: str,
    env: Mapping[str, str],
) -> CheckResult:
    cache_root = local_root / "cache"
    _ensure_roots(local_root, cache_root)
    environment = _subprocess_environment(local_root, cache_root, base=env)

    def run() -> tuple[str, str]:
        completed = _run(
            [
                python,
                "-m",
                "solar_apps.frontends.app_v1.cli",
                "--smoke-test",
                mode,
                "--no-show",
            ],
            cwd=apps_root,
            env=environment,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT_S,
        )
        if completed.returncode != 0:
            return (
                "fail",
                f"smoke {mode} exited rc={completed.returncode}: "
                f"{_tail_text(completed.stderr)}",
            )
        line = next(
            (
                item
                for item in completed.stdout.splitlines()
                if item.startswith(SMOKE_PREFIX)
            ),
            None,
        )
        if line is None:
            return "fail", f"smoke {mode} produced no {SMOKE_PREFIX} line"
        payload = json.loads(line.removeprefix(SMOKE_PREFIX))
        valid, detail = _validate_smoke(mode, payload)
        return ("pass" if valid else "fail"), detail

    return _measure(f"smoke-{mode}", "app_v1.smoke", run)


def legacy_not_run_check(spec: FrontendSpec) -> CheckResult:
    return CheckResult(
        f"legacy-{spec.id}",
        "legacy.servers",
        "not_run",
        (
            f"[reason=disabled-by-default] toolkit={spec.toolkit} legacy server "
            "checks are offline by default; rerun with --include-legacy-servers"
        ),
        0.0,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _probe_http(
    process: subprocess.Popen[str],
    port: int,
    *,
    timeout: float,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout
    last_error = "no HTTP response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return "unsafe-startup", f"process exited rc={process.returncode}"
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=1.0
            ) as response:
                code = int(response.status)
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.3)
            continue
        if 200 <= code < 400:
            return "ok", str(code)
        return "http-error", f"HTTP {code}"
    return "startup-timeout", last_error


def _port_has_listener(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", int(port))) == 0


def _wait_for_port_closed(port: int, *, timeout: float = 5.0) -> bool:
    """Read-only bounded wait until the probe port stops listening."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_has_listener(port):
            return True
        time.sleep(0.1)
    return not _port_has_listener(port)


def _start_legacy_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: object,
) -> tuple[subprocess.Popen[Any], int | None]:
    """Start a server in a dedicated process tree; return (Popen, POSIX PGID)."""

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        return process, None
    process = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        env=dict(env),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, os.getpgid(process.pid)


def _process_group_alive(pgid: int | None) -> bool:
    """Return whether a recorded POSIX process group still exists."""

    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError, OSError:
        return True
    return True


def _wait_for_group_exit_and_kill(pgid: int, *, grace_seconds: float = 5.0) -> None:
    """Wait a bounded time for the group to exit, then escalate to SIGKILL."""

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:
        pass


def _terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    pgid: int | None,
    root_pid: int | None = None,
    platform: str | None = None,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate only the process tree started for this health check."""

    if (platform or os.name) == "nt":
        _terminate_windows_tree(process, root_pid=root_pid)
    else:
        _terminate_posix_group(process, pgid, grace_seconds=grace_seconds)


def _terminate_posix_group(
    process: subprocess.Popen[Any],
    pgid: int | None,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """TERM, bounded group-exit wait, KILL, then reap the launcher."""

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError, PermissionError:
            pass
    else:
        try:
            process.terminate()
        except OSError:
            pass
    if pgid is not None:
        _wait_for_group_exit_and_kill(pgid, grace_seconds=grace_seconds)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _terminate_windows_tree(
    process: subprocess.Popen[Any], *, root_pid: int | None = None
) -> None:
    """Terminate the whole Windows process tree via taskkill."""

    pid = root_pid if root_pid is not None else getattr(process, "pid", None)
    if pid is not None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except OSError, subprocess.SubprocessError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass


def _run_legacy_server_check(
    spec: FrontendSpec,
    *,
    include_legacy_servers: bool,
    help_text: str,
    apps_root: Path,
    local_root: Path,
    python: str,
    env: Mapping[str, str],
) -> CheckResult:
    def run() -> tuple[str, str]:
        if not include_legacy_servers:
            return legacy_not_run_check(spec).status, legacy_not_run_check(spec).detail
        port = _free_port()
        argv = [python, "-m", spec.entry_module]
        if spec.toolkit == "flask":
            argv.extend(
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--allowed-roots",
                    str(local_root / "allowed"),
                ]
            )
        else:
            argv.extend(["--port", str(port)])
            if "--no-browser" in help_text:
                argv.append("--no-browser")
            if "--auto-stop-idle-sec" in help_text:
                argv.extend(["--auto-stop-idle-sec", str(LEGACY_IDLE_SECONDS)])
        log_path = local_root / "logs" / f"legacy-{spec.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                process, pgid = _start_legacy_process(
                    argv,
                    cwd=apps_root,
                    env=env,
                    stdout=log_handle,
                )
        except OSError as exc:
            return (
                "not_run",
                f"[reason=startup-error] toolkit={spec.toolkit} "
                f"{type(exc).__name__}: {exc}",
            )
        try:
            outcome, info = _probe_http(process, port, timeout=LEGACY_PROBE_TIMEOUT_S)
        finally:
            _terminate_process_tree(process, pgid=pgid, root_pid=process.pid)
        if outcome == "ok":
            if not _wait_for_port_closed(port, timeout=5.0):
                return (
                    "not_run",
                    f"[reason=cleanup-failed] toolkit={spec.toolkit} port "
                    f"{port} still listening after process-tree termination",
                )
            return (
                "pass",
                f"toolkit={spec.toolkit} http=127.0.0.1:{port} status={info}",
            )
        tail = _tail(log_path)
        return (
            "not_run",
            f"[reason={outcome}] toolkit={spec.toolkit} {info}; log tail: {tail}",
        )

    return _measure(f"legacy-{spec.id}", "legacy.servers", run)


def run_health(
    include_legacy_servers: bool = False,
    *,
    apps_root: str | os.PathLike[str] | None = None,
    local_root: str | os.PathLike[str] | None = None,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Execute the complete offline health matrix and return the report."""

    resolved_apps_root = (
        Path(apps_root)
        if apps_root is not None
        else Path(__file__).resolve().parents[2]
    )
    interpreter = python or sys.executable
    provided_local = Path(local_root).expanduser() if local_root is not None else None
    managed_root: tempfile.TemporaryDirectory[str] | None = None
    if provided_local is not None:
        resolved_local = provided_local
    else:
        managed_root = tempfile.TemporaryDirectory(prefix=HEALTH_PREFIX)
        resolved_local = Path(managed_root.name) / "Local"
    try:
        return _run_health_matrix(
            include_legacy_servers=include_legacy_servers,
            apps_root=resolved_apps_root,
            local_root=resolved_local,
            python=interpreter,
            env=env,
        )
    finally:
        if managed_root is not None:
            managed_root.cleanup()


def _run_health_matrix(
    *,
    include_legacy_servers: bool,
    apps_root: Path,
    local_root: Path,
    python: str,
    env: Mapping[str, str] | None,
) -> dict[str, object]:
    """Run every catalog-driven check and build the report envelope."""

    cache_root = local_root / "cache"
    _ensure_roots(local_root, cache_root)
    subprocess_env = _subprocess_environment(local_root, cache_root, base=env)

    started = now_utc()
    checks: list[CheckResult] = []
    checks.extend(_catalog_checks())

    for spec in FRONTENDS:
        checks.append(
            _run_entry_check(
                spec,
                apps_root=apps_root,
                python=python,
                env=subprocess_env,
            )
        )
    help_texts: dict[str, str] = {}
    for spec in FRONTENDS:
        check, help_text = _run_help_check(
            spec,
            apps_root=apps_root,
            python=python,
            env=subprocess_env,
        )
        checks.append(check)
        help_texts[spec.id] = help_text

    for module in MODULES:
        check, _payload = _run_page_check(
            module.module_id,
            apps_root=apps_root,
            local_root=local_root,
            allowed_root=local_root / "allowed",
            python=python,
            env=env,
        )
        checks.append(check)

    for mode in ("basic", "cancel", "project"):
        checks.append(
            _run_smoke_check(
                mode,
                apps_root=apps_root,
                local_root=local_root,
                python=python,
                env=env,
            )
        )

    for spec in FRONTENDS:
        if spec.toolkit in LEGACY_TOOLKITS:
            checks.append(
                _run_legacy_server_check(
                    spec,
                    include_legacy_servers=include_legacy_servers,
                    help_text=help_texts.get(spec.id, ""),
                    apps_root=apps_root,
                    local_root=local_root,
                    python=python,
                    env=subprocess_env,
                )
            )

    return build_report(checks, started, now_utc())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = now_utc()
    try:
        report = run_health(include_legacy_servers=args.include_legacy_servers)
    except Exception as exc:  # noqa: BLE001 - the report must always be written
        report = build_report(
            [
                CheckResult(
                    "health-runner",
                    "health.runner",
                    "fail",
                    f"[reason=runner-error] {type(exc).__name__}: {exc}",
                    0.0,
                )
            ],
            started,
            now_utc(),
        )
    atomic_write_json(args.output, report)
    return 1 if report["overall_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckResult",
    "LEGACY_TOOLKITS",
    "SCHEMA_VERSION",
    "aggregate_status",
    "atomic_write_json",
    "build_parser",
    "build_report",
    "check_ids",
    "legacy_not_run_check",
    "main",
    "now_utc",
    "run_health",
]
