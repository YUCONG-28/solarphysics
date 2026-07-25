"""Windows/macOS native path selection with fail-closed root validation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..processes import (
    miniforge_subprocess_environment,
    selected_python_executable,
)
from .memory import PathMemoryContext, RecentPathMemory
from .semantics import path_is_within, path_key

__all__ = [
    "DialogRequest",
    "DialogSelection",
    "NativeDialogBusyError",
    "NativeDialogError",
    "NativeDialogForbiddenError",
    "NativeDialogRequestError",
    "NativeDialogUnavailableError",
    "NativeDialogUnsupportedError",
    "NativePathDialogService",
    "is_path_within_roots",
    "validate_allowed_path",
]

_DIALOG_MODES = frozenset({"open_file", "open_files", "select_directory", "save_file"})
_EXTENSION_RE = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
_DIALOG_APPEAR_TIMEOUT_SECONDS = 5.0


class NativeDialogError(RuntimeError):
    """Base error carrying the HTTP status used by Flask adapters."""

    status_code = 503


class NativeDialogRequestError(NativeDialogError):
    status_code = 400


class NativeDialogForbiddenError(NativeDialogError):
    status_code = 403


class NativeDialogBusyError(NativeDialogError):
    status_code = 409


class NativeDialogUnsupportedError(NativeDialogError):
    status_code = 501


class NativeDialogUnavailableError(NativeDialogError):
    status_code = 503


@dataclass(frozen=True)
class DialogRequest:
    """Validated request sent to the isolated Qt dialog worker."""

    mode: str
    title: str = "Select local path"
    initial_path: str = ""
    extensions: tuple[str, ...] = ()
    default_suffix: str = ""
    memory_context: PathMemoryContext | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DialogRequest:
        if not isinstance(payload, Mapping):
            raise NativeDialogRequestError("Dialog request must be a JSON object.")
        mode = str(payload.get("mode", "")).strip()
        if mode not in _DIALOG_MODES:
            raise NativeDialogRequestError(f"Unsupported dialog mode: {mode!r}")
        title = str(payload.get("title") or "Select local path").strip()
        if not title or len(title) > 160 or any(ord(char) < 32 for char in title):
            raise NativeDialogRequestError("Dialog title is invalid.")
        initial_path = str(payload.get("initial_path") or "").strip()
        raw_extensions = payload.get("extensions") or []
        if not isinstance(raw_extensions, Sequence) or isinstance(
            raw_extensions, (str, bytes, bytearray)
        ):
            raise NativeDialogRequestError("extensions must be a JSON array.")
        if len(raw_extensions) > 32:
            raise NativeDialogRequestError("At most 32 file extensions are allowed.")
        extensions = tuple(_normalize_extension(value) for value in raw_extensions)
        default_suffix = str(payload.get("default_suffix") or "").strip()
        if default_suffix:
            default_suffix = _normalize_extension(default_suffix)
        if mode != "save_file" and default_suffix:
            raise NativeDialogRequestError(
                "default_suffix is only valid for save_file dialogs."
            )
        try:
            memory_context = PathMemoryContext.from_payload(
                payload.get("memory_context")
            )
        except ValueError as exc:
            raise NativeDialogRequestError(str(exc)) from exc
        return cls(
            mode=mode,
            title=title,
            initial_path=initial_path,
            extensions=tuple(dict.fromkeys(extensions)),
            default_suffix=default_suffix,
            memory_context=memory_context,
        )

    def to_worker_payload(self, *, initial_path: str) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "title": self.title,
            "initial_path": initial_path,
            "extensions": list(self.extensions),
            "default_suffix": self.default_suffix,
        }


@dataclass(frozen=True)
class DialogSelection:
    status: str
    paths: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": self.status,
            "paths": [str(path) for path in self.paths],
        }


def _normalize_extension(value: Any) -> str:
    extension = str(value).strip()
    if extension in {"*", ".*"}:
        return "*"
    if extension.startswith("*."):
        extension = extension[1:]
    if not extension.startswith("."):
        extension = f".{extension}"
    if not _EXTENSION_RE.fullmatch(extension):
        raise NativeDialogRequestError(f"Invalid file extension: {value!r}")
    return extension.casefold()


def _normalize_roots(values: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        root = Path(value).expanduser().resolve(strict=False)
        key = path_key(root)
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return tuple(roots)


def is_path_within_roots(
    path: str | os.PathLike[str], roots: Iterable[str | os.PathLike[str]]
) -> bool:
    """Return whether a resolved path stays inside at least one allowed root."""

    candidate = Path(path).expanduser().resolve(strict=False)
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve(strict=False)
        if path_is_within(candidate, root):
            return True
    return False


def validate_allowed_path(
    value: str | os.PathLike[str],
    *,
    allowed_roots: Iterable[str | os.PathLike[str]],
    kind: str,
    base_directory: str | os.PathLike[str] | None = None,
    default_suffix: str = "",
) -> Path:
    """Resolve one typed or selected path and enforce its kind/root policy."""

    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        raise NativeDialogRequestError("Path is required.")
    roots = _normalize_roots(allowed_roots)
    if not roots:
        raise NativeDialogForbiddenError("No allowed filesystem roots are configured.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = Path(base_directory or Path.cwd()).expanduser().resolve(strict=False)
        candidate = base / candidate
    if kind == "file":
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise NativeDialogRequestError(f"File does not exist: {candidate}") from exc
        if not resolved.is_file():
            raise NativeDialogRequestError(f"Not a file: {resolved}")
    elif kind == "directory":
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise NativeDialogRequestError(
                f"Directory does not exist: {candidate}"
            ) from exc
        if not resolved.is_dir():
            raise NativeDialogRequestError(f"Not a directory: {resolved}")
    elif kind in {"output_directory", "save_file"}:
        if kind == "save_file" and default_suffix and not candidate.suffix:
            candidate = candidate.with_suffix(_normalize_extension(default_suffix))
        resolved = candidate.resolve(strict=False)
        anchor = resolved if kind == "output_directory" else resolved.parent
        existing = anchor
        if kind == "output_directory":
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
        elif not existing.exists():
            raise NativeDialogRequestError(
                f"Save-file parent directory does not exist: {existing}"
            )
        try:
            existing = existing.resolve(strict=True)
        except FileNotFoundError as exc:
            raise NativeDialogRequestError(
                f"No existing parent directory for: {candidate}"
            ) from exc
        if not existing.is_dir():
            raise NativeDialogRequestError(f"Not a directory: {existing}")
    else:
        raise NativeDialogRequestError(f"Unsupported path kind: {kind!r}")
    if not is_path_within_roots(resolved, roots):
        raise NativeDialogForbiddenError(f"Path is outside allowed roots: {resolved}")
    return resolved


def _wait_for_worker_ready(process: Any, ready_path: Path) -> bool:
    deadline = time.monotonic() + _DIALOG_APPEAR_TIMEOUT_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        if ready_path.is_file():
            return True
        time.sleep(0.05)
    return False


class NativePathDialogService:
    """Spawn and validate one platform-native dialog at a time."""

    def __init__(
        self,
        allowed_roots: Iterable[str | os.PathLike[str]],
        *,
        python_executable: str | os.PathLike[str] | None = None,
        runner: Callable[..., Any] | None = None,
        platform_name: str | None = None,
        memory: RecentPathMemory | None = None,
        worker_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.allowed_roots = _normalize_roots(allowed_roots)
        self._runner = runner
        if python_executable is None:
            selected_python = selected_python_executable()
            default_worker_environment = miniforge_subprocess_environment(
                python_executable=selected_python,
                inherit_path=False,
            )
        else:
            selected_python = Path(python_executable).expanduser().resolve(strict=False)
            if runner is None:
                current_python = selected_python_executable()
                if os.path.normcase(str(selected_python)) != os.path.normcase(
                    str(current_python)
                ):
                    raise RuntimeError(
                        "Native dialog worker must use the running Miniforge interpreter"
                    )
                default_worker_environment = miniforge_subprocess_environment(
                    python_executable=current_python,
                    inherit_path=False,
                )
            else:
                default_worker_environment = None
        self.python_executable = str(selected_python)
        self.worker_environment = (
            dict(worker_environment)
            if worker_environment is not None
            else default_worker_environment
        )
        self.platform_name = platform_name or sys.platform
        self.memory = memory
        self._lock = threading.Lock()

    @property
    def supported(self) -> bool:
        return self.platform_name in {"win32", "darwin"}

    def select(self, payload: Mapping[str, Any] | DialogRequest) -> DialogSelection:
        request = (
            payload
            if isinstance(payload, DialogRequest)
            else DialogRequest.from_payload(payload)
        )
        if not self.supported:
            raise NativeDialogUnsupportedError(
                "Native path dialogs are only available on Windows and macOS."
            )
        if not self.allowed_roots:
            raise NativeDialogForbiddenError(
                "No allowed filesystem roots are configured."
            )
        if not self._lock.acquire(blocking=False):
            raise NativeDialogBusyError("Another native path dialog is already open.")
        try:
            response = self._run_worker(request)
            selection = self._validate_worker_response(request, response)
            if (
                self.memory is not None
                and selection.status == "selected"
                and selection.paths
            ):
                self.memory.remember(
                    context=request.memory_context,
                    dialog_mode=request.mode,
                    paths=selection.paths,
                )
            return selection
        finally:
            self._lock.release()

    def _run_worker(self, request: DialogRequest) -> Mapping[str, Any]:
        initial_path = (
            self.memory.resolve_initial(
                context=request.memory_context,
                dialog_mode=request.mode,
                current_value=request.initial_path,
            )
            if self.memory is not None
            else self._safe_initial_directory(request.initial_path)
        )
        worker_payload = request.to_worker_payload(initial_path=initial_path)
        command = [
            self.python_executable,
            "-m",
            "solar_apps.platform.paths.dialog_worker",
        ]
        try:
            if self._runner is None:
                with tempfile.TemporaryDirectory(
                    prefix="solar-native-dialog-ready-"
                ) as temporary:
                    ready_path = Path(temporary) / "visible"
                    process_payload = dict(worker_payload)
                    process_payload["_ready_path"] = str(ready_path)
                    completed = self._run_worker_process(
                        command,
                        input_text=json.dumps(process_payload),
                        ready_path=(
                            ready_path
                            if request.mode == "select_directory"
                            and self.platform_name == "win32"
                            else None
                        ),
                    )
            else:
                completed = self._runner(
                    command,
                    input=json.dumps(worker_payload),
                    text=True,
                    capture_output=True,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env=self.worker_environment,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NativeDialogUnavailableError(
                f"Could not start the native path dialog: {exc}"
            ) from exc
        if int(getattr(completed, "returncode", 1)) != 0:
            detail = str(getattr(completed, "stderr", "")).strip()
            raise NativeDialogUnavailableError(
                detail or "The native path dialog process failed."
            )
        stdout = str(getattr(completed, "stdout", "")).strip()
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise NativeDialogUnavailableError(
                "The native path dialog returned an invalid response."
            ) from exc
        if not isinstance(response, Mapping):
            raise NativeDialogUnavailableError(
                "The native path dialog returned an invalid response."
            )
        return response

    def _run_worker_process(
        self,
        command: list[str],
        *,
        input_text: str,
        ready_path: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=self.worker_environment,
        )
        if process.stdin is None:
            raise NativeDialogUnavailableError(
                "Native path dialog worker stdin was not created."
            )
        process.stdin.write(input_text)
        process.stdin.close()
        process.stdin = None
        startup_error = ""
        if ready_path is not None and not _wait_for_worker_ready(process, ready_path):
            if process.poll() is None:
                startup_error = (
                    "Windows native directory dialog did not become visible."
                )
                process.kill()
        stdout, stderr = process.communicate()
        stderr = str(stderr).strip()
        if startup_error:
            stderr = startup_error if not stderr else f"{stderr}\n{startup_error}"
        return subprocess.CompletedProcess(
            command,
            int(process.returncode or 0),
            str(stdout),
            stderr,
        )

    def _validate_worker_response(
        self, request: DialogRequest, response: Mapping[str, Any]
    ) -> DialogSelection:
        status = str(response.get("status", "")).strip()
        if status == "cancelled":
            return DialogSelection(status="cancelled")
        if status != "selected":
            raise NativeDialogUnavailableError(
                "The native path dialog returned an unknown status."
            )
        raw_paths = response.get("paths")
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) and path.strip() for path in raw_paths
        ):
            raise NativeDialogUnavailableError(
                "The native path dialog returned invalid paths."
            )
        expected_many = request.mode == "open_files"
        if not raw_paths or (not expected_many and len(raw_paths) != 1):
            raise NativeDialogUnavailableError(
                "The native path dialog returned an unexpected path count."
            )
        kind = {
            "open_file": "file",
            "open_files": "file",
            "select_directory": "directory",
            "save_file": "save_file",
        }[request.mode]
        selected = tuple(
            validate_allowed_path(
                path,
                allowed_roots=self.allowed_roots,
                kind=kind,
                default_suffix=request.default_suffix,
            )
            for path in raw_paths
        )
        return DialogSelection(status="selected", paths=selected)

    def _safe_initial_directory(self, value: str) -> str:
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = candidate.resolve(strict=False)
            if candidate.exists() and candidate.is_file():
                candidate = candidate.parent
            while not candidate.exists() and candidate != candidate.parent:
                candidate = candidate.parent
            if candidate.is_dir() and is_path_within_roots(
                candidate, self.allowed_roots
            ):
                return str(candidate)
        for root in self.allowed_roots:
            if root.is_dir():
                return str(root)
        return str(self.allowed_roots[0])
