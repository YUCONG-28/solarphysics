"""Isolated PySide6 worker for Windows/macOS native file and folder dialogs."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_WINDOW_APPEAR_TIMEOUT_SECONDS = 5.0


def _name_filters(extensions: list[str]) -> list[str]:
    patterns = ["*" if item == "*" else f"*{item}" for item in extensions]
    if not patterns:
        return ["All files (*)"]
    return [f"Supported files ({' '.join(patterns)})", "All files (*)"]


def _windows_desktop_is_interactive() -> bool:
    if sys.platform != "win32":
        return False

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user_object_name = 2
    desktop_read_objects = 0x0001

    def object_name(handle: int) -> str:
        required = wintypes.DWORD()
        user32.GetUserObjectInformationW(
            handle, user_object_name, None, 0, ctypes.byref(required)
        )
        if not required.value:
            return ""
        buffer = ctypes.create_unicode_buffer(required.value // 2 + 1)
        if not user32.GetUserObjectInformationW(
            handle,
            user_object_name,
            buffer,
            ctypes.sizeof(buffer),
            ctypes.byref(required),
        ):
            return ""
        return buffer.value

    station_name = object_name(user32.GetProcessWindowStation())
    current_desktop = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
    input_desktop = user32.OpenInputDesktop(0, False, desktop_read_objects)
    if not input_desktop:
        return False
    try:
        return (
            station_name.casefold() == "winsta0"
            and object_name(current_desktop).casefold()
            == object_name(input_desktop).casefold()
        )
    finally:
        user32.CloseDesktop(input_desktop)


def _keep_windows_directory_dialog_topmost(
    title: str, stop_event: threading.Event, ready_path: str = ""
) -> None:
    """Promote the native Windows dialog created behind QFileDialog."""

    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )
    process_id = os.getpid()
    hwnd_topmost = wintypes.HWND(-1)
    swp_flags = 0x0001 | 0x0002 | 0x0010  # NOSIZE | NOMOVE | NOACTIVATE
    deadline = time.monotonic() + _WINDOW_APPEAR_TIMEOUT_SECONDS
    ready_reported = False

    while not stop_event.is_set() and (ready_reported or time.monotonic() < deadline):
        matches: list[int] = []

        def find_dialog(hwnd: int, _lparam: int) -> bool:
            window_process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_process_id))
            if window_process_id.value != process_id or not user32.IsWindowVisible(
                hwnd
            ):
                return True
            title_length = user32.GetWindowTextLengthW(hwnd)
            title_buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, title_length + 1)
            class_buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))
            if title_buffer.value == title and class_buffer.value == "#32770":
                matches.append(hwnd)
                return False
            return True

        callback = enum_windows_callback(find_dialog)
        user32.EnumWindows(callback, 0)
        if matches:
            if user32.SetWindowPos(matches[0], hwnd_topmost, 0, 0, 0, 0, swp_flags):
                # Windows 11's native shell dialog can accept HWND_TOPMOST
                # without retaining the corresponding extended style. Keep
                # raising it as well so later normal windows cannot cover it.
                user32.BringWindowToTop(matches[0])
                if ready_path and not ready_reported:
                    Path(ready_path).touch()
                ready_reported = True
        stop_event.wait(0.05)
    if not stop_event.is_set() and not ready_reported:
        print(
            "Windows native directory dialog did not become visible.",
            file=sys.stderr,
            flush=True,
        )
        os._exit(2)


def run_dialog(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one QFileDialog and return its small JSON-compatible result."""

    mode = str(payload["mode"])
    windows_directory = mode == "select_directory" and sys.platform == "win32"
    if windows_directory and not _windows_desktop_is_interactive():
        raise RuntimeError(
            "Windows native directory dialog is unavailable from the current "
            "non-interactive desktop."
        )
    topmost_stop = threading.Event()
    topmost_thread = None
    if windows_directory:
        topmost_thread = threading.Thread(
            target=_keep_windows_directory_dialog_topmost,
            args=(
                str(payload["title"]),
                topmost_stop,
                str(payload.get("_ready_path") or ""),
            ),
            name="native-directory-dialog-topmost",
            daemon=True,
        )
        topmost_thread.start()

    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication(["solar-native-path-dialog"])
    dialog = QFileDialog()
    dialog.setOption(QFileDialog.DontUseNativeDialog, False)
    dialog.setWindowTitle(str(payload["title"]))
    initial_path = str(payload.get("initial_path") or "")
    if initial_path:
        dialog.setDirectory(initial_path)
    extensions = [str(item) for item in payload.get("extensions") or []]
    dialog.setNameFilters(_name_filters(extensions))
    if mode == "open_file":
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)
    elif mode == "open_files":
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFiles)
    elif mode == "select_directory":
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
    elif mode == "save_file":
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        suffix = str(payload.get("default_suffix") or "").lstrip(".")
        if suffix:
            dialog.setDefaultSuffix(suffix)
    else:
        raise ValueError(f"Unsupported dialog mode: {mode!r}")
    try:
        accepted = dialog.exec() == QFileDialog.Accepted
        return {
            "status": "selected" if accepted else "cancelled",
            "paths": [str(path) for path in dialog.selectedFiles()] if accepted else [],
        }
    finally:
        topmost_stop.set()
        if topmost_thread is not None:
            topmost_thread.join(timeout=1.0)
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise TypeError("Worker payload must be a JSON object.")
        result = run_dialog(payload)
    except Exception as exc:  # The parent converts worker failures into HTTP 503.
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
