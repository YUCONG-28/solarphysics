"""Internal timing and logging helpers."""

from __future__ import annotations

import functools
import logging
import os
import time


def timing_decorator(func):
    """Print the execution time of ``func`` while preserving its metadata."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[{func.__name__}] execution time: {end_time - start_time:.3f} s")
        return result

    return wrapper


class SolarLogger:
    """Logger configured for local solar-data processing workflows.

    Handlers are installed once per stream/file and shared across instances so
    that constructing a second logger no longer destroys the first one's
    handlers.
    """

    _console_installed = False
    _file_handlers: set[str] = set()

    def __init__(self, log_file: str | None = None, level: str = "INFO"):
        self.logger = logging.getLogger("solar_data_processing")
        self.logger.setLevel(getattr(logging, level))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if not SolarLogger._console_installed:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            SolarLogger._console_installed = True

        if log_file:
            resolved = os.path.abspath(log_file)
            if resolved not in SolarLogger._file_handlers:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                SolarLogger._file_handlers.add(resolved)

    def debug(self, msg: str) -> None:
        """Log a debug message."""

        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an informational message."""

        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""

        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""

        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""

        self.logger.critical(msg)


__all__ = ["SolarLogger", "timing_decorator"]
