from __future__ import annotations

from solar_toolkit._utils.logging import SolarLogger


def test_solar_logger_instances_share_and_accumulate_handlers(tmp_path) -> None:
    """A second logger must not destroy the first instance's handlers."""
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"

    first = SolarLogger(log_file=str(first_log))
    first.info("first message")
    count_after_first = len(first.logger.handlers)

    second = SolarLogger(log_file=str(second_log))
    second.info("second message")

    # The first instance's file handler must still be attached.
    assert len(second.logger.handlers) > count_after_first
    base_names = {
        getattr(handler, "baseFilename", "") for handler in second.logger.handlers
    }
    assert str(first_log) in base_names
    assert str(second_log) in base_names


def test_solar_logger_console_handler_installed_once() -> None:
    first = SolarLogger()
    second = SolarLogger()
    stream_handlers = [
        handler
        for handler in second.logger.handlers
        if getattr(handler, "baseFilename", "") == ""
    ]
    # Only the single shared console handler exists.
    assert len(stream_handlers) == 1
    assert first.logger is second.logger
