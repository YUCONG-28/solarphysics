from __future__ import annotations

from solar_apps.frontends.radio.roi_lightcurve import (
    roi_lightcurve_launcher as launcher,
)


class _Process:
    returncode = 1


def test_idle_shutdown_is_a_success(monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        launcher, "_wait_with_auto_stop", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(launcher.webbrowser, "open", lambda _url: None)

    assert launcher.main(["--port", "8766", "--no-browser"]) == 0


def test_localized_netstat_output_is_decoded_safely(monkeypatch) -> None:
    observed = {}

    class _Completed:
        stdout = None

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return _Completed()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher._has_browser_connection(8766) is False
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
