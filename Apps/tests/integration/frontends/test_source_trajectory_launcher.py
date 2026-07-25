from __future__ import annotations

from types import SimpleNamespace

from solar_apps.frontends.radio.source_trajectory import source_app_launcher as launcher


def test_intentional_idle_shutdown_returns_success(monkeypatch) -> None:
    process = SimpleNamespace(returncode=1)
    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8767)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        launcher, "_wait_with_auto_stop", lambda *_args, **_kwargs: True
    )

    assert launcher.main(["--no-browser"]) == 0


def test_localized_netstat_output_is_decoded_safely(monkeypatch) -> None:
    observed = {}

    def fake_run(*_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher._has_browser_connection(8767) is False
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
