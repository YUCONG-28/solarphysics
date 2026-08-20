"""Focused contracts for the directory-driven App health matrix."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar_apps.cli import health
from solar_apps.cli import router
from solar_apps.frontends.app_v1.catalog import MODULES
from solar_apps.frontends.catalog import FRONTENDS


def test_health_tool_is_registered_in_router() -> None:
    assert router.TOOL_TARGETS["health"] == "solar_apps.cli.health"


def test_router_dispatches_health_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        router,
        "forward_main",
        lambda module, arguments, **_kwargs: calls.append((module, list(arguments)))
        or 0,
    )
    assert router.main(["tools", "health", "--output", "report.json"]) == 0
    assert calls == [("solar_apps.cli.health", ["--output", "report.json"])]


def test_check_ids_derive_from_catalogues_only() -> None:
    ids = set(health.check_ids())
    assert {f"entry-{frontend.id}" for frontend in FRONTENDS} <= ids
    assert {f"help-{frontend.id}" for frontend in FRONTENDS} <= ids
    assert {f"page-{module.module_id}" for module in MODULES} <= ids
    assert {
        f"legacy-{frontend.id}"
        for frontend in FRONTENDS
        if frontend.toolkit in health.LEGACY_TOOLKITS
    } <= ids
    assert len(FRONTENDS) == len([item for item in ids if item.startswith("entry-")])
    assert len(FRONTENDS) == len([item for item in ids if item.startswith("help-")])
    assert len(MODULES) == len([item for item in ids if item.startswith("page-")])


def test_legacy_check_ids_only_cover_web_toolkits() -> None:
    legacy_ids = {item for item in health.check_ids() if item.startswith("legacy-")}
    expected = {
        f"legacy-{frontend.id}"
        for frontend in FRONTENDS
        if frontend.toolkit in health.LEGACY_TOOLKITS
    }
    assert legacy_ids == expected
    non_web = {
        f"legacy-{frontend.id}"
        for frontend in FRONTENDS
        if frontend.toolkit not in health.LEGACY_TOOLKITS
    }
    assert legacy_ids.isdisjoint(non_web)


def test_default_legacy_checks_are_not_run_with_reason() -> None:
    checks = [
        health.legacy_not_run_check(spec)
        for spec in FRONTENDS
        if spec.toolkit in health.LEGACY_TOOLKITS
    ]
    assert checks
    for check in checks:
        assert check.status == "not_run"
        assert "reason=disabled-by-default" in check.detail
        assert "toolkit=" in check.detail


def test_report_schema_and_status_aggregation() -> None:
    started = health.now_utc()
    finished = health.now_utc()
    checks = [
        health.CheckResult("catalog-frontends", "catalog", "pass", "ok", 0.1),
        health.CheckResult("legacy-workbench", "legacy.servers", "not_run", "off", 0.0),
    ]
    report = health.build_report(checks, started, finished)
    assert set(report) == {
        "schema_version",
        "overall_status",
        "started_at_utc",
        "finished_at_utc",
        "checks",
    }
    assert report["schema_version"] == health.SCHEMA_VERSION
    assert report["overall_status"] == "pass"
    for item in report["checks"]:
        assert set(item) == {
            "id",
            "category",
            "status",
            "detail",
            "duration_seconds",
        }
        assert item["status"] in {"pass", "fail", "not_run"}
    failed = health.build_report(
        [health.CheckResult("page-workbench", "app_v1.pages", "fail", "boom", 0.2)],
        started,
        finished,
    )
    assert failed["overall_status"] == "fail"
    assert health.aggregate_status(checks) == "pass"
    assert (
        health.aggregate_status(
            [health.CheckResult("x", "app_v1.pages", "fail", "boom", 0.2)]
        )
        == "fail"
    )


def test_atomic_write_json_creates_parents_and_leaves_no_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "report.json"
    health.atomic_write_json(
        target,
        {"schema_version": health.SCHEMA_VERSION, "checks": []},
    )
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == health.SCHEMA_VERSION
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]
    health.atomic_write_json(target, {"schema_version": "2.0"})
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == "2.0"
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]


def test_main_exits_nonzero_and_writes_report_when_required_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_health(**_kwargs: object) -> dict[str, object]:
        return health.build_report(
            [health.CheckResult("page-workbench", "app_v1.pages", "fail", "boom", 0.1)],
            health.now_utc(),
            health.now_utc(),
        )

    monkeypatch.setattr(health, "run_health", fake_run_health)
    target = tmp_path / "report.json"
    assert health.main(["--output", str(target)]) == 1
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["overall_status"] == "fail"
    assert report["checks"][0]["status"] == "fail"


def test_main_exits_zero_when_all_required_checks_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        health,
        "run_health",
        lambda **_kwargs: health.build_report([], health.now_utc(), health.now_utc()),
    )
    target = tmp_path / "report.json"
    assert health.main(["--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["overall_status"] == "pass"


def test_main_writes_fail_report_when_runner_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("matrix exploded")

    monkeypatch.setattr(health, "run_health", boom)
    target = tmp_path / "report.json"
    assert health.main(["--output", str(target)]) == 1
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["overall_status"] == "fail"
    assert report["checks"][0]["id"] == "health-runner"


def _fast_run_health(
    monkeypatch: pytest.MonkeyPatch, *, local_root: Path | None = None
) -> dict[str, object]:
    def fake_check(*_args: object, **_kwargs: object) -> health.CheckResult:
        return health.CheckResult("fake", "catalog", "pass", "ok", 0.0)

    monkeypatch.setattr(health, "_catalog_checks", lambda: [])
    monkeypatch.setattr(health, "_run_entry_check", fake_check)
    monkeypatch.setattr(health, "_run_help_check", lambda *_a, **_k: (fake_check(), ""))
    monkeypatch.setattr(
        health, "_run_page_check", lambda *_a, **_k: (fake_check(), None)
    )
    monkeypatch.setattr(health, "_run_smoke_check", fake_check)
    monkeypatch.setattr(health, "_run_legacy_server_check", fake_check)
    return health.run_health(local_root=local_root)


def test_run_health_cleans_default_temporary_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    real_temporary_directory = health.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args: object, **kwargs: object):
        temporary = real_temporary_directory(*args, **kwargs)
        created.append(temporary.name)
        return temporary

    monkeypatch.setattr(
        health.tempfile, "TemporaryDirectory", tracking_temporary_directory
    )
    report = _fast_run_health(monkeypatch)

    assert report["overall_status"] == "pass"
    assert created
    assert all(not Path(name).exists() for name in created)


def test_run_health_preserves_injected_local_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = tmp_path / "injected-local"

    def unexpected_temporary_directory(*_args: object, **_kwargs: object):
        raise AssertionError(
            "TemporaryDirectory must not be created for an injected local_root"
        )

    monkeypatch.setattr(
        health.tempfile, "TemporaryDirectory", unexpected_temporary_directory
    )
    report = _fast_run_health(monkeypatch, local_root=local)

    assert report["overall_status"] == "pass"
    assert local.is_dir()
    assert (local / "allowed").is_dir()


def test_probe_http_accepts_only_2xx_and_3xx(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(poll=lambda: None)

    class _SuccessfulResponse:
        status = 204

        def __enter__(self) -> _SuccessfulResponse:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr(
        health.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _SuccessfulResponse(),
    )
    assert health._probe_http(process, 9998, timeout=5) == ("ok", "204")

    def error_urlopen(url: str, timeout: float = 1.0):
        raise urllib.error.HTTPError(url, 503, "unavailable", None, None)

    monkeypatch.setattr(health.urllib.request, "urlopen", error_urlopen)
    assert health._probe_http(process, 9998, timeout=5) == ("http-error", "HTTP 503")


def test_terminate_posix_group_escalates_to_sigkill_when_group_survives_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == health.signal.SIGKILL:
            raise ProcessLookupError("group gone")

    monkeypatch.setattr(health.os, "killpg", fake_killpg)
    process = SimpleNamespace(
        poll=lambda: 0,
        pid=111,
        wait=lambda timeout=None: None,
        terminate=lambda: None,
        kill=lambda: None,
    )

    health._terminate_process_tree(
        process, pgid=9876, platform="posix", grace_seconds=0.05
    )

    signals = [sig for _pgid, sig in calls]
    assert health.signal.SIGTERM in signals
    assert health.signal.SIGKILL in signals
    term_index = signals.index(health.signal.SIGTERM)
    kill_index = signals.index(health.signal.SIGKILL)
    assert term_index < kill_index
    assert calls[-1][0] == 9876


def test_start_legacy_process_windows_uses_new_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured.update(kwargs)
        return SimpleNamespace(pid=333)

    monkeypatch.setattr(health.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(health.os, "name", "nt")

    process, pgid = health._start_legacy_process(
        ["python", "-m", "demo"],
        cwd=Path("."),
        env={},
        stdout=object(),
    )

    assert pgid is None
    assert process.pid == 333
    assert "creationflags" in captured
    assert "start_new_session" not in captured


def test_terminate_windows_tree_uses_taskkill_and_never_killpg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(health.subprocess, "run", fake_run)
    monkeypatch.setattr(
        health.os,
        "killpg",
        lambda *_args: pytest.fail("killpg must not be used on Windows"),
    )
    process = SimpleNamespace(
        pid=222, wait=lambda timeout=None: None, kill=lambda: None
    )

    health._terminate_process_tree(process, pgid=None, root_pid=222, platform="nt")

    assert commands and commands[0][:4] == ["taskkill", "/PID", "222", "/T"]
    assert "/F" in commands[0]


def test_legacy_check_maps_http_error_to_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = next(item for item in FRONTENDS if item.toolkit == "flask")
    monkeypatch.setattr(health, "_free_port", lambda: 55555)
    monkeypatch.setattr(
        health,
        "_start_legacy_process",
        lambda *_args, **_kwargs: (SimpleNamespace(pid=4242), 4242),
    )
    monkeypatch.setattr(
        health,
        "_probe_http",
        lambda *_args, **_kwargs: ("http-error", "HTTP 500"),
    )
    monkeypatch.setattr(health, "_terminate_process_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(health, "_port_has_listener", lambda port: False)

    check = health._run_legacy_server_check(
        spec,
        include_legacy_servers=True,
        help_text="",
        apps_root=tmp_path,
        local_root=tmp_path,
        python="python",
        env={},
    )

    assert check.status == "not_run"
    assert "reason=http-error" in check.detail


def test_legacy_check_reports_cleanup_failed_when_port_stays_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = next(item for item in FRONTENDS if item.toolkit == "flask")
    monkeypatch.setattr(health, "_free_port", lambda: 55556)
    monkeypatch.setattr(
        health,
        "_start_legacy_process",
        lambda *_args, **_kwargs: (SimpleNamespace(pid=4243), 4243),
    )
    monkeypatch.setattr(health, "_probe_http", lambda *_args, **_kwargs: ("ok", "200"))
    monkeypatch.setattr(health, "_terminate_process_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(health, "_wait_for_port_closed", lambda port, **kwargs: False)

    check = health._run_legacy_server_check(
        spec,
        include_legacy_servers=True,
        help_text="",
        apps_root=tmp_path,
        local_root=tmp_path,
        python="python",
        env={},
    )

    assert check.status == "not_run"
    assert "reason=cleanup-failed" in check.detail


def test_measure_reports_invalid_status_value() -> None:
    check = health._measure("sample", "catalog", lambda: ("bogus", "original"))

    assert check.status == "fail"
    assert "invalid status 'bogus'" in check.detail
    assert "original" not in check.detail


def test_help_check_duration_includes_subprocess_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    def slow_run(*_args: object, **_kwargs: object):
        time.sleep(0.15)
        return SimpleNamespace(returncode=0, stdout="--no-browser\n", stderr="")

    monkeypatch.setattr(health, "_run", slow_run)
    spec = FRONTENDS[0]

    check, help_text = health._run_help_check(
        spec, apps_root=Path("."), python="python", env={}
    )

    assert check.status == "pass"
    assert check.duration_seconds >= 0.15
    assert "--no-browser" in help_text


@pytest.mark.parametrize("inject_local", [False, True])
def test_run_health_cleans_managed_root_even_when_checks_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inject_local: bool,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> list[health.CheckResult]:
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(health, "_catalog_checks", boom)

    if inject_local:
        local = tmp_path / "injected-local"

        def unexpected_temporary_directory(*_args: object, **_kwargs: object):
            raise AssertionError(
                "TemporaryDirectory must not be created for an injected local_root"
            )

        monkeypatch.setattr(
            health.tempfile, "TemporaryDirectory", unexpected_temporary_directory
        )
        with pytest.raises(RuntimeError, match="catalog exploded"):
            health.run_health(local_root=local)
        assert local.is_dir()
        return

    created: list[str] = []
    real_temporary_directory = health.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args: object, **kwargs: object):
        temporary = real_temporary_directory(*args, **kwargs)
        created.append(temporary.name)
        return temporary

    monkeypatch.setattr(
        health.tempfile, "TemporaryDirectory", tracking_temporary_directory
    )
    with pytest.raises(RuntimeError, match="catalog exploded"):
        health.run_health()
    assert created
    assert all(not Path(name).exists() for name in created)


def test_check_ids_match_actual_matrix_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[str] = []

    def fake_measure(check_id: str, category: str, fn):
        recorded.append(check_id)
        return health.CheckResult(check_id, category, "pass", "ok", 0.0)

    monkeypatch.setattr(health, "_measure", fake_measure)
    report = health.run_health(local_root=tmp_path / "Local")

    assert report["overall_status"] == "pass"
    assert tuple(recorded) == health.check_ids()
    assert "catalog-frontends" in health.check_ids()
    assert "catalog-app-v1-modules" in health.check_ids()


def test_wait_for_port_closed_accepts_delayed_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([True, False])
    monkeypatch.setattr(health, "_port_has_listener", lambda port: next(states))

    assert health._wait_for_port_closed(12345, timeout=5.0) is True


def test_wait_for_port_closed_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_port_has_listener", lambda port: True)

    assert health._wait_for_port_closed(12345, timeout=0.2) is False


def test_legacy_check_passes_after_port_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = next(item for item in FRONTENDS if item.toolkit == "flask")
    monkeypatch.setattr(health, "_free_port", lambda: 55557)
    monkeypatch.setattr(
        health,
        "_start_legacy_process",
        lambda *_args, **_kwargs: (SimpleNamespace(pid=4244), 4244),
    )
    monkeypatch.setattr(health, "_probe_http", lambda *_args, **_kwargs: ("ok", "200"))
    monkeypatch.setattr(health, "_terminate_process_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(health, "_wait_for_port_closed", lambda port, **kwargs: True)

    check = health._run_legacy_server_check(
        spec,
        include_legacy_servers=True,
        help_text="",
        apps_root=tmp_path,
        local_root=tmp_path,
        python="python",
        env={},
    )

    assert check.status == "pass"
    assert "status=200" in check.detail


def test_atomic_write_json_removes_temporary_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.json"
    health.atomic_write_json(target, {"ok": True})

    def failing_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(health.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        health.atomic_write_json(target, {"bad": True})

    leftovers = [path.name for path in tmp_path.iterdir() if path.name != "report.json"]
    assert leftovers == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
