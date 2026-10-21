"""Help and rejected CLI input must not create private runtime state."""

import pytest

from solar_apps.cli import admin, router
from solar_apps.platform.paths.allowed_roots import AllowedRootPolicyError
from solar_apps.workflows.radio.dispatcher import _COMMANDS


@pytest.fixture
def private_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "uninitialized"
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(runtime))
    for key in (
        "SOLAR_APPS_CONFIG",
        "SOLAR_PHYSICS_CONFIG",
        "SOLAR_APPS_ALLOWED_ROOTS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(router, "inspect_miniforge_runtime", lambda: None)
    monkeypatch.setattr(
        admin,
        "initialize_runtime",
        lambda *args, **kwargs: pytest.fail("implicit runtime initialization"),
    )
    return runtime


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["frontend", "--help"],
        ["workflow", "--help"],
        ["tools", "--help"],
        ["admin", "--help"],
        ["workflow", "radio", "--help"],
        ["tools", "release", "prepare", "--help"],
        ["frontend", "image-composer", "--help"],
    ],
)
def test_help_is_available_without_runtime_files(private_runtime, arguments):
    try:
        status = router.main(arguments)
    except SystemExit as exc:
        status = exc.code
    assert status == 0
    assert not private_runtime.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["unknown"],
        ["frontend", "unknown"],
        ["workflow", "unknown"],
        ["tools", "unknown"],
        ["workflow", "radio", "unknown"],
        ["tools", "release", "unknown"],
        ["frontend", "image-composer", "--unknown"],
    ],
)
def test_invalid_commands_do_not_initialize(private_runtime, arguments):
    try:
        status = router.main(arguments)
    except SystemExit as exc:
        status = exc.code
    assert status == 2
    assert not private_runtime.exists()


def test_radio_help_lists_every_registered_workflow(private_runtime, capsys):
    assert router.main(["workflow", "radio", "--help"]) == 0
    output = capsys.readouterr().out
    assert all(command in output for command in _COMMANDS)


def test_frontend_missing_initialization_has_an_actionable_error(
    private_runtime, capsys
):
    assert router.main(["frontend", "workbench"]) == 2
    assert "admin init" in capsys.readouterr().err
    assert not private_runtime.exists()


def test_initialized_frontend_keeps_its_existing_dispatch(private_runtime, monkeypatch):
    config = private_runtime / "configs/paths.local.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("apps:\n  allowed_roots: []\n")
    calls = []
    monkeypatch.setattr(
        router,
        "forward_main",
        lambda target, args, **kwargs: calls.append((target, args)) or 0,
    )
    assert router.main(["frontend", "workbench", "--port", "9000"]) == 0
    assert calls == [(router.FRONTEND_TARGETS["workbench"], ["--port", "9000"])]
    assert list(private_runtime.rglob("*")) == [config.parent, config]


@pytest.mark.parametrize("config_env", ["SOLAR_APPS_CONFIG", "SOLAR_PHYSICS_CONFIG"])
def test_custom_configuration_does_not_require_default_runtime_config(
    private_runtime, tmp_path, monkeypatch, config_env
):
    config = tmp_path / "custom.yaml"
    config.write_text("apps:\n  allowed_roots: []\n")
    monkeypatch.setenv(config_env, str(config))
    calls = []
    monkeypatch.setattr(
        router, "forward_main", lambda *args, **kwargs: calls.append(args) or 0
    )
    assert router.main(["frontend", "workbench"]) == 0
    assert len(calls) == 1
    assert not private_runtime.exists()


def test_explicit_allowed_roots_keep_existing_override_contract(
    private_runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr(router, "forward_main", lambda *args, **kwargs: 0)
    assert router.main(["frontend", "workbench", "--allowed-roots", str(tmp_path)]) == 0
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    assert router.main(["frontend", "workbench"]) == 0
    assert not private_runtime.exists()


def test_known_runtime_and_allowed_root_errors_are_clear(
    private_runtime, monkeypatch, capsys
):
    from solar_apps.platform.layout import RuntimeLayout

    monkeypatch.setattr(
        router,
        "forward_main",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AllowedRootPolicyError("configured root is too broad")
        ),
    )
    assert router.main(["workflow", "radio", "centers"]) == 2
    assert "invalid allowed-root configuration" in capsys.readouterr().err
    monkeypatch.setattr(
        RuntimeLayout,
        "discover",
        lambda: (_ for _ in ()).throw(ValueError("runtime points into public source")),
    )
    assert router.main(["workflow", "radio", "centers"]) == 2
    assert "invalid private runtime configuration" in capsys.readouterr().err
    assert not private_runtime.exists()


@pytest.mark.parametrize(
    "error", [ValueError("scientific bug"), RuntimeError("implementation bug")]
)
def test_unexpected_execution_errors_are_not_hidden(
    private_runtime, monkeypatch, error
):
    monkeypatch.setattr(
        router, "forward_main", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )
    with pytest.raises(type(error), match=str(error)):
        router.main(["workflow", "radio", "centers"])
