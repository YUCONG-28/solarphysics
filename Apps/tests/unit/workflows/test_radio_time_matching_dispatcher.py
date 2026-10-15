"""Public radio routing keeps acceptance inputs explicit and unmodified."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from solar_apps.cli import router
from solar_apps.workflows.radio import dispatcher


def test_public_workflow_forwards_acceptance_window_without_opening_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = [
        "--event-root", "/allowed/synthetic-event",
        "--muser-directory", "/allowed/synthetic-muser",
        "--output", "/allowed/synthetic-results",
        "--time-start", "2000-01-01T00:00:00.000000001Z",
        "--time-end", "2000-01-01T00:00:01Z",
    ]
    calls = []

    def acceptance_main(forwarded):
        calls.append((list(forwarded), sys.argv[0]))
        return 17

    def import_target(name):
        assert name == "solar_apps.workflows.radio.time_matching_acceptance"
        return SimpleNamespace(main=acceptance_main)

    monkeypatch.setattr(dispatcher, "import_module", import_target)
    original_program = sys.argv[0]

    assert router.main(["workflow", "radio", "time-matching-acceptance", *arguments]) == 17
    assert calls == [(arguments, "solar-apps workflow radio time-matching-acceptance")]
    assert sys.argv[0] == original_program


def test_acceptance_route_restores_program_when_target_rejects_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(_arguments):
        raise SystemExit(2)

    monkeypatch.setattr(
        dispatcher, "import_module", lambda _name: SimpleNamespace(main=reject)
    )
    original_program = sys.argv[0]
    with pytest.raises(SystemExit) as exc:
        dispatcher.main(["time-matching-acceptance", "--unknown"])
    assert exc.value.code == 2
    assert sys.argv[0] == original_program
