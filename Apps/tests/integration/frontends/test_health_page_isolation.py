"""Real per-module App 1.0 page isolation subprocesses for the health matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from solar_apps.cli import health
from solar_apps.frontends.app_v1.catalog import MODULES

APPS_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("module_id", [module.module_id for module in MODULES])
def test_app_v1_page_runs_in_its_own_offscreen_process(
    tmp_path: Path, module_id: str
) -> None:
    check, payload = health._run_page_check(
        module_id,
        apps_root=APPS_ROOT,
        local_root=tmp_path / "Local",
        allowed_root=tmp_path / "allowed",
    )
    assert check.status == "pass", check.detail
    assert payload is not None
    assert payload["selected_module"] == module_id
    assert payload["registered_count"] == len(MODULES)
    assert payload["foreign_qt_loaded"] is False
    assert payload["running_qprocess_count"] == 0
    assert payload["process_running"] is False
