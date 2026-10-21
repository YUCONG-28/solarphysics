"""The formal HMI entry point requires a caller-selected scientific region."""

from __future__ import annotations

import pytest

from solar_apps.workflows.hmi.overlay_cli import _DEFAULT_CONFIG, _build_parser


def test_overlay_requires_explicit_roi():
    parser = _build_parser(dict(_DEFAULT_CONFIG))
    with pytest.raises(SystemExit) as result:
        parser.parse_args([])
    assert result.value.code == 2


def test_overlay_accepts_explicit_cli_and_private_config_roi():
    parser = _build_parser(dict(_DEFAULT_CONFIG))
    assert parser.parse_args(["--roi-bounds", "-5", "-4", "6", "7"]).roi_bounds == [
        -5.0,
        -4.0,
        6.0,
        7.0,
    ]
    parser = _build_parser({**_DEFAULT_CONFIG, "roi_bounds": (-5, -4, 6, 7)})
    assert parser.parse_args([]).roi_bounds == (-5, -4, 6, 7)


def test_overlay_help_remains_available_without_roi(capsys):
    with pytest.raises(SystemExit) as result:
        _build_parser(dict(_DEFAULT_CONFIG)).parse_args(["--help"])
    assert result.value.code == 0
    assert "X_MIN Y_MIN X_MAX Y_MAX" in capsys.readouterr().out
