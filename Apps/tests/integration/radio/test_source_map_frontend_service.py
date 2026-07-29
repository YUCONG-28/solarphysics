from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.radio import source_map_workflow as workflow
from solar_apps.frontends.radio.source_map.service import (
    PathPolicy,
    discover_candidates,
    parse_request_config,
    validate_advanced_config,
)


def _write_radio_fits(path: Path, *, polarization: str, bunit: str = "K") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["DATE-OBS"] = "2025-01-24T04:48:30"
    header["FREQ"] = 149.0
    header["POLAR"] = polarization
    header["BUNIT"] = bunit
    header["CRVAL1"] = 0.0
    header["CRPIX1"] = 4.5
    header["CDELT1"] = 10.0
    header["CRVAL2"] = 0.0
    header["CRPIX2"] = 4.5
    header["CDELT2"] = 10.0
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32), header=header).writeto(path)


def test_allowed_root_policy_rejects_outside_path(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = PathPolicy([allowed])
    assert policy.resolve(allowed / "future", must_exist=False) == allowed / "future"
    with pytest.raises(PermissionError):
        policy.resolve(tmp_path / "outside", must_exist=False)


def test_advanced_json_rejects_unknown_and_protected_options() -> None:
    assert validate_advanced_config({"scale_factor": 2.5}, workflow.DEFAULT_CONFIG) == {
        "scale_factor": 2.5
    }
    with pytest.raises(ValueError, match="Unknown"):
        validate_advanced_config({"not_a_real_option": 1}, workflow.DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="protected"):
        validate_advanced_config({"output_dir": "elsewhere"}, workflow.DEFAULT_CONFIG)


def test_frontend_frequency_list_preserves_integer_directory_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bands"
    source.mkdir()
    policy = PathPolicy([tmp_path])
    cfg = parse_request_config(
        {
            "mode": "multi_band",
            "source_path": str(source),
            "output_dir": str(tmp_path / "output"),
            "frequencies": "149, 164.5",
            "polarization": "RR",
            "gaussian_overlay": False,
            "spectrogram_panel": False,
            "background_mode": "off",
            "cmap": "hot",
            "color_range_mode": "auto",
            "start_idx": 0,
            "end_idx": None,
            "advanced": {},
        },
        policy=policy,
    )
    assert cfg["multi_band_freqs"] == [149, 164.5]
    assert cfg["start_idx"] == 0
    assert cfg["end_idx"] is None


def test_frontend_input_slice_overrides_event_config(tmp_path: Path) -> None:
    source = tmp_path / "radio"
    source.mkdir()
    policy = PathPolicy([tmp_path])

    cfg = parse_request_config(
        {
            "mode": "single_band",
            "source_path": str(source),
            "output_dir": str(tmp_path / "output"),
            "polarization": "RR",
            "start_idx": 2,
            "end_idx": 5,
            "advanced": {},
        },
        policy=policy,
    )

    assert cfg["start_idx"] == 2
    assert cfg["end_idx"] == 5


def test_seven_digit_calendar_date_ignores_dated_parent_directory(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path / "app-v1-realdata-20260728T051015Z" / "149MHz_2025124_044829_000.fits"
    )

    observed = workflow.radio_datetime_from_header_or_path(
        fits.Header(),
        str(path),
        {"date_format": "auto"},
    )

    assert observed is not None
    assert observed.isoformat() == "2025-01-24T04:48:29"


def test_single_band_rr_ll_discovery_freezes_matched_pair(tmp_path: Path) -> None:
    rr_path = tmp_path / "149MHz" / "RR" / "frame.fits"
    ll_path = tmp_path / "149MHz" / "LL" / "frame.fits"
    _write_radio_fits(rr_path, polarization="RR")
    _write_radio_fits(ll_path, polarization="LL")
    policy = PathPolicy([tmp_path])
    cfg = parse_request_config(
        {
            "mode": "single_band",
            "source_path": str(rr_path),
            "output_dir": str(tmp_path / "output"),
            "polarization": "RR+LL",
            "gaussian_overlay": False,
            "spectrogram_panel": False,
            "background_mode": "off",
            "cmap": "hot",
            "color_range_mode": "auto",
            "advanced": {},
        },
        policy=policy,
    )
    candidates = discover_candidates(cfg, policy=policy)
    assert len(candidates) == 1
    assert candidates[0]["paths"] == [str(rr_path), str(ll_path)]
    assert candidates[0]["pairing_status"] == "RR+LL matched"
    assert candidates[0]["frequencies_mhz"] == [149.0]
