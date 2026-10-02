from __future__ import annotations

from pathlib import Path

import pandas as pd

from solar_toolkit.radio.quicklook import (
    filter_valid_gaussian_centers,
    resolve_gaussian_csv,
    run_gaussian_newkirk_quicklook,
)

EVENT_CONFIG = {
    "user": {"data": {"multi_band_freqs": [149.0, 164.0]}},
    "output": {
        "output_dir": "outputs/radio/2000-01-01",
        "analysis_subdir": "gaussian_spectrogram_overlay",
        "gaussian_diagnostics_csv": "radio_gaussian_fit_diagnostics.csv",
    },
}


def _gaussian_row(
    *,
    time: str,
    freq: float,
    x_arcsec: float,
    y_arcsec: float,
    quality_flag: str = "ok",
    overlay_valid: bool = True,
    trajectory_valid: bool = True,
) -> dict:
    return {
        "source_file": f"{freq:g}MHz_test.fits",
        "time": time,
        "freq": freq,
        "polarization": "RR+LL",
        "center_x_arcsec": x_arcsec,
        "center_y_arcsec": y_arcsec,
        "center_x_pixel": 128.0,
        "center_y_pixel": 128.0,
        "fwhm_valid": True,
        "overlay_valid": overlay_valid,
        "trajectory_valid": trajectory_valid,
        "quality_flag": quality_flag,
        "quality_flag_detail": "",
    }


def test_filter_valid_gaussian_centers_uses_quality_and_visibility_flags():
    data = pd.DataFrame(
        [
            _gaussian_row(
                time="20000101044840000",
                freq=149.0,
                x_arcsec=1100.0,
                y_arcsec=0.0,
            ),
            _gaussian_row(
                time="20000101044841000",
                freq=149.0,
                x_arcsec=100.0,
                y_arcsec=100.0,
                overlay_valid=False,
            ),
            _gaussian_row(
                time="20000101044842000",
                freq=164.0,
                x_arcsec=1110.0,
                y_arcsec=10.0,
                quality_flag="unphysical_size",
                trajectory_valid=False,
            ),
        ]
    )

    result = filter_valid_gaussian_centers(data)

    assert len(result) == 1
    assert result.loc[0, "freq"] == 149.0
    assert result.loc[0, "center_x_arcsec"] == 1100.0


def test_quicklook_generates_isolated_outputs_from_gaussian_csv(tmp_path):
    gaussian_csv = tmp_path / "radio_gaussian_fit_diagnostics.csv"
    output_dir = tmp_path / "quicklook_outputs"
    pd.DataFrame(
        [
            _gaussian_row(
                time="20000101044840000",
                freq=149.0,
                x_arcsec=1100.0,
                y_arcsec=0.0,
            ),
            _gaussian_row(
                time="20000101044841000",
                freq=149.0,
                x_arcsec=100.0,
                y_arcsec=100.0,
            ),
            _gaussian_row(
                time="20000101044842000",
                freq=164.0,
                x_arcsec=1110.0,
                y_arcsec=10.0,
                quality_flag="unphysical_size",
                overlay_valid=False,
                trajectory_valid=False,
            ),
        ]
    ).to_csv(gaussian_csv, index=False)

    result = run_gaussian_newkirk_quicklook(
        gaussian_csv=gaussian_csv,
        config_name=EVENT_CONFIG,
        output_dir=output_dir,
    )

    expected_files = {
        "valid_centers_csv": "radio_gaussian_valid_centers.csv",
        "height_rows_csv": "gaussian_newkirk_height_rows.csv",
    }
    for key, filename in expected_files.items():
        path = output_dir / filename
        assert Path(result[key]) == path
        assert path.exists()
        assert path.stat().st_size > 0

    height_plot = Path(result["height_plot"])
    trajectory_plot = Path(result["trajectory_plot"])
    assert height_plot.name == (
        "0001_20000101T044840Z-20000101T044842Z_" "radio_newkirk_height_comparison.png"
    )
    assert trajectory_plot.name == (
        "0002_20000101T044840Z-20000101T044841Z_"
        "radio_149mhz_lcp_plus_rcp_source_trajectory.png"
    )
    assert height_plot.exists()
    assert trajectory_plot.exists()
    assert not (output_dir / "event_gaussian_newkirk_height_comparison.png").exists()
    assert not (output_dir / "gaussian_center_trajectory.png").exists()
    valid_centers = pd.read_csv(output_dir / "radio_gaussian_valid_centers.csv")
    height_rows = pd.read_csv(output_dir / "gaussian_newkirk_height_rows.csv")

    assert len(valid_centers) == 2
    assert set(valid_centers["quality_flag"]) == {"ok"}
    assert height_rows["gaussian_projected_height_valid"].isin([False]).any()
    assert result["input_csv"] == str(gaussian_csv)
    assert result["summary"]["gaussian_rows"] == 3
    assert result["summary"]["valid_trajectory_rows"] == 2
    assert result["summary"]["projected_height_valid_count"] == 1
    assert result["summary"]["projected_height_invalid_count"] == 2


def test_resolves_default_gaussian_csv_from_config_without_requiring_local_data():
    resolved = resolve_gaussian_csv(
        gaussian_csv=None,
        config_name=EVENT_CONFIG,
    )

    assert (
        resolved
        == Path("outputs/radio/2000-01-01")
        / "gaussian_spectrogram_overlay"
        / "radio_gaussian_fit_diagnostics.csv"
    )
