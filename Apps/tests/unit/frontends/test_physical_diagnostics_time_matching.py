"""The CLI must pass its drift input to height matching, preserving milliseconds."""

import pandas as pd


def test_cli_uses_supplied_drift_table(tmp_path, monkeypatch):
    from solar_apps.workflows.radio.physical_diagnostics_cli import (
        run_physical_diagnostics,
    )
    from solar_apps.workflows.radio import quicklook
    from solar_toolkit.radio import config

    monkeypatch.setattr(config, "load_radio_user_config", lambda name: ({}, {}))
    monkeypatch.setattr(
        config,
        "load_radio_diagnostic_presentation_config",
        lambda name: {
            "enable_static_summary": False,
            "enable_html_dashboard": False,
            "comparison_frequency_mhz": [150.0],
        },
    )
    monkeypatch.setattr(
        quicklook, "build_quicklook_config", lambda name: {"solar_radius_arcsec": 960.0}
    )
    gaussian = tmp_path / "gaussian.csv"
    drift = tmp_path / "drift.csv"
    pd.DataFrame(
        [
            dict(
                time="20250124044831 11",
                freq=150.0,
                center_x_arcsec=1200.0,
                center_y_arcsec=-300.0,
                quality_flag="ok",
                trajectory_valid=True,
            )
        ]
    ).to_csv(gaussian, index=False)
    pd.DataFrame(
        [
            dict(
                label="known_burst",
                t_start="2025-01-24T04:48:30.011",
                t_end="2025-01-24T04:48:32.011",
                f_start_mhz=160.0,
                f_end_mhz=140.0,
                drift_rate_mhz_s=-10.0,
            )
        ]
    ).to_csv(drift, index=False)
    result = run_physical_diagnostics(
        gaussian_csv=gaussian, drift_csv=drift, output_dir=tmp_path / "out"
    )
    rows = pd.read_csv(result["artifacts"]["height_rows_csv"], dtype={"time": str})
    assert rows["time"].eq("20250124044831 11").all()
    assert rows.drift_label.eq("known_burst").all()
