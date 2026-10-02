"""Presentation and annotation exports for the radio ROI Streamlit app."""

from __future__ import annotations

import io
import json

import matplotlib.figure
import numpy as np
import pandas as pd

from solar_apps.frontends.radio.roi_lightcurve import roi_lightcurve_app as app
from solar_apps.frontends.radio.roi_lightcurve import (
    roi_lightcurve_application as presentation,
)
from solar_toolkit.radio.roi_lightcurve import RadioRoi


def _curve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": pd.date_range(
                "2000-01-01T04:48:45", periods=4, freq="s", tz="UTC"
            ).astype(str),
            "freq_mhz": [149.0] * 4,
            "polarization": ["L+R"] * 4,
            "quality_flag": ["ok"] * 4,
            "raw_sum": [-5.0, 1.0, 10.0, 100.0],
            "bunit": ["Jy/beam"] * 4,
        }
    )


def _roi() -> RadioRoi:
    return RadioRoi.from_box(-120.0, -80.0, 90.0, 140.0, label="source-a")


def test_analysis_export_fragment_routes_path_policy_to_export_only(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    analysis = _curve_df()
    path_policy = object()

    def fake_analysis(
        _st,
        selected_paths,
        references,
        roi,
        settings,
        display_config,
    ) -> pd.DataFrame:
        observed["analysis_args"] = (
            selected_paths,
            references,
            roi,
            settings,
            display_config,
        )
        return analysis

    def fake_export(
        _st,
        df,
        selected_paths,
        references,
        roi,
        settings,
        display_config,
        received_path_policy,
    ) -> None:
        observed["export_args"] = (
            df,
            selected_paths,
            references,
            roi,
            settings,
            display_config,
            received_path_policy,
        )

    monkeypatch.setattr(app, "_render_analysis_step", fake_analysis)
    monkeypatch.setattr(app, "_render_export_step", fake_export)

    selected_paths = ["frame.fits"]
    references = [object()]
    roi = _roi()
    settings = {"metric": "raw_sum"}
    display_config = {"mode": "linear"}
    app._render_analysis_and_export_steps.__wrapped__(
        selected_paths,
        references,
        roi,
        settings,
        display_config,
        path_policy,
    )

    assert observed["analysis_args"] == (
        selected_paths,
        references,
        roi,
        settings,
        display_config,
    )
    assert observed["export_args"] == (
        analysis,
        selected_paths,
        references,
        roi,
        settings,
        display_config,
        path_policy,
    )


def test_coordinate_export_is_closed_and_records_hpc_arcsec() -> None:
    artifacts = presentation.build_radio_roi_artifacts(
        _curve_df(),
        _roi(),
        metric="raw_sum",
        lightcurve_plot_style="line",
        lightcurve_y_transform="log10",
        selected_products=("coordinates_csv", "json", "lightcurve_png"),
    )

    coordinates = pd.read_csv(io.BytesIO(artifacts["coordinates_csv"]))
    assert list(coordinates["vertex_order"]) == [0, 1, 2, 3, 4]
    assert coordinates.iloc[0][["hpln_arcsec", "hplt_arcsec"]].tolist() == (
        coordinates.iloc[-1][["hpln_arcsec", "hplt_arcsec"]].tolist()
    )
    assert coordinates["is_closure_vertex"].tolist() == [False] * 4 + [True]
    assert set(coordinates["coordinate_system"]) == {"HPLN/HPLT arcsec"}
    assert coordinates.iloc[0][
        ["left_arcsec", "bottom_arcsec", "right_arcsec", "top_arcsec"]
    ].tolist() == [-120.0, -80.0, 90.0, 140.0]

    metadata = json.loads(artifacts["json"].decode("utf-8"))
    plot = metadata["settings"]["lightcurve_plot"]
    assert plot["style"] == "line"
    assert plot["y_transform"] == "log10"
    assert plot["metric_unit"] == "Jy/beam * pixel"
    assert metadata["outputs"]["coordinates_csv"] == "radio_roi_coordinates.csv"
    assert artifacts["lightcurve_png"].startswith(b"\x89PNG")


def test_line_plot_uses_log10_values_and_labels_units(tmp_path, monkeypatch) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture_figure(figure, *_args, **_kwargs) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)

    presentation._plot_radio_roi_lightcurve(
        _curve_df(),
        tmp_path / "curve.png",
        metric="raw_sum",
        plot_style="line",
        y_transform="log10",
    )

    axis = captured["figure"].axes[0]
    assert len(axis.lines) == 1
    assert not axis.collections
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.0, 1.0, 2.0])
    assert axis.get_ylabel() == "log10(raw_sum / 1 [Jy/beam * pixel])"


def test_scatter_plot_keeps_independent_points(tmp_path, monkeypatch) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture_figure(figure, *_args, **_kwargs) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)

    presentation._plot_radio_roi_lightcurve(
        _curve_df(),
        tmp_path / "curve.png",
        metric="raw_sum",
        plot_style="scatter",
        y_transform="linear",
    )

    axis = captured["figure"].axes[0]
    assert not axis.lines
    assert len(axis.collections) == 1
    np.testing.assert_allclose(
        axis.collections[0].get_offsets()[:, 1],
        [-5.0, 1.0, 10.0, 100.0],
    )


def test_app_log_frame_and_cache_signature_track_plot_options() -> None:
    transformed = app._lightcurve_metric_frame(
        _curve_df(),
        "raw_sum",
        y_transform="log10",
    )
    assert transformed["raw_sum"].tolist() == [0.0, 1.0, 2.0]

    kwargs = {
        "analysis_result_signature": "result-a",
        "product_keys": ("coordinates_csv", "lightcurve_png"),
        "metric": "raw_sum",
        "reference_identities": [],
        "display_config": {},
    }
    scatter_log = app._export_signature(
        **kwargs,
        lightcurve_plot_style="scatter",
        lightcurve_y_transform="log10",
    )
    assert (
        app._export_signature(
            **kwargs,
            lightcurve_plot_style="line",
            lightcurve_y_transform="log10",
        )
        != scatter_log
    )
    assert (
        app._export_signature(
            **kwargs,
            lightcurve_plot_style="scatter",
            lightcurve_y_transform="linear",
        )
        != scatter_log
    )
