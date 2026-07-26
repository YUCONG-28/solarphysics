"""Interactive ROI contracts for the AIA radio composite frontend."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.aia_radio_composite.models import CompositeRequest
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    apply_radio_roi_to_request,
    build_top_panel_selection_figure,
    radio_roi_from_selection,
    radio_roi_json,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering.composite_renderer import (
    TopPanelArtifact,
)
from solar_toolkit.radio.centers import POL_RCP, RadioImage
from solar_toolkit.radio.roi_lightcurve import RadioRoi


def _request(tmp_path: Path) -> CompositeRequest:
    return CompositeRequest(
        aia_directory=tmp_path / "aia",
        aia_wave=171,
        aia_time=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        radio_directory=tmp_path / "radio",
        radio_frequency=149.0,
        polarization="RR",
        roi_type="box",
        roi_vertices_arcsec=(
            (-2.0, -2.0),
            (2.0, -2.0),
            (2.0, 2.0),
            (-2.0, 2.0),
        ),
        spectrum_type="dart",
        spectrum_path=tmp_path / "dart",
    )


def _artifact() -> TopPanelArtifact:
    from PIL import Image
    import io

    image = Image.new("RGB", (200, 100), color=(32, 32, 32))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return TopPanelArtifact(
        image_png=output.getvalue(),
        metadata={
            "schema_version": 1,
            "coordinate_system": "HPLN/HPLT arcsec",
            "image": {"width": 200, "height": 100, "sha256": "synthetic"},
            "panels": [
                {
                    "id": "aia-radio-top",
                    "bbox_normalized": [0.1, 0.1, 0.9, 0.9],
                    "xlim_arcsec": [-100.0, 100.0],
                    "ylim_arcsec": [-50.0, 50.0],
                }
            ],
        },
    )


def _radio_artifact() -> TopPanelArtifact:
    header = fits.Header()
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 3.0
    header["CRPIX2"] = 3.0
    header["CDELT1"] = 10.0
    header["CDELT2"] = 10.0
    frame = RadioImage(
        path=Path("/data/radio_149_RR.fits"),
        hdu_index=0,
        image=np.arange(25, dtype=float).reshape(5, 5),
        header=header,
        pol=POL_RCP,
        freq_mhz=149.0,
        obs_time=datetime(2025, 1, 24, 4, 48, 32),
    )
    return TopPanelArtifact(
        image_png=b"unused",
        metadata={
            "render": {
                "display_extent_arcsec": [-80.0, 80.0, -70.0, 70.0],
            }
        },
        radio_frame=frame,
    )


def test_top_panel_uses_existing_source_map_selection_figure() -> None:
    """The top PNG is exposed through the established Source Map selector."""

    figure = build_top_panel_selection_figure(_artifact(), roi_mode="box")

    assert figure.layout.dragmode == "select"
    assert figure.layout.xaxis.range == (-100.0, 100.0)
    assert figure.layout.yaxis.range == (-50.0, 50.0)
    assert figure.layout.xaxis.constrain == "domain"
    assert figure.layout.yaxis.constrain == "domain"
    assert figure.layout.yaxis.scaleanchor == "x"
    assert len(figure.layout.images) == 1


def test_radio_frame_uses_existing_radio_reference_roi_selector() -> None:
    """A current artifact selects the raw radio source, never the AIA PNG."""

    artifact = _radio_artifact()
    figure = build_top_panel_selection_figure(
        artifact,
        roi_mode="box",
    )

    assert figure.layout.dragmode == "select"
    assert figure.layout.xaxis.range == (-80.0, 80.0)
    assert figure.layout.yaxis.range == (-70.0, 70.0)
    assert figure.layout.meta["roi_coordinate_source"] == "radio_source_frame"
    assert artifact.radio_frame is not None
    assert figure.layout.meta["radio_path"] == str(artifact.radio_frame.path)
    assert figure.layout.meta["display_percentiles"] == [90.0, 99.0]
    assert figure.layout.images == ()
    assert figure.data[0].type == "heatmap"
    assert np.array_equal(np.asarray(figure.data[0].z), np.arange(25).reshape(5, 5))
    assert figure.data[1].name == "Selection grid"


def test_radio_reference_intensity_percentiles_are_user_controlled() -> None:
    figure = build_top_panel_selection_figure(
        _radio_artifact(),
        roi_mode="box",
        low_percentile=80.0,
        high_percentile=95.0,
    )

    assert figure.layout.meta["display_percentiles"] == [80.0, 95.0]
    assert figure.data[0].zmin == pytest.approx(np.percentile(np.arange(25), 80.0))
    assert figure.data[0].zmax == pytest.approx(np.percentile(np.arange(25), 95.0))


def test_confirmed_roi_is_drawn_on_lasso_selection_figure() -> None:
    """An existing arcsec ROI is projected into the reusable Plotly figure."""

    roi = RadioRoi.from_polygon(
        [(-20.0, -10.0), (10.0, -15.0), (25.0, 20.0)],
        label="Burst",
    )
    figure = build_top_panel_selection_figure(
        _artifact(),
        roi=roi,
        roi_mode="lasso",
    )

    assert figure.layout.dragmode == "lasso"
    assert figure.data[-1].name == "Burst"
    assert list(figure.data[-1].x) == [-20.0, 10.0, 25.0, -20.0]


def test_box_selection_becomes_arcsec_radio_roi() -> None:
    """A Plotly box event delegates to ``selection_to_radio_roi``."""

    roi = radio_roi_from_selection(
        {
            "selection": {
                "box": [
                    {
                        "x": [-30.0, 20.0],
                        "y": [-10.0, 40.0],
                    }
                ]
            }
        },
        roi_mode="box",
        label="Core",
    )

    assert roi is not None
    assert roi.kind == "box"
    assert roi.label == "Core"
    assert roi.bounds_arcsec == {
        "left": -30.0,
        "bottom": -10.0,
        "right": 20.0,
        "top": 40.0,
    }


def test_lasso_selection_preserves_arcsec_vertices() -> None:
    """A Plotly lasso event retains its HPLN/HPLT vertex sequence."""

    roi = radio_roi_from_selection(
        {
            "selection": {
                "lasso": {
                    "x": [-20.0, 0.0, 25.0, 5.0],
                    "y": [-10.0, -25.0, 5.0, 30.0],
                }
            }
        },
        roi_mode="lasso",
        label="Lasso",
    )

    assert roi is not None
    assert roi.kind == "polygon"
    assert roi.vertices_arcsec == (
        (-20.0, -10.0),
        (0.0, -25.0),
        (25.0, 5.0),
        (5.0, 30.0),
    )


def test_empty_selection_returns_none() -> None:
    """Clearing a Plotly selection does not fabricate an ROI."""

    assert radio_roi_from_selection(None, roi_mode="box") is None
    assert radio_roi_from_selection({}, roi_mode="lasso") is None


def test_confirmed_roi_updates_request_without_pixel_coordinates(
    tmp_path: Path,
) -> None:
    """Confirmed ROI persistence uses only HPLN/HPLT arcsec."""

    original = _request(tmp_path)
    roi = RadioRoi.from_polygon(
        [(-20.0, -10.0), (10.0, -15.0), (25.0, 20.0)],
        label="Burst",
    )

    updated = apply_radio_roi_to_request(original, roi)
    payload = updated.to_dict()
    roi_payload = radio_roi_json(roi)

    assert updated is not original
    assert original.roi_type == "box"
    assert updated.roi_type == "lasso"
    assert updated.roi_vertices_arcsec == roi.vertices_arcsec
    assert roi_payload["coordinate_system"] == "HPLN/HPLT arcsec"
    assert "pixel" not in json.dumps(payload).lower()
    assert "pixel" not in json.dumps(roi_payload).lower()


@pytest.mark.parametrize("mode", ["circle", "", "polygon"])
def test_roi_mode_fails_closed(mode: str) -> None:
    """Only the requested box and lasso interaction modes are accepted."""

    with pytest.raises(ValueError, match="box or lasso"):
        build_top_panel_selection_figure(_artifact(), roi_mode=mode)
