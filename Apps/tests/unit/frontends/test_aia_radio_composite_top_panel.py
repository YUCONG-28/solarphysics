"""AIA selection, radio Gaussian, and top-panel rendering contracts."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from PIL import Image

from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    AiaSelection,
    RadioGaussianSelection,
    fit_radio_gaussian_selection,
    load_aia_selection,
    select_radio_frame,
)
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    aia_adapter,
    radio_adapter,
)
from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite import app as composite_app
from solar_apps.frontends.radio.aia_radio_composite.models import CompositeRequest
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    TopPanelArtifact,
    render_top_panel,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    composite_renderer,
)
from solar_toolkit.aia.background import AiaBackground
from solar_toolkit.radio.centers import POL_LCP, POL_RCP, POL_SUM, RadioImage
from solar_toolkit.radio.gaussian import GaussianFitResult


def _request(tmp_path: Path, **overrides: object) -> CompositeRequest:
    values: dict[str, object] = {
        "aia_directory": tmp_path / "aia",
        "aia_wave": 171,
        "aia_time": datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        "radio_directory": tmp_path / "radio",
        "radio_frequency": 149.0,
        "polarization": "RR",
        "roi_type": "box",
        "roi_vertices_arcsec": (
            (-20.0, -20.0),
            (20.0, -20.0),
            (20.0, 20.0),
            (-20.0, 20.0),
        ),
        "spectrum_type": "dart",
        "spectrum_path": tmp_path / "dart",
    }
    values.update(overrides)
    return CompositeRequest(**values)


def _background(
    path: str = "/data/aia171.fits",
    *,
    wavelength: int = 171,
) -> AiaBackground:
    return AiaBackground(
        path=path,
        z=np.arange(100, dtype=float).reshape(10, 10),
        x_arcsec=np.linspace(-90.0, 90.0, 10),
        y_arcsec=np.linspace(-90.0, 90.0, 10),
        label=f"AIA {wavelength}",
        obs_time=pd.Timestamp("2025-01-24T04:48:31"),
        wavelength=str(wavelength),
    )


def _radio_header() -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 5.5
    header["CRPIX2"] = 5.5
    header["CDELT1"] = 10.0
    header["CDELT2"] = 10.0
    return header


def _radio_image(
    path: Path,
    *,
    polarization: str = POL_RCP,
    frequency_mhz: float = 149.0,
    observed: datetime | None = None,
    value: float = 1.0,
) -> RadioImage:
    return RadioImage(
        path=path,
        hdu_index=0,
        image=np.full((10, 10), value, dtype=float),
        header=_radio_header(),
        pol=polarization,
        freq_mhz=frequency_mhz,
        obs_time=observed or datetime(2025, 1, 24, 4, 48, 32),
    )


def _fit(*, sigma: float = 1.0, quality_flag: str = "ok") -> GaussianFitResult:
    y, x = np.mgrid[0:10, 0:10]
    model = np.exp(-((x - 4.5) ** 2 + (y - 4.5) ** 2) / (2.0 * sigma**2))
    return GaussianFitResult(
        model=model,
        gaussian_only_model=model,
        center_pixel=(4.5, 4.5),
        center_arcsec=(0.0, 0.0),
        sigma_pixel=(sigma, sigma),
        theta_rad=0.0,
        amplitude=10.0,
        background_level=0.0,
        noise_sigma=0.5,
        snr=20.0,
        residual_rms=0.1,
        quality_flag=quality_flag,
        covariance=None,
        mask_pixel_count=30,
        source_file="/data/radio.fits",
    )


def _aia_selection(*, wavelength: int = 171) -> AiaSelection:
    return AiaSelection(
        background=_background(
            f"/data/aia{wavelength}.fits",
            wavelength=wavelength,
        ),
        requested_time_utc=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        matched_time_utc=datetime(2025, 1, 24, 4, 48, 31, tzinfo=UTC),
        delta_seconds=1.0,
        candidate_count=3,
    )


def _radio_selection(
    fit: GaussianFitResult | None,
    *,
    frequency_mhz: float = 149.0,
) -> RadioGaussianSelection:
    path = (
        Path("/data/radio.fits")
        if frequency_mhz == 149.0
        else Path(f"/data/radio-{frequency_mhz:g}.fits")
    )
    return RadioGaussianSelection(
        frame=_radio_image(
            path,
            frequency_mhz=frequency_mhz,
        ),
        fit_result=fit,
        extent_arcsec=(-50.0, 50.0, -50.0, 50.0),
        image_origin="lower",
        gaussian_config={
            "gaussian_overlay_display_mode": "contours_and_fwhm",
            "draw_gaussian_contours": True,
            "draw_gaussian_center": True,
            "draw_gaussian_fwhm_ellipse": True,
            "gaussian_hide_all_when_fit_invalid": True,
            "draw_low_quality_gaussian_contours": False,
            "gaussian_quality_requirements": {
                "require_quality_ok": True,
                "max_fwhm_arcsec": 100.0,
                "max_center_peak_distance_arcsec": 50.0,
                "min_snr": 5.0,
                "max_residual_rms_fraction": 0.8,
            },
            "fit_snr_threshold": 5.0,
        },
        requested_time_utc=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        matched_time_utc=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        delta_seconds=0.0,
        candidate_count=1,
        failure_diagnostics={},
    )


def test_aia_adapter_filters_wavelength_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the requested AIA wavelength reaches the existing reader."""

    table = pd.DataFrame(
        {
            "path": ["/data/aia94.fits", "/data/aia171.fits", "/data/aia193.fits"],
            "obs_time": pd.to_datetime(
                [
                    "2025-01-24T04:48:32",
                    "2025-01-24T04:48:31",
                    "2025-01-24T04:48:32",
                ]
            ),
            "wavelength": ["94", "171.0", "193"],
        }
    )
    read_paths: list[str] = []
    monkeypatch.setattr(aia_adapter, "scan_aia_folder", lambda *_args, **_kwargs: table)

    def fake_read(path: str, **_kwargs: object) -> AiaBackground:
        read_paths.append(path)
        return _background(path)

    monkeypatch.setattr(aia_adapter, "read_aia_background", fake_read)

    selection = load_aia_selection(_request(tmp_path), max_dt_seconds=5.0)

    assert read_paths == ["/data/aia171.fits"]
    assert selection.candidate_count == 1
    assert selection.delta_seconds == pytest.approx(1.0)
    assert selection.matched_time_utc.tzinfo is UTC


def test_aia_adapter_fails_closed_when_wave_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different available wavelength is never used as a fallback."""

    monkeypatch.setattr(
        aia_adapter,
        "scan_aia_folder",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "path": ["/data/aia193.fits"],
                "obs_time": pd.to_datetime(["2025-01-24T04:48:32"]),
                "wavelength": ["193"],
            }
        ),
    )

    with pytest.raises(FileNotFoundError, match="AIA 171"):
        load_aia_selection(_request(tmp_path))


def test_radio_adapter_uses_nearest_existing_radio_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Radio selection delegates file reading and chooses the nearest UTC frame."""

    early = _radio_image(
        tmp_path / "early.fits",
        observed=datetime(2025, 1, 24, 4, 48, 25),
    )
    near = _radio_image(
        tmp_path / "near.fits",
        observed=datetime(2025, 1, 24, 4, 48, 31, 900000),
    )
    images = {early.path: early, near.path: near}
    monkeypatch.setattr(
        radio_adapter,
        "select_radio_files",
        lambda *_args, **_kwargs: list(images),
    )
    monkeypatch.setattr(
        radio_adapter,
        "iter_radio_images",
        lambda path, **_kwargs: iter([images[Path(path)]]),
    )

    selected, count, delta = select_radio_frame(
        _request(tmp_path),
        max_dt_seconds=10.0,
    )

    assert selected.path == near.path
    assert count == 2
    assert delta == pytest.approx(0.1)


def test_radio_candidate_index_loads_only_the_selected_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [
        tmp_path / f"149MHz_2025124_0448{second:02d}_000.fits"
        for second in (30, 32, 34)
    ]
    images = {
        path: _radio_image(
            path,
            observed=datetime(2025, 1, 24, 4, 48, second),
        )
        for path, second in zip(paths, (30, 32, 34), strict=True)
    }
    reads: list[Path] = []
    monkeypatch.setattr(
        radio_adapter,
        "select_radio_files",
        lambda *_args, **_kwargs: paths,
    )

    def fake_iter(path: str | Path, **_kwargs: object):
        reads.append(Path(path))
        return iter([images[Path(path)]])

    monkeypatch.setattr(radio_adapter, "iter_radio_images", fake_iter)

    selected, count, delta = select_radio_frame(_request(tmp_path))

    assert selected.path == paths[1]
    assert count == 3
    assert delta == 0.0
    assert reads == [paths[1]]


def test_radio_adapter_reuses_existing_rr_ll_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RR+LL selection is synthesized by ``maybe_make_sum_images``."""

    observed = datetime(2025, 1, 24, 4, 48, 32)
    left = _radio_image(
        tmp_path / "LL.fits",
        polarization=POL_LCP,
        observed=observed,
        value=2.0,
    )
    right = _radio_image(
        tmp_path / "RR.fits",
        polarization=POL_RCP,
        observed=observed,
        value=3.0,
    )
    images = {left.path: left, right.path: right}
    monkeypatch.setattr(
        radio_adapter,
        "select_radio_files",
        lambda *_args, **_kwargs: list(images),
    )
    monkeypatch.setattr(
        radio_adapter,
        "iter_radio_images",
        lambda path, **_kwargs: iter([images[Path(path)]]),
    )

    selected, count, delta = select_radio_frame(
        _request(tmp_path, polarization="RR+LL"),
    )

    assert selected.pol == POL_SUM
    np.testing.assert_allclose(selected.image, 5.0)
    assert count == 1
    assert delta == 0.0


def test_radio_gaussian_adapter_calls_canonical_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter forwards the selected image to the canonical fitter."""

    frame = _radio_image(tmp_path / "radio.fits")
    monkeypatch.setattr(
        radio_adapter,
        "select_radio_frame",
        lambda *_args, **_kwargs: (frame, 1, 0.0),
    )
    captured: dict[str, object] = {}

    def fake_fit(
        data: np.ndarray,
        extent: tuple[float, float, float, float],
        config: dict[str, object],
        **kwargs: object,
    ) -> GaussianFitResult:
        captured.update(
            {
                "data": data,
                "extent": extent,
                "config": config,
                "kwargs": kwargs,
            }
        )
        return _fit()

    monkeypatch.setattr(
        radio_adapter,
        "fit_elliptical_gaussian_on_radio_image",
        fake_fit,
    )

    selection = fit_radio_gaussian_selection(_request(tmp_path))

    assert captured["data"] is frame.image
    assert captured["extent"] == (-50.0, 50.0, -50.0, 50.0)
    assert captured["kwargs"]["image_origin"] == "lower"
    assert selection.gaussian_config["gaussian_overlay_display_mode"] == (
        "contours_and_fwhm"
    )
    assert selection.fit_result is not None


def test_gaussian_user_visibility_overrides_are_not_reset() -> None:
    config = radio_adapter._gaussian_config(
        149.0,
        overrides={
            "draw_gaussian_center": False,
            "draw_gaussian_contours": False,
        },
    )

    assert config["draw_gaussian_center"] is False
    assert config["draw_gaussian_contours"] is False


def test_top_panel_renders_valid_gaussian_center_contour_and_fwhm() -> None:
    """A valid existing Gaussian fit produces a PNG and quality metadata."""

    artifact = render_top_panel(_aia_selection(), _radio_selection(_fit()), dpi=80)

    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(artifact.image_png)) as image:
        assert image.width > 0
        assert image.height > 0
    fit_metadata = artifact.metadata["radio"]["fit"]
    assert fit_metadata["quality_flag"] == "ok"
    assert fit_metadata["overlay_valid"] is True
    assert fit_metadata["fwhm_major_arcsec"] == pytest.approx(23.55)
    assert artifact.metadata["coordinate_system"] == "HPLN/HPLT arcsec"
    assert artifact.metadata["image"]["width"] > 0
    assert artifact.metadata["image"]["height"] > 0
    assert len(artifact.metadata["image"]["sha256"]) == 64
    assert artifact.metadata["panels"][0]["id"] == "aia-radio-top"
    assert len(artifact.metadata["panels"][0]["bbox_normalized"]) == 4
    assert artifact.radio_frame is not None
    assert artifact.radio_frame.path == Path("/data/radio.fits")
    assert np.array_equal(
        artifact.radio_frame.image,
        _radio_selection(_fit()).frame.image,
    )
    json.dumps(artifact.metadata)


def test_top_panel_renders_multiple_selected_radio_frequencies() -> None:
    selections = (
        _radio_selection(_fit(), frequency_mhz=149.0),
        _radio_selection(_fit(), frequency_mhz=164.0),
        _radio_selection(_fit(), frequency_mhz=190.0),
    )

    artifact = render_top_panel(_aia_selection(), selections, dpi=80)

    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert [item.freq_mhz for item in artifact.radio_frames] == [
        149.0,
        164.0,
        190.0,
    ]
    assert artifact.radio_frame is artifact.radio_frames[0]
    assert artifact.metadata["render"]["radio_overlay_frequency_count"] == 3
    assert [item["frequency_mhz"] for item in artifact.metadata["radios"]] == [
        149.0,
        164.0,
        190.0,
    ]


def test_top_panel_honors_hidden_gaussian_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _radio_selection(_fit())
    config = dict(selection.gaussian_config)
    config["draw_gaussian_center"] = False
    object.__setattr__(selection, "gaussian_config", config)
    received: list[bool] = []

    def fake_overlay(
        _axis: object,
        _fit_result: object,
        _extent: object,
        _shape: object,
        overlay_config: dict[str, object],
    ) -> None:
        received.append(bool(overlay_config["draw_gaussian_center"]))

    monkeypatch.setattr(
        composite_renderer,
        "overlay_gaussian_fit_on_axis",
        fake_overlay,
    )

    artifact = render_top_panel(_aia_selection(), selection, dpi=80)

    assert received == [False]
    assert artifact.metadata["render"]["draw_gaussian_center"] == [False]


def test_top_panel_does_not_draw_contours_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.patches import Ellipse

    selection = _radio_selection(_fit())
    config = dict(selection.gaussian_config)
    config.update(
        composite_app._gaussian_display_overrides(
            show_center=False,
            show_contours=False,
            contour_percent=95.0,
        )
    )
    object.__setattr__(selection, "gaussian_config", config)
    contour_calls: list[object] = []
    ellipse_calls: list[object] = []
    original_contour = Axes.contour
    original_add_patch = Axes.add_patch

    def spy_contour(axis: Axes, *args: object, **kwargs: object):
        contour_calls.append(axis)
        return original_contour(axis, *args, **kwargs)

    def spy_add_patch(axis: Axes, patch: object):
        if isinstance(patch, Ellipse):
            ellipse_calls.append(patch)
        return original_add_patch(axis, patch)

    monkeypatch.setattr(Axes, "contour", spy_contour)
    monkeypatch.setattr(Axes, "add_patch", spy_add_patch)

    artifact = render_top_panel(_aia_selection(), selection, dpi=80)

    assert contour_calls == []
    assert ellipse_calls == []
    assert artifact.metadata["render"]["draw_gaussian_contours"] == [False]
    assert artifact.metadata["render"]["draw_gaussian_fwhm_ellipse"] == [False]


def test_top_panel_draws_only_requested_contour_without_implicit_fwhm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.patches import Ellipse

    selection = _radio_selection(_fit())
    config = dict(selection.gaussian_config)
    config.update(
        composite_app._gaussian_display_overrides(
            show_center=False,
            show_contours=True,
            contour_percent=95.0,
        )
    )
    object.__setattr__(selection, "gaussian_config", config)
    contour_calls: list[object] = []
    ellipse_calls: list[object] = []
    original_contour = Axes.contour
    original_add_patch = Axes.add_patch

    def spy_contour(axis: Axes, *args: object, **kwargs: object):
        contour_calls.append(axis)
        return original_contour(axis, *args, **kwargs)

    def spy_add_patch(axis: Axes, patch: object):
        if isinstance(patch, Ellipse):
            ellipse_calls.append(patch)
        return original_add_patch(axis, patch)

    monkeypatch.setattr(Axes, "contour", spy_contour)
    monkeypatch.setattr(Axes, "add_patch", spy_add_patch)

    artifact = render_top_panel(_aia_selection(), selection, dpi=80)

    assert len(contour_calls) == 1
    assert ellipse_calls == []
    assert artifact.metadata["render"]["draw_gaussian_contours"] == [True]
    assert artifact.metadata["render"]["draw_gaussian_fwhm_ellipse"] == [False]


def test_top_panel_custom_hpc_range_expands_canvas() -> None:
    artifact = render_top_panel(
        _aia_selection(),
        _radio_selection(_fit()),
        dpi=80,
        display_extent_arcsec=(-180.0, 220.0, -140.0, 160.0),
    )

    panel = artifact.metadata["panels"][0]
    render = artifact.metadata["render"]
    assert panel["xlim_arcsec"] == [-180.0, 220.0]
    assert panel["ylim_arcsec"] == [-140.0, 160.0]
    assert render["aia_extent_arcsec"] == [-100.0, 100.0, -100.0, 100.0]
    assert render["display_extent_arcsec"] == [
        -180.0,
        220.0,
        -140.0,
        160.0,
    ]
    assert render["canvas_extended_beyond_observation"] is True
    assert render["radio_gaussian_evaluated_on_display_canvas"] is True


def test_multi_aia_grid_is_seamless_and_only_labels_outer_axes() -> None:
    aias = tuple(
        _aia_selection(wavelength=wavelength)
        for wavelength in (94, 131, 171, 193, 211, 304)
    )

    artifact = render_top_panel(
        aias,
        _radio_selection(_fit()),
        dpi=80,
        display_extent_arcsec=(-180.0, 220.0, -140.0, 160.0),
        extended_canvas_color="white",
    )

    panels = artifact.metadata["panels"]
    render = artifact.metadata["render"]
    assert render["panel_spacing"] == {"wspace": 0.0, "hspace": 0.0}
    assert render["outer_coordinate_labels_only"] is True
    assert render["extended_canvas_color"] == "white"
    assert panels[0]["bbox_normalized"][2] == pytest.approx(
        panels[1]["bbox_normalized"][0]
    )
    assert panels[1]["bbox_normalized"][2] == pytest.approx(
        panels[2]["bbox_normalized"][0]
    )
    assert panels[0]["bbox_normalized"][3] == pytest.approx(
        panels[3]["bbox_normalized"][1]
    )
    assert [panel["show_x_coordinates"] for panel in panels] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert [panel["show_y_coordinates"] for panel in panels] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_top_panel_rejects_unknown_extended_canvas_color() -> None:
    with pytest.raises(
        ValueError,
        match="extended_canvas_color must be black or white",
    ):
        render_top_panel(
            _aia_selection(),
            _radio_selection(_fit()),
            extended_canvas_color="gray",
        )


def test_radio_gaussian_model_is_evaluated_beyond_observation_extent() -> None:
    fit = _fit(sigma=4.0)
    fit.center_pixel = (9.0, 4.5)
    fit.center_arcsec = (45.0, 0.0)

    grid_x, _grid_y, model = composite_renderer._expanded_gaussian_model(
        fit,
        (-50.0, 50.0, -50.0, 50.0),
        (10, 10),
        "lower",
        (-120.0, 120.0, -100.0, 100.0),
        samples_per_axis=240,
    )

    outside_radio_observation = grid_x > 50.0
    assert np.nanmax(model[outside_radio_observation]) > fit.amplitude * 0.5


def test_top_panel_rejects_degenerate_hpc_range() -> None:
    with pytest.raises(ValueError, match="left < right"):
        render_top_panel(
            _aia_selection(),
            _radio_selection(_fit()),
            display_extent_arcsec=(10.0, -10.0, -20.0, 20.0),
        )


def test_top_panel_preserves_gaussian_quality_rejection() -> None:
    """An oversized fit is rejected by the existing overlay quality control."""

    fit = _fit(sigma=20.0)
    artifact = render_top_panel(_aia_selection(), _radio_selection(fit), dpi=80)

    fit_metadata = artifact.metadata["radio"]["fit"]
    assert fit_metadata["quality_flag"] == "unphysical_size"
    assert fit_metadata["quality_flag_detail"] == "skipped_large_fwhm"
    assert fit_metadata["overlay_valid"] is False
    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_top_panel_renders_aia_when_gaussian_fit_is_unavailable() -> None:
    """A fit failure remains visible and does not suppress the AIA context."""

    selection = _radio_selection(None)
    object.__setattr__(
        selection,
        "failure_diagnostics",
        {"quality_flag": "mask_too_small"},
    )

    artifact = render_top_panel(_aia_selection(), selection, dpi=80)

    assert artifact.metadata["radio"]["fit"] == {
        "available": False,
        "quality_flag": "mask_too_small",
        "quality_flag_detail": "mask_too_small",
        "overlay_valid": False,
    }
    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_application_build_top_panel_orchestrates_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The application layer connects adapters to rendering without science."""

    aia = _aia_selection()
    radio = _radio_selection(_fit())
    expected = TopPanelArtifact(image_png=b"png", metadata={"ok": True})
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        application,
        "load_aia_selection",
        lambda request, **kwargs: calls.append(("aia", kwargs)) or aia,
    )
    monkeypatch.setattr(
        application,
        "load_radio_candidates",
        lambda request, **kwargs: (
            calls.append(("radio_candidates", kwargs)) or (radio.frame,)
        ),
    )
    monkeypatch.setattr(
        application,
        "scan_aia_catalog",
        lambda request: pd.DataFrame(),
    )
    monkeypatch.setattr(
        application,
        "fit_radio_gaussian_frame",
        lambda frame, **kwargs: calls.append(("radio_fit", kwargs)) or radio,
    )
    monkeypatch.setattr(
        application,
        "render_top_panel",
        lambda aia_value, radio_value, **kwargs: (
            calls.append(("render", kwargs)) or expected
        ),
    )

    result = application.build_top_panel(
        _request(tmp_path),
        aia_max_dt_seconds=10.0,
        radio_max_dt_seconds=2.0,
        pair_time_tolerance_sec=0.1,
        max_aia_pixels=512,
        display_extent_arcsec=(-200.0, 200.0, -150.0, 150.0),
        dpi=120,
    )

    assert result is expected
    assert [name for name, _ in calls] == [
        "radio_candidates",
        "radio_fit",
        "aia",
        "render",
    ]
    assert calls[0][1] == {"pair_time_tolerance_sec": 0.1}
    assert calls[2][1]["max_dt_seconds"] == 10.0
    assert calls[2][1]["max_pixels"] == 512
    assert calls[3][1] == {
        "dpi": 120,
        "display_extent_arcsec": (-200.0, 200.0, -150.0, 150.0),
        "extended_canvas_color": "black",
    }


def test_application_build_top_panel_fits_every_selected_frequency_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aia = _aia_selection()
    fitted: list[float] = []
    rendered: list[tuple[RadioGaussianSelection, ...]] = []
    expected = TopPanelArtifact(image_png=b"png", metadata={"ok": True})
    monkeypatch.setattr(
        application, "scan_aia_catalog", lambda _request: pd.DataFrame()
    )
    monkeypatch.setattr(
        application, "load_aia_selection", lambda _request, **_kwargs: aia
    )
    monkeypatch.setattr(
        application,
        "load_radio_candidates",
        lambda request, **_kwargs: (
            _radio_selection(
                _fit(),
                frequency_mhz=request.radio_frequency,
            ).frame,
        ),
    )

    def fake_fit(
        frame: RadioImage,
        **_kwargs: object,
    ) -> RadioGaussianSelection:
        fitted.append(frame.freq_mhz)
        return _radio_selection(_fit(), frequency_mhz=frame.freq_mhz)

    def fake_render(
        _aia: tuple[AiaSelection, ...],
        radios: tuple[RadioGaussianSelection, ...],
        **_kwargs: object,
    ) -> TopPanelArtifact:
        rendered.append(radios)
        return expected

    monkeypatch.setattr(application, "fit_radio_gaussian_frame", fake_fit)
    monkeypatch.setattr(application, "render_top_panel", fake_render)

    result = application.build_top_panel(
        _request(tmp_path),
        radio_frequencies_mhz=(149.0, 164.0, 190.0),
    )

    assert result is expected
    assert fitted == [149.0, 164.0, 190.0]
    assert len(rendered) == 1
    assert [item.frame.freq_mhz for item in rendered[0]] == [
        149.0,
        164.0,
        190.0,
    ]


def test_top_panel_renders_selected_aia_waves_in_order_as_grid() -> None:
    artifact = render_top_panel(
        (
            _aia_selection(wavelength=193),
            _aia_selection(wavelength=94),
            _aia_selection(wavelength=171),
            _aia_selection(wavelength=335),
        ),
        _radio_selection(_fit()),
        dpi=60,
    )

    assert [panel["wavelength"] for panel in artifact.metadata["panels"]] == [
        "193",
        "94",
        "171",
        "335",
    ]
    assert artifact.metadata["render"]["grid_rows"] == 2
    assert artifact.metadata["render"]["grid_columns"] == 3
    assert artifact.metadata["reference_radio_frequency_mhz"] == 149.0
    assert artifact.metadata["reference_radio_time_utc"].startswith(
        "2025-01-24T04:48:32"
    )
