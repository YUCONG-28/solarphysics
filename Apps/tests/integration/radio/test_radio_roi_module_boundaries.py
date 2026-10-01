"""ROI compatibility ownership and real synthetic FITS workflow regression."""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.roi_lightcurve import roi_lightcurve_app as app
from solar_apps.frontends.radio.roi_lightcurve import _roi_lightcurve_app_core as core
from solar_toolkit.radio import roi_lightcurve as scientific


@pytest.mark.parametrize(
    "module_name",
    [
        "_roi_app_state",
        "_roi_app_files",
        "_roi_app_regions",
        "_roi_app_analysis",
        "_roi_app_display",
        "_roi_app_page",
        "_roi_lightcurve_app_core",
    ],
)
def test_private_module_can_load_before_public_facade(module_name, tmp_path) -> None:
    package = "solar_apps.frontends.radio.roi_lightcurve"
    code = (
        "import importlib; "
        f"module = importlib.import_module('{package}.{module_name}'); "
        f"app = importlib.import_module('{package}.roi_lightcurve_app'); "
        "assert app.build_parser().parse_args([]).metric is None"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_facade_retains_responsibility_owners_and_single_reference_cache() -> None:
    owners = {
        "state": (
            "DEFAULT_APP_SETTINGS",
            "_RoiImportChoice",
            "_ReferencePlan",
            "build_parser",
        ),
        "files": (
            "build_file_manifest",
            "_plan_reference_grid",
            "_cached_first_radio_image",
        ),
        "regions": (
            "selection_to_radio_roi",
            "_parse_roi_import_document",
            "_stage_imported_roi",
        ),
        "analysis": (
            "_analysis_signature",
            "_export_signature",
            "_cached_lightcurve_png",
        ),
        "display": (
            "build_reference_figure",
            "_write_prepared_artifacts",
            "_zip_artifacts",
        ),
        "page": ("main", "_run_streamlit_app", "_render_roi_step"),
    }
    for responsibility, names in owners.items():
        module = importlib.import_module(
            f"solar_apps.frontends.radio.roi_lightcurve._roi_app_{responsibility}"
        )
        for name in names:
            assert getattr(app, name) is getattr(core, name) is getattr(module, name)
    assert app.extract_radio_roi_lightcurve is scientific.extract_radio_roi_lightcurve
    assert app.RadioRoi is scientific.RadioRoi
    assert set(app.__all__) == {
        "DEFAULT_APP_SETTINGS",
        "build_file_manifest",
        "build_parser",
        "build_reference_figure",
        "default_settings_path",
        "discover_frequency_options",
        "load_app_settings",
        "parse_row_selection_expression",
        "main",
        "resolve_app_settings",
        "save_app_settings",
        "selection_to_radio_roi",
    }


def _write_frames(folder):
    for frequency, factor in [(149.0, 1.0), (164.0, 2.0)]:
        for frame in range(2):
            image = np.arange(16, dtype=float).reshape(4, 4) * factor + frame * 10.0
            header = fits.Header(
                {
                    "CTYPE1": "HPLN-TAN",
                    "CTYPE2": "HPLT-TAN",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CRPIX1": 1.0,
                    "CRPIX2": 1.0,
                    "CRVAL1": 0.0,
                    "CRVAL2": 0.0,
                    "CDELT1": 1.0,
                    "CDELT2": 1.0,
                    "FREQ": frequency,
                    "FREQUNIT": "MHz",
                    "BUNIT": "Jy/beam",
                    "DATE-OBS": f"2025-01-24T04:48:{45 + frame}",
                }
            )
            name = f"{frequency:g}MHz_20250124T0448{45 + frame}_LCP.fits"
            fits.writeto(folder / name, image, header)


def test_synthetic_manifest_reference_extraction_cache_and_export(
    tmp_path, monkeypatch
) -> None:
    _write_frames(tmp_path)
    files = importlib.import_module(
        "solar_apps.frontends.radio.roi_lightcurve._roi_app_files"
    )
    with monkeypatch.context() as guard:
        guard.setattr(
            files,
            "iter_radio_images",
            lambda *_a, **_kw: (_ for _ in ()).throw(
                AssertionError("manifest planning decoded a FITS image")
            ),
        )
        manifest = app.build_file_manifest(tmp_path)
        primary_row = int(
            manifest.loc[manifest["inferred_freq_mhz"].eq(149.0), "row"].iloc[0]
        )
        plans = app._plan_reference_grid(
            manifest,
            primary_frequency=149.0,
            anchor_number=primary_row,
            preview_polarization="LCP",
            pair_tolerance_sec=0.5,
        )
    assert len(manifest) == 4
    assert [plan.freq_mhz for plan in plans] == [149.0, 164.0]
    app._cached_first_radio_image.cache_clear()
    try:
        references, metadata = app._materialize_reference_grid(plans)
        again, _ = app._materialize_reference_grid(plans)
        assert app._cached_first_radio_image.cache_info().misses == 2
        assert app._cached_first_radio_image.cache_info().hits == 2
        assert references[0].image is again[0].image
        assert not references[0].image.flags.writeable
        assert [item["freq_mhz"] for item in metadata] == [149.0, 164.0]
        np.testing.assert_array_equal(references[0].image, np.arange(16).reshape(4, 4))

        roi = scientific.RadioRoi.from_box(1.0, 1.0, 2.0, 2.0, label="synthetic")
        selected_paths = manifest["path"].tolist()
        curve = app.extract_radio_roi_lightcurve(
            tmp_path, files=selected_paths, roi=roi, polarization="LCP"
        )
        assert curve["roi_pixel_count"].tolist() == [4] * 4
        np.testing.assert_array_equal(
            curve.sort_values(["freq_mhz", "obs_time"])["raw_sum"],
            [30.0, 70.0, 60.0, 100.0],
        )
        settings = {
            **app.DEFAULT_APP_SETTINGS,
            "radio_dir": str(tmp_path),
            "polarization": "LCP",
        }
        identities = app._selected_file_identities(selected_paths)
        signature = app._analysis_signature(
            selected_paths, roi, settings, file_identities=identities
        )
        assert (
            app._analysis_signature(
                selected_paths,
                roi,
                {**settings, "metric": "raw_mean"},
                file_identities=identities,
            )
            == signature
        )
        changed = [dict(item) for item in identities]
        changed[0]["mtime_ns"] += 1
        assert (
            app._analysis_signature(
                selected_paths, roi, settings, file_identities=changed
            )
            != signature
        )
        assert (
            app._analysis_signature(
                selected_paths,
                scientific.RadioRoi.from_box(0, 0, 1, 1),
                settings,
                file_identities=identities,
            )
            != signature
        )

        st = SimpleNamespace(
            session_state={
                "analysis_result_signature": app._dataframe_content_signature(curve),
                "reference_metadata": metadata,
                "loaded_manifest": manifest,
                "selected_paths": selected_paths,
            }
        )
        kwargs = {
            "analysis_result_signature": st.session_state["analysis_result_signature"],
            "metric": "raw_sum",
        }
        png = app._cached_lightcurve_png(st, curve, roi, **kwargs)
        assert app._cached_lightcurve_png(st, curve, roi, **kwargs) is png
        assert len(st.session_state["lightcurve_png_cache"]) == 1
        app._cached_lightcurve_png(st, curve, roi, **{**kwargs, "metric": "raw_mean"})
        assert len(st.session_state["lightcurve_png_cache"]) == 2

        artifacts = app._build_cached_export_artifacts(
            st,
            curve,
            roi,
            selected_paths=selected_paths,
            references=references,
            settings=settings,
            display_config={},
            product_keys=("csv", "coordinates_csv", "json", "lightcurve_png"),
        )
        exported = pd.read_csv(io.BytesIO(artifacts["csv"]))
        np.testing.assert_allclose(exported["raw_sum"], curve["raw_sum"])
        assert json.loads(artifacts["json"])["roi"]["label"] == "synthetic"
        assert artifacts["lightcurve_png"] == png
        outputs = app._write_prepared_artifacts(
            artifacts,
            tmp_path / "exports",
            filenames=st.session_state["export_artifact_filenames"],
        )
        assert {key: outputs[key].read_bytes() for key in artifacts} == artifacts
    finally:
        app._cached_first_radio_image.cache_clear()
