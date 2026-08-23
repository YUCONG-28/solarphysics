# SPDX-License-Identifier: GPL-3.0-only
"""Focused tests for the App 1.0 STEREO EUVI plot module and worker."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from astropy.io import fits

matplotlib.use("Agg")

from solar_apps.frontends.app_v1.function_catalog import (
    DEFAULT_FUNCTION_CATALOG,
)  # noqa: E402
from solar_apps.workflows.visualization.stereo_euvi_plot import (  # noqa: E402
    EuvPlotConfig,
    build_manifest,
    plot_euvi_overview,
)


def _make_euvi_fits(
    directory: Path, wavelength: int, date_obs: str, seed: int = 1
) -> Path:
    data = np.random.default_rng(seed).random((16, 16), dtype=np.float32)
    header = fits.Header()
    header["NAXIS"] = 2
    header["CRPIX1"] = 8.0
    header["CRPIX2"] = 8.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 10.0
    header["CDELT2"] = 10.0
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["WAVELNTH"] = wavelength
    header["DATE-OBS"] = date_obs
    header["EXPTIME"] = 1.0
    header["DETECTOR"] = "EUVI"
    header["OBSRVTRY"] = "STEREO-A"
    path = (
        directory
        / f"euvi_{wavelength}_{date_obs.replace(':', '').replace('-', '')}.fits"
    )
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return path


def test_build_manifest_groups_and_sorts(tmp_path: Path) -> None:
    _make_euvi_fits(tmp_path, 171, "2025-01-24T04:48:40", seed=2)
    _make_euvi_fits(tmp_path, 195, "2025-01-24T04:48:30", seed=3)
    _make_euvi_fits(tmp_path, 171, "2025-01-24T04:48:30", seed=1)

    manifest = build_manifest(tmp_path, [171, 195])

    assert [record["wavelength"] for record in manifest] == [171, 171, 195]
    assert [record["date_obs"] for record in manifest] == [
        "2025-01-24T04:48:30",
        "2025-01-24T04:48:40",
        "2025-01-24T04:48:30",
    ]
    assert all(record["filename"] for record in manifest)


def test_build_manifest_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path)


def test_worker_emits_artifacts(monkeypatch, tmp_path: Path, capsys) -> None:
    produced = tmp_path / "a.png"
    monkeypatch.setattr(
        "solar_apps.workflows.visualization.stereo_euvi_plot.plot_euvi_overview",
        lambda config: [produced],
    )

    from solar_apps.frontends.app_v1.stereo_euvi_worker import main

    returncode = main(
        [
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert returncode == 0
    assert "APP_V1_EVENT" in captured.out
    assert "a.png" in captured.out


def test_function_spec_builds_arguments() -> None:
    function = DEFAULT_FUNCTION_CATALOG.get("stereo-euvi-plot")
    allowed = Path("/data").resolve()

    module, argv, values = function.build_arguments(
        {
            "input_dir": str(allowed / "euvi"),
            "output_dir": str(allowed / "out"),
            "wavelengths": [171, 195],
        },
        allowed_roots=[str(allowed)],
    )

    assert module == "solar_apps.frontends.app_v1.stereo_euvi_worker"
    assert "--input-dir" in argv
    assert "--output-dir" in argv
    assert values["wavelengths"] == [171, 195]

    with pytest.raises(ValueError):
        function.build_arguments(
            {
                "input_dir": str(Path("/outside/euvi").resolve()),
                "output_dir": str(allowed / "out"),
            },
            allowed_roots=[str(allowed)],
        )


@pytest.mark.skipif(
    not hasattr(__import__("cv2"), "VideoWriter"),
    reason="OpenCV unavailable",
)
def test_plot_euvi_overview_produces_pngs(tmp_path: Path) -> None:
    _make_euvi_fits(tmp_path, 171, "2025-01-24T04:48:30", seed=1)
    _make_euvi_fits(tmp_path, 195, "2025-01-24T04:48:30", seed=2)

    config = EuvPlotConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        wavelengths=(171, 195),
    )
    produced = plot_euvi_overview(config)

    assert produced
    assert produced[0].exists()
    assert produced[0].suffix == ".png"
