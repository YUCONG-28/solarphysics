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
    header["SYNTHET"] = True
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
    _make_euvi_fits(tmp_path, 171, "2000-01-01T12:00:40", seed=2)
    _make_euvi_fits(tmp_path, 195, "2000-01-01T12:00:30", seed=3)
    _make_euvi_fits(tmp_path, 171, "2000-01-01T12:00:30", seed=1)

    manifest = build_manifest(tmp_path, [171, 195])

    assert [record["wavelength"] for record in manifest] == [171, 171, 195]
    assert [record["date_obs"] for record in manifest] == [
        "2000-01-01T12:00:30",
        "2000-01-01T12:00:40",
        "2000-01-01T12:00:30",
    ]
    assert all(record["filename"] for record in manifest)


def test_build_manifest_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path)


def test_worker_emits_artifacts(monkeypatch, tmp_path: Path, capsys) -> None:
    _make_euvi_fits(tmp_path, 171, "2000-01-01T12:00:30")
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

    assert values["calibration"] == "python-preprocess"
    assert function.parameter("ssw_root").group == "advanced"
    assert function.parameter("idl_executable").group == "advanced"
    _, calibration_args, calibration_values = function.build_arguments(
        {
            "input_dir": str(allowed / "euvi"),
            "output_dir": str(allowed / "out"),
            "calibration": "secchi-prep",
            "ssw_root": str(allowed / "ssw"),
            "idl_executable": "custom-idl",
        },
        allowed_roots=[str(allowed)],
    )
    assert "--calibration" in calibration_args
    assert "secchi-prep" in calibration_args
    assert calibration_values["ssw_root"] == str(allowed / "ssw")
    assert "--ssw-root" in calibration_args
    assert "custom-idl" in calibration_args
    assert (
        function.normalize_parameters(
            {"input_dir": str(allowed / "euvi"), "calibration": "legacy"}
        )["calibration"]
        == "legacy"
    )

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
    _make_euvi_fits(tmp_path, 171, "2000-01-01T12:00:30", seed=1)
    _make_euvi_fits(tmp_path, 195, "2000-01-01T12:00:30", seed=2)

    config = EuvPlotConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        wavelengths=(171, 195),
    )
    produced = plot_euvi_overview(config)

    assert produced
    assert produced[0].exists()
    assert produced[0].suffix == ".png"


def test_verified_secchi_output_is_not_exposure_divided_twice(tmp_path):
    import json
    from solar_toolkit.map.secchi import RECIPE, file_sha256, provenance_path
    from solar_apps.workflows.visualization.stereo_euvi_plot import exposure_normalized

    path = _make_euvi_fits(tmp_path, 171, "2000-01-01T12:00:30")
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[:] = 20.0
        hdus[0].header["EXPTIME"] = 4.0
        hdus[0].header["BUNIT"] = "DN/s"
        hdus[0].header["SPPREP"] = RECIPE
    provenance_path(path).write_text(
        json.dumps(
            dict(recipe=RECIPE, status="verified", output_sha256=file_sha256(path))
        )
    )
    assert np.all(exposure_normalized(path).data == 20.0)
    provenance_path(path).unlink()
    with pytest.raises(FileNotFoundError):
        exposure_normalized(path)


def test_worker_calibration_failure_does_not_render(monkeypatch, tmp_path, capsys):
    from solar_apps.frontends.app_v1.stereo_euvi_worker import main

    def unexpected(config):
        pytest.fail("Renderer must not be called when calibration is unavailable")

    monkeypatch.setattr(
        "solar_apps.workflows.visualization.stereo_euvi_plot.plot_euvi_overview",
        unexpected,
    )
    assert (
        main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--calibration",
                "secchi-prep",
            ]
        )
        == 1
    )
    assert "requires --ssw-root" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["overview", "movie"])
def test_worker_explicit_secchi_prep_delegates_without_python_normalization(
    monkeypatch, tmp_path, capsys, mode
):
    import json
    from solar_apps.frontends.app_v1.stereo_euvi_worker import main
    from solar_apps.workflows.visualization import stereo_euvi_plot as plotting
    from solar_toolkit.map.secchi import RECIPE, file_sha256, provenance_path

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    nested = input_dir / "nested"
    nested.mkdir()
    sources = [
        _make_euvi_fits(input_dir, 171, "2000-01-01T12:00:30"),
        _make_euvi_fits(nested, 171, "2000-01-01T12:01:30"),
    ]
    _make_euvi_fits(input_dir, 195, "2000-01-01T12:00:30")
    output_dir = tmp_path / "out"
    calibrated = output_dir / "calibrated"
    ssw_root = tmp_path / "ssw"
    calls = []

    def runtime(root, executable):
        calls.append(("runtime", root, executable))
        return executable, {}

    def prepare(source, destination, *, ssw_root, idl_executable):
        calls.append(("prepare", source, destination, ssw_root, idl_executable))
        destination.mkdir(parents=True, exist_ok=True)
        product = destination / f"{source.stem}_secchi_prep.fits"
        with fits.open(source) as hdus:
            hdus[0].data[:] = 20.0
            hdus[0].header["EXPTIME"] = 4.0
            hdus[0].header["BUNIT"] = "DN/s"
            hdus[0].header["SPPREP"] = RECIPE
            hdus.writeto(product)
        provenance_path(product).write_text(
            json.dumps(
                dict(
                    recipe=RECIPE,
                    status="verified",
                    output_sha256=file_sha256(product),
                )
            )
        )
        return product

    def forbidden(*args, **kwargs):
        pytest.fail("SECCHI_PREP products must bypass Python preprocessing")

    def render(config):
        assert config.input_dir == calibrated
        assert config.wavelengths == (171,)
        records = plotting.build_manifest(config.input_dir, config.wavelengths)
        assert len(records) == len(sources)
        for record in records:
            prepared = plotting.exposure_normalized(Path(record["path"]))
            assert np.all(prepared.data == 20.0)
        return [output_dir / ("result.png" if mode == "overview" else "result.mp4")]

    monkeypatch.setattr("solar_toolkit.map.secchi.runtime_environment", runtime)
    monkeypatch.setattr("solar_toolkit.map.secchi.prepare_euvi", prepare)
    monkeypatch.setattr(
        "solar_toolkit.map.euvi_preprocessing.prepare_euvi_map", forbidden
    )
    monkeypatch.setattr("solar_toolkit.map.euvi_preprocessing.prepared_map", forbidden)
    monkeypatch.setattr(
        plotting,
        "plot_euvi_overview" if mode == "overview" else "make_roi_movie",
        render,
    )
    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--mode",
                mode,
                "--wavelengths",
                "171",
                "--calibration",
                "secchi-prep",
                "--ssw-root",
                str(ssw_root),
                "--idl-executable",
                "mock-idl",
            ]
        )
        == 0
    )
    assert calls == [("runtime", ssw_root, "mock-idl")] + [
        ("prepare", source, calibrated, ssw_root, "mock-idl")
        for source in sorted(sources)
    ]
    assert not (output_dir / "preprocessing_audit.json").exists()
    events = [
        json.loads(line.removeprefix("APP_V1_EVENT "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("APP_V1_EVENT ")
    ]
    assert [
        event["payload"]["percent"] for event in events if event["kind"] == "progress"
    ] == [25, 50, 100]
    assert events[-1]["payload"] == {"status": "succeeded", "artifact_count": 1}


@pytest.mark.parametrize(
    "unit, expected, normalized",
    [("DN", 5, True), ("photon/s", 20, True), ("", 20, False)],
)
def test_jet_crop_preserves_unit_and_exposure_decision(
    tmp_path, unit, expected, normalized
):
    from solar_apps.workflows.jet_lab.pilot import native_cutout
    from solar_toolkit.map.jet_annotations import intensity_per_second

    p = _make_euvi_fits(tmp_path, 304, "2000-01-01T12:00:30")
    with fits.open(p, mode="update") as hdus:
        hdus[0].data[:] = 20
        hdus[0].header["EXPTIME"] = 4
        hdus[0].header["BUNIT"] = unit
    result = native_cutout(p, [-20, 20, -20, 20], tmp_path / "crop.fits")
    assert np.all(result.data == expected)
    assert np.all(intensity_per_second(result) == expected)
    assert bool(result.meta["jetnorm"]) == normalized
    assert str(result.meta.get("bunit", "")) == ("DN/s" if unit == "DN" else unit)
    assert (tmp_path / "crop.fits.preprocessing.json").exists()


@pytest.mark.parametrize("calibration", [None, "python-preprocess", "legacy"])
def test_python_worker_does_not_invoke_idl(monkeypatch, tmp_path, calibration):
    import json
    from solar_apps.frontends.app_v1.stereo_euvi_worker import main, build_parser

    def forbidden(*args, **kwargs):
        pytest.fail("IDL must not be used")

    monkeypatch.setattr("solar_toolkit.map.secchi.prepare_euvi", forbidden)
    monkeypatch.setattr("solar_toolkit.map.secchi.runtime_environment", forbidden)
    _make_euvi_fits(tmp_path, 304, "2000-01-01T12:00:30")
    monkeypatch.setattr(
        "solar_apps.workflows.visualization.stereo_euvi_plot.plot_euvi_overview",
        lambda c: [],
    )
    args = ["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    assert build_parser().parse_args(args).calibration == "python-preprocess"
    if calibration is not None:
        args.extend(["--calibration", calibration])
    assert main(args) == 0
    audit = json.loads((tmp_path / "out/preprocessing_audit.json").read_text())
    assert audit[0]["status"] == "limited_preprocessing"
    assert audit[0]["exposure_action"] == "unchanged_requires_review"


def test_completion_hashes_follow_updated_pairing_and_nested_manifest(tmp_path):
    import json
    from solar_apps.workflows.jet_lab.preprocessing_check import finalize_manifest
    from solar_toolkit.map.jet_annotations import file_sha256

    (tmp_path / "paired_samples.json").write_text(
        json.dumps({"pairs": [{"id": "304"}]})
    )
    (tmp_path / "PREPARATION_COMPLETE.json").write_text('{"obsolete": true}')
    nested = tmp_path / "example"
    nested.mkdir()
    (nested / "COMPLETE.json").write_text("{}")
    finalize_manifest(tmp_path)
    (tmp_path / "paired_samples.json").write_text(
        json.dumps({"pairs": [{"id": "171"}, {"id": "304"}]})
    )
    finalize_manifest(tmp_path)
    for marker in ("COMPLETE.json", "PREPARATION_COMPLETE.json"):
        record = json.loads((tmp_path / marker).read_text())
        assert "example/COMPLETE.json" in record["sha256"]
        assert all(file_sha256(tmp_path / p) == h for p, h in record["sha256"].items())
    assert (
        json.loads((tmp_path / "PREPARATION_COMPLETE.json").read_text())["pair_count"]
        == 2
    )


def test_crop_cache_includes_roi_and_requires_sidecar(tmp_path):
    import sunpy.map
    from solar_apps.workflows.jet_lab.pilot import native_cutout
    from solar_apps.workflows.jet_lab.cutout_cache import identity, verify
    from solar_toolkit.map.jet_annotations import file_sha256

    source = _make_euvi_fits(tmp_path, 304, "2000-01-01T12:00:30")
    dest = tmp_path / "cut.fits"
    bounds = [-20, 20, -20, 20]
    native_cutout(source, bounds, dest)
    m = sunpy.map.Map(source)
    key = identity(m, file_sha256(source), bounds)
    verify(dest, key)
    with pytest.raises(ValueError, match="Different ROI"):
        verify(dest, identity(m, file_sha256(source), [-10, 20, -20, 20]))
    Path(str(dest) + ".preprocessing.json").unlink()
    with pytest.raises(OSError):
        verify(dest, key)
