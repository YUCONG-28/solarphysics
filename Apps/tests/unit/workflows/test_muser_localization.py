import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.radio import muser_localization as workflow


def make_input(path):
    h = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 20.0,
            "CRPIX2": 20.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 42.0,
            "CDELT2": 42.0,
            "CTYPE4": "STOKES",
            "CRVAL4": 1,
            "DATE-OBS": "2020-01-02T04:48:30",
            "BUNIT": "Jy/beam",
        }
    )
    y, x = np.indices((40, 40))
    data = np.stack(
        [
            2
            + (10 + i) * np.exp(-0.5 * (((x - 20.3) / 3) ** 2 + ((y - 19.1) / 4) ** 2))
            for i in range(21)
        ]
    )[:, None, None]
    hdus = [fits.PrimaryHDU(data, h)]
    for name, values in [
        ("FMHZ", np.arange(21) * 4.88 + 148.477),
        ("BMAJ", np.ones(21)),
        ("BMIN", np.ones(21) * 0.5),
        ("BPA", np.arange(21)),
    ]:
        hdus.append(fits.ImageHDU(values, name=name))
    fits.HDUList(hdus).writeto(path)


def job(path="input.fits", plane=20):
    return {
        "path": str(path),
        "plane_index": plane,
        "roi_xyxy": [3, 3, 37, 37],
        "source_id": "test-source",
    }


def test_explicit_plane_and_provenance_roundtrip(tmp_path):
    source = tmp_path / "input.fits"
    make_input(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "new_run"
    manifest = workflow.run_localization(
        {"jobs": [job()]}, output, base_directory=tmp_path
    )
    assert manifest["status"] == "completed"
    assert manifest["job_count"] == 1
    row = json.loads((output / "jobs/job_000000.json").read_text())
    assert row["plane_index"] == 20
    assert row["frequency_mhz"] == pytest.approx(148.477 + 20 * 4.88)
    assert row["center_x_pixel"] == pytest.approx(20.3, abs=0.01)
    assert row["input_sha256"] == digest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    for record in manifest["files"]:
        assert (
            hashlib.sha256((output / record["path"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    with pytest.raises(FileExistsError):
        workflow.run_localization({"jobs": [job()]}, output, base_directory=tmp_path)


def test_invalid_plane_and_missing_input_are_retained(tmp_path):
    make_input(tmp_path / "input.fits")
    output = tmp_path / "failed_run"
    workflow.run_localization(
        {"jobs": [job(plane=-1), job(plane=21), job("missing.fits")]},
        output,
        base_directory=tmp_path,
    )
    rows = json.loads((output / "sources.json").read_text())
    assert len(rows) == 3 and all(not row["fit_success"] for row in rows)
    manifest = json.loads((output / "COMPLETE.json").read_text())
    assert manifest["failed_job_count"] == 3
    assert manifest["status"] == "completed_with_failed_jobs"


def test_cli_config_relative_paths_and_nonzero_failed_jobs(tmp_path):
    make_input(tmp_path / "input.fits")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"jobs": [job(), job(plane=99)]}))
    assert (
        workflow.main(["--config", str(config), "--output", str(tmp_path / "cli_run")])
        == 2
    )


def test_save_failure_propagates_without_completion(tmp_path, monkeypatch):
    make_input(tmp_path / "input.fits")
    original = workflow._write_json

    def fail_jobs(path, value):
        if path.parent.name == "jobs":
            raise OSError("simulated unavailable output storage")
        original(path, value)

    monkeypatch.setattr(workflow, "_write_json", fail_jobs)
    output = tmp_path / "storage_failure"
    with pytest.raises(OSError, match="unavailable output"):
        workflow.run_localization({"jobs": [job()]}, output, base_directory=tmp_path)
    assert not (output / "COMPLETE.json").exists()


def test_input_change_does_not_mark_complete(tmp_path, monkeypatch):
    source = tmp_path / "input.fits"
    make_input(source)
    original = workflow.fit_radio_plane

    def mutate_input(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.setval(source, "HISTORY", value="modified during processing")
        return result

    monkeypatch.setattr(workflow, "fit_radio_plane", mutate_input)
    output = tmp_path / "changed_input"
    with pytest.raises(RuntimeError, match="Input changed"):
        workflow.run_localization({"jobs": [job()]}, output, base_directory=tmp_path)
    assert not (output / "COMPLETE.json").exists()
