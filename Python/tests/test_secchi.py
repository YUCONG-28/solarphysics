"""Contract tests for official SECCHI_PREP integration (no IDL emulation)."""

import json

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.map.secchi import (
    RECIPE,
    _idl_string,
    file_sha256,
    prepare_euvi,
    provenance_path,
    runtime_environment,
    verify_prepared,
)


def test_runtime_missing_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="IDL executable"):
        runtime_environment(tmp_path, "/nonexistent/idl")


def test_idl_paths_are_quoted_not_commands():
    assert _idl_string("a'b") == "'a''b'"
    with pytest.raises(ValueError):
        _idl_string("a\nexit")


def test_provenance_detects_modified_calibrated_pixels(tmp_path):
    path = tmp_path / "prepared.fits"
    fits.PrimaryHDU(np.ones((8, 8)), fits.Header({"SPPREP": RECIPE})).writeto(path)
    provenance_path(path).write_text(
        json.dumps(
            dict(recipe=RECIPE, status="verified", output_sha256=file_sha256(path))
        )
    )
    assert verify_prepared(path)["status"] == "verified"
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[0, 0] = 2
    with pytest.raises(ValueError, match="SHA-256"):
        verify_prepared(path)


def test_reject_already_processed_before_runtime(tmp_path):
    path = tmp_path / "processed.fits"
    fits.PrimaryHDU(
        np.ones((8, 8)),
        fits.Header(
            {"DETECTOR": "EUVI", "EXPTIME": 2, "FILENAME": "20250124_040000_14euA.fts"}
        ),
    ).writeto(path)
    with pytest.raises(ValueError, match="double calibration"):
        prepare_euvi(
            path, tmp_path / "out", ssw_root=tmp_path, idl_executable="/nonexistent/idl"
        )
