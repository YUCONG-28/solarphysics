"""Sample preparation uses caller-selected times and native synthetic inputs."""

import csv
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_apps.workflows.jet_lab import pilot


def test_missing_sample_times_fails_before_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="sample_times_utc"):
        pilot.prepare({"output_directory": str(output)})
    assert not output.exists()


def test_configured_times_prepare_native_samples_without_event_defaults(
    tmp_path, monkeypatch
):
    rows = []
    yy, xx = np.mgrid[:32, :32]
    for instrument in ("AIA", "EUVI"):
        for second in (0, 20):
            date = f"2000-01-01T12:00:{second:02d}"
            header = fits.Header(
                dict(
                    CTYPE1="HPLN-TAN",
                    CTYPE2="HPLT-TAN",
                    CUNIT1="arcsec",
                    CUNIT2="arcsec",
                    CRPIX1=16,
                    CRPIX2=16,
                    CRVAL1=0,
                    CRVAL2=0,
                    CDELT1=1,
                    CDELT2=1,
                    DATE_OBS=date,
                    EXPTIME=2,
                    DSUN_OBS=149600000000.0,
                    HGLN_OBS=0,
                    HGLT_OBS=0,
                    RSUN_REF=695700000,
                    WAVELNTH=304,
                    WAVEUNIT="angstrom",
                    BUNIT="DN",
                    SYNTHET=True,
                    TELESCOP="SDO/AIA" if instrument == "AIA" else "STEREO A",
                    INSTRUME="AIA" if instrument == "AIA" else "SECCHI",
                    DETECTOR="AIA" if instrument == "AIA" else "EUVI",
                    OBSRVTRY="SDO" if instrument == "AIA" else "STEREO_A",
                )
            )
            path = tmp_path / f"synthetic_{instrument}_{second}.fits"
            fits.writeto(
                path, 10 + np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 20), header
            )
            midpoint = Time(f"2000-01-01T12:00:{second + 1:02d}")
            rows.append(
                dict(
                    instrument=instrument,
                    band=304,
                    path=str(path),
                    midpoint_utc=midpoint.isot,
                    unix=midpoint.unix,
                    solar_unix=midpoint.unix - 149600000000.0 / 299792458,
                )
            )
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(pilot, "synthetic_qa", lambda output: None)
    monkeypatch.setattr(
        pilot,
        "registered_difference",
        lambda *args: (None, {"status": "not_requested"}),
    )
    output = tmp_path / "output"
    selected = ["2000-01-01T12:00:01", "2000-01-01T12:00:21"]
    pilot.prepare(
        dict(
            output_directory=str(output),
            inventory=str(inventory),
            sample_times_utc=selected,
            bands=[304],
            roi_arcsec={"AIA": [-5, 5, -5, 5], "EUVI": [-5, 5, -5, 5]},
        )
    )
    manifest = json.loads((output / "paired_samples.json").read_text())
    assert len(manifest["pairs"]) == 2
    assert [pair["split"] for pair in manifest["pairs"]] == ["sample", "sample"]
    assert [pair["views"][0]["info"]["midpoint_utc"] for pair in manifest["pairs"]] == [
        time + ".000" for time in selected
    ]
    assert all(
        fits.getheader(output / view["image"])["SYNTHET"]
        for pair in manifest["pairs"]
        for view in pair["views"]
    )
