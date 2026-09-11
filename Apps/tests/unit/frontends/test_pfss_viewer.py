"""The PFSS result viewer must work without the numerical backend."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from solar_apps.frontends.app_v1.pfss_adapter import PFSSAdapter
from solar_apps.platform.paths.native_dialog import NativeDialogForbiddenError
from solar_toolkit.modeling.pfss.bundle import save_bundle


def fixture_bundle(path):
    from astropy.io import fits

    path.mkdir()
    header = fits.Header(
        dict(
            CTYPE1="HPLN-TAN",
            CTYPE2="HPLT-TAN",
            CUNIT1="arcsec",
            CUNIT2="arcsec",
            CRPIX1=32,
            CRPIX2=32,
            CRVAL1=900,
            CRVAL2=-200,
            CDELT1=5,
            CDELT2=5,
            EXPTIME=2,
            DATE_OBS="2025-01-24T04:48:30",
            DSUN_OBS=149597870700,
            HGLN_OBS=0,
            HGLT_OBS=0,
            RSUN_REF=695700000,
            TELESCOP="SDO/AIA",
            WAVELNTH=171,
            WAVEUNIT="angstrom",
        )
    )
    fits.writeto(path / "aia_display.fits", np.ones((64, 64)), header)
    config = {
        "output_dir": "/remote/run",
        "pfss": {"rss": 2.5, "nphi": 360, "ns": 180, "nr": 35},
        "seed_hpc_box": None,
    }
    (path / "resolved_config.json").write_text(json.dumps(config))
    (path / "sources.csv").write_text(
        "source_id,time_utc,frequency_mhz,center_x_arcsec,center_y_arcsec\ns,2025-01-24T04:48:30Z,150,900,-200\n"
    )
    (path / "newkirk_forward_ranking.csv").write_text(
        "fieldline_id,multiplier,harmonic,median,count,max\na,1,1,42,1,42\n"
    )
    lines = [
        dict(
            fieldline_id="a",
            xyz_carrington_rsun=np.array([[1, 0, 0], [2, 0, 0]]),
            trace_valid=True,
            classification="open_positive",
        )
    ]
    save_bundle(
        path,
        lines,
        [dict(hpc_arcsec=np.array([[800, -150], [950, -210]]))],
        {"config": config["pfss"], "source_count": 1, "selected_fieldline_ids": ["a"]},
    )


def test_adapter_rejects_incomplete_and_outside(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    bundle = root / "run"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError):
        PFSSAdapter([root]).load(bundle)
    with pytest.raises(NativeDialogForbiddenError):
        PFSSAdapter([bundle]).load(root)


def test_export_preserves_original_and_does_not_run(tmp_path):
    bundle = tmp_path / "run"
    fixture_bundle(bundle)
    adapter = PFSSAdapter([tmp_path])
    adapter.load(bundle)
    original = json.loads((bundle / "resolved_config.json").read_text())
    target = adapter.export_config(
        tmp_path / "request.json",
        original,
        pfss=original["pfss"],
        seed_hpc_box=[800, 850, -200, -150],
    )
    result = json.loads(target.read_text())
    assert (
        result["output_dir"] != "/remote/run"
        and original["output_dir"] == "/remote/run"
    )
    assert result["seed_hpc_box"] == [800, 850, -200, -150]
    with pytest.raises(FileExistsError):
        adapter.export_config(
            target, original, pfss=original["pfss"], seed_hpc_box=None
        )


def test_native_viewer_themes_reload_filter_and_cleanup(tmp_path):
    bundle = tmp_path / "run"
    fixture_bundle(bundle)
    script = """
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.frontends.app_v1.pfss_page import PFSSPanel
from solar_apps.frontends.app_v1.theme import AppV1ThemeController
root=Path(sys.argv[1]); repo=Path.cwd()
app=QApplication([])
theme=AppV1ThemeController(app)
layout=RuntimeLayout.discover(repo,environ={"SOLAR_APPS_LOCAL_ROOT":str(root/"Local")})
panel=PFSSPanel(layout,allowed_roots=[root]);panel.resize(1400,850)
for mode in ["light","dark","auto"]:
    theme.set_mode(mode);panel.load_result(root/"run");app.processEvents()
    assert panel.lines.count()==1 and len(panel.lines.selectedItems())==1
    panel.classification.setCurrentText("closed");assert panel.lines.count()==0
    panel.classification.setCurrentText("all")
    panel.frequency.setCurrentIndex(1);panel.redraw();app.processEvents()
assert "sunkit_magex" not in sys.modules
panel.close();panel.deleteLater();app.processEvents();app.quit()
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
