"""Native sessions never need a shared observer or image reprojection."""

import os
import subprocess
import sys


def test_native_pair_loader_keeps_missing_view_and_checks_identity():
    from types import SimpleNamespace

    import pytest
    from solar_apps.frontends.jet_lab.native_frames import (
        load_native_pair,
        native_pair_status,
    )

    document = SimpleNamespace(
        sha256="b",
        difference_sha256=None,
        info={"midpoint_utc": "2000-01-01T12:00:00", "dsun_m": 1.49e11},
    )
    pair = {
        "AIA": None,
        "EUVI": {"image_sha256": "b"},
        "status": "no_other_frame",
        "delta_emission_s": None,
        "repeated_other": False,
    }
    docs = load_native_pair(pair, {"b": document}, {}, lambda _: None)
    assert docs == [None, document]
    assert "缺配" in native_pair_status(docs, pair)
    document.sha256 = "changed"
    with pytest.raises(ValueError, match="校验"):
        load_native_pair(pair, {"b": document}, {}, lambda _: None)


def test_native_timeline_export_restore_without_mapping(tmp_path):
    script = r"""
import json, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab import common_view
from solar_toolkit.map.jet_annotations import file_sha256
out=Path(sys.argv[1]);app=QApplication([])
def forbidden(*a,**kw):raise AssertionError('Native flow called reprojection')
for name in ('build_mapping','sample_mapping','reproject_photosphere','native_roi_mask'):
 setattr(common_view,name,forbidden)
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=24,CRPIX2=20,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
records=[]
for i in range(4):
 hh=h.copy();hh['DATE_OBS']=f'2000-01-01T12:00:{i*12:02d}'
 fits.writeto(out/f'{i}.fits',np.ones((40,48))*10+i,hh)
 records.append(dict(instrument='AIA' if i%2==0 else 'EUVI',band=304,image=f'{i}.fits',
  image_sha256=file_sha256(out/f'{i}.fits'),midpoint_utc=f'2000-01-01T12:00:{i*12+1:02d}',dsun_m=149600000000.0))
timeline=out/'timeline.json';timeline.write_text(json.dumps(dict(schema='solarphysics.jet_lab.timeline',version=1,frames=records)))
(out/'COMPLETE.json').write_text(json.dumps({'sha256':{'timeline.json':file_sha256(timeline)}}))
w=JetLabWindow([out]);assert w.native_only;c=w.common;w.show()
# Isolate session serialization from numerical tests; scientific reconstruction has its own suite.
w.reconstruction_result=None;w.reconstruction_signature=None
w.refresh_reconstruction=lambda:None
w.reconstruction_current=lambda:w.reconstruction_signature=='current'
def calculate():
 w.reconstruction_result=dict(points=[dict(number=1,role='jet',order=1,endpoint='inner',
  numerical_valid=True,geometry_valid=False,xyz_Rsun=np.array([1.1,.2,.3]),height_Rsun=.158,
  gap_Mm=.01,ray_angle_deg=29,residual_arcsec=np.array([.1,.2]),issues=['uncertainty_unavailable'])],
  display_geometry_used=False, summary={'axis':{
   'valid':True,'geometry_valid':False,'method':'mean_centered_equal_weight_pca',
   'method_version':1,'citation_doi':'10.1007/s11207-023-02122-9',
   'point_residuals':[{'number':1,'distance_Rsun':.001,'distance_Mm':.6957}]}})
 w.reconstruction_result['joint_fits']=dict(selected_model='line',line=dict(fitted_points=[dict(
  number=1,xyz_Rsun=[1.1,.2,.3],predicted_pixel_xy=[[20,21],[22,23]],
  residual_pixel_xy=[[.1,.2],[.3,.4]],residual_px=[.2236,.5],issues=[])]))
 w.reconstruction_result['joint_robust']=dict(selected_model='line',line=dict(fitted_points=[]))
 w.reconstruction_signature='current'
 return w.reconstruction_result
w.calculate_3d=calculate
w.recovery.timer.stop();w.undo_history.timer.stop()
def wait():
 end=time.monotonic()+15
 while time.monotonic()<end:
  app.processEvents();time.sleep(.02)
  if c.future is None and not c.debounce.isActive():return
 raise AssertionError(c.status.text())
c.load_timeline(timeline);wait()
assert c.reference is None and c.reference_hash is None and c.last_arrays is None
assert not c.cache and not c.preview_cache
assert w.panes[1].doc is not None
# A missing AIA frame must not stop display of the actual EUVI observation.
pair=dict(c.paired[0]);pair.update(AIA=None,status='outside_tolerance',delta_emission_s=None)
c.pending_frame=pair;c.request();wait()
assert w.panes[0].doc is None and w.panes[1].doc is not None
assert '缺配' in c.status.text()
# An explicitly chosen pair is retained through save/reload; no rematching on restore.
w.open_image(0,out/'0.fits');wait()
assert c.pair_metadata() is None  # Old missing/matched metadata cannot attach to new originals.
for pane in w.panes:
 pane.doc.add_tiepoint([20,20],1,'possible','UI test candidate')
assert all(not pane.doc.state['axis'] for pane in w.panes)
before=[json.dumps(p.doc.state,sort_keys=True) for p in w.panes]
calculate();path=c.save_to(out/'native_session')
s=json.loads(path.read_text());assert s['version']==2 and s['reconstruction_current']
calc=json.loads((path.parent/'calculation_input.json').read_text())
assert calc['requires_remote_execution'] is False and calc['export_kind']=='candidate_geometry'
assert not calc['display_shell_used_for_geometry']
assert (path.parent/'calculation_pair/annotations.json').exists()
assert calc['axis_diagnostic']=='jet_axis.json'
axis=json.loads((path.parent/'jet_axis.json').read_text())
assert axis['axis']['citation_doi']=='10.1007/s11207-023-02122-9'
assert axis['axis']['geometry_valid'] is False
assert '1,0.001,0.6957' in (path.parent/'jet_axis_residuals.csv').read_text()
assert '1.1,0.2,0.3' in (path.parent/'reconstruction_points.csv').read_text()
import csv
joint=json.loads((path.parent/'joint_fits.json').read_text())
assert joint['selected_model']=='line' and joint['result']['line']['fitted_points'][0]['number']==1
row=next(csv.DictReader((path.parent/'joint_fits.csv').open()))
assert row['model']=='line' and row['selected_model']=='line' and row['B_residual_px']=='0.5'
assert (path.parent/'joint_robust.json').exists() and (path.parent/'joint_robust.csv').exists()
checks=json.loads((path.parent/'COMPLETE.json').read_text())['sha256']
assert all(file_sha256(path.parent/name)==digest for name,digest in checks.items())
c.restore_session(path);wait()
assert [json.dumps(p.doc.state,sort_keys=True) for p in w.panes]==before
assert w.reconstruction_signature=='current' and c.reference is None
w.reconstruction_signature='old'
stale=c.save_to(out/'stale_session')
assert json.loads((stale.parent/'calculation_input.json').read_text())['export_kind']=='diagnostic_only'
assert not (stale.parent/'reconstruction.json').exists()
assert (stale.parent/'reconstruction_history.json').exists()
assert not (stale.parent/'jet_axis.json').exists()
assert (stale.parent/'jet_axis_history.json').exists()
assert not (stale.parent/'joint_fits.json').exists()
assert (stale.parent/'joint_fits_history.csv').exists()
# Read a previous-format session and retain its shared-view settings only as history.
legacy=out/'legacy';import shutil;shutil.copytree(path.parent,legacy)
s['version']=1;s.pop('reconstruction_result');s.pop('reconstruction_signature')
(legacy/'session.json').write_text(json.dumps(s))
v=json.loads((legacy/'view_state.json').read_text());v['window_view_mode']='common';v['height_rsun']=.2
(legacy/'view_state.json').write_text(json.dumps(v))
(legacy/'COMPLETE.json').write_text(json.dumps({'sha256':{str(p.relative_to(legacy)):file_sha256(p) for p in legacy.rglob('*') if p.is_file() and p!=legacy/'COMPLETE.json'}}))
c.restore_session(legacy/'session.json');wait()
assert w.view_mode=='native' and c.historical_shared_view['height_rsun']==.2
assert c.reference is None and not c.cache
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.close();app.processEvents();assert c.closed
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
