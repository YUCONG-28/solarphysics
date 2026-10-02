"""Exercise real native controls and geometry without a projection backend."""

import os
import subprocess
import sys


def test_native_buttons_geometry_staleness_undo_and_save(tmp_path):
    script = r"""
import json,sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QPushButton,QFileDialog
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab import common_view
out=Path(sys.argv[1]);app=QApplication([])
def forbidden(*a,**kw):raise AssertionError('Image projection forbidden')
for n in ('build_mapping','sample_mapping','reproject_photosphere','native_roi_mask'):
 setattr(common_view,n,forbidden)
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',CRPIX1=64,CRPIX2=64,
 CRVAL1=800,CRVAL2=-160,CDELT1=1,CDELT2=1,EXPTIME=2,DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,
 HGLN_OBS=0,HGLT_OBS=-5.0,RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):
 hh=h.copy();hh['HGLN_OBS']=i*29
 fits.writeto(out/f'{i}.fits',np.ones((128,128)),hh)
w=JetLabWindow([out]);w.show();app.processEvents();w.undo_history.timer.stop();w.recovery.timer.stop()
for i in range(2):w.open_image(i,out/f'{i}.fits')
w.start_marking()
for n in range(1,4):
 w.tie_id.setValue(n)
 w.point_endpoint.setCurrentIndex(1 if n==1 else 2 if n==3 else 0)
 for i in range(2):w.perform(lambda i=i,n=n:w.mark_native_point(i,[60+n*5,64-n*3]),i)
assert all(len(p.doc.state['tiepoints'])==3 and not p.doc.state['axis'] for p in w.panes)
assert w.undo_history.move(-1)
assert len(w.panes[1].doc.state['tiepoints'])==2
assert w.undo_history.move(1)
assert len(w.panes[1].doc.state['tiepoints'])==3
w.tie_id.setValue(1)
next(b for b in w.findChildren(QPushButton) if b.text()=='设为内端').click()
w.tie_id.setValue(3)
next(b for b in w.findChildren(QPushButton) if b.text()=='设为外端').click()
assert all(p.doc.state['tiepoints'][0]['endpoint']=='inner' and p.doc.state['tiepoints'][2]['endpoint']=='outer' for p in w.panes)
button=next(b for b in w.findChildren(QPushButton) if b.text()=='计算三维')
button.click();app.processEvents()
from PyQt6.QtTest import QTest
for _ in range(1000):
 if w.reconstruction_dialog is not None:break
 QTest.qWait(10)
assert w.reconstruction_dialog is not None, (w.point_hint.text(), w._fit_future, w._fit_generation)
assert w.reconstruction_dialog.isVisible()
assert w.reconstruction_current() and len(w.reconstruction_result['points'])==3
assert not w.reconstruction_result['geometry_valid']
assert w.undo_history.move(-1)
assert w.reconstruction_result is None and w.reconstruction_dialog is None
assert w.undo_history.move(1) and w.reconstruction_current()
geometry=w.geometry_signature()
for theme in ('Light','Dark','Auto'):
 w.theme.setCurrentText(theme);w.resize(1280,800);app.processEvents()
 w.low.setValue(2);w.fit_images()
 assert w.geometry_signature()==geometry and w.reconstruction_current()
w.tie_id.setValue(3)
w.perform(lambda:w.mark_native_point(1,[78,54]),1)
assert not w.reconstruction_current() and '过期' in w.result_summary.text()
assert w.undo_history.move(-1) and w.reconstruction_current()
path=w.common.save_to(out/'revision')
assert path.exists()
complete=json.loads((path.parent/'COMPLETE.json').read_text())
assert 'reconstruction.json' in complete['sha256']
w.restore(path)
assert w.reconstruction_current() and len(w.panes[0].doc.state['tiepoints'])==3
assert w.point_endpoint.currentData()=='outer'
# Fault injection: saving failure must not delete marks or previous files.
old=w.common.save_to
QFileDialog.getExistingDirectory=lambda *a,**kw:str(out)
w.common.save_to=lambda *a,**kw:(_ for _ in ()).throw(OSError('read-only destination'))
assert not w.save_dialog();assert path.exists() and len(w.panes[1].doc.state['tiepoints'])==3
w.common.save_to=old
w.can_discard=lambda:True;w.close();assert w.closed_cleanly
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
