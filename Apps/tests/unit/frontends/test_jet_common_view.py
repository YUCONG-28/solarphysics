"""Run the real asynchronous viewer in an isolated Qt6 process."""

import os
import subprocess
import sys


def test_common_view_native_state_timeline_export(tmp_path):
    script = r"""
import sys,time,json
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_toolkit.map.jet_annotations import file_sha256
out=Path(sys.argv[1]);app=QApplication([])
yy,xx=np.mgrid[:96,:128]
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',CRPIX1=64,CRPIX2=48,CRVAL1=850,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,DATE_OBS='2000-01-01T12:00:30',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(4):
    hh=h.copy();hh['DATE_OBS']=f'2000-01-01T12:00:{i*12:02d}';fits.writeto(out/f'{i}.fits',2*(5+80*np.exp(-.5*((yy-48)/3)**2)),hh)
w=JetLabWindow([out],native_only=False);w.show();w.open_image(0,out/'0.fits');w.open_image(1,out/'1.fits');c=w.common;c.show()
def wait():
    end=time.monotonic()+15
    while time.monotonic()<end:
        app.processEvents();time.sleep(.02)
        if c.future is None and c.roi_future is None and not c.debounce.isActive():return
    raise AssertionError(c.status.text())
c.request();wait();assert c.last_arrays is not None
doc=w.panes[0].doc;doc.parameters(method='manual',value=15,min_area=1);doc.select([20,48]);doc.trace([20,48]);doc.confirm();original=json.dumps(doc.state,sort_keys=True)
for mode in ['common','native','split']:
    w.set_view_mode(mode);wait();app.processEvents()
    assert w.common.isVisible()==(mode!='native')
    assert w.native_splitter.isVisible()==(mode!='common')
    assert json.dumps(doc.state,sort_keys=True)==original
assert w.vertical_splitter.count()==2 and w.native_splitter.height()>=320
w.vertical_splitter.setSizes([450,550]);app.processEvents()
assert all(n>0 for n in w.vertical_splitter.sizes())
for idx in range(4):
    c.height.setCurrentIndex(idx);wait()
    assert json.dumps(doc.state,sort_keys=True)==original
w.view_layer.setCurrentIndex(1);w.redraw();wait();assert json.dumps(doc.state,sort_keys=True)==original
w.view_layer.setCurrentIndex(0)
c.height.setCurrentIndex(0);wait();c.make_draft([[10,40],[110,40],[110,57],[10,57]]);wait()
assert doc.sha256 in c.native_drafts
assert json.dumps(doc.state,sort_keys=True)==original
c.accept_roi(0);assert doc.state['roi_runs'];doc.undo();assert json.dumps(doc.state,sort_keys=True)==original
for mode in range(3):c.display.setCurrentIndex(mode);app.processEvents()
c.axes[0].set_xlim(20,100);c.axes[0].set_ylim(35,60);assert c.limits[0]==(20.,100.)
doc.add_tiepoint([30,48],1,'manual_confirmed','bright knot');w.panes[1].doc.add_tiepoint([30,48],1,'manual_confirmed','bright knot')
r=c.precheck();assert not r['display_geometry_used'] and not r['geometry_valid']
reference=c.reference_hash
records=[]
for i in range(4):
    records.append(dict(instrument='AIA' if i%2==0 else 'EUVI',band=304,image=f'{i}.fits',image_sha256=file_sha256(out/f'{i}.fits'),midpoint_utc=f'2000-01-01T12:00:{i*12+1:02d}',dsun_m=149600000000.0))
(out/'timeline.json').write_text(json.dumps(dict(schema='solarphysics.jet_lab.timeline',version=1,frames=records)))
(out/'COMPLETE.json').write_text(json.dumps({'sha256':{'timeline.json':file_sha256(out/'timeline.json')}}))
c.load_timeline(out/'timeline.json');wait();assert c.reference_hash==reference
saved_axis=doc.state['axis'].copy();c.slider.setValue(c.slider.maximum());c.start_job();c.height.setCurrentIndex(1);wait();assert c.current_pair==c.paired[-1];c.slider.setValue(0);wait();assert c.documents[doc.sha256].state['axis']==saved_axis
w.panes[1].display_limits=[3,98];w.colour.setCurrentText('viridis')
path=c.save_to(out/'session',for_compute=True);assert path.exists()
session=json.loads(path.read_text())
assert all('/' in d['annotation'] and '\\' not in d['annotation'] for d in session['documents'])
assert all('\\' not in frame['image'] for frame in session['frames'])
checks=json.loads((path.parent/'COMPLETE.json').read_text())['sha256']
assert 'calculation_pair/annotations.json' in checks and all('\\' not in name for name in checks)
assert all(file_sha256(path.parent/name)==digest for name,digest in checks.items())
calc=json.loads((path.parent/'calculation_input.json').read_text());assert not calc['display_shell_used_for_geometry']
undo_before=len(c.documents[doc.sha256]._undo)
c.restore_session(path);wait();assert c.documents[doc.sha256].state['axis']==saved_axis
assert len(c.documents[doc.sha256]._undo)==undo_before
assert w.panes[1].display_limits==[3,98] and w.colour.currentText()=='viridis'
live=c.documents[doc.sha256];live.dirty=True
try:c.save_to(path.parent)
except FileExistsError:pass
else:raise AssertionError('Existing outputs must not be overwritten')
assert live.dirty
partial=out/'partial';partial.mkdir();(partial/'session.json').write_text(path.read_text());(partial/'COMPLETE.json').write_text('{"sha256":{}}')
try:c.restore_session(partial/'session.json')
except ValueError:pass
else:raise AssertionError('Partial session accepted')
for theme in ['Light','Dark','Auto']:
    w.apply_theme(theme);app.processEvents();w.grab().save(str(out/(theme+'.png')))
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
c.request();w.close();app.processEvents();assert w.closed_cleanly and c.closed
fresh=JetLabWindow([out],native_only=False);c=fresh.common;c.load_timeline(out/"timeline.json");c.request();wait()
assert c.reference is not None and fresh.panes[0].doc is not None
fresh.close();app.processEvents();assert c.closed
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=100,
    )
    assert result.returncode == 0, result.stdout + result.stderr
