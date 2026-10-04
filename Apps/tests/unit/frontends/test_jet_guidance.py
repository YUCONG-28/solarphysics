"""Offline help, responsive workspaces and identity-checked recovery."""

import os
import subprocess
import sys


def test_guide_layout_recovery_and_state_protection(tmp_path):
    script = r"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QMessageBox,QLineEdit
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab.workflow_guide import annotation_signature,workflow_status
out=Path(sys.argv[1]);app=QApplication([])
y,x=np.mgrid[:240,:300]
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=150,CRPIX2=120,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:30',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
data=2*(5+80*np.exp(-.5*((y-(180-.3*x))/4)**2))
for i in range(2):fits.writeto(out/f'synthetic_{i}.fits',data+i,h)
w=JetLabWindow([out],native_only=False);w.recovery.timer.stop();w.recovery.directory=out/'recovery';w.recovery.path=w.recovery.directory/'draft.json'
w.show();w.open_image(0,out/'synthetic_0.fits');w.open_image(1,out/'synthetic_1.fits')
assert not w.confirm_button.isEnabled()
for pane in w.panes:
 d=pane.doc;d.parameters(method='manual',value=25,min_area=1);d.select([40,168]);d.trace([40,168]);d.confirm()
 for number,xy in enumerate(([60,162],[130,141],[220,114]),1):d.add_tiepoint(xy,number,'manual_confirmed','合成特征')
w.sync_controls();assert w.confirm_button.isEnabled()
signature=annotation_signature(w);statuses=workflow_status(w)[0]
assert not hasattr(w.guide, 'show_help')
for i in range(7):w.guide.steps.setCurrentRow(i);app.processEvents()
assert annotation_signature(w)==signature and workflow_status(w)[0]==statuses
w.guide.steps.setCurrentRow(4)
def wait():
 end=time.monotonic()+15
 while time.monotonic()<end:
  app.processEvents();time.sleep(.02)
  if w.common.future is None and w.common.roi_future is None and not w.common.debounce.isActive():return
 raise AssertionError(w.common.status.text())
for theme in ('Light','Dark','Auto'):
 w.apply_theme(theme)
 for width,height in ((1280,800),(1440,900),(1920,1080)):
  w.resize(width,height);w.set_view_mode('native');app.processEvents()
  assert w.width()<=width and w.height()<=height,(w.size(),width,height)
  assert min(p.canvas.height() for p in w.panes)>300
  w.grab().save(str(out/f'native_{theme}_{width}.png'))
 # Wide title metrics must wrap without forcing the dual-view workspace wider.
 fonts=[pane.title.font() for pane in w.panes]
 for pane in w.panes:
  font=pane.title.font();font.setPointSize(24);pane.title.setFont(font)
 for _ in range(5):app.processEvents()
 w.resize(1280,800)
 for _ in range(5):app.processEvents()
 assert w.width()<=1280 and w.height()<=800,w.size()
 assert w.minimumSizeHint().width()<=1280,w.minimumSizeHint()
 assert all(p.isVisible() and p.title.isVisible() for p in w.panes)
 assert min(p.canvas.height() for p in w.panes)>300
 for pane,font in zip(w.panes,fonts):pane.title.setFont(font)
 app.processEvents()
w.guide.hide();w.resize(1440,900);w.set_view_mode('split');wait()
w.vertical_splitter.setSizes([350,450]);app.processEvents();ratio=w.vertical_splitter.sizes()
w.set_view_mode('common');wait();w.set_view_mode('split');wait()
assert max(abs(a-b) for a,b in zip(ratio,w.vertical_splitter.sizes()))<4
w.set_view_mode('common');wait();w.guide.show();app.processEvents()
w.grab().save(str(out/'workspace.png'))
w.set_view_mode('native');w.resize(1440,900);app.processEvents();w.redraw()
w.grab().save(str(out/'annotation.png'))
assert annotation_signature(w)==signature
# No real sleep: simulate the inactivity clock and verify atomic save.
w.recovery.poll();assert not w.recovery.path.exists()
w.recovery.changed_at-=2.1;w.recovery.poll();assert w.recovery.path.is_file()
saved=w.recovery.path.read_bytes();w.doc.reverse();assert annotation_signature(w)!=signature
w.recovery.restore(w.recovery.path);assert annotation_signature(w)==signature
w.doc.checkpoint('continue editing');w.recovery.poll();w.recovery.saved_at-=31;w.recovery.poll()
assert w.recovery.path.read_bytes()!=saved
assert all(p.suffix=='.json' for p in w.recovery.directory.iterdir())
# A failed atomic replacement keeps the last draft and the unsaved edits.
import solar_apps.frontends.jet_lab.recovery as recovery_module
previous_bytes=w.recovery.path.read_bytes();replace=recovery_module.os.replace
def fail_replace(*args):raise OSError('simulated write failure')
recovery_module.os.replace=fail_replace
try:
 try:w.recovery.write()
 except OSError:pass
 else:raise AssertionError('Save failure swallowed')
finally:recovery_module.os.replace=replace
assert w.recovery.path.read_bytes()==previous_bytes and w.doc.dirty
assert all(p.suffix=='.json' for p in w.recovery.directory.iterdir())
before=annotation_signature(w)
envelope=json.loads(w.recovery.path.read_text());key=next(iter(envelope['payload']['documents']))
envelope['payload']['documents']['0'*64]=envelope['payload']['documents'].pop(key)
body=json.dumps(envelope['payload'],ensure_ascii=False,sort_keys=True)
envelope['sha256']=hashlib.sha256(body.encode()).hexdigest()
bad=w.recovery.directory/'bad.json';bad.write_text(json.dumps(envelope))
try:w.recovery.restore(bad)
except ValueError:pass
else:raise AssertionError('Image identity mismatch accepted')
assert annotation_signature(w)==before
broken=w.recovery.directory/'partial.json';broken.write_text('{')
try:w.recovery.restore(broken)
except ValueError:pass
else:raise AssertionError('Partial draft accepted')
assert annotation_signature(w)==before
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.close();app.processEvents();assert w.closed_cleanly and not w.recovery.timer.isActive()
print('Guide, layout, state, recovery and cleanup passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
