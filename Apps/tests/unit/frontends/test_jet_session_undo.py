"""Undo/redo crosses all seven stages, never deleting published local revisions."""

import os
import subprocess
import sys


def test_decorated_actions_receive_qt_button_signals(tmp_path):
    script = r"""
import sys
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QPushButton,QFileDialog,QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
app=QApplication([]);w=JetLabWindow([Path(sys.argv[1])],native_only=False);w.show();app.processEvents()
QFileDialog.getExistingDirectory=lambda *a,**k:''
QFileDialog.getOpenFileName=lambda *a,**k:('','')
def click(title, root=w):
 matches=[b for b in root.findChildren(QPushButton) if b.text()==title]
 assert len(matches)==1,(title,len(matches))
 matches[0].click();app.processEvents()
for title in ('保存新修订','恢复标注','以当前AIA为参考','几何预检','保存时序会话','导出计算输入'):
 click(title)
def choose_source():
 dialog=app.activeModalWidget();assert dialog.windowTitle()=='选择数据来源'
 click('加载配对样本',dialog)
QTimer.singleShot(0,choose_source)
click('打开事件')
assert app.activeModalWidget() is None
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.close();app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_session_undo_all_stages_and_identity_failure(tmp_path):
    script = r"""
import json,sys,time
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab.workflow_guide import annotation_signature
out=Path(sys.argv[1]); app=QApplication([])
y,x=np.mgrid[:100,:140]
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=70,CRPIX2=50,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:30',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(3):fits.writeto(out/f'{i}.fits',2*(5+80*np.exp(-.5*((y-50)/4)**2))+i,h)
w=JetLabWindow([out],native_only=False);w.show();app.processEvents();w.recovery.timer.stop();u=w.undo_history;u.timer.stop()
def settle():
 end=time.monotonic()+15
 while time.monotonic()<end:
  app.processEvents();time.sleep(.02)
  if w.common.future is None and not w.common.debounce.isActive():return
 raise AssertionError(w.common.status.text())
def undo():
 assert u.move(-1),w.statusBar().currentMessage()
 settle();u.capture()
def redo():
 assert u.move(1),w.statusBar().currentMessage()
 settle();u.capture()
# Stage 1: undo loading each original image, including empty initial workspace.
w.open_image(0,out/'0.fits');settle();u.capture()
w.open_image(1,out/'1.fits');settle();u.capture()
undo();assert w.panes[1].doc is None and w.panes[0].doc is not None
undo();assert all(p.doc is None for p in w.panes)
redo();assert w.panes[0].doc is not None and w.panes[1].doc is None
redo();assert w.panes[1].doc is not None
# Stage 2: inspection report and workspace selection.
w.show_processing();settle();u.capture();text=w.message.toPlainText()
undo();redo();assert w.message.toPlainText()==text
# Stage 3: display height, crop and per-view ROI draft.
w.set_view_mode('common');w.common.height.setCurrentIndex(2);settle();u.capture()
undo();assert w.common.height.currentIndex()==0
redo();assert w.common.height.currentIndex()==2
mask=np.zeros(w.doc.raw.shape,dtype=bool);mask[35:65,10:130]=True
key=w.panes[0].doc.sha256
w.common.native_drafts[key]=(mask,{'display_height':.1});u.capture('ROI draft')
w.common.accept_roi(0);settle();u.capture();assert key not in w.common.native_drafts
undo();assert key in w.common.native_drafts and not w.panes[0].doc.state.get('roi_runs')
redo();assert w.panes[0].doc.state.get('roi_runs')
w.set_view_mode('native');settle();u.capture()
# Stages 4/5: segmentation, branch confirmation and ordered points, on each side.
for i in range(2):
 w.active.setCurrentIndex(i);u.capture()
 for fn in (lambda:w.doc.parameters(method='manual',value=25,min_area=1),
            lambda:w.doc.select([40,50]),lambda:w.doc.trace([40,50]),lambda:w.doc.confirm(),
            lambda:w.doc.add_tiepoint([50,50],1,'possible','test')):
  before=annotation_signature(w);w.perform(fn);settle();u.capture();after=annotation_signature(w)
  assert before!=after
  undo();assert annotation_signature(w)==before
  redo();assert annotation_signature(w)==after
# Restore previous frame with full annotations; no aliasing between snapshots.
before=annotation_signature(w);w.open_image(1,out/'2.fits');settle();u.capture()
undo();assert annotation_signature(w)==before
redo();assert annotation_signature(w)!=before
undo();assert annotation_signature(w)==before
# Stage 6: precheck has its own undo boundary and never manufactures validity.
w.common.precheck();settle();u.capture();assert w.last_precheck is not None
undo();assert w.last_precheck is None
redo();assert w.last_precheck is not None
# Stage 7: save/export undo restores dirty state, keeps immutable files.
path=w.common.save_to(out/'export',for_compute=True);settle();u.capture()
files={str(p):p.read_bytes() for p in (out/'export').rglob('*') if p.is_file()}
undo();assert files=={p:Path(p).read_bytes() for p in files}
redo();assert w.last_output==str(out/'export')
# Loading a saved session and clearing an observation can both be reversed.
before=annotation_signature(w);w.perform(lambda:w.doc.reverse());settle();u.capture();edited=annotation_signature(w)
w.common.restore_session(path);settle();u.capture();assert annotation_signature(w)==before
undo();assert annotation_signature(w)==edited
# New change after undo invalidates redo.
w.perform(lambda:w.doc.add_tiepoint([65,50],2,'uncertain','new'));settle();u.capture()
assert not w.redo_button.isEnabled()
# Identity failure must leave current documents and history cursor untouched.
w.open_image(1,out/'2.fits');settle();u.capture();current=annotation_signature(w);index=u.cursor
original=(out/'1.fits').read_bytes();(out/'1.fits').write_bytes(b'corrupt')
assert not u.move(-1);assert annotation_signature(w)==current and u.cursor==index
(out/'1.fits').write_bytes(original)
# Display timer groups idle view changes; state is retained through themes.
for theme in ('Dark','Light','Auto'):
 w.theme.setCurrentText(theme);app.processEvents();u.capture()
 w.grab().save(str(out/(theme+'.png')))
for i in range(65):
 w.tie_id.setValue(i+1);u.capture()
assert len(u.entries)<=51 and sum(len(e[1]) for e in u.entries)<=u.max_bytes
# Timed display edits use the same history and can be reversed.
u.timer.start();w.feature.setText('timer captured display setting')
end=time.monotonic()+.7
while time.monotonic()<end:app.processEvents();time.sleep(.02)
u.timer.stop();undo();assert w.feature.text()!='timer captured display setting'
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.close();app.processEvents();assert not u.timer.isActive() and not u.entries
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
