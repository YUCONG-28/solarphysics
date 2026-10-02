"""Display interactions must not edit scientific annotation state."""

import os
import subprocess
import sys


def test_zoom_pan_focus_and_session_layout(tmp_path):
    script = r"""
import sys,time
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication,QMessageBox
from matplotlib.backend_bases import MouseEvent
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab.workflow_guide import annotation_signature
out=Path(sys.argv[1]); app=QApplication([])
y,x=np.mgrid[:120,:160]
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=80,CRPIX2=60,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:30',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):fits.writeto(out/f'{i}.fits',2*(5+80*np.exp(-.5*((y-60)/4)**2))+i,h)
w=JetLabWindow([out],native_only=False); w.show(); w.recovery.timer.stop()
for i in range(2):w.open_image(i,out/f'{i}.fits')
for p in w.panes:
 p.doc.parameters(method='manual',value=25,min_area=1)
 p.doc.select([40,60]);p.doc.trace([40,60]);p.doc.confirm()
 p.doc.add_tiepoint([60,60],1,'manual_confirmed','synthetic')
w.sync_controls();w.redraw();app.processEvents()
before=annotation_signature(w)
assert not w.common.timeline_controls.isVisible()
def wait():
 end=time.monotonic()+20
 while time.monotonic()<end:
  app.processEvents();time.sleep(.02)
  if w.common.future is None and not w.common.debounce.isActive():return
 raise AssertionError(w.common.status.text())
for mode in ('tie','polygon','rectangle','edit'):
 w.mode.setCurrentIndex(w.mode.findData(mode));app.processEvents()
 p=w.panes[0];p.canvas.draw()
 xp,yp=p.ax.transData.transform([60,60])
 e=MouseEvent('scroll_event',p.canvas,xp,yp,button='up',step=1)
 anchor=(e.xdata,e.ydata);old=p.ax.get_xlim()
 p.canvas.callbacks.process('scroll_event',e);p.canvas.draw()
 assert np.ptp(p.ax.get_xlim()) < np.ptp(old)
 assert np.allclose(p.ax.transData.transform(anchor),[xp,yp],atol=1)
 start=p.ax.get_xlim();pixel=p.ax.transData.transform([60,60])
 for name,offset in [('button_press_event',0),('motion_notify_event',15),('button_release_event',15)]:
  p.canvas.callbacks.process(name,MouseEvent(name,p.canvas,pixel[0]+offset,pixel[1],button=2))
 assert not np.allclose(start,p.ax.get_xlim())
 assert annotation_signature(w)==before
 if mode=='polygon':assert not p.selector.verts
 w.fit_images();assert np.allclose(p.ax.get_xlim(),[-.5,159.5])
for index in (0,1):
 w.focus_native(index);app.processEvents()
 assert w.native_focus==index and w.active.currentIndex()==index
 assert w.panes[index].isVisible() and not w.panes[1-index].isVisible()
 w.focus_native(index);app.processEvents();assert all(p.isVisible() for p in w.panes)
w.set_view_mode('split');wait();w.vertical_splitter.setSizes([350,450]);app.processEvents()
sizes=w.vertical_splitter.sizes();w.image_priority.setChecked(True);wait()
assert w.view_mode=='common' and w.guide.isHidden() and not w.parameter_toggle.isChecked()
w.common.operation.setCurrentIndex(2);wait();w.common.canvas.draw()
axes=w.common.figure.axes;pixel=axes[0].transData.transform([60,60])
for name,offset in [('button_press_event',0),('motion_notify_event',12),('button_release_event',12)]:
 w.common.canvas.callbacks.process(name,MouseEvent(name,w.common.canvas,pixel[0]+offset,pixel[1],button=2))
w.common.canvas.callbacks.process('scroll_event',MouseEvent('scroll_event',w.common.canvas,*pixel,button='up',step=1))
assert np.allclose(axes[0].get_xlim(),axes[1].get_xlim())
assert np.allclose(axes[0].get_ylim(),axes[1].get_ylim())
assert all(not selector.verts for selector in w.common.selectors)
assert not w.common.native_drafts and annotation_signature(w)==before
w.fit_images();w.common.operation.setCurrentIndex(0)
w.reset_view_layout();assert w.view_mode=='common'
w.image_priority.setChecked(False);wait()
assert w.view_mode=='split' and not w.guide.isHidden() and w.parameter_toggle.isChecked()
w.focus_native(1);w.parameter_toggle.setChecked(False);w.guide.hide()
w.common.timeline_toggle.setChecked(True)
path=w.common.save_to(out/'session')
w.focus_native(1);w.parameter_toggle.setChecked(True);w.guide.show()
w.active.setCurrentIndex(0)
w.common.timeline_toggle.setChecked(False)
w.common.restore_session(path);wait()
assert w.native_focus==1 and w.active.currentIndex()==1 and not w.panes[0].isVisible()
assert not w.parameter_toggle.isChecked() and w.guide.isHidden()
assert w.common.timeline_toggle.isChecked()
assert annotation_signature(w)==before
for theme in ('Light','Dark','Auto'):
 w.apply_theme(theme)
 for mode in ('native','common','split'):
  w.set_view_mode(mode);wait();w.resize(1280,800);app.processEvents()
  assert w.width()<=1280 and w.height()<=800
  w.grab().save(str(out/f'{theme}_{mode}.png'))
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.close();app.processEvents()
assert all(not p.navigation.connections for p in w.panes)
assert not w.common.navigation.connections
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
