"""Exercise canvas mouse events, including the right-hand navigation lock."""

import os
import subprocess
import sys


def test_native_canvas_clicks_after_zoom_and_pan(tmp_path):
    script = r"""
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from solar_apps.frontends.jet_lab.window import JetLabWindow

out=Path(sys.argv[1]); app=QApplication([])
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=64,CRPIX2=64,CRVAL1=800,CRVAL2=-160,CDELT1=1,CDELT2=1,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):
 hh=h.copy();hh['HGLN_OBS']=i*29;hh['INSTRUME']='AIA' if i==0 else 'EUVI'
 fits.writeto(out/f'{i}.fits',np.ones((128,128)),hh)
def click(pane, xy):
 pane.canvas.draw();app.processEvents()
 x,y=pane.ax.transData.transform(xy);r=pane.canvas.device_pixel_ratio
 pos=QPoint(round(x/r),round((pane.canvas.figure.bbox.height-y)/r))
 QTest.mouseClick(pane.canvas,Qt.MouseButton.LeftButton,pos=pos);app.processEvents()
for theme in ('Light','Dark','Auto'):
 w=JetLabWindow([out]);w.show();app.processEvents()
 w.recovery.timer.stop();w.undo_history.timer.stop()
 for i in range(2):w.open_image(i,out/f'{i}.fits')
 w.theme.setCurrentText(theme);w.start_marking();app.processEvents()
 left,right=w.panes
 click(left,[50,60]); assert len(left.doc.state['tiepoints'])==1
 for nav in ('zoom','pan'):
  # Zoomed viewport is preserved by the return-to-marking action.
  right.ax.set_xlim(30,90);right.ax.set_ylim(30,90)
  limits=(right.ax.get_xlim(),right.ax.get_ylim())
  old=[dict(t) for t in right.doc.state['tiepoints']]
  right.toolbar._actions[nav].trigger();app.processEvents()
  assert right.toolbar.mode and '当前不标点' in right.mark_button.text()
  click(right,[55,60])
  assert right.doc.state['tiepoints']==old
  assert '不会保存对应点' in w.statusBar().currentMessage()
  QTest.mouseClick(right.mark_button,Qt.MouseButton.LeftButton);app.processEvents()
  assert not right.toolbar.mode and '标点模式' in right.mark_button.text()
  assert (right.ax.get_xlim(),right.ax.get_ylim())==limits
  click(right,[55,60]); assert len(right.doc.state['tiepoints'])==1
  original=list(right.doc.state['tiepoints'][0]['pixel_xy'])
  click(right,[62,65]); changed=right.doc.state['tiepoints'][0]['pixel_xy']
  assert len(right.doc.state['tiepoints'])==1 and np.linalg.norm(np.array(changed)-original)>3
  assert w.undo_history.move(-1)
  np.testing.assert_allclose(w.panes[1].doc.state['tiepoints'][0]['pixel_xy'],original)
  assert w.undo_history.move(1)
  assert len(w.panes[0].doc.state['tiepoints'])==1
  left,right=w.panes
 w.can_discard=lambda:True;w.close();assert w.closed_cleanly
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
