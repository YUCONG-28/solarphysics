"""Accessible coordinate editing and explicit window-only screenshot feedback."""

import os
import subprocess
import sys


def test_control_identity_coordinate_actions_and_screenshot(tmp_path):
    script = r"""
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QDialog,QFileDialog,QWidget,QPushButton
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab.accessibility import install_accessibility,identify_widgets,save_window_screenshot
from solar_apps.frontends.jet_lab.coordinate_dialog import NativeCoordinateDialog,edit_native_coordinate
out=Path(sys.argv[1]);app=QApplication([])
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=32,CRPIX2=24,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):fits.writeto(out/f'{i}.fits',np.ones((48,64))*10+i,h)
w=JetLabWindow([out]);w.show()
manager=install_accessibility(w);assert install_accessibility(w) is manager
for i in range(2):w.open_image(i,out/f'{i}.fits')
app.processEvents();identify_widgets(w)
widgets=[w,*w.findChildren(QWidget)]
identifiers=[obj.accessibleIdentifier() for obj in widgets]
assert all(identifiers) and len(set(identifiers))==len(identifiers)
before={id(obj):(obj.objectName(),obj.accessibleIdentifier()) for obj in widgets}
identify_widgets(w)
assert before=={id(obj):(obj.objectName(),obj.accessibleIdentifier()) for obj in widgets}
# Dynamic dialogs receive IDs too, and an existing Qt objectName is retained.
d=QDialog(w);b=QPushButton('dynamic',d);b.setObjectName('keep_this_name');d.show();app.processEvents()
assert b.objectName()=='keep_this_name' and b.accessibleIdentifier()
d.close();d.deleteLater();app.processEvents()
for i,xy in enumerate(([11.25,12.5],[21.75,22.125])):
 w.perform(lambda i=i,xy=xy:w.mark_native_point(i,xy),i)
dialog=NativeCoordinateDialog(w)
assert dialog.coordinate_values()==[21.75,22.125]
# Programmatic/AX state changes need not send a clicked signal.
dialog.side_buttons[0].setChecked(True)
assert dialog.index==0 and dialog.coordinate_values()==[11.25,12.5]
assert dialog.x.accessibleIdentifier()!=dialog.y.accessibleIdentifier()
assert dialog.x.accessibleName()=='原生 x 数值输入'
assert dialog.y.accessibleName()=='原生 y 数值输入'
for invalid in ('nan','64','-1','1,2',''):
 dialog.x.setText(invalid)
 assert dialog.coordinate_values() is None and not dialog.apply_button.isEnabled()
dialog.x.setText('11.25');assert dialog.apply_button.isEnabled()
dialog.deleteLater();app.processEvents()
def accept():
 d=QApplication.activeModalWidget();assert isinstance(d,NativeCoordinateDialog)
 d.side_buttons[1].setChecked(True);d.x.setText('24.5');d.y.setText('25.75');d.apply_button.click()
QTimer.singleShot(0,accept)
assert edit_native_coordinate(w)
assert w.panes[1].doc.state['tiepoints'][0]['pixel_xy']==[24.5,25.75]
w.undo_history.capture();assert w.undo_history.move(-1)
assert w.panes[1].doc.state['tiepoints'][0]['pixel_xy']==[21.75,22.125]
assert w.undo_history.move(1)
assert w.panes[1].doc.state['tiepoints'][0]['pixel_xy']==[24.5,25.75]
def cancel():
 d=QApplication.activeModalWidget();assert d.coordinate_values()==[24.5,25.75]
 d.x.setText('1');d.cancel_button.click()
before=[deepcopy(p.doc.state) for p in w.panes]
QTimer.singleShot(0,cancel);assert not edit_native_coordinate(w)
assert [p.doc.state for p in w.panes]==before
QFileDialog.getSaveFileName=lambda *a,**k:(str(out/'window.png'),'PNG (*.png)')
assert save_window_screenshot(w,w)==out/'window.png'
assert (out/'window.png').read_bytes().startswith(b'\x89PNG')
assert [p.doc.state for p in w.panes]==before
outside=out.parent/'forbidden.png'
QFileDialog.getSaveFileName=lambda *a,**k:(str(outside),'PNG (*.png)')
assert save_window_screenshot(w,w) is None and not outside.exists()
assert '截图未保存' in w.statusBar().currentMessage()
assert any(b.text()=='保存窗口截图' for b in w.findChildren(QPushButton))
w.can_discard=lambda:True;w.close();app.processEvents()
assert manager.closed and manager.owner is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
