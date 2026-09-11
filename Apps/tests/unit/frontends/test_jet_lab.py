"""Exercise the actual Qt/Matplotlib controls in a clean Qt6 process."""

import os
import subprocess
import sys


def test_jet_lab_ui_roundtrip_themes_and_cleanup(tmp_path):
    script = r"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication, QMessageBox
from matplotlib.backend_bases import MouseEvent
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_toolkit.map.jet_annotations import save_session
out=Path(sys.argv[1]); yy,xx=np.mgrid[:96,:128]
header=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=64,CRPIX2=48,CRVAL1=850,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2025-01-24T04:48:30',DSUN_OBS=147298498065.7,HGLN_OBS=0,HGLT_OBS=-5.7,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=171,WAVEUNIT='angstrom',BUNIT='DN'))
fits.writeto(out/'test.fits',2*(5+80*np.exp(-.5*((yy-48)/3)**2)),header)
app=QApplication([]);window=JetLabWindow([out]);window.show();app.processEvents()
window.open_image(0,out/'test.fits');window.open_image(1,out/'test.fits')
window.active.setCurrentIndex(0);doc=window.doc
window.perform(lambda:doc.parameters(method='manual',value=25,min_area=1))
pane=window.panes[0]
window.mode.setCurrentIndex(window.mode.findData('rectangle'));app.processEvents();pane.canvas.draw()
for event_name,xy in [('button_press_event',[10,40]),('motion_notify_event',[115,57]),('button_release_event',[115,57])]:
 xp,yp=pane.ax.transData.transform(xy)
 pane.canvas.callbacks.process(event_name,MouseEvent(event_name,pane.canvas,xp,yp,button=1))
assert doc.state['roi'] is not None and len(doc.state['roi'])==4
def click(mode,xy,event_name='button_press_event'):
 window.mode.setCurrentIndex(window.mode.findData(mode));app.processEvents();pane.canvas.draw()
 xp,yp=pane.ax.transData.transform(xy)
 event=MouseEvent(event_name,pane.canvas,xp,yp,button=1)
 pane.canvas.callbacks.process(event_name,event)
click('select',[20,48]);assert doc.selected is not None
click('inner',[20,48]);assert doc.state['axis_status']=='pending' and len(doc.paths)
window.perform(doc.confirm);assert doc.state['axis_status']=='automatic_confirmed'
before=np.array(doc.state['axis']);original=doc.raw.copy()
endpoint=before[0]
click('edit',endpoint)
click('edit',endpoint+[.15,.15],'button_release_event')
assert doc.state['axis_status']=='manual_corrected'
window.perform(doc.undo);assert np.array_equal(before,doc.state['axis'])
pane.ax.set_xlim(10,80);pane.ax.set_ylim(40,60)
click('tie',[31.25,48.125]);point=doc.state['tiepoints'][0]
assert np.max(abs(np.array(point['pixel_xy'])-[31.25,48.125]))<.1
assert not window.panes[1].doc.state['tiepoints']
window.perform(doc.reverse);window.perform(doc.undo);assert np.array_equal(before,doc.state['axis'])
window.low.setValue(5);window.colour.setCurrentText('inferno');app.processEvents()
assert np.array_equal(original,doc.raw)
window.perform(window.measure);assert len(window.sections[0])==10
window.perform(window.show_profiles);assert window.profile_dialog.isVisible()
path=save_session(out/'annotation',[doc,window.panes[1].doc]);window.restore(path)
assert np.array_equal(before,window.doc.state['axis'])
# Pending parameter edits must stay with their original view when switching.
window.value.setValue(26);window.active.setCurrentIndex(1)
assert window.panes[0].doc.state['parameters']['value']==26
assert window.panes[1].doc.state['parameters']['value']==85
window.active.setCurrentIndex(0);window.perform(window.doc.undo)
for theme in ['Light','Dark','Auto']:
 window.apply_theme(theme);app.processEvents();window.grab().save(str(out/(theme+'.png')))
window.doc.checkpoint('dirty')
original_question=QMessageBox.question
QMessageBox.question=lambda *args,**kwargs: QMessageBox.StandardButton.Cancel
window.close();assert not window.closed_cleanly
QMessageBox.question=lambda *args,**kwargs: QMessageBox.StandardButton.Discard
window.close();assert window.closed_cleanly and not window.timer.isActive()
assert all(not p.connections and p.selector is None for p in window.panes)
QMessageBox.question=original_question
print('UI acceptance passed')
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "qtagg"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_jet_lab_help_import_safe():
    result = subprocess.run(
        [sys.executable, "-m", "solar_apps.frontends.jet_lab.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and "--allowed-roots" in result.stdout
