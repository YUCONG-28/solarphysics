"""Exercise native-pane redraw caches and unthrottled scientific click positions."""

import os
import subprocess
import sys


def test_native_artists_cache_cursor_and_auxiliary_transitions(tmp_path):
    script = r"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox
from solar_toolkit.map.jet_annotations import JetDocument
from solar_toolkit.map import jet_reconstruction
from solar_apps.frontends.jet_lab import observation_pane as pane_module

out=Path(sys.argv[1]);app=QApplication([])
header=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=64,CRPIX2=64,CRVAL1=800,CRVAL2=-160,CDELT1=1,CDELT2=1,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):
 h=header.copy();h['HGLN_OBS']=i*29
 fits.writeto(out/f'{i}.fits',np.arange(128*128).reshape(128,128).astype(float),h)
w=QMainWindow();w.native_only=True;w.focus_native=lambda i:None;w.start_marking=lambda:None
w.mode=QComboBox();w.mode.addItem('tie','tie')
w.view_layer=QComboBox();w.view_layer.addItems(['original','difference'])
w.stretch=QComboBox();w.stretch.addItems(['Asinh','Linear'])
w.colour=QComboBox();w.colour.addItem('gray');w.active=QComboBox();w.active.addItems(['A','B'])
w.sections={};w.common=SimpleNamespace(native_drafts={});w.reconstruction_current=lambda:False
w.epipolar_anchor=None;w.notice=lambda t:None
w.panes=[pane_module.ObservationPane(w,i) for i in range(2)]
for i,p in enumerate(w.panes):p.doc=JetDocument(out/f'{i}.fits',lazy_segmentation=True);p.draw()
left,right=w.panes
assert 'AIA' in left.canvas.accessibleName() and 'STEREO EUVI' in right.canvas.accessibleName()
image=right.ax.images[0];array=image.get_array()
limits=((25.,90.),(20.,100.));right.ax.set_xlim(limits[0]);right.ax.set_ylim(limits[1])
right.draw();array=image.get_array()
w.epipolar_anchor=(0,left.doc.sha256,[45.,55.])
with patch.object(jet_reconstruction,'epipolar_native_pixels',wraps=jet_reconstruction.epipolar_native_pixels) as epipolar:
 for i in range(6):
  right.doc.state['tiepoints']=[dict(number=1,pixel_xy=[50+i,60])]
  right.draw()
 assert epipolar.call_count==1
 assert right.ax.images[0] is image and image.get_array() is array
 assert (right.ax.get_xlim(),right.ax.get_ylim())==limits
 assert len(right.ax.images)==1 and len(right.ax.lines)==2
 assert not right.doc.segmentation_ready
 right.display_limits=[5.,95.];right.draw()
 assert image.get_array() is not array and epipolar.call_count==1
 w.epipolar_anchor=(0,left.doc.sha256,[46.,55.]);right.draw();assert epipolar.call_count==2
 for n in range(10):
  w.epipolar_anchor=(0,left.doc.sha256,[n+10.,55.]);right.draw()
 assert len(right._epipolar_cache)==8
right.tabs.setCurrentIndex(1);right.draw()
assert right.doc.segmentation_ready and len(right.ax.images)==2
mask=right.ax.images[1];right.draw();assert right.ax.images[1] is mask
right.tabs.setCurrentIndex(0);right.draw();assert not mask.get_visible()
assert right.ax.images[0] is image
original_ties=deepcopy(right.doc.state['tiepoints'])
w.reconstruction_current=lambda:True
w.reconstruction_result={'points':[], 'joint_fits':{'selected_model':'line','line':{'valid':True,'fitted_points':[
 {'number':1,'predicted_pixel_xy':[[40.,50.],[44.,54.]]},
 {'number':2,'predicted_pixel_xy':[[45.,55.],[49.,59.]]}]}}}
right.draw()
fitted=[line for line in right.ax.lines if line.get_color()=='#26d9ef']
assert len(fitted)==1 and fitted[0].get_marker()=='D'
np.testing.assert_array_equal(fitted[0].get_xdata(),[44.,49.])
assert right.doc.state['tiepoints']==original_ties
w.reconstruction_current=lambda:False;right.draw()
assert not any(line.get_color()=='#26d9ef' for line in right.ax.lines)
w.reconstruction_current=lambda:True;w.reconstruction_result['joint_fits']['line']['valid']=False;right.draw()
assert not any(line.get_color()=='#26d9ef' for line in right.ax.lines)
for _ in range(3):
 right.set_mode('rectangle');right.set_mode('tie')
assert not right.ax.patches
# Cursor formatting is capped without quantising or replacing click coordinates.
clock=[10.]
with patch.object(pane_module,'monotonic',side_effect=lambda:clock[0]), patch.object(pane_module,'pixel_hpc',wraps=pane_module.pixel_hpc) as hpc:
 for i in range(120):
  clock[0]=10+i/120;right.format_coordinate(40+i/100,50)
 assert 20 <= hpc.call_count <= 30
 clicks=[];w.mark_native_point=lambda i,xy:clicks.append((i,xy))
 w.perform=lambda fn,index:fn()
 right.press(SimpleNamespace(button=1,inaxes=right.ax,xdata=51.123456,ydata=62.654321))
 assert clicks==[(1,[51.123456,62.654321])]
for p in w.panes:p.cleanup()
w.close();app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
