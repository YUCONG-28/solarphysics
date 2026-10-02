"""Display sampling retains native coordinates, missing data, and zoom detail."""

import os
import subprocess
import sys

import numpy as np

from solar_apps.frontends.jet_lab.display_preview import (
    sample_native_view,
    preview_region_mask,
)


def test_preview_affine_centres_extent_flip_and_native_detail():
    y, x = np.mgrid[:997, :1531]
    data = x + 3.0 * y
    original = data.copy()
    limits = ((10.5, 1420.5), (20.5, 970.5))
    preview = sample_native_view(data, limits, (100, 80))
    assert preview.data.shape == (80, 100)
    assert preview.extent == (10.5, 1420.5, 20.5, 970.5)
    xs = 10.5 + (np.arange(100) + 0.5) * 1410 / 100
    ys = 20.5 + (np.arange(80) + 0.5) * 950 / 80
    np.testing.assert_allclose(preview.data, xs[None, :] + 3 * ys[:, None], rtol=1e-6)
    flipped = sample_native_view(data, (limits[0][::-1], limits[1][::-1]), (100, 80))
    np.testing.assert_array_equal(flipped.data, preview.data)
    assert flipped.extent == preview.extent
    detailed = sample_native_view(data, ((400.5, 500.5), (200.5, 250.5)), (300, 300))
    assert np.shares_memory(detailed.data, data)
    np.testing.assert_array_equal(detailed.data, data[201:251, 401:501])
    np.testing.assert_array_equal(data, original)


def test_missing_stripe_partial_edge_and_mask_alignment():
    data = np.ones((997, 1531))
    data[:, 711] = np.nan
    preview = sample_native_view(data, ((-10, 2000), (-5, 1001)), (37, 43))
    assert preview.extent == (-0.5, 1530.5, -0.5, 996.5)
    assert np.isnan(preview.data).any(axis=1).all()
    shown = preview_region_mask(np.ones(data.shape, dtype=bool), preview)
    np.testing.assert_array_equal(shown, np.isfinite(preview.data))
    detail = sample_native_view(data, ((709.5, 713.5), (0.5, 4.5)), (100, 100))
    assert np.isnan(detail.data[:, 1]).all()
    assert np.isfinite(detail.data[:, [0, 2, 3]]).all()
    outside = sample_native_view(data, ((2000, 2100), (500, 600)), (100, 100))
    assert outside.slices is None and np.isnan(outside.data).all()
    assert not preview_region_mask(np.ones(data.shape, dtype=bool), outside).any()


def test_pane_zoom_refresh_preserves_limits_originals_and_masks(tmp_path):
    script = r"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import sys
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox
from solar_toolkit.map.jet_annotations import JetDocument
from solar_apps.frontends.jet_lab.observation_pane import ObservationPane
out=Path(sys.argv[1]);app=QApplication([])
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=1024,CRPIX2=1024,CRVAL1=0,CRVAL2=0,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
data=np.arange(2048**2,dtype=float).reshape(2048,2048);data[:,450]=np.nan
fits.writeto(out/'large.fits',data,h)
w=QMainWindow();w.native_only=True;w.focus_native=lambda i:None;w.start_marking=lambda:None
for key,items in [('mode',['tie']),('view_layer',['original','difference']),('stretch',['Linear']),('colour',['gray']),('active',['A','B'])]:
 c=QComboBox();[c.addItem(t,t) for t in items];setattr(w,key,c)
w.sections={};w.common=SimpleNamespace(native_drafts={});w.reconstruction_current=lambda:False
w.epipolar_anchor=None;w.notice=lambda t:None
p=ObservationPane(w,0);w.panes=[p];w.setCentralWidget(p);w.resize(650,750);w.show();app.processEvents()
p.doc=JetDocument(out/'large.fits',lazy_segmentation=True);raw=p.doc.raw.copy()
p.doc.state['tiepoints']=[{'number':1,'pixel_xy':[451.123456,481.654321]}]
ties=deepcopy(p.doc.state['tiepoints']);p.draw();app.processEvents();p._refresh_preview()
assert max(p._image_artist.get_array().shape)<=1024
assert p._image_artist.get_extent()==[-.5,2047.5,-.5,2047.5] or tuple(p._image_artist.get_extent())==(-.5,2047.5,-.5,2047.5)
limits=((500.5,400.5),(510.5,460.5))
p.ax.set_xlim(limits[0]);p.ax.set_ylim(limits[1]);p._refresh_preview()
assert p._preview.data.shape==(50,100)
np.testing.assert_array_equal(p._preview.data,raw[461:511,401:501])
assert (p.ax.get_xlim(),p.ax.get_ylim())==limits
assert tuple(p._image_artist.get_extent())==(400.5,500.5,460.5,510.5)
assert np.ma.getmaskarray(p._image_artist.get_array())[:,49].all()
# Region overlay uses identical native edges and retains the missing stripe.
p.doc.segmentation_ready=True;p.doc.selected=np.ones(raw.shape,dtype=bool)
p.tabs.setCurrentIndex(1);p.draw()
assert tuple(p._mask_artist.get_extent())==tuple(p._image_artist.get_extent())
assert (p._mask_artist.get_array()[:,49,3]==0).all()
assert (p.ax.get_xlim(),p.ax.get_ylim())==limits
np.testing.assert_array_equal(p.doc.raw,raw);assert p.doc.state['tiepoints']==ties
# Zoom back out: same bounded display and full native extent, no stale crop.
p.ax.set_xlim(-.5,2047.5);p.ax.set_ylim(-.5,2047.5);p._refresh_preview()
assert max(p._image_artist.get_array().shape)<=1024
assert tuple(p._image_artist.get_extent())==(-.5,2047.5,-.5,2047.5)
# Large constant values must not turn an epsilon-sized span into 0/0.
p.doc.raw=np.full(raw.shape,123456.789);p.tabs.setCurrentIndex(0)
with np.errstate(divide='raise',invalid='raise'):
 p.draw()
assert np.isfinite(p._image_artist.get_array()).all()
assert not np.any(p._image_artist.get_array())
p.cleanup();assert not p._preview_timer.isActive();w.close();app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
