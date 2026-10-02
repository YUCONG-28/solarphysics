"""Native metadata protection and descriptive axis rendering."""

import os
import subprocess
import sys


def test_metadata_apply_discard_cancel_and_undo(tmp_path):
    script = r"""
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
out=Path(sys.argv[1]); app=QApplication([])
h=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=64,CRPIX2=64,CRVAL1=800,CRVAL2=-160,CDELT1=1,CDELT2=1,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):
 hh=h.copy();hh['HGLN_OBS']=i*29
 fits.writeto(out/f'{i}.fits',np.ones((128,128)),hh)
w=JetLabWindow([out]);w.show();app.processEvents()
w.undo_history.timer.stop();w.recovery.timer.stop()
for i in range(2):w.open_image(i,out/f'{i}.fits')
for i in range(2):w.perform(lambda i=i:w.mark_native_point(i,[65,60]),i)
assert not w.point_metadata_pending()
old_pixels=[list(p.doc.state['tiepoints'][0]['pixel_xy']) for p in w.panes]
w.point_role.setCurrentIndex(w.point_role.findData('reference'))
assert w.point_metadata_pending() and '尚未应用' in w.metadata_pending_label.text()
# Exercise the actual Qt dialog buttons, separately from its caller actions.
for label, answer in [('应用后继续','apply'),('放弃说明修改','discard'),('返回编辑','return')]:
 def click_choice(label=label):
  dialog=QApplication.activeModalWidget()
  assert isinstance(dialog,QMessageBox)
  next(b for b in dialog.buttons() if b.text()==label).click()
 QTimer.singleShot(0,click_choice)
 assert w.point_metadata_choice('测试')==answer
w.point_metadata_choice=lambda action:'return'
w.tie_id.setValue(2)
assert w.tie_id.value()==1 and w.point_metadata_pending()
assert w.calculate_3d() is None and w.reconstruction_result is None
assert w.common.save_to(out/'cancelled') is None and not (out/'cancelled').exists()
w.point_metadata_choice=lambda action:'discard'
w.tie_id.setValue(2)
assert w.tie_id.value()==2 and not w.point_metadata_pending()
assert all(p.doc.state['tiepoints'][0]['role']=='jet' for p in w.panes)
w.tie_id.setValue(1)
w.point_role.setCurrentIndex(w.point_role.findData('reference'))
w.undo_history.capture('修改点用途')
w.point_metadata_choice=lambda action:'apply'
assert w.ensure_point_metadata_applied('测试')
assert all(p.doc.state['tiepoints'][0]['role']=='reference' for p in w.panes)
assert not w.point_metadata_pending()
assert w.undo_history.move(-1)
assert all(p.doc.state['tiepoints'][0]['role']=='jet' for p in w.panes)
assert w.point_metadata_pending() and w.point_role.currentData()=='reference'
assert w.undo_history.move(1) and not w.point_metadata_pending()
assert [p.doc.state['tiepoints'][0]['pixel_xy'] for p in w.panes]==old_pixels
w.feature.setText('A bright knot')
w.point_metadata_choice=lambda action:'apply'
w.tie_id.setValue(2)
assert all(p.doc.state['tiepoints'][0]['feature']=='A bright knot' for p in w.panes)
for theme in ('Light','Dark','Auto'):
 w.theme.setCurrentText(theme); app.processEvents()
 assert not w.point_metadata_pending()
w.can_discard=lambda:True;w.close();assert w.closed_cleanly
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_axis_quality_presentation_and_no_gap_bridge():
    script = r"""
import numpy as np
from matplotlib.figure import Figure
from solar_apps.frontends.jet_lab.reconstruction_view import reconstruction_quality,axis_description,draw_reconstruction
rows=[dict(number=n,xyz_Rsun=[1.1+0.1*n,0.01*n,0],role='jet',numerical_valid=True,
 admissible_candidate=True,below_photosphere=False,visible=[True,True],
 identity_status=['manual_confirmed','manual_confirmed'],
 input_correspondences=[{},{}],issues=[]) for n in (1,2,4,5)]
axis=dict(valid=True,directed=False,used_point_numbers=[1,2,4,5],centroid_xyz_Rsun=[1.4,0.03,0],
 direction_xyz=[1,0,0],radial_unit_xyz=[0,1,0],unsigned_radial_angle_deg=90,rms_perpendicular_Mm=2.3)
report=dict(points=rows,issues=['time_pairing_unverified'],provenance={'reference_frame':'HeliographicStonyhurst'},
 summary={'axis':axis,'polyline_segments':[{'xyz_Rsun':[r['xyz_Rsun'] for r in rows[:2]]},
                                       {'xyz_Rsun':[r['xyz_Rsun'] for r in rows[2:]]}]})
text='\n'.join(reconstruction_quality(report))
for label in ('数值求解','物理条件','身份判断','时间配对','定位误差'):
 assert label in text
assert '尚未验证' in text and '0/4' in text
assert '方向未定' in axis_description(axis) and '不是定位误差' in axis_description(axis)
figure=Figure();ax=figure.add_subplot(111,projection='3d');draw_reconstruction(ax,report)
orange=[l for l in ax.lines if l.get_color()=='orange']
assert len(orange)==2 and all(len(l.get_data_3d()[0])==2 for l in orange)
assert any(l.get_label()=='PCA axis' for l in ax.lines)
# All candidates and the radial arrow tip fit inside an equal-scale cube.
# The arrow extends beyond the data's narrow y range in this fixture.
limits=np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
radial_tip=np.array(axis['centroid_xyz_Rsun'])+0.4*0.4*np.array(axis['radial_unit_xyz'])
shown=np.vstack([[r['xyz_Rsun'] for r in rows],radial_tip])
assert np.all(shown.min(axis=0)>limits[:,0]) and np.all(shown.max(axis=0)<limits[:,1])
np.testing.assert_allclose(np.diff(limits,axis=1).ravel(),np.repeat(limits[0,1]-limits[0,0],3))
np.testing.assert_allclose(ax.get_box_aspect(),np.repeat(ax.get_box_aspect()[0],3))
assert all(len(a.get_major_locator().tick_values(*limit))<=6
           for a,limit in zip((ax.xaxis,ax.yaxis,ax.zaxis),limits))
axis['directed']=True;axis['radial_angle_deg']=170
assert '170.00' in axis_description(axis)
# Legacy outputs omit segment evidence: preserve points, do not reconnect them.
report['summary']={'polyline_xyz_Rsun':[r['xyz_Rsun'] for r in rows]}
ax.clear();draw_reconstruction(ax,report)
assert not any(l.get_color()=='orange' for l in ax.lines)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
