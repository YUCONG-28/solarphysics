"""Native IO commits, recovery drafts and interaction-driven history contracts."""

import os
import subprocess
import sys


_SETUP = r"""
import json, sys, time, shutil, threading
from pathlib import Path
from copy import deepcopy
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import QApplication, QMessageBox
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_toolkit.map.jet_annotations import file_sha256
out=Path(sys.argv[1]); app=QApplication([])
header=fits.Header(dict(CTYPE1='HPLN-TAN',CTYPE2='HPLT-TAN',CUNIT1='arcsec',CUNIT2='arcsec',
 CRPIX1=24,CRPIX2=20,CRVAL1=800,CRVAL2=-170,CDELT1=.6,CDELT2=.6,EXPTIME=2,
 DATE_OBS='2000-01-01T12:00:00',DSUN_OBS=149600000000.0,HGLN_OBS=0,HGLT_OBS=-5.0,
 RSUN_REF=695700000,TELESCOP='SDO/AIA',INSTRUME='AIA',WAVELNTH=304,WAVEUNIT='angstrom',BUNIT='DN',SYNTHET=True))
for i in range(2):
 h=header.copy();h['DATE_OBS']=f'2000-01-01T12:00:{i*12:02d}'
 fits.writeto(out/f'{i}.fits',np.ones((40,48))*10+i,h)
w=JetLabWindow([out]); c=w.common; w.show()
def drain(seconds=.3):
 end=time.monotonic()+seconds
 while time.monotonic()<end: app.processEvents();time.sleep(.01)
def wait():
 end=time.monotonic()+12
 while time.monotonic()<end:
  drain(.02)
  if c.future is None and not c.debounce.isActive(): return
 raise AssertionError(c.status.text())
"""

_CLEANUP = r"""
QMessageBox.question=lambda *a,**k:QMessageBox.StandardButton.Discard
w.ensure_point_metadata_applied=lambda *a:True
w.close();app.processEvents();assert c.closed
"""


def _run(script, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _SETUP + script + _CLEANUP, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_controller_async_io_and_cancel_do_not_build_shared_view(tmp_path):
    _run(
        r"""
assert 'solar_apps.frontends.jet_lab.common_view' not in sys.modules
assert not hasattr(c,'figure') and not hasattr(c,'canvas') and not c.cache
c.open_image_async(0,out/'0.fits');wait()
assert w.panes[0].doc and not w.panes[0].doc.segmentation_ready
original=w.panes[0].doc
# Check the full list and both frames before replacing any of the current sample.
bad=dict(schema='solarphysics.jet_lab.samples',version=1,pairs=[dict(id='bad',split='tuning',
 views=[dict(image='0.fits',image_sha256=file_sha256(out/'0.fits')),
        dict(image='1.fits',image_sha256='wrong')])])
(out/'samples.json').write_text(json.dumps(bad))
c.load_manifest_async(out/'samples.json');wait()
assert w.panes[0].doc is original and w.manifest is None
assert '失败' in c.status.text()
bad['pairs'][0]['views'][1]['image_sha256']=file_sha256(out/'1.fits')
(out/'samples.json').write_text(json.dumps(bad))
c.load_manifest_async(out/'samples.json');wait()
assert all(p.doc for p in w.panes) and w.sample_id=='bad'
assert all(not p.doc.segmentation_ready for p in w.panes)
# Completed asynchronous frame changes are one coherent undo step: original
# images, displayed pairing and slider index are restored together.
first,second=[p.doc for p in w.panes]
pair0=dict(AIA=dict(image_sha256=first.sha256),EUVI=dict(image_sha256=second.sha256),status='manual_pair')
pair1=dict(AIA=dict(image_sha256=second.sha256),EUVI=dict(image_sha256=first.sha256),status='manual_pair')
c.paired=[pair0,pair1];c.current_pair=pair0;c.slider.setRange(0,1)
drain(.6);w.undo_history.capture('paired originals')
c.step(1);wait();assert c.slider.value()==1 and w.panes[0].doc.sha256==second.sha256
w.undo_history.capture();assert w.undo_history.move(-1)
assert c.slider.value()==0 and w.panes[0].doc.sha256==first.sha256
assert w.undo_history.move(1)
assert c.slider.value()==1 and w.panes[0].doc.sha256==second.sha256
gate=threading.Event();started=threading.Event();applied=[]
def old(cancelled):
 started.set();gate.wait(5);return 'old'
c.run_load(old,applied.append,'old')
assert started.wait(2)
c.run_load(lambda cancelled:'new',applied.append,'new')
gate.set();wait();assert applied==['new']
# A pending loading step can be cancelled without touching annotations.
gate.clear();started.clear();c.run_load(old,applied.append,'cancel')
assert started.wait(2)
before=[deepcopy(p.doc.state) for p in w.panes]
c.cancel_pending();gate.set();drain(.2)
assert applied==['new'] and [p.doc.state for p in w.panes]==before
assert not c.poll.isActive() and not c.debounce.isActive()
# A user can keep marking while a read is in flight. Its completion must not
# replace those edits, even if the file read itself succeeded.
gate.clear();started.clear();c.run_load(old,applied.append,'changed')
assert started.wait(2)
w.panes[0].doc.add_tiepoint([19,18],1,'possible','edit while reading')
gate.set();wait();assert applied==['new']
assert '标注已修改' in c.status.text()
assert 'solar_apps.frontends.jet_lab.common_view' not in sys.modules
""",
        tmp_path,
    )


def test_session_restore_and_pending_recovery_are_transactional(tmp_path):
    _run(
        r"""
for i in range(2):w.open_image(i,out/f'{i}.fits')
for pane in w.panes:pane.doc.add_tiepoint([20,20],1,'possible','baseline')
w.sync_point_editor()
w.fit_options={'include_curve':True}
# A real second-document disk write failure must roll back every dirty/save flag.
import solar_apps.frontends.jet_lab.session_io as storage
real_save=storage.save_session;calls=[]
def fail_second(*args,**kwargs):
 calls.append(args[0])
 if len(calls)==2:raise OSError('second document write failed')
 return real_save(*args,**kwargs)
storage.save_session=fail_second
w.last_saved=False;w.last_output='previous saved location';w.saved_signature='previous saved signature'
before_state=[deepcopy(p.doc.state) for p in w.panes]
try:
 try:c.save_to(out/'partial')
 except OSError:pass
 else:raise AssertionError('save failure swallowed')
finally:storage.save_session=real_save
assert len(calls)==2 and all(p.doc.dirty for p in w.panes)
assert [p.doc.state for p in w.panes]==before_state
assert w.last_saved is False and w.last_output=='previous saved location'
assert w.saved_signature=='previous saved signature' and not (out/'partial/COMPLETE.json').exists()
path=c.save_to(out/'session')
previous_docs=[p.doc for p in w.panes]
previous=[deepcopy(p.doc.state) for p in w.panes]
# A ROI error found after all originals have loaded cannot half-restore the UI.
v=json.loads((path.parent/'view_state.json').read_text())
v['native_roi_drafts']={previous_docs[0].sha256:dict(shape=[40,48],runs=[[0,999999]],evidence={})}
(path.parent/'view_state.json').write_text(json.dumps(v))
check=path.parent/'COMPLETE.json'; complete=json.loads(check.read_text())
complete['sha256']['view_state.json']=file_sha256(path.parent/'view_state.json')
check.write_text(json.dumps(complete))
try:c.restore_session(path)
except ValueError as exc:assert 'ROI' in str(exc)
else:raise AssertionError('malformed restore accepted')
assert [p.doc for p in w.panes]==previous_docs and [p.doc.state for p in w.panes]==previous
# Restore drafts keep unapplied UI values, while preserving committed annotation.
c.current_pair=dict(AIA=dict(image_sha256=previous_docs[0].sha256),
 EUVI=dict(image_sha256=previous_docs[1].sha256),status='manual_pair',delta_emission_s=None)
w.feature.setText('not applied yet');assert w.point_metadata_pending()
for pane in w.panes:pane.doc.dirty=False
w.recovery.directory=out/'recovery';w.recovery.path=w.recovery.directory/'pending.json'
payload=w.recovery.payload();assert payload['point_editor']['values']['feature']=='not applied yet'
assert payload['current_pair']['AIA']['image_sha256']==previous_docs[0].sha256
draft=w.recovery.write(payload)
w.feature.setText('temporary');w.ensure_point_metadata_applied=lambda *a:True
w.recovery.restore(draft)
assert w.feature.text()=='not applied yet' and w.point_metadata_pending()
assert w.panes[0].doc.state['tiepoints'][0]['feature']=='baseline'
assert c.current_pair['status']=='manual_pair' and w.fit_options['include_curve']
# Invalid editor metadata must fail without changing current objects or values.
payload['point_editor']['values']['role']='invalid'
w.recovery.path=w.recovery.directory/'invalid.json';invalid=w.recovery.write(payload)
before=[p.doc for p in w.panes]
try:w.recovery.restore(invalid)
except ValueError as exc:assert '说明' in str(exc)
else:raise AssertionError('malformed metadata accepted')
assert [p.doc for p in w.panes]==before and w.feature.text()=='not applied yet'
""",
        tmp_path,
    )


def test_formal_restore_labels_use_image_identity_and_restore_current_point(tmp_path):
    _run(
        r"""
for i in range(2): w.open_image(i,out/f'{i}.fits')
hashes=[p.doc.sha256 for p in w.panes]
for pane in w.panes:
 pane.doc.add_tiepoint([20,20],3,'possible','restore selection test')
w.tie_id.setValue(3);w.active.setCurrentIndex(1);w.sync_point_editor()
# The startup combo selection is deliberately different from the saved pair.
samples=[dict(id='startup',views=[dict(image_sha256=h) for h in reversed(hashes)]),
         dict(id='restored',views=[dict(image_sha256=h) for h in hashes])]
w.manifest=(out,dict(pairs=samples));w.pairs.addItems(['startup','restored'])
w.sample_id='restored'
path=c.save_to(out/'selected_session')
view=json.loads((path.parent/'view_state.json').read_text())
assert view['current_point_number']==3 and view['active_editor_side']==1
w.tie_id.setValue(1);w.active.setCurrentIndex(0);w.pairs.setCurrentIndex(0)
c.restore_session(path)
assert w.pairs.currentIndex()==1 and w.loaded_pair_index==1
assert w.pairs.currentText()=='restored' and w.sample_id=='restored'
assert w.tie_id.value()==3 and w.active.currentIndex()==1
assert w._point_editor_number==3 and not w.point_metadata_pending()
assert all(p.doc.state['tiepoints'][0]['number']==3 for p in w.panes)
# A restored pair absent from the current manifest cannot retain a stale label.
w.manifest=(out,dict(pairs=samples[:1]));w.pairs.clear();w.pairs.addItem('startup')
c.restore_session(path)
assert w.pairs.currentIndex()==-1 and w.pairs.currentText()=='' and w.loaded_pair_index==-1
from solar_apps.frontends.jet_lab.native_frames import matching_sample_index
assert matching_sample_index((out,dict(pairs=[samples[1],samples[1]])),[p.doc for p in w.panes])==-1
""",
        tmp_path,
    )


def test_idle_does_not_snapshot_and_event_window_is_configurable(tmp_path):
    _run(
        r"""
for i in range(2):w.open_image(i,out/f'{i}.fits')
drain(1.2)
counts={'undo':0,'recovery':0}
old_snapshot=w.undo_history.snapshot;old_payload=w.recovery.payload
def snapshot():counts['undo']+=1;return old_snapshot()
def payload():counts['recovery']+=1;return old_payload()
w.undo_history.snapshot=snapshot;w.recovery.payload=payload
drain(1.1);assert counts=={'undo':0,'recovery':0},counts
# An actual view adjustment makes one coalesced history capture.
w.low.setValue(2);drain(.8);assert counts['undo']>=1
from solar_apps.frontends.jet_lab.session_validation import validate_event_profile
for profile in ({'bands':[0]}, {'bands':[304,304]},
                {'start_utc':'2000-01-03','end_utc':'2000-01-02'}):
 try:validate_event_profile(profile)
 except ValueError:pass
 else:raise AssertionError(profile)
c.event_profile={'bands':[304],'start_utc':'2000-01-02T12:00:10','end_utc':'2000-01-02T12:00:40'}
records=[]
for name in ('AIA','EUVI'):
 for second in (0,20,40,60):
  records.append(dict(instrument=name,band=304,midpoint_utc=f'2000-01-02T12:00:{second:02d}' if second<60 else '2000-01-02T12:01:00',
    image_sha256=name+str(second),dsun_m=149600000000.0))
c.records=records;c.background.blockSignals(True);c.background.setChecked(False);c.background.blockSignals(False)
c.band.setCurrentText('304');c.rebuild_pairs(select=False)
assert len(c.paired)==2
before=c.records
try:c.set_timeline_records([dict(band=171)])
except ValueError:pass
else:raise AssertionError('incompatible event bands accepted')
assert c.records is before
""",
        tmp_path,
    )
