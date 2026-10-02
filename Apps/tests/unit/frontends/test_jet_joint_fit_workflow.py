"""Native joint fitting UI: asynchronous acceptance, staleness and preservation."""

import os
from pathlib import Path
import subprocess
import sys


def test_joint_fit_async_selection_cancel_and_roundtrip(tmp_path):
    science_tests = Path(__file__).resolve().parents[4] / "Python" / "tests"
    script = r"""
import sys,json,time,threading
from pathlib import Path
from copy import deepcopy
import numpy as np
from PyQt6.QtWidgets import QApplication,QPushButton,QMessageBox
from PyQt6.QtTest import QTest
sys.path.insert(0,sys.argv[2])
from test_jet_reconstruction import observations
from solar_apps.frontends.jet_lab.window import JetLabWindow
from solar_apps.frontends.jet_lab import fit_workflow
out=Path(sys.argv[1]);app=QApplication([])
t=np.linspace(0,1,7)
start=np.array([.65,.82,-.2]);end=np.array([.73,.89,-.23]);bend=np.array([-.01,.008,0.])
direction=end-start;bend-=direction*np.dot(bend,direction)/np.dot(direction,direction)
xyz=start+t[:,None]*direction+4*t[:,None]*(1-t[:,None])*bend
docs=observations(out,points=xyz)
w=JetLabWindow([out]);w.show();app.processEvents()
assert not w.common.has_unsaved()  # Opening an empty workspace is not a modification.
for p,d in zip(w.panes,docs):p.doc=d
w.common.current_pair={'status':'matched','tolerance_s':15,
 'AIA':{'image_sha256':docs[0].sha256},'EUVI':{'image_sha256':docs[1].sha256},
 'image_sha256':[d.sha256 for d in docs]}
w.common.remember();w.sync_point_editor();w.redraw()
original=deepcopy([d.state['tiepoints'] for d in docs])
for d in docs:d.dirty=False  # Equivalent to restoring a clean annotations bundle.
assert not w.common.has_unsaved()
questions=[]
def cancel_question(*args,**kwargs):
 questions.append(args[2]);return QMessageBox.StandardButton.Cancel
QMessageBox.question=cancel_question
w.fit_curve.setChecked(True)
assert w.common.has_unsaved_science() and not w.can_discard() and questions
# Isolate the result-only case: options already belong to the saved baseline.
w.common.mark_science_saved();assert not w.common.has_unsaved()
w.start_joint_fit()
def wait_for(predicate):
 deadline=time.monotonic()+15
 while not predicate() and time.monotonic()<deadline:QTest.qWait(10)
 assert predicate(),w.point_hint.text()
wait_for(lambda:w.reconstruction_current())
assert w.reconstruction_dialog.isVisible()
result=w.reconstruction_result
assert result['joint_fits']['line']['valid'] and result['joint_fits']['curve']['valid']
assert result['joint_fits']['selected_model']=='line'
assert [d.state['tiepoints'] for d in docs]==original
assert not any(d.dirty for d in docs)
assert w.common.has_unsaved_science() and not w.can_discard()
w.fit_model.setCurrentIndex(1);w.select_fit_model()
assert result['joint_fits']['selected_model']=='curve'
assert w.undo_history.move(-1) and w.reconstruction_result['joint_fits']['selected_model']=='line'
assert w.undo_history.move(1) and w.reconstruction_result['joint_fits']['selected_model']=='curve'
w.start_joint_fit(robust=True);wait_for(lambda:'joint_robust' in (w.reconstruction_result or {}))
assert w.reconstruction_result['joint_robust']['loss']=='soft_l1'
saved=w.common.save_to(out/'saved')
assert not w.common.has_unsaved()
assert 'joint_fits.json' in json.loads((saved.parent/'COMPLETE.json').read_text())['sha256']
before=deepcopy(w.reconstruction_result)
w.restore(saved);assert w.reconstruction_current() and w.reconstruction_result==before
assert not w.common.has_unsaved() and w.can_discard()
# Selecting a scientific model changes persisted results even when pixels do not.
w.fit_model.setCurrentIndex(1);w.select_fit_model()
assert w.common.has_unsaved_science() and not w.can_discard()
assert w.undo_history.move(-1) and not w.common.has_unsaved()
assert w.undo_history.move(1) and w.common.has_unsaved_science()
w.restore(saved);assert not w.common.has_unsaved()
# Deterministic worker barrier: edit while old scientific work is in flight.
compute=fit_workflow.compute_native_fit;release=threading.Event();entered=threading.Event()
def gated(*args,**kw):
 entered.set();release.wait(5);return compute(*args,**kw)
fit_workflow.compute_native_fit=gated
w.start_joint_fit();wait_for(entered.is_set)
w.tie_id.setValue(2)
old=w.panes[0].doc.state['tiepoints'][1]['pixel_xy']
w.perform(lambda:w.mark_native_point(0,[old[0]+1,old[1]]),0)
release.set();wait_for(lambda:w._fit_future is None)
assert '丢弃过期' in w.point_hint.text() and not w.reconstruction_current()
release.clear();entered.clear();w.start_joint_fit();wait_for(entered.is_set)
next(b for b in w.fit_progress.findChildren(QPushButton) if b.text()=='取消计算').click()
release.set();app.processEvents();assert w._fit_future is None
assert not w.reconstruction_current()
fit_workflow.compute_native_fit=compute
w.can_discard=lambda:True;w.close();assert w.closed_cleanly and w._fit_closed
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(science_tests)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert result.returncode == 0, result.stdout + result.stderr
