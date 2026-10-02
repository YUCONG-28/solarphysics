"""The simplified guide follows native correspondences, not segmentation steps."""

import os
import subprocess
import sys


def test_native_guide_needs_no_mask_or_axis():
    script = r"""
from types import SimpleNamespace
from solar_apps.frontends.jet_lab.workflow_guide import annotation_signature, workflow_status, readable_reason
documents = [
    SimpleNamespace(sha256=str(i), info={}, state={"tiepoints": [{"number": 1}], "axis": []})
    for i in range(2)
]
owner = SimpleNamespace(
    native_only=True, panes=[SimpleNamespace(doc=d) for d in documents],
    reconstruction_result=None, reconstruction_current=lambda: False,
    common=SimpleNamespace(has_unsaved=lambda: True), saved_signature=None, last_saved=False,
)
done, details = workflow_status(owner)
assert done == [True, True, False, False, False]
assert "1 组" in details[1]
documents[1].state["tiepoints"].append({"number": 2})
assert not workflow_status(owner)[0][1]
assert "缺少编号：2" in workflow_status(owner)[1][1]
documents[1].state["tiepoints"].pop()
owner.reconstruction_result = {"points": [{"number": 1, "numerical_valid": True}]}
owner.reconstruction_current = lambda: True
owner.saved_signature = annotation_signature(owner)
owner.last_saved = True
owner.common.has_unsaved = lambda: False
assert workflow_status(owner)[0] == [True] * 5
owner.reconstruction_current = lambda: False
assert "过期" in workflow_status(owner)[1][2]
for reason in ("time_pairing_unverified", "explicit_unique_endpoints_required",
               "pair_or_input_conditions_unresolved", "below_photosphere",
               "time_pairing_ambiguous_nearest", "source_time_outside_tolerance",
               "role_conflict", "order_conflict", "endpoint_conflict",
               "unknown_roles_excluded", "solar_occultation", "no_correspondences",
               "reprojection_exceeds_3sigma", "invalid_native_coordinates"):
    assert not readable_reason(reason).startswith("需要核查："), reason
assert readable_reason("view_1:missing_or_invalid_hgln_obs").startswith("右侧：缺少")
assert "光球以下" in readable_reason("below_photosphere")
assert "尚未" in readable_reason("time_pairing_unverified")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
