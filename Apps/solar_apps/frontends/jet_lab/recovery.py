"""Atomic private recovery drafts; never a substitute for a scientific export."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from PyQt6.QtCore import QTimer, QSignalBlocker
from solar_apps.platform.layout import RuntimeLayout
from solar_toolkit.map.jet_annotations import json_safe


class Recovery:
    def __init__(self, owner, directory=None):
        self.owner = owner
        self.directory = Path(
            directory or RuntimeLayout.discover().state_dir / "jet_lab_recovery"
        )
        self.path = self.directory / (uuid4().hex + ".json")
        self.signature = None
        self.saved_signature = None
        self.changed_at = self.saved_at = time.monotonic()
        self.timer = QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)

    def notify_changed(self):
        """Schedule a draft only after an actual interaction or mutation."""
        self.timer.start()

    def payload(self):
        c = self.owner.common
        c.remember()
        docs = deepcopy(c.document_snapshots)
        for key, d in c.documents.items():
            docs[key] = dict(
                path=str(d.path),
                difference=str(d.difference_path) if d.difference_path else None,
                difference_sha256=d.difference_sha256,
                state=deepcopy(d.state),
                history=deepcopy(d.history),
                undo=deepcopy(d._undo),
                dirty=d.dirty,
            )
        editor = None
        if hasattr(self.owner, "point_editor_values") and hasattr(
            self.owner, "point_role"
        ):
            editor = dict(
                number=self.owner.tie_id.value(),
                values=self.owner.point_editor_values(),
                baseline=deepcopy(getattr(self.owner, "_point_editor_baseline", None)),
            )
        return dict(
            schema="jet_lab.recovery",
            version=2,
            documents=docs,
            active=[p.doc.sha256 if p.doc else None for p in self.owner.panes],
            sample_id=self.owner.sample_id,
            frames=deepcopy(c.records),
            current_pair=c.pair_metadata(),
            point_editor=editor,
            event_profile=deepcopy(getattr(c, "event_profile", {})),
            fit_options=deepcopy(getattr(self.owner, "fit_options", {})),
        )

    def poll(self):
        try:
            c = self.owner.common
            c.remember()
            pending = (
                hasattr(self.owner, "point_role")
                and self.owner.point_metadata_pending()
            )
            if not pending and not c.has_unsaved():
                return
            payload = self.payload()
            text = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(text.encode()).hexdigest()
            now = time.monotonic()
            if digest != self.signature:
                self.signature, self.changed_at = digest, now
            if digest != self.saved_signature and (
                now - self.changed_at >= 2 or now - self.saved_at >= 30
            ):
                self.write(payload)
                self.saved_signature, self.saved_at = digest, now
            elif digest != self.saved_signature:
                self.timer.start()
        except (OSError, ValueError) as exc:
            self.owner.statusBar().showMessage(f"恢复草稿未保存：{exc}")

    def write(self, payload=None):
        payload = self.payload() if payload is None else payload
        self.directory.mkdir(parents=True, exist_ok=True)
        body = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True)
        envelope = dict(
            payload=payload, sha256=hashlib.sha256(body.encode()).hexdigest()
        )
        temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as f:
                temp = Path(f.name)
                json.dump(json_safe(envelope), f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
        finally:
            if temp and temp.exists():
                temp.unlink()
        self.owner.statusBar().showMessage(
            "恢复草稿已保存；正式标注仍请另存新修订", 4000
        )
        return self.path

    def restore(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("恢复草稿"):
            return
        self.apply_prepared(self.prepare_restore(path))

    def prepare_restore(self, path, cancelled=lambda: False):
        """Validate/read all recovery inputs without touching the live session."""
        from .session_validation import checked_hash

        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("恢复文件必须来自本地草稿目录")
        record = json.loads(path.read_text())
        body = json.dumps(record["payload"], ensure_ascii=False, sort_keys=True)
        if hashlib.sha256(body.encode()).hexdigest() != record["sha256"]:
            raise ValueError("草稿不完整或校验失败")
        payload = record["payload"]
        if payload.get("schema") != "jet_lab.recovery" or payload.get(
            "version"
        ) not in (1, 2):
            raise ValueError("不支持的恢复草稿格式")
        docs = {}
        active = payload["active"]
        for digest, snap in payload["documents"].items():
            if cancelled():
                raise InterruptedError("草稿读取已取消")
            p = self.owner.validate(snap["path"])
            if checked_hash(p, cancelled) != digest:
                raise ValueError("草稿原图身份已改变；不会移植标注")
            if snap.get("difference"):
                p = self.owner.validate(snap["difference"])
                if (
                    not snap.get("difference_sha256")
                    or checked_hash(p, cancelled) != snap["difference_sha256"]
                ):
                    raise ValueError("差分身份无法验证")
            if digest in active:
                docs[digest] = self.owner.common.thaw(snap)
        if len(active) != 2 or any(d is not None and d not in docs for d in active):
            raise ValueError("恢复草稿缺少当前图像")
        frames = payload.get("frames", [])
        from .session_validation import (
            validate_event_profile,
            validate_frame_metadata,
            validate_fit_options,
        )

        validate_fit_options(payload.get("fit_options", {}))
        for frame in frames:
            validate_frame_metadata(frame)
            p = self.owner.validate(frame["_path"])
            if checked_hash(p, cancelled) != frame["image_sha256"]:
                raise ValueError("时序原图身份不一致")
            if frame.get("_difference"):
                p = self.owner.validate(frame["_difference"])
                if checked_hash(p, cancelled) != frame.get("difference_sha256"):
                    raise ValueError("时序差分身份不一致")
        pair = payload.get("current_pair")
        if pair is not None:
            for name, digest in zip(("AIA", "EUVI"), active, strict=True):
                entry = pair.get(name)
                if (entry.get("image_sha256") if entry else None) != digest:
                    raise ValueError("草稿配对与原图身份不同")
        editor = payload.get("point_editor")
        profile = payload.get("event_profile", {})
        validate_event_profile(profile)
        if editor is not None:
            if (
                type(editor.get("number")) is not int
                or not 1 <= editor["number"] <= 9999
            ):
                raise ValueError("草稿当前编号无效")
            for values in (editor.get("values"), editor.get("baseline")):
                if not isinstance(values, dict) or values.get("role") not in (
                    "jet",
                    "reference",
                    "unknown",
                ):
                    raise ValueError("草稿点说明无效")
                if values.get("identity_status") not in (
                    "possible",
                    "manual_confirmed",
                    "uncertain",
                ):
                    raise ValueError("草稿身份说明无效")
                if values.get("endpoint") not in (None, "inner", "outer"):
                    raise ValueError("草稿端点说明无效")
                if not isinstance(values.get("feature", ""), str):
                    raise ValueError("草稿特征说明无效")
                if values.get("order") is not None and (
                    type(values["order"]) is not int or not 1 <= values["order"] <= 9999
                ):
                    raise ValueError("草稿顺序无效")
        if cancelled():
            raise InterruptedError("草稿读取已取消")
        return payload, docs

    def apply_prepared(self, prepared):
        payload, docs = prepared
        frames = payload.get("frames", [])
        pair = payload.get("current_pair")
        editor = payload.get("point_editor")
        profile = payload.get("event_profile", {})
        c = self.owner.common
        if hasattr(c, "cancel_pending"):
            c.cancel_pending()
        else:
            c.play.stop()
            c.generation += 1
            c.debounce.stop()
            for future in (c.future, c.roi_future):
                if future:
                    future.cancel()
        c.pending_frame = None
        c.documents = docs
        c.document_snapshots = {
            k: v for k, v in payload["documents"].items() if k not in docs
        }
        c.records = frames
        if hasattr(c, "event_profile"):
            c.event_profile = deepcopy(profile)
        c.paired = []
        c.reference = None
        c.reference_hash = None
        c.current_pair = pair
        c.native_drafts = {}
        for pane, digest in zip(self.owner.panes, payload["active"]):
            pane.doc = docs.get(digest)
        self.owner.sample_id = payload.get("sample_id", "")
        self.owner.last_precheck = None
        self.owner.reconstruction_result = None
        self.owner.reconstruction_signature = None
        self.owner.fit_options = deepcopy(payload.get("fit_options", {}))
        if hasattr(self.owner, "fit_curve"):
            self.owner.sync_fit_controls()
        self.owner.last_saved = False
        if frames:
            c.rebuild_pairs(select=False)
        self.owner.sync_controls()
        if getattr(self.owner, "native_only", False):
            self.owner.sync_point_editor()
            if editor:
                blocker = QSignalBlocker(self.owner.tie_id)
                self.owner.tie_id.setValue(editor["number"])
                del blocker
                self.owner._point_editor_number = editor["number"]
                self.owner._point_editor_baseline = deepcopy(editor["baseline"])
                self.owner.restore_point_editor_values(editor["values"])
        self.owner.redraw()

    def close(self):
        self.timer.stop()
