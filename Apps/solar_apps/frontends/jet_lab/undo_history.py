"""Bounded session undo/redo, independent of immutable exported revisions."""

from contextlib import contextmanager
from functools import wraps
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import zlib

import numpy as np
from PyQt6.QtCore import QTimer, QSignalBlocker
from PyQt6.QtWidgets import QApplication
from solar_toolkit.map.jet_annotations import file_sha256, json_safe


def undoable(label):
    """Group nested synchronous actions; async results commit when settled."""

    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            owner = getattr(self, "owner", self)
            history = getattr(owner, "undo_history", None)
            if history is None or history.restoring:
                return fn(self, *args, **kwargs)
            with history.action(label):
                return fn(self, *args, **kwargs)

        return call

    return decorate


def document_state(doc):
    if doc is None:
        return None
    return dict(
        path=str(doc.path),
        sha256=doc.sha256,
        difference=str(doc.difference_path) if doc.difference_path else None,
        difference_sha256=doc.difference_sha256,
        state=doc.state,
        history=doc.history,
        undo=doc._undo,
        dirty=doc.dirty,
    )


class SessionUndo:
    """Keep up to 50 transitions / 64 MiB of compressed JSON in this session.

    No image arrays or Python objects are persisted. Files are verified before
    restoration; saved/exported directories are never removed by undo.
    """

    def __init__(self, owner):
        self.owner = owner
        self.restoring = False
        self.depth = 0
        self.label = "调整视图 / 操作设置"
        self.entries = []
        self.cursor = -1
        self.layout_revision = 0
        self.max_bytes = 64 * 1024**2
        self.timer = QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.capture)
        for splitter in (owner.vertical_splitter, owner.native_splitter):
            splitter.splitterMoved.connect(self.layout_changed)
        self.capture("初始工作区")
        # Controls and completed draws schedule one coalesced snapshot. Idle
        # workspaces no longer serialize/compress all documents every 350 ms.
        from PyQt6.QtWidgets import (
            QComboBox,
            QAbstractSpinBox,
            QLineEdit,
            QAbstractButton,
            QTabBar,
        )

        for widget in owner.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.schedule_capture)
        for widget in owner.findChildren(QAbstractSpinBox):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.schedule_capture)
        for widget in owner.findChildren(QLineEdit):
            widget.textChanged.connect(self.schedule_capture)
        for widget in owner.findChildren(QAbstractButton):
            widget.toggled.connect(self.schedule_capture)
        for widget in owner.findChildren(QTabBar):
            widget.currentChanged.connect(self.schedule_capture)
        self._draw_connections = [
            (p.canvas, p.canvas.mpl_connect("draw_event", self.schedule_capture))
            for p in owner.panes
        ]

    def schedule_capture(self, *_):
        if self.restoring:
            return
        self.timer.start()
        recovery = getattr(self.owner, "recovery", None)
        if recovery and hasattr(recovery, "notify_changed"):
            recovery.notify_changed()

    def layout_changed(self, *_):
        if not self.restoring:
            self.layout_revision += 1
            self.schedule_capture()

    @staticmethod
    def fingerprint(state):
        # Qt adjusts these sizes after painting; only an actual divider gesture
        # increments layout_revision. Automatic layout must not consume Undo.
        comparable = dict(state, view=dict(state["view"]))
        comparable.pop("split_sizes", None)
        for key in ("vertical_panel_sizes", "native_panel_sizes"):
            comparable["view"].pop(key, None)
        return hashlib.sha256(
            json.dumps(comparable, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def busy(self):
        w, c = self.owner, self.owner.common
        return (
            w.timer.isActive()
            or c.pending_frame is not None
            or c.roi_future is not None
            or c.future is not None
            or c.play.isActive()
            or bool(QApplication.mouseButtons())
            or QApplication.activeModalWidget() is not None
        )

    def snapshot(self):
        w, c = self.owner, self.owner.common
        c.remember()
        active = [document_state(p.doc) for p in w.panes]
        # Retain non-active frames without copying immutable image arrays.
        active_keys = {d["sha256"] for d in active if d}
        inactive = {
            k: v for k, v in c.document_snapshots.items() if k not in active_keys
        }
        inactive.update(
            {
                k: document_state(d)
                for k, d in c.documents.items()
                if k not in active_keys
            }
        )

        def limits(p):
            return (
                [
                    [round(float(x), 6) for x in pair]
                    for pair in (p.ax.get_xlim(), p.ax.get_ylim())
                ]
                if p.doc
                else None
            )

        widgets = {
            name: getattr(w, name).currentIndex()
            for name in (
                "active",
                "mode",
                "tie_status",
                "view_layer",
                "sections_box",
                "theme",
            )
        }
        return json_safe(
            dict(
                active=active,
                inactive=inactive,
                sample_id=w.sample_id,
                manifest=[str(w.manifest[0]), w.manifest[1]] if w.manifest else None,
                loaded_pair_index=w.loaded_pair_index,
                frames=c.records,
                paired=c.paired,
                view=c.session_state(),
                native_limits=[limits(p) for p in w.panes],
                tabs=[p.tabs.currentIndex() for p in w.panes],
                widgets=widgets,
                tie_id=w.tie_id.value(),
                feature=w.feature.text(),
                operation=c.operation.currentIndex(),
                epipolar_anchor=w.epipolar_anchor,
                guide_step=w.guide.steps.currentRow(),
                last_precheck=w.last_precheck,
                precheck_signature=getattr(w, "precheck_signature", None),
                reconstruction_result=getattr(w, "reconstruction_result", None),
                reconstruction_signature=getattr(w, "reconstruction_signature", None),
                fit_options=deepcopy(getattr(w, "fit_options", {})),
                point_editor=(
                    [
                        w.point_role.currentIndex(),
                        w.point_order.value(),
                        w.point_endpoint.currentIndex(),
                        w.parameter_panel.currentIndex(),
                        w.point_hint.text(),
                    ]
                    if hasattr(w, "point_role")
                    else None
                ),
                point_editor_baseline=getattr(w, "_point_editor_baseline", None),
                point_editor_number=getattr(w, "_point_editor_number", None),
                last_saved=w.last_saved,
                saved_signature=getattr(w, "saved_signature", None),
                saved_science_signature=getattr(w, "saved_science_signature", None),
                saved_output=getattr(w, "last_output", None),
                sections=w.sections,
                message=w.message.toPlainText(),
                details=w.raw_details.toPlainText(),
                image_priority=w.image_priority.isChecked(),
                priority_previous=w._priority_previous,
                split_sizes=w._split_sizes,
                layout_revision=self.layout_revision,
            )
        )

    @contextmanager
    def action(self, label, *, capture_before=True):
        if self.depth == 0:
            if capture_before:
                self.capture()
            self.label = label
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                self.capture()
                self.schedule_capture()

    def capture(self, label=None):
        if self.restoring or self.depth or self.busy():
            return False
        try:
            state = self.snapshot()
            body = json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
            digest = self.fingerprint(state)
            if self.cursor >= 0 and digest == self.entries[self.cursor][0]:
                self.entries[self.cursor] = (
                    digest,
                    zlib.compress(body),
                    self.entries[self.cursor][2],
                )
                self.refresh()
                return False
            data = zlib.compress(body)
            if len(data) > self.max_bytes:
                self.owner.statusBar().showMessage(
                    "当前标注超过撤销容量；请保存新修订。"
                )
                return False
            self.entries = self.entries[: self.cursor + 1]
            self.entries.append((digest, data, label or self.label))
            while (
                len(self.entries) > 51
                or sum(len(e[1]) for e in self.entries) > self.max_bytes
            ):
                self.entries.pop(0)
            self.cursor = len(self.entries) - 1
            self.label = "调整视图 / 操作设置"
            self.refresh()
            return True
        except (ValueError, TypeError, OSError) as exc:
            self.owner.statusBar().showMessage(f"撤销记录未更新：{exc}")
            return False

    def refresh(self):
        w = self.owner
        w.undo_button.setEnabled(self.cursor > 0 or self.busy())
        w.redo_button.setEnabled(self.cursor + 1 < len(self.entries))
        w.undo_button.setToolTip(
            "撤销：" + (self.entries[self.cursor][2] if self.cursor > 0 else "无历史")
        )
        w.redo_button.setToolTip(
            "重做："
            + (
                self.entries[self.cursor + 1][2]
                if self.cursor + 1 < len(self.entries)
                else "无历史"
            )
        )

    def cancel_jobs(self):
        c = self.owner.common
        pending = (
            c.pending_frame is not None or c.roi_future is not None or c.play.isActive()
        )
        if hasattr(self.owner, "_fit_generation"):
            pending = pending or getattr(self.owner, "_fit_future", None) is not None
            self.owner.cancel_fit()
        if hasattr(c, "cancel_pending"):
            pending = pending or c.future is not None
            c.cancel_pending()
            return pending
        c.play.stop()
        c.flash.stop()
        c.debounce.stop()
        c.poll.stop()
        c.generation += 1
        for future in (c.future, c.roi_future):
            if future is not None:
                future.cancel()
        c.future = c.roi_future = None
        c.pending_frame = None
        return pending

    def move(self, direction):
        w = self.owner
        w.update_parameters()
        pending = self.cancel_jobs()
        changed = self.capture()
        # A draft not yet computed is itself the current cancellable step.
        if direction < 0 and pending and not changed:
            w.statusBar().showMessage("已撤销尚未完成的预览 / 选区任务；标注保持不变。")
            self.refresh()
            return False
        target = self.cursor + direction
        if not 0 <= target < len(self.entries):
            self.refresh()
            return False
        old_cursor = self.cursor
        self.restoring = True
        try:
            state = json.loads(zlib.decompress(self.entries[target][1]))
            self.restore(state)
            self.cursor = target
            # Rendering may normalize harmless widget/layout details.
            normalized = self.snapshot()
            body = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
            _, _, label = self.entries[target]
            self.entries[target] = (
                self.fingerprint(normalized),
                zlib.compress(body),
                label,
            )
            w.statusBar().showMessage(
                (
                    "已撤销：" + self.entries[old_cursor][2]
                    if direction < 0
                    else "已重做：" + label
                )
                + "。已保存 / 导出文件保留；需要新结果时另存修订。",
                12000,
            )
            return True
        except (ValueError, OSError, KeyError, TypeError) as exc:
            w.statusBar().showMessage(f"不能恢复该步骤：{exc}；当前标注保留。")
            return False
        finally:
            self.restoring = False
            self.refresh()

    def restore(self, s):
        """Validate and load all required inputs before changing the workspace."""
        w, c = self.owner, self.owner.common

        def thaw(record):
            if record is None:
                return None
            path = w.validate(record["path"])
            if file_sha256(path) != record["sha256"]:
                raise ValueError("原图身份改变")
            if record.get("difference"):
                path = w.validate(record["difference"])
                if file_sha256(path) != record["difference_sha256"]:
                    raise ValueError("差分身份改变")
            return c.thaw(record)

        docs = [thaw(r) for r in s["active"]]
        ref_hash = s["view"]["reference_image_sha256"]
        reference = next((d for d in docs if d and d.sha256 == ref_hash), None)
        if reference is None and ref_hash in s["inactive"]:
            record = dict(s["inactive"][ref_hash], sha256=ref_hash)
            reference = thaw(record)
        if ref_hash and reference is None:
            raise ValueError("参考图像缺少身份记录")
        controls = [
            getattr(w, name)
            for name in (
                "pairs",
                "active",
                "mode",
                "tie_status",
                "view_layer",
                "sections_box",
                "theme",
                "tie_id",
                "feature",
                "parameter_toggle",
                "image_priority",
                "stretch",
                "colour",
                "view_selector",
                "method",
                "value",
                "polarity",
                "opening",
                "closing",
                "area",
                "source",
                "low",
                "high",
                "branch",
            )
        ]
        controls += [
            getattr(c, name)
            for name in (
                "original",
                "display",
                "layer",
                "height",
                "operation",
                "band",
                "master",
                "background",
                "slider",
                "timeline_toggle",
            )
        ]
        controls += [p.tabs for p in w.panes] + [w.guide.steps]
        if hasattr(w, "point_role"):
            controls += [w.point_role, w.point_order, w.point_endpoint]
        blockers = [QSignalBlocker(obj) for obj in controls]
        try:
            self.cancel_jobs()
            w.timer.stop()
            w.pending_parameters = None
            self.layout_revision = s["layout_revision"]
            if w.profile_dialog:
                w.profile_dialog.close()
                w.profile_dialog = None
            if getattr(w, "reconstruction_dialog", None):
                w.reconstruction_dialog.close()
                w.reconstruction_dialog = None
            c.applying = True
            c.documents = {d.sha256: d for d in docs if d}
            c.document_snapshots = s["inactive"]
            c.records = s["frames"]
            c.paired = s["paired"]
            c.current_pair = s["view"]["current_pair"]
            if hasattr(c, "event_profile"):
                c.event_profile = deepcopy(s["view"].get("event_profile", {}))
            c.reference = reference.map if reference else None
            c.reference_hash = ref_hash
            c.last_arrays = None
            c.native_drafts = {}
            for digest, draft in s["view"]["native_roi_drafts"].items():
                mask = np.zeros(draft["shape"], dtype=bool)
                for start, stop in draft["runs"]:
                    mask.ravel()[start:stop] = True
                c.native_drafts[digest] = (mask, draft["evidence"])
            for pane, d, tab, lim in zip(
                w.panes, docs, s["tabs"], s["view"]["display_limits"]
            ):
                pane.doc = d
                pane.tabs.setCurrentIndex(tab)
                pane.display_limits = lim
            w.sample_id = s["sample_id"]
            w.manifest = (
                (Path(s["manifest"][0]), s["manifest"][1]) if s["manifest"] else None
            )
            w.loaded_pair_index = s["loaded_pair_index"]
            w.pairs.clear()
            if w.manifest:
                for p in w.manifest[1]["pairs"]:
                    w.pairs.addItem(f"{p['id']} · {p['split']}")
                w.pairs.setCurrentIndex(w.loaded_pair_index)
            for name, index in s["widgets"].items():
                getattr(w, name).setCurrentIndex(index)
            w.tie_id.setValue(s["tie_id"])
            w.feature.setText(s["feature"])
            w.epipolar_anchor = s["epipolar_anchor"]
            w.guide.steps.setCurrentRow(s["guide_step"])
            v = s["view"]
            w.native_focus = v["native_focus"]
            w._split_sizes = s["split_sizes"]
            w.parameter_toggle.setChecked(v["parameters_visible"])
            w.parameter_panel.setVisible(v["parameters_visible"])
            w.guide.setVisible(v["guide_visible"])
            w.image_priority.setChecked(s["image_priority"])
            w._priority_previous = s["priority_previous"]
            w.stretch.setCurrentText(v["stretch"])
            w.colour.setCurrentText(v["colour"])
            c.display.setCurrentIndex(v["display_mode"])
            c.layer.setCurrentIndex(v["layer"])
            c.height.setCurrentIndex(c.height.findData(v["height_rsun"]))
            c.operation.setCurrentIndex(s["operation"])
            if getattr(w, "native_only", False) and c.band.findText(str(v["band"])) < 0:
                c.band.addItem(str(v["band"]))
            c.band.setCurrentText(v["band"])
            c.master.setCurrentIndex(v["master"])
            c.background.setChecked(v["background"])
            c.slider.setRange(0, max(0, len(c.paired) - 1))
            c.slider.setValue(v["frame_index"])
            c.timeline_toggle.setChecked(v["timeline_controls_visible"])
            c.timeline_controls.setVisible(v["timeline_controls_visible"])
            w.set_view_mode(v["window_view_mode"])
            w.vertical_splitter.setSizes(v["vertical_panel_sizes"])
            w.native_splitter.setSizes(v["native_panel_sizes"])
            w.sections = {int(k): val for k, val in s["sections"].items()}
            w.last_precheck = s["last_precheck"]
            w.precheck_signature = s["precheck_signature"]
            w.reconstruction_result = s.get("reconstruction_result")
            w.reconstruction_signature = s.get("reconstruction_signature")
            w.fit_options = deepcopy(s.get("fit_options", {}))
            if hasattr(w, "fit_curve"):
                w.sync_fit_controls()
            if s.get("point_editor") and hasattr(w, "point_role"):
                role, order, endpoint, panel, hint = s["point_editor"]
                w.point_role.setCurrentIndex(role)
                w.point_order.setValue(order)
                w.point_endpoint.setCurrentIndex(endpoint)
                w.parameter_panel.setCurrentIndex(panel)
                w.point_hint.setText(hint)
                w._point_editor_number = (
                    s.get("point_editor_number") or w.tie_id.value()
                )
                if s.get("point_editor_baseline") is not None:
                    w._point_editor_baseline = s["point_editor_baseline"]
                    w.refresh_point_metadata_pending()
                else:
                    w.reset_point_editor_baseline()
            w.last_saved = s["last_saved"]
            w.saved_signature = s["saved_signature"]
            w.last_output = s["saved_output"]
            w.saved_science_signature = s.get("saved_science_signature")
            w.sync_controls()
            w.redraw()
            for pane, limits in zip(w.panes, s["native_limits"]):
                if pane.doc and limits:
                    pane.ax.set_xlim(limits[0])
                    pane.ax.set_ylim(limits[1])
                    pane.canvas.draw_idle()
            c.limits = v["limits"]
            w.message.setPlainText(s["message"])
            w.raw_details.setPlainText(s["details"])
            w.guide.refresh()
        finally:
            c.applying = False
            del blockers
        w.apply_theme(w.theme.currentText())
        if w.view_mode != "native":
            c.request()

    def close(self):
        self.timer.stop()
        for canvas, connection in self._draw_connections:
            canvas.mpl_disconnect(connection)
        self._draw_connections.clear()
        self.entries.clear()
