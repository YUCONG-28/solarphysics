"""Native timeline/session controller, independent of shared-view rendering."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, CancelledError
from copy import deepcopy
import json
from threading import Event

import numpy as np
from astropy.time import Time
from PyQt6.QtCore import QTimer, Qt, QSignalBlocker
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QLabel,
    QToolButton,
    QFileDialog,
)

from solar_toolkit.map.jet_timeline import pair_frames
from solar_toolkit.map.jet_annotations import JetDocument, load_session
from .session_documents import DocumentStore
from .session_io import SessionStorage
from .session_validation import (
    prepare_timeline,
    prepare_session,
    prepare_manifest,
    checked_hash,
    validate_event_profile,
)
from .native_frames import (
    load_native_pair,
    native_pair_status,
    sample_pairing,
    matching_sample_index,
)


class NativeSession(DocumentStore, SessionStorage, QWidget):
    """Compatibility surface for legacy sessions, without a projection canvas.

    ``event_profile`` may contain ``bands``, ``start_utc`` and ``end_utc``.
    Without a supplied interval every actual timestamp remains available.
    Historical display fields are inert compatibility values, never jobs.
    """

    def __init__(self, owner, event_profile=None):
        super().__init__(owner)
        self.owner = owner
        self.event_profile = validate_event_profile(event_profile or {})
        self.reference = self.reference_hash = None
        self.records, self.paired = [], []
        self.documents, self.document_snapshots = {}, {}
        self.native_drafts, self.historical_shared_view = {}, {}
        self.current_pair = self.pending_frame = self.last_arrays = None
        self.limits = None
        self.stride = 1
        self.applying = self.closed = False
        self.cache, self.preview_cache = OrderedDict(), OrderedDict()
        self.cache_bytes = self.preview_cache_bytes = 0
        self.generation = 0
        self.future = self.roi_future = None
        self._cancel = Event()
        self._apply_job = None
        self._job_signature = None
        self._job_label = "加载原图"
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jet-native-io"
        )
        # Lightweight compatibility values keep version 1/2 and undo readable.
        self.height = QComboBox(self)
        for h in (0.0, 0.05, 0.10, 0.20):
            self.height.addItem(str(h), h)
        self.display = QComboBox(self)
        self.display.addItems(["native", "legacy_overlay", "legacy_flash"])
        self.layer = QComboBox(self)
        self.layer.addItems(["intensity", "difference"])
        self.operation = QComboBox(self)
        self.operation.addItems(["native", "legacy_rectangle", "legacy_polygon"])
        self.original = QCheckBox(self)
        self.original.setChecked(True)
        for widget in (
            self.height,
            self.display,
            self.layer,
            self.operation,
            self.original,
        ):
            widget.hide()
        self.timeline_toggle = QToolButton(self)
        self.timeline_toggle.setCheckable(True)
        self.timeline_toggle.setText("时序")
        self.timeline_controls = QWidget()
        row = QHBoxLayout(self.timeline_controls)
        row.setContentsMargins(0, 0, 0, 0)
        for title, action in (
            ("加载时序", self.open_timeline),
            ("前一帧", lambda: self.step(-1)),
            ("播放/暂停", self.toggle_play),
            ("后一帧", lambda: self.step(1)),
            ("取消加载", self.cancel_pending),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, fn=action: fn())
            row.addWidget(button)
        self.band = QComboBox()
        self.band.addItems(
            [str(x) for x in self.event_profile.get("bands", [304, 171])]
        )
        self.master = QComboBox()
        self.master.addItems(["按EUVI帧", "按AIA帧"])
        self.background = QCheckBox("全部时段")
        self.background.setChecked(
            not any(self.event_profile.get(k) for k in ("start_utc", "end_utc"))
        )
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        for widget in (self.band, self.master, self.background):
            row.addWidget(widget)
        row.addWidget(self.slider, 1)
        self.status = QLabel("加载原始双视角")
        self.timeline_controls.hide()
        self.timeline_toggle.toggled.connect(self.timeline_controls.setVisible)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(80)
        self.debounce.timeout.connect(self.start_native_job)
        self.poll = QTimer(self)
        self.poll.setInterval(30)
        self.poll.timeout.connect(self.collect)
        self.play = QTimer(self)
        self.play.setInterval(600)
        self.play.timeout.connect(self.play_step)
        self.flash = QTimer(self)  # Compatibility timer remains permanently stopped.
        self.slider.valueChanged.connect(self.select_time)
        self.band.currentIndexChanged.connect(self.rebuild_pairs)
        self.master.currentIndexChanged.connect(self.rebuild_pairs)
        self.background.toggled.connect(self.rebuild_pairs)

    def set_status(self, message):
        self.status.setText(message)
        self.owner.statusBar().showMessage(message)

    def render(self, *_):
        self.set_status(
            native_pair_status([p.doc for p in self.owner.panes], self.current_pair)
        )

    def sync_sample_selection(self):
        """Keep the sample label bound to displayed originals after a restore."""
        index = matching_sample_index(
            self.owner.manifest, [p.doc for p in self.owner.panes]
        )
        self.owner.loaded_pair_index = index
        blocker = QSignalBlocker(self.owner.pairs)
        self.owner.pairs.setCurrentIndex(index)
        del blocker
        if index >= 0:
            self.owner.sample_id = self.owner.manifest[1]["pairs"][index]["id"]

    def reset_science_for_annotations(self):
        """Loading a saved annotation bundle must not inherit another fit."""
        self.owner.reconstruction_result = None
        self.owner.reconstruction_signature = None
        self.owner.last_precheck = None
        self.owner.precheck_signature = None
        self.owner.fit_options = {"include_curve": False}
        if hasattr(self.owner, "fit_curve"):
            self.owner.sync_fit_controls()
        self.mark_science_saved()
        self.owner.last_saved = True
        from .workflow_guide import annotation_signature

        self.owner.saved_signature = annotation_signature(self.owner)

    def request(self, *_):
        if self.closed or self.applying:
            return
        self.remember()
        if self.pending_frame is not None:
            self.debounce.start()
        else:
            self.render()

    def cancel_pending(self, *_):
        self._cancel.set()
        self.generation += 1
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self._apply_job = None
        self.pending_frame = None
        self.debounce.stop()
        self.poll.stop()
        self.play.stop()
        index = self.current_pair_index()
        if index is not None:
            blocker = QSignalBlocker(self.slider)
            self.slider.setValue(index)
            del blocker
        self.set_status("已取消加载；当前原图和标注保留")

    def current_pair_index(self):
        """Return the displayed pair's index, excluding an uncommitted request."""
        if self.current_pair is None:
            return None

        def identity(pair):
            return tuple(
                (pair.get(name) or {}).get("image_sha256") for name in ("AIA", "EUVI")
            )

        wanted = identity(self.current_pair)
        return next(
            (i for i, pair in enumerate(self.paired) if identity(pair) == wanted), None
        )

    def run_load(self, work, apply, label="加载原图"):
        """Run ``work(cancelled)`` off-thread, commit only the newest result."""
        if self.closed:
            return
        history = getattr(self.owner, "undo_history", None)
        if history:
            history.capture()
        self.cancel_pending()
        token = self.generation
        self._cancel = cancel = Event()
        self._apply_job, self._job_label = apply, label
        from .workflow_guide import annotation_signature

        self._job_signature = annotation_signature(self.owner)

        def run():
            value = work(cancel.is_set)
            if cancel.is_set():
                raise InterruptedError("读取已取消")
            return token, value

        self.future = self.executor.submit(run)
        self.poll.start()
        self.set_status(label + "…可取消，现有标注保持不变")

    def open_image_async(self, index, path):
        """Stage one original FITS off-thread, changing the pane on success only."""
        if index not in (0, 1):
            raise ValueError("原图侧别无效")
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("加载原图"):
            return
        path = self.owner.validate(path)
        self.owner.update_parameters()
        self.remember()

        def apply(document):
            pane = self.owner.panes[index]
            pane.disconnect_selector()
            pane.doc = document
            self.current_pair = None
            self.sync_sample_selection()
            self.owner.sections.pop(index, None)
            self.owner.active.setCurrentIndex(index)
            self.owner.sync_controls()
            self.owner.sync_point_editor()
            pane.draw(reset=True)
            pane.set_mode(self.owner.mode.currentData())
            self.remember()
            self.owner.refresh_reconstruction()
            self.render()

        self.run_load(
            lambda cancelled: JetDocument(path, lazy_segmentation=True),
            apply,
            "加载原始 FITS",
        )

    def _prepare_sample(self, manifest, index, known, snapshots, cancelled):
        base, data = manifest
        sample = data["pairs"][index]
        documents = []
        for view in sample["views"]:
            if cancelled():
                raise InterruptedError("读取已取消")
            path = self.owner.validate(base / view["image"])
            digest = view["image_sha256"]
            if checked_hash(path, cancelled) != digest:
                raise ValueError("样本原图身份已变化")
            document = known.get(digest)
            if document is None and digest in snapshots:
                document = self.thaw(snapshots[digest])
            if document is None:
                document = JetDocument(path, lazy_segmentation=True)
            if document.sha256 != digest:
                raise ValueError("样本原图校验不一致")
            if view.get("difference"):
                difference = self.owner.validate(base / view["difference"])
                if checked_hash(difference, cancelled) != view["difference_sha256"]:
                    raise ValueError("样本差分身份已变化")
                if document.difference_sha256 != view["difference_sha256"]:
                    # Never modify the live cached document from a worker thread.
                    document = self.thaw(
                        dict(
                            path=str(document.path),
                            difference=None,
                            state=deepcopy(document.state),
                            history=deepcopy(document.history),
                            undo=deepcopy(document._undo),
                            dirty=document.dirty,
                        )
                    )
                    document.load_difference(difference)
            documents.append(document)
        return documents, sample_pairing(sample, documents)

    def _apply_sample(self, manifest, index, result, *, replace_manifest=False):
        documents, pair = result
        base, data = manifest
        if replace_manifest:
            self.owner.manifest = manifest
            blocker = QSignalBlocker(self.owner.pairs)
            self.owner.pairs.clear()
            for sample in data["pairs"]:
                self.owner.pairs.addItem(f"{sample['id']} · {sample['split']}")
            del blocker
        self.owner.sample_id = data["pairs"][index]["id"]
        self.owner.loaded_pair_index = index
        blocker = QSignalBlocker(self.owner.pairs)
        self.owner.pairs.setCurrentIndex(index)
        del blocker
        if not self.records:
            wavelength = int(float(documents[0].map.meta["wavelnth"]))
            blocker = QSignalBlocker(self.band)
            if self.band.findText(str(wavelength)) < 0:
                self.band.addItem(str(wavelength))
            self.band.setCurrentText(str(wavelength))
            del blocker
        self.apply_pair(pair, documents)

    def load_manifest_async(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("加载配对样本"):
            return
        path = self.owner.validate(path)
        self.owner.update_parameters()
        self.remember()
        known, snapshots = dict(self.documents), deepcopy(self.document_snapshots)

        def work(cancelled):
            manifest = prepare_manifest(path, self.owner.validate, cancelled)
            index = next(
                (
                    i
                    for i, p in enumerate(manifest[1]["pairs"])
                    if p["split"] == "tuning"
                ),
                0,
            )
            return (
                manifest,
                index,
                self._prepare_sample(manifest, index, known, snapshots, cancelled),
            )

        self.run_load(
            work,
            lambda data: self._apply_sample(*data, replace_manifest=True),
            "校验并加载样本",
        )

    def select_pair_async(self, index):
        if not self.owner.manifest:
            return
        if not 0 <= index < len(self.owner.manifest[1]["pairs"]):
            raise ValueError("样本编号超出范围")
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("切换样本"):
            return
        self.owner.update_parameters()
        self.remember()
        manifest = deepcopy(self.owner.manifest)
        known, snapshots = dict(self.documents), deepcopy(self.document_snapshots)
        self.run_load(
            lambda cancelled: self._prepare_sample(
                manifest, index, known, snapshots, cancelled
            ),
            lambda result: self._apply_sample(manifest, index, result),
            "加载配对样本",
        )

    def restore_annotations_async(self, path):
        """Dispatch old annotation bundles and full sessions through one UI API."""
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("恢复标注"):
            return
        path = self.owner.validate(path)

        def work(cancelled):
            info = json.loads(path.read_text())
            if cancelled():
                raise InterruptedError("读取已取消")
            if info.get("schema") == "solarphysics.jet_lab.session":
                return "session", prepare_session(
                    path, self.owner.validate, cancelled, lazy_segmentation=True
                )
            return "annotations", load_session(
                path, self.owner.validate, lazy_segmentation=True
            )

        def apply(value):
            kind, payload = value
            if getattr(self.owner, "reconstruction_dialog", None):
                self.owner.reconstruction_dialog.close()
                self.owner.reconstruction_dialog = None
            if kind == "session":
                self.apply_prepared_session(payload)
                self.owner.sync_point_editor()
            else:
                docs, bundle = payload
                self.owner.sample_id = bundle["sample_id"]
                self.apply_pair(None, docs)
                self.reset_science_for_annotations()
                self.owner.refresh_reconstruction()

        self.run_load(work, apply, "校验并恢复标注")

    def restore_recovery_async(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("恢复草稿"):
            return
        self.run_load(
            lambda cancelled: self.owner.recovery.prepare_restore(path, cancelled),
            self.owner.recovery.apply_prepared,
            "校验并恢复草稿",
        )

    def start_native_job(self):
        if self.closed or self.pending_frame is None:
            return
        pair = deepcopy(self.pending_frame)
        known, snapshots = dict(self.documents), deepcopy(self.document_snapshots)

        def work(cancelled):
            if cancelled():
                raise InterruptedError()
            return load_native_pair(
                pair, known, snapshots, self.thaw, lazy_segmentation=True
            )

        playing = self.play.isActive()
        self.run_load(work, lambda docs: self.apply_pair(pair, docs), "加载实际原图帧")
        if playing:
            self.play.start()
        self.pending_frame = pair

    def collect(self):
        if self.future is None or not self.future.done():
            return
        future, apply, label = self.future, self._apply_job, self._job_label
        input_signature = self._job_signature
        self.future = None
        self._apply_job = None
        self.poll.stop()
        self.pending_frame = None
        try:
            token, value = future.result()
            if self.closed or token != self.generation:
                return
            from .workflow_guide import annotation_signature

            if annotation_signature(self.owner) != input_signature:
                self.set_status(
                    "加载期间标注已修改，已丢弃旧加载结果；当前原图与新标注保留，请重新加载"
                )
                return
            guard = getattr(self.owner, "ensure_point_metadata_applied", None)
            if guard and not guard(label):
                self.set_status("保留当前帧：说明尚未应用")
                return
            if annotation_signature(self.owner) != input_signature:
                self.set_status("点说明已应用；原加载结果已过期，请重新加载")
                return
            history = getattr(self.owner, "undo_history", None)
            if history:
                with history.action(label, capture_before=False):
                    apply(value)
            else:
                apply(value)
        except (InterruptedError, CancelledError):
            self.set_status("已取消加载；当前工作区保留")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception(
                "Native session operation failed: %s", label
            )
            self.set_status(f"{label}失败：{exc}；当前工作区保留")

    def apply_pair(self, pair, docs):
        self.applying = True
        try:
            for pane, doc in zip(self.owner.panes, docs, strict=True):
                pane.doc = doc
            self.current_pair = pair
            self.sync_sample_selection()
            index = self.current_pair_index()
            if index is not None:
                blocker = QSignalBlocker(self.slider)
                self.slider.setValue(index)
                del blocker
            for d in docs:
                if d:
                    self.document_snapshots.pop(d.sha256, None)
            self.remember()
            self.owner.sections.clear()
            self.owner.sync_controls()
            self.owner.sync_point_editor()
            for pane in self.owner.panes:
                pane.draw(reset=True)
            self.owner.refresh_reconstruction()
            self.render()
        finally:
            self.applying = False

    def rebuild_pairs(self, *_args, select=True):
        if not self.records:
            return
        self.play.stop()
        self.remember()
        current = self.owner.panes[1 if self.master.currentIndex() == 0 else 0].doc
        current_time = Time(current.info["midpoint_utc"]).unix if current else None
        records = self.records
        if not self.background.isChecked():
            start, end = self.event_profile.get("start_utc"), self.event_profile.get(
                "end_utc"
            )
            records = [
                r
                for r in records
                if (not start or Time(r["midpoint_utc"]) >= Time(start))
                and (not end or Time(r["midpoint_utc"]) <= Time(end))
            ]
        master = "EUVI" if self.master.currentIndex() == 0 else "AIA"
        self.paired = pair_frames(records, master, int(self.band.currentText()))
        index = (
            int(
                np.argmin(
                    [
                        abs(Time(p[master]["midpoint_utc"]).unix - current_time)
                        for p in self.paired
                    ]
                )
            )
            if self.paired and current_time is not None
            else 0
        )
        blocker = QSignalBlocker(self.slider)
        self.slider.setRange(0, max(0, len(self.paired) - 1))
        self.slider.setValue(index)
        del blocker
        if self.paired and select:
            self.select_time(index)
        elif not self.paired:
            self.set_status("当前波段 / 时窗没有可用帧")

    def set_timeline_records(self, records):
        bands = sorted({int(r["band"]) for r in records})
        allowed = self.event_profile.get("bands")
        if allowed:
            bands = [b for b in bands if b in allowed]
        if not bands:
            raise ValueError("事件配置与时序没有共同波段")
        self.records = records
        previous = self.band.currentText()
        blocker = QSignalBlocker(self.band)
        self.band.clear()
        self.band.addItems([str(b) for b in bands])
        if previous in [str(b) for b in bands]:
            self.band.setCurrentText(previous)
        del blocker
        self.timeline_toggle.setChecked(True)
        self.rebuild_pairs()

    def load_timeline(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("加载时序"):
            return
        path = self.owner.validate(path)
        self.run_load(
            lambda cancelled: prepare_timeline(path, self.owner.validate, cancelled),
            self.set_timeline_records,
            "校验时序输入",
        )

    def restore_session_async(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("恢复会话"):
            return
        path = self.owner.validate(path)
        self.run_load(
            lambda cancelled: prepare_session(
                path, self.owner.validate, cancelled, lazy_segmentation=True
            ),
            self.apply_prepared_session,
            "校验并恢复会话",
        )

    def select_time(self, index):
        if not 0 <= index < len(self.paired):
            return
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("切换时间帧"):
            self.play.stop()
            return
        self.owner.update_parameters()
        self.remember()
        history = getattr(self.owner, "undo_history", None)
        if history:
            history.capture()
        self.pending_frame = self.paired[index]
        self.request()

    def step(self, delta):
        index = self.slider.value() + delta
        if not 0 <= index < len(self.paired):
            self.play.stop()
            return
        self.slider.setValue(index)

    def toggle_play(self):
        if self.play.isActive():
            self.play.stop()
        else:
            self.play.start()

    def play_step(self):
        if self.future is None and not self.debounce.isActive():
            self.step(1)

    def open_timeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 timeline.json 或 session.json",
            str(self.owner.allowed_roots[0]),
            "JSON (*.json)",
        )
        if path:
            try:
                if path.endswith("session.json"):
                    if self.owner.can_discard():
                        self.restore_session_async(path)
                else:
                    self.load_timeline(path)
            except (ValueError, OSError) as exc:
                self.set_status(str(exc))

    def precheck(self):
        self.play.stop()
        return self.owner.calculate_3d()

    def cleanup(self):
        self.closed = True
        self.cancel_pending()
        self.flash.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
