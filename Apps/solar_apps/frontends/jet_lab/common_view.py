"""Shared-view browsing. All scientific editing remains in native panes."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, PolygonSelector
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QCheckBox,
    QSlider,
    QFileDialog,
    QToolButton,
)
from PyQt6.QtCore import Qt

from solar_toolkit.map.jet_annotations import (
    file_sha256,
    json_safe,
)
from solar_toolkit.map.jet_viewpoint import (
    build_mapping,
    sample_mapping,
    native_roi_mask,
    reproject_photosphere,
)
from solar_toolkit.map.jet_timeline import pair_frames, geometry_precheck


from .session_io import SessionStorage
from .session_documents import DocumentStore
from .undo_history import undoable
from .view_navigation import compact_toolbar, ViewNavigation
from .native_frames import load_native_pair, native_pair_status


class CommonView(DocumentStore, SessionStorage, QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.reference = None
        self.reference_hash = None
        self.records = []
        self.paired = []
        self.documents = {}
        self.current_pair = None
        self.document_snapshots = {}
        self.native_drafts = {}
        self.last_arrays = None
        self.stride = 1
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jet-preview"
        )
        self.future = None
        self.generation = 0
        self.closed = False
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.preview_cache = OrderedDict()
        self.preview_cache_bytes = 0
        self.selectors = []
        self.limits = None
        self.syncing_limits = False
        self.pending_frame = None
        self.roi_future = None
        self.applying = False
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        self.display = QComboBox()
        self.display.addItems(["并排", "透明叠加", "闪烁"])
        self.height = QComboBox()
        for h in [0.0, 0.05, 0.10, 0.20]:
            self.height.addItem(f"显示假设：高度 {h:g} R☉", h)
        self.layer = QComboBox()
        self.layer.addItems(["原图显示", "差分显示"])
        self.operation = QComboBox()
        self.operation.addItems(["浏览", "矩形候选区", "多边形候选区"])
        for w in [self.display, self.height, self.layer, self.operation]:
            bar.addWidget(w)
        reset = QPushButton("以当前AIA为参考")
        reset.clicked.connect(lambda checked=False: self.reset_reference())
        bar.addWidget(reset)
        self.original = QCheckBox("展开原图确认")
        self.original.setChecked(True)
        self.original.toggled.connect(self.show_original)
        self.original.hide()  # Kept for old session compatibility, not a second layout switch.
        native = QPushButton("原图标注")
        native.clicked.connect(lambda: self.owner.set_view_mode("native"))
        bar.addWidget(native)
        self.timeline_toggle = QToolButton()
        self.timeline_toggle.setText("时序 ▸")
        self.timeline_toggle.setCheckable(True)
        bar.addWidget(self.timeline_toggle)
        self.timeline_controls = QWidget()
        row = QHBoxLayout(self.timeline_controls)
        row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.timeline_controls)
        self.timeline_controls.hide()
        self.timeline_toggle.toggled.connect(self.timeline_controls.setVisible)
        self.timeline_toggle.toggled.connect(
            lambda visible: self.timeline_toggle.setText(
                "时序 ▾" if visible else "时序 ▸"
            )
        )
        for title, fn in [
            ("加载时序", self.open_timeline),
            ("前一帧", lambda: self.step(-1)),
            ("播放/暂停", self.toggle_play),
            ("后一帧", lambda: self.step(1)),
        ]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, action=fn: action())
            row.addWidget(b)
        self.band = QComboBox()
        self.band.addItems(["304", "171"])
        row.addWidget(self.band)
        self.master = QComboBox()
        self.master.addItems(["按EUVI帧", "按AIA帧"])
        row.addWidget(self.master)
        self.background = QCheckBox("全部时段")
        self.background.setChecked(True)
        row.addWidget(self.background)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        row.addWidget(self.slider, 1)
        self.status = QLabel("共同视角仅辅助辨认；请在原图中确认结构。")
        self.status.setWordWrap(True)
        self.status.setMaximumHeight(48)
        layout.addWidget(self.status)
        self.figure = Figure(figsize=(10, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(160)
        self.toolbar = compact_toolbar(self.canvas, self)
        self.navigation = ViewNavigation(self.canvas, self.toolbar)
        nav_row = QHBoxLayout()
        layout.addLayout(nav_row)
        nav_row.addWidget(self.toolbar, 1)
        enlarge = QPushButton("放大共同视角")
        enlarge.clicked.connect(lambda: self.owner.set_view_mode("common"))
        nav_row.addWidget(enlarge)
        layout.addWidget(self.canvas, 1)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        for title, fn in [
            ("确认左侧ROI草稿", lambda: self.accept_roi(0)),
            ("确认右侧ROI草稿", lambda: self.accept_roi(1)),
            ("几何预检", self.precheck),
            ("保存时序会话", self.save_dialog),
            ("导出计算输入", self.export_dialog),
        ]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, action=fn: action())
            actions.addWidget(b)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(120)
        self.debounce.timeout.connect(self.start_job)
        self.poll = QTimer(self)
        self.poll.setInterval(40)
        self.poll.timeout.connect(self.collect)
        self.play = QTimer(self)
        self.play.setInterval(600)
        self.play.timeout.connect(self.play_step)
        self.flash = QTimer(self)
        self.flash.setInterval(650)
        self.flash.timeout.connect(self.flash_tick)
        self.flash_phase = False
        for w in [self.height, self.layer]:
            w.currentIndexChanged.connect(self.request)
        self.display.currentIndexChanged.connect(self.render)
        self.operation.currentIndexChanged.connect(self.render)
        self.slider.valueChanged.connect(self.select_time)
        self.master.currentIndexChanged.connect(self.rebuild_pairs)
        self.band.currentIndexChanged.connect(self.rebuild_pairs)
        self.background.toggled.connect(self.rebuild_pairs)

    def show_original(self, visible):
        if getattr(self.owner, "native_only", False):
            return
        self.owner.set_view_mode("split" if visible else "common")

    @undoable("更换显示参考")
    def reset_reference(self):
        if getattr(self.owner, "native_only", False):
            return
        d = self.owner.panes[0].doc
        if d is not None:
            self.reference = d.map
            self.reference_hash = d.sha256
            self.limits = None
            self.request()

    def request(self, *args):
        if self.closed or self.applying:
            return
        self.remember()
        if getattr(self.owner, "native_only", False):
            self.reference = None
            self.reference_hash = None
            self.generation += 1
            if self.pending_frame is not None:
                self.debounce.start()
            else:
                self.render()
            return
        if self.reference is None:
            d = self.owner.panes[0].doc
            if d is not None:
                self.reference = d.map
                self.reference_hash = d.sha256
        self.generation += 1
        self.debounce.start()

    def start_job(self):
        if getattr(self.owner, "native_only", False):
            return self.start_native_job()
        if self.closed or (self.reference is None and self.pending_frame is None):
            return
        if self.future is not None and not self.future.done():
            self.debounce.start()
            return
        token = self.generation
        ref = self.reference
        rh = self.reference_hash
        h = self.height.currentData()
        difference = self.layer.currentIndex() == 1
        docs = [p.doc for p in self.owner.panes]
        pair = self.pending_frame
        known = dict(self.documents)
        snapshots = dict(self.document_snapshots)

        def work():
            if pair is not None:
                ds = load_native_pair(pair, known, snapshots, self.thaw)
            else:
                ds = docs
            reference = ref if ref is not None else (ds[0].map if ds[0] else None)
            reference_hash = (
                rh if ref is not None else (ds[0].sha256 if ds[0] else None)
            )
            if reference is None:
                raise ValueError("当前帧缺少AIA参考图，请选择有AIA配对的实际帧")
            stride = max(1, int(np.ceil(max(reference.data.shape) / 1024)))
            arrays = []
            for d in ds:
                if d is None:
                    arrays.append(None)
                    continue
                key = (d.sha256, reference_hash, h, stride)
                mapping = self.cache.get(key)
                if mapping is None:
                    mapping = build_mapping(d.map, reference, h, stride)
                    size = (
                        mapping.source_x.nbytes
                        + mapping.source_y.nbytes
                        + mapping.valid.nbytes
                    )
                    while self.cache and self.cache_bytes + size > 128 * 1024**2:
                        _, old = self.cache.popitem(last=False)
                        self.cache_bytes -= (
                            old.source_x.nbytes + old.source_y.nbytes + old.valid.nbytes
                        )
                    if size <= 128 * 1024**2:
                        self.cache[key] = mapping
                        self.cache_bytes += size
                else:
                    self.cache.move_to_end(key)
                data = d.difference if difference else d.raw
                if data is None:
                    arrays.append(None)
                elif h == 0:
                    preview_key = (
                        *key,
                        d.difference_sha256 if difference else "intensity",
                        "sunpy_reproject_v1",
                    )
                    preview = self.preview_cache.get(preview_key)
                    if preview is None:
                        preview = reproject_photosphere(
                            d.map, reference, data, stride=stride, mapping=mapping
                        )
                        while (
                            self.preview_cache
                            and self.preview_cache_bytes + preview.nbytes > 64 * 1024**2
                        ):
                            _, old = self.preview_cache.popitem(last=False)
                            self.preview_cache_bytes -= old.nbytes
                        if preview.nbytes <= 64 * 1024**2:
                            self.preview_cache[preview_key] = preview
                            self.preview_cache_bytes += preview.nbytes
                    else:
                        self.preview_cache.move_to_end(preview_key)
                    arrays.append(preview)
                else:
                    arrays.append(sample_mapping(data, mapping))
            return token, ds, arrays, pair, reference, reference_hash, stride

        self.future = self.executor.submit(work)
        self.poll.start()
        self.status.setText("正在更新共同视角…原始标注保持不变。")

    def start_native_job(self):
        """Load originals without constructing any reference image or mapping."""
        if self.closed or self.pending_frame is None:
            return
        if self.future is not None and not self.future.done():
            self.debounce.start()
            return
        token, pair = self.generation, self.pending_frame
        known, snapshots = dict(self.documents), deepcopy(self.document_snapshots)

        def work():
            docs = load_native_pair(pair, known, snapshots, self.thaw)
            return token, docs, None, pair, None, None, 1

        self.future = self.executor.submit(work)
        self.poll.start()
        self.status.setText("正在加载实际原图帧…各视角坐标保持独立。")

    def collect(self):
        if self.future is None or not self.future.done():
            return
        future = self.future
        self.future = None
        self.poll.stop()
        try:
            token, docs, arrays, pair, reference, reference_hash, stride = (
                future.result()
            )
            if token != self.generation:
                return
            self.reference, self.reference_hash, self.stride = (
                reference,
                reference_hash,
                stride,
            )
            if pair is not None:
                self.pending_frame = None
                guard = getattr(self.owner, "ensure_point_metadata_applied", None)
                if guard and not guard("显示已加载的配对帧"):
                    self.play.stop()
                    self.status.setText("保留当前帧：点说明尚未应用")
                    return
                self.applying = True
                try:
                    for pane, d in zip(self.owner.panes, docs):
                        pane.doc = d
                    for d in docs:
                        if d is not None:
                            self.document_snapshots.pop(d.sha256, None)
                    self.remember()
                    self.owner.sections.clear()
                    self.owner.sync_controls()
                    for pane in self.owner.panes:
                        pane.draw(reset=True)
                    self.current_pair = pair
                    if getattr(self.owner, "native_only", False):
                        self.owner.sync_point_editor()
                finally:
                    self.applying = False
            self.last_arrays = arrays
            self.render()
            if getattr(self.owner, "native_only", False):
                self.owner.refresh_reconstruction()
        except Exception as exc:
            self.pending_frame = None
            self.status.setText(f"原图/预览未更新：{exc}")

    def render(self, *args):
        if getattr(self.owner, "native_only", False):
            self.flash.stop()
            self.status.setText(
                native_pair_status([p.doc for p in self.owner.panes], self.current_pair)
            )
            return
        if self.last_arrays is None:
            return
        for s in self.selectors:
            s.disconnect_events()
        self.selectors = []
        self.figure.clear()
        mode = self.display.currentIndex()
        axes = (
            self.figure.subplots(1, 2) if mode == 0 else [self.figure.add_subplot(111)]
        )
        axes = list(np.atleast_1d(axes))
        self.axes = axes
        self.images = []
        dims = self.reference.data.shape
        for i, a in enumerate(self.last_arrays):
            ax = axes[i] if mode == 0 else axes[0]
            if a is not None and np.isfinite(a).any():
                values = a[np.isfinite(a)]
                lo, hi = np.percentile(values, self.owner.panes[i].display_limits)
                hi = max(hi, lo + 1e-12)
                z = np.clip((a - lo) / (hi - lo), 0, 1)
                if self.owner.stretch.currentText() == "Asinh":
                    z = np.arcsinh(z * 10) / np.arcsinh(10)
                im = ax.imshow(
                    np.ma.masked_invalid(z),
                    origin="lower",
                    extent=(
                        -self.stride / 2,
                        (a.shape[1] - 0.5) * self.stride,
                        -self.stride / 2,
                        (a.shape[0] - 0.5) * self.stride,
                    ),
                    cmap="gray" if i == 0 or mode == 0 else "magma",
                    alpha=0.5 if i == 1 and mode == 1 else 1,
                    interpolation="nearest",
                )
                self.images.append(im)
            else:
                ax.text(
                    0.5,
                    0.5,
                    ["AIA", "EUVI"][i] + ": no valid pair / layer / overlap",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set(
                xlabel="AIA ref. x [px]",
                ylabel="AIA ref. y [px]",
            )
            if mode == 0:
                ax.set_title(["AIA", "EUVI → AIA (display only)"][i], fontsize=10)
        for ax in axes:
            foreground = self.owner.palette().windowText().color().name()
            ax.set_facecolor(self.owner.palette().base().color().name())
            ax.tick_params(colors=foreground, labelsize=9)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            ax.xaxis.label.set_color(foreground)
            ax.yaxis.label.set_color(foreground)
            ax.title.set_color(foreground)
            for spine in ax.spines.values():
                spine.set_color(foreground)
            ax.set_xlim(-0.5, dims[1] - 0.5)
            ax.set_ylim(-0.5, dims[0] - 0.5)
            if self.limits:
                ax.set_xlim(self.limits[0])
                ax.set_ylim(self.limits[1])
            ax.callbacks.connect("xlim_changed", self.link_limits)
            ax.callbacks.connect("ylim_changed", self.link_limits)
            if self.operation.currentIndex() == 1:

                def rectangle(a, b):
                    if self.toolbar.mode or a.xdata is None or b.xdata is None:
                        return
                    self.make_draft(
                        [
                            [a.xdata, a.ydata],
                            [b.xdata, a.ydata],
                            [b.xdata, b.ydata],
                            [a.xdata, b.ydata],
                        ]
                    )

                self.selectors.append(
                    RectangleSelector(ax, rectangle, useblit=False, button=[1])
                )
            elif self.operation.currentIndex() == 2:
                selector = PolygonSelector(ax, self.make_draft, useblit=False)
                selector.validButtons = [1]
                self.selectors.append(selector)
        self.figure.set_facecolor(self.owner.palette().window().color().name())
        self.canvas.draw_idle()
        if mode == 2:
            self.flash.start()
        else:
            self.flash.stop()
        pair = self.current_pair
        text = "显示球壳为假设；透明处不补图；请在原图确认。"
        if pair:
            stamps = [
                f"{inst}: " + (pair[inst]["midpoint_utc"] if pair[inst] else "缺配")
                for inst in ["AIA", "EUVI"]
            ]
            text = (
                " | ".join(stamps)
                + f" | {pair['status']} | 发射时间差 "
                + (
                    f"{pair['delta_emission_s']:.3f} s"
                    if pair["delta_emission_s"] is not None
                    else "未定"
                )
            )
            if pair["repeated_other"]:
                text += " | 重复使用配对帧，不是新观测"
        self.status.setText(
            text
            + (
                " | SunPy官方重投影 · 光球假设"
                if self.height.currentData() == 0
                else " | 等高球壳预览 · 条件显示"
            )
        )

    def link_limits(self, ax):
        if self.syncing_limits:
            return
        self.syncing_limits = True
        try:
            self.limits = (ax.get_xlim(), ax.get_ylim())
            for other in self.axes:
                if other is not ax:
                    other.set_xlim(self.limits[0])
                    other.set_ylim(self.limits[1])
            self.canvas.draw_idle()
        finally:
            self.syncing_limits = False

    def flash_tick(self):
        if len(self.images) == 2:
            self.flash_phase = not self.flash_phase
            self.images[0].set_visible(self.flash_phase)
            self.images[1].set_visible(not self.flash_phase)
            self.canvas.draw_idle()

    @undoable("共同视角 ROI 草稿")
    def make_draft(self, vertices):
        if getattr(self.owner, "native_only", False):
            self.status.setText("本流程仅支持原图 ROI；共同视角映射已停用。")
            return
        if self.toolbar.mode:
            return
        self.play.stop()
        h = self.height.currentData()
        token = self.generation
        ref, digest = self.reference, self.reference_hash
        docs = [p.doc for p in self.owner.panes if p.doc is not None]
        if self.roi_future:
            self.roi_future.cancel()

        def work():
            drafts = {}
            for d in docs:
                mask = native_roi_mask(d.map, ref, vertices, h)
                drafts[d.sha256] = (
                    mask,
                    {
                        "height_rsun": h,
                        "reference_image_sha256": digest,
                        "display_polygon": vertices,
                        "interpretation": "display_assumption_not_measured_depth",
                    },
                )
            return token, drafts

        self.roi_future = self.executor.submit(work)
        self.status.setText("正在后台映射原图ROI草稿…")
        QTimer.singleShot(40, self.collect_roi)

    def collect_roi(self):
        if self.closed or self.roi_future is None:
            return
        if not self.roi_future.done():
            QTimer.singleShot(40, self.collect_roi)
            return
        future, self.roi_future = self.roi_future, None
        try:
            token, drafts = future.result()
            if token != self.generation:
                self.status.setText("视角或帧已改变，过期ROI草稿已丢弃，请重新圈选。")
                return
            self.native_drafts.update(drafts)
            for pane in self.owner.panes:
                pane.draw()
            self.status.setText(
                "原图中青色草稿保留断裂与缺测。请分别确认两侧ROI，再在原图选支段和对应点。"
            )
        except Exception as exc:
            self.status.setText(str(exc))

    @undoable("确认原图 ROI")
    def accept_roi(self, index):
        d = self.owner.panes[index].doc
        if d is None or d.sha256 not in self.native_drafts:
            return
        mask, evidence = self.native_drafts[d.sha256]
        self.owner.perform(lambda: d.set_native_roi(mask, evidence), index)
        self.native_drafts.pop(d.sha256, None)
        self.owner.redraw()

    def rebuild_pairs(self, *args, select=True):
        if not self.records:
            return
        self.play.stop()
        self.remember()
        from astropy.time import Time

        current = self.owner.panes[self.master.currentIndex() == 0].doc
        current_time = Time(current.info["midpoint_utc"]).unix if current else None
        records = self.records
        profile = getattr(self.owner, "event_profile", {}) or {}
        if not self.background.isChecked():
            start, end = profile.get("start_utc"), profile.get("end_utc")
            records = [
                r
                for r in records
                if (not start or Time(r["midpoint_utc"]) >= Time(start))
                and (not end or Time(r["midpoint_utc"]) <= Time(end))
            ]
        self.paired = pair_frames(
            records,
            "EUVI" if self.master.currentIndex() == 0 else "AIA",
            int(self.band.currentText()),
        )
        inst = "EUVI" if self.master.currentIndex() == 0 else "AIA"
        index = (
            int(
                np.argmin(
                    [
                        abs(Time(p[inst]["midpoint_utc"]).unix - current_time)
                        for p in self.paired
                    ]
                )
            )
            if self.paired and current_time
            else next((i for i, p in enumerate(self.paired) if p["AIA"] is not None), 0)
        )
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.paired) - 1))
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        if self.paired and select:
            self.select_time(index)
        elif not self.paired:
            self.status.setText("当前波段/时窗没有可用帧")

    def select_time(self, index):
        if not 0 <= index < len(self.paired):
            return
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("切换时间帧"):
            self.play.stop()
            if self.current_pair in self.paired:
                self.slider.blockSignals(True)
                self.slider.setValue(self.paired.index(self.current_pair))
                self.slider.blockSignals(False)
            return
        self.owner.update_parameters()
        self.remember()
        self.pending_frame = self.paired[index]
        self.request()

    def step(self, delta):
        if not self.paired:
            return
        index = self.slider.value() + delta
        if index >= len(self.paired) or index < 0:
            self.play.stop()
            return
        self.slider.setValue(index)

    def toggle_play(self):
        if self.play.isActive():
            self.play.stop()
        else:
            self.play.start()

    def play_step(self):
        if (
            self.future is None
            and self.roi_future is None
            and not self.debounce.isActive()
        ):
            self.step(1)

    @undoable("加载时序")
    def load_timeline(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("加载时序"):
            return
        path = Path(self.owner.validate(path))
        data = json.loads(path.read_text())
        if (
            data.get("schema") != "solarphysics.jet_lab.timeline"
            or data.get("version") != 1
        ):
            raise ValueError("Unsupported timeline")
        complete = path.parent / "COMPLETE.json"
        if not complete.is_file():
            raise ValueError("时序仍在同步，缺少完成清单")
        checks = json.loads(complete.read_text())["sha256"]
        if checks.get(path.name) != file_sha256(path):
            raise ValueError("时序清单校验失败")
        records = []
        for r in data["frames"]:
            r = dict(r)
            r["_path"] = str(self.owner.validate(path.parent / r["image"]))
            if file_sha256(r["_path"]) != r["image_sha256"]:
                raise ValueError("时序图像尚未完整同步")
            if r.get("difference"):
                r["_difference"] = str(
                    self.owner.validate(path.parent / r["difference"])
                )
                if file_sha256(r["_difference"]) != r["difference_sha256"]:
                    raise ValueError("差分尚未完整同步")
            records.append(r)
        self.records = records
        self.timeline_toggle.setChecked(True)
        self.rebuild_pairs()

    def open_timeline(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "打开timeline.json或session.json",
            str(self.owner.allowed_roots[0]),
            "JSON (*.json)",
        )
        if p:
            try:
                if Path(p).name == "session.json":
                    if not self.owner.can_discard():
                        return
                    self.restore_session(p)
                else:
                    self.load_timeline(p)
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    @undoable("几何预检")
    def precheck(self):
        self.play.stop()
        if getattr(self.owner, "native_only", False):
            return self.owner.calculate_3d()
        report = geometry_precheck([p.doc for p in self.owner.panes])
        from .workflow_guide import precheck_summary, annotation_signature

        self.owner.last_precheck = report
        self.owner.precheck_signature = annotation_signature(self.owner)
        self.owner.notice(precheck_summary(report))
        self.owner.raw_details.setPlainText(
            json.dumps(json_safe(report), ensure_ascii=False, indent=2)
        )
        self.owner.sections_box.setCurrentIndex(3)
        if hasattr(self.owner, "guide"):
            self.owner.guide.refresh()
        self.owner.set_view_mode("native")
        self.owner.parameter_toggle.setChecked(True)
        return report

    def cleanup(self):
        self.closed = True
        self.navigation.close()
        self.generation += 1
        for timer in [self.debounce, self.poll, self.play, self.flash]:
            timer.stop()
        for selector in self.selectors:
            selector.disconnect_events()
        if self.future:
            self.future.cancel()
        if self.roi_future:
            self.roi_future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
