"""Small PyQt6 annotation workbench; views retain their own native coordinates."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from solar_apps.platform.paths.native_dialog import validate_allowed_path
from solar_toolkit.map.jet_annotations import (
    JetDocument,
    file_sha256,
    load_session,
    pixel_hpc,
    save_session,
)


class ObservationPane(QWidget):
    def __init__(self, owner, index):
        super().__init__()
        self.owner, self.index = owner, index
        self.doc = None
        self.selector = None
        self.dragging = None
        self.drag_preview = None
        layout = QVBoxLayout(self)
        self.title = QLabel("AIA / 视角 A" if index == 0 else "EUVI / 视角 B")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.tabs = QTabBar()
        for label in ("原图", "分割叠加", "轴线与横截面"):
            self.tabs.addTab(label)
        self.tabs.setCurrentIndex(1)
        self.tabs.currentChanged.connect(lambda: self.draw())
        layout.addWidget(self.tabs)
        self.figure = Figure(figsize=(5, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.connections = [
            self.canvas.mpl_connect("button_press_event", self.press),
            self.canvas.mpl_connect("button_release_event", self.release),
            self.canvas.mpl_connect("motion_notify_event", self.motion),
        ]

    def draw(self, reset=False):
        self.disconnect_selector()
        limits = (
            (self.ax.get_xlim(), self.ax.get_ylim())
            if self.ax.images and not reset
            else None
        )
        self.ax.clear()
        self.drag_preview = None
        doc = self.doc
        if doc is None:
            self.ax.text(
                0.5, 0.5, "Open FITS", transform=self.ax.transAxes, ha="center"
            )
            self.canvas.draw_idle()
            return
        original = self.tabs.currentIndex() == 0
        data = doc.raw if original else doc.data
        finite = data[np.isfinite(data)]
        lo, hi = (
            np.percentile(finite, [self.owner.low.value(), self.owner.high.value()])
            if len(finite)
            else (0, 1)
        )
        hi = max(hi, lo + np.finfo(float).eps)
        scaled = np.clip((data - lo) / (hi - lo), 0, 1)
        if self.owner.stretch.currentText() == "Asinh":
            scaled = np.arcsinh(scaled * 10) / np.arcsinh(10)
        self.ax.imshow(
            np.ma.masked_invalid(scaled),
            origin="lower",
            cmap=self.owner.colour.currentText(),
            interpolation="nearest",
        )
        if not original:
            shown = doc.selected if doc.selected is not None else doc.labels > 0
            overlay = np.zeros((*shown.shape, 4))
            overlay[shown] = [0.1, 0.9, 0.7, 0.32]
            self.ax.imshow(overlay, origin="lower", interpolation="nearest")
        roi = doc.state["roi"]
        if roi:
            p = np.vstack([roi, roi[0]])
            self.ax.plot(*p.T, color="cyan", lw=1)
        if self.tabs.currentIndex() == 2:
            sk = np.asarray(doc.state["original_skeleton"])
            if len(sk):
                self.ax.scatter(*sk.T, s=2, color="white", alpha=0.5)
            axis = np.asarray(doc.state["axis"])
            if len(axis):
                self.ax.plot(*axis.T, color="#ffae45", lw=1.8)
                self.ax.scatter(*axis[[0, -1]].T, color=["lime", "red"], s=45, zorder=5)
                controls = np.asarray(doc.state["control_points"])
                if self.owner.mode.currentData() in {"edit", "add", "delete"} and len(
                    controls
                ):
                    self.ax.scatter(
                        *controls.T, s=9, facecolor="none", edgecolor="yellow"
                    )
            extension = np.asarray(doc.state["extension"])
            if len(extension):
                self.ax.plot(*extension.T, "--", color="magenta", lw=1.5)
            for section in self.owner.sections.get(self.index, []):
                p, normal = np.array(section["axis_pixel"]), np.array(
                    section["normal_pixel"]
                )
                line = np.array([p - 10 * normal, p + 10 * normal])
                self.ax.plot(*line.T, color="cyan", alpha=0.7, lw=0.7)
        for tie in doc.state["tiepoints"]:
            x, y = tie["pixel_xy"]
            self.ax.plot(x, y, "+", color="magenta", ms=10)
            self.ax.annotate(
                str(tie["number"]),
                (x, y),
                color="white",
                xytext=(5, 5),
                textcoords="offset points",
            )
        self.ax.set_xlabel("native pixel x (zero based)")
        self.ax.set_ylabel("native pixel y")
        dark = self.owner.palette().color(QPalette.ColorRole.Window).lightness() < 128
        text_colour = "#edf1f5" if dark else "#1b2430"
        self.ax.set_facecolor("#15191f" if dark else "white")
        self.ax.tick_params(colors=text_colour)
        self.ax.xaxis.label.set_color(text_colour)
        self.ax.yaxis.label.set_color(text_colour)
        self.ax.format_coord = self.format_coordinate
        if limits:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        info = doc.info
        self.title.setText(
            f'{info["observatory"]} / {info["detector"]} · {info["wavelength"]}\n曝光中点 UTC {info["midpoint_utc"]}\n{doc.path.name}'
        )
        self.canvas.draw_idle()
        self.set_mode(self.owner.mode.currentData())

    def format_coordinate(self, x, y):
        if not self.doc:
            return ""
        tx, ty = pixel_hpc(self.doc.map, [[x, y]])[0]
        return f"pixel ({x:.2f}, {y:.2f}) | HPC ({tx:.2f}, {ty:.2f}) arcsec"

    def disconnect_selector(self):
        if self.selector:
            self.selector.set_active(False)
            self.selector.disconnect_events()
            self.selector = None

    def set_mode(self, mode):
        self.disconnect_selector()
        if mode == "rectangle":

            def rectangle(a, b):
                vertices = [
                    [a.xdata, a.ydata],
                    [b.xdata, a.ydata],
                    [b.xdata, b.ydata],
                    [a.xdata, b.ydata],
                ]
                self.owner.perform(lambda: self.doc.set_roi(vertices), self.index)

            self.selector = RectangleSelector(
                self.ax, rectangle, useblit=False, button=[1], minspanx=2, minspany=2
            )
        elif mode == "polygon":
            self.selector = PolygonSelector(
                self.ax,
                lambda vertices: self.owner.perform(
                    lambda: self.doc.set_roi(vertices), self.index
                ),
                useblit=False,
            )

    def press(self, event):
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or self.doc is None
            or self.toolbar.mode
        ):
            return
        self.owner.active.setCurrentIndex(self.index)
        xy = [event.xdata, event.ydata]
        mode = self.owner.mode.currentData()
        if mode in {"rectangle", "polygon", "navigate"}:
            return

        def action():
            if mode == "select":
                self.doc.select(xy)
            elif mode == "inner":
                info = self.doc.trace(xy)
                self.owner.notice(info["reason"])
                self.tabs.setCurrentIndex(2)
            elif mode == "tie":
                self.doc.add_tiepoint(
                    xy,
                    self.owner.tie_id.value(),
                    self.owner.tie_status.currentData(),
                    self.owner.feature.text(),
                )
            elif mode == "remove_tie":
                ties = self.doc.state["tiepoints"]
                if not ties:
                    return
                index = int(
                    np.argmin(
                        [np.linalg.norm(np.array(t["pixel_xy"]) - xy) for t in ties]
                    )
                )
                self.doc.checkpoint("remove_tiepoint")
                ties.pop(index)
            elif mode == "extension":
                if not self.doc.state["axis"]:
                    raise ValueError("先确认实测轴线，再添加推测延伸")
                self.doc.checkpoint("conditional_extension")
                self.doc.state["extension"] = [self.doc.state["axis"][-1], xy]
            else:
                controls = np.asarray(self.doc.state["control_points"])
                if len(controls) < 2:
                    raise ValueError("先选择区域并指定内端")
                i = int(np.argmin(np.linalg.norm(controls - xy, axis=1)))
                if mode == "edit":
                    self.dragging = (i, controls.copy())
                elif mode == "delete":
                    self.doc.edit(np.delete(controls, i, axis=0))
                elif mode == "add":
                    midpoints = (controls[:-1] + controls[1:]) / 2
                    j = int(np.argmin(np.linalg.norm(midpoints - xy, axis=1)))
                    self.doc.edit(np.insert(controls, j + 1, xy, axis=0))

        self.owner.perform(action, self.index)

    def motion(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.owner.statusBar().showMessage(
                self.format_coordinate(event.xdata, event.ydata)
            )
            if self.dragging is not None:
                i, original = self.dragging
                controls = original.copy()
                controls[i] = [event.xdata, event.ydata]
                if self.drag_preview is None:
                    (self.drag_preview,) = self.ax.plot(
                        *controls.T, "--", color="cyan", lw=1
                    )
                else:
                    self.drag_preview.set_data(*controls.T)
                self.canvas.draw_idle()

    def release(self, event):
        if self.dragging is None:
            return
        i, controls = self.dragging
        self.dragging = None
        if event.inaxes == self.ax and event.xdata is not None:
            controls[i] = [event.xdata, event.ydata]
            self.owner.perform(lambda: self.doc.edit(controls), self.index)

    def cleanup(self):
        self.disconnect_selector()
        for cid in self.connections:
            self.canvas.mpl_disconnect(cid)
        self.connections.clear()
        self.figure.clear()


class JetLabWindow(QMainWindow):
    def __init__(self, allowed_roots):
        super().__init__()
        self.allowed_roots = tuple(allowed_roots)
        self.default_palette = QApplication.instance().palette()
        self.setWindowTitle("Jet Lab · 喷流提取测试版")
        self.resize(1440, 920)
        self.sections = {}
        self.manifest = None
        self.loaded_pair_index = -1
        self.sample_id = ""
        self.syncing_controls = False
        self.pending_parameters = None
        self.closed_cleanly = False
        main = QWidget()
        layout = QVBoxLayout(main)
        self.setCentralWidget(main)
        top = QHBoxLayout()
        layout.addLayout(top)
        for title, fn in (
            ("打开 A", lambda: self.open_dialog(0)),
            ("打开 B", lambda: self.open_dialog(1)),
            ("加载样本", self.manifest_dialog),
            ("恢复标注", self.restore_dialog),
            ("保存新标注", self.save_dialog),
            ("盲重复：清空重做", self.blind_repeat),
        ):
            button = QPushButton(title)
            button.clicked.connect(fn)
            top.addWidget(button)
        self.pairs = QComboBox()
        self.pairs.setMinimumWidth(240)
        self.pairs.activated.connect(self.select_pair)
        top.addWidget(self.pairs, 1)
        self.theme = QComboBox()
        self.theme.addItems(["Auto", "Light", "Dark"])
        self.theme.currentTextChanged.connect(self.apply_theme)
        top.addWidget(self.theme)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        self.panes = []
        controls = QWidget()
        form = QFormLayout(controls)
        self.active = QComboBox()
        self.active.addItems(["A / 左侧", "B / 右侧"])
        self.active.currentIndexChanged.connect(self.sync_controls)
        form.addRow("当前编辑", self.active)
        self.mode = QComboBox()
        for label, value in (
            ("浏览", "navigate"),
            ("矩形区域", "rectangle"),
            ("多边形区域（双击完成）", "polygon"),
            ("点击目标区域", "select"),
            ("点击内端 → 自动轴线", "inner"),
            ("拖动轴线点 / 端点", "edit"),
            ("增加控制点", "add"),
            ("删除控制点", "delete"),
            ("标记对应特征", "tie"),
            ("删除最近特征", "remove_tie"),
            ("推测延伸（虚线）", "extension"),
        ):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self.change_mode)
        form.addRow("操作", self.mode)
        self.source = QComboBox()
        self.source.addItems(["intensity", "difference"])
        self.source.activated.connect(
            lambda: self.perform(lambda: self.doc.set_source(self.source.currentText()))
        )
        form.addRow("测量数组", self.source)
        button = QPushButton("加载已配准差分")
        button.clicked.connect(self.difference_dialog)
        form.addRow(button)
        self.method = QComboBox()
        self.method.addItems(["percentile", "manual", "otsu"])
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e9, 1e9)
        self.value.setValue(85)
        self.value.setDecimals(3)
        self.polarity = QComboBox()
        self.polarity.addItems(["positive", "negative"])
        self.opening = QSpinBox()
        self.opening.setRange(0, 20)
        self.closing = QSpinBox()
        self.closing.setRange(0, 20)
        self.area = QSpinBox()
        self.area.setRange(0, 1000000)
        self.area.setValue(8)
        for label, widget in (
            ("阈值方法", self.method),
            ("阈值 / 百分位", self.value),
            ("增强 / 暗化", self.polarity),
            ("开运算半径（像素）", self.opening),
            ("闭运算半径（像素）", self.closing),
            ("最小面积（像素）", self.area),
        ):
            form.addRow(label, widget)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_parameters)
        for widget in (self.method, self.polarity):
            widget.currentIndexChanged.connect(self.schedule_parameters)
        for widget in (self.value, self.opening, self.closing, self.area):
            widget.valueChanged.connect(self.schedule_parameters)
        self.branch = QComboBox()
        self.branch.activated.connect(
            lambda i: self.perform(lambda: self.doc.choose_branch(i))
        )
        form.addRow("候选支段 · 待确认", self.branch)
        for title, fn in (
            ("确认当前支段", lambda: self.doc.confirm()),
            ("反转内外端", lambda: self.doc.reverse()),
            ("撤销", lambda: self.doc.undo()),
            ("清除 ROI", lambda: self.doc.set_roi(None)),
            ("计算长度 / 方向 / 宽度", self.measure),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, f=fn: self.perform(f))
            form.addRow(button)
        self.tie_id = QSpinBox()
        self.tie_id.setRange(1, 9999)
        form.addRow("两侧使用同一编号", self.tie_id)
        self.tie_status = QComboBox()
        for label, value in (
            ("可能对应", "possible"),
            ("已人工确认身份", "manual_confirmed"),
            ("不确定", "uncertain"),
        ):
            self.tie_status.addItem(label, value)
        form.addRow("身份判断 ≠ 三维验证", self.tie_status)
        self.feature = QLineEdit()
        self.feature.setPlaceholderText("亮结 / 弯折 / 连续亮脊…")
        form.addRow("特征说明", self.feature)
        self.low = QDoubleSpinBox()
        self.low.setRange(0, 49)
        self.low.setValue(1)
        self.high = QDoubleSpinBox()
        self.high.setRange(51, 100)
        self.high.setValue(99.5)
        self.stretch = QComboBox()
        self.stretch.addItems(["Asinh", "Linear"])
        self.colour = QComboBox()
        self.colour.addItems(["gray", "inferno", "viridis", "coolwarm"])
        for label, widget in (
            ("显示低百分位", self.low),
            ("显示高百分位", self.high),
            ("显示拉伸", self.stretch),
            ("显示色表", self.colour),
        ):
            form.addRow(label, widget)
            signal = (
                widget.valueChanged
                if isinstance(widget, QDoubleSpinBox)
                else widget.currentIndexChanged
            )
            signal.connect(lambda: self.redraw())
        self.message = QTextEdit()
        self.message.setReadOnly(True)
        self.message.setMinimumHeight(150)
        form.addRow(self.message)
        profile_button = QPushButton("查看横截面强度轮廓")
        profile_button.clicked.connect(lambda: self.perform(self.show_profiles))
        form.addRow(profile_button)
        self.profile_dialog = None
        self.panes = [ObservationPane(self, i) for i in (0, 1)]
        for pane in self.panes:
            splitter.addWidget(pane)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(310)
        splitter.addWidget(scroll)
        splitter.setSizes([550, 550, 330])
        self.notice(
            "先选择 ROI，再点击目标区域与内端。橙色轴线待人工确认；所有长度和方向均为投影量。"
        )
        self.apply_theme("Auto")

    @property
    def doc(self):
        doc = self.panes[self.active.currentIndex()].doc
        if doc is None:
            raise ValueError("请先打开当前视角的 FITS")
        return doc

    def validate(self, path, kind="file"):
        return validate_allowed_path(path, allowed_roots=self.allowed_roots, kind=kind)

    def notice(self, text):
        self.message.setPlainText(str(text))

    def perform(self, action, index=None):
        if index is not None:
            self.active.setCurrentIndex(index)
        try:
            self.sections.pop(self.active.currentIndex(), None)
            result = action()
            self.sync_controls()
            self.redraw()
            if isinstance(result, str):
                self.notice(result)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            self.notice(f"未应用操作：{exc}")

    def schedule_parameters(self):
        if not self.syncing_controls:
            self.pending_parameters = (
                self.active.currentIndex(),
                dict(
                    method=self.method.currentText(),
                    value=self.value.value(),
                    opening=self.opening.value(),
                    closing=self.closing.value(),
                    min_area=self.area.value(),
                    polarity=self.polarity.currentText(),
                ),
            )
            self.timer.start()

    def update_parameters(self):
        self.timer.stop()
        pending = self.pending_parameters
        self.pending_parameters = None
        if pending is not None:
            index, parameters = pending
            doc = self.panes[index].doc
            if doc is not None:
                self.perform(lambda: doc.parameters(**parameters))

    def sync_controls(self):
        if self.timer.isActive():
            self.update_parameters()
        self.timer.stop()
        if not self.panes or self.panes[self.active.currentIndex()].doc is None:
            return
        self.syncing_controls = True
        p = self.doc.state["parameters"]
        self.method.setCurrentText(p["method"])
        self.value.setValue(p["value"])
        self.polarity.setCurrentText(p["polarity"])
        self.opening.setValue(p["opening"])
        self.closing.setValue(p["closing"])
        self.area.setValue(p["min_area"])
        self.source.setCurrentText(self.doc.state["source"])
        self.branch.clear()
        for i, path in enumerate(self.doc.paths):
            self.branch.addItem(
                f"支段 {i+1} · {np.linalg.norm(np.diff(path,axis=0),axis=1).sum():.1f} px"
            )
        self.branch.setCurrentIndex(self.doc.state["branch_index"])
        self.syncing_controls = False
        if self.doc.selection_reason not in {"component_selected", "restored"}:
            self.notice(
                f"区域状态：{self.doc.selection_reason}\n实际阈值：{self.doc.threshold}\n轴线：{self.doc.state['axis_status']}"
            )

    def change_mode(self):
        self.redraw()
        for pane in self.panes:
            pane.set_mode(self.mode.currentData())

    def redraw(self):
        for pane in self.panes:
            pane.draw()

    def open_image(self, index, path):
        doc = JetDocument(self.validate(path))
        self.panes[index].disconnect_selector()
        self.panes[index].doc = doc
        self.sections.pop(index, None)
        self.active.setCurrentIndex(index)
        self.sync_controls()
        self.panes[index].draw(reset=True)
        self.panes[index].set_mode(self.mode.currentData())

    def can_discard(self):
        if self.timer.isActive():
            self.update_parameters()
        if not any(p.doc and p.doc.dirty for p in self.panes):
            return True
        answer = QMessageBox.question(
            self,
            "尚未保存",
            "存在未保存标注。保存为新记录后继续，还是放弃修改？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_dialog()
        return answer == QMessageBox.StandardButton.Discard

    def open_dialog(self, index):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开原始 FITS",
            str(self.allowed_roots[0]),
            "FITS (*.fits *.fts *.fit)",
        )
        if path and self.can_discard():
            self.perform(lambda: self.open_image(index, path))

    def difference_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "已配准差分（带 JETREG 和原图哈希）",
            str(self.allowed_roots[0]),
            "FITS (*.fits *.fts)",
        )
        if path:
            self.perform(lambda: self.doc.load_difference(self.validate(path)))

    def save_dialog(self):
        if self.timer.isActive():
            self.update_parameters()
        if not any(p.doc for p in self.panes):
            self.notice("没有可保存的观测")
            return False
        folder = QFileDialog.getExistingDirectory(
            self, "选择标注保存父目录", str(self.allowed_roots[0])
        )
        if not folder:
            return False
        try:
            destination = self.validate(
                Path(folder)
                / ("annotation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
                "output_directory",
            )
            result = save_session(
                destination, [p.doc for p in self.panes], sample_id=self.sample_id
            )
            self.notice(f"已保存新标注：\n{result}")
            return True
        except (ValueError, OSError) as exc:
            self.notice(str(exc))
            return False

    def restore_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "恢复 annotations.json", str(self.allowed_roots[0]), "JSON (*.json)"
        )
        if path and self.can_discard():
            self.perform(lambda: self.restore(path))

    def restore(self, path):
        docs, bundle = load_session(path, self.validate)
        for pane, doc in zip(self.panes, docs):
            pane.doc = doc
            pane.draw(reset=True)
        self.sample_id = bundle["sample_id"]
        self.sections.clear()
        self.sync_controls()

    def manifest_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 paired_samples.json",
            str(self.allowed_roots[0]),
            "JSON (*.json)",
        )
        if path and self.can_discard():
            self.perform(lambda: self.load_manifest(path))

    def load_manifest(self, path):
        path = self.validate(path)
        data = json.loads(path.read_text())
        if (
            data.get("schema") != "solarphysics.jet_lab.samples"
            or data.get("version") != 1
        ):
            raise ValueError("Unsupported sample manifest")
        for pair in data["pairs"]:
            for view in pair["views"]:
                for key in ("image", "difference"):
                    if view.get(key):
                        resolved = self.validate(path.parent / view[key])
                        if file_sha256(resolved) != view[key + "_sha256"]:
                            raise ValueError(
                                "Sample is still syncing or checksum mismatch"
                            )
        self.manifest = (path.parent, data)
        self.pairs.clear()
        for pair in data["pairs"]:
            self.pairs.addItem(f'{pair["id"]} · {pair["split"]}')
        first = next(
            (i for i, p in enumerate(data["pairs"]) if p["split"] == "tuning"), 0
        )
        self.pairs.setCurrentIndex(first)
        self.select_pair(first, checked=True)

    def select_pair(self, index, checked=False):
        if not self.manifest:
            return
        if not checked and not self.can_discard():
            self.pairs.setCurrentIndex(self.loaded_pair_index)
            return

        def load():
            base, data = self.manifest
            pair = data["pairs"][index]
            # Build both first: failure cannot leave a half-switched pair.
            docs = []
            for view in pair["views"]:
                doc = JetDocument(self.validate(base / view["image"]))
                if view.get("difference"):
                    doc.load_difference(self.validate(base / view["difference"]))
                docs.append(doc)
            for pane, doc in zip(self.panes, docs):
                pane.doc = doc
                pane.draw(reset=True)
            self.sample_id = pair["id"]
            self.loaded_pair_index = index
            self.sections.clear()

        self.perform(load)

    def blind_repeat(self):
        if not self.can_discard():
            return

        def reset():
            for pane in self.panes:
                if pane.doc:
                    old = pane.doc
                    pane.doc = JetDocument(old.path)
                    if old.difference_path:
                        pane.doc.load_difference(old.difference_path)
            self.sections.clear()
            self.sample_id = self.sample_id.split("__repeat")[0] + "__repeat"
            self.notice("已隐藏并清空本次视图标注；先前保存记录保留。请独立重复操作。")

        self.perform(reset)

    def measure(self):
        result = self.doc.measurements()
        self.sections[self.active.currentIndex()] = result.pop("widths")
        summary = json.dumps(result, ensure_ascii=False, indent=2)
        widths = self.sections[self.active.currentIndex()]
        summary += "\n横截面拟合：\n" + "\n".join(
            f'{i+1}: {r["gaussian"]["status"]}; mask={r["mask_width_arcsec"]}; FWHM={r["gaussian"].get("fwhm_arcsec")}'
            for i, r in enumerate(widths)
        )
        self.notice(summary)
        self.panes[self.active.currentIndex()].tabs.setCurrentIndex(2)
        return summary

    def show_profiles(self):
        measurements = self.doc.measurements()
        sections = measurements["widths"]
        if not sections:
            raise ValueError("先选择并确认轴线")
        if self.profile_dialog:
            self.profile_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("原始强度横截面 · FWHM 不是中心定位误差")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        choices = QComboBox()
        choices.addItems([f"横截面 {i+1}" for i in range(len(sections))])
        layout.addWidget(choices)
        figure = Figure(layout="constrained")
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)

        def plot(index):
            figure.clear()
            ax = figure.add_subplot(111)
            section = sections[index]
            fit = section["gaussian"]
            x = np.array(section["offset_pixel"])
            y = np.array(section["intensity"])
            ax.plot(x, y, ".-", label="original intensity")
            if fit["status"] == "valid":
                fitted = fit["background"] + fit["amplitude"] * np.exp(
                    -0.5 * ((x - fit["centre_pixel"]) / fit["sigma_pixel"]) ** 2
                )
                ax.plot(
                    x, fitted, label=f'Gaussian FWHM {fit["fwhm_arcsec"]:.2f} arcsec'
                )
            ax.set_title(fit["status"])
            ax.set_xlabel("normal offset / native pixel")
            ax.set_ylabel("intensity / s")
            ax.legend()
            canvas.draw_idle()

        choices.currentIndexChanged.connect(plot)
        plot(0)
        dialog.finished.connect(lambda: figure.clear())
        self.profile_dialog = dialog
        dialog.show()

    def apply_theme(self, mode):
        palette = QPalette(self.default_palette)
        if mode in {"Light", "Dark"}:
            dark = mode == "Dark"
            for role, colour in (
                (QPalette.ColorRole.Window, "#20242b" if dark else "#f6f7f9"),
                (QPalette.ColorRole.Base, "#15191f" if dark else "#ffffff"),
                (QPalette.ColorRole.Text, "#edf1f5" if dark else "#1b2430"),
                (QPalette.ColorRole.WindowText, "#edf1f5" if dark else "#1b2430"),
                (QPalette.ColorRole.Button, "#303741" if dark else "#e5e9ee"),
                (QPalette.ColorRole.ButtonText, "#edf1f5" if dark else "#1b2430"),
            ):
                palette.setColor(role, QColor(colour))
        self.setPalette(palette)
        dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        for pane in self.panes:
            pane.figure.set_facecolor("#20242b" if dark else "white")
        self.redraw()

    def closeEvent(self, event):
        if not self.can_discard():
            event.ignore()
            return
        self.timer.stop()
        if self.profile_dialog:
            self.profile_dialog.close()
        for pane in self.panes:
            pane.cleanup()
        self.closed_cleanly = True
        event.accept()
