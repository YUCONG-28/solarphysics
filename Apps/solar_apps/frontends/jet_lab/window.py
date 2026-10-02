"""Small PyQt6 annotation workbench; views retain their own native coordinates."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPalette, QKeySequence, QShortcut
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
    QToolBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from solar_apps.platform.paths.native_dialog import validate_allowed_path
from solar_toolkit.map.jet_annotations import (
    JetDocument,
    file_sha256,
    load_session,
    save_session,
)


from .observation_pane import ObservationPane
from .undo_history import SessionUndo, undoable
from .native_workflow import NativeWorkflow


class JetLabWindow(NativeWorkflow, QMainWindow):
    def __init__(self, allowed_roots, *, native_only=True, event_profile=None):
        super().__init__()
        self.native_only = native_only
        from .session_validation import validate_event_profile

        self.event_profile = validate_event_profile(event_profile or {})
        self.reconstruction_result = None
        self.reconstruction_signature = None
        self.allowed_roots = tuple(allowed_roots)
        self.default_palette = QApplication.instance().palette()
        self.setWindowTitle("Jet Lab")
        self.resize(1440, 920)
        self.sections = {}
        self.manifest = None
        self.loaded_pair_index = -1
        self.sample_id = ""
        self.syncing_controls = False
        self.pending_parameters = None
        self.closed_cleanly = False
        self.epipolar_anchor = None
        main = QWidget()
        layout = QVBoxLayout(main)
        self.setCentralWidget(main)
        top = QHBoxLayout()
        layout.addLayout(top)
        open_button = QPushButton("打开事件")
        open_button.clicked.connect(self.open_event_dialog)
        top.addWidget(open_button)
        for title, fn in [
            ("恢复标注", self.restore_dialog),
            ("保存新修订", self.save_dialog),
        ]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, action=fn: action())
            top.addWidget(button)
        self.undo_button = QPushButton("撤销")
        self.redo_button = QPushButton("重做")
        self.undo_button.clicked.connect(lambda: self.undo_history.move(-1))
        self.redo_button.clicked.connect(lambda: self.undo_history.move(1))
        top.addWidget(self.undo_button)
        top.addWidget(self.redo_button)
        self.pairs = QComboBox()
        self.pairs.setMinimumWidth(175)
        self.pairs.activated.connect(
            lambda index: (
                self.common.select_pair_async(index)
                if self.native_only
                else self.select_pair(index)
            )
        )
        top.addWidget(self.pairs, 1)
        self.theme = QComboBox()
        self.theme.addItems(["Auto", "Light", "Dark"])
        self.theme.currentTextChanged.connect(self.apply_theme)
        top.addWidget(self.theme)
        self.common_toggle = QPushButton("AIA共同视角 / 原始双视角")
        self.common_toggle.hide()
        view_bar = QHBoxLayout()
        layout.addLayout(view_bar)
        self.view_selector = QComboBox()
        for title, value in [
            ("原图标注", "native"),
            ("共同视角浏览", "common"),
            ("上下对照（大屏）", "split"),
        ]:
            self.view_selector.addItem(title, value)
        self.view_selector.currentIndexChanged.connect(
            lambda: self.set_view_mode(self.view_selector.currentData())
        )
        view_bar.addWidget(self.view_selector)
        guide_button = QPushButton("操作流程")
        view_bar.addWidget(guide_button)
        guide_button.clicked.connect(
            lambda: self.guide.setVisible(not self.guide.isVisible())
        )
        recovery_button = QPushButton("恢复草稿")
        view_bar.addWidget(recovery_button)
        recovery_button.clicked.connect(self.restore_recovery)
        reset_layout = QPushButton("重置布局")
        reset_layout.clicked.connect(self.reset_view_layout)
        view_bar.addWidget(reset_layout)
        self.parameter_toggle = QPushButton("参数面板")
        self.parameter_toggle.setCheckable(True)
        self.parameter_toggle.setChecked(True)
        self.parameter_toggle.toggled.connect(
            lambda checked: self.parameter_panel.setVisible(checked)
        )
        view_bar.addWidget(self.parameter_toggle)
        self.image_priority = QPushButton("图像优先")
        self.image_priority.setCheckable(True)
        self.image_priority.setToolTip(
            "收起导航和参数，放大图像；再次点击恢复之前的工作区。"
        )
        self.image_priority.toggled.connect(self.set_image_priority)
        view_bar.addWidget(self.image_priority)
        fit_button = QPushButton("适合窗口")
        fit_button.clicked.connect(self.fit_images)
        fit_button.setToolTip(
            "恢复整幅图的查看范围，保留 ROI、轴线和对应点。滚轮缩放，中键拖动。"
        )
        view_bar.addWidget(fit_button)
        view_bar.addStretch(1)
        self.view_mode = "native"
        self.native_focus = None
        self._priority_previous = None
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setHandleWidth(10)
        self.vertical_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.vertical_splitter, 1)
        splitter = self.native_splitter = QSplitter()
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.setMinimumHeight(320)
        self.panes = []
        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        form = QFormLayout()
        control_layout.addLayout(form)
        self.sections_box = QToolBox()
        control_layout.addWidget(self.sections_box)

        def section(title):
            page = QWidget()
            page_form = QFormLayout(page)
            self.sections_box.addItem(page, title)
            return page_form

        self.active = QComboBox()
        self.active.addItems(["A / 左侧", "B / 右侧"])
        self.active.currentIndexChanged.connect(self.sync_controls)
        form.addRow("当前编辑", self.active)
        form = section("区域与分割")
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
        form.addRow("重新分割所用数组", self.source)
        self.view_layer = QComboBox()
        self.view_layer.addItems(["原图", "已配准差分"])
        self.view_layer.currentIndexChanged.connect(self.redraw)
        form.addRow("仅切换显示图层", self.view_layer)
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
        form = section("轴线与对应点")
        self.branch = QComboBox()
        self.branch.activated.connect(
            lambda i: self.perform(lambda: self.doc.choose_branch(i))
        )
        form.addRow("候选支段 · 待确认", self.branch)
        for title, fn in (
            ("确认当前支段", lambda: self.doc.confirm()),
            ("反转内外端", lambda: self.doc.reverse()),
            ("清除 ROI", lambda: self.doc.set_roi(None)),
            ("计算长度 / 方向 / 宽度", self.measure),
        ):
            button = QPushButton(title)
            if title == "确认当前支段":
                self.confirm_button = button
                button.setEnabled(False)
                button.setToolTip("先选连通区域，再点击内端生成候选轴线")
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
        form = section("显示设置")
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
            signal.connect(self.display_changed)
        form = section("检查结果")
        self.message = QTextEdit()
        self.message.setReadOnly(True)
        self.message.setMinimumHeight(150)
        form.addRow(self.message)
        self.raw_details = QTextEdit()
        self.raw_details.setReadOnly(True)
        self.raw_details.hide()
        details = QPushButton("展开 / 收起原始记录")
        details.clicked.connect(
            lambda: self.raw_details.setVisible(not self.raw_details.isVisible())
        )
        form.addRow(details)
        form.addRow(self.raw_details)
        profile_button = QPushButton("查看横截面强度轮廓")
        profile_button.clicked.connect(lambda: self.perform(self.show_profiles))
        form.addRow(profile_button)
        self.profile_dialog = None
        self.panes = [ObservationPane(self, i) for i in (0, 1)]
        for pane in self.panes:
            splitter.addWidget(pane)
        scroll = self.parameter_panel = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(310)
        splitter.addWidget(scroll)
        splitter.setSizes([550, 550, 330])
        if native_only:
            from .native_session import NativeSession

            self.common = NativeSession(self, event_profile=event_profile)
        else:
            from .common_view import CommonView

            self.common = CommonView(self)
        self.vertical_splitter.addWidget(self.common)
        self.vertical_splitter.addWidget(splitter)
        self.vertical_splitter.handle(1).setCursor(Qt.CursorShape.SizeVerCursor)
        self.vertical_splitter.setStretchFactor(0, 4)
        self.vertical_splitter.setStretchFactor(1, 6)
        self.common.hide()
        self.common_toggle.clicked.connect(self.toggle_common)
        self.notice(
            "加载原始双视角，分别标记同一特征后计算三维候选；无需 ROI 或轴线。"
            if self.native_only
            else "先选择 ROI，再点击目标区域与内端。橙色轴线待人工确认；所有长度和方向均为投影量。"
        )
        self.apply_theme("Auto")
        self._split_sizes = [1, 1]
        self.last_precheck = None
        self.last_saved = False
        from .workflow_guide import WorkflowGuide
        from .recovery import Recovery

        self.guide = WorkflowGuide(self)
        self.recovery = Recovery(self)
        if native_only:
            self.setup_native_workflow(layout, view_bar)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1440, available.width() - 30), min(900, available.height() - 40)
        )
        self.undo_history = SessionUndo(self)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(lambda: self.undo_history.move(-1))
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(lambda: self.undo_history.move(1))
        self.setAccessibleName("Jet Lab 原始双视角重建")
        for widget in self.findChildren(QPushButton):
            widget.setAccessibleName(widget.text())
        self.pairs.setAccessibleName("观测配对样本")
        self.theme.setAccessibleName("界面主题")
        from .accessibility import install_accessibility

        self.accessibility = install_accessibility(self)

    def toggle_common(self):
        if self.native_only:
            return
        self.set_view_mode("native" if self.common.isVisible() else "common")

    def open_event_dialog(self):
        """Explicit buttons remain accessible on macOS and with keyboard input."""
        dialog = QDialog(self)
        dialog.setWindowTitle("选择数据来源")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("选择已有配对、时序，或分别加载两个视角的原图。"))
        for title, action in (
            ("加载配对样本", self.manifest_dialog),
            ("加载时序", self.common.open_timeline),
            ("打开左侧 FITS", lambda: self.open_dialog(0)),
            ("打开右侧 FITS", lambda: self.open_dialog(1)),
            ("盲重复：清空重做", self.blind_repeat),
        ):
            button = QPushButton(title)
            button.clicked.connect(
                lambda checked=False, fn=action: (dialog.accept(), fn())
            )
            layout.addWidget(button)
        cancel = QPushButton("取消")
        cancel.clicked.connect(dialog.reject)
        layout.addWidget(cancel)
        dialog.exec()

    def set_view_mode(self, mode):
        if self.native_only:
            mode = "native"
        if self.view_mode == "split" and mode != "split":
            self._split_sizes = self.vertical_splitter.sizes()
        self.view_mode = mode
        self.view_selector.blockSignals(True)
        self.view_selector.setCurrentIndex(self.view_selector.findData(mode))
        self.view_selector.blockSignals(False)
        self.common.setVisible(mode != "native")
        self.native_splitter.setVisible(mode != "common")
        self.common.original.blockSignals(True)
        self.common.original.setChecked(mode != "common")
        self.common.original.blockSignals(False)
        self.parameter_toggle.setEnabled(mode != "common")
        for index, pane in enumerate(self.panes):
            pane.setVisible(self.native_focus in (None, index))
        if mode == "native":
            self.common.play.stop()
            self.common.flash.stop()
        else:
            self.common.request()
            if mode == "split":
                self.vertical_splitter.setSizes(
                    getattr(self, "_split_sizes", [400, 600])
                )

    def reset_view_layout(self):
        self._split_sizes = [1, 1]
        self.native_focus = None
        self.set_view_mode(self.view_mode)
        if self.view_mode == "split":
            self.vertical_splitter.setSizes([1, 1])
        for pane in self.panes:
            pane.enlarge.setText("放大此图")
        self.native_splitter.setSizes([550, 550, 330])

    def focus_native(self, index):
        if self.native_focus is None:
            self._native_pair_sizes = self.native_splitter.sizes()
        self.native_focus = None if self.native_focus == index else index
        self.active.setCurrentIndex(index)
        self.set_view_mode("native")
        if self.native_focus is None:
            self.native_splitter.setSizes(
                getattr(self, "_native_pair_sizes", [550, 550, 330])
            )
        else:
            panel = (
                min(380, max(260, self.parameter_panel.width()))
                if self.parameter_toggle.isChecked()
                else 0
            )
            sizes = [0, 0, panel]
            sizes[index] = max(1, self.native_splitter.width() - panel)
            self.native_splitter.setSizes(sizes)
        for i, pane in enumerate(self.panes):
            pane.enlarge.setText("返回双图" if self.native_focus == i else "放大此图")

    def set_image_priority(self, enabled):
        if enabled:
            self._priority_previous = (
                not self.guide.isHidden(),
                self.parameter_toggle.isChecked(),
                self.view_mode,
            )
            self.guide.hide()
            self.parameter_toggle.setChecked(False)
            if self.view_mode == "split":
                self.set_view_mode("common")
        elif self._priority_previous:
            guide, parameters, mode = self._priority_previous
            self.guide.setVisible(guide)
            self.parameter_toggle.setChecked(parameters)
            self.set_view_mode(mode)

    def fit_images(self):
        """Restore the full displayed images, without modifying measurement ROI."""
        if self.view_mode != "native":
            self.common.limits = None
            self.common.render()
        if self.view_mode != "common":
            for pane in self.panes:
                if pane.doc:
                    h, w = pane.doc.raw.shape
                    pane.ax.set_xlim(-0.5, w - 0.5)
                    pane.ax.set_ylim(-0.5, h - 0.5)
                    pane.canvas.draw_idle()

    def display_changed(self, *args):
        if self.panes and not self.syncing_controls:
            self.panes[self.active.currentIndex()].display_limits = [
                self.low.value(),
                self.high.value(),
            ]
        self.redraw()

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
        if self.native_only:
            self.statusBar().showMessage(str(text).replace("\n", " "), 20000)

    @undoable("编辑标注 / 计算")
    def perform(self, action, index=None):
        if index is not None:
            self.active.setCurrentIndex(index)
        try:
            self.sections.pop(self.active.currentIndex(), None)
            result = action()
            self.last_precheck = None
            self.last_saved = False
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
            self.confirm_button.setEnabled(False)
            return
        self.confirm_button.setEnabled(bool(self.doc.state["axis"]))
        if self.native_focus is not None:
            self.native_focus = self.active.currentIndex()
            for i, pane in enumerate(self.panes):
                pane.setVisible(i == self.native_focus)
                pane.enlarge.setText(
                    "返回双图" if i == self.native_focus else "放大此图"
                )
        for index, pane in enumerate(self.panes):
            if pane.doc:
                self.active.setItemText(
                    index,
                    f"{'● ' if index == self.active.currentIndex() else ''}{pane.doc.info['detector']} / {'左侧' if index == 0 else '右侧'}",
                )
        self.syncing_controls = True
        p = self.doc.state["parameters"]
        self.method.setCurrentText(p["method"])
        self.value.setValue(p["value"])
        self.polarity.setCurrentText(p["polarity"])
        self.opening.setValue(p["opening"])
        self.closing.setValue(p["closing"])
        self.area.setValue(p["min_area"])
        self.source.setCurrentText(self.doc.state["source"])
        self.low.blockSignals(True)
        self.high.blockSignals(True)
        self.low.setValue(self.panes[self.active.currentIndex()].display_limits[0])
        self.high.setValue(self.panes[self.active.currentIndex()].display_limits[1])
        self.low.blockSignals(False)
        self.high.blockSignals(False)
        self.branch.clear()
        for i, path in enumerate(self.doc.paths):
            self.branch.addItem(
                f"支段 {i + 1} · {np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.1f} px"
            )
        self.branch.setCurrentIndex(self.doc.state["branch_index"])
        self.syncing_controls = False
        if not self.native_only and self.doc.selection_reason not in {
            "component_selected",
            "restored",
        }:
            reason = {
                "select_component": "请点击目标连通区域，再点击内端生成轴线",
                "component_disappeared": "原区域消失，请调整阈值后重新选择",
                "component_split": "区域发生分裂，请重新确认支段",
            }.get(self.doc.selection_reason, "区域改变，请重新确认目标支段")
            self.notice(
                f"{reason}\n实际阈值：{self.doc.threshold}\n"
                "断裂可能来自阈值、缺测或结构不连续；可查看差分，不自动跨缺口连接。"
            )

    def change_mode(self):
        self.redraw()
        for pane in self.panes:
            pane.set_mode(self.mode.currentData())

    def redraw(self, *args):
        if self.native_only:
            self.refresh_reconstruction()
        if hasattr(self, "guide"):
            self.guide.refresh()
        for pane in self.panes:
            pane.draw()
        if not self.native_only and hasattr(self, "common") and self.common.isVisible():
            self.common.request()

    @undoable("加载原图")
    def open_image(self, index, path):
        if self.native_only and not self.ensure_point_metadata_applied("加载原图"):
            return
        doc = JetDocument(self.validate(path), lazy_segmentation=self.native_only)
        self.panes[index].disconnect_selector()
        self.panes[index].doc = doc
        self.sections.pop(index, None)
        self.active.setCurrentIndex(index)
        self.sync_controls()
        self.panes[index].draw(reset=True)
        self.panes[index].set_mode(self.mode.currentData())
        if hasattr(self, "common"):
            self.common.request()

    @undoable("检查数据")
    def show_processing(self):
        from .workflow_guide import readable_reason

        entries = [p.doc.info.get("preprocessing", {}) for p in self.panes if p.doc]
        self.raw_details.setPlainText(json.dumps(entries, ensure_ascii=False, indent=2))
        lines = [
            "Python 预处理：仅按声明单位处理曝光，保留头部指向。",
            "未执行完整仪器校准。",
        ]
        for pane in self.panes:
            if pane.doc:
                info = pane.doc.info
                prep = info.get("preprocessing", {})
                lines.append(
                    f"{info['detector']}：输出单位 {prep.get('output_unit') or '未声明'}；头部曝光 {info['exposure_s']:.4g} 秒"
                )
                reasons = prep.get("reasons", []) + info.get("geometry_issues", [])
                lines.extend("• " + readable_reason(r) for r in dict.fromkeys(reasons))
        self.notice("\n".join(lines))
        self.set_view_mode("native")
        self.sections_box.setCurrentIndex(3)

    def restore_recovery(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "恢复本地草稿", str(self.recovery.directory), "JSON (*.json)"
        )
        if path and self.can_discard():
            if self.native_only:
                self.common.restore_recovery_async(path)
            else:
                self.perform(lambda: self.recovery.restore(path))

    def can_discard(self):
        if self.native_only and not self.ensure_point_metadata_applied("离开当前标注"):
            return False
        if self.timer.isActive():
            self.update_parameters()
        docs = list(
            getattr(getattr(self, "common", None), "documents", {}).values()
        ) + [p.doc for p in self.panes if p.doc]
        if not any(d.dirty for d in docs) and not self.common.has_unsaved():
            return True
        answer = QMessageBox.question(
            self,
            "尚未保存",
            "存在未保存标注、拟合结果或设置。保存为新记录后继续，还是放弃修改？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            if self.common.records:
                self.common.save_dialog()
                return not self.common.has_unsaved()
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
            if self.native_only:
                self.common.open_image_async(index, path)
            else:
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

    @undoable("保存新修订")
    def save_dialog(self):
        if self.native_only and not self.ensure_point_metadata_applied("保存修订"):
            return False
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
            if self.native_only:
                result = self.common.save_to(destination)
            else:
                result = save_session(
                    destination, [p.doc for p in self.panes], sample_id=self.sample_id
                )
            if result is None:
                return False
            self.notice(f"已保存新标注：\n{result}")
            self.last_saved = True
            self.last_output = str(result)
            from .workflow_guide import annotation_signature

            self.saved_signature = annotation_signature(self)
            self.guide.refresh()
            return True
        except (ValueError, OSError) as exc:
            self.notice(str(exc))
            return False

    def restore_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "恢复 session.json 或 annotations.json",
            str(self.allowed_roots[0]),
            "JSON (*.json)",
        )
        if path and self.can_discard():
            if self.native_only:
                self.common.restore_annotations_async(path)
            else:
                self.perform(lambda: self.restore(path))

    @undoable("恢复标注")
    def restore(self, path):
        if self.native_only and not self.ensure_point_metadata_applied("恢复标注"):
            return
        bundle_info = json.loads(self.validate(path).read_text())
        if getattr(self, "reconstruction_dialog", None):
            self.reconstruction_dialog.close()
            self.reconstruction_dialog = None
        if bundle_info.get("schema") == "solarphysics.jet_lab.session":
            self.common.restore_session(path)
            if self.native_only:
                self.sync_point_editor()
            self.redraw()
            return
        docs, bundle = load_session(
            path, self.validate, lazy_segmentation=self.native_only
        )
        for pane, doc in zip(self.panes, docs):
            pane.doc = doc
            pane.draw(reset=True)
        self.sample_id = bundle["sample_id"]
        self.sections.clear()
        self.sync_controls()
        if self.native_only:
            self.common.reset_science_for_annotations()
            self.sync_point_editor()
        if hasattr(self, "common"):
            self.common.remember()
            self.common.request()

    def manifest_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 paired_samples.json",
            str(self.allowed_roots[0]),
            "JSON (*.json)",
        )
        if path and self.can_discard():
            if self.native_only:
                self.common.load_manifest_async(path)
            else:
                self.perform(lambda: self.load_manifest(path))

    @undoable("加载配对样本")
    def load_manifest(self, path):
        if self.native_only and not self.ensure_point_metadata_applied("加载配对样本"):
            return
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
            self.pairs.addItem(f"{pair['id']} · {pair['split']}")
        first = next(
            (i for i, p in enumerate(data["pairs"]) if p["split"] == "tuning"), 0
        )
        self.pairs.setCurrentIndex(first)
        self.select_pair(first, checked=True)

    @undoable("切换样本")
    def select_pair(self, index, checked=False):
        if not self.manifest:
            return
        if self.native_only and not self.ensure_point_metadata_applied("切换样本"):
            self.pairs.blockSignals(True)
            self.pairs.setCurrentIndex(self.loaded_pair_index)
            self.pairs.blockSignals(False)
            return
        if not self.native_only and not checked and not self.can_discard():
            self.pairs.setCurrentIndex(self.loaded_pair_index)
            return

        def load():
            self.common.remember()
            base, data = self.manifest
            pair = data["pairs"][index]
            # Build both first: failure cannot leave a half-switched pair.
            docs = []
            for view in pair["views"]:
                path = self.validate(base / view["image"])
                digest = view["image_sha256"]
                doc = self.common.documents.get(digest)
                if doc is None and digest in self.common.document_snapshots:
                    doc = self.common.thaw(self.common.document_snapshots[digest])
                if doc is None:
                    doc = JetDocument(path, lazy_segmentation=self.native_only)
                if view.get("difference") and not doc.difference_path:
                    doc.load_difference(self.validate(base / view["difference"]))
                docs.append(doc)
            from .native_frames import sample_pairing

            pairing = sample_pairing(pair, docs)
            for pane, doc in zip(self.panes, docs):
                pane.doc = doc
                pane.draw(reset=True)
            self.sample_id = pair["id"]
            self.common.current_pair = pairing
            self.loaded_pair_index = index
            self.pairs.setCurrentIndex(index)
            if not self.common.records:
                self.common.band.blockSignals(True)
                wavelength = docs[0].map.meta.get("wavelnth")
                if wavelength in (171, 304):
                    self.common.band.setCurrentText(str(int(wavelength)))
                self.common.band.blockSignals(False)
            self.sections.clear()
            if self.native_only:
                self.sync_point_editor()

        self.perform(load)

    @undoable("清空重做")
    def blind_repeat(self):
        if not self.can_discard():
            return

        def reset():
            for pane in self.panes:
                if pane.doc:
                    old = pane.doc
                    pane.doc = JetDocument(old.path, lazy_segmentation=self.native_only)
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
            f"{i + 1}: {r['gaussian']['status']}; mask={r['mask_width_arcsec']}; FWHM={r['gaussian'].get('fwhm_arcsec')}"
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
        if getattr(self, "reconstruction_dialog", None):
            self.reconstruction_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("原始强度横截面 · FWHM 不是中心定位误差")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        choices = QComboBox()
        choices.addItems([f"横截面 {i + 1}" for i in range(len(sections))])
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
                    x, fitted, label=f"Gaussian FWHM {fit['fwhm_arcsec']:.2f} arcsec"
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
        # Qt containers and Matplotlib toolbars may hold explicit palettes.
        for child in self.findChildren(QWidget):
            child.setPalette(palette)
        dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        bg = palette.window().color().name()
        fg = palette.windowText().color().name()
        base = palette.base().color().name()
        button = palette.button().color().name()
        self.setStyleSheet(
            f"QWidget {{color: {fg}; background-color: {bg};}}"
            f"QLineEdit, QTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{background-color: {base};}}"
            f"QPushButton, QToolButton, QToolBox::tab, QTabBar::tab {{background-color: {button}; padding: 4px;}}"
            "QPushButton:disabled, QToolButton:disabled {color: #808892;}"
            "QListWidget::item:selected, QToolBox::tab:selected, QTabBar::tab:selected {background-color: #2877af; color: white;}"
        )
        handle = "#58616d" if dark else "#b3bdc9"
        for splitter in [self.vertical_splitter, self.native_splitter]:
            splitter.setStyleSheet(
                f"QSplitter::handle {{background: {handle};}} "
                "QSplitter::handle:hover {background: #398ad7;}"
            )
        for pane in self.panes:
            pane.figure.set_facecolor("#20242b" if dark else "white")
        self.redraw()

    def closeEvent(self, event):
        if not self.can_discard():
            event.ignore()
            return
        if hasattr(self, "accessibility"):
            self.accessibility.close()
        self.timer.stop()
        if hasattr(self, "_fit_executor"):
            self.close_fit_workflow()
        self.recovery.close()
        self.undo_history.close()
        if self.guide.dialog:
            self.guide.dialog.close()
        self.common.cleanup()
        if self.profile_dialog:
            self.profile_dialog.close()
        for pane in self.panes:
            pane.cleanup()
        self.closed_cleanly = True
        event.accept()
