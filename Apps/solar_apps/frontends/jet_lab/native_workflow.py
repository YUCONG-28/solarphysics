"""Correspondence-first native viewing; no image reprojection or depth prior."""

from copy import deepcopy
import hashlib
import json

import numpy as np
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QStackedWidget,
    QScrollArea,
    QTabWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from .undo_history import undoable
from .native_frames import native_pair_status
from .point_metadata_editor import PointMetadataEditor
from .reconstruction_view import (
    axis_description,
    reconstruction_quality,
    draw_reconstruction,
)
from .fit_workflow import FitWorkflow, fit_description


class NativeWorkflow(FitWorkflow, PointMetadataEditor):
    def setup_native_workflow(self, layout, view_bar):
        """Retain auxiliary controls without letting them gate triangulation."""
        self.view_selector.hide()
        self.parameter_toggle.setText("对应点面板")
        for position, (title, action) in enumerate(
            (
                ("前一帧", lambda: self.native_step(-1)),
                ("后一帧", lambda: self.native_step(1)),
                ("标记对应点", self.start_marking),
                ("计算三维", self.calculate_and_show),
            )
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, fn=action: fn())
            view_bar.insertWidget(position, button)
        self.native_times = QLabel("请加载一组原图配对")
        self.native_times.setWordWrap(True)
        layout.insertWidget(2, self.native_times)
        layout.insertWidget(3, self.common.timeline_controls)
        self.common.timeline_controls.hide()
        # A stacked third panel preserves the existing three-pane sizing contract.
        auxiliary = self.parameter_panel
        self.parameter_panel = QStackedWidget()
        self.parameter_panel.setMinimumWidth(295)
        self.parameter_panel.setMaximumWidth(420)
        self.native_splitter.replaceWidget(2, self.parameter_panel)
        panel = QWidget()
        box = QVBoxLayout(panel)
        heading = QLabel("对应点 · 原始双视角")
        heading.setStyleSheet("font-weight: bold; font-size: 15px")
        box.addWidget(heading)
        self.point_hint = QLabel("依次在两幅原图中标记同一特征。无需分割或轴线。")
        self.point_hint.setWordWrap(True)
        box.addWidget(self.point_hint)
        self.point_table = QTableWidget(0, 6)
        self.point_table.setHorizontalHeaderLabels(
            ["编号", "A", "B", "用途", "顺序", "端点"]
        )
        self.point_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.point_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.point_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.point_table.horizontalHeader().setStretchLastSection(True)
        self.point_table.verticalHeader().hide()
        self.point_table.setMaximumHeight(205)
        self.point_table.cellClicked.connect(self.select_point_row)
        box.addWidget(self.point_table)
        # Reuse the original identity widgets; saved formats and undo stay compatible.
        form = QFormLayout()
        box.addLayout(form)
        for text, widget in (
            ("当前编号", self.tie_id),
            ("特征说明", self.feature),
            ("身份判断", self.tie_status),
        ):
            old = widget.parentWidget().layout()
            if isinstance(old, QFormLayout):
                old.labelForField(widget).hide()
                old.removeWidget(widget)
            form.addRow(text, widget)
        self.point_role = QComboBox()
        for label, value in (
            ("喷流点", "jet"),
            ("周围参照点", "reference"),
            ("待指定", "unknown"),
        ):
            self.point_role.addItem(label, value)
        self.point_order = QSpinBox()
        self.point_order.setRange(1, 9999)
        self.point_endpoint = QComboBox()
        for label, value in (
            ("中间点 / 未指定", None),
            ("内端", "inner"),
            ("外端", "outer"),
        ):
            self.point_endpoint.addItem(label, value)
        form.addRow("用途", self.point_role)
        form.addRow("沿喷流顺序", self.point_order)
        # Direct actions also work with macOS accessibility clients whose Qt
        # popup-menu activation is unreliable. The combo retains undo state.
        self.point_endpoint.setParent(panel)
        self.point_endpoint.hide()
        endpoints = QWidget()
        endpoint_row = QHBoxLayout(endpoints)
        endpoint_row.setContentsMargins(0, 0, 0, 0)
        for title, index in (("设为内端", 1), ("中间点", 0), ("设为外端", 2)):
            button = QPushButton(title)
            button.clicked.connect(
                lambda checked=False, value=index: self.assign_endpoint(value)
            )
            endpoint_row.addWidget(button)
        form.addRow("端点", endpoints)
        self.tie_id.valueChanged.connect(self.point_number_changed)
        self.metadata_pending_label = QLabel()
        self.metadata_pending_label.setWordWrap(True)
        box.addWidget(self.metadata_pending_label)
        for widget in (self.tie_status, self.point_role, self.point_endpoint):
            widget.currentIndexChanged.connect(self.refresh_point_metadata_pending)
        self.feature.textChanged.connect(self.refresh_point_metadata_pending)
        self.point_order.valueChanged.connect(self.refresh_point_metadata_pending)
        self.reset_point_editor_baseline()
        for titles in [
            (
                ("应用点说明", self.apply_point_metadata),
                ("下一特征", self.next_feature),
            ),
            (
                ("删除此编号", self.delete_feature),
                ("精确坐标…", self.coordinate_dialog),
            ),
            (
                ("查看三维 / 残差", self.show_reconstruction_results),
                ("辅助工具", self.show_auxiliary),
            ),
        ]:
            row = QHBoxLayout()
            box.addLayout(row)
            for title, action in titles:
                button = QPushButton(title)
                button.clicked.connect(lambda checked=False, fn=action: fn())
                row.addWidget(button)
        self.result_summary = QLabel(
            "尚未计算。所得空间点是对应身份和时间稳定性假设下的候选。"
        )
        self.result_summary.setWordWrap(True)
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setMinimumHeight(75)
        result_scroll.setMaximumHeight(210)
        result_scroll.setWidget(self.result_summary)
        box.addWidget(result_scroll)
        self.setup_fit_workflow(box)
        box.addStretch()
        self.parameter_panel.addWidget(panel)
        aux_container = QWidget()
        aux_layout = QVBoxLayout(aux_container)
        back = QPushButton("返回对应点")
        back.clicked.connect(lambda: self.parameter_panel.setCurrentIndex(0))
        aux_layout.addWidget(back)
        aux_layout.addWidget(auxiliary)
        self.parameter_panel.addWidget(aux_container)
        auxiliary.show()
        self.native_splitter.setSizes([530, 530, 310])
        self.guide.hide()
        self.setWindowTitle("Jet Lab · 原始双视角三维重建")
        self.reconstruction_dialog = None
        self.start_marking()

    def geometry_signature(self):
        from solar_toolkit.map.jet_reconstruction import RECONSTRUCTION_VERSION
        from solar_toolkit.map.jet_geometry_fit import GEOMETRY_FIT_VERSION

        payload = {
            "documents": [
                (
                    (
                        p.doc.sha256,
                        p.doc.info.get("midpoint_utc"),
                        p.doc.state["tiepoints"],
                    )
                    if p.doc
                    else None
                )
                for p in self.panes
            ],
            "pairing": self.common.pair_metadata(),
            "algorithm_version": RECONSTRUCTION_VERSION,
            "fit_version": GEOMETRY_FIT_VERSION,
            "fit_options": getattr(self, "fit_options", {}),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def reconstruction_current(self):
        return (
            self.reconstruction_result is not None
            and self.reconstruction_signature == self.geometry_signature()
        )

    def refresh_reconstruction(self):
        if not hasattr(self, "result_summary"):
            return
        self.native_times.setText(
            native_pair_status([p.doc for p in self.panes], self.common.current_pair)
        )
        if self.reconstruction_result is None:
            text = "尚未计算。无需分割或轴线；至少一对对应点可计算候选坐标。"
            if getattr(self, "reconstruction_dialog", None):
                self.reconstruction_dialog.close()
                self.reconstruction_dialog = None
        elif not self.reconstruction_current():
            text = "结果已过期：原图、配对或标注已改变。请重新计算；历史文件保留。"
            if getattr(self, "reconstruction_dialog", None):
                self.reconstruction_dialog.setWindowTitle("已过期 · 历史三维候选")
                self.reconstruction_dialog.result_status.setText(text)
        else:
            r = self.reconstruction_result
            count = sum(bool(p.get("numerical_valid")) for p in r["points"])
            s = r.get("summary", {})
            text = f"已算出 {count} 个空间候选点。"
            length = s.get("polyline_length_Mm")
            if length is not None:
                label = "已支持分段合计" if s.get("polyline_has_gaps") else "有序折线"
                text += f"\n{label} {length:.3f} Mm（不补造缺失轴段）。"
            if s.get("chord_Mm") is not None:
                text += f"\n指定内外端弦长 {s['chord_Mm']:.3f} Mm。"
            text += (
                "\n绿色圆圈为结果回投；洋红十字为原图标点。无误差依据时不报告置信度。"
            )
            text += "\n" + axis_description(s.get("axis"))
            text += "\n" + fit_description(r.get("joint_fits"))
            text += "\n" + "\n".join(reconstruction_quality(r))
            if s.get("incomplete"):
                text += "\n长度约束不完整，请查看缺段、顺序或时间条件。"
            from .workflow_guide import readable_reason

            reasons = list(dict.fromkeys(r.get("issues", []) + s.get("issues", [])))
            text += "\n" + "；".join(map(readable_reason, reasons)) if reasons else ""
        self.result_summary.setText(text)
        self.refresh_point_table()

    def refresh_point_table(self):
        if not hasattr(self, "point_table"):
            return
        lookup = [
            {t["number"]: t for t in p.doc.state["tiepoints"]} if p.doc else {}
            for p in self.panes
        ]
        numbers = sorted(set(lookup[0]) | set(lookup[1]))
        self.point_table.setRowCount(len(numbers))
        n = self.tie_id.value()
        present = [n in side for side in lookup]
        if all(present):
            hint = f"编号 {n} 两侧已标记；请核对身份，再选下一特征。"
        elif any(present):
            hint = f"编号 {n}：请在{'EUVI 右侧' if present[0] else 'AIA 左侧'}点击同一特征；黄虚线仅作提示。"
        else:
            hint = f"编号 {n} 尚未标记。在两幅原图分别点击同一特征，无需分割或轴线。"
        self.point_hint.setText(hint)
        roles = {"jet": "喷流", "reference": "参照", "unknown": "待指定"}
        for row, n in enumerate(numbers):
            t = lookup[0].get(n, lookup[1].get(n))
            values = [
                n,
                "✓" if n in lookup[0] else "缺",
                "✓" if n in lookup[1] else "缺",
                roles.get(t.get("role", "unknown"), "冲突"),
                t.get("order") or "—",
                {"inner": "内", "outer": "外", None: "—"}.get(t.get("endpoint"), "—"),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(t.get("feature", ""))
                self.point_table.setItem(row, col, item)

    def select_point_row(self, row, _column):
        self.tie_id.setValue(int(self.point_table.item(row, 0).text()))
        # The valueChanged handler protects pending descriptions, including
        # cancelling a row selection. Do not overwrite the editor a second time.

    def sync_point_editor(self):
        if not hasattr(self, "point_role"):
            return
        n = self.tie_id.value()
        t = next(
            (
                t
                for p in self.panes
                if p.doc
                for t in p.doc.state["tiepoints"]
                if t["number"] == n
            ),
            None,
        )
        widgets = [
            self.feature,
            self.tie_status,
            self.point_role,
            self.point_order,
            self.point_endpoint,
        ]
        blockers = [QSignalBlocker(w) for w in widgets]
        if t:
            self.feature.setText(t.get("feature", ""))
            self.tie_status.setCurrentIndex(
                self.tie_status.findData(t.get("identity_status", "possible"))
            )
            self.point_role.setCurrentIndex(
                self.point_role.findData(t.get("role", "unknown"))
            )
            self.point_order.setValue(t.get("order") or n)
            self.point_endpoint.setCurrentIndex(
                self.point_endpoint.findData(t.get("endpoint"))
            )
        else:
            self.feature.clear()
            self.tie_status.setCurrentIndex(0)
            self.point_role.setCurrentIndex(0)
            self.point_order.setValue(n)
            self.point_endpoint.setCurrentIndex(0)
        del blockers
        self.reset_point_editor_baseline()

    def start_marking(self):
        self.set_view_mode("native")
        self.mode.setCurrentIndex(self.mode.findData("tie"))
        for pane in self.panes:
            if pane.toolbar.mode:
                if str(pane.toolbar.mode) == "pan/zoom":
                    pane.toolbar.pan()
                else:
                    pane.toolbar.zoom()
            pane.update_interaction_status()
        if hasattr(self, "point_hint"):
            self.point_hint.setText(
                "在 AIA 和 EUVI 原图分别点击同一特征。重选同编号会移动该侧点，可撤销。"
            )
            self.parameter_panel.setCurrentIndex(0)
            self.parameter_toggle.setChecked(True)

    def point_metadata(self):
        role = self.point_role.currentData()
        return dict(
            role=role,
            order=self.point_order.value() if role == "jet" else None,
            endpoint=self.point_endpoint.currentData() if role == "jet" else None,
        )

    @undoable("指定喷流端点")
    def assign_endpoint(self, index):
        self.point_endpoint.setCurrentIndex(index)
        self.apply_point_metadata()

    def mark_native_point(self, index, xy):
        doc = self.panes[index].doc
        number = self.tie_id.value()
        was_pending = self.point_metadata_pending()
        existing = next(
            (t for t in doc.state["tiepoints"] if t["number"] == number), None
        )
        if existing is not None:
            # Validate replacement before touching the old record.
            from solar_toolkit.map.jet_annotations import pixel_hpc

            x, y = np.rint(xy).astype(int)
            if not (
                0 <= x < doc.raw.shape[1] and 0 <= y < doc.raw.shape[0]
            ) or not np.isfinite(doc.raw[y, x]):
                raise ValueError("目标坐标在原图以外或缺测处")
            hpc = pixel_hpc(doc.map, [xy])[0].tolist()
            doc.checkpoint("move_native_tiepoint")
            existing.update(
                pixel_xy=list(map(float, xy)), hpc_arcsec=hpc, verified_3d=False
            )
        else:
            doc.add_tiepoint(
                xy,
                number,
                self.tie_status.currentData(),
                self.feature.text(),
                **self.point_metadata(),
            )
        if not was_pending:
            self.reset_point_editor_baseline()
        else:
            self.refresh_point_metadata_pending()
        self.epipolar_anchor = (index, doc.sha256, list(map(float, xy)))
        other = self.panes[1 - index].doc
        paired = other and any(t["number"] == number for t in other.state["tiepoints"])
        self.point_hint.setText(
            f"编号 {number} 两侧已标记；请核对身份，再选下一特征。"
            if paired
            else f"编号 {number}：请在{'EUVI 右侧' if index == 0 else 'AIA 左侧'}点击同一特征；黄虚线仅为极线提示。"
        )

    @undoable("应用对应点用途 / 顺序")
    def apply_point_metadata(self):
        n = getattr(self, "_point_editor_number", self.tie_id.value())
        for pane in self.panes:
            if pane.doc:
                for t in pane.doc.state["tiepoints"]:
                    if t["number"] == n:
                        pane.doc.checkpoint("tiepoint_metadata")
                        t.update(
                            self.point_metadata(),
                            feature=self.feature.text(),
                            identity_status=self.tie_status.currentData(),
                            verified_3d=False,
                        )
        self.last_saved = False
        self._point_editor_baseline = deepcopy(self.point_editor_values())
        self.refresh_point_metadata_pending()
        self.redraw()

    @undoable("下一特征")
    def next_feature(self):
        if not self.ensure_point_metadata_applied("标记下一特征"):
            return
        numbers = [
            t["number"] for p in self.panes if p.doc for t in p.doc.state["tiepoints"]
        ]
        self.tie_id.setValue(max(numbers, default=0) + 1)
        self.start_marking()

    @undoable("删除对应点")
    def delete_feature(self):
        for p in self.panes:
            if p.doc and any(
                t["number"] == self.tie_id.value() for t in p.doc.state["tiepoints"]
            ):
                p.doc.checkpoint("remove_correspondence")
                p.doc.state["tiepoints"] = [
                    t
                    for t in p.doc.state["tiepoints"]
                    if t["number"] != self.tie_id.value()
                ]
        self.epipolar_anchor = None
        self.last_saved = False
        self.redraw()

    def coordinate_dialog(self):
        from .coordinate_dialog import edit_native_coordinate

        return edit_native_coordinate(self)

    def native_step(self, delta):
        if not self.ensure_point_metadata_applied("切换观测帧"):
            return
        if self.common.records:
            self.common.step(delta)
        elif self.manifest:
            index = max(
                0,
                min(len(self.manifest[1]["pairs"]) - 1, self.loaded_pair_index + delta),
            )
            self.common.select_pair_async(index)

    @undoable("计算原图三维候选")
    def calculate_3d(self):
        if not self.ensure_point_metadata_applied("计算三维"):
            return None
        from .fit_workflow import compute_native_fit

        self.reconstruction_result = compute_native_fit(
            [p.doc for p in self.panes],
            self.common.pair_metadata(),
            self.fit_options,
            lambda: False,
        )
        self.reconstruction_signature = self.geometry_signature()
        self.last_precheck = deepcopy(self.reconstruction_result)
        from .workflow_guide import annotation_signature

        self.precheck_signature = annotation_signature(self)
        self.last_saved = False
        self.refresh_reconstruction()
        self.redraw()
        return self.reconstruction_result

    def calculate_and_show(self):
        try:
            self.start_joint_fit()
        except (ValueError, OSError, KeyError) as exc:
            self.point_hint.setText(f"计算未完成：{exc}")

    def show_auxiliary(self):
        self.parameter_toggle.setChecked(True)
        self.parameter_panel.setCurrentIndex(1)

    def show_reconstruction_results(self):
        if not self.reconstruction_result:
            self.point_hint.setText("请先点击“计算三维”。至少一对原图对应点即可。")
            return
        if self.reconstruction_dialog:
            self.reconstruction_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("三维候选坐标与原图残差")
        available = self.screen().availableGeometry()
        dialog.resize(
            min(1050, int(available.width() * 0.94)),
            min(760, int(available.height() * 0.90)),
        )
        layout = QVBoxLayout(dialog)
        details_tabs = QTabWidget()
        details_tabs.setMinimumHeight(160)
        details_tabs.setMaximumHeight(260)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        status = QLabel("\n".join(reconstruction_quality(self.reconstruction_result)))
        status.setWordWrap(True)
        details_layout.addWidget(status)
        dialog.result_status = status
        fit_note = QLabel(
            axis_description(self.reconstruction_result.get("summary", {}).get("axis"))
        )
        fit_note.setWordWrap(True)
        details_layout.addWidget(fit_note)
        details_layout.addStretch()
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setWidget(details)
        details_tabs.addTab(details_scroll, "检查条件与主轴")
        fit_details = QLabel(
            fit_description(self.reconstruction_result.get("joint_fits"))
        )
        fit_details.setWordWrap(True)
        fit_area = QScrollArea()
        fit_area.setWidgetResizable(True)
        fit_area.setWidget(fit_details)
        details_tabs.addTab(fit_area, "联合拟合")
        from .fit_results import fit_comparison_table, fit_residual_table

        details_tabs.addTab(
            fit_comparison_table(self.reconstruction_result), "模型对照"
        )
        details_tabs.addTab(
            fit_residual_table(self.reconstruction_result), "拟合回投残差"
        )
        rows = self.reconstruction_result["points"]
        from .workflow_guide import readable_reason

        table = QTableWidget(len(rows), 7)
        table.setHorizontalHeaderLabels(
            [
                "编号",
                "XYZ / R☉",
                "h / R☉",
                "射线距 Mm",
                "A / B 残差 ″",
                "交会角 °",
                "诊断",
            ]
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        def fmt(value):
            return "—" if value is None else f"{value:.5f}"

        for i, r in enumerate(rows):
            values = [
                r["number"],
                ", ".join(map(fmt, r.get("xyz_Rsun") or [])),
                fmt(r.get("height_Rsun")),
                fmt(r.get("gap_Mm")),
                " / ".join(map(fmt, r.get("residual_arcsec") or [])),
                fmt(r.get("ray_angle_deg")),
                "; ".join(
                    map(
                        readable_reason,
                        r.get("issues", []) or [r.get("reason", "candidate")],
                    )
                ),
            ]
            for j, v in enumerate(values):
                table.setItem(i, j, QTableWidgetItem(str(v)))
        table.resizeColumnsToContents()
        details_tabs.addTab(table, "逐点坐标与残差")
        layout.addWidget(details_tabs, 1)
        # mplot3d's projected axis labels are not fully included by constrained
        # layout. Reserve an explicit lower margin above the return button.
        figure = Figure(figsize=(6, 3))
        figure.subplots_adjust(left=0.03, right=0.96, bottom=0.16, top=0.89)
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot(111, projection="3d")
        draw_reconstruction(ax, self.reconstruction_result)
        frame = self.reconstruction_result.get("provenance", {})
        context = QLabel(
            f"参考系：{frame.get('reference_frame', '未记录')}；参考时刻 UTC：{frame.get('reference_time_utc', '未记录')}\n"
            "洋红虚线：PCA 主轴；青色实线：联合拟合；绿色箭头：局部径向。拟合不确认物质运动或对应身份。"
        )
        context.setWordWrap(True)
        layout.addWidget(context)
        layout.addWidget(canvas, 2)
        from .accessibility import screenshot_button

        layout.addWidget(screenshot_button(self, dialog))
        close = QPushButton("返回原图检查")
        close.clicked.connect(dialog.close)
        layout.addWidget(close)
        dialog.finished.connect(lambda: figure.clear())
        self.reconstruction_dialog = dialog
        dialog.show()
