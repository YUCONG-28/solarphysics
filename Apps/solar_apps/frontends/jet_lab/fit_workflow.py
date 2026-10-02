"""Cancellable native-image fitting; workers never receive mutable UI state."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

from PyQt6.QtCore import QTimer, Qt, QSignalBlocker
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QProgressDialog,
)

from .undo_history import undoable


def compute_native_fit(documents, pairing, options, cancelled, *, robust=False):
    from solar_toolkit.map.jet_reconstruction import reconstruct_jet
    from solar_toolkit.map.jet_geometry_fit import fit_jet_geometry

    if cancelled():
        raise InterruptedError("Cancelled")
    result = reconstruct_jet(documents, pairing=pairing)
    result["joint_fits"] = fit_jet_geometry(
        documents,
        reconstruction=result,
        pairing=pairing,
        cancelled=cancelled,
        include_curve=options.get("include_curve", False),
    )
    if robust:
        result["joint_robust"] = fit_jet_geometry(
            documents,
            reconstruction=result,
            pairing=pairing,
            cancelled=cancelled,
            include_curve=options.get("include_curve", False),
            loss="soft_l1",
        )
    return result


def fit_description(comparison):
    if not comparison:
        return "尚无双原图联合拟合；原始空间点与 PCA 结果保留。"
    model = comparison.get("selected_model", "line")
    fit = comparison.get(model) or {}
    label = "联合直线" if model == "line" else "单弯曲线（人工选择）"
    if not fit.get("valid"):
        from .workflow_guide import readable_reason

        reasons = (
            fit.get("issues")
            or comparison.get("issues")
            or [fit.get("status", "unavailable")]
        )
        return label + "未满足条件：" + "；".join(map(readable_reason, reasons))
    residuals = fit.get("per_view_rmse_px") or []
    residual_text = " / ".join(f"{r:.3f}" for r in residuals)
    length_label = (
        "指定内外端间拟合长度"
        if fit.get("length_interpretation")
        == "fitted_span_between_explicit_inner_outer_features"
        else "拟合样本跨度（非完整喷流长度）"
    )
    text = f"{label}：两侧像素 RMSE {residual_text} px；{length_label} {fit['length_Mm']:.3f} Mm。"
    if fit.get("radial_angle_deg") is not None:
        text += f"\n径向夹角 {fit['radial_angle_deg']:.2f}°。"
    if not fit.get("directed", True):
        text += "\n无向拟合：缺少可靠内外端，不解释为物质运动方向。"
        if fit.get("unsigned_radial_angle_deg") is not None:
            text += f"径向锐夹角 {fit['unsigned_radial_angle_deg']:.2f}°。"
    validation = fit.get("validation") or {}
    if validation.get("pixel_rmse") is not None:
        text += f"\n留一内部点轨迹检验 RMSE {validation['pixel_rmse']:.3f} px。"
        text += f"有效验证折 {validation.get('valid_fold_count', 0)}/{validation.get('expected_fold_count', 0)}；不完整时不能视为完整稳定性检验。"
        text += "留出点只重求沿轨迹位置，检验轨迹相容性，不是盲预测三维坐标。"
    return text + "\n拟合不确认对应身份；残差与敏感性均不是置信区间。"


class FitWorkflow:
    def setup_fit_workflow(self, layout):
        self.fit_options = {"include_curve": False}
        self._fit_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jet-native-fit"
        )
        self._fit_future = None
        self._fit_generation = 0
        self._fit_cancel = Event()
        self._fit_closed = False
        self._fit_poll = QTimer(self)
        self._fit_poll.setInterval(30)
        self._fit_poll.timeout.connect(self.collect_fit)
        self.fit_progress = None
        self.fit_settings = QWidget()
        box = QVBoxLayout(self.fit_settings)
        box.setContentsMargins(0, 0, 0, 0)
        self.fit_curve = QCheckBox("比较单弯曲线（需 ≥6 对有序喷流点）")
        self.fit_curve.setAccessibleName("比较单弯曲线")
        self.fit_curve.toggled.connect(self.update_fit_options)
        box.addWidget(self.fit_curve)
        robust = QPushButton("计算稳健敏感性对照")
        robust.setAccessibleName(robust.text())
        robust.clicked.connect(lambda: self.start_joint_fit(robust=True))
        box.addWidget(robust)
        self.fit_model = QComboBox()
        self.fit_model.addItem("使用联合直线（默认）", "line")
        self.fit_model.addItem("使用单弯曲线（需计算后选择）", "curve")
        self.fit_model.setAccessibleName("采用的拟合模型")
        self.fit_model.activated.connect(self.select_fit_model)
        box.addWidget(self.fit_model)
        # Explicit actions are also usable when the native accessibility bridge
        # cannot open a Qt combo popup. Keep the combo for saved/API compatibility.
        choices = QHBoxLayout()
        for title, model in (("采用直线", "line"), ("采用曲线", "curve")):
            button = QPushButton(title)
            button.setAccessibleName(title)
            button.clicked.connect(
                lambda checked=False, value=model: self.choose_fit_model(value)
            )
            choices.addWidget(button)
        box.addLayout(choices)
        note = QLabel("稳健损失仅作敏感性检查；不会删除标点或自动认定曲线更真实。")
        note.setWordWrap(True)
        box.addWidget(note)
        advanced = QPushButton("拟合选项")
        advanced.setCheckable(True)
        advanced.toggled.connect(self.fit_settings.setVisible)
        layout.addWidget(advanced)
        layout.addWidget(self.fit_settings)
        self.fit_settings.hide()

    @undoable("修改拟合选项")
    def update_fit_options(self, *_):
        self.fit_options = {"include_curve": self.fit_curve.isChecked()}
        self.refresh_reconstruction()

    def sync_fit_controls(self):
        blockers = [QSignalBlocker(self.fit_curve), QSignalBlocker(self.fit_model)]
        self.fit_curve.setChecked(self.fit_options.get("include_curve", False))
        selected = ((self.reconstruction_result or {}).get("joint_fits") or {}).get(
            "selected_model", "line"
        )
        self.fit_model.setCurrentIndex(self.fit_model.findData(selected))
        del blockers

    @undoable("选择拟合模型")
    def select_fit_model(self, *_):
        if not self.reconstruction_current():
            self.point_hint.setText("请先重新计算当前标注的拟合。")
            self.sync_fit_controls()
            return
        result = self.reconstruction_result.get("joint_fits", {})
        model = self.fit_model.currentData()
        if not result.get(model, {}).get("valid"):
            self.point_hint.setText(
                "该模型尚不可用；请检查点数、顺序、端点及拟合诊断。"
            )
            self.sync_fit_controls()
            return
        result["selected_model"] = model
        result["selection_source"] = "user_explicit_choice"
        self.last_saved = False
        self.redraw()
        if self.reconstruction_dialog:
            self.show_reconstruction_results()

    def choose_fit_model(self, model):
        """Choose an already computed model explicitly; never recompute clicks."""
        self.fit_model.setCurrentIndex(self.fit_model.findData(model))
        self.select_fit_model()

    def start_joint_fit(self, *, robust=False):
        if not self.ensure_point_metadata_applied("重建及拟合"):
            return
        from solar_toolkit.map.jet_geometry_fit import freeze_jet_documents

        self.cancel_fit()
        self._fit_cancel = Event()
        token = self._fit_generation
        signature = self.geometry_signature()
        documents = freeze_jet_documents([p.doc for p in self.panes])
        pairing = deepcopy(self.common.pair_metadata())
        options = deepcopy(self.fit_options)
        cancelled = self._fit_cancel
        self._fit_job = (token, signature)
        self._fit_future = self._fit_executor.submit(
            compute_native_fit,
            documents,
            pairing,
            options,
            cancelled.is_set,
            robust=robust,
        )
        self.fit_progress = QProgressDialog(
            "计算原生双视角及联合拟合…", "取消计算", 0, 0, self
        )
        self.fit_progress.setWindowTitle("原图联合拟合")
        self.fit_progress.setWindowModality(Qt.WindowModality.NonModal)
        self.fit_progress.setMinimumDuration(0)
        self.fit_progress.canceled.connect(self.cancel_fit)
        self.fit_progress.show()
        self._fit_poll.start()

    def cancel_fit(self):
        self._fit_generation += 1
        self._fit_cancel.set()
        self._fit_poll.stop()
        if self._fit_future:
            self._fit_future.cancel()
        self._fit_future = None
        self._close_fit_progress()

    def _close_fit_progress(self):
        dialog, self.fit_progress = self.fit_progress, None
        if dialog is not None:
            # Programmatic completion must not emit the user's cancel action.
            dialog.blockSignals(True)
            dialog.close()
            dialog.deleteLater()

    def collect_fit(self):
        future = self._fit_future
        if future is None or not future.done():
            return
        token, signature = self._fit_job
        self._fit_future = None
        self._fit_poll.stop()
        self._close_fit_progress()
        try:
            result = future.result()
        except InterruptedError:
            return
        except Exception as exc:
            self.point_hint.setText(f"计算失败，已有标注和结果保留：{exc}")
            return
        if (
            self._fit_closed
            or token != self._fit_generation
            or signature != self.geometry_signature()
        ):
            self.point_hint.setText("计算期间输入已改变，已丢弃过期结果。请重新计算。")
            return
        self.accept_fit(result, signature)
        self.show_reconstruction_results()

    @undoable("重建及双原图联合拟合")
    def accept_fit(self, result, signature):
        self.reconstruction_result = result
        self.reconstruction_signature = signature
        self.last_precheck = deepcopy(result)
        from .workflow_guide import annotation_signature

        self.precheck_signature = annotation_signature(self)
        self.last_saved = False
        self.sync_fit_controls()
        self.redraw()

    def close_fit_workflow(self):
        self._fit_closed = True
        self.cancel_fit()
        self._fit_executor.shutdown(wait=False, cancel_futures=True)
