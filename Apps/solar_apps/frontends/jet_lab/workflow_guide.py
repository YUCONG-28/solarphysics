"""Read-only workflow navigation and operation status."""

import json
import hashlib
import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QDockWidget,
)

STEPS = [
    "加载事件",
    "检查数据",
    "共同视角找结构",
    "原图确认区域",
    "轴线与对应点",
    "几何预检",
    "保存与导出",
]
TIPS = [
    "加载已有样本、时序或两幅 FITS。",
    "查看单位、曝光、观测者与未校准项目。",
    "显示重合不等于同一结构；圈选候选 ROI 后回原图确认。",
    "两侧分别选区域、目标连通块和内端；断裂不强行连接。",
    "确认支段和方向，再在两侧分别标记同一特征。",
    "检查编号、顺序和残差；缺误差不生成 3σ 通过状态。",
    "保存新修订；无效几何导出为诊断材料，不能作为有效高度输入。",
]
NATIVE_STEPS = ["加载原图", "标记同一特征", "计算三维", "检查残差", "保存新修订"]
NATIVE_TIPS = [
    "加载 AIA、EUVI 原图配对；核对波段、实际 UTC、观测者和时间配对状态。",
    "先在 AIA 点选亮结/弯折，再到 EUVI 独立辨认同一特征；编号相同，顺序从内向外。",
    "一组点可求空间候选；两组求线段；三组以上形成有序折线。分割和轴线不是前提。",
    "检查两条射线间距及各自原图残差；缺定位与演化误差时不显示 3σ 通过。",
    "保存原生坐标、候选三维坐标、结果状态和输入校验和；旧修订始终保留。",
]


def readable_reason(reason):
    reason = str(reason or "unknown_reason")
    label = {
        "unit_or_normalization_history_ambiguous": "单位未声明或归一化历史冲突",
        "invalid_exposure_for_normalization": "曝光时间无效，保留原值",
        "telemetry_missing_blocks_not_decoded": "存在未解析的遥测缺块",
        "invalid_missing_block_count": "缺测计数字段无效",
        "no_valid_pixels": "没有有效像素",
        "invalid_helioprojective_wcs": "原生 WCS 无效，不能作坐标测量",
        "missing_or_invalid_hgln_obs": "缺少可靠观测者经度",
        "missing_or_invalid_hglt_obs": "缺少可靠观测者纬度",
        "missing_or_invalid_dsun_obs": "缺少可靠日地距离",
        "missing_view": "尚未加载两个视角",
        "same_native_image_in_both_views": "两侧是同一幅原图，不能三角测量",
        "duplicate_numbers": "对应点编号重复",
        "axis_pending": "轴线待确认",
        "axis_missing": "缺少轴线",
        "axis_order_disagreement": "两侧对应点沿轴顺序不一致",
        "weak_parallax": "视差角过小，深度约束不足",
        "behind_observer": "交会位于观测者后方",
        "localization_sigma_and_evolution_not_available": "缺定位误差和时间演化约束",
        "missing_counterpart": "另一侧缺少同编号特征",
        "ok": "射线数值求解完成，仍为候选",
        "time_pairing_unverified": "尚未验证实际帧的时间配对",
        "time_pairing_no_other_frame": "缺少另一视角的实际配对帧",
        "time_pairing_outside_tolerance": "最近帧超过时间容差，未配对",
        "time_pairing_ambiguous_nearest": "多个最近帧同样接近，时间配对不唯一",
        "time_pairing_cadence_unavailable": "采样间隔不足，未自动确认时间配对",
        "time_pairing_incompatible": "该对应点所用两帧的时间配对不相容",
        "time_tolerance_unavailable": "缺少可靠的时间配对容差",
        "center_time_outside_tolerance": "日心光行时修正后的时间差超限",
        "source_time_outside_tolerance": "按候选位置修正光行时后，时间差仍超限",
        "pairing_image_identity_mismatch": "配对记录的图像身份与当前原图不一致",
        "different_wavelengths": "两幅原图波段不同，不能视为同波段对应",
        "wavelength_unavailable": "缺少可靠波段信息",
        "invalid_point_number": "对应点编号不是有效整数",
        "duplicate_number": "同一原图内出现重复编号",
        "role_conflict": "两侧对该编号的喷流点/参照点用途不一致",
        "order_conflict": "两侧沿喷流顺序不一致或顺序无效",
        "endpoint_conflict": "两侧内端/外端标记不一致",
        "duplicate_native_position": "多个编号标在同一原生像素位置",
        "native_pixel_missing": "标记位置落在原图缺测区域",
        "invalid_native_coordinates": "原生坐标无效、越界或无法转换",
        "below_photosphere": "候选点位于光球以下，不能作为可见日冕喷流点",
        "solar_occultation": "候选点至少在一个视角被太阳遮挡",
        "identity_and_evolution_unverified": "结构身份和曝光之间的演化尚未验证",
        "invalid_localization_sigma": "定位误差无效，必须为有限正值",
        "reprojection_exceeds_3sigma": "原图重投影残差超过提供的定位误差 3σ",
        "no_correspondences": "尚无可计算的同编号对应点",
        "point_diagnostics_require_review": "部分对应点存在需核查的问题，详见逐点诊断",
        "jet_order_missing": "喷流点缺少沿轴顺序，不能累计有序长度",
        "jet_points_unusable": "部分指定喷流点不可用，轴段不完整",
        "duplicate_jet_order": "多个喷流点使用同一沿轴顺序",
        "unknown_roles_excluded": "未指定用途的点已排除于喷流长度计算",
        "explicit_unique_endpoints_required": "请明确指定唯一内端和唯一外端，才能计算端点弦长与方向",
        "endpoint_unusable": "指定内端或外端不可用，请核查该点诊断",
        "endpoint_order_missing": "内端或外端缺少沿轴顺序",
        "endpoint_order_conflict": "内端顺序必须小于外端顺序",
        "coincident_endpoints": "内外端三维位置重合，方向无法确定",
        "points_outside_endpoint_order": "存在顺序位于指定内外端范围之外的喷流点",
        "no_explicit_jet_points": "尚未明确指定喷流点；参照点不参与长度计算",
        "pair_or_input_conditions_unresolved": "配对或输入条件尚有未解决项，长度仅为条件候选",
        "candidate_polyline": "由有序候选点连接形成的三维折线",
        "insufficient_ordered_points": "可用有序喷流点不足，不能形成三维折线",
        "at_least_three_distinct_admissible_jet_points_required": "至少需要三个不同且可接受的喷流候选点才能拟合主轴",
        "principal_eigenline_not_unique": "点集没有唯一主方向，不能给出喷流主轴",
        "explicit_unique_endpoints_required_for_direction": "缺少唯一有效内外端，仅显示无向主轴",
        "endpoint_excluded_from_fit": "指定端点未进入有效点集，主轴方向未定",
        "endpoints_do_not_orient_principal_axis": "指定端点重合或不能确定主轴正负方向",
        "centroid_radial_direction_undefined": "质心处径向无法确定",
        "not_explicit_jet_point": "非明确喷流点，不参与主轴拟合",
        "inadmissible_candidate": "候选点未满足参与拟合的条件",
        "duplicate_point_number": "点编号重复，不参与拟合",
        "invalid_position": "三维坐标无效",
        "duplicate_position": "三维位置重复，不增加拟合权重",
        "curve_not_requested": "未启用曲线对照，默认使用直线",
        "at_least_six_ordered_features_required": "曲线需至少六对完整、有序喷流点及唯一内外端",
        "rank_deficient_fit": "拟合参数不可唯一约束；保留诊断，不认定可用模型",
        "optimizer_not_converged": "拟合未收敛",
        "fitted_nodes_not_resolvably_ordered": "拟合点不能保持可分辨顺序",
        "fitted_physical_conditions_failed": "拟合位置存在遮挡或光球以下等物理问题",
        "reconstruction_annotations_stale": "原重建与当前标注不同，请重新计算",
        "unknown_reason": "尚未给出诊断原因",
    }
    match = re.fullmatch(r"view_([01])[:_](.+)", reason)
    if match:
        return ("左侧：" if match[1] == "0" else "右侧：") + readable_reason(match[2])
    return label.get(reason, "需要核查：" + reason)


def annotation_signature(owner):
    return hashlib.sha256(
        json.dumps(
            [(p.doc.sha256, p.doc.state) if p.doc else None for p in owner.panes],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def workflow_status(owner):
    if getattr(owner, "native_only", False):
        return native_workflow_status(owner)
    docs = [p.doc for p in owner.panes]
    loaded = all(d is not None for d in docs)
    ready = loaded and all(
        not d.info.get("preprocessing", {}).get("reasons", [])
        and not d.info.get("geometry_issues", [])
        for d in docs
    )
    roi = loaded and all(d.state.get("roi") or d.state.get("roi_runs") for d in docs)
    axes = loaded and all(
        d.state["axis_status"] != "pending" and len(d.state["axis"]) >= 2 for d in docs
    )
    ties = loaded and all(d.state["tiepoints"] for d in docs)
    # Inspection of help never changes these measurement-derived statuses.
    signature = annotation_signature(owner)
    checked = (
        getattr(owner, "last_precheck", None) is not None
        and getattr(owner, "precheck_signature", None) == signature
    )
    saved = getattr(owner, "saved_signature", None) == signature
    done = [
        loaded,
        ready,
        bool(owner.common.native_drafts) or roi,
        roi,
        axes and ties,
        checked,
        loaded and not owner.common.has_unsaved() and saved,
    ]
    details = list(TIPS)
    if loaded:
        reasons = sorted(
            {
                r
                for d in docs
                for r in (
                    d.info.get("preprocessing", {}).get("reasons", [])
                    + d.info.get("geometry_issues", [])
                )
            }
        )
        details[1] = (
            "可浏览；定量条件未满足：" + "；".join(map(readable_reason, reasons))
            if reasons
            else "已核对有限预处理条件；完整仪器校准未执行。"
        )
    if checked:
        details[5] = "预检已执行；" + (
            "几何可用"
            if owner.last_precheck.get("geometry_valid")
            else "几何仍未验证，请查看缺失条件。"
        )
    return done, details


def native_workflow_status(owner):
    """Native workflow status depends on actual data, never visits to help pages."""
    docs = [p.doc for p in owner.panes]
    loaded = all(d is not None for d in docs)
    numbers = [
        [t["number"] for t in d.state.get("tiepoints", [])] if d else [] for d in docs
    ]
    common = set(numbers[0]) & set(numbers[1])
    duplicates = any(len(n) != len(set(n)) for n in numbers)
    missing = set(numbers[0]) ^ set(numbers[1])
    paired = loaded and bool(common) and not duplicates and not missing
    current = bool(owner.reconstruction_current())
    report = owner.reconstruction_result or {}
    saved = getattr(owner, "saved_signature", None) == annotation_signature(owner)
    done = [
        loaded,
        paired,
        current,
        current,
        loaded
        and saved
        and bool(getattr(owner, "last_saved", False))
        and not owner.common.has_unsaved(),
    ]
    details = list(NATIVE_TIPS)
    if loaded:
        issues = sorted(
            {reason for d in docs for reason in d.info.get("geometry_issues", [])}
        )
        details[0] = "两幅原图已加载，保持不同视角。" + (
            "几何条件：" + "；".join(map(readable_reason, issues))
            if issues
            else "请核对顶部实际时间与配对状态。"
        )
    details[1] = f"已有 {len(common)} 组同编号候选。" + (
        "编号重复，请先处理。"
        if duplicates
        else (
            "另一侧缺少编号：" + "、".join(map(str, sorted(missing)))
            if missing
            else "分别确认结构身份；参照点不参与喷流长度。"
        )
    )
    if current:
        count = sum(
            bool(row.get("numerical_valid")) for row in report.get("points", [])
        )
        details[2] = f"已计算 {count} 个数值可用候选；这不等于物理对应已确认。"
        details[3] = (
            "残差与失败原因已显示。请检查错配、顺序和时间演化；候选坐标不自动通过科学验证。"
        )
    elif report:
        details[2] = "已有结果过期：原图或对应点已改变，请重新计算。"
    return done, details


def precheck_summary(report):
    lines = [
        "几何预检："
        + ("可用" if report.get("geometry_valid") else "尚未验证；不据此生成唯一高度")
    ]
    for reason in report.get("issues", []):
        lines.append("• " + readable_reason(reason))
    for row in report.get("points", []):
        residual = row.get("residual_arcsec")
        text = " / ".join(f"{x:.2f}″" for x in residual) if residual else "未计算"
        reason = row.get("reason", "未验证")
        reason = readable_reason(reason)
        lines.append(f"编号 {row['number']}：重投影残差 {text}；{reason}")
    return "\n".join(lines)


class WorkflowGuide(QDockWidget):
    def __init__(self, owner):
        super().__init__("操作流程", owner)
        self.owner = owner
        self.dialog = None
        self.setObjectName("jet_workflow_guide")
        body = QWidget()
        box = QVBoxLayout(body)
        self.steps = QListWidget()
        self.steps.setMaximumHeight(205)
        box.addWidget(self.steps)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        box.addWidget(self.explanation)
        row = QHBoxLayout()
        box.addLayout(row)
        go = QPushButton("前往操作")
        go.clicked.connect(self.go)
        row.addWidget(go)
        box.addStretch()
        self.setWidget(body)
        self.steps.currentRowChanged.connect(self.explain)
        self.setMinimumWidth(215)
        self.setMaximumWidth(285)
        owner.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self)
        self.save_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Save), owner, activated=self.save
        )
        self.refresh()

    def save(self):
        if getattr(self.owner, "native_only", False):
            self.owner.save_dialog()
            return
        if self.owner.common.records:
            self.owner.common.save_dialog()
        else:
            self.owner.save_dialog()

    def refresh(self):
        index = max(self.steps.currentRow(), 0)
        done, details = workflow_status(self.owner)
        self.details = details
        self.steps.blockSignals(True)
        self.steps.clear()
        labels = NATIVE_STEPS if getattr(self.owner, "native_only", False) else STEPS
        index = min(index, len(labels) - 1)
        for i, label in enumerate(labels):
            self.steps.addItem(f"{'✓' if done[i] else '○'} {i+1}. {label}")
        self.steps.setCurrentRow(index)
        self.steps.blockSignals(False)
        self.explain(index)

    def explain(self, index):
        self.explanation.setText(self.details[max(0, index)])

    def go(self):
        index = self.steps.currentRow()
        w = self.owner
        if getattr(w, "native_only", False):
            if index == 0:
                w.open_event_dialog()
            elif index == 1:
                w.start_marking()
            elif index == 2:
                w.calculate_3d()
            elif index == 3:
                w.refresh_reconstruction()
                if hasattr(w, "show_reconstruction_results"):
                    w.show_reconstruction_results()
                else:
                    w.parameter_toggle.setChecked(True)
                    w.sections_box.setCurrentIndex(3)
            else:
                self.save()
            return
        if index == 0:
            w.manifest_dialog()
        elif index == 1:
            w.show_processing()
        elif index == 2:
            w.set_view_mode("common")
        elif index in (3, 4):
            w.set_view_mode("native")
            w.mode.setCurrentIndex(
                w.mode.findData("rectangle" if index == 3 else "tie")
            )
        elif index == 5:
            w.common.precheck()
        else:
            self.save()
