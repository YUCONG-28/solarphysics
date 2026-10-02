"""Prepare immutable session inputs completely before changing the workspace."""

from pathlib import Path
import hashlib
import json

import numpy as np
from astropy.time import Time

from solar_toolkit.map.jet_annotations import load_session


def validate_event_profile(profile):
    """Validate only user-supplied browsing limits, never invent an event date."""
    if not isinstance(profile, dict):
        raise ValueError("事件配置无效")
    bands = profile.get("bands")
    if bands is not None and (
        not isinstance(bands, list)
        or not bands
        or any(type(b) is not int or b <= 0 for b in bands)
        or len(set(bands)) != len(bands)
    ):
        raise ValueError("事件波段必须是非重复的正整数列表")
    for key in ("start_utc", "end_utc"):
        if profile.get(key):
            Time(profile[key])
    if all(profile.get(k) for k in ("start_utc", "end_utc")) and Time(
        profile["start_utc"]
    ) >= Time(profile["end_utc"]):
        raise ValueError("事件起止时间顺序无效")
    return dict(profile)


def checked_hash(path, cancelled=lambda: False):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if cancelled():
                raise InterruptedError("读取已取消")
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame_metadata(record):
    if record.get("instrument") not in ("AIA", "EUVI"):
        raise ValueError("时序仪器标识无效")
    Time(record["midpoint_utc"])
    for key, label in (("band", "波段"), ("dsun_m", "观测者距离")):
        if not np.isfinite(float(record[key])) or float(record[key]) <= 0:
            raise ValueError("时序" + label + "无效")


def validate_fit_options(options):
    if not isinstance(options, dict) or (
        "include_curve" in options and type(options["include_curve"]) is not bool
    ):
        raise ValueError("拟合选项格式无效")


def resolve_frames(records, base, validate, cancelled=lambda: False):
    result = []
    for source in records:
        if cancelled():
            raise InterruptedError("读取已取消")
        record = dict(source)
        validate_frame_metadata(record)
        record["_path"] = str(validate(base / record["image"]))
        if checked_hash(record["_path"], cancelled) != record["image_sha256"]:
            raise ValueError("时序图像尚未完整同步")
        if record.get("difference"):
            record["_difference"] = str(validate(base / record["difference"]))
            if (
                checked_hash(record["_difference"], cancelled)
                != record["difference_sha256"]
            ):
                raise ValueError("差分尚未完整同步")
        result.append(record)
    return result


def prepare_timeline(path, validate, cancelled=lambda: False):
    path = Path(validate(path))
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
    if checks.get(path.name) != checked_hash(path, cancelled):
        raise ValueError("时序清单校验失败")
    return resolve_frames(data["frames"], path.parent, validate, cancelled)


def prepare_manifest(path, validate, cancelled=lambda: False):
    """Validate all manifest identities before offering a new sample list."""
    path = Path(validate(path))
    data = json.loads(path.read_text())
    if data.get("schema") != "solarphysics.jet_lab.samples" or data.get("version") != 1:
        raise ValueError("Unsupported sample manifest")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("样本清单没有配对")
    identifiers = set()
    for pair in pairs:
        if not isinstance(pair.get("id"), str) or pair["id"] in identifiers:
            raise ValueError("样本编号缺失或重复")
        identifiers.add(pair["id"])
        if not isinstance(pair.get("split"), str) or len(pair.get("views", [])) != 2:
            raise ValueError("样本必须包含两侧原图与用途")
        for view in pair["views"]:
            for key in ("image", "difference"):
                if view.get(key):
                    source = validate(path.parent / view[key])
                    if checked_hash(source, cancelled) != view.get(key + "_sha256"):
                        raise ValueError("样本尚未完整同步或原图身份已变化")
            if not view.get("image"):
                raise ValueError("样本缺少原图")
    return path.parent, data


def prepare_session(
    path, validate, cancelled=lambda: False, *, lazy_segmentation=False
):
    path = Path(validate(path))
    checks = json.loads((path.parent / "COMPLETE.json").read_text())["sha256"]
    if not {path.name, "view_state.json"} <= checks.keys():
        raise ValueError("会话完成清单缺少必要文件")
    for name, digest in checks.items():
        if checked_hash(validate(path.parent / name), cancelled) != digest:
            raise ValueError("会话仍在同步或校验失败")
    session = json.loads(path.read_text())
    view = json.loads((path.parent / "view_state.json").read_text())
    if session.get("schema") != "solarphysics.jet_lab.session" or session.get(
        "version"
    ) not in (1, 2):
        raise ValueError("Unsupported session")
    # Reject NaN/Inf before Qt fields or plotting can receive them.
    json.dumps([session, view], allow_nan=False)
    validate_fit_options(session.get("fit_options", {}))
    required = {
        "height_rsun",
        "display_mode",
        "layer",
        "master",
        "frame_index",
        "band",
        "background",
        "reference_image_sha256",
        "limits",
    }
    if not required <= view.keys():
        raise ValueError("会话显示设置缺少必要字段")
    for key in (
        "background",
        "original_visible",
        "parameters_visible",
        "guide_visible",
        "timeline_controls_visible",
    ):
        if key in view and type(view[key]) is not bool:
            raise ValueError("会话开关设置无效：" + key)
    if view["limits"] is not None:
        bounds = np.asarray(view["limits"], float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("视野范围无效")
    validate_event_profile(view.get("event_profile", {}))
    documents = {}
    for entry in session["documents"]:
        if cancelled():
            raise InterruptedError("读取已取消")
        docs, _ = load_session(
            validate(path.parent / entry["annotation"]),
            validate,
            lazy_segmentation=lazy_segmentation,
        )
        docs = [d for d in docs if d is not None]
        if (
            len(docs) != 1
            or docs[0].sha256 != entry["sha256"]
            or entry["sha256"] in documents
        ):
            raise ValueError("会话图像身份不一致或重复")
        docs[0]._undo = entry.get("undo_stack", [])
        documents[docs[0].sha256] = docs[0]
    active = session["active_images"]
    if (
        not isinstance(active, list)
        or len(active) != 2
        or any(k is not None and k not in documents for k in active)
    ):
        raise ValueError("会话缺少有效的当前图像")
    for key, choices in (
        ("display_mode", range(3)),
        ("layer", range(2)),
        ("master", range(2)),
    ):
        if type(view[key]) is not int or view[key] not in choices:
            raise ValueError("无效显示设置：" + key)
    if view["height_rsun"] not in (0, 0.05, 0.1, 0.2):
        raise ValueError("不支持的历史显示球壳")
    if type(view["frame_index"]) is not int or view["frame_index"] < 0:
        raise ValueError("时序索引无效")
    if int(view["band"]) <= 0:
        raise ValueError("波段无效")
    for key, length in (("native_panel_sizes", 3), ("vertical_panel_sizes", 2)):
        value = view.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != length
            or any(type(x) is not int or x < 0 for x in value)
        ):
            raise ValueError("窗口分隔设置无效")
    limits = view.get("display_limits", [[1, 99.5], [1, 99.5]])
    array = np.asarray(limits, float)
    if (
        array.shape != (2, 2)
        or not np.isfinite(array).all()
        or np.any(array[:, 0] >= array[:, 1])
    ):
        raise ValueError("亮度设置无效")
    if view.get("native_display_layer", 0) not in (0, 1):
        raise ValueError("原图图层无效")
    if view.get("window_view_mode", "native") not in ("native", "common", "split"):
        raise ValueError("工作区模式无效")
    if view.get("native_focus") not in (None, 0, 1):
        raise ValueError("当前视角无效")
    if "current_point_number" in view and (
        type(view["current_point_number"]) is not int
        or not 1 <= view["current_point_number"] <= 9999
    ):
        raise ValueError("当前对应点编号无效")
    if "active_editor_side" in view and (
        type(view["active_editor_side"]) is not int
        or view["active_editor_side"] not in (0, 1)
    ):
        raise ValueError("当前编辑侧别无效")
    reference = view["reference_image_sha256"]
    if reference is not None and reference not in documents:
        raise ValueError("历史参考图像缺失")
    drafts = {}
    for digest, draft in view.get("native_roi_drafts", {}).items():
        if (
            digest not in documents
            or list(documents[digest].raw.shape) != draft["shape"]
        ):
            raise ValueError("ROI草稿与原图身份/形状不一致")
        mask = np.zeros(draft["shape"], dtype=bool)
        for start, stop in draft["runs"]:
            if (
                type(start) is not int
                or type(stop) is not int
                or not 0 <= start < stop <= mask.size
            ):
                raise ValueError("ROI草稿区间无效")
            mask.ravel()[start:stop] = True
        drafts[digest] = (mask, draft["evidence"])
    frames = resolve_frames(session["frames"], path.parent, validate, cancelled)
    pair = view.get("current_pair")
    if pair is not None:
        for name, digest in zip(("AIA", "EUVI"), active, strict=True):
            record = pair.get(name)
            if (record.get("image_sha256") if record else None) != digest:
                raise ValueError("保存的配对与当前图像身份不同")
        if not isinstance(pair.get("status"), str):
            raise ValueError("保存的配对状态无效")
    report = session.get("reconstruction_result")
    if report is not None:
        if not isinstance(report, dict) or not isinstance(report.get("points"), list):
            raise ValueError("三维结果格式无效")
        for row in report["points"]:
            if not isinstance(row, dict) or type(row.get("number")) is not int:
                raise ValueError("三维点格式无效")
            xyz = row.get("xyz_Rsun")
            if xyz is not None and (
                np.asarray(xyz).shape != (3,) or not np.isfinite(xyz).all()
            ):
                raise ValueError("三维坐标无效")
        summary = report.get("summary", {})
        if not isinstance(summary, dict) or not isinstance(
            summary.get("axis", {}), dict
        ):
            raise ValueError("主轴结果格式无效")
        for key in ("joint_fits", "joint_robust"):
            comparison = report.get(key)
            if comparison is not None:
                if not isinstance(comparison, dict) or comparison.get(
                    "selected_model", "line"
                ) not in ("line", "curve"):
                    raise ValueError("拟合结果格式无效")
                for name in ("line", "curve"):
                    model = comparison.get(name)
                    if model is not None and (
                        not isinstance(model, dict)
                        or not isinstance(model.get("fitted_points", []), list)
                    ):
                        raise ValueError("拟合点格式无效")
    return session, view, documents, drafts, frames
