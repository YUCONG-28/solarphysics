"""Read-only presentation of native-ray candidates and descriptive PCA results."""

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from .workflow_guide import readable_reason


def reconstruction_quality(report):
    """Five separate checks; no presentation action upgrades geometry validity."""
    rows = report.get("points", [])
    count = len(rows)
    numerical = sum(bool(p.get("numerical_valid")) for p in rows)
    physical = sum(
        bool(p.get("numerical_valid"))
        and p.get("below_photosphere") is False
        and p.get("visible") == [True, True]
        for p in rows
    )
    identity = sum(
        p.get("identity_status") == ["manual_confirmed", "manual_confirmed"]
        for p in rows
    )
    sigma = 0
    for p in rows:
        values = [
            t.get("sigma_arcsec") if t else None
            for t in p.get("input_correspondences", [])
        ]
        try:
            a = np.asarray(values, float)
            sigma += bool(a.shape == (2,) and np.isfinite(a).all() and (a > 0).all())
        except (TypeError, ValueError):
            pass
    codes = list(report.get("issues", [])) + [
        c for p in rows for c in p.get("issues", [])
    ]
    time_codes = list(
        dict.fromkeys(
            c for c in codes if "time" in c or c == "pairing_image_identity_mismatch"
        )
    )
    pairing = report.get("provenance", {}).get("pairing") or {}
    time_text = (
        "；".join(map(readable_reason, time_codes))
        if time_codes
        else (
            "已通过记录中的采样配对；曝光间演化仍需检查"
            if pairing.get("status") == "matched"
            else "缺少可验证的采样配对证据"
        )
    )
    return [
        f"数值求解：{numerical}/{count} 个点可解；不可解点保留诊断。",
        f"物理条件：{physical}/{count} 个点位于光球外且两侧可见；不确认结构身份。",
        f"身份判断：{identity}/{count} 个点两侧均已人工确认；请检查未确认特征。",
        "时间配对：" + time_text + "。",
        f"定位误差：{sigma}/{count} 个点两侧已提供有限正误差；时间演化和完整不确定度未评估。",
    ]


def axis_description(axis):
    if not axis:
        return "此历史结果不含主轴拟合；重新计算可生成。"
    if not axis.get("valid"):
        return "主轴未确定：" + "；".join(map(readable_reason, axis.get("issues", [])))
    n = len(axis.get("used_point_numbers", []))
    rms = axis.get("rms_perpendicular_Mm")
    lines = [f"主轴：{n} 个喷流候选点等权拟合。"]
    if axis.get("directed") and axis.get("radial_angle_deg") is not None:
        lines.append(f"内→外方向与质心处径向夹角 {axis['radial_angle_deg']:.2f}°。")
    else:
        angle = axis.get("unsigned_radial_angle_deg")
        lines.append(
            "方向未定；"
            + (
                f"无向轴与径向锐夹角 {angle:.2f}°。"
                if angle is not None
                else "径向夹角不可用。"
            )
        )
    if rms is not None:
        lines.append(f"离轴 RMS {rms:.3f} Mm，仅为空间散布，不是定位误差或置信区间。")
    if n == 3:
        lines.append("仅三个点：主轴描述当前候选，尚无重复测量稳定性证据。")
    return "\n".join(lines)


def draw_reconstruction(ax, report):
    """Plot actual candidates and supported segments; never bridge a missing node."""
    rows = report.get("points", [])
    positions = []
    for row in rows:
        value = row.get("xyz_Rsun")
        if (
            value is not None
            and np.asarray(value).shape == (3,)
            and np.isfinite(value).all()
        ):
            p = np.asarray(value, float)
            positions.append(p)
            ax.scatter(
                *p,
                marker="+" if row.get("role") == "jet" else "x",
                color="tab:blue" if row.get("admissible_candidate") else "tab:red",
            )
            ax.text(*p, str(row["number"]), fontsize=8)
    summary = report.get("summary", {})
    segments = summary.get("polyline_segments")
    if segments is None:
        # Old reports cannot prove that omitted nodes were not bridged.
        segments = []
    for segment in segments:
        points = segment.get("xyz_Rsun", [])
        if len(points) >= 2:
            ax.plot(*np.asarray(points).T, color="orange", lw=1.3)
    fit = summary.get("axis", {})
    if fit.get("valid"):
        center = np.asarray(fit["centroid_xyz_Rsun"], float)
        direction = np.asarray(fit["direction_xyz"], float)
        selected = [
            np.asarray(r["xyz_Rsun"])
            for r in rows
            if r["number"] in fit["used_point_numbers"]
        ]
        span = max(float(np.ptp((np.asarray(selected) - center) @ direction)), 0.01)
        ends = center + np.array([-0.55, 0.55])[:, None] * span * direction
        ax.plot(*ends.T, color="magenta", lw=1.5, ls="--", label="PCA axis")
        if fit.get("directed"):
            ax.quiver(
                *center,
                *direction,
                length=span * 0.5,
                color="magenta",
                arrow_length_ratio=0.15,
            )
        radial = fit.get("radial_unit_xyz")
        if radial is not None:
            ax.quiver(
                *center,
                *radial,
                length=span * 0.4,
                color="green",
                arrow_length_ratio=0.18,
                label="Local radial",
            )
        positions.extend(ends)
        if radial is not None:
            positions.append(center + span * 0.4 * np.asarray(radial))
        ax.legend(loc="upper left", fontsize=8)
    joint = report.get("joint_fits") or {}
    chosen = joint.get(joint.get("selected_model", "line")) or {}
    samples = np.asarray(chosen.get("sampled_curve_xyz_Rsun") or [], float)
    if samples.ndim == 2 and samples.shape[1] == 3 and np.isfinite(samples).all():
        ax.plot(
            *samples.T,
            color="darkcyan",
            lw=2,
            ls="-" if chosen.get("valid") else ":",
            label="Joint fit" if chosen.get("valid") else "Joint fit diagnostic",
        )
        positions.extend(samples)
        ax.legend(loc="upper left", fontsize=8)
    for name in "xyz":
        getattr(ax, f"set_{name}label")(
            name.upper() + " / Rsun", fontsize=8, labelpad=4
        )
        axis = getattr(ax, f"{name}axis")
        axis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        axis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(labelsize=7, pad=0)
    if positions:
        # Quiver does not reliably autoscale to its arrow tip in mplot3d.
        # Include every displayed position and use equal physical scales so
        # changing a narrow coordinate range cannot visually change the angle.
        bounds = np.asarray(positions, float)
        midpoint = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        half_width = max(float(np.ptp(bounds, axis=0).max()) / 2, 0.005) * 1.2
        for name, center in zip("xyz", midpoint, strict=True):
            getattr(ax, f"set_{name}lim")(center - half_width, center + half_width)
    ax.set_box_aspect((1, 1, 1))
    frame = report.get("provenance", {}).get(
        "reference_frame", "Reference frame unavailable"
    )
    ax.set_title("Native-ray candidates · " + frame, fontsize=10)
