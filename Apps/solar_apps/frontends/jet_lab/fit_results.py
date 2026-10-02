"""Read-only fit tables with explicit model and loss provenance."""

from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem


def _table(headers, rows):
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.setItem(i, j, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()
    return table


def _fmt(value):
    return "—" if value is None else f"{value:.4f}"


def fit_comparison_table(report):
    from .workflow_guide import readable_reason

    rows = []
    for key, loss in (("joint_fits", "普通最小二乘"), ("joint_robust", "稳健敏感性")):
        group = report.get(key) or {}
        for name, title in (("line", "直线"), ("curve", "单弯曲线")):
            fit = group.get(name) or {}
            if not fit:
                continue
            validation = fit.get("validation") or {}
            rows.append(
                [
                    loss,
                    title,
                    "候选可用" if fit.get("valid") else "未通过条件",
                    _fmt(fit.get("pixel_rmse")),
                    _fmt(validation.get("pixel_rmse")),
                    _fmt(fit.get("length_Mm")),
                    _fmt(validation.get("max_endpoint_direction_change_deg")),
                    "; ".join(
                        map(
                            readable_reason,
                            fit.get("issues", []) + group.get("issues", []),
                        )
                    ),
                ]
            )
    return _table(
        [
            "目标",
            "模型",
            "状态",
            "原图 RMSE px",
            "留点轨迹 RMSE px",
            "长度/跨度 Mm",
            "方向敏感性 °",
            "诊断（不等于置信度）",
        ],
        rows,
    )


def fit_residual_table(report):
    rows = []
    for key, loss in (("joint_fits", "普通"), ("joint_robust", "稳健")):
        group = report.get(key) or {}
        for model in ("line", "curve"):
            for point in (group.get(model) or {}).get("fitted_points", []):
                for side, label in enumerate(("AIA", "EUVI")):
                    observed = point["observed_pixel_xy"][side]
                    predicted = point["predicted_pixel_xy"][side]
                    rows.append(
                        [
                            loss,
                            model,
                            point["number"],
                            label,
                            ", ".join(map(_fmt, observed)),
                            ", ".join(map(_fmt, predicted)),
                            _fmt(point["residual_px"][side]),
                            "; ".join(point.get("issues", [])),
                        ]
                    )
    return _table(
        ["目标", "模型", "编号", "原图", "点击 x,y", "拟合回投 x,y", "偏离 px", "诊断"],
        rows,
    )
