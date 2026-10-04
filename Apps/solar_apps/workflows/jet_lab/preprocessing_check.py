"""Audit configured native-image preparation and WCS preservation."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
import pandas as pd
import sunpy.map

from .pilot import prepare, dump
from solar_toolkit.map.euvi_preprocessing import (
    prepare_euvi_map,
    intensity_array,
    quantitative_issues,
)
from solar_toolkit.map.jet_annotations import file_sha256, pixel_hpc
from solar_toolkit.map.jet_viewpoint import reproject_photosphere


def finalize_manifest(out):
    """Publish consistent completion records after all report mutations."""
    out = Path(out)
    markers = {out / "COMPLETE.json", out / "PREPARATION_COMPLETE.json"}
    checks = {
        p.relative_to(out).as_posix(): file_sha256(p)
        for p in out.rglob("*")
        if p.is_file() and p not in markers
    }
    pairs = json.loads((out / "paired_samples.json").read_text())["pairs"]
    dump(
        out / "PREPARATION_COMPLETE.json",
        {"version": 1, "pair_count": len(pairs), "sha256": checks},
    )
    checks["PREPARATION_COMPLETE.json"] = file_sha256(out / "PREPARATION_COMPLETE.json")
    dump(out / "COMPLETE.json", {"version": 1, "sha256": checks})


def run(config):
    # Fail any unexpected process invocation in this workflow, including IDL.
    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "External process invocation forbidden in Python preparation test"
        )

    original_popen = subprocess.Popen
    subprocess.Popen = forbidden
    try:
        return _run(config)
    finally:
        subprocess.Popen = original_popen


def _run(config):
    prepare(config)
    out = Path(config["output_directory"])
    # The pilot completion marker must not advertise this extended run as complete.
    (out / "COMPLETE.json").unlink(missing_ok=True)
    (out / "PREPARATION_COMPLETE.json").unlink(missing_ok=True)
    bundle = json.loads((out / "paired_samples.json").read_text())
    inventory = pd.read_csv(config["inventory"])
    records, panels, inputs = (
        [],
        [],
        {config["inventory"]: file_sha256(Path(config["inventory"]))},
    )
    panel_titles = {}
    for pair in bundle["pairs"]:
        maps = [sunpy.map.Map(out / v["image"]) for v in pair["views"]]
        aia, euvi = maps
        source = Path(pair["views"][1]["parent"])
        raw = sunpy.map.Map(source)
        prepared = prepare_euvi_map(raw)
        x, y = (int(round(float(euvi.meta[k]))) for k in ("jxoff", "jyoff"))
        h, w = euvi.data.shape
        raw_crop = raw.data[y : y + h, x : x + w]
        expected = prepared.data[y : y + h, x : x + w]
        np.testing.assert_allclose(euvi.data, expected, rtol=0, atol=0, equal_nan=True)
        px = np.array([[0, 0], [w // 2, h // 2], [w - 1, h - 1]], float)
        delta = pixel_hpc(euvi, px) - pixel_hpc(raw, px + [x, y])
        scale = min(
            abs(raw.scale.axis1.to_value(u.arcsec / u.pix)),
            abs(raw.scale.axis2.to_value(u.arcsec / u.pix)),
        )
        error = float(np.max(np.linalg.norm(delta, axis=1)) / scale)
        if error >= 0.1:
            raise ValueError("Native crop WCS changed")
        band = int(euvi.meta["wavelnth"])
        aia_rows = inventory[(inventory.instrument == "AIA") & (inventory.band == band)]
        tolerance = float(np.median(np.diff(np.sort(aia_rows.solar_unix.unique()))) / 2)
        dt = float(pair["emission_time_difference_s"])
        pair["pairing_status"] = (
            "matched" if abs(dt) <= tolerance else "outside_tolerance"
        )
        pair["tolerance_s"] = tolerance
        projected = (
            reproject_photosphere(euvi, aia, euvi.data)
            if pair["pairing_status"] == "matched"
            else np.full(aia.data.shape, np.nan)
        )
        differences = []
        for view, m in zip(pair["views"], maps):
            differences.append(
                sunpy.map.Map(out / view["difference"]).data
                if view["difference"]
                else np.full(m.data.shape, np.nan)
            )
            parent = Path(view["parent"])
            inputs[str(parent)] = file_sha256(parent)
        finite = np.isfinite(raw_crop) & np.isfinite(euvi.data) & (raw_crop != 0)
        ratio = (
            float(np.median(euvi.data[finite] / raw_crop[finite]))
            if finite.any()
            else None
        )
        record = dict(
            pair=pair["id"],
            band=band,
            aia_midpoint=pair["views"][0]["info"]["midpoint_utc"],
            euvi_midpoint=pair["views"][1]["info"]["midpoint_utc"],
            pairing_status=pair["pairing_status"],
            delta_emission_s=dt,
            tolerance_s=tolerance,
            native_wcs_error_pixel=error,
            output_to_raw_ratio=ratio,
            projected_valid_fraction=float(np.isfinite(projected).mean()),
            source_sha256=file_sha256(source),
            **prepared.audit,
        )
        records.append(record)
        _, aia_unit, aia_action, _ = intensity_array(aia)
        record["aia_preprocessing"] = dict(
            unit=aia_unit, exposure_action=aia_action, reasons=quantitative_issues(aia)
        )
        au = aia_unit or "unit unspecified; raw values"
        eu = prepared.unit or "unit unspecified"
        panel_titles[pair["id"]] = [
            f"EUVI original ({prepared.audit['input_unit']})",
            f"EUVI Python preparation ({eu})",
            f"AIA ({au})",
            "EUVI in AIA view: photosphere assumption",
            f"EUVI consecutive difference ({eu})",
            f"AIA difference ({au})",
        ]
        panels.append(
            (
                pair["id"],
                band,
                [
                    raw_crop,
                    euvi.data,
                    aia.data,
                    projected,
                    differences[1],
                    differences[0],
                ],
            )
        )
    # Fixed scales across each band and panel type.
    norms = {}
    for band in sorted({panel[1] for panel in panels}):
        for column in range(6):
            arrays = [p[2][column] for p in panels if p[1] == band]
            values = (
                np.concatenate([a[np.isfinite(a)] for a in arrays])
                if arrays
                else np.array([])
            )
            if not len(values):
                norms[band, column] = None
            elif column >= 4:
                limit = max(float(np.percentile(abs(values), 99)), 1e-9)
                norms[band, column] = plt.Normalize(-limit, limit)
            else:
                lo, hi = np.percentile(values, [1, 99.5])
                hi = max(hi, lo + 1e-9)
                norms[band, column] = AsinhNorm(
                    linear_width=max((hi - lo) / 30, 1e-9), vmin=lo, vmax=hi
                )
    for name, band, arrays in panels:
        titles = panel_titles[name]
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
        for column, (ax, data, title) in enumerate(zip(axes.flat, arrays, titles)):
            im = ax.imshow(
                data,
                origin="lower",
                cmap="RdBu_r" if column >= 4 else "gray",
                norm=norms[band, column],
            )
            ax.set_title(title)
            ax.set_xlabel("Native pixel (reference AIA for projection)")
            cb = fig.colorbar(im, ax=ax, shrink=0.7)
            if norms[band, column] is not None:
                ticks = np.linspace(
                    norms[band, column].vmin, norms[band, column].vmax, 4
                )
                cb.set_ticks(ticks, labels=[f"{v:.3g}" for v in ticks])
            if not np.isfinite(data).any():
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable / registration not accepted",
                    transform=ax.transAxes,
                    ha="center",
                )
        fig.suptitle(f"{name}: limited preprocessing; pointing retained; no IDL")
        fig.savefig(out / f"{name}_comparison.png", dpi=140)
        plt.close(fig)
    for audit in json.loads((out / "registration_audit.json").read_text()):
        if audit.get("previous_image"):
            p = Path(audit["previous_image"])
            inputs[str(p)] = file_sha256(p)
    limits = json.loads((out / "unresolved.json").read_text())
    limits.extend(
        {"pair": r["pair"], "reason": warning}
        for r in records
        for warning in r["warnings"]
    )
    limits.extend(
        {"pair": r["pair"], "instrument": "AIA", "reason": reason}
        for r in records
        for reason in r["aia_preprocessing"]["reasons"]
    )
    limits.append(
        {"reason": "full_instrument_calibration_and_jet_identity_not_validated"}
    )
    dump(out / "unresolved.json", limits)
    dump(out / "paired_samples.json", bundle)
    dump(out / "preprocessing_audit.json", records)
    dump(
        out / "input_manifest.json",
        [{"path": p, "sha256": h} for p, h in sorted(inputs.items())],
    )
    dump(
        out / "runtime_check.json",
        {
            "idl_on_path": shutil.which("idl"),
            "external_processes_forbidden": True,
            "idl_invoked": False,
            "pair_count": len(records),
        },
    )
    rows = [
        "|组|AIA UTC|EUVI UTC|发射时间差/s|比例（输出/原图）|WCS误差/像素|",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in records:
        rows.append(
            f"|{r['pair']}|{r['aia_midpoint']}|{r['euvi_midpoint']}|{r['delta_emission_s']:.3f}|{r['output_to_raw_ratio']:.6g}|{r['native_wcs_error_pixel']:.3g}|"
        )
    report = "# 纯 Python EUVI 预处理与共同视角测试\n\n"
    report += "此工作流不调用 IDL。输出是有限预处理，并非 SECCHI_PREP 完整校准。SunPy 用于 FITS/WCS 和显示重投影。\n\n"
    report += "## 输入处理结果\n\n" + "\n".join(rows) + "\n\n"
    report += "## 处理证据与限制\n\n已执行：明确 DN 按头部 EXPTIME 归一化、保留无效像素及原生 WCS、同仪器时序配准和球面显示投影。已是计数率的数据不会再次除曝光。\n\n"
    report += "单位未声明的 AIA 保留原值，仅供浏览；不标为 DN/s，不产生有效定量差分。各组状态见 preprocessing_audit.json 的 aia_preprocessing。\n\n"
    report += "头部已有：逐帧 HISTORY 证据见 preprocessing_audit.json；不会重复执行 rectify、指向或滚转更新。外部归档历史可能含 IDL 字样，它不表示此工作流调用 IDL。\n\n"
    report += "未执行：偏置、SEB 反处理、有效曝光重算、平场、响应和新版指向校准。采用头部曝光不等于已验证全部机载处理。遥测缺块若无法解析则明确报限；零值不会一律掩蔽。\n\n"
    report += "差分采用同通道前一帧；外围配准区只是工程假设，未通过配准时保留缺测，不生成替代差分。同波段同类图使用固定色标。两仪器色标分别设置，不作绝对辐射比较。\n\n"
    report += "共同视角采用高度 0 球面显示假设；角度未被人为调整，显示重合不能确认喷流身份或三维高度。原生标注不随显示投影改变。\n\n"
    report += "## 对照图片\n\n" + "\n\n".join(
        f"![{p[0]}]({p[0]}_comparison.png)" for p in panels
    )
    report += "\n\n## 复现与使用\n\n使用已选解释器执行 `python -m solar_apps.workflows.jet_lab.preprocessing_check --config resolved_config.json`，复现时将输出目录改为新的目录。Jet Lab 的“加载样本”选择 paired_samples.json；不要恢复旧标注覆盖新图像身份。\n\n"
    report += "[SunPy EUVIMap](https://docs.sunpy.org/en/stable/generated/api/sunpy.map.sources.EUVIMap.html)；[官方 SECCHI_PREP 处理说明](https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/doc/euvi_prep.html)。\n"
    (out / "REPORT.md").write_text(report)
    finalize_manifest(out)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
