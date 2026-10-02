# 模块用途: 将 DEM 诊断结果与射电源形态叠加比较。
# 主要输入: DEM 分析结果、AIA 图像和射电源数据。
# 主要输出/运行说明: 输出热等离子体结构与射电辐射位置的对比图。
"""
DEM/radio source overlay calculations.
"""

import argparse
import glob
import os
import re
from copy import deepcopy
from datetime import datetime, timezone

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import configured_scientific_image_path

__all__ = [
    "CONFIG",
    "SolarMap",
    "build_parser",
    "find_matching_radio",
    "get_display_extent",
    "get_radio_time",
    "get_tb_extent",
    "load_radio",
    "load_tb",
    "main",
    "make_tb_colormap",
    "plot_tb",
]


# ============================================================
#  字体配置（必须在其他 matplotlib 调用之前执行）
# ============================================================
def _setup_font() -> None:
    """为 Windows/macOS 环境配置中文字体，并修复负号显示问题。"""
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            rcParams["font.sans-serif"] = [font] + rcParams["font.sans-serif"]
            rcParams["font.family"] = "sans-serif"
            break
    rcParams["axes.unicode_minus"] = False


# ============================================================
#  配置（所有可调参数集中于此）
# ============================================================
CONFIG = {
    # ┌─────────────────────────────────────────────────────────┐
    # │  文件路径                                                │
    # └─────────────────────────────────────────────────────────┘
    "aia_fits_path": "data/aia/example.fits",
    "tb_data_path": "data/radio/Tb_223MHz.npy",
    # ┌─────────────────────────────────────────────────────────┐
    # │  射电源叠加开关 & 路径                                   │
    # │                                                          │
    # │  overlay_radio : True  → 叠加射电等高线                 │
    # │                  False → 仅显示 Tb 图，跳过射电相关步骤 │
    # └─────────────────────────────────────────────────────────┘
    "overlay_radio": True,
    "radio_sources_dir": "data/radio/223MHz/RR",
    "radio_sources_pattern": "223MHz_*.fits",
    # ┌─────────────────────────────────────────────────────────┐
    # │  射电源时间匹配策略                                      │
    # │                                                          │
    # │  time_match_level 决定匹配粒度（粗 → 细）：             │
    # │    "minute" → 筛选与 AIA 同年月日时分（HH:MM）的文件， │
    # │               再选 |秒差| 最小的一帧  ← 默认，推荐     │
    # │    "hour"   → 筛选同小时内，选最近的一帧               │
    # │    "any"    → 不预筛选，全目录选时间差最小的            │
    # │                                                          │
    # │  自动降级：若当前粒度无候选，自动放宽至下一粒度并警告   │
    # └─────────────────────────────────────────────────────────┘
    "time_match_level": "minute",
    # ┌─────────────────────────────────────────────────────────┐
    # │  Tb 网格参数                                            │
    # │  像素尺度 3 arcsec/pixel，覆盖范围 ±1150 arcsec         │
    # │  验证：(1150-(-1150))/3 + 1 ≈ 767 → shape(767,767) 吻合│
    # └─────────────────────────────────────────────────────────┘
    "tb_pixel_size": 3,  # arcsec/pixel
    "tb_xmin": -1150,  # arcsec
    "tb_xmax": 1150,
    "tb_ymin": -1150,
    "tb_ymax": 1150,
    # ┌─────────────────────────────────────────────────────────┐
    # │  显示模式                                               │
    # │  "full"       : AIA 完整视场                            │
    # │  "solar_disk" : 日面中心 ± rsun × solar_padding_factor  │
    # │  "custom"     : 手动指定 display_x/y_range（arcsec）    │
    # └─────────────────────────────────────────────────────────┘
    "display_mode": "custom",
    "display_x_range": (-1600, 1600),
    "display_y_range": (-1600, 1600),
    "solar_padding_factor": 1.15,
    # ┌─────────────────────────────────────────────────────────┐
    # │  画布与 Tb 色彩                                         │
    # └─────────────────────────────────────────────────────────┘
    "figsize": (10, 9),
    "dpi": 300,
    "colorbar_label": "Tb (MK)",
    # 颜色映射节点（黑 → 暗红 → 橙红 → 金黄 → 白）
    "cmap_colors": ["#000000", "#8B0000", "#FF4500", "#FFD700", "#FFFFFF"],
    "cmap_positions": [0.0, 0.2, 0.4, 0.7, 1.0],
    # 动态范围（百分位数裁剪，抑制极端值干扰色标）
    "percentile_low": 1,
    "percentile_high": 99,
    # ┌─────────────────────────────────────────────────────────┐
    # │  太阳轮廓叠加                                           │
    # └─────────────────────────────────────────────────────────┘
    "draw_optical_limb": True,  # True：叠加白色虚线 AIA 光学日面
    "draw_radio_limb": False,  # True：叠加红色虚线射电日面估计
    "radio_limb_factor": 1.08,  # 射电日面半径 = rsun × radio_limb_factor
    # ┌─────────────────────────────────────────────────────────┐
    # │  射电源强度梯度等高线样式                               │
    # │                                                          │
    # │  radio_contour_levels : 相对强度阈值（0~1），           │
    # │                         层数越多梯度越细腻              │
    # │  radio_contour_colors : 与 levels 一一对应的颜色        │
    # │  radio_contour_linewidths : 与 levels 一一对应的线宽    │
    # │  radio_smooth_sigma   : 高斯平滑 σ（像素），            │
    # │                         越大轮廓越平滑                  │
    # └─────────────────────────────────────────────────────────┘
    "radio_contour_levels": [0.3, 0.6, 0.9, 0.95, 0.99],
    "radio_contour_colors": ["cyan", "deepskyblue", "lime", "orange", "red"],
    "radio_contour_linewidths": [1.0, 1.5, 2.0, 2.5, 3.0],
    "radio_smooth_sigma": 1.5,
    # ┌─────────────────────────────────────────────────────────┐
    # │  网格与输出                                             │
    # └─────────────────────────────────────────────────────────┘
    "show_grid": True,
    "save_figure": True,
    "output_filename": "brightness_temperature_radio_overlay",
}
_DEFAULT_CONFIG = deepcopy(CONFIG)
CONFIG = deepcopy(_DEFAULT_CONFIG)


# ============================================================
#  AIA 数据封装
# ============================================================
class SolarMap:
    """
    读取 SDO/AIA lev1 FITS 文件，解算 WCS 坐标并提取太阳几何参数。

    属性
    ----
    extent     : [x_left, x_right, y_bottom, y_top]（arcsec）
    nx, ny     : 图像像素数
    rsun       : 太阳视半径（arcsec）
    sun_center : (cx, cy) 日面中心坐标（arcsec）
    obs_time   : 格式化时间字符串，如 '2000-01-01 04:47:47 UT'
    obs_dt     : datetime 对象，用于与射电文件做时间匹配
    """

    def __init__(self, path: str) -> None:
        with fits.open(path) as hdul:
            self.header = hdul[1].header
        self._build_wcs()
        self._extract_solar_geometry()
        self.obs_time = self._parse_obs_time(path)
        self.obs_dt = _str_to_datetime(self.obs_time)

    def _build_wcs(self) -> None:
        h = self.header
        nx, ny = h["NAXIS1"], h["NAXIS2"]
        dx, dy = h["CDELT1"], h["CDELT2"]
        x = h["CRVAL1"] + (np.arange(nx) + 1 - h["CRPIX1"]) * dx
        y = h["CRVAL2"] + (np.arange(ny) + 1 - h["CRPIX2"]) * dy
        self.extent = [
            x.min() - abs(dx) / 2,
            x.max() + abs(dx) / 2,
            y.min() - abs(dy) / 2,
            y.max() + abs(dy) / 2,
        ]
        self.nx, self.ny = nx, ny

    def _extract_solar_geometry(self) -> None:
        h = self.header
        self.rsun = h["RSUN_OBS"] if "RSUN_OBS" in h else h["R_SUN"] * abs(h["CDELT1"])
        self.sun_center = (h["CRVAL1"], h["CRVAL2"])

    def _parse_obs_time(self, path: str) -> str:
        for key in ("T_OBS", "DATE-OBS", "DATE_OBS"):
            if key in self.header:
                raw = str(self.header[key]).strip()
                m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", raw)
                if m:
                    return f"{m.group(1)} {m.group(2)} UT"
                m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", raw)
                if m:
                    return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)} UT"
        fname = os.path.basename(path)
        m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})Z?", fname)
        if m:
            return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)} UT"
        m = re.search(r"(\d{4})(\d{2})(\d{2})[_T]?(\d{2})(\d{2})(\d{2})", fname)
        if m:
            return (
                f"{m.group(1)}-{m.group(2)}-{m.group(3)} "
                f"{m.group(4)}:{m.group(5)}:{m.group(6)} UT"
            )
        return "Unknown"


# ============================================================
#  时间工具
# ============================================================


def _str_to_datetime(time_str: str):
    """
    将 'YYYY-MM-DD HH:MM:SS[.mmm] UT' 解析为 datetime 对象。
    解析失败时返回 None。
    """
    if not time_str or time_str == "Unknown":
        return None
    # 带毫秒
    m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\.\d+", time_str)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # 不带毫秒
    m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", time_str)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def get_radio_time(path: str) -> str:
    """
    提取射电 FITS 文件的观测时间字符串。
    优先读取 FITS 头文件关键字，回退到文件名正则匹配。
    返回格式：'YYYY-MM-DD HH:MM:SS[.mmm] UT'
    """
    try:
        with fits.open(path) as hdul:
            header = hdul[0].header
        for key in ("DATE-OBS", "T_OBS", "DATE_OBS"):
            if key in header:
                raw = str(header[key]).strip()
                m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", raw)
                if m:
                    return f"{m.group(1)} {m.group(2)} UT"
                m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", raw)
                if m:
                    return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)} UT"
    except Exception:
        pass

    fname = os.path.basename(path)
    # CSRH/MUSER 合成示例：149MHz_2000101_000000_000.fits（月份无前导零）
    m = re.search(r"(\d{4})(\d{1,2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", fname)
    if m:
        year = m.group(1)
        month = m.group(2).zfill(2)
        day = m.group(3)
        hh, mm, ss, ms = m.group(4), m.group(5), m.group(6), m.group(7)
        return f"{year}-{month}-{day} {hh}:{mm}:{ss}.{ms} UT"
    # 标准格式：YYYYMMDD_HHMMSS 或 YYYYMMDD-HHMMSS
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})(\d{2})", fname)
    if m:
        return (
            f"{m.group(1)}-{m.group(2)}-{m.group(3)} "
            f"{m.group(4)}:{m.group(5)}:{m.group(6)} UT"
        )
    # 连续 14 位：YYYYMMDDHHMMSS
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", fname)
    if m:
        return (
            f"{m.group(1)}-{m.group(2)}-{m.group(3)} "
            f"{m.group(4)}:{m.group(5)}:{m.group(6)} UT"
        )
    return "Unknown"


def find_matching_radio(files: list, aia_dt: datetime) -> tuple:
    """
    从射电文件列表中选出与 AIA 时间最接近的一帧。

    匹配策略（由 CONFIG["time_match_level"] 控制）
    ──────────────────────────────────────────────
    "minute"（默认）
        先筛选与 AIA 同年月日时分（YYYY-MM-DD HH:MM）的候选文件，
        再从中选 |秒数差| 最小的一帧。
        → 例：AIA 时间 04:47:47，则只考虑 04:47:xx 的射电文件，
          选 |秒 - 47| 最小者（如 04:47:46、04:47:48 中取更近的）。

    "hour"
        先筛选同年月日小时（YYYY-MM-DD HH）的候选文件，
        再选时间差最小的。

    "any"
        不做时间预筛选，直接在全目录中选时间差最小的。

    自动降级
        若当前粒度无候选，自动放宽至下一粒度并打印警告。

    Parameters
    ----------
    files  : list[str]，射电 FITS 路径列表（已 sorted）
    aia_dt : datetime，AIA 观测时刻

    Returns
    -------
    best_path : str，最佳匹配文件路径
    best_time : str，对应的时间字符串
    dt_diff   : float | None，时间差（秒）；无法计算时为 None
    """
    if aia_dt is None:
        print("  [警告] AIA 时间解析失败，默认选择第一帧射电文件")
        return files[0], get_radio_time(files[0]), None

    # ── 为每个文件解析 datetime ────────────────────────────────
    parsed = []  # [(path, time_str, dt), ...]
    for p in files:
        t_str = get_radio_time(p)
        t_dt = _str_to_datetime(t_str)
        parsed.append((p, t_str, t_dt))

    aia_key_minute = aia_dt.strftime("%Y-%m-%d %H:%M")
    aia_key_hour = aia_dt.strftime("%Y-%m-%d %H")

    def _filter_by_key(key_fn, aia_key):
        return [
            (p, ts, dt)
            for p, ts, dt in parsed
            if dt is not None and key_fn(dt) == aia_key
        ]

    level = CONFIG["time_match_level"]
    candidates = []

    if level == "minute":
        candidates = _filter_by_key(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M"), aia_key_minute
        )
        if not candidates:
            print(f"  [降级] 同分钟（{aia_key_minute}）无候选，改用同小时筛选")
            candidates = _filter_by_key(
                lambda dt: dt.strftime("%Y-%m-%d %H"), aia_key_hour
            )
        if not candidates:
            print(f"  [降级] 同小时（{aia_key_hour}）无候选，改用全局最近匹配")
            candidates = [(p, ts, dt) for p, ts, dt in parsed if dt is not None]

    elif level == "hour":
        candidates = _filter_by_key(lambda dt: dt.strftime("%Y-%m-%d %H"), aia_key_hour)
        if not candidates:
            print(f"  [降级] 同小时（{aia_key_hour}）无候选，改用全局最近匹配")
            candidates = [(p, ts, dt) for p, ts, dt in parsed if dt is not None]

    else:  # "any"
        candidates = [(p, ts, dt) for p, ts, dt in parsed if dt is not None]

    if not candidates:
        print("  [警告] 所有文件均无法解析时间，默认选择第一帧")
        return files[0], get_radio_time(files[0]), None

    # ── 选时间差最小的一帧 ─────────────────────────────────────
    best_path, best_time, best_dt = min(
        candidates, key=lambda x: abs((x[2] - aia_dt).total_seconds())
    )
    dt_diff = abs((best_dt - aia_dt).total_seconds())

    return best_path, best_time, dt_diff


# ============================================================
#  射电数据加载
# ============================================================


def load_radio(path: str) -> tuple:
    """
    读取射电成像 FITS，解算 WCS 坐标，返回强度图和坐标范围。

    Returns
    -------
    data   : 2D ndarray，射电强度图
    extent : [x_min, x_max, y_min, y_max]（arcsec）
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    data = np.nan_to_num(data)
    ny, nx = data.shape

    crpix1 = header.get("CRPIX1", 1)
    crpix2 = header.get("CRPIX2", 1)
    crval1 = header.get("CRVAL1", 0)
    crval2 = header.get("CRVAL2", 0)
    cdelt1 = header.get("CDELT1", 1)
    cdelt2 = header.get("CDELT2", 1)

    x = crval1 + (np.arange(nx) + 1 - crpix1) * cdelt1
    y = crval2 + (np.arange(ny) + 1 - crpix2) * cdelt2

    return data, [x.min(), x.max(), y.min(), y.max()]


# ============================================================
#  辅助函数
# ============================================================


def get_display_extent(aia_map: SolarMap) -> list:
    """返回最终绘图坐标范围 [x_min, x_max, y_min, y_max]（arcsec）。"""
    mode = CONFIG["display_mode"]
    if mode == "full":
        return aia_map.extent
    if mode == "solar_disk":
        cx, cy = aia_map.sun_center
        r = aia_map.rsun * CONFIG["solar_padding_factor"]
        return [cx - r, cx + r, cy - r, cy + r]
    if mode == "custom":
        xmin, xmax = CONFIG["display_x_range"]
        ymin, ymax = CONFIG["display_y_range"]
        return [xmin, xmax, ymin, ymax]
    return aia_map.extent


def make_tb_colormap() -> LinearSegmentedColormap:
    """构建 Tb 专用色图（黑→暗红→橙红→金黄→白）。"""
    return LinearSegmentedColormap.from_list(
        "tb_solar",
        list(zip(CONFIG["cmap_positions"], CONFIG["cmap_colors"], strict=False)),
        N=256,
    )


def load_tb(path: str) -> np.ndarray:
    """加载 Tb 数组（K → MK），替换 NaN/Inf 为 0。"""
    data = np.load(path).astype(np.float64)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) / 1e6


def get_tb_extent() -> list:
    """返回 Tb 图像的 imshow extent（像素外边缘，arcsec）。"""
    half = CONFIG["tb_pixel_size"] / 2.0
    return [
        CONFIG["tb_xmin"] - half,
        CONFIG["tb_xmax"] + half,
        CONFIG["tb_ymin"] - half,
        CONFIG["tb_ymax"] + half,
    ]


# ============================================================
#  绘图
# ============================================================


def _draw_limbs(ax, sun_center, rsun) -> None:
    """叠加光学日面（白色虚线圆）和射电日面（红色虚线圆）轮廓。"""
    cx, cy = sun_center
    if CONFIG["draw_optical_limb"]:
        ax.plot(
            cx,
            cy,
            "+",
            color="white",
            markersize=15,
            markeredgewidth=1.5,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                radius=rsun,
                fill=False,
                color="white",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                zorder=5,
                label=f'AIA optical limb  ({rsun:.0f}")',
            )
        )
    if CONFIG["draw_radio_limb"]:
        r_radio = rsun * CONFIG["radio_limb_factor"]
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                radius=r_radio,
                fill=False,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                zorder=5,
                label=f"223 MHz radio limb  ({r_radio:.0f}\", ×{CONFIG['radio_limb_factor']:.3f})",
            )
        )


def _draw_radio_gradient(ax, radio_data: np.ndarray, radio_extent: list) -> list:
    """
    叠加射电源强度梯度等高线（仿 DEM_RS.py plot_overlay 步骤 5–7）。

    流程
    ────
    1. 高斯平滑，消除噪声碎小等高线
    2. 将 CONFIG 中的相对阈值（0~1）映射到数据实际值域
    3. ax.contour() 绘制多层等高线

    Returns
    -------
    list[Line2D]：图例代理线
    """
    radio_s = gaussian_filter(radio_data, CONFIG["radio_smooth_sigma"])
    rmin, rmax = radio_s.min(), radio_s.max()
    levels = [rmin + lv * (rmax - rmin) for lv in CONFIG["radio_contour_levels"]]

    ax.contour(
        radio_s,
        levels=levels,
        extent=radio_extent,
        origin="lower",
        colors=CONFIG["radio_contour_colors"],
        linewidths=CONFIG["radio_contour_linewidths"],
        zorder=6,
    )

    proxy = [
        Line2D(
            [0],
            [0],
            color=CONFIG["radio_contour_colors"][i],
            linewidth=CONFIG["radio_contour_linewidths"][i],
            label=f"Radio {int(CONFIG['radio_contour_levels'][i] * 100)}%",
        )
        for i in range(len(CONFIG["radio_contour_levels"]))
    ]
    return proxy


def plot_tb(
    tb_data: np.ndarray,
    aia_map: SolarMap,
    radio_data: np.ndarray = None,
    radio_extent: list = None,
    radio_time: str = "",
) -> tuple:
    """
    绘制 Tb 底图，叠加日面轮廓，并（可选）叠加射电强度梯度等高线。

    Parameters
    ----------
    tb_data      : (ny, nx) ndarray，亮温度（MK）
    aia_map      : SolarMap
    radio_data   : 2D ndarray | None；为 None 时跳过等高线
    radio_extent : list | None
    radio_time   : str，射电观测时间（用于标题）

    Returns
    -------
    fig, ax
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    # ── Tb 底图 ──────────────────────────────────────────────
    vmin = np.nanpercentile(tb_data, CONFIG["percentile_low"])
    vmax = np.nanpercentile(tb_data, CONFIG["percentile_high"])
    im = ax.imshow(
        tb_data,
        extent=get_tb_extent(),
        origin="lower",
        cmap=make_tb_colormap(),
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    # ── 日面轮廓 ─────────────────────────────────────────────
    _draw_limbs(ax, aia_map.sun_center, aia_map.rsun)

    # ── 射电强度梯度等高线（overlay_radio=True 时才会传入数据）
    radio_proxy = []
    if radio_data is not None and radio_extent is not None:
        radio_proxy = _draw_radio_gradient(ax, radio_data, radio_extent)

    # ── 坐标轴 ───────────────────────────────────────────────
    x0, x1, y0, y1 = get_display_extent(aia_map)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("Solar X (arcsec)", fontsize=12)
    ax.set_ylabel("Solar Y (arcsec)", fontsize=12)
    ax.tick_params(labelsize=10)
    if CONFIG["show_grid"]:
        ax.grid(True, alpha=0.3, linestyle="--")

    # ── 图例 ─────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    handles += radio_proxy
    labels += [p.get_label() for p in radio_proxy]
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize=9,
            framealpha=0.7,
            facecolor="black",
            labelcolor="white",
        )

    # ── 色标 ─────────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(CONFIG["colorbar_label"], fontsize=12, labelpad=10)

    # ── 标题（射电叠加时附加射电时间）───────────────────────
    subtitle = f"AIA : {aia_map.obs_time}"
    if radio_time:
        subtitle += f"\nRadio: {radio_time}"
    fig.suptitle(
        f"Tb at 223.0 MHz\n{subtitle}",
        fontsize=13,
        fontweight="bold",
        color="black",
        linespacing=1.8,
    )
    plt.tight_layout()
    return fig, ax


# ============================================================
#  主程序
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the compatibility command-line parser for this recipe."""

    return argparse.ArgumentParser(
        description="Overlay DEM brightness temperature and radio-source contours."
    )


def main(argv=None) -> int:
    global CONFIG

    build_parser().parse_known_args(argv)
    CONFIG = load_script_config("dem_radio_source_overlay", _DEFAULT_CONFIG)
    sep = "=" * 60

    # ── Step 1：读取 AIA ─────────────────────────────────────
    print(sep)
    print("Step 1  从 AIA FITS 读取坐标与太阳几何参数")
    aia_map = SolarMap(CONFIG["aia_fits_path"])
    print(f"  图像尺寸   : ({aia_map.ny} × {aia_map.nx}) px")
    print(f"  extent X   : [{aia_map.extent[0]:.1f}, {aia_map.extent[1]:.1f}] arcsec")
    print(f"  extent Y   : [{aia_map.extent[2]:.1f}, {aia_map.extent[3]:.1f}] arcsec")
    print(f"  rsun       : {aia_map.rsun:.2f} arcsec")
    print(f"  sun_center : {aia_map.sun_center} arcsec")
    print(f"  obs_time   : {aia_map.obs_time}")

    # ── Step 2：加载 Tb ──────────────────────────────────────
    print("\nStep 2  加载 Tb 数据")
    tb_data = load_tb(CONFIG["tb_data_path"])
    ny_tb, nx_tb = tb_data.shape
    tb_ext = get_tb_extent()
    print(f"  形状     : ({ny_tb} × {nx_tb}) px")
    print(f"  像素尺度 : {CONFIG['tb_pixel_size']} arcsec/pixel")
    print(f"  值域     : [{tb_data.min():.4f}, {tb_data.max():.4f}] MK")
    print(f"  extent X : [{tb_ext[0]:.1f}, {tb_ext[1]:.1f}] arcsec")
    print(f"  extent Y : [{tb_ext[2]:.1f}, {tb_ext[3]:.1f}] arcsec")

    # ── Step 3：射电源时间匹配（受 overlay_radio 开关控制）──
    radio_data = None
    radio_extent = None
    radio_time = ""

    if not CONFIG["overlay_radio"]:
        print("\nStep 3  overlay_radio = False，跳过射电叠加")

    else:
        print(
            f"\nStep 3  加载射电源 FITS"
            f"（时间匹配策略：{CONFIG['time_match_level']}）"
        )

        radio_files = sorted(
            glob.glob(
                os.path.join(
                    CONFIG["radio_sources_dir"], CONFIG["radio_sources_pattern"]
                )
            )
        )

        if not radio_files:
            print("  [警告] 未找到射电 FITS 文件，跳过强度梯度绘制")
        else:
            print(f"  共找到 {len(radio_files)} 个射电文件")
            print(f"  AIA 时间  : {aia_map.obs_time}")

            best_path, radio_time, dt_diff = find_matching_radio(
                radio_files, aia_map.obs_dt
            )
            diff_str = f"{dt_diff:.3f} 秒" if dt_diff is not None else "未知"
            print(f"  最佳匹配  : {os.path.basename(best_path)}")
            print(f"  射电时间  : {radio_time}")
            print(f"  时间差    : {diff_str}")

            radio_data, radio_extent = load_radio(best_path)
            print(f"  形状      : {radio_data.shape}")
            print(
                f"  extent X  : [{radio_extent[0]:.1f}, {radio_extent[1]:.1f}] arcsec"
            )
            print(
                f"  extent Y  : [{radio_extent[2]:.1f}, {radio_extent[3]:.1f}] arcsec"
            )

    # ── Step 4：绘图 ─────────────────────────────────────────
    print("\nStep 4  绘图")
    fig, ax = plot_tb(
        tb_data,
        aia_map,
        radio_data=radio_data,
        radio_extent=radio_extent,
        radio_time=radio_time,
    )

    if CONFIG["save_figure"]:
        output_path = configured_scientific_image_path(
            CONFIG.get("output_filename"),
            sequence=1,
            start_time=aia_map.obs_dt,
            instrument="aia_dem_radio",
            channel="223mhz",
            product="brightness_temperature_radio_overlay",
            generated_at=datetime.now(timezone.utc),
        )
        fig.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        print(f"  图像已保存至: {output_path}")

    plt.show()
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
