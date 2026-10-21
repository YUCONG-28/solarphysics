"""Plot a precomputed brightness-temperature map using AIA reference geometry.

The historical command name is retained; this recipe does not invert AIA data.
"""

# ============================================================
#  导入
# ============================================================
import argparse
from copy import deepcopy
from datetime import datetime, timezone

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from solar_toolkit.visualization.plotting import setup_chinese_font
from solar_toolkit.xray_dem.brightness_temperature import (
    AiaMapGeometry as SolarMap,
    brightness_temperature_extent,
    load_brightness_temperature,
)

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import configured_scientific_image_path

__all__ = [
    "CONFIG",
    "SolarMap",
    "build_parser",
    "get_display_extent",
    "get_tb_extent",
    "load_tb",
    "main",
    "make_tb_colormap",
    "plot_tb",
]


# ============================================================
#  字体配置（必须在其他 matplotlib 调用之前执行）
# ============================================================
def _setup_font() -> None:
    """Configure the shared installed-font policy for scientific plots."""
    setup_chinese_font()


# ============================================================
#  配置（所有可调参数集中于此）
# ============================================================
CONFIG = {
    # ── 文件路径 ─────────────────────────────────────────────
    "aia_fits_path": "data/aia/example.fits",
    "tb_data_path": "data/radio/Tb_149MHz.npy",
    # ── Tb 网格参数（来自生成 Tb 数据时的配置文件）─────────────
    # 像素尺度：3 arcsec/pixel
    # 空间范围：X/Y ∈ [-1150, 1150] arcsec
    # 验证：(1150 - (-1150)) / 3 + 1 ≈ 767，与 Tb 数组形状 (767, 767) 吻合
    "tb_pixel_size": 3,  # arcsec/pixel
    "tb_xmin": -1150,  # arcsec
    "tb_xmax": 1150,  # arcsec
    "tb_ymin": -1150,  # arcsec
    "tb_ymax": 1150,  # arcsec
    # ── 显示模式 ─────────────────────────────────────────────
    # "full"       : 显示 AIA 完整视场
    # "solar_disk" : 以日面中心为基准，向外留白 solar_padding_factor 倍太阳半径
    # "custom"     : 手动指定 X/Y 范围（角秒）
    "display_mode": "custom",
    "display_x_range": (-1150, 1150),  # 与 Tb 网格范围一致（arcsec）
    "display_y_range": (-1150, 1150),  # 与 Tb 网格范围一致（arcsec）
    "solar_padding_factor": 1.15,  # solar_disk 模式留白系数
    # ── 画布与色彩 ───────────────────────────────────────────
    "figsize": (10, 9),
    "dpi": 300,
    "colorbar_label": "Tb (MK)",
    # 颜色映射：黑→暗红→橙红→金黄→白（从低温到高温）
    "cmap_colors": ["#000000", "#8B0000", "#FF4500", "#FFD700", "#FFFFFF"],
    "cmap_positions": [0.0, 0.2, 0.4, 0.7, 1.0],
    # 显示动态范围（百分位数裁剪，抑制极端值对色标的干扰）
    "percentile_low": 1,
    "percentile_high": 99,
    # ── 叠加轮廓 ─────────────────────────────────────────────
    # 白色虚线：AIA 光学日面（rsun，来自 FITS 头文件）
    "draw_optical_limb": True,
    # 红色虚线：149 MHz 射电日面估计（约 1.02–1.05 × rsun）
    # 调整 radio_limb_factor 使红圈与 Tb 可见边缘重合
    "draw_radio_limb": False,
    "radio_limb_factor": 1.08,
    # ── 网格与输出 ───────────────────────────────────────────
    "show_grid": True,
    "save_figure": False,
    "output_filename": "brightness_temperature_map",
}
_DEFAULT_CONFIG = deepcopy(CONFIG)
CONFIG = deepcopy(_DEFAULT_CONFIG)


# ============================================================
#  AIA 几何由 solar_toolkit.xray_dem.brightness_temperature 提供
# ============================================================


# ============================================================
#  辅助函数
# ============================================================


def get_display_extent(aia_map: SolarMap) -> list:
    """Return the configured bounds through the shared geometry helper."""
    return aia_map.display_extent(
        CONFIG["display_mode"],
        solar_padding_factor=CONFIG["solar_padding_factor"],
        x_range=CONFIG["display_x_range"],
        y_range=CONFIG["display_y_range"],
    )


def make_tb_colormap() -> LinearSegmentedColormap:
    """构建 Tb 专用色图（黑→暗红→橙红→金黄→白）。"""
    return LinearSegmentedColormap.from_list(
        "tb_solar",
        list(zip(CONFIG["cmap_positions"], CONFIG["cmap_colors"], strict=False)),
        N=256,
    )


def load_tb(path: str) -> np.ndarray:
    """Load the precomputed Kelvin map as MK using the shared display policy."""
    return load_brightness_temperature(path)


def get_tb_extent() -> list:
    """Return the configured Tb pixel-edge extent."""
    return brightness_temperature_extent(
        pixel_size=CONFIG["tb_pixel_size"],
        x_range=(CONFIG["tb_xmin"], CONFIG["tb_xmax"]),
        y_range=(CONFIG["tb_ymin"], CONFIG["tb_ymax"]),
    )


# ============================================================
#  绘图
# ============================================================


def _draw_limbs(ax, sun_center, rsun, config) -> None:
    """在坐标轴上叠加光学日面和射电日面轮廓。"""
    cx, cy = sun_center

    if config["draw_optical_limb"]:
        # 日面中心十字
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
        # 白色虚线圆：AIA 光学日面
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

    if config["draw_radio_limb"]:
        # 红色虚线圆：149 MHz 射电日面估计
        r_radio = rsun * config["radio_limb_factor"]
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
                label=f"149 MHz radio limb  ({r_radio:.0f}\",  ×{config['radio_limb_factor']:.3f})",
            )
        )


def plot_tb(tb_data: np.ndarray, aia_map: SolarMap) -> tuple:
    """
    绘制 Tb 图像，叠加太阳轮廓。

    Parameters
    ----------
    tb_data : ndarray, shape (ny, nx)
        亮温度数组（与 AIA 同一物理覆盖范围，直接使用 aia_map.extent）。
    aia_map : SolarMap
        提供 WCS extent、rsun、sun_center。

    Returns
    -------
    fig, ax : matplotlib Figure 和 Axes 对象
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    # ── Tb 图像 ───────────────────────────────────────────────
    vmin = np.nanpercentile(tb_data, CONFIG["percentile_low"])
    vmax = np.nanpercentile(tb_data, CONFIG["percentile_high"])

    # Tb 的空间范围由其生成网格决定（3 arcsec/pixel，±1150 arcsec），
    # 而非 AIA 的像素网格，因此使用 get_tb_extent() 而非 aia_map.extent
    tb_extent = get_tb_extent()

    im = ax.imshow(
        tb_data,
        extent=tb_extent,
        origin="lower",
        cmap=make_tb_colormap(),
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    # ── 太阳轮廓（光学 + 射电日面）───────────────────────────
    _draw_limbs(ax, aia_map.sun_center, aia_map.rsun, CONFIG)

    # ── 坐标轴范围、标签、网格 ───────────────────────────────
    x0, x1, y0, y1 = get_display_extent(aia_map)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("Solar X (arcsec)", fontsize=12)
    ax.set_ylabel("Solar Y (arcsec)", fontsize=12)
    ax.tick_params(labelsize=10)
    if CONFIG["show_grid"]:
        ax.grid(True, alpha=0.3, linestyle="--")

    # ── 图例、色标、标题 ─────────────────────────────────────
    ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.7,
        facecolor="black",
        labelcolor="white",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(CONFIG["colorbar_label"], fontsize=12, labelpad=10)

    fig.suptitle(
        f"Tb at 149.0 MHz\nAIA ref: {aia_map.obs_time}",
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
        description="Plot an AIA-referenced differential-emission-measure product."
    )


def main(argv=None) -> int:
    global CONFIG

    build_parser().parse_known_args(argv)
    CONFIG = load_script_config("sdo_aia_dem_inversion", _DEFAULT_CONFIG)
    sep = "=" * 60

    # ── Step 1：读取 AIA WCS 与太阳几何参数 ──────────────────
    print(sep)
    print("Step 1  从 AIA FITS 读取坐标与太阳几何参数")
    aia_map = SolarMap(CONFIG["aia_fits_path"])
    print(f"  图像尺寸   : ({aia_map.ny} × {aia_map.nx}) px")
    print(f"  extent X   : [{aia_map.extent[0]:.1f}, {aia_map.extent[1]:.1f}] arcsec")
    print(f"  extent Y   : [{aia_map.extent[2]:.1f}, {aia_map.extent[3]:.1f}] arcsec")
    print(f"  rsun       : {aia_map.rsun:.2f} arcsec")
    print(f"  sun_center : {aia_map.sun_center} arcsec")
    print(f"  obs_time   : {aia_map.obs_time}")

    # ── Step 2：加载 Tb 数据 ──────────────────────────────────
    print("\nStep 2  加载 Tb 数据")
    tb_data = load_tb(CONFIG["tb_data_path"])
    ny_tb, nx_tb = tb_data.shape
    tb_extent = get_tb_extent()
    print(f"  形状     : ({ny_tb} × {nx_tb}) px")
    print(f"  像素尺度 : {CONFIG['tb_pixel_size']} arcsec/pixel")
    print(f"  值域     : [{tb_data.min():.4f}, {tb_data.max():.4f}] MK")
    print(f"  extent X : [{tb_extent[0]:.1f}, {tb_extent[1]:.1f}] arcsec")
    print(f"  extent Y : [{tb_extent[2]:.1f}, {tb_extent[3]:.1f}] arcsec")

    # ── Step 3：绘图 ──────────────────────────────────────────
    print("\nStep 3  绘图")
    fig, ax = plot_tb(tb_data, aia_map)

    if CONFIG["save_figure"]:
        output_path = configured_scientific_image_path(
            CONFIG.get("output_filename"),
            sequence=1,
            start_time=str(aia_map.obs_time).replace(" UT", ""),
            instrument="aia_dem",
            channel="149mhz",
            product="brightness_temperature_map",
            generated_at=datetime.now(timezone.utc),
        )
        fig.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches="tight")
        print(f"  图像已保存至: {output_path}")

    plt.show()
    print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
