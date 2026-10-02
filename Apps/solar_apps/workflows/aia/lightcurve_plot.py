# 模块用途: 绘制 AIA 光变/流量表格数据，支持多波段和时间范围配置。
# 主要输入: AIA_Flux_data.py 生成的表格数据。
# 主要输出/运行说明: 输出科研绘图风格的光变曲线图。
"""
Plot caller-supplied AIA light curves.

"""

import argparse
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import build_scientific_image_filename

__all__ = [
    "get_log10_bounds",
    "load_single_file",
    "main",
    "normalize_data",
    "plot_multi_data",
]


def _configure_plot_style() -> None:
    """Apply the historical font settings only when plotting is requested."""
    plt.rcParams["font.family"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_single_file(file_path: str) -> tuple[list[datetime], list[float], str]:
    """加载单个CSV文件的数据，过滤非正值（对数坐标要求）"""
    p = Path(file_path)
    if not p.exists():
        print(f"警告：文件 {file_path} 不存在，已跳过")
        return None, None, p.name

    time_list = []
    flux_list = []
    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) != 2:
                    continue
                # 解析时间和通量数据
                dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                flux = float(row[1])
                # 对数坐标不能处理非正值，过滤无效数据
                if flux > 0:
                    time_list.append(dt)
                    flux_list.append(flux)
        print(f"成功加载 {file_path}，共 {len(time_list)} 条有效数据（已过滤非正值）")
        return time_list, flux_list, p.name
    except Exception as e:
        print(f"加载 {file_path} 失败: {str(e)}，已跳过")
        return None, None, p.name


def normalize_data(data: list[float]) -> list[float]:
    """归一化数据到[0, 1]范围（当前未使用）"""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:  # 避免除零
        return [0.0 for _ in data]
    return [(x - min_val) / (max_val - min_val) for x in data]


def get_log10_bounds(data: list[float]) -> tuple[float, float]:
    """计算数据的10的幂次边界，确保覆盖所有数据且为10的倍数"""
    if not data:
        return 1.0, 10.0  # 默认范围

    min_val = min(data)
    max_val = max(data)

    # 计算下限：小于等于最小值的最大10的幂次
    lower_exp = math.floor(math.log10(min_val))
    lower_bound = 10**lower_exp

    # 计算上限：大于等于最大值的最小10的幂次
    upper_exp = math.ceil(math.log10(max_val))
    upper_bound = 10**upper_exp

    return lower_bound, upper_bound


def plot_multi_data(
    all_data: list[tuple[list[datetime], list[float], str]],
    output_dir: str,
    file_names: list[str],
    mark_times: list[datetime] = None,
):
    """在同一张图上绘制多个文件的数据，标题包含所选文件名信息，支持标记特定时间点"""
    if not all_data:
        print("没有有效数据用于绘图！")
        return

    _configure_plot_style()

    # 准备不同的线条样式用于区分不同文件
    line_styles = ["-", "--", "-.", ":"]
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    style_idx = 0

    plt.figure(figsize=(16, 12))

    # 收集所有时间和通量数据用于设置坐标轴范围
    all_times = []
    all_fluxes = []
    for time_list, flux_list, _ in all_data:
        if time_list:
            all_times.extend(time_list)
        if flux_list:
            all_fluxes.extend(flux_list)

    # 确定X轴范围
    if not all_times or not all_fluxes:
        print("没有有效时间或通量数据，无法绘图")
        return
    x_min, x_max = min(all_times), max(all_times)

    # 计算Y轴对数坐标范围（10的倍数）
    y_min, y_max = get_log10_bounds(all_fluxes)

    for time_list, flux_list, label in all_data:
        if not time_list or not flux_list:
            continue

        # 循环使用线条样式和颜色
        ls = line_styles[style_idx % len(line_styles)]
        color = colors[style_idx % len(colors)]
        plt.plot(
            time_list, flux_list, linestyle=ls, color=color, linewidth=1.5, label=label
        )
        style_idx += 1

    # 绘制标记时间点的垂直线
    if mark_times:
        for mark_time in mark_times:
            # 检查标记时间是否在数据时间范围内
            if x_min <= mark_time <= x_max:
                plt.axvline(
                    x=mark_time,
                    color="purple",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label=f'Marker: {mark_time.strftime("%H:%M:%S")}',
                )
            else:
                print(f"警告：标记时间 {mark_time} 超出数据时间范围，已跳过")

    # 图表配置
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("通量", fontsize=12)
    plt.title("The postflareloop region", fontsize=14)
    plt.legend(loc="best")  # 显示图例（使用文件名作为标签）

    # 设置X轴范围
    plt.xlim(x_min, x_max)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # 设置Y轴为对数坐标
    ax.set_yscale("log")

    # 配置对数坐标刻度，使用科学计数法显示
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))

    # 设置Y轴范围为10的倍数区间
    plt.ylim(y_min, y_max)

    # 时间轴刻度设置
    time_interval_minutes = 5
    tick_times = []
    current_tick = x_min
    # 确保起始刻度是5分钟的整数倍
    minutes_to_add = (5 - (current_tick.minute % 5)) % 5
    current_tick += timedelta(minutes=minutes_to_add)

    while current_tick <= x_max:
        tick_times.append(current_tick)
        current_tick += timedelta(minutes=time_interval_minutes)

    # 添加垂直参考线
    for tick in tick_times:
        plt.axvline(x=tick, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    ax.xaxis.set_major_locator(ticker.FixedLocator(mdates.date2num(tick_times)))
    plt.xticks(rotation=45, ha="right")

    plt.grid(alpha=0.2, which="both")  # 显示主、次网格线
    plt.tight_layout()

    output_path = Path(output_dir) / build_scientific_image_filename(
        sequence=1,
        start_time=x_min,
        end_time=x_max,
        instrument="aia",
        product="postflare_loop_lightcurve",
        generated_at=datetime.now(timezone.utc),
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"对比曲线图已保存至：{output_path}")
    plt.show()


def main(argv=None) -> int:
    path_config = load_script_config(
        "sdo_aia_lightcurve_plot",
        {
            "input_dir": "data/aia/flux",
            "output_dir": "outputs/aia/flux",
        },
    )
    # 解析命令行参数（修改为按文件名选取）
    parser = argparse.ArgumentParser(
        description="从自定义文件夹读取指定文件名的CSV文件并绘制对比曲线"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=path_config["input_dir"],
        help="数据文件所在的文件夹路径（例如：<data-directory>）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=path_config["output_dir"],
        help="图像输出文件夹路径（默认与输入文件夹相同）",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=[
            "aia_94_selected_2.csv",
            "aia_131_selected_2.csv",
            "aia_171_selected_2.csv",
            "aia_193_selected_2.csv",
            "aia_211_selected_2.csv",
            "aia_304_selected_2.csv",
            "aia_1600_selected_2.csv",
        ],  # 设置默认文件列表
        required=False,  # 可选参数（因为有默认值）
        help="要选取的CSV文件名列表（例如：file1.csv file2.csv file3.csv）",
    )
    parser.add_argument(
        "--times",
        type=str,
        nargs="*",
        default=[],
        help='需要标记的时间点列表（格式：YYYY-MM-DD HH:MM:SS，例如："2000-01-01 00:00:00" "2000-01-01 00:01:00"）',
    )
    args = parser.parse_args(argv)

    input_dir = args.dir
    input_path = Path(input_dir)
    # 验证文件夹是否存在
    if not input_path.is_dir():
        print(f"错误：文件夹 '{input_dir}' 不存在，请检查路径是否正确")
        return 0

    # 处理输出文件夹
    output_dir = args.output if args.output else input_dir
    output_path = Path(output_dir)
    # 确保输出文件夹存在
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析时间点
    mark_times = []
    if args.times:
        for time_str in args.times:
            try:
                mark_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                mark_times.append(mark_time)
                print(f"已添加时间标记: {mark_time}")
            except ValueError:
                print(
                    f"警告：时间格式错误 '{time_str}'，请使用YYYY-MM-DD HH:MM:SS格式，已跳过"
                )

    # 获取文件夹中所有CSV文件（用于验证）
    all_csv_files = [
        f.name for f in input_path.iterdir() if f.is_file() and f.suffix == ".csv"
    ]

    if not all_csv_files:
        print(f"警告：文件夹 '{input_dir}' 中没有找到CSV文件")
        return 0

    # 验证并筛选用户指定的文件
    selected_files = []
    for file in args.files:
        # 确保文件名以.csv结尾
        if not file.endswith(".csv"):
            file += ".csv"
        if file in all_csv_files:
            selected_files.append(file)
        else:
            print(f"警告：文件 '{file}' 不存在于文件夹 '{input_dir}' 中，已跳过")

    if not selected_files:
        print("错误：没有有效的文件可供处理")
        return 0

    print(f"已选择文件：{selected_files}")

    # 加载并处理选中的文件数据
    all_data = []
    for file in selected_files:
        file_path = str(input_path / file)
        time_list, flux_list, label = load_single_file(file_path)
        if time_list and flux_list:
            all_data.append((time_list, flux_list, label))

    # 绘制所有数据
    plot_multi_data(all_data, output_dir, selected_files, mark_times)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
