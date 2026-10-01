"""Source-map display ranges and directory statistics with explicit unit domains."""

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from ._source_map_helpers import (
    _combine_polarization_data,
    background_enabled_for_display,
)
from ._source_map_selection import _filter_bad_radio_files, _match_rr_ll_by_time


def _workflow():
    # Resolve retained monkeypatch anchors only after the facade has loaded.
    from . import source_map_workflow

    return source_map_workflow


def _global_range_one(fp):
    """
    Read a single file, return (nanmin, nanmax).
    For parallel calls, exceptions are caught independently.
    """
    try:
        # 处理元组情况：如果是元组，取第一个文件（RR文件）
        if isinstance(fp, tuple):
            file_path = fp[0]
        else:
            file_path = fp

        data, _ = _workflow().read_fits(file_path)
        return float(np.nanmin(data)), float(np.nanmax(data))
    except Exception as e:
        warnings.warn(f"Skipping file {fp}: {e}", stacklevel=2)
        return None


def compute_global_range(
    file_list: list, fixed_vmin=None, fixed_vmax=None, max_workers: int = 4
):
    """
    Traverse all files in parallel to compute global [vmin, vmax].
    When fixed_vmin / fixed_vmax are not None, return directly and skip statistics.

    【Optimization】Use ThreadPoolExecutor:
      - Reading FITS + nanmin/nanmax is a mix of I/O and NumPy computation,
        ThreadPool can parallelize I/O, NumPy releases GIL so computation can also be parallel,
        and it avoids process fork overhead, memory usage is lower.
    """
    if fixed_vmin is not None and fixed_vmax is not None:
        return fixed_vmin, fixed_vmax

    mins, maxs = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_global_range_one, fp): fp for fp in file_list}
        with tqdm(
            total=len(file_list), desc="Computing global color range", unit="files"
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    mins.append(result[0])
                    maxs.append(result[1])
                pbar.update(1)

    gmin = fixed_vmin if fixed_vmin is not None else float(np.nanmin(mins))
    gmax = fixed_vmax if fixed_vmax is not None else float(np.nanmax(maxs))
    print(f"Global color range: [{gmin:.3e}, {gmax:.3e}]")
    return gmin, gmax


def _calculate_range(data: np.ndarray, cfg: dict, is_global: bool = False) -> tuple:
    """
    计算数据范围。

    参数:
    ----------
    data : np.ndarray
        输入数据（通常是对数化后的数据）
    cfg : dict
        配置字典
    is_global : bool
        是否为全局范围计算

    返回:
    -------
    tuple : (vmin, vmax)
    """
    method = cfg.get("per_band_range_method", "fixed_percentile")
    min_log_range = cfg.get("min_log_range", 0.5)

    if len(data) == 0:
        return 0, 1

    if method == "percentile":
        # 使用5%和95%分位数
        low = np.percentile(data, 5)
        high = np.percentile(data, 95)

        # 如果范围太小，使用1%和99%分位数
        if high - low < min_log_range:
            low = np.percentile(data, 1)
            high = np.percentile(data, 99)

    elif method == "fixed_percentile":
        # 使用用户自定义的百分位数
        percentiles = cfg.get("per_band_percentiles", [66, 95])
        if len(percentiles) == 2:
            low = np.percentile(data, percentiles[0])
            high = np.percentile(data, percentiles[1])
        else:
            # 如果配置错误，使用默认值
            low = np.percentile(data, 5)
            high = np.percentile(data, 95)

        # 如果范围太小，给出警告但保持用户设置
        if high - low < min_log_range:
            warnings.warn(
                f"颜色范围过小 ({high - low:.3f})，考虑调整百分位数设置。", stacklevel=2
            )

    elif method == "minmax":
        # 使用最小最大值方法
        low = np.min(data)
        high = np.max(data)

        # 如果范围太小，给出警告
        if high - low < min_log_range:
            warnings.warn(
                f"颜色范围过小 ({high - low:.3f})，考虑使用百分位数方法。", stacklevel=2
            )
    else:
        # 默认使用5%和95%分位数
        low = np.percentile(data, 5)
        high = np.percentile(data, 95)

        if high - low < min_log_range:
            low = np.percentile(data, 1)
            high = np.percentile(data, 99)

    return low, high


def _calculate_per_band_ranges(all_log_data: list, cfg: dict) -> tuple:
    """
    为每个波段计算合适的对数数据范围。

    参数:
    ----------
    all_log_data : list
        所有波段的对数数据列表
    cfg : dict
        配置字典

    返回:
    -------
    tuple : (band_vmins, band_vmaxs)
    """
    band_vmins = []
    band_vmaxs = []

    for _idx, log_data in enumerate(all_log_data):
        valid_data = log_data[~np.isnan(log_data)]
        if len(valid_data) > 0:
            # 使用配置的方法计算范围
            vmin_band, vmax_band = _calculate_range(valid_data, cfg, is_global=False)
            band_vmins.append(vmin_band)
            band_vmaxs.append(vmax_band)
        else:
            # 如果没有有效数据，使用默认范围
            band_vmins.append(0)
            band_vmaxs.append(1)

    return band_vmins, band_vmaxs


def _resolve_multi_band_display_ranges(
    all_log_data: list[np.ndarray],
    cfg: dict,
    vmin=None,
    vmax=None,
) -> tuple[list[float], list[float]]:
    """Resolve the final log10 display range for every multi-band panel."""

    valid_parts = [data[np.isfinite(data)] for data in all_log_data]
    valid_parts = [data for data in valid_parts if data.size]
    if not valid_parts:
        return [0.0] * len(all_log_data), [1.0] * len(all_log_data)

    all_valid = np.concatenate(valid_parts)
    mode = str(cfg.get("color_range_mode", "auto") or "auto").lower()
    if mode == "fixed":
        raw_vmin = cfg.get("fixed_vmin")
        raw_vmax = cfg.get("fixed_vmax")
        if raw_vmin is None:
            raw_vmin = vmin
        if raw_vmax is None:
            raw_vmax = vmax
        low, high = _linear_limits_to_log10(raw_vmin, raw_vmax, all_valid)
        return [low] * len(all_log_data), [high] * len(all_log_data)

    if mode == "global":
        if vmin is not None and vmax is not None:
            low, high = _linear_limits_to_log10(vmin, vmax, all_valid)
        else:
            low, high = float(np.min(all_valid)), float(np.max(all_valid))
        return [low] * len(all_log_data), [high] * len(all_log_data)

    if mode != "auto":
        warnings.warn(
            f"Unknown multi-band color range mode {mode!r}; using auto.",
            stacklevel=2,
        )
    if (
        "fixed_band_vmins" in cfg
        and "fixed_band_vmaxs" in cfg
        and not background_enabled_for_display(cfg)
    ):
        band_vmins = [float(value) for value in cfg["fixed_band_vmins"]]
        band_vmaxs = [float(value) for value in cfg["fixed_band_vmaxs"]]
    else:
        band_vmins, band_vmaxs = _calculate_per_band_ranges(all_log_data, cfg)
    if cfg.get("use_per_band_colormap", True):
        return band_vmins, band_vmaxs

    low, high = _calculate_range(all_valid, cfg, is_global=True)
    return [float(low)] * len(all_log_data), [float(high)] * len(all_log_data)


def _linear_limits_to_log10(
    vmin, vmax, available_log_values: np.ndarray
) -> tuple[float, float]:
    if vmin is None or vmax is None:
        raise ValueError("A multi-band fixed range requires both vmin and vmax")
    raw_low = float(vmin)
    raw_high = float(vmax)
    if not np.isfinite(raw_low) or not np.isfinite(raw_high):
        raise ValueError("Multi-band color limits must be finite")
    if raw_low < 0 or raw_high <= 0 or raw_low >= raw_high:
        raise ValueError(
            "Multi-band color limits must satisfy 0 <= vmin < vmax and vmax > 0"
        )
    low = float(np.min(available_log_values)) if raw_low == 0 else math.log10(raw_low)
    high = math.log10(raw_high)
    if low >= high:
        raise ValueError("Multi-band color limits collapse after log10 conversion")
    return low, high


def _compute_fixed_band_ranges(cfg: dict) -> tuple:
    """
    为每个波段计算固定的颜色范围（基于该波段所有文件的数据）。

    参数:
    ----------
    cfg : dict
        配置字典

    返回:
    -------
    tuple : (band_vmins, band_vmaxs)
        每个波段的固定颜色范围
    """
    root = cfg["multi_band_root"]
    freqs = cfg["multi_band_freqs"]
    pattern = cfg["band_dir_pattern"]
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", None)

    combine_polarizations = cfg.get("combine_polarizations", False)
    polarization = cfg.get("polarization", "RR")

    band_vmins = []
    band_vmaxs = []

    print("开始计算每个波段的固定颜色范围...")

    for _freq_idx, freq in enumerate(tqdm(freqs, desc="计算波段颜色范围", unit="波段")):
        all_band_data = []

        if combine_polarizations and polarization == "RR+LL":
            # 读取RR和LL两个文件夹的所有文件
            rr_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"])
            )
            ll_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["ll_dir_suffix"])
            )

            # 获取两个文件夹的文件列表
            rr_files = _workflow()._sorted_fits_for_band(
                rr_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            ll_files = _workflow()._sorted_fits_for_band(
                ll_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )

            # ── 基于文件名时间戳做精确匹配 ─────────────────────────
            time_tolerance = cfg.get("time_tolerance_seconds", 1.0)
            tolerance_ms = time_tolerance * 1000
            rr_files = _filter_bad_radio_files(
                rr_files, freq, cfg["rr_dir_suffix"], cfg, drop_bad=True
            )
            ll_files = _filter_bad_radio_files(
                ll_files, freq, cfg["ll_dir_suffix"], cfg, drop_bad=True
            )
            matched_pairs = _match_rr_ll_by_time(rr_files, ll_files, tolerance_ms, cfg)

            if not matched_pairs:
                warnings.warn(
                    f"频率 {freq}MHz: RR和LL时间匹配失败，无有效数据", stacklevel=2
                )
                band_vmins.append(0)
                band_vmaxs.append(1)
                continue

            # 读取所有文件的数据
            for rr_path, ll_path in matched_pairs:
                try:
                    rr_data, rr_header = _workflow().read_fits(rr_path)
                    ll_data, ll_header = _workflow().read_fits(ll_path)

                    # 组合数据（加权平均或简单相加）
                    combined_data = _combine_polarization_data(rr_data, ll_data, cfg)

                    # 对数化处理
                    mask = combined_data > 0
                    log_data = np.full_like(combined_data, np.nan, dtype=np.float64)
                    log_data[mask] = np.log10(combined_data[mask])

                    # 收集有效数据
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 0:
                        all_band_data.extend(valid_data)

                except Exception as e:
                    warnings.warn(
                        f"读取文件时出错（频率 {freq}MHz）: {e}", stacklevel=2
                    )
                    continue
        else:
            # 普通模式：只读取指定偏振的文件
            band_dir = os.path.join(root, pattern.format(freq=freq, polar=polarization))
            files = _workflow()._sorted_fits_for_band(
                band_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            files = _filter_bad_radio_files(
                files, freq, polarization, cfg, drop_bad=True
            )

            # 读取所有文件的数据
            for file_path in files:
                try:
                    img_data, header = _workflow().read_fits(file_path)

                    # 对数化处理
                    mask = img_data > 0
                    log_data = np.full_like(img_data, np.nan, dtype=np.float64)
                    log_data[mask] = np.log10(img_data[mask])

                    # 收集有效数据
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 0:
                        all_band_data.extend(valid_data)

                except Exception as e:
                    warnings.warn(
                        f"读取文件时出错（频率 {freq}MHz）: {e}", stacklevel=2
                    )
                    continue

        # 计算该波段的固定颜色范围
        if len(all_band_data) > 0:
            all_band_data_array = np.array(all_band_data)
            vmin_band, vmax_band = _calculate_range(
                all_band_data_array, cfg, is_global=False
            )
            band_vmins.append(vmin_band)
            band_vmaxs.append(vmax_band)

            print(
                f"频率 {freq}MHz: 固定颜色范围 = [{vmin_band:.3f}, {vmax_band:.3f}] (基于{len(all_band_data)}个数据点)"
            )
        else:
            # 如果没有有效数据，使用默认范围
            band_vmins.append(0)
            band_vmaxs.append(1)
            print(f"频率 {freq}MHz: 警告 - 没有有效数据，使用默认范围")

    print("每个波段的固定颜色范围计算完成！")
    return band_vmins, band_vmaxs
