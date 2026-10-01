"""Input discovery, time parsing, and radio/AIA/HMI pairing.

Functions retain the historical core dependency hooks; scientific operations
and product ordering are unchanged.
"""

# ruff: noqa: F401, I001

from ._overlay_context import (
    Config,
    _DATETIME_FMTS,
    _RE_AIA_NEW_PAT,
    _RE_AIA_PATS,
    _RE_HMI_NEW_PAT,
    _RE_HMI_PAT,
    _RE_RADIO_PAT_YYYYJJJ,
    _RE_RADIO_PAT_YYYYMMDD,
    _nearest_radio_entry_index,
    _parse_millisecond_suffix,
    _slice_file_list,
    datetime,
    fits,
    glob,
    os,
    re,
    timedelta,
    warnings,
)


def _workflow():
    """Resolve historical core replacement hooks only when a function runs."""

    from . import _overlay_workflow_core

    return _overlay_workflow_core


def _parse_flexible_datetime(date_str: str) -> datetime | None:
    """灵活解析各种时间字符串格式，返回 datetime 或 None"""
    date_str = date_str.strip()

    # ── 17 位纯数字格式（YYYYMMDDHHMMSSmmm）─────────────────
    if len(date_str) == 17 and date_str.isdigit():
        try:
            return datetime(
                int(date_str[0:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
                int(date_str[8:10]),
                int(date_str[10:12]),
                int(date_str[12:14]),
                int(date_str[14:17]) * 1000,
            )
        except Exception:
            pass

    # ── 下划线分隔格式（YYYYJJJ_HHMMSS_SSS 或 YYYYmDD_HHMMSS）──
    if "_" in date_str:
        parts = date_str.split("_")
        if len(parts) >= 2:
            date_part, time_part = parts[0], parts[1]
            if len(date_part) in {6, 7}:
                year = int(date_part[:4])
                parsed_date = None
                if len(date_part) == 6:
                    try:
                        parsed_date = datetime(
                            year, int(date_part[4:5]), int(date_part[5:6])
                        )
                    except ValueError:
                        pass
                try:
                    if parsed_date is None and len(date_part) == 7:
                        parsed_date = datetime(
                            year, int(date_part[4:5]), int(date_part[5:7])
                        )
                except ValueError:
                    pass
                if parsed_date is None:
                    try:
                        parsed_date = datetime(year, 1, 1) + timedelta(
                            days=int(date_part[4:]) - 1
                        )
                    except Exception:
                        pass
                if parsed_date is not None and len(time_part) == 6:
                    microsecond = 0
                    if len(parts) > 2 and parts[2]:
                        microsecond = (
                            _workflow()._parse_millisecond_suffix(parts[2]) * 1000
                        )
                    return datetime(
                        parsed_date.year,
                        parsed_date.month,
                        parsed_date.day,
                        int(time_part[0:2]),
                        int(time_part[2:4]),
                        int(time_part[4:6]),
                        microsecond,
                    )

    # ── 规范化小数部分 ────────────────────────────────────────
    if "." in date_str:
        integer_part, decimal_part = date_str.split(".", 1)
        date_str = f"{integer_part}.{decimal_part.ljust(6, '0')[:6]}"

    # ── 逐一尝试标准格式 ──────────────────────────────────────
    for fmt in _DATETIME_FMTS:
        try:
            s = date_str
            if ".%f" in fmt and "." not in s:
                s = s + ".0"
            if "." in s and ".%f" in fmt:
                ip, dp = s.split(".", 1)
                s = f"{ip}.{dp.ljust(6, '0')[:6]}"
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # ── 14+ 位纯数字兜底 ──────────────────────────────────────
    if len(date_str) >= 14 and date_str[:14].isdigit():
        try:
            microsecond = 0
            remaining = date_str[14:]
            if remaining.startswith("."):
                microsecond = int(remaining[1:].ljust(6, "0")[:6])
            elif remaining.isdigit():
                if len(remaining) == 3:
                    microsecond = int(remaining) * 1000
                else:
                    microsecond = int(remaining.ljust(6, "0")[:6])
            return datetime(
                int(date_str[0:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
                int(date_str[8:10]),
                int(date_str[10:12]),
                int(date_str[12:14]),
                microsecond,
            )
        except Exception:
            pass

    return None


def parse_radio_time_from_filename(filename: str) -> datetime | None:
    """从射电文件名提取观测时间"""
    basename = os.path.basename(filename)

    m = _RE_RADIO_PAT_YYYYJJJ.search(basename)
    if m:
        t = _workflow()._parse_flexible_datetime(
            f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        )
        if t:
            return t

    m = _RE_RADIO_PAT_YYYYMMDD.search(basename)
    if m:
        t = _workflow()._parse_flexible_datetime(f"{m.group(1)}_{m.group(2)}")
        if t:
            return t

    try:
        if os.path.exists(filename):
            date_obs = str(
                _workflow().fits.getheader(filename, 0).get("DATE-OBS", "")
            ).strip()
            if date_obs:
                return _workflow()._parse_flexible_datetime(date_obs)
    except Exception:
        pass

    return None


def parse_aia_time_from_filename(filename: str) -> datetime | None:
    """从 AIA 文件名或头文件提取观测时间"""
    basename = os.path.basename(filename)

    try:
        if os.path.exists(filename):
            date_obs = str(
                _workflow().fits.getheader(filename, 0).get("DATE-OBS", "")
            ).strip()
            if date_obs:
                t = _workflow()._parse_flexible_datetime(date_obs)
                if t:
                    return t
    except Exception:
        pass

    m = _RE_AIA_NEW_PAT.search(basename)
    if m:
        t = _workflow()._parse_flexible_datetime(m.group(1).rstrip("Z"))
        if t:
            return t

    for pat in _RE_AIA_PATS:
        m = pat.search(basename)
        if m:
            t = _workflow()._parse_flexible_datetime(m.group(1).rstrip("Z"))
            if t:
                return t

    for digits in re.findall(r"\d{4,}", basename):
        if len(digits) >= 8:
            t = _workflow()._parse_flexible_datetime(digits)
            if t:
                return t

    return None


def parse_hmi_time_from_filename(filename: str) -> datetime | None:
    """从 HMI 文件名提取观测时间"""
    basename = os.path.basename(filename)

    m = _RE_HMI_NEW_PAT.search(basename)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    m = _RE_HMI_PAT.search(basename)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    return None


def _parse_time_from_filename(filename: str) -> tuple[str, int] | None:
    """
    从文件名中解析时间信息（精确到毫秒），用于时间对齐匹配。

    文件名格式: 149MHz_2025124_043739_681.fits
      - 日期部分:   2025124  (YYYYDDD，7~8位)
      - 时间部分:   043739   (HHMMSS，6位)
      - 毫秒部分:   681      (1~3位，不足3位按实际值处理)

    返回: (date_str, total_ms) 或 None
      - date_str  : 日期字符串，用于跨天判断
      - total_ms  : 当天从0点起的毫秒数，用于数值比较
    """
    # 匹配: _日期(7-8位)_时间(6位)_毫秒(1-3位)
    pattern = r"_(\d{6,8})_(\d{6})_(\d{1,3})"
    match = re.search(pattern, filename)
    if match:
        date_part = match.group(1)  # e.g. "2025124"
        time_part = match.group(2)  # e.g. "043739"
        ms_str = match.group(3)  # e.g. "681"

        hh = int(time_part[0:2])
        mm = int(time_part[2:4])
        ss = int(time_part[4:6])
        # Parse suffix as integer milliseconds: _13 means 13 ms, not 130 ms.
        ms = _workflow()._parse_millisecond_suffix(ms_str)

        total_ms = (hh * 3600 + mm * 60 + ss) * 1000 + ms
        return (date_part, total_ms)

    # 降级：仅匹配 _日期_时间（无毫秒字段）
    pattern_no_ms = r"_(\d{6,8})_(\d{6})"
    match2 = re.search(pattern_no_ms, filename)
    if match2:
        date_part = match2.group(1)
        time_part = match2.group(2)
        hh = int(time_part[0:2])
        mm = int(time_part[2:4])
        ss = int(time_part[4:6])
        total_ms = (hh * 3600 + mm * 60 + ss) * 1000
        return (date_part, total_ms)

    return None


def _match_rr_ll_by_time(
    rr_files: list, ll_files: list, tolerance_ms: float = 10.0
) -> list[tuple[str, str]]:
    """
    根据文件名时间戳将RR与LL文件逐一配对（毫秒级精度）。

    算法:
      1. 解析所有LL文件的时间戳，建立 {(date, ms): path} 索引。
      2. 遍历每个RR文件，先精确匹配，再在容差范围内找最近邻。
      3. 返回已匹配的 [(rr_path, ll_path), ...] 列表，并报告未匹配数量。

    Parameters
    ----------
    rr_files      : RR文件路径列表（已排序）
    ll_files      : LL文件路径列表（已排序）
    tolerance_ms  : 时间匹配容差（毫秒），默认10ms

    Returns
    -------
    matched_pairs : list of (rr_path, ll_path)
    """
    # 构建LL时间索引: {(date, total_ms): ll_path}
    ll_index: dict[tuple[str, int], str] = {}
    ll_no_parse: list[str] = []
    for ll_path in ll_files:
        parsed = _workflow()._parse_time_from_filename(os.path.basename(ll_path))
        if parsed is None:
            ll_no_parse.append(ll_path)
        else:
            key = parsed  # (date_str, total_ms)
            if key not in ll_index:
                ll_index[key] = ll_path

    if ll_no_parse:
        warnings.warn(
            f"有 {len(ll_no_parse)} 个LL文件无法从文件名解析时间，将被跳过。",
            stacklevel=2,
        )

    matched_pairs: list[tuple[str, str]] = []
    unmatched_rr: list[str] = []

    # 将LL索引按日期分组，加速搜索
    from collections import defaultdict

    ll_by_date: dict[str, list[tuple[int, str]]] = defaultdict(
        list
    )  # {date_str: [(total_ms, ll_path), ...]}
    for (date_str, total_ms), ll_path in ll_index.items():
        ll_by_date[date_str].append((total_ms, ll_path))
    # 每个日期内按ms排序，以便未来二分查找（当前数据量不大，线性也可）
    for date_str in ll_by_date:
        ll_by_date[date_str].sort(key=lambda x: x[0])

    for rr_path in rr_files:
        parsed = _workflow()._parse_time_from_filename(os.path.basename(rr_path))
        if parsed is None:
            unmatched_rr.append(rr_path)
            warnings.warn(
                f"RR文件 {os.path.basename(rr_path)} 无法解析时间，跳过。", stacklevel=2
            )
            continue

        rr_date, rr_ms = parsed

        # ① 精确匹配
        if (rr_date, rr_ms) in ll_index:
            matched_pairs.append((rr_path, ll_index[(rr_date, rr_ms)]))
            continue

        # ② 容差范围内最近邻匹配
        candidates = ll_by_date.get(rr_date, [])
        best_ll_path = None
        best_diff = float("inf")
        for ll_ms, ll_path in candidates:
            diff = abs(rr_ms - ll_ms)
            if diff < best_diff:
                best_diff = diff
                best_ll_path = ll_path

        if best_ll_path is not None and best_diff <= tolerance_ms:
            matched_pairs.append((rr_path, best_ll_path))
        else:
            unmatched_rr.append(rr_path)
            if best_diff != float("inf"):
                warnings.warn(
                    f"RR文件 {os.path.basename(rr_path)} 找不到时间匹配的LL文件 "
                    f"(最近差值={best_diff:.1f}ms > 容差={tolerance_ms:.1f}ms)，跳过。",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"RR文件 {os.path.basename(rr_path)} 在LL目录中找不到同日期文件，跳过。",
                    stacklevel=2,
                )

    if unmatched_rr:
        print(
            f"  时间匹配结果: 成功 {len(matched_pairs)} 对，"
            f"RR未匹配 {len(unmatched_rr)} 个。"
        )
    else:
        print(f"  时间匹配结果: 全部 {len(matched_pairs)} 对成功匹配。")

    return matched_pairs


def _multi_wave_overlay_enabled(cfg: Config) -> bool:
    wavelengths = getattr(cfg, "aia_panel_wavelengths", None)
    return bool(wavelengths and len(wavelengths) > 1)


def _fits_files_in_dir(path: str, start_idx=None, end_idx=None) -> list[str]:
    files = sorted(glob.glob(os.path.join(path, "*.fits")))
    return _workflow()._slice_file_list(files, start_idx, end_idx)


def _radio_time_key_from_item(item) -> tuple[str, int] | None:
    path = item[0] if isinstance(item, tuple) else item
    return _workflow()._parse_time_from_filename(os.path.basename(path))


def _build_common_radio_slots_from_entries(
    per_band_entries: list[list[tuple[tuple[str, int], object]]],
    tolerance_ms: float,
) -> list[list[object]]:
    if not per_band_entries or any(not entries for entries in per_band_entries):
        return []
    reference_index = min(
        range(len(per_band_entries)), key=lambda index: len(per_band_entries[index])
    )
    reference_entries = sorted(
        per_band_entries[reference_index], key=lambda entry: (entry[0][0], entry[0][1])
    )
    used_by_band = [set() for _entries in per_band_entries]
    slots = []
    for ref_key, _ref_item in reference_entries:
        slot = []
        matched_indices = []
        for band_index, entries in enumerate(per_band_entries):
            match_index = _workflow()._nearest_radio_entry_index(
                entries, ref_key, used_by_band[band_index], tolerance_ms
            )
            if match_index is None:
                slot = []
                matched_indices = []
                break
            matched_indices.append(match_index)
            slot.append(entries[match_index][1])
        if not slot:
            continue
        slot_times = [
            per_band_entries[band_index][match_index][0][1]
            for band_index, match_index in enumerate(matched_indices)
        ]
        if max(slot_times) - min(slot_times) > tolerance_ms:
            continue
        for band_index, match_index in enumerate(matched_indices):
            used_by_band[band_index].add(match_index)
        slots.append(slot)
    return slots


def build_radio_time_slots_for_overlay(cfg: Config) -> list[tuple[int, dict]]:
    """Build radio-first multi-frequency slots for raw AIA overlay frames."""
    per_band_entries = []
    tolerance_ms = float(getattr(cfg, "multi_band_time_tolerance_seconds", 0.1)) * 1000
    for band in cfg.selected_bands:
        band_dir = os.path.join(cfg.radio_base_dir, band)
        if cfg.combine_polarizations and cfg.polarization_mode == "RR+LL":
            rr_dir = os.path.join(band_dir, cfg.rr_dir_suffix)
            ll_dir = os.path.join(band_dir, cfg.ll_dir_suffix)
            rr_files = _workflow()._fits_files_in_dir(
                rr_dir, cfg.radio_start_idx, cfg.radio_end_idx
            )
            ll_files = _workflow()._fits_files_in_dir(
                ll_dir, cfg.radio_start_idx, cfg.radio_end_idx
            )
            matched_items = _workflow()._match_rr_ll_by_time(
                rr_files, ll_files, cfg.time_tolerance_seconds * 1000
            )
            polarization = "RR+LL"
        else:
            pol_dir = os.path.join(band_dir, cfg.polarization_mode)
            search_dir = pol_dir if os.path.isdir(pol_dir) else band_dir
            matched_items = _workflow()._fits_files_in_dir(
                search_dir, cfg.radio_start_idx, cfg.radio_end_idx
            )
            polarization = cfg.polarization_mode

        entries = []
        for item in matched_items:
            key = _workflow()._radio_time_key_from_item(item)
            path_for_time = item[0] if isinstance(item, tuple) else item
            radio_time = _workflow().parse_radio_time_from_filename(path_for_time)
            if key is None or radio_time is None:
                continue
            entries.append((key, (band, item, polarization, radio_time)))
        per_band_entries.append(entries)

    common_slots = _workflow()._build_common_radio_slots_from_entries(
        per_band_entries, tolerance_ms
    )
    output_slots = []
    for slot_index, slot_items in enumerate(common_slots):
        single_slice_bands = {}
        for band, file_item, polarization, radio_time in slot_items:
            single_slice_bands[band] = [(file_item, polarization, radio_time)]
        output_slots.append((slot_index, single_slice_bands))
    return output_slots


def _aia_panel_wavelengths(cfg: Config) -> list[int]:
    return [int(wave) for wave in (cfg.aia_panel_wavelengths or [])]


def _aia_panel_dir(cfg: Config, wavelength: int) -> str:
    template = cfg.aia_panel_base_dir_template
    if template:
        return str(template).format(wave=wavelength, wavelength=wavelength)
    if str(cfg.aia_wavelength) == str(wavelength):
        return cfg.aia_base_dir
    return os.path.join(os.path.dirname(cfg.aia_base_dir), str(wavelength))


def _nearest_file_by_time(
    files: list[str],
    parser,
    target_time: datetime,
    threshold_seconds: float,
) -> str | None:
    candidates = []
    for path in files:
        parsed_time = parser(path)
        if parsed_time is None:
            parsed_time = parser(os.path.basename(path))
        if parsed_time is None:
            continue
        dt = abs((parsed_time - target_time).total_seconds())
        if dt <= threshold_seconds:
            candidates.append((dt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _slot_reference_time(single_slice_bands: dict) -> datetime | None:
    times = []
    for file_list in single_slice_bands.values():
        for _file_item, _polarization, radio_time in file_list:
            if radio_time is not None:
                times.append(radio_time)
    if not times:
        return None
    return min(times) + (max(times) - min(times)) / 2


def build_multi_wave_matched_pairs(
    cfg: Config,
) -> list[tuple[dict[int, str], str | None, list]]:
    """Match radio-first slots to the nearest six-wave AIA files and HMI file."""
    wavelengths = _workflow()._aia_panel_wavelengths(cfg)
    if not wavelengths:
        return []

    aia_files_by_wave = {
        wave: _workflow()._fits_files_in_dir(
            _workflow()._aia_panel_dir(cfg, wave),
            cfg.aia_file_start_idx,
            cfg.aia_file_end_idx,
        )
        for wave in wavelengths
    }
    if any(not files for files in aia_files_by_wave.values()):
        missing = [wave for wave, files in aia_files_by_wave.items() if not files]
        raise FileNotFoundError(f"Missing AIA FITS files for wavelengths: {missing}")

    hmi_files = (
        sorted(glob.glob(os.path.join(cfg.hmi_base_dir, "*.fits")))
        if cfg.overlay_hmi
        else []
    )
    radio_slots = _workflow().build_radio_time_slots_for_overlay(cfg)
    matched_pairs = []
    for slot_index, single_slice_bands in radio_slots:
        reference_time = _workflow()._slot_reference_time(single_slice_bands)
        if reference_time is None:
            continue
        matched_aia = {}
        for wave, files in aia_files_by_wave.items():
            aia_file = _workflow()._nearest_file_by_time(
                files,
                _workflow().parse_aia_time_from_filename,
                reference_time,
                float(cfg.aia_time_threshold_seconds),
            )
            if aia_file is None:
                matched_aia = {}
                break
            matched_aia[wave] = aia_file
        if not matched_aia:
            continue
        hmi_file = None
        if hmi_files:
            hmi_file = _workflow()._nearest_file_by_time(
                hmi_files,
                _workflow().parse_hmi_time_from_filename,
                reference_time,
                float(cfg.hmi_time_threshold) * 3600.0,
            )
        matched_pairs.append(
            (matched_aia, hmi_file, [(slot_index, single_slice_bands)])
        )
    return matched_pairs


def build_matched_pairs(cfg: Config) -> list[tuple[str, str | None, list]]:
    """
    构建任务列表：将 AIA 文件前后指定时间内的所有射电数据
    按时间顺序切分为一个个独立的“切片”（slice），用于生成序列帧。
    """
    aia_files = sorted(glob.glob(os.path.join(cfg.aia_base_dir, "*.fits")))
    if not aia_files:
        raise FileNotFoundError(f"在 {cfg.aia_base_dir} 中未找到 AIA fits 文件")

    start = cfg.aia_file_start_idx if cfg.aia_file_start_idx is not None else 0
    end = cfg.aia_file_end_idx if cfg.aia_file_end_idx is not None else len(aia_files)
    aia_files = aia_files[start:end]

    hmi_files = (
        sorted(glob.glob(os.path.join(cfg.hmi_base_dir, "*.fits")))
        if cfg.overlay_hmi
        else []
    )

    # 提前缓存并解析所有射电文件的时间
    radio_cache = []
    for band in cfg.selected_bands:
        band_dir = os.path.join(cfg.radio_base_dir, band)
        rr_dir = (
            os.path.join(band_dir, cfg.rr_dir_suffix)
            if cfg.combine_polarizations
            else band_dir
        )
        ll_dir = (
            os.path.join(band_dir, cfg.ll_dir_suffix)
            if cfg.combine_polarizations
            else None
        )

        rr_files = (
            sorted(glob.glob(os.path.join(rr_dir, "*.fits")))
            if os.path.isdir(rr_dir)
            else []
        )
        ll_files = (
            sorted(glob.glob(os.path.join(ll_dir, "*.fits")))
            if ll_dir and os.path.isdir(ll_dir)
            else []
        )

        if cfg.combine_polarizations and rr_files and ll_files:
            pairs = _workflow()._match_rr_ll_by_time(
                rr_files, ll_files, cfg.time_tolerance_seconds * 1000
            )
            for rr_path, ll_path in pairs:
                t = _workflow().parse_radio_time_from_filename(rr_path)
                if t:
                    radio_cache.append(
                        {
                            "path": (rr_path, ll_path),
                            "band": band,
                            "pol": "RR+LL",
                            "time": t,
                        }
                    )
        else:
            files = (
                rr_files
                if rr_files
                else (
                    ll_files
                    if ll_files
                    else glob.glob(os.path.join(band_dir, "*.fits"))
                )
            )
            for rf in files:
                t = _workflow().parse_radio_time_from_filename(rf)
                if t:
                    radio_cache.append(
                        {
                            "path": rf,
                            "band": band,
                            "pol": cfg.polarization_mode,
                            "time": t,
                        }
                    )

    matched_pairs = []
    for aia_file in aia_files:
        aia_time = _workflow().parse_aia_time_from_filename(os.path.basename(aia_file))
        if not aia_time:
            continue

        # 匹配最近的 HMI
        best_hmi = None
        if cfg.overlay_hmi and hmi_files:
            hmi_diffs = []
            for hf in hmi_files:
                ht = _workflow().parse_hmi_time_from_filename(hf)
                if ht:
                    hmi_diffs.append((hf, abs((ht - aia_time).total_seconds())))
            valid_hmis = [x for x in hmi_diffs if x[1] <= cfg.hmi_time_threshold * 3600]
            if valid_hmis:
                best_hmi = min(valid_hmis, key=lambda x: x[1])[0]

        # 归类落在这个 AIA 时间窗口内的所有射电帧
        band_groups = {}
        for rc in radio_cache:
            dt = abs((rc["time"] - aia_time).total_seconds())
            if dt <= cfg.radio_time_threshold:
                band_groups.setdefault(rc["band"], []).append(
                    (rc["path"], rc["pol"], rc["time"], dt)
                )

        if not band_groups:
            continue

        for band in band_groups:
            band_groups[band].sort(key=lambda x: x[2])  # 严格按照时间排序
            band_groups[band] = band_groups[band][: cfg.max_radio_per_band]

        min_count = min(len(v) for v in band_groups.values())
        if min_count == 0:
            continue

        # 横向构建切片，生成序列子任务 (切片结构：{ 频段: [(文件, 偏振, 时间)] })
        tasks_for_aia = []
        for idx in range(min_count):
            slc = {
                band: [band_groups[band][idx][:3]]
                for band in band_groups
                if idx < len(band_groups[band])
            }
            if slc:
                tasks_for_aia.append((idx, slc))

        matched_pairs.append((aia_file, best_hmi, tasks_for_aia))

    return matched_pairs
