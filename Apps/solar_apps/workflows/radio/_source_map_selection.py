"""Source-map input discovery, frozen collections, quality flags and time slots."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

from ._source_map_helpers import (
    _require_study_mode,
    _raw_quality_filter_enabled,
    _nearest_time_entry_index,
    _build_slots_by_position,
    _candidate_slot_index,
    get_time_from_header,
)


def _workflow():
    # Resolve retained monkeypatch anchors only after the facade has loaded.
    from . import source_map_workflow

    return source_map_workflow


_RAW_QUALITY_BAD_REASONS_KEY = "_raw_quality_bad_file_reasons"
_RAW_QUALITY_FILE_FLAGS_KEY = "_raw_quality_file_quality_flags"


def get_sorted_fits(
    data_dir: str,
    start: int,
    end: int | None,
    *,
    study_mode: str | None = None,
) -> list:
    """Return sorted list of FITS file paths within the specified range."""
    mode = _require_study_mode(study_mode)
    frozen = _frozen_files_for_band(Path(data_dir), required=mode == "confirmatory")
    if frozen is not None:
        return [str(path) for path in frozen]
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".fits")
    )
    if not all_files:
        raise FileNotFoundError(f"目录 {data_dir} 中未找到 FITS 文件")
    selected = all_files[start:end]
    if not selected:
        raise ValueError(f"索引范围 [{start}, {end}) 内没有文件，请检查参数")
    return selected


def _sorted_fits_for_band(
    band_dir: str,
    start_idx: int,
    end_idx,
    *,
    study_mode: str | None = None,
) -> list:
    """Get sorted list of FITS files in the specified band directory"""
    if not os.path.isdir(band_dir):
        raise ValueError(f"波段目录不存在：{band_dir}")

    mode = _require_study_mode(study_mode)
    frozen = _frozen_files_for_band(Path(band_dir), required=mode == "confirmatory")
    if frozen is not None:
        return [str(path) for path in frozen]

    all_files = sorted(
        os.path.join(band_dir, f)
        for f in os.listdir(band_dir)
        if f.lower().endswith(".fits")
    )
    if not all_files:
        raise ValueError(f"波段目录 {band_dir} 中未找到 FITS 文件")

    total = len(all_files)
    end = total if end_idx is None else min(end_idx, total)
    selected = all_files[start_idx:end]
    if not selected:
        raise ValueError(f"索引范围 [{start_idx}, {end}) 内没有文件")
    return selected


def _parse_utc_z(value: object, *, label: str) -> datetime.datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value)
        is None
    ):
        raise ValueError(f"Frozen collection {label} must be UTC with a Z suffix")
    try:
        parsed = datetime.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"Frozen collection {label} is invalid") from exc
    if parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"Frozen collection {label} must be UTC")
    return parsed


def _utc_z_millis(value: datetime.datetime) -> str:
    return (
        value.astimezone(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _expected_frozen_record_id(observed: datetime.datetime, relative_path: str) -> str:
    identity = f"{_utc_z_millis(observed)}\0{relative_path}"
    return "radio-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _frozen_files_for_band(
    band_dir: Path, *, required: bool = False
) -> list[Path] | None:
    """Return and verify an explicit collection, bypassing positional slices."""

    manifest = next(
        (
            parent / ".frozen-collection-v1.json"
            for parent in (band_dir, *band_dir.parents)
            if (parent / ".frozen-collection-v1.json").is_file()
        ),
        None,
    )
    if manifest is None:
        if required:
            raise FileNotFoundError(
                f"Confirmatory radio selection requires .frozen-collection-v1.json: "
                f"{band_dir}"
            )
        return None
    root = manifest.parent.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen collection must be a JSON object")
    if payload.get("schema") != "solar-radio-frozen-collection-v1":
        raise ValueError(f"Unsupported frozen collection: {manifest}")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Frozen collection selection must be an object")
    start = _parse_utc_z(selection.get("start_utc"), label="selection.start_utc")
    end = _parse_utc_z(selection.get("end_utc"), label="selection.end_utc")
    if end <= start:
        raise ValueError("Frozen collection selection must satisfy end_utc > start_utc")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen collection records must be a non-empty list")
    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(records)
    ):
        raise ValueError("Frozen collection record_count mismatch")
    resolved_band = band_dir.resolve()
    selected: list[tuple[datetime.datetime, str, Path]] = []
    record_ids: set[str] = set()
    paths: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Frozen collection record must be an object")
        record_id = item.get("record_id")
        if not isinstance(record_id, str) or not record_id.startswith("radio-"):
            raise ValueError("Frozen collection record_id is invalid")
        if record_id in record_ids:
            raise ValueError(f"Frozen collection duplicate record_id: {record_id}")
        record_ids.add(record_id)
        observed = _parse_utc_z(item.get("observed_utc"), label="record.observed_utc")
        if not start <= observed < end:
            raise ValueError(f"Frozen record outside selection interval: {record_id}")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Frozen record path is invalid: {record_id}")
        if relative_path in paths:
            raise ValueError(f"Frozen collection duplicate path: {relative_path}")
        paths.add(relative_path)
        expected_record_id = _expected_frozen_record_id(observed, relative_path)
        if record_id != expected_record_id:
            raise ValueError(f"Frozen record_id is not bound to UTC/path: {record_id}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Frozen collection path escaped its root") from exc
        size = item.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Frozen record bytes is invalid: {record_id}")
        digest_value = item.get("sha256")
        if (
            not isinstance(digest_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
        ):
            raise ValueError(f"Frozen record SHA is invalid: {record_id}")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Frozen collection file missing/size mismatch: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != digest_value:
            raise ValueError(f"Frozen collection SHA mismatch: {path}")
        if path.parent == resolved_band:
            selected.append((observed, record_id, path))
    if not selected:
        raise ValueError(f"Frozen collection has no records for {band_dir}")
    return [path for _observed, _record_id, path in sorted(selected)]


def _raw_quality_path_key(path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _iter_raw_quality_item_paths(item) -> list[str]:
    if isinstance(item, (tuple, list)):
        paths: list[str] = []
        for part in item:
            paths.extend(_iter_raw_quality_item_paths(part))
        return paths
    text = os.fspath(item)
    if "|" in text:
        return [part for part in text.split("|") if part]
    return [text]


def _remember_raw_quality_rows(cfg: dict, rows: list) -> None:
    bad_reasons = dict(cfg.get(_RAW_QUALITY_BAD_REASONS_KEY, {}) or {})
    file_flags = dict(cfg.get(_RAW_QUALITY_FILE_FLAGS_KEY, {}) or {})
    for row in rows:
        raw_path = str(row.source_file)
        path_key = _raw_quality_path_key(raw_path)
        file_flags[path_key] = str(row.quality_flag)
        if row.quality_flag == "bad":
            bad_reasons[path_key] = str(row.reason)
        else:
            bad_reasons.pop(path_key, None)
    cfg[_RAW_QUALITY_BAD_REASONS_KEY] = bad_reasons
    cfg[_RAW_QUALITY_FILE_FLAGS_KEY] = file_flags


def _raw_quality_bad_reasons_for_item(item, cfg: dict) -> list[str]:
    bad_reasons = cfg.get(_RAW_QUALITY_BAD_REASONS_KEY, {}) or {}
    reasons: list[str] = []
    seen = set()
    for path in _iter_raw_quality_item_paths(item):
        for key in (path, _raw_quality_path_key(path)):
            reason = bad_reasons.get(key)
            if reason and key not in seen:
                reasons.append(str(reason))
                seen.add(key)
    return reasons


def _raw_quality_item_is_bad(item, cfg: dict) -> bool:
    return bool(_raw_quality_bad_reasons_for_item(item, cfg))


def _raw_quality_bad_frame_output_dir(output_dir: str, cfg: dict, *parts) -> Path:
    bad_subdir = str(
        cfg.get("raw_quality_bad_frame_output_subdir", "raw_quality_bad_frames")
        or "raw_quality_bad_frames"
    ).strip()
    if not bad_subdir:
        bad_subdir = "raw_quality_bad_frames"
    return (
        Path(output_dir)
        / _workflow()._plot_output_subdir(cfg)
        / bad_subdir
        / Path(*parts)
    )


def _filter_bad_radio_files(
    files: list, freq, polarization: str, cfg: dict, *, drop_bad: bool = False
) -> list:
    """Classify raw FITS quality, optionally dropping bad files for statistics only."""
    if not _raw_quality_filter_enabled(cfg):
        return files
    if not files:
        return files

    from solar_toolkit.radio.raw_quality import filter_bad_radio_fits_files

    result = filter_bad_radio_fits_files(
        files,
        frequency_mhz=float(freq),
        polarization=str(polarization),
    )
    _remember_raw_quality_rows(cfg, result.file_rows)
    rejected = result.rejected_rows
    if rejected:
        print(
            f"  Raw-quality filter {freq}MHz/{polarization}: "
            f"flagged {len(rejected)}/{len(files)} bad"
        )
        for row in rejected[:5]:
            print(f"    reject {os.path.basename(row.source_file)}: {row.reason}")
        if len(rejected) > 5:
            print(f"    ... {len(rejected) - 5} more rejected files")
    if drop_bad:
        return result.accepted_files
    return list(files)


class TimeParser:
    """时间解析器，支持多种日期格式"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.date_format = cfg.get("date_format", "auto")
        self.fallback = cfg.get("time_parsing_fallback", True)

    def parse_date_part(self, date_str):
        """解析日期字符串，返回(年份, 天数)"""
        if len(date_str) == 6:
            # 格式: YYDDD (6位)
            year = int(date_str[0:2])
            # 假设20xx年
            full_year = 2000 + year if year < 100 else year
            day_of_year = int(date_str[2:])
            return full_year, day_of_year
        elif len(date_str) == 7:
            # 格式: YYYYDDD (7位)
            year = int(date_str[0:4])
            day_of_year = int(date_str[4:])
            return year, day_of_year
        elif len(date_str) == 8:
            # 格式: YYYYDDDD (8位，不常见）
            year = int(date_str[0:4])
            day_of_year = int(date_str[4:])
            return year, day_of_year
        else:
            raise ValueError(f"不支持的日期格式长度: {len(date_str)}位")

    def parse_time_from_filename(self, filename):
        """从文件名解析时间信息（精确到毫秒），支持多种格式。

        支持的格式:
        1. 6位日期+毫秒: 149MHz_000001_000000_000.fits
        2. 7位日期+毫秒: 149MHz_2000001_000000_000.fits
        3. 6位日期无毫秒: 149MHz_000001_000000.fits
        4. 7位日期无毫秒: 149MHz_2000001_000000.fits

        返回: (date_key, total_ms) 或 None
          - date_key : 用于跨天比较的日期键
          - total_ms : 当天从0点起的毫秒数
        """
        import re

        # 从配置获取正则表达式模式
        patterns = self.cfg.get("filename_patterns", {})
        pattern_with_ms = patterns.get("with_ms", r"_(\d{6,8})_(\d{6})_(\d{1,3})")
        pattern_without_ms = patterns.get("without_ms", r"_(\d{6,8})_(\d{6})")

        # 尝试带毫秒的模式
        match = re.search(pattern_with_ms, filename)
        if match:
            date_part = match.group(1)  # 如 "000001" 或 "2000001"
            time_part = match.group(2)  # 如 "000000"
            ms_str = match.group(3)  # 如 "000"

            # 解析时间部分
            hh = int(time_part[0:2])
            mm = int(time_part[2:4])
            ss = int(time_part[4:6])

            # Parse suffix as integer milliseconds: _13 means 13 ms, not 130 ms.
            ms = int(ms_str[:3])

            total_ms = (hh * 3600 + mm * 60 + ss) * 1000 + ms

            # 根据配置的日期格式生成date_key
            if self.date_format == "auto":
                # 自动根据长度选择
                date_key = date_part  # 使用原始字符串作为键
            else:
                # 使用指定格式解析并生成标准键
                year, day_of_year = self.parse_date_part(date_part)
                # 生成标准格式的日期键: YYYY-DDD
                date_key = f"{year:04d}-{day_of_year:03d}"

            return (date_key, total_ms)

        # 尝试不带毫秒的模式
        match = re.search(pattern_without_ms, filename)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            hh = int(time_part[0:2])
            mm = int(time_part[2:4])
            ss = int(time_part[4:6])
            total_ms = (hh * 3600 + mm * 60 + ss) * 1000

            if self.date_format == "auto":
                date_key = date_part
            else:
                year, day_of_year = self.parse_date_part(date_part)
                date_key = f"{year:04d}-{day_of_year:03d}"

            return (date_key, total_ms)

        # 容错模式：尝试更宽松的匹配
        if self.fallback:
            # 尝试匹配任何看起来像时间格式的部分
            fallback_pattern = r"_(\d{6,8})_(\d{6})"
            match = re.search(fallback_pattern, filename)
            if match:
                date_part = match.group(1)
                time_part = match.group(2)

                # 尝试解析时间
                try:
                    hh = int(time_part[0:2])
                    mm = int(time_part[2:4])
                    ss = int(time_part[4:6])
                    total_ms = (hh * 3600 + mm * 60 + ss) * 1000

                    if self.date_format == "auto":
                        date_key = date_part
                    else:
                        year, day_of_year = self.parse_date_part(date_part)
                        date_key = f"{year:04d}-{day_of_year:03d}"

                    return (date_key, total_ms)
                except ValueError, IndexError:
                    pass

        return None


def _parse_time_from_filename(filename):
    """从文件名解析时间信息（精确到毫秒），用于时间对齐匹配。

    这是向后兼容的包装函数，使用新的TimeParser类。

    文件名格式:
      - 合成6位日期示例: 149MHz_000001_000000_000.fits
      - 合成7位日期示例: 149MHz_2000001_000000_000.fits

    返回: (date_str, total_ms) 或 None
      - date_str  : 日期字符串，用于跨天判断
      - total_ms  : 当天从0点起的毫秒数，用于数值比较
    """
    # 创建解析器实例，使用默认配置（日期格式自动检测）
    parser = TimeParser({"date_format": "auto", "time_parsing_fallback": True})

    result = parser.parse_time_from_filename(filename)
    if result:
        date_key, total_ms = result
        # 为了向后兼容，返回原始格式
        return (date_key, total_ms)
    return None


def create_time_parser(cfg=None):
    """创建时间解析器实例

    参数:
    ----------
    cfg : dict, 可选
        配置字典，如果为None则使用全局CONFIG

    返回:
    -------
    TimeParser 实例
    """
    if cfg is None:
        cfg = _workflow().CONFIG
    return TimeParser(cfg)


def _check_time_alignment(
    rr_header, ll_header, rr_path, ll_path, tolerance_seconds=1.0
):
    """检查RR和LL文件的时间对齐情况。

    优先从文件名解析精确时间戳（毫秒级），
    次之从FITS header解析，最后退回字符串比较。
    """
    tolerance_ms = tolerance_seconds * 1000

    # ── 1. 优先用文件名时间戳（更精确、更可靠） ──────────────────
    rr_parsed = _parse_time_from_filename(os.path.basename(rr_path))
    ll_parsed = _parse_time_from_filename(os.path.basename(ll_path))

    if rr_parsed is not None and ll_parsed is not None:
        rr_date, rr_ms = rr_parsed
        ll_date, ll_ms = ll_parsed
        if rr_date != ll_date:
            print(
                f"警告: RR/LL文件日期不一致 (RR={rr_date}, LL={ll_date})\n"
                f"  RR: {os.path.basename(rr_path)}\n"
                f"  LL: {os.path.basename(ll_path)}"
            )
            return False
        diff_ms = abs(rr_ms - ll_ms)
        if diff_ms > tolerance_ms:
            print(
                f"警告: RR和LL文件时间差 {diff_ms:.1f} ms 超过容差 {tolerance_ms:.1f} ms\n"
                f"  RR: {os.path.basename(rr_path)}\n"
                f"  LL: {os.path.basename(ll_path)}"
            )
            return False
        return True

    # ── 2. 降级：从FITS header解析 ───────────────────────────────
    from astropy.time import Time

    rr_time_str = get_time_from_header(rr_header)
    ll_time_str = get_time_from_header(ll_header)

    if rr_time_str == "Unknown" or ll_time_str == "Unknown":
        # 无法从任何来源获取时间，假设对齐
        return True

    try:
        rr_time = Time(rr_time_str, format="isot", scale="utc")
        ll_time = Time(ll_time_str, format="isot", scale="utc")
        time_diff_s = abs((rr_time - ll_time).sec)
        if time_diff_s > tolerance_seconds:
            print(
                f"警告: RR和LL文件时间差 {time_diff_s:.3f} 秒超过容差 {tolerance_seconds} 秒\n"
                f"  RR: {rr_time_str}\n"
                f"  LL: {ll_time_str}"
            )
            return False
        return True
    except Exception as e:
        print(f"时间解析失败: {e}，使用字符串比较")
        return rr_time_str == ll_time_str


def _match_rr_ll_by_time(
    rr_files: list, ll_files: list, tolerance_ms: float = 10.0, cfg=None
):
    """根据文件名时间戳将RR与LL文件逐一配对（毫秒级精度）。

    参数:
    ----------
    rr_files      : RR文件路径列表（已排序）
    ll_files      : LL文件路径列表（已排序）
    tolerance_ms  : 时间匹配容差（毫秒），默认10ms
    cfg          : 配置字典，用于时间解析

    返回:
    -------
    matched_pairs : list of (rr_path, ll_path)
    """
    # 创建时间解析器
    parser = create_time_parser(cfg)

    # 构建LL时间索引: {(date_key, total_ms): ll_path}
    ll_index: dict = {}
    ll_no_parse: list = []
    for ll_path in ll_files:
        parsed = parser.parse_time_from_filename(os.path.basename(ll_path))
        if parsed is None:
            ll_no_parse.append(ll_path)
        else:
            key = parsed  # (date_key, total_ms)
            if key in ll_index:
                # 重复时间戳：保留先出现的（有序列表中索引更小的）
                pass
            else:
                ll_index[key] = ll_path

    if ll_no_parse:
        warnings.warn(
            f"有 {len(ll_no_parse)} 个LL文件无法从文件名解析时间，将被跳过。",
            stacklevel=2,
        )

    matched_pairs: list = []
    unmatched_rr: list = []

    # 将LL索引按日期分组，加速搜索
    from collections import defaultdict

    ll_by_date: dict = defaultdict(list)  # {date_key: [(total_ms, ll_path), ...]}
    for (date_key, total_ms), ll_path in ll_index.items():
        ll_by_date[date_key].append((total_ms, ll_path))
    # 每个日期内按ms排序
    for date_key in ll_by_date:
        ll_by_date[date_key].sort(key=lambda x: x[0])

    for rr_path in rr_files:
        parsed = parser.parse_time_from_filename(os.path.basename(rr_path))
        if parsed is None:
            unmatched_rr.append(rr_path)
            warnings.warn(
                f"RR文件 {os.path.basename(rr_path)} 无法解析时间，跳过。", stacklevel=2
            )
            continue

        rr_date_key, rr_ms = parsed

        # ① 精确匹配
        if (rr_date_key, rr_ms) in ll_index:
            matched_pairs.append((rr_path, ll_index[(rr_date_key, rr_ms)]))
            continue

        # ② 容差范围内最近邻匹配
        candidates = ll_by_date.get(rr_date_key, [])
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


def _radio_item_time_key(item, parser):
    path = item[0] if isinstance(item, tuple) else item
    return parser.parse_time_from_filename(os.path.basename(path))


def _build_slots_by_common_time(per_band: list, cfg: dict) -> list | None:
    """Build multi-band slots by matching nearest parsed times across all bands."""
    parser = create_time_parser(cfg)
    per_band_entries = []
    for band_items in per_band:
        entries = []
        for item in band_items:
            key = _radio_item_time_key(item, parser)
            if key is None:
                return None
            entries.append((key, item))
        per_band_entries.append(entries)

    if not per_band_entries:
        return []
    if any(not entries for entries in per_band_entries):
        return []

    tolerance_ms = float(cfg.get("multi_band_time_tolerance_seconds", 0.1)) * 1000.0
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
            match_index = _nearest_time_entry_index(
                entries, ref_key, used_by_band[band_index], tolerance_ms
            )
            if match_index is None:
                slot = []
                matched_indices = []
                break
            matched_indices.append(match_index)
            slot.append(entries[match_index][1])
        if slot:
            slot_times = [
                per_band_entries[band_index][match_index][0][1]
                for band_index, match_index in enumerate(matched_indices)
            ]
            if max(slot_times) - min(slot_times) > tolerance_ms:
                continue
            for band_index, match_index in enumerate(matched_indices):
                used_by_band[band_index].add(match_index)
            slots.append(slot)

    used_count = sum(len(used) for used in used_by_band)
    total_count = sum(len(entries) for entries in per_band_entries)
    dropped = total_count - used_count
    if dropped:
        print(
            "Dropped "
            f"{dropped} band-time entries because not every band has a usable match."
        )
    return slots


def _build_multi_band_slots(cfg: dict) -> list:
    """
    Build multi-band synthesis time slots (each slot contains files of each band at the same time).

    【Optimization】Inner time slot construction changed to zip(*per_band),
    eliminating nested for loops, slightly faster and more concise code.
    """
    root = cfg["multi_band_root"]
    freqs = cfg["multi_band_freqs"]
    pattern = cfg["band_dir_pattern"]
    polarization = cfg["polarization"]
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", None)

    # 检查是否启用左右旋数据加和
    combine_polarizations = cfg.get("combine_polarizations", False)
    time_tolerance = cfg.get("time_tolerance_seconds", 1.0)

    per_band = []
    for freq in freqs:
        if combine_polarizations and polarization == "RR+LL":
            # 左右旋数据加和模式：需要读取RR和LL两个目录的文件
            rr_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"])
            )
            ll_dir = os.path.join(
                root, pattern.format(freq=freq, polar=cfg["ll_dir_suffix"])
            )

            rr_files = _workflow()._sorted_fits_for_band(
                rr_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            ll_files = _workflow()._sorted_fits_for_band(
                ll_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            rr_files = _filter_bad_radio_files(
                rr_files, freq, cfg["rr_dir_suffix"], cfg
            )
            ll_files = _filter_bad_radio_files(
                ll_files, freq, cfg["ll_dir_suffix"], cfg
            )

            # ── 基于文件名时间戳做精确匹配（毫秒级） ──────────────
            tolerance_ms = time_tolerance * 1000  # 秒 → 毫秒
            print(
                f"  频率 {freq}MHz: RR={len(rr_files)} 文件, "
                f"LL={len(ll_files)} 文件，开始时间匹配 (容差={tolerance_ms:.1f}ms)..."
            )
            # 传递cfg给匹配函数，以便使用正确的时间解析配置
            combined_files = _match_rr_ll_by_time(rr_files, ll_files, tolerance_ms, cfg)

            if not combined_files:
                raise ValueError(
                    f"频率 {freq}MHz: RR和LL文件时间匹配失败，没有找到任何匹配对"
                )

            per_band.append(combined_files)
        else:
            # 普通模式：只读取指定偏振的文件
            band_dir = os.path.join(root, pattern.format(freq=freq, polar=polarization))
            files = _workflow()._sorted_fits_for_band(
                band_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            files = _filter_bad_radio_files(files, freq, polarization, cfg)
            per_band.append(files)

    # ★ 优化：zip 直接转置二维列表，替代双层 for 循环
    slots = _build_slots_by_common_time(per_band, cfg)
    if slots is None:
        if _require_study_mode(cfg.get("study_mode")) == "confirmatory":
            raise ValueError(
                "Confirmatory multi-band selection requires parseable UTC times; "
                "positional fallback is forbidden"
            )
        print(
            "Warning: could not parse all radio times; "
            "falling back to positional slots."
        )
        slots = _build_slots_by_position(per_band)

    print(f"Built {len(slots)} time slots, each slot contains {len(freqs)} bands")
    print(f"Polarization: {polarization}")
    if combine_polarizations and polarization == "RR+LL":
        print("Mode: RR + LL data combination")
        if cfg.get("weighted_average", False):
            print(
                f"Weighted average: RR weight={cfg['rr_weight']}, LL weight={cfg['ll_weight']}"
            )
        else:
            print("Simple summation")
    return slots


def _workspace_source_map_selection(cfg: dict) -> dict | None:
    """Decode an explicit web-workspace source-map selection, if supplied."""

    raw_value = cfg.get("selected_source_map_json") or cfg.get("source_map_selection")
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        try:
            selection = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "selected_source_map_json must be valid JSON. Preview again."
            ) from exc
    else:
        selection = raw_value
    if not isinstance(selection, dict):
        raise ValueError("selected_source_map_json must be a JSON object.")
    schema_version = selection.get("schema_version", 1)
    if schema_version not in (1, "1"):
        raise ValueError("Unsupported source-map selection schema version.")
    mode = str(selection.get("mode") or "").strip()
    if mode not in {"single_band", "multi_band"}:
        raise ValueError("Source-map selection must declare single_band or multi_band.")
    items = selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Source-map selection must include at least one item.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Source-map selection items must be JSON objects.")
    return {
        "schema_version": 1,
        "mode": mode,
        "candidate_ids": list(selection.get("candidate_ids") or []),
        "items": items,
    }


def _normalized_selection_path(value) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _source_map_slot_file_paths(slot: list) -> list[str]:
    paths: list[str] = []
    for item in slot:
        if isinstance(item, (tuple, list)):
            paths.extend(os.fspath(path) for path in item)
        else:
            paths.append(os.fspath(item))
    return paths


def _selected_workspace_slot_items(
    slots: list, selection: dict
) -> list[tuple[int, list]]:
    if selection["mode"] != "multi_band":
        raise ValueError("Multi-band source-map run requires a multi-band selection.")
    selected: list[tuple[int, list]] = []
    seen_indices: set[int] = set()
    for item in selection["items"]:
        slot_index = _candidate_slot_index(item)
        if slot_index < 0 or slot_index >= len(slots):
            raise ValueError(
                "Selected source-map slot is no longer available. Preview again."
            )
        if slot_index in seen_indices:
            continue
        expected_paths = item.get("paths")
        if not isinstance(expected_paths, list) or not expected_paths:
            raise ValueError(
                "Selected source-map slot is missing its source paths. Preview again."
            )
        actual_paths = _source_map_slot_file_paths(slots[slot_index])
        if tuple(map(_normalized_selection_path, expected_paths)) != tuple(
            map(_normalized_selection_path, actual_paths)
        ):
            raise ValueError(
                "Selected source-map slot no longer matches the current input "
                "folder. Preview again."
            )
        selected.append((slot_index, slots[slot_index]))
        seen_indices.add(slot_index)
    if not selected:
        raise ValueError("No source-map slots were selected.")
    return selected


def _selected_workspace_files(selection: dict, cfg: dict) -> list[str]:
    if selection["mode"] != "single_band":
        raise ValueError("Single-band source-map run requires a single-band selection.")
    selected: list[str] = []
    combine = (
        bool(cfg.get("combine_polarizations")) and cfg.get("polarization") == "RR+LL"
    )
    for item in selection["items"]:
        paths = item.get("paths")
        run_path = item.get("run_path")
        if not run_path and isinstance(paths, list) and paths:
            run_path = paths[0]
        if not isinstance(run_path, str) or not run_path.strip():
            raise ValueError(
                "Selected source-map file is missing its run path. Preview again."
            )
        if not os.path.isfile(run_path):
            raise FileNotFoundError(
                f"Selected source-map FITS file does not exist: {run_path}"
            )
        if combine:
            if not isinstance(paths, list) or len(paths) < 2:
                raise ValueError(
                    "RR+LL source-map run requires a matched RR/LL preview selection."
                )
            missing = [path for path in paths if not os.path.isfile(os.fspath(path))]
            if missing:
                raise FileNotFoundError(
                    "RR+LL source-map run is missing matched polarization file(s): "
                    + ", ".join(map(str, missing))
                )
        selected.append(run_path)
    if not selected:
        raise ValueError("No source-map files were selected.")
    return selected
