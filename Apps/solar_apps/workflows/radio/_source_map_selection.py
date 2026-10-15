"""Source-map input discovery, frozen collections, quality flags and time slots."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path

from ._source_map_helpers import (
    _raw_quality_filter_enabled,
    _require_study_mode,
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
    """Check complete UTC observation times without assuming missing times align."""
    from solar_toolkit.radio.time_matching import observation_stamp
    from solar_toolkit.time import seconds_to_ns

    try:
        left = observation_stamp(
            rr_header, path=rr_path, fallback_parser=_radio_stamp_from_filename
        )
        right = observation_stamp(
            ll_header, path=ll_path, fallback_parser=_radio_stamp_from_filename
        )
        return abs(right.tai_ns - left.tai_ns) <= seconds_to_ns(tolerance_seconds)
    except ValueError, TypeError:
        return False


def _match_rr_ll_by_time(
    rr_files: list,
    ll_files: list,
    tolerance_ms: float = 10.0,
    cfg=None,
    *,
    frequency_mhz=None,
):
    """Strict maximum monotone one-to-one RR/LL pairs using FITS UTC and grid checks."""
    from solar_toolkit.radio.time_matching import index_radio_frames, pair_radio_frames

    def fallback(path):
        return _radio_stamp_from_filename(path, cfg)

    left = index_radio_frames(
        rr_files,
        frequency_mhz=frequency_mhz,
        polarization="RR",
        fallback_parser=fallback,
    )
    right = index_radio_frames(
        ll_files,
        frequency_mhz=frequency_mhz,
        polarization="LL",
        fallback_parser=fallback,
    )
    window = _image_time_window(cfg, [*rr_files, *ll_files])
    if frequency_mhz is not None:
        paired = pair_radio_frames(
            left, right, tolerance_seconds=float(tolerance_ms) / 1000, **window
        )
    else:
        # Distinct observing frequencies are independent physical streams.
        # Equal-time HDUs may have different storage orders in the two files.
        groups = []
        for frame in left:
            frequency = frame.payload.get("frequency_mhz")
            if frequency is None:
                continue
            bucket = next(
                (
                    values
                    for freq, values in groups
                    if _workflow().np.isclose(freq, frequency, rtol=1e-5, atol=1e-6)
                ),
                None,
            )
            if bucket is None:
                bucket = []
                groups.append((frequency, bucket))
            bucket.append(frame)
        paired = []
        peer_groups = [[] for _ in groups]
        for frame in right:
            frequency = frame.payload.get("frequency_mhz")
            if frequency is None:
                continue
            eligible = [
                index
                for index, (value, _) in enumerate(groups)
                if _workflow().np.isclose(value, frequency, rtol=1e-5, atol=1e-6)
            ]
            if eligible:
                bucket = min(
                    eligible,
                    key=lambda index: (abs(groups[index][0] - frequency), index),
                )
                peer_groups[bucket].append(frame)
        for (frequency, members), peers in zip(groups, peer_groups):
            paired.extend(
                pair_radio_frames(
                    members,
                    peers,
                    tolerance_seconds=float(tolerance_ms) / 1000,
                    **window,
                )
            )
        paired.sort(key=lambda frame: (frame.time.tai_ns, frame.frame_id))
    pairs = [_item_from_logical_frame(frame) for frame in paired]
    print(
        f"  时间匹配结果: 成功 {len(pairs)} 对，RR未匹配 {len(left) - len(pairs)} 个逻辑图像。"
    )
    return pairs


def _radio_item_time_key(item, parser):
    path = item[0] if isinstance(item, tuple) else item
    return parser.parse_time_from_filename(os.path.basename(path))


def _build_slots_by_common_time(per_band: list, cfg: dict) -> list | None:
    """Keep every longest-stream frame; cover other streams monotonically with reuse."""
    from solar_toolkit.radio.time_matching import selection_metadata
    from solar_toolkit.time import TimeFrame, TimeStream, build_sequence_match_plan

    streams = []
    source_ids = []
    try:
        for index, items in enumerate(per_band):
            source_id = f"band:{index:04d}"
            source_ids.append(source_id)
            logical_items = []
            for item in items:
                if getattr(item, "logical_frame", None) is not None:
                    logical_items.append(item)
                elif isinstance(item, (tuple, list)):
                    if len(item) != 2:
                        raise ValueError(
                            "A source-map polarization pair requires two exact members"
                        )
                    logical_items.extend(
                        _match_rr_ll_by_time(
                            [item[0]],
                            [item[1]],
                            float(cfg.get("time_tolerance_seconds", 1.0)) * 1000,
                            cfg,
                        )
                    )
                else:
                    logical_items.extend(
                        _index_source_items([item], cfg, apply_window=False)
                    )
            frames = [
                TimeFrame(item.logical_frame.frame_id, item.logical_frame.time, item)
                for item in logical_items
            ]
            streams.append(
                TimeStream(
                    source_id,
                    frames,
                    tolerance_seconds=float(
                        cfg.get("multi_band_time_tolerance_seconds", 0.1)
                    ),
                )
            )
    except ValueError as exc:
        if any(
            marker in str(exc)
            for marker in ("Expected one image observation", "Unsupported", "Ambiguous")
        ):
            raise ValueError(
                "Unsupported source-map logical image layout: " + str(exc)
            ) from exc
        return None
    except OSError, TypeError:
        return None
    if not any(stream.frames for stream in streams):
        return []
    plan = build_sequence_match_plan(streams, **_image_time_window(cfg))
    cfg["_radio_time_match_plan"] = plan.to_dict()
    slots = []
    for row in plan.rows:
        inputs = [
            row.selections[source_id].frame.payload
            for source_id in source_ids
            if row.selections[source_id].frame is not None
        ]
        metadata = selection_metadata(row, plan=plan)
        metadata["window"] = plan.to_dict()["window"]
        metadata["master_source_id"] = plan.master_source_id
        item_index = 0
        for source_id in source_ids:
            selection = row.selections[source_id]
            if selection.frame is not None:
                value = selection.frame.payload
                metadata["selections"][source_id]["paths"] = (
                    list(value) if isinstance(value, (tuple, list)) else [str(value)]
                )
                metadata["selections"][source_id]["locator"] = _logical_locator(
                    value.logical_frame
                )
                metadata["selections"][source_id]["slot_item_index"] = item_index
                item_index += 1
        slots.append(RadioTimeSlot(inputs, time_match=metadata))
    return slots


def _build_multi_band_slots(cfg: dict) -> list:
    """
    Build multi-band synthesis time slots (each slot contains files of each band at the same time).

    Slots retain the longest valid stream and monotonically reuse the other bands.
    """
    root = cfg["multi_band_root"]
    freqs = cfg["multi_band_freqs"]
    pattern = cfg["band_dir_pattern"]
    polarization = cfg["polarization"]
    start_idx, end_idx = _image_file_slice(cfg)

    # 检查是否启用左右旋数据加和
    combine_polarizations = cfg.get("combine_polarizations", False)
    time_tolerance = cfg.get("time_tolerance_seconds", 1.0)

    per_band = []
    for freq in freqs:
        if combine_polarizations and polarization == "RR+LL":
            # 左右旋数据加和模式：需要读取RR和LL两个目录的文件
            rr_dir = _workflow().os.path.join(
                root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"])
            )
            ll_dir = _workflow().os.path.join(
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
            try:
                combined_files = _match_rr_ll_by_time(
                    rr_files, ll_files, tolerance_ms, cfg, frequency_mhz=float(freq)
                )
            except (OSError, ValueError, TypeError) as exc:
                if str(cfg.get("study_mode", "")).casefold() == "confirmatory":
                    raise ValueError(
                        "Cannot index frozen radio observations; positional fallback is forbidden: "
                        + str(exc)
                    ) from exc
                raise

            per_band.append(combined_files)
        else:
            # 普通模式：只读取指定偏振的文件
            band_dir = _workflow().os.path.join(
                root, pattern.format(freq=freq, polar=polarization)
            )
            files = _workflow()._sorted_fits_for_band(
                band_dir, start_idx, end_idx, study_mode=cfg.get("study_mode")
            )
            files = _filter_bad_radio_files(files, freq, polarization, cfg)
            try:
                indexed = _index_source_items(
                    files,
                    cfg,
                    frequency_mhz=float(freq),
                    polarization=polarization,
                    apply_window=False,
                )
            except (OSError, ValueError, TypeError) as exc:
                if str(cfg.get("study_mode", "")).casefold() == "confirmatory":
                    raise ValueError(
                        "Cannot index frozen radio observations; positional fallback is forbidden: "
                        + str(exc)
                    ) from exc
                raise
            per_band.append(indexed)

    # Freeze the complete longest-stream UTC plan before rendering.
    slots = _build_slots_by_common_time(per_band, cfg)
    if slots is None:
        raise ValueError(
            "Multi-band selection requires parseable UTC times; positional fallback is forbidden"
        )

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
        slot_index = _workflow()._candidate_slot_index(item)
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
        expected_time = item.get("time_match")
        if expected_time:
            actual_time = getattr(slots[slot_index], "time_match", {})
            expected_frames = {
                source: value.get("frame_id")
                for source, value in expected_time.get("selections", {}).items()
            }
            actual_frames = {
                source: value.get("frame_id")
                for source, value in actual_time.get("selections", {}).items()
            }
            if (
                expected_time.get("algorithm_version") != 3
                or expected_time.get("window") != actual_time.get("window")
                or expected_time.get("output_frame_id")
                != actual_time.get("output_frame_id")
                or expected_time.get("master_frame_id")
                != actual_time.get("master_frame_id")
                or expected_frames != actual_frames
            ):
                raise ValueError(
                    "Selected source-map logical images changed. Preview again."
                )
            chosen_slot = _restore_source_map_slot(slots[slot_index], expected_time)
        else:
            chosen_slot = slots[slot_index]
        selected.append((slot_index, chosen_slot))
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
        if not _workflow().os.path.isfile(run_path):
            raise FileNotFoundError(
                f"Selected source-map FITS file does not exist: {run_path}"
            )
        if combine:
            if not isinstance(paths, list) or len(paths) < 2:
                raise ValueError(
                    "RR+LL source-map run requires a matched RR/LL preview selection."
                )
            missing = [
                path
                for path in paths
                if not _workflow().os.path.isfile(_workflow().os.fspath(path))
            ]
            if missing:
                raise FileNotFoundError(
                    "RR+LL source-map run is missing matched polarization file(s): "
                    + ", ".join(map(str, missing))
                )
        expected_time = item.get("time_match")
        if expected_time:
            from solar_toolkit.time import build_sequence_match_plan

            current_window = build_sequence_match_plan(
                [], **_image_time_window(cfg, paths or [run_path])
            ).to_dict()["window"]
            expected_window = expected_time.get("window")
            if (expected_window is not None and expected_window != current_window) or (
                expected_window is None
                and (current_window["start_utc"] or current_window["end_utc"])
            ):
                raise ValueError(
                    "Selected source-map image UTC interval changed. Preview again."
                )
            selected.append(
                _restore_source_map_slot(
                    [paths if combine else run_path], expected_time, cfg=cfg
                )[0]
            )
        else:
            selected.append(run_path)
    if not selected:
        raise ValueError("No source-map files were selected.")
    return selected


def _radio_stamp_from_filename(path, cfg=None):
    """Resolve the configured radio calendar and retain the full filename fraction."""
    import re

    from solar_toolkit.time import normalize_time

    if not path:
        return None
    path = str(path).split("|")[0]
    name = _workflow().os.path.basename(path)
    match = re.search(r"_(\d{6,8})_(\d{6})(?:_(\d{1,9}))?(?!\d)", name)
    if match is None:
        from solar_toolkit.time import extract_timestamp_from_filename

        return extract_timestamp_from_filename(name)
    date_part, clock, fraction = match.groups()
    base_date = _workflow()._parse_radio_date_part(date_part, path, cfg)
    if base_date is None:
        return None
    if fraction and len(fraction) <= 3:
        fraction = f"{int(fraction):03d}"
    text = f"{base_date.isoformat()}T{clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    if fraction:
        text += "." + fraction
    scale = "tai" if name.upper().find("_TAI") >= 0 else "utc"
    try:
        return normalize_time(text, source="filename", scale=scale)
    except ValueError, TypeError:
        return None


def _logical_item_header(item):
    """Return selected-layer metadata without reading its pixels."""
    frame = item.logical_frame
    payload = frame.payload
    if payload.get("pair_members"):
        payload = payload["pair_members"][0]
    header = _workflow().fits.Header(list(payload["header_cards"]))
    if payload.get("frequency_mhz") is not None:
        header["FREQ"] = payload["frequency_mhz"]
        header["FREQUNIT"] = "MHz"
    header["POLAR"] = payload["polarization"]
    header["DATE-OBS"] = frame.time.utc
    header["TIMESYS"] = "UTC"
    return header


class RadioFramePath(str):
    """Path-compatible exact image locator, including HDU and cube plane."""

    def __new__(cls, path, logical_frame=None):
        value = super().__new__(cls, str(path))
        value.logical_frame = logical_frame
        return value

    def __getnewargs__(self):
        return str(self), self.logical_frame


class RadioFramePair(tuple):
    """Tuple-compatible verified RR/LL pair with exact member locators."""

    def __new__(cls, paths, logical_frame=None):
        value = super().__new__(cls, paths)
        value.logical_frame = logical_frame
        return value

    def __getnewargs__(self):
        return tuple(self), self.logical_frame


def _pair_member_frame(member, tai_ns):
    from solar_toolkit.radio.time_matching import observation_stamp
    from solar_toolkit.time import TimeFrame, TimeStamp

    try:
        stamp = observation_stamp(member["header"])
        if stamp.tai_ns != int(tai_ns):
            raise ValueError("Member observation timestamp differs from its exact pair")
    except ValueError:
        stamp = TimeStamp.from_tai_ns(int(tai_ns), source=member.get("time_source"))
    return TimeFrame(member["frame_id"], stamp, member)


def _item_from_logical_frame(frame):
    payload = frame.payload
    if payload.get("pair_members"):
        members = [
            _pair_member_frame(member, stamp)
            for member, stamp in zip(
                payload["pair_members"], payload["actual_times_tai_ns"]
            )
        ]
        return RadioFramePair(
            [RadioFramePath(member.payload["path"], member) for member in members],
            frame,
        )
    return RadioFramePath(payload["path"], frame)


def _image_time_window(cfg=None, paths=()):
    """Resolve image bounds independently of spectrogram display settings."""
    cfg = cfg or {}
    start, end = cfg.get("image_time_start"), cfg.get("image_time_end")
    if start is not None or end is not None:
        return {"time_start": start, "time_end": end}
    roots = [
        cfg.get(key) for key in ("multi_band_root", "data_dir", "single_file_path")
    ]
    roots.extend(paths)
    for value in roots:
        if value is None:
            continue
        directory = _workflow().Path(value)
        if directory.is_file():
            directory = directory.parent
        for parent in (directory, *directory.parents):
            manifest = parent / ".frozen-collection-v1.json"
            if manifest.is_file():
                payload = _workflow().json.loads(manifest.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("schema") != "solar-radio-frozen-collection-v1"
                ):
                    raise ValueError("Unsupported frozen collection image window")
                selection = payload.get("selection")
                if (
                    not isinstance(selection, dict)
                    or not selection.get("start_utc")
                    or not selection.get("end_utc")
                ):
                    raise ValueError(
                        "Frozen collection requires an explicit image UTC interval"
                    )
                window = {
                    "time_start": selection["start_utc"],
                    "time_end": selection["end_utc"],
                }
                from solar_toolkit.time import filter_time_frames

                filter_time_frames([], **window)
                return window
    return {"time_start": None, "time_end": None}


def _index_source_items(
    paths, cfg, *, frequency_mhz=None, polarization=None, apply_window=True
):
    from solar_toolkit.radio.time_matching import index_radio_frames
    from solar_toolkit.time import filter_time_frames

    paths = list(paths)
    frames = index_radio_frames(
        paths,
        frequency_mhz=frequency_mhz,
        polarization=polarization,
        fallback_parser=lambda path: _radio_stamp_from_filename(path, cfg),
    )
    if apply_window:
        frames = filter_time_frames(frames, **_image_time_window(cfg, paths))
    return [_item_from_logical_frame(frame) for frame in frames]


def _image_file_slice(cfg):
    window = _image_time_window(cfg)
    if any(value is not None for value in window.values()) and not cfg.get(
        "image_window_use_indices", False
    ):
        return 0, None
    return cfg.get("start_idx", 0), cfg.get("end_idx")


def _logical_locator(frame):
    """Compact JSON snapshot; reconstruction reads this exact HDU, never rematches."""
    payload = frame.payload
    result = {
        key: payload.get(key)
        for key in (
            "path",
            "paths",
            "hdu_index",
            "plane_index",
            "plane_indices",
            "shape",
            "frequency_mhz",
            "polarization",
            "size_bytes",
            "mtime_ns",
        )
    }
    result.update(
        frame_id=frame.frame_id,
        time_tai_ns=frame.time.tai_ns,
        time_utc=frame.time.utc + "Z",
        time_raw=frame.time.raw,
        time_scale=frame.time.scale,
        time_source=frame.time.source,
    )
    if payload.get("pair_members"):
        result["pair_members"] = [
            _logical_locator(_pair_member_frame(member, stamp))
            for member, stamp in zip(
                payload["pair_members"], payload["actual_times_tai_ns"]
            )
        ]
        result["pair_delta_ns"] = payload["pair_delta_ns"]
    return result


def _logical_frame_from_locator(locator):
    from solar_toolkit.time import TimeFrame, TimeStamp

    payload = dict(locator)
    stamp = TimeStamp.from_tai_ns(
        int(locator["time_tai_ns"]),
        source=locator.get("time_source"),
        raw=locator.get("time_raw"),
    )
    from dataclasses import replace

    stamp = replace(stamp, scale=locator.get("time_scale", stamp.scale))
    if locator.get("pair_members"):
        members = [
            _logical_frame_from_locator(member) for member in locator["pair_members"]
        ]
        payload.update(
            pair_members=tuple(member.payload for member in members),
            actual_times_tai_ns=tuple(member.time.tai_ns for member in members),
            actual_times_utc=tuple(member.time.utc for member in members),
        )
    else:
        path = _workflow().Path(locator["path"])
        stat = path.stat()
        if (
            stat.st_size != locator["size_bytes"]
            or stat.st_mtime_ns != locator["mtime_ns"]
        ):
            raise ValueError(f"Planned radio source changed: {path}")
        with _workflow().fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
            header = hdus[0].header.copy()
            if int(locator["hdu_index"]) != 0:
                header.update(hdus[int(locator["hdu_index"])].header)
        payload["header_cards"] = tuple(
            (card.keyword, card.value, card.comment) for card in header.cards
        )
        payload["header"] = {
            key: list(header[key]) if key in {"COMMENT", "HISTORY"} else header[key]
            for key in header
        }
    return TimeFrame(locator["frame_id"], stamp, payload)


def _restore_source_map_slot(items, time_match, *, cfg=None):
    """Restore frozen JSON candidates without rediscovering image planes or pairs."""
    selections = (time_match or {}).get("selections", {})
    if selections and (time_match or {}).get("algorithm_version") != 3:
        raise ValueError(
            "Legacy source-map matching plan requires a new hierarchy preview"
        )
    explicit_config = cfg is not None
    cfg = cfg or _workflow().CONFIG
    output = []
    for index, item in enumerate(items):
        exact = [
            selection
            for selection in selections.values()
            if selection.get("locator") and selection.get("slot_item_index", 0) == index
        ]
        if len(exact) > 1:
            raise ValueError("Ambiguous source-map logical image locator")
        if exact:
            paths = item if isinstance(item, (tuple, list)) else [item]
            locator_paths = exact[0]["locator"].get("paths") or [
                exact[0]["locator"]["path"]
            ]
            if tuple(map(_normalized_selection_path, paths)) != tuple(
                map(_normalized_selection_path, locator_paths)
            ):
                raise ValueError(
                    "Source-map locator does not match the selected source paths"
                )
            restored = _item_from_logical_frame(
                _logical_frame_from_locator(exact[0]["locator"])
            )
        elif getattr(item, "logical_frame", None) is not None:
            restored = item
        elif isinstance(item, (list, tuple)):
            matches = _match_rr_ll_by_time(
                [item[0]],
                [item[1]],
                float(cfg.get("time_tolerance_seconds", 0.001)) * 1000,
                cfg,
            )
            if len(matches) != 1:
                raise ValueError(
                    "A frozen source-map pair requires one exact logical image per member"
                )
            restored = matches[0]
        else:
            indexed = _index_source_items([item], cfg)
            if len(indexed) != 1:
                raise ValueError(
                    "A frozen source-map slot requires an exact locator for multiple logical images"
                )
            restored = indexed[0]
        output.append(restored)
    if explicit_config:
        from solar_toolkit.time import build_sequence_match_plan, filter_time_frames

        paths = [
            str(path)
            for item in items
            for path in (item if isinstance(item, (list, tuple)) else [item])
        ]
        window = _image_time_window(cfg, paths)
        current_window = build_sequence_match_plan([], **window).to_dict()["window"]
        expected_window = (time_match or {}).get("window")
        if (expected_window is not None and expected_window != current_window) or (
            expected_window is None
            and (current_window["start_utc"] or current_window["end_utc"])
        ):
            raise ValueError("Source-map image UTC interval changed. Preview again.")
        frames = [item.logical_frame for item in output]
        if len(filter_time_frames(frames, **window)) != len(frames):
            raise ValueError(
                "Planned source-map observation is outside the image UTC interval. Preview again."
            )
    return RadioTimeSlot(output, time_match=time_match)


def _strict_single_band_pairs(files, cfg):
    """Pre-pair selected single-band observations instead of guessing same filenames."""
    grouped = {}
    for path in files:
        parent = _workflow().Path(path).parent
        if parent.name not in {
            cfg.get("rr_dir_suffix", "RR"),
            cfg.get("ll_dir_suffix", "LL"),
        }:
            grouped.setdefault(None, []).append(str(path))
        else:
            grouped.setdefault(parent.parent, []).append(str(path))
    output = []
    for root, selected in grouped.items():
        if root is None:
            output.extend(
                _match_rr_ll_by_time(
                    selected,
                    selected,
                    float(cfg.get("time_tolerance_seconds", 0.001)) * 1000,
                    cfg,
                )
            )
            continue
        rr_dir = root / cfg.get("rr_dir_suffix", "RR")
        ll_dir = root / cfg.get("ll_dir_suffix", "LL")
        selected_rr = [
            path for path in selected if _workflow().Path(path).parent == rr_dir
        ]
        selected_ll = [
            path for path in selected if _workflow().Path(path).parent == ll_dir
        ]
        rr = selected_rr or sorted(str(path) for path in rr_dir.glob("*.fits"))
        ll = selected_ll or sorted(str(path) for path in ll_dir.glob("*.fits"))
        output.extend(
            _match_rr_ll_by_time(
                rr, ll, float(cfg.get("time_tolerance_seconds", 0.001)) * 1000, cfg
            )
        )
    return output


def _prepare_single_band_inputs(files, cfg):
    """Enumerate exact planes and freeze RR/LL pairs once before batch rendering."""
    if cfg.get("combine_polarizations"):
        pairs = [item for item in files if isinstance(item, RadioFramePair)]
        plain = [item for item in files if not isinstance(item, RadioFramePair)]
        pairs.extend(_strict_single_band_pairs(plain, cfg) if plain else [])
        from solar_toolkit.time import filter_time_frames

        retained = {
            frame.frame_id
            for frame in filter_time_frames(
                [pair.logical_frame for pair in pairs], **_image_time_window(cfg)
            )
        }
        pairs = [pair for pair in pairs if pair.logical_frame.frame_id in retained]
        cfg["_single_band_pair_map"] = {
            pair[0].logical_frame.frame_id: pair for pair in pairs
        }
        return [pair[0] for pair in pairs]
    output = []
    for item in files:
        output.extend(
            [item]
            if getattr(item, "logical_frame", None) is not None
            else _index_source_items([item], cfg, polarization=cfg.get("polarization"))
        )
    from solar_toolkit.time import filter_time_frames

    retained = {
        frame.frame_id
        for frame in filter_time_frames(
            [item.logical_frame for item in output], **_image_time_window(cfg)
        )
    }
    return [item for item in output if item.logical_frame.frame_id in retained]


def _pair_time_metadata(pair, cfg):
    from solar_toolkit.radio.time_matching import frame_for_item, selection_metadata
    from solar_toolkit.time import TimeStream, build_sequence_match_plan

    frame = getattr(pair, "logical_frame", None) or frame_for_item(
        pair, fallback_parser=lambda path: _radio_stamp_from_filename(path, cfg)
    )
    source_id = "radio:RR+LL" if isinstance(pair, (tuple, list)) else "radio:single"
    plan = build_sequence_match_plan(
        [TimeStream(source_id, [frame])], **_image_time_window(cfg)
    )
    if not plan.rows:
        raise ValueError(
            "Source-map observation is outside the selected image UTC interval"
        )
    result = selection_metadata(plan.rows[0], plan=plan)
    result["window"] = plan.to_dict()["window"]
    result["master_source_id"] = plan.master_source_id
    selected = result["selections"][source_id]
    if getattr(pair, "logical_frame", None) is not None:
        selected.update(
            locator=_logical_locator(frame),
            slot_item_index=0,
            paths=(
                list(map(str, pair)) if isinstance(pair, (tuple, list)) else [str(pair)]
            ),
        )
    return result


class RadioTimeSlot(list):
    """List-compatible inputs with the actual master UTC and display provenance."""

    def __init__(self, items=(), *, time_match=None):
        super().__init__(items)
        self.time_match = time_match or {}
