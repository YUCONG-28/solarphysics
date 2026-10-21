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
    """Parse shared calendar formats and the explicit legacy short-date format.

    Six-digit dates mean YYYYMD. Seven-digit dates retain the historical
    YYYYMDD-first interpretation, falling back to YYYYDDD only when that
    calendar date is invalid. No fractional digits or trailing text are cut.
    """
    import pandas as pd

    from solar_toolkit.radio._time import table_timestamp

    result = table_timestamp(date_str)
    if not pd.isna(result):
        return result
    match = re.fullmatch(r"(\d{6,7})_(\d{6})(?:_(\d{1,9}))?", str(date_str).strip())
    if match is None:
        return None
    date_part, clock, fraction = match.groups()
    year = int(date_part[:4])
    try:
        date = datetime(year, int(date_part[4:5]), int(date_part[5:]))
    except ValueError:
        if len(date_part) != 7:
            return None
        try:
            date = datetime.strptime(date_part, "%Y%j")
            if date.year != year:
                return None
        except ValueError:
            return None
    text = (
        date.strftime("%Y-%m-%d") + "T" + ":".join((clock[:2], clock[2:4], clock[4:]))
    )
    if fraction:
        text += "." + fraction
    result = table_timestamp(text)
    return None if pd.isna(result) else result


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

    合成文件名示例: 149MHz_2000001_000000_000.fits
      - 日期部分:   2000001  (YYYYDDD，7位)
      - 时间部分:   000000   (HHMMSS，6位)
      - 毫秒部分:   000      (1~3位，不足3位按实际值处理)

    返回: (date_str, total_ms) 或 None
      - date_str  : 日期字符串，用于跨天判断
      - total_ms  : 当天从0点起的毫秒数，用于数值比较
    """
    # 匹配: _日期(7-8位)_时间(6位)_毫秒(1-3位)
    pattern = r"_(\d{6,8})_(\d{6})_(\d{1,3})"
    match = re.search(pattern, filename)
    if match:
        date_part = match.group(1)  # e.g. "2000001"
        time_part = match.group(2)  # e.g. "000000"
        ms_str = match.group(3)  # e.g. "000"

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


def _match_rr_ll_by_time(rr_files: list, ll_files: list, tolerance_ms: float = 10.0):
    """Use the shared strict UTC/grid/frequency one-to-one radio pairing."""
    from solar_toolkit.radio.time_matching import index_radio_frames, pair_radio_frames

    left = index_radio_frames(
        rr_files, fallback_parser=parse_radio_time_from_filename, strict=False
    )
    right = index_radio_frames(
        ll_files, fallback_parser=parse_radio_time_from_filename, strict=False
    )
    return [
        tuple(frame.payload["paths"])
        for frame in pair_radio_frames(
            left, right, tolerance_seconds=float(tolerance_ms) / 1000
        )
    ]


def _multi_wave_overlay_enabled(cfg: Config) -> bool:
    wavelengths = getattr(cfg, "aia_panel_wavelengths", None)
    return bool(wavelengths and len(wavelengths) > 1)


def _fits_files_in_dir(path: str, start_idx=None, end_idx=None) -> list[str]:
    files = sorted(glob.glob(os.path.join(path, "*.fits")))
    return _workflow()._slice_file_list(files, start_idx, end_idx)


def _radio_time_key_from_item(item) -> tuple[str, int] | None:
    path = item[0] if isinstance(item, tuple) else item
    return _workflow()._parse_time_from_filename(os.path.basename(path))


def _build_common_radio_slots_from_entries(per_band_entries, tolerance_ms):
    """Compatibility wrapper for legacy entry lists using the unified fixed plan."""
    from solar_toolkit.time import (
        TimeFrame,
        TimeStamp,
        TimeStream,
        build_sequence_match_plan,
        normalize_time,
    )

    streams = []
    for band_index, entries in enumerate(per_band_entries):
        frames = []
        for index, (key, item) in enumerate(entries):
            if (
                isinstance(item, tuple)
                and len(item) >= 4
                and isinstance(item[3], datetime)
            ):
                stamp = normalize_time(item[3])
            else:
                midnight = _parse_flexible_datetime(f"{key[0]}_000000_0")
                if midnight is None:
                    raise ValueError("Legacy radio entry has no parseable UTC date")
                stamp = TimeStamp.from_tai_ns(
                    normalize_time(midnight).tai_ns + int(key[1]) * 1_000_000
                )
            frames.append(
                TimeFrame(f"band:{band_index:04d};entry:{index}", stamp, item)
            )
        streams.append(
            TimeStream(
                f"band:{band_index:04d}",
                frames,
                tolerance_seconds=float(tolerance_ms) / 1000,
            )
        )
    if not any(stream.frames for stream in streams):
        return []
    plan = build_sequence_match_plan(streams)
    return [
        [
            selection.frame.payload
            for selection in row.selections.values()
            if selection.frame is not None
        ]
        for row in plan.rows
    ]


def build_radio_time_slots_for_overlay(cfg: Config) -> list[tuple[int, dict]]:
    """Keep all longest-radio-stream observations with monotone display reuse."""
    from solar_toolkit.time import build_sequence_match_plan

    streams = _overlay_radio_streams(cfg)
    if not any(stream.frames for stream in streams):
        return []
    plan = build_sequence_match_plan(streams, **_overlay_image_window(cfg))
    cfg._time_match_plan = plan.to_dict()
    return [
        (index, _overlay_slice(row, plan=plan)) for index, row in enumerate(plan.rows)
    ]


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
    from solar_toolkit.time import elapsed_seconds

    candidates = []
    for path in files:
        parsed_time = parser(path)
        if parsed_time is None:
            parsed_time = parser(os.path.basename(path))
        if parsed_time is None:
            continue
        dt = abs(elapsed_seconds(parsed_time, target_time))
        if dt <= threshold_seconds:
            candidates.append((dt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _slot_reference_time(single_slice_bands: dict) -> datetime | None:
    metadata = getattr(single_slice_bands, "time_match", {})
    if metadata.get("output_time_utc") or metadata.get("master_time_utc"):
        return datetime.fromisoformat(
            (metadata.get("output_time_utc") or metadata["master_time_utc"]).rstrip("Z")
        ).replace(tzinfo=None)
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
    """All image routes share the longest route's complete fixed time axis."""
    wavelengths = _workflow()._aia_panel_wavelengths(cfg)
    if not wavelengths:
        return []
    plan = _build_overlay_plan(cfg, wavelengths)
    if plan is None:
        return []
    matched = []
    for index, row in enumerate(plan.rows):
        aia = {
            wave: row.selections[f"aia:{wave}"].frame.payload
            for wave in wavelengths
            if row.selections[f"aia:{wave}"].frame is not None
        }
        hmi = row.selections.get("hmi")
        matched.append(
            (
                aia,
                hmi.frame.payload if hmi and hmi.frame else None,
                [(index, _overlay_slice(row, plan=plan))],
            )
        )
    return matched


def build_matched_pairs(cfg: Config) -> list[tuple[str | None, str | None, list]]:
    """Single AIA overlays use the same complete fixed plan as multi-wave overlays."""
    wave = int(cfg.aia_wavelength)
    plan = _build_overlay_plan(cfg, [wave])
    if plan is None:
        return []
    matched = []
    for index, row in enumerate(plan.rows):
        aia = row.selections[f"aia:{wave}"].frame
        hmi = row.selections.get("hmi")
        matched.append(
            (
                aia.payload if aia else None,
                hmi.frame.payload if hmi and hmi.frame else None,
                [(index, _overlay_slice(row, plan=plan))],
            )
        )
    return matched


class RadioOverlaySlice(dict):
    """Legacy band task mapping carrying the exact master UTC and matching flags."""

    def __init__(self, *args, time_match=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_match = time_match or {}


def _overlay_radio_streams(cfg):
    from solar_toolkit.radio.time_matching import index_radio_frames, pair_radio_frames
    from solar_toolkit.time import TimeFrame, TimeStream
    from . import (
        source_map_workflow,
    )  # Initialize the shared facade before its input adapter.
    from . import _source_map_selection as inputs

    window = _overlay_image_window(cfg)
    radio_start, radio_end = _overlay_file_indices(
        cfg, "radio", cfg.radio_start_idx, cfg.radio_end_idx
    )
    streams = []
    for band in cfg.selected_bands:
        band_dir = os.path.join(cfg.radio_base_dir, band)
        frequency = re.search(r"(\d+(?:\.\d+)?)", str(band))
        frequency = float(frequency.group(1)) if frequency else None

        def fallback(path):
            return inputs._radio_stamp_from_filename(path, {"date_format": "auto"})

        if cfg.combine_polarizations and cfg.polarization_mode == "RR+LL":
            rr = index_radio_frames(
                _fits_files_in_dir(
                    os.path.join(band_dir, cfg.rr_dir_suffix), radio_start, radio_end
                ),
                frequency_mhz=frequency,
                polarization="RR",
                fallback_parser=fallback,
            )
            ll = index_radio_frames(
                _fits_files_in_dir(
                    os.path.join(band_dir, cfg.ll_dir_suffix), radio_start, radio_end
                ),
                frequency_mhz=frequency,
                polarization="LL",
                fallback_parser=fallback,
            )
            frames = pair_radio_frames(
                rr, ll, tolerance_seconds=cfg.time_tolerance_seconds, **window
            )
            polarization = "RR+LL"
        else:
            pol_dir = os.path.join(band_dir, cfg.polarization_mode)
            frames = index_radio_frames(
                _fits_files_in_dir(
                    pol_dir if os.path.isdir(pol_dir) else band_dir,
                    radio_start,
                    radio_end,
                ),
                frequency_mhz=frequency,
                polarization=cfg.polarization_mode,
                fallback_parser=fallback,
            )
            polarization = cfg.polarization_mode
        wrapped = []
        for frame in frames:
            item = inputs._item_from_logical_frame(frame)
            actual = datetime.fromisoformat(frame.time.utc).replace(tzinfo=None)
            wrapped.append(
                TimeFrame(
                    frame.frame_id, frame.time, (band, item, polarization, actual)
                )
            )
        streams.append(
            TimeStream(
                f"radio:{band}",
                wrapped,
                tolerance_seconds=cfg.multi_band_time_tolerance_seconds,
            )
        )
    return streams


def _overlay_image_window(cfg):
    from . import _source_map_selection as inputs

    return inputs._image_time_window(
        {
            "image_time_start": cfg.image_time_start,
            "image_time_end": cfg.image_time_end,
            "multi_band_root": cfg.radio_base_dir,
        }
    )


def _overlay_file_indices(cfg, source, start, end):
    if any(
        value is not None for value in _overlay_image_window(cfg).values()
    ) and source not in (cfg.image_index_sources or []):
        return None, None
    return start, end


def _overlay_slice(row, *, plan):
    from solar_toolkit.radio.time_matching import selection_metadata

    values = {}
    for source_id, selection in row.selections.items():
        if source_id.startswith("radio:") and selection.frame is not None:
            band, item, polarization, actual = selection.frame.payload
            values[band] = [(item, polarization, actual)]
    metadata = selection_metadata(row, plan=plan)
    from . import _source_map_selection as inputs

    for source_id, selection in row.selections.items():
        if selection.frame is None:
            continue
        item = (
            selection.frame.payload[1]
            if source_id.startswith("radio:")
            else selection.frame.payload
        )
        if getattr(item, "logical_frame", None) is not None:
            metadata["selections"][source_id]["locator"] = inputs._logical_locator(
                item.logical_frame
            )
            metadata["selections"][source_id]["paths"] = (
                list(map(str, item)) if isinstance(item, tuple) else [str(item)]
            )
    return RadioOverlaySlice(values, time_match=metadata)


def _build_overlay_plan(cfg, wavelengths):
    from solar_toolkit.radio.time_matching import index_radio_frames
    from solar_toolkit.time import (
        TimeFrame,
        TimeStream,
        build_sequence_match_plan,
        extract_timestamp_from_filename,
    )

    streams = _overlay_radio_streams(cfg)
    from . import _source_map_selection as inputs

    for wave in wavelengths:
        files = _fits_files_in_dir(
            _aia_panel_dir(cfg, wave),
            *_overlay_file_indices(
                cfg, "aia", cfg.aia_file_start_idx, cfg.aia_file_end_idx
            ),
        )
        frames = index_radio_frames(
            files, fallback_parser=extract_timestamp_from_filename
        )
        streams.append(
            TimeStream(
                f"aia:{wave}",
                [
                    TimeFrame(
                        frame.frame_id,
                        frame.time,
                        inputs._item_from_logical_frame(frame),
                    )
                    for frame in frames
                ],
                tolerance_seconds=float(cfg.aia_time_threshold_seconds),
            )
        )
    if cfg.overlay_hmi:
        files = _fits_files_in_dir(cfg.hmi_base_dir)
        frames = index_radio_frames(
            files, fallback_parser=extract_timestamp_from_filename
        )
        streams.append(
            TimeStream(
                "hmi",
                [
                    TimeFrame(
                        frame.frame_id,
                        frame.time,
                        inputs._item_from_logical_frame(frame),
                    )
                    for frame in frames
                ],
                tolerance_seconds=float(cfg.hmi_time_threshold) * 3600,
            )
        )
    if not any(stream.frames for stream in streams):
        return None
    plan = build_sequence_match_plan(streams, **_overlay_image_window(cfg))
    cfg._time_match_plan = plan.to_dict()
    return plan
