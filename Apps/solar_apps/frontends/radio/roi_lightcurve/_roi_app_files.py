"""Lightweight FITS manifests and reference-image planning and decoding.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
    RadioImage,
    infer_polarization,
    iter_radio_images,
    parse_datetime_value,
    parse_frequency_mhz,
    parse_time_from_filename,
    select_radio_files,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
)
from ._roi_app_helpers import _roi_app_facade


from ._roi_app_state import (
    ROI_KEYS,
    ANALYSIS_KEYS,
    EXPORT_KEYS,
    REFERENCE_KEYS,
    _REFERENCE_DECODER_VERSION,
    _REFERENCE_PLANE_CACHE_SIZE,
    _NAT_INT64,
    _ReferencePlan,
)


def build_file_manifest(
    radio_dir: str | Path,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
    freqs: list[float] | tuple[float, ...] | None = None,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
) -> pd.DataFrame:
    """Build a lightweight path manifest without loading FITS image arrays."""

    folder = Path(radio_dir).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Radio data folder does not exist: {folder}")
    files = select_radio_files(
        folder,
        pattern=pattern,
        recursive=recursive,
        time_start=time_start,
        time_end=time_end,
    )
    blank_header = fits.Header()
    freq_set = _normalize_frequency_selection(freqs)
    rows: list[dict[str, Any]] = []
    for path in files:
        freq_mhz = _parse_frequency_hint_mhz(path, blank_header)
        if freq_set and not _frequency_matches_any(freq_mhz, freq_set):
            continue
        stat = path.stat()
        obs_time = parse_time_from_filename(path)
        rows.append(
            {
                "row": len(rows) + 1,
                "path": str(path),
                "relative_path": _relative_path_text(path, folder),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "size_mib": round(float(stat.st_size) / 1024.0 / 1024.0, 3),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
                "inferred_freq_mhz": freq_mhz,
                "inferred_polarization": infer_polarization(path, blank_header),
                "inferred_obs_time": (
                    obs_time.isoformat(timespec="milliseconds") if obs_time else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def discover_frequency_options(
    radio_dir: str | Path,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
) -> pd.DataFrame:
    """Scan FITS paths and return available path-inferred frequency options."""

    folder = Path(radio_dir).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Radio data folder does not exist: {folder}")
    files = select_radio_files(folder, pattern=pattern, recursive=recursive)
    blank_header = fits.Header()
    counts: dict[float, int] = {}
    unknown = 0
    for path in files:
        freq_mhz = _parse_frequency_hint_mhz(path, blank_header)
        if not np.isfinite(freq_mhz):
            unknown += 1
            continue
        key = round(float(freq_mhz), 6)
        counts[key] = counts.get(key, 0) + 1
    rows = [
        {"freq_mhz": freq, "file_count": count}
        for freq, count in sorted(counts.items(), key=lambda item: item[0])
    ]
    result = pd.DataFrame(rows)
    result.attrs["unknown_frequency_count"] = unknown
    result.attrs["total_file_count"] = len(files)
    return result


def _load_manifest_into_state(
    st: Any, settings: dict[str, Any], path_policy: PathAccessPolicy
) -> None:
    try:
        radio_dir = path_policy.input_directory(settings["radio_dir"])
        manifest = build_file_manifest(
            radio_dir,
            pattern=settings["pattern"],
            recursive=bool(settings["recursive"]),
            freqs=settings.get("selected_freqs_mhz") or None,
        )
    except Exception as exc:  # noqa: BLE001 - visible app error.
        st.error(str(exc))
        return
    st.session_state["loaded_full_manifest"] = manifest
    st.session_state["loaded_source_signature"] = _source_signature(settings)
    _apply_loaded_time_filter(st, settings, clear_selection=True)


def _apply_loaded_time_filter(
    st: Any, settings: dict[str, Any], *, clear_selection: bool = False
) -> None:
    full_manifest = st.session_state.get("loaded_full_manifest")
    if full_manifest is None:
        st.warning("Load selected frequencies before applying a time range.")
        return
    try:
        manifest = _filter_manifest_by_time(
            full_manifest,
            time_start=settings.get("time_start") or None,
            time_end=settings.get("time_end") or None,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["loaded_manifest"] = manifest
    st.session_state["dataset_signature"] = _roi_app_facade()._dataset_signature(
        settings, manifest
    )
    if clear_selection:
        _roi_app_facade()._clear_keys(
            st,
            (
                "selected_paths",
                *REFERENCE_KEYS,
                *ROI_KEYS,
                *ANALYSIS_KEYS,
                *EXPORT_KEYS,
            ),
        )
    else:
        selected = set(st.session_state.get("selected_paths", []))
        valid = set(manifest["path"].astype(str)) if not manifest.empty else set()
        _set_selected_paths(st, selected & valid, manifest)
        _roi_app_facade()._clear_keys(
            st, (*REFERENCE_KEYS, *ROI_KEYS, *ANALYSIS_KEYS, *EXPORT_KEYS)
        )
    st.session_state["roi_chart_generation"] = (
        int(st.session_state.get("roi_chart_generation", 0)) + 1
    )
    st.success(f"Loaded {len(manifest):,} FITS files.")


def _set_selected_paths(
    st: Any, paths: set[str], manifest: pd.DataFrame | None = None
) -> None:
    selected = _order_paths_by_manifest(paths, manifest)
    previous = _order_paths_by_manifest(
        st.session_state.get("selected_paths", []), manifest
    )
    if selected == previous:
        return
    st.session_state["selected_paths"] = selected
    st.session_state["selection_revision"] = (
        int(st.session_state.get("selection_revision", 0)) + 1
    )
    _roi_app_facade()._clear_keys(
        st, (*REFERENCE_KEYS, *ROI_KEYS, *ANALYSIS_KEYS, *EXPORT_KEYS)
    )
    st.session_state["roi_chart_generation"] = (
        int(st.session_state.get("roi_chart_generation", 0)) + 1
    )


def _discover_frequencies_into_state(
    st: Any, settings: dict[str, Any], path_policy: PathAccessPolicy
) -> None:
    try:
        radio_dir = path_policy.input_directory(settings["radio_dir"])
        options = discover_frequency_options(
            radio_dir,
            pattern=settings["pattern"],
            recursive=bool(settings["recursive"]),
        )
    except Exception as exc:  # noqa: BLE001 - visible app error.
        st.error(str(exc))
        return
    st.session_state["frequency_options"] = options
    st.session_state["frequency_source_signature"] = _source_signature(settings)
    _roi_app_facade()._clear_keys(
        st,
        (
            "loaded_manifest",
            "loaded_full_manifest",
            "selected_paths",
            *REFERENCE_KEYS,
            *ROI_KEYS,
            *ANALYSIS_KEYS,
            *EXPORT_KEYS,
        ),
    )
    total = int(options.attrs.get("total_file_count", 0))
    unknown = int(options.attrs.get("unknown_frequency_count", 0))
    st.success(
        f"Discovered {len(options):,} frequency bands from {total:,} matching FITS files."
    )
    if unknown:
        st.warning(
            f"{unknown:,} matching files did not expose a frequency in the path or filename."
        )


def _source_signature(settings: dict[str, Any]) -> str:
    payload = {
        "radio_dir": str(settings.get("radio_dir", "")),
        "pattern": str(settings.get("pattern", "")),
        "recursive": bool(settings.get("recursive", True)),
    }
    return json.dumps(payload, sort_keys=True)


def _frequency_options_list(options: pd.DataFrame | None) -> list[float]:
    if options is None or options.empty or "freq_mhz" not in options.columns:
        return []
    return [float(item) for item in options["freq_mhz"].dropna().tolist()]


def _default_selected_frequencies(options: list[float], stored: Any) -> list[float]:
    if not options:
        return []
    requested = _normalize_frequency_selection(stored)
    if not requested:
        return options
    selected = [freq for freq in options if _frequency_matches_any(freq, requested)]
    return selected or options


def _normalize_frequency_selection(freqs: Any) -> set[float]:
    if freqs in (None, ""):
        return set()
    values = freqs if isinstance(freqs, (list, tuple, set)) else [freqs]
    selected: set[float] = set()
    for value in values:
        try:
            freq = float(value)
        except TypeError, ValueError:
            continue
        if np.isfinite(freq):
            selected.add(freq)
    return selected


def _frequency_matches_any(value: float, freqs: set[float]) -> bool:
    if not np.isfinite(value):
        return False
    return any(
        abs(float(value) - float(freq)) <= max(1e-6, abs(float(freq)) * 1e-5)
        for freq in freqs
    )


def _parse_frequency_hint_mhz(path: Path, header: fits.Header) -> float:
    for part in reversed(path.parts):
        match = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ghz|mhz|khz|hz)\b",
            part,
            flags=re.IGNORECASE,
        )
        if match:
            value = float(match.group("value"))
            unit = match.group("unit").lower()
            if unit == "ghz":
                return value * 1000.0
            if unit == "khz":
                return value / 1000.0
            if unit == "hz":
                return value / 1_000_000.0
            return value
    return parse_frequency_mhz(path, header)


def _manifest_time_range_hint(manifest: pd.DataFrame | None) -> dict[str, Any]:
    if (
        manifest is None
        or manifest.empty
        or "inferred_obs_time" not in manifest.columns
    ):
        return {}
    times = pd.to_datetime(
        manifest["inferred_obs_time"].replace("", pd.NA), errors="coerce"
    )
    timed = times.dropna()
    if timed.empty:
        return {
            "start": "",
            "end": "",
            "timed_count": 0,
            "untimed_count": len(manifest),
        }
    return {
        "start": timed.min().isoformat(timespec="milliseconds"),
        "end": timed.max().isoformat(timespec="milliseconds"),
        "timed_count": int(timed.size),
        "untimed_count": int(times.isna().sum()),
    }


def _filter_manifest_by_time(
    manifest: pd.DataFrame,
    *,
    time_start: str | datetime | None,
    time_end: str | datetime | None,
) -> pd.DataFrame:
    start = _parse_optional_time(time_start, "Start time")
    end = _parse_optional_time(time_end, "End time")
    if start is not None and end is not None and start > end:
        raise ValueError("Start time must be earlier than or equal to end time.")
    if start is None and end is None:
        return manifest.copy()
    times = pd.to_datetime(
        manifest["inferred_obs_time"].replace("", pd.NA), errors="coerce"
    )
    mask = times.notna()
    if start is not None:
        mask &= times >= pd.Timestamp(start)
    if end is not None:
        mask &= times <= pd.Timestamp(end)
    return manifest.loc[mask].copy()


def _parse_optional_time(value: str | datetime | None, label: str) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = parse_datetime_value(value)
    if parsed is None:
        raise ValueError(f"{label} is not a recognized time: {value}")
    return parsed


def parse_row_selection_expression(raw: str) -> list[int]:
    """Parse comma-separated File # values and inclusive ranges."""

    text = str(raw or "").strip()
    if not text:
        return []
    values: list[int] = []
    seen: set[int] = set()
    for token in [item.strip() for item in text.split(",") if item.strip()]:
        match = re.fullmatch(r"(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?", token)
        if not match:
            raise ValueError(f"Invalid File # token: {token!r}")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            raise ValueError(f"File # ranges must be increasing: {token!r}")
        for value in range(start, end + 1):
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def _apply_number_selection(
    st: Any,
    manifest: pd.DataFrame,
    filtered: pd.DataFrame,
    expression: str,
    action: str,
) -> None:
    try:
        numbers = parse_row_selection_expression(expression)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not numbers:
        st.warning("Enter one or more File # values, for example 1,3,8-20.")
        return
    row_to_path = dict(
        zip(filtered["row"].astype(int), filtered["path"].astype(str), strict=False)
    )
    selected_paths = {
        row_to_path[number] for number in numbers if number in row_to_path
    }
    skipped = [number for number in numbers if number not in row_to_path]
    current = set(st.session_state.get("selected_paths", []))
    if action == "Replace":
        updated = selected_paths
    elif action == "Remove":
        updated = current - selected_paths
    else:
        updated = current | selected_paths
    _set_selected_paths(st, updated, manifest)
    if skipped:
        st.warning(
            f"Skipped {len(skipped):,} File # values outside the current filtered table."
        )


def _order_paths_by_manifest(paths: Any, manifest: pd.DataFrame | None) -> list[str]:
    unique_paths = {str(path) for path in (paths or [])}
    if (
        manifest is None
        or manifest.empty
        or "path" not in manifest.columns
        or "row" not in manifest.columns
    ):
        return sorted(unique_paths)
    row_by_path = dict(
        zip(manifest["path"].astype(str), manifest["row"].astype(int), strict=False)
    )
    return sorted(unique_paths, key=lambda path: (row_by_path.get(path, 10**12), path))


def _filter_manifest(manifest: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return manifest
    text = query.strip().casefold()
    searchable = manifest[
        [
            "relative_path",
            "inferred_obs_time",
            "inferred_polarization",
            "inferred_freq_mhz",
        ]
    ].astype(str)
    mask = searchable.apply(
        lambda column: column.str.casefold().str.contains(text, regex=False)
    ).any(axis=1)
    return manifest.loc[mask].copy()


def _page_paths(filtered: pd.DataFrame, *, page: int, page_size: int) -> list[str]:
    start = (int(page) - 1) * int(page_size)
    end = start + int(page_size)
    return filtered.iloc[start:end]["path"].astype(str).tolist()


def _load_first_radio_image(path: str | Path) -> RadioImage:
    for item in iter_radio_images(path):
        return item
    raise RuntimeError(f"No usable 2D radio image plane found in {path}")


@lru_cache(maxsize=_REFERENCE_PLANE_CACHE_SIZE)
def _cached_first_radio_image(
    path: str,
    size_bytes: int,
    mtime_ns: int,
    decoder_version: str = _REFERENCE_DECODER_VERSION,
) -> RadioImage:
    del size_bytes, mtime_ns, decoder_version
    item = _load_first_radio_image(path)
    image = np.asarray(item.image, dtype=float)
    image.setflags(write=False)
    return RadioImage(
        path=Path(path),
        hdu_index=item.hdu_index,
        image=image,
        header=item.header.copy(),
        pol=item.pol,
        freq_mhz=float(item.freq_mhz),
        obs_time=item.obs_time,
        source_label=item.source_label,
    )


def _cached_reference_image(path: str | Path) -> RadioImage:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _cached_first_radio_image(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _clone_reference_image(item: RadioImage, *, source_label: str) -> RadioImage:
    return RadioImage(
        path=item.path,
        hdu_index=item.hdu_index,
        image=item.image,
        header=item.header.copy(),
        pol=item.pol,
        freq_mhz=float(item.freq_mhz),
        obs_time=item.obs_time,
        source_label=source_label,
    )


def _manifest_frequency_values(manifest: pd.DataFrame) -> list[float]:
    values = [
        float(item)
        for item in manifest.get("inferred_freq_mhz", pd.Series(dtype=float))
        .dropna()
        .unique()
        .tolist()
        if np.isfinite(float(item))
    ]
    return sorted(values)


def _load_reference_grid(
    available: pd.DataFrame,
    *,
    primary_frequency: float,
    anchor_number: int,
    preview_polarization: str,
    pair_tolerance_sec: float,
) -> tuple[list[RadioImage], list[dict[str, Any]]]:
    plans = _plan_reference_grid(
        available,
        primary_frequency=primary_frequency,
        anchor_number=anchor_number,
        preview_polarization=preview_polarization,
        pair_tolerance_sec=pair_tolerance_sec,
    )
    return _materialize_reference_grid(plans)


def _plan_reference_grid(
    available: pd.DataFrame,
    *,
    primary_frequency: float,
    anchor_number: int,
    preview_polarization: str,
    pair_tolerance_sec: float,
) -> list[_ReferencePlan]:
    if available.empty:
        return []
    working = available.copy()
    working["_freq_value"] = pd.to_numeric(
        working["inferred_freq_mhz"], errors="coerce"
    )
    finite_values = sorted(
        float(value)
        for value in working["_freq_value"].dropna().unique().tolist()
        if np.isfinite(float(value))
    )
    canonical: list[float] = []
    value_to_group: dict[float, float] = {}
    for value in finite_values:
        matched = next(
            (item for item in canonical if _frequency_matches_any(value, {item})),
            None,
        )
        if matched is None:
            matched = value
            canonical.append(value)
        value_to_group[value] = matched
    working["_freq_group"] = working["_freq_value"].map(value_to_group)
    working["_obs_time_parsed"] = pd.to_datetime(
        working["inferred_obs_time"],
        utc=True,
        errors="coerce",
    )
    anchor_rows = working.loc[working["row"].astype(int).eq(int(anchor_number))]
    if anchor_rows.empty:
        raise ValueError("Anchor File # must be one of the selected rows.")
    anchor_time = _row_time(anchor_rows.iloc[0])
    primary_group = next(
        (
            freq
            for freq in canonical
            if _frequency_matches_any(freq, {primary_frequency})
        ),
        None,
    )
    ordered_freqs = ([primary_group] if primary_group is not None else []) + [
        freq for freq in canonical if freq != primary_group
    ]
    grouped = {
        float(freq): group.copy()
        for freq, group in working.dropna(subset=["_freq_group"]).groupby(
            "_freq_group",
            sort=False,
        )
    }
    plans: list[_ReferencePlan] = []
    for freq in ordered_freqs:
        same_freq = grouped.get(float(freq))
        if same_freq is None or same_freq.empty:
            continue
        row: pd.Series | None = None
        paired_row: pd.Series | None = None
        if preview_polarization == POL_SUM:
            left_rows = same_freq.loc[
                same_freq["inferred_polarization"].astype(str).eq(POL_LCP)
            ]
            right_rows = same_freq.loc[
                same_freq["inferred_polarization"].astype(str).eq(POL_RCP)
            ]
            row, paired_row = _select_paired_rows(
                left_rows,
                right_rows,
                anchor_time=anchor_time,
                pair_tolerance_sec=pair_tolerance_sec,
            )
        if row is None:
            candidates = same_freq
            if preview_polarization in {POL_LCP, POL_RCP}:
                filtered = same_freq.loc[
                    same_freq["inferred_polarization"]
                    .astype(str)
                    .eq(preview_polarization)
                ]
                if not filtered.empty:
                    candidates = filtered
            row = _nearest_manifest_row(candidates, anchor_time)
            paired_row = None
        if row is None:
            continue
        row_time = _row_time(row)
        delta_sec = (
            abs((row_time - anchor_time).total_seconds())
            if row_time is not None and anchor_time is not None
            else math.nan
        )
        plans.append(
            _ReferencePlan(
                freq_mhz=float(freq),
                row=int(row["row"]),
                path=str(row["path"]),
                paired_row=int(paired_row["row"]) if paired_row is not None else None,
                paired_path=str(paired_row["path"]) if paired_row is not None else None,
                obs_time=str(row.get("inferred_obs_time", "")),
                delta_from_anchor_sec=float(delta_sec),
                polarization=(
                    POL_SUM
                    if paired_row is not None
                    else str(row.get("inferred_polarization", preview_polarization))
                ),
            )
        )
    return plans


def _select_paired_rows(
    left_rows: pd.DataFrame,
    right_rows: pd.DataFrame,
    *,
    anchor_time: datetime | None,
    pair_tolerance_sec: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    if left_rows.empty or right_rows.empty:
        return None, None
    left_ns = _manifest_time_ns(left_rows)
    right_ns = _manifest_time_ns(right_rows)
    left_positions = np.flatnonzero(left_ns != _NAT_INT64)
    right_positions = np.flatnonzero(right_ns != _NAT_INT64)
    if not left_positions.size or not right_positions.size:
        return None, None
    right_row_numbers = pd.to_numeric(right_rows["row"], errors="coerce").to_numpy(
        dtype=float
    )
    order = np.lexsort((right_row_numbers[right_positions], right_ns[right_positions]))
    sorted_right_positions = right_positions[order]
    sorted_right_ns = right_ns[sorted_right_positions]
    insertion = np.searchsorted(sorted_right_ns, left_ns[left_positions])
    candidate_slots = np.stack(
        (
            np.clip(insertion - 1, 0, len(sorted_right_ns) - 1),
            np.clip(insertion, 0, len(sorted_right_ns) - 1),
        ),
        axis=1,
    )
    candidate_right_positions = sorted_right_positions[candidate_slots]
    pair_delta_ns = np.abs(
        right_ns[candidate_right_positions] - left_ns[left_positions, None]
    )
    tolerance_ns = max(0, int(round(float(pair_tolerance_sec) * 1_000_000_000.0)))
    valid = pair_delta_ns <= tolerance_ns
    if not np.any(valid):
        return None, None
    if anchor_time is None:
        anchor_delta_ns = np.zeros(left_positions.size, dtype=np.int64)
    else:
        anchor_ns = int(pd.Timestamp(anchor_time).value)
        anchor_delta_ns = np.abs(left_ns[left_positions] - anchor_ns)
    score = anchor_delta_ns[:, None] + pair_delta_ns
    valid_left_slots, valid_candidate_slots = np.nonzero(valid)
    selected_left_positions = left_positions[valid_left_slots]
    selected_right_positions = candidate_right_positions[
        valid_left_slots,
        valid_candidate_slots,
    ]
    left_row_numbers = pd.to_numeric(left_rows["row"], errors="coerce").to_numpy(
        dtype=float
    )
    ranking = np.lexsort(
        (
            right_row_numbers[selected_right_positions],
            left_row_numbers[selected_left_positions],
            score[valid_left_slots, valid_candidate_slots],
        )
    )
    best = int(ranking[0])
    return (
        left_rows.iloc[int(selected_left_positions[best])],
        right_rows.iloc[int(selected_right_positions[best])],
    )


def _materialize_reference_grid(
    plans: list[_ReferencePlan],
) -> tuple[list[RadioImage], list[dict[str, Any]]]:
    references: list[RadioImage] = []
    metadata: list[dict[str, Any]] = []
    for plan in plans:
        left_item = _cached_reference_image(plan.path)
        paired_row = plan.paired_row
        paired_path = plan.paired_path
        if paired_path:
            right_item = _cached_reference_image(paired_path)
            if left_item.image.shape == right_item.image.shape:
                image = np.asarray(left_item.image, dtype=float) + np.asarray(
                    right_item.image,
                    dtype=float,
                )
                image.setflags(write=False)
                item = RadioImage(
                    path=Path(plan.path),
                    hdu_index=left_item.hdu_index,
                    image=image,
                    header=left_item.header.copy(),
                    pol=POL_SUM,
                    freq_mhz=float(left_item.freq_mhz),
                    obs_time=left_item.obs_time or right_item.obs_time,
                    source_label=f"File #{plan.row} + #{paired_row} LCP+RCP preview",
                )
            else:
                paired_row = None
                paired_path = None
                item = _clone_reference_image(
                    left_item,
                    source_label=f"File #{plan.row}",
                )
        else:
            item = _clone_reference_image(
                left_item,
                source_label=f"File #{plan.row}",
            )
        references.append(item)
        metadata.append(
            {
                "freq_mhz": float(plan.freq_mhz),
                "file_number": int(plan.row),
                "path": str(plan.path),
                "paired_file_number": paired_row,
                "paired_path": str(paired_path or ""),
                "obs_time": plan.obs_time,
                "delta_from_anchor_sec": float(plan.delta_from_anchor_sec),
                "polarization": item.pol,
            }
        )
    return references, metadata


def _reference_grid_signature(
    available: pd.DataFrame,
    plans: list[_ReferencePlan],
    *,
    primary_frequency: float,
    anchor_number: int,
    preview_polarization: str,
    pair_tolerance_sec: float,
) -> str:
    digest = hashlib.sha256()
    request = (
        "reference-grid-v2",
        f"{float(primary_frequency):.12g}",
        str(int(anchor_number)),
        str(preview_polarization),
        f"{float(pair_tolerance_sec):.9f}",
    )
    digest.update("|".join(request).encode("utf-8"))
    digest.update(b"\n")
    columns = [
        "row",
        "path",
        "inferred_freq_mhz",
        "inferred_polarization",
        "inferred_obs_time",
    ]
    for values in available.sort_values("row")[columns].itertuples(
        index=False, name=None
    ):
        digest.update("|".join(str(value) for value in values).encode("utf-8"))
        digest.update(b"\n")
    for plan in plans:
        for path in (plan.path, plan.paired_path):
            if not path:
                continue
            resolved = Path(path).expanduser().resolve()
            stat = resolved.stat()
            digest.update(f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _reference_reuse_signature(
    st: Any,
    *,
    primary_frequency: float,
    anchor_number: int,
    preview_polarization: str,
    pair_tolerance_sec: float,
) -> str:
    digest = hashlib.sha256()
    request = (
        "reference-reuse-v1",
        str(st.session_state.get("dataset_signature", "")),
        str(int(st.session_state.get("selection_revision", 0))),
        f"{float(primary_frequency):.12g}",
        str(int(anchor_number)),
        str(preview_polarization),
        f"{float(pair_tolerance_sec):.9f}",
    )
    digest.update("|".join(request).encode())
    digest.update(b"\n")
    metadata = list(st.session_state.get("reference_metadata") or [])
    for meta in metadata:
        for path in (meta.get("path"), meta.get("paired_path")):
            if not path:
                continue
            resolved = Path(path).expanduser().resolve()
            try:
                stat = resolved.stat()
                identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
            except OSError:
                identity = f"{resolved}|missing"
            digest.update(identity.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _nearest_manifest_row(
    candidates: pd.DataFrame, anchor_time: datetime | None
) -> pd.Series | None:
    if candidates.empty:
        return None
    if anchor_time is None:
        return candidates.sort_values("row").iloc[0]
    scored = candidates.copy()
    time_ns = _manifest_time_ns(scored)
    anchor_ns = int(pd.Timestamp(anchor_time).value)
    time_delta = np.full(time_ns.shape, np.inf, dtype=float)
    valid = time_ns != _NAT_INT64
    time_delta[valid] = np.abs(time_ns[valid] - anchor_ns) / 1_000_000_000.0
    scored["_time_delta"] = time_delta
    return scored.sort_values(["_time_delta", "row"]).iloc[0]


def _manifest_time_ns(frame: pd.DataFrame) -> np.ndarray:
    source = (
        frame["_obs_time_parsed"]
        if "_obs_time_parsed" in frame.columns
        else pd.to_datetime(frame["inferred_obs_time"], utc=True, errors="coerce")
    )
    return (
        pd.to_datetime(source, utc=True, errors="coerce")
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )


def _row_time(row: pd.Series) -> datetime | None:
    value = row.get("inferred_obs_time", "")
    parsed = parse_datetime_value(value)
    return parsed


def _relative_path_text(path: Path, folder: Path) -> str:
    try:
        return str(path.relative_to(folder))
    except ValueError:
        return str(path)


__all__ = [
    "build_file_manifest",
    "discover_frequency_options",
    "_load_manifest_into_state",
    "_apply_loaded_time_filter",
    "_set_selected_paths",
    "_discover_frequencies_into_state",
    "_source_signature",
    "_frequency_options_list",
    "_default_selected_frequencies",
    "_normalize_frequency_selection",
    "_frequency_matches_any",
    "_parse_frequency_hint_mhz",
    "_manifest_time_range_hint",
    "_filter_manifest_by_time",
    "_parse_optional_time",
    "parse_row_selection_expression",
    "_apply_number_selection",
    "_order_paths_by_manifest",
    "_filter_manifest",
    "_page_paths",
    "_load_first_radio_image",
    "_cached_first_radio_image",
    "_cached_reference_image",
    "_clone_reference_image",
    "_manifest_frequency_values",
    "_load_reference_grid",
    "_plan_reference_grid",
    "_select_paired_rows",
    "_materialize_reference_grid",
    "_reference_grid_signature",
    "_reference_reuse_signature",
    "_nearest_manifest_row",
    "_manifest_time_ns",
    "_row_time",
    "_relative_path_text",
]
