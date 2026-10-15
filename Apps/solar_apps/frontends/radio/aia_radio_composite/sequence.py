"""Adapt actual image catalogs to the shared immutable sequence plan."""

from __future__ import annotations

from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
)


def radio_source_id(frequency):
    return f"radio:{float(frequency):g}"


def aia_source_id(wavelength):
    return f"aia:{int(wavelength)}"


def build_image_sequence_plan(
    radio_candidates,
    aia_catalog,
    *,
    wavelengths,
    aia_tolerance_seconds,
    radio_tolerance_seconds,
    time_start=None,
    time_end=None,
):
    """Keep every logical image within the explicitly selected half-open window."""
    streams = []
    for frequency, candidates in radio_candidates.items():
        frames = []
        for index, candidate in enumerate(candidates):
            stamp = getattr(candidate, "time_stamp", None)
            if stamp is None:
                if candidate.obs_time is None:
                    continue
                stamp = normalize_time(candidate.obs_time, source="radio observation")
            logical = getattr(candidate, "logical_frames", ())
            identity = "+".join(frame.frame_id for frame in logical)
            if not identity:
                paths = getattr(candidate, "paths", ())
                identity = "+".join(str(path) for path in paths)
                identity += f"#hdu={getattr(candidate, 'hdu_index', 0)};frequency={frequency};index={index}"
            if logical:
                identity += ";versions=" + ",".join(
                    f"{frame.payload.get('size_bytes')}:{frame.payload.get('mtime_ns')}"
                    for frame in logical
                )
            frames.append(TimeFrame(identity, stamp, candidate))
        streams.append(
            TimeStream(radio_source_id(frequency), frames, radio_tolerance_seconds)
        )
    for wavelength in wavelengths:
        frames = []
        if aia_catalog is not None and not aia_catalog.empty:
            for index, row in aia_catalog.iterrows():
                try:
                    if abs(float(row["wavelength"]) - wavelength) > 1e-6:
                        continue
                    quality = row.get("quality")
                    if (
                        quality is not None
                        and str(quality) not in ("nan", "None")
                        and int(quality) != 0
                    ):
                        continue
                    exposure = row.get("exposure_seconds")
                    if exposure is not None and str(exposure) not in ("nan", "None"):
                        import math

                        if not math.isfinite(float(exposure)) or float(exposure) <= 0:
                            continue
                    stamp = normalize_time(
                        row.get("utc") or row["obs_time"],
                        source=row.get("time_source", "AIA observation"),
                    )
                    payload = row.to_dict()
                    identity = f"{row['path']}#hdu={row.get('hdu_index', 0)}"
                    identity += (
                        f";version={row.get('size_bytes')}:{row.get('mtime_ns')}"
                    )
                    frames.append(TimeFrame(identity, stamp, payload))
                except TypeError, ValueError:
                    continue
        streams.append(
            TimeStream(aia_source_id(wavelength), frames, aia_tolerance_seconds)
        )
    return build_sequence_match_plan(streams, time_start=time_start, time_end=time_end)


def match_row_metadata(row):
    from solar_toolkit.radio.time_matching import selection_metadata

    metadata = selection_metadata(row)
    return {
        **metadata,
        "sources": {
            source: {
                "frame_id": item.frame.frame_id if item.frame else None,
                "actual_time_utc": item.frame.time.utc if item.frame else None,
                "delta_ns": item.delta_ns,
                "within_tolerance": item.within_tolerance,
                "use_count": item.use_count,
                "reused": item.reused,
                "sorted_index": item.sorted_index,
                "status": (
                    "missing"
                    if item.frame is None
                    else ("matched" if item.within_tolerance else "outside_tolerance")
                ),
                "parent_match": metadata["selections"][source]["parent_match"],
            }
            for source, item in row.selections.items()
        },
    }
