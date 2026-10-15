"""FITS observation indexing and strict radio-polarization pairing.

Sequence display matching belongs to ``solar_toolkit.time``.  This adapter
keeps the original FITS time, image identity, and physical pairing checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from astropy.io import fits

from solar_toolkit.time import (
    TimeFrame,
    TimeStamp,
    filter_time_frames,
    monotone_one_to_one,
    normalize_time,
)

__all__ = [
    "compatible_radio_payloads",
    "frame_for_item",
    "index_radio_frames",
    "materialize_radio_frame",
    "observation_stamp",
    "pair_radio_frames",
    "selection_metadata",
]

_GRID_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
    "CROTA2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
)


def observation_stamp(header, *, path=None, fallback_parser=None) -> TimeStamp:
    """Read a complete observation timestamp, then an explicit filename fallback."""

    for key in ("T_OBS", "DATE-OBS", "DATE_OBS"):
        value = str(header.get(key, "")).strip().strip("'")
        if not value:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            clock = str(header.get("TIME-OBS", header.get("TIME_OBS", ""))).strip()
            if not clock:
                continue
            value = f"{value}T{clock}"
        if not (
            re.search(r"\d{2}:\d{2}:\d{2}", value)
            or re.match(r"\d{14}", value)
            or re.search(r"T\d{6}", value)
        ):
            continue
        scale = (
            None
            if re.search(r"(?:_(?:TAI|UTC)|\s+(?:TAI|UTC)|Z)$", value, re.IGNORECASE)
            else header.get("TIMESYS")
        )
        try:
            return normalize_time(value, source=key, scale=scale)
        except (ValueError, TypeError):
            continue
    if path is not None and fallback_parser is not None:
        value = fallback_parser(str(path))
        if value is not None:
            return normalize_time(value, source="filename")
    raise ValueError(f"No complete observation timestamp: {path or 'FITS header'}")


def index_radio_frames(
    paths: Iterable[str | Path],
    *,
    frequency_mhz=None,
    polarization=None,
    fallback_parser: Callable | None = None,
    strict: bool = True,
) -> list[TimeFrame]:
    """Index logical image planes using explicit frequency/Stokes axes.

    Non-singleton unsupported axes (including time cubes without explicit
    per-plane timestamps) are rejected rather than counted as one observation.
    ``plane_indices`` index every non-spatial numpy axis in array order.
    """
    from itertools import product

    from .centers import (
        POL_UNKNOWN,
        infer_parent_directory_polarization,
        infer_polarization,
        normalize_pol_text,
        parse_frequency_mhz,
        stokes_code_to_pol,
    )

    output = []
    for path_value in sorted({Path(value) for value in paths}, key=str):
        path = Path(path_value)
        try:
            snapshot = path.stat()
            with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
                frequency_table = ()
                if "FMHZ" in hdus:
                    values = hdus["FMHZ"].data
                    if getattr(values.dtype, "names", None):
                        values = values[values.dtype.names[0]]
                    frequency_table = tuple(
                        float(value) for value in np.asarray(values).ravel()
                    )
                image_found = False
                for hdu_index, hdu in enumerate(hdus):
                    if not isinstance(
                        hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
                    ):
                        continue
                    dimensions = int(hdu.header.get("NAXIS", 0))
                    if dimensions < 2:
                        continue
                    header = hdus[0].header.copy()
                    if hdu_index != 0:
                        header.update(hdu.header)
                    if any(
                        str(header.get(f"CTYPE{axis}", ""))
                        .upper()
                        .startswith(("FREQ", "TIME", "STOKES", "WAVE", "ENER"))
                        for axis in (1, 2)
                    ):
                        # Dynamic spectra are sampled signals, not image frames.
                        continue
                    image_found = True
                    stamp = observation_stamp(
                        header, path=path, fallback_parser=fallback_parser
                    )
                    axes = list(range(dimensions, 2, -1))
                    sizes = [int(header.get(f"NAXIS{axis}", 1)) for axis in axes]
                    frequency_axes = [
                        axis
                        for axis in axes
                        if str(header.get(f"CTYPE{axis}", ""))
                        .upper()
                        .startswith("FREQ")
                    ]
                    active_frequency_axes = [
                        axis
                        for axis in frequency_axes
                        if int(header.get(f"NAXIS{axis}", 1)) > 1
                    ]
                    if len(active_frequency_axes) > 1:
                        raise ValueError(f"Ambiguous multiple frequency axes: {path}")
                    wcs_frequency_axis = next(
                        iter(active_frequency_axes or frequency_axes), None
                    )
                    table_axis = None
                    if frequency_table and hdu_index == 0:
                        possible = [
                            axis
                            for axis in axes
                            if int(header.get(f"NAXIS{axis}", 1))
                            == len(frequency_table)
                        ]
                        if len(possible) == 1:
                            table_axis = possible[0]
                        elif len(frequency_table) > 1:
                            raise ValueError(f"Ambiguous FMHZ image axis: {path}")
                    for axis, size in zip(axes, sizes, strict=True):
                        kind = str(header.get(f"CTYPE{axis}", "")).upper()
                        if (
                            table_axis is not None
                            and axis != table_axis
                            and size > 1
                            and kind.startswith("FREQ")
                        ):
                            raise ValueError(
                                f"Ambiguous multiple frequency axes: {path}"
                            )
                        if (
                            size > 1
                            and axis != table_axis
                            and not kind.startswith(("FREQ", "STOKES"))
                        ):
                            raise ValueError(
                                f"Unsupported non-singleton image axis {axis} ({kind}): {path}"
                            )
                    for indices in product(*(range(size) for size in sizes)):
                        indexed = dict(zip(axes, indices, strict=True))
                        freq = parse_frequency_mhz(path, header)
                        freq = float(freq) if np.isfinite(freq) else None
                        pol = infer_polarization(path, header)
                        parent_pol = infer_parent_directory_polarization(path)
                        plane_index = None
                        for axis, pixel in indexed.items():
                            kind = str(header.get(f"CTYPE{axis}", "")).upper()
                            coordinate = float(header.get(f"CRVAL{axis}", 0)) + (
                                pixel + 1 - float(header.get(f"CRPIX{axis}", 1))
                            ) * float(header.get(f"CDELT{axis}", 1))
                            if axis == table_axis:
                                plane_index = pixel
                                freq = frequency_table[pixel]
                            elif axis == wcs_frequency_axis and table_axis is None:
                                unit = str(header.get(f"CUNIT{axis}", "Hz")).lower()
                                factor = {
                                    "hz": 1e-6,
                                    "khz": 0.001,
                                    "mhz": 1.0,
                                    "ghz": 1000.0,
                                }.get(unit)
                                if factor is None:
                                    raise ValueError(f"Unknown frequency unit: {unit}")
                                freq = coordinate * factor
                            elif kind.startswith("STOKES"):
                                if (
                                    int(header.get(f"NAXIS{axis}", 1)) > 1
                                    or parent_pol == POL_UNKNOWN
                                ):
                                    pol = stokes_code_to_pol(coordinate)
                        if polarization is not None and pol != normalize_pol_text(
                            polarization
                        ):
                            continue
                        if frequency_mhz is not None:
                            if freq is None:
                                continue
                            if frequency_table:
                                nearest = int(
                                    np.argmin(
                                        np.abs(
                                            np.asarray(frequency_table)
                                            - float(frequency_mhz)
                                        )
                                    )
                                )
                                if (
                                    plane_index != nearest
                                    or abs(freq - float(frequency_mhz)) > 5.0
                                ):
                                    continue
                            elif not np.isclose(
                                freq, frequency_mhz, rtol=1e-5, atol=1e-6
                            ):
                                continue
                        identity = f"{path}#hdu={hdu_index};planes={','.join(map(str, indices))};size={snapshot.st_size};mtime_ns={snapshot.st_mtime_ns}"
                        header_cards = tuple(
                            (card.keyword, card.value, card.comment)
                            for card in header.cards
                        )
                        header_values = {
                            key: list(header[key])
                            if key in {"COMMENT", "HISTORY"}
                            else header[key]
                            for key in header
                        }
                        payload = {
                            "frame_id": identity,
                            "path": str(path),
                            "paths": (str(path),),
                            "hdu_index": hdu_index,
                            "header": header_values,
                            "header_cards": header_cards,
                            "shape": (
                                int(header.get("NAXIS2", 0)),
                                int(header.get("NAXIS1", 0)),
                            ),
                            "frequency_mhz": freq,
                            "frequencies_mhz": frequency_table,
                            "polarization": pol,
                            "time_source": stamp.source,
                            "plane_index": plane_index,
                            "plane_indices": tuple(indices),
                            "size_bytes": snapshot.st_size,
                            "mtime_ns": snapshot.st_mtime_ns,
                        }
                        output.append(TimeFrame(identity, stamp, payload))
                if not image_found:
                    raise ValueError(f"No image HDU: {path}")
        except (OSError, ValueError, TypeError, KeyError):
            if strict:
                raise
    return sorted(output, key=lambda frame: (frame.time.tai_ns, frame.frame_id))


def compatible_radio_payloads(left, right) -> bool:
    """Require matching pixels, WCS, units and observing frequency for RR+LL."""

    if tuple(left.get("shape", ())) != tuple(right.get("shape", ())):
        return False
    lh, rh = left.get("header", {}), right.get("header", {})
    if (
        str(lh.get("BUNIT", "")).strip().casefold()
        != str(rh.get("BUNIT", "")).strip().casefold()
    ):
        return False
    if any(lh.get(key) != rh.get(key) for key in _GRID_KEYS):
        return False
    lf, rf = left.get("frequency_mhz"), right.get("frequency_mhz")
    if lf is None or rf is None:
        return False
    return bool(np.isclose(lf, rf, rtol=1e-5, atol=1e-6))


def pair_radio_frames(
    left, right, *, tolerance_seconds, time_start=None, time_end=None
) -> list[TimeFrame]:
    """Filter each original polarization first, then maximize strict pairs."""

    left, right = list(left), list(right)
    identities = [frame.frame_id for frame in (*left, *right)]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "An original polarization frame may appear only once in pairing inputs"
        )
    left = filter_time_frames(left, time_start=time_start, time_end=time_end)
    right = filter_time_frames(right, time_start=time_start, time_end=time_end)
    matches = monotone_one_to_one(
        left,
        right,
        tolerance_seconds=tolerance_seconds,
        compatibility=lambda a, b: compatible_radio_payloads(a.payload, b.payload),
    )
    output = []
    for match in matches:
        a, b = match.left, match.right
        midpoint = TimeStamp.from_tai_ns(
            (a.time.tai_ns + b.time.tai_ns) // 2, source="RR+LL midpoint"
        )
        payload = dict(a.payload)
        payload.update(
            {
                "paths": tuple(a.payload["paths"]) + tuple(b.payload["paths"]),
                "actual_times_utc": (a.time.utc, b.time.utc),
                "actual_times_tai_ns": (a.time.tai_ns, b.time.tai_ns),
                "pair_delta_ns": match.delta_ns,
                "polarization": "L+R",
                "pairing_algorithm_version": 2,
                "pair_members": (a.payload, b.payload),
            }
        )
        output.append(
            TimeFrame(
                f"{a.frame_id}+{b.frame_id}",
                midpoint,
                payload,
                member_times=(a.time, b.time),
            )
        )
    return output


def frame_for_item(item, *, fallback_parser=None) -> TimeFrame:
    """Wrap an existing path or validated polarization pair for sequence display."""

    paths = tuple(item) if isinstance(item, (tuple, list)) else (item,)
    members = []
    for path in paths:
        indexed = index_radio_frames([path], fallback_parser=fallback_parser)
        if len(indexed) != 1:
            raise ValueError(f"Expected one image observation per input: {path}")
        members.append(indexed[0])
    if len(members) == 1:
        frame = members[0]
        return TimeFrame(frame.frame_id, frame.time, item)
    stamp = TimeStamp.from_tai_ns(
        sum(frame.time.tai_ns for frame in members) // len(members),
        source="polarization pair midpoint",
    )
    return TimeFrame(
        "+".join(frame.frame_id for frame in members),
        stamp,
        item,
        member_times=tuple(frame.time for frame in members),
    )


def selection_metadata(row, *, source_ids=None, plan=None) -> dict:
    """JSON-safe per-row provenance consumed by radio rendering and frontends."""

    edges = row.to_dict().get("parent_matches", {})
    hierarchy = list(plan.source_order) if plan is not None else []
    if not hierarchy and edges:
        hierarchy = [next(iter(edges.values()))["parent_source_id"], *edges]
    if not hierarchy:
        hierarchy = [
            source for source, item in row.selections.items() if item.frame is not None
        ]
    root_source = (
        plan.master_source_id
        if plan is not None
        else (hierarchy[0] if hierarchy else None)
    )
    output_source = (
        plan.output_source_id
        if plan is not None
        else (hierarchy[-1] if hierarchy else None)
    )
    output = row.output_frame or row.master_frame
    selected = {}
    for source_id, selection in row.selections.items():
        if source_ids is not None and source_id not in source_ids:
            continue
        selected[source_id] = {
            "frame_id": selection.frame.frame_id if selection.frame else None,
            "actual_time_utc": selection.frame.time.utc + "Z"
            if selection.frame
            else None,
            "actual_time_tai_ns": selection.frame.time.tai_ns
            if selection.frame
            else None,
            "delta_ns": selection.delta_ns,
            "within_tolerance": selection.within_tolerance,
            "reused": selection.reused,
            "use_count": selection.use_count,
            "original_index": selection.original_index,
            "sorted_index": selection.sorted_index,
            "parent_match": edges.get(source_id),
        }
    return {
        "algorithm": "hierarchical_monotone_full_coverage_minimum_absolute_delta",
        "algorithm_version": 3,
        "master_source_id": root_source,
        "root_source_id": root_source,
        "output_source_id": output_source,
        "source_order": hierarchy,
        "master_time_utc": row.master_frame.time.utc + "Z",
        "master_time_tai_ns": row.master_frame.time.tai_ns,
        "master_frame_id": row.master_frame.frame_id,
        "output_frame_id": output.frame_id,
        "output_time_utc": row.time.utc + "Z",
        "output_time_tai_ns": row.time.tai_ns,
        "delta_reference": "output_frame",
        "parent_matches": edges,
        "selections": selected,
    }


def materialize_radio_frame(frame: TimeFrame):
    """Load the exact planned HDU/planes, rejecting changed source files."""

    def load(payload):
        if payload.get("pair_members"):
            values = [load(member) for member in payload["pair_members"]]
            return sum((value[0] for value in values[1:]), values[0][0].copy()), values[
                0
            ][1]
        path = Path(payload["path"])
        stat = path.stat()
        if (
            payload.get("size_bytes") != stat.st_size
            or payload.get("mtime_ns") != stat.st_mtime_ns
        ):
            raise ValueError(f"Planned radio source changed: {path}")
        indices = tuple(payload.get("plane_indices", ()))

        def read_plane(memmap):
            with fits.open(path, memmap=memmap) as hdus:
                hdu = hdus[payload["hdu_index"]]
                return np.asarray(
                    hdu.data[indices + (slice(None), slice(None))], dtype=float
                ).copy()

        try:
            pixels = read_plane(True)
        except ValueError as exc:
            if (
                "memory-mapped" not in str(exc)
                and "BZERO" not in str(exc)
                and "BSCALE" not in str(exc)
            ):
                raise
            pixels = read_plane(False)
        if pixels.ndim != 2 or pixels.shape != tuple(payload["shape"]):
            raise ValueError(f"Planned image plane shape changed: {path}")
        if payload.get("header_cards") is not None:
            header = fits.Header(list(payload["header_cards"]))
        else:
            header = fits.Header()
            for key, value in payload["header"].items():
                if key in {"COMMENT", "HISTORY"} and isinstance(value, (tuple, list)):
                    for line in value:
                        header.append((key, line))
                else:
                    header[key] = value
        header["NAXIS"] = 2
        if "WCSAXES" in header:
            header["WCSAXES"] = 2
        for key in list(header):
            if re.fullmatch(
                r"(?:NAXIS|CTYPE|CUNIT|CRVAL|CRPIX|CDELT|CROTA)(?:[3-9]|[1-9]\d+)", key
            ):
                del header[key]
        if payload.get("frequency_mhz") is not None:
            header["FREQ"] = payload["frequency_mhz"]
            header["FREQUNIT"] = "MHz"
        header["POLAR"] = payload["polarization"]
        return pixels, header

    pixels, header = load(frame.payload)
    header["POLAR"] = frame.payload["polarization"]
    header["DATE-OBS"] = frame.time.utc
    if "T_OBS" in header:
        header["T_OBS"] = frame.time.utc
    header["TIMESYS"] = "UTC"
    return pixels, header
