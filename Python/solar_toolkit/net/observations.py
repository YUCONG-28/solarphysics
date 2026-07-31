"""Unified remote-observation search and download contracts.

This module is intentionally UI-free.  It provides the reusable provider
registry used by App 1.0 while keeping all downloaded observations below an
explicit caller-selected root.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
MAX_QUERY_SECONDS = 24 * 60 * 60
MAX_QUERY_RECORDS = 10_000
JSOC_BASE_URL = "https://jsoc1.stanford.edu"
STEREO_BASE_URL = "https://stereo-ssc.nascom.nasa.gov/data/ins_data"
SOAR_DATA_URL = "http://soar.esac.esa.int/soar-sl-tap/data"
_RECORD_PROVIDERS = {
    "sdo-aia-euv": "jsoc",
    "sdo-aia-uv": "jsoc",
    "sdo-hmi-los": "jsoc",
    "stereo-euvi": "ssc",
    "soho-lasco": "sdac",
    "goes-suvi": "noaa",
    "solar-orbiter-eui": "soar",
}
_PROVIDER_HOSTS = {
    "jsoc": "jsoc1.stanford.edu",
    "ssc": "stereo-ssc.nascom.nasa.gov",
    "noaa": "data.ngdc.noaa.gov",
    "soar": "soar.esac.esa.int",
}


@dataclass(frozen=True, slots=True)
class ObservationProductSpec:
    """One selectable remote observation product."""

    product_id: str
    title: str
    provider: str
    mission: str
    instrument: str
    spacecraft: tuple[str, ...] = ()
    detectors: tuple[str, ...] = ()
    wavelengths: tuple[int, ...] = ()
    levels: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("fits",)
    search_capability: str = "remote-metadata"
    url_resolution_capability: str = "direct"

    @property
    def source_id(self) -> str:
        """Stable source ID used by UI and workflow registries."""

        return self.product_id


PRODUCTS: dict[str, ObservationProductSpec] = {
    "sdo-aia-euv": ObservationProductSpec(
        "sdo-aia-euv",
        "SDO/AIA EUV",
        "jsoc",
        "sdo",
        "aia",
        spacecraft=("sdo",),
        wavelengths=(94, 131, 171, 193, 211, 304, 335),
        levels=("1",),
    ),
    "sdo-aia-uv": ObservationProductSpec(
        "sdo-aia-uv",
        "SDO/AIA UV",
        "jsoc",
        "sdo",
        "aia",
        spacecraft=("sdo",),
        wavelengths=(1600, 1700),
        levels=("1",),
    ),
    "sdo-hmi-los": ObservationProductSpec(
        "sdo-hmi-los",
        "SDO/HMI line-of-sight magnetogram",
        "jsoc",
        "sdo",
        "hmi",
        spacecraft=("sdo",),
        detectors=("magnetogram",),
        levels=("1.5",),
    ),
    "stereo-euvi": ObservationProductSpec(
        "stereo-euvi",
        "STEREO/SECCHI EUVI",
        "vso-stereo",
        "stereo",
        "euvi",
        spacecraft=("stereo-a", "stereo-b"),
        detectors=("euvi",),
        wavelengths=(171, 195, 284, 304),
        levels=("0.5",),
        formats=("fts",),
    ),
    "soho-lasco": ObservationProductSpec(
        "soho-lasco",
        "SOHO/LASCO",
        "vso-lasco",
        "soho",
        "lasco",
        spacecraft=("soho",),
        detectors=("c2", "c3", "c1"),
        levels=("0.5",),
        formats=("fts",),
        url_resolution_capability="vso-getdata-url-file",
    ),
    "goes-suvi": ObservationProductSpec(
        "goes-suvi",
        "GOES/SUVI Level 2 composite",
        "noaa-suvi",
        "goes",
        "suvi",
        spacecraft=("goes16", "goes18"),
        wavelengths=(94, 131, 171, 195, 284, 304),
        levels=("2",),
    ),
    "solar-orbiter-eui": ObservationProductSpec(
        "solar-orbiter-eui",
        "Solar Orbiter/EUI",
        "soar",
        "solar-orbiter",
        "eui",
        spacecraft=("solar-orbiter",),
        levels=("L1", "L2", "L3"),
    ),
}


def _utc(value: str | dt.datetime, *, label: str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        rendered = str(value).strip().replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(rendered)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _clean_choices(
    values: Sequence[str], allowed: Sequence[str], *, label: str
) -> tuple[str, ...]:
    clean = tuple(dict.fromkeys(str(item).strip().lower() for item in values if item))
    unknown = sorted(set(clean) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {', '.join(unknown)}")
    return clean


@dataclass(frozen=True, slots=True)
class ObservationQueryV1:
    """Versioned request for one product family."""

    query_id: str
    product_id: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    spacecraft: tuple[str, ...] = ()
    detectors: tuple[str, ...] = ()
    wavelengths: tuple[int, ...] = ()
    level: str | None = None
    sample_seconds: int | None = None
    source_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported observation-query schema")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.query_id):
            raise ValueError("query_id must be lowercase kebab-case")
        try:
            spec = PRODUCTS[self.product_id]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported observation product: {self.product_id}"
            ) from exc
        if self.source_id not in (None, spec.provider):
            raise ValueError(
                f"Product {self.product_id} requires source {spec.provider}"
            )
        start = _utc(self.start_utc, label="start_utc")
        end = _utc(self.end_utc, label="end_utc")
        if end <= start:
            raise ValueError("end_utc must be after start_utc")
        if (end - start).total_seconds() > MAX_QUERY_SECONDS:
            raise ValueError("Observation searches are limited to 24 hours")
        spacecraft = _clean_choices(
            self.spacecraft or spec.spacecraft,
            spec.spacecraft,
            label="spacecraft",
        )
        detectors = _clean_choices(
            self.detectors or spec.detectors,
            spec.detectors,
            label="detector",
        )
        waves = tuple(dict.fromkeys(int(item) for item in self.wavelengths))
        unknown_waves = sorted(set(waves) - set(spec.wavelengths))
        if unknown_waves:
            raise ValueError(
                "Unsupported wavelength(s): " + ", ".join(map(str, unknown_waves))
            )
        if self.sample_seconds is not None and int(self.sample_seconds) <= 0:
            raise ValueError("sample_seconds must be positive or null")
        if self.level and spec.levels and self.level not in spec.levels:
            raise ValueError(
                f"Unsupported level {self.level!r}; choose from "
                + ", ".join(spec.levels)
            )
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(self, "spacecraft", spacecraft)
        object.__setattr__(self, "detectors", detectors)
        object.__setattr__(self, "wavelengths", waves)
        object.__setattr__(self, "source_id", spec.provider)
        if self.sample_seconds is not None:
            object.__setattr__(self, "sample_seconds", int(self.sample_seconds))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_utc"] = _iso(self.start_utc)
        result["end_utc"] = _iso(self.end_utc)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObservationQueryV1:
        return cls(
            query_id=str(value["query_id"]),
            product_id=str(value["product_id"]),
            start_utc=_utc(str(value["start_utc"]), label="start_utc"),
            end_utc=_utc(str(value["end_utc"]), label="end_utc"),
            spacecraft=tuple(value.get("spacecraft") or ()),
            detectors=tuple(value.get("detectors") or ()),
            wavelengths=tuple(value.get("wavelengths") or ()),
            level=None if value.get("level") in (None, "") else str(value["level"]),
            sample_seconds=(
                None
                if value.get("sample_seconds") in (None, "")
                else int(value["sample_seconds"])
            ),
            source_id=(
                None
                if value.get("source_id") in (None, "")
                else str(value["source_id"])
            ),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class RemoteObservationV1:
    """One normalized result returned by a provider search."""

    record_id: str
    product_id: str
    provider: str
    mission: str
    spacecraft: str
    instrument: str
    detector: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    remote_locator: str
    filename: str
    target_relative_path: str
    wavelength: int | None = None
    level: str | None = None
    format: str = "fits"
    size_bytes: int | None = None
    size_is_estimate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported remote-observation schema")
        if self.product_id not in PRODUCTS:
            raise ValueError(f"Unsupported observation product: {self.product_id}")
        expected_provider = _RECORD_PROVIDERS[self.product_id]
        if self.provider != expected_provider:
            raise ValueError(
                f"Product {self.product_id} requires provider {expected_provider}"
            )
        _validate_remote_locator(self.provider, self.remote_locator)
        start = _utc(self.start_utc, label="start_utc")
        end = _utc(self.end_utc, label="end_utc")
        if end < start:
            raise ValueError("Remote observation end precedes start")
        relative = PurePosixPath(self.target_relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("target_relative_path must be a safe relative path")
        if Path(self.filename).name != self.filename:
            raise ValueError("filename must not contain directory components")
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("size_bytes cannot be negative")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_utc"] = _iso(self.start_utc)
        result["end_utc"] = _iso(self.end_utc)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteObservationV1:
        fields = dict(value)
        fields["start_utc"] = _utc(str(fields["start_utc"]), label="start_utc")
        fields["end_utc"] = _utc(str(fields["end_utc"]), label="end_utc")
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class ObservationDownloadItemV1:
    """Terminal state for one requested observation."""

    record_id: str
    local_path: str
    status: str
    bytes_written: int = 0
    sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationCollectionV1:
    """Versioned receipt for a selected observation collection."""

    collection_id: str
    items: tuple[ObservationDownloadItemV1, ...]
    created_at_utc: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_id": self.collection_id,
            "created_at_utc": _iso(self.created_at_utc),
            "items": [asdict(item) for item in self.items],
        }


def product_specs() -> tuple[ObservationProductSpec, ...]:
    """Return registered products in stable UI order."""

    return tuple(PRODUCTS.values())


def _record_id(provider: str, locator: str) -> str:
    digest = hashlib.sha256(f"{provider}\0{locator}".encode()).hexdigest()[:24]
    return f"record-{digest}"


def _validate_remote_locator(provider: str, locator: str) -> None:
    if provider == "sdac":
        relative = PurePosixPath(locator)
        is_vso_archive_fileid = relative.is_absolute() and locator.startswith(
            "/archive/"
        )
        if (
            "://" in locator
            or "\\" in locator
            or "\0" in locator
            or (relative.is_absolute() and not is_vso_archive_fileid)
            or ".." in relative.parts
            or not relative.parts
        ):
            raise ValueError("LASCO locator must be a VSO fileid")
        return
    expected_host = _PROVIDER_HOSTS[provider]
    parsed = urllib.parse.urlsplit(locator)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != expected_host:
        raise ValueError(
            f"Remote locator must use the fixed {expected_host} provider endpoint"
        )


def _safe_name(value: str, fallback: str) -> str:
    name = Path(urllib.parse.urlsplit(value).path).name or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if name in {"", ".", ".."}:
        raise ValueError("Provider returned an unsafe filename")
    return name


def _target_path(
    *,
    mission: str,
    instrument: str,
    spacecraft: str,
    observed: dt.datetime,
    filename: str,
    detector: str = "",
    wavelength: int | None = None,
) -> str:
    date = observed.strftime("%Y%m%d")
    if mission == "stereo":
        parts = (spacecraft, "euvi", date, str(wavelength or "unknown"), filename)
    elif mission == "goes":
        parts = (spacecraft, "suvi", date, str(wavelength or "unknown"), filename)
    elif mission == "solar-orbiter":
        parts = ("solar-orbiter", "eui", date, detector or "unknown", filename)
    elif mission == "soho":
        parts = ("soho", "lasco", date, detector, filename)
    elif instrument == "aia":
        parts = ("sdo", "aia", date, str(wavelength or "unknown"), filename)
    else:
        parts = ("sdo", "hmi", date, detector or "magnetogram", filename)
    return PurePosixPath(*parts).as_posix()


def search_observations(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    """Search the registered provider for one normalized query."""

    provider = PRODUCTS[query.product_id].provider
    if provider == "jsoc":
        records = _search_jsoc(query)
    elif provider == "vso-stereo":
        records = _search_stereo(query)
    elif provider == "vso-lasco":
        records = _search_lasco(query)
    elif provider == "noaa-suvi":
        records = _search_suvi(query)
    elif provider == "soar":
        records = _search_soar(query)
    else:  # pragma: no cover - registry validation prevents this
        raise ValueError(f"Unsupported observation provider: {provider}")
    unique = {f"{item.spacecraft}\0{item.remote_locator}": item for item in records}
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.start_utc,
            item.spacecraft,
            item.detector,
            item.wavelength or 0,
            item.filename,
        ),
    )
    if query.sample_seconds:
        sampled: list[RemoteObservationV1] = []
        last_by_stream: dict[tuple[str, str, str, int | None], dt.datetime] = {}
        cadence = dt.timedelta(seconds=query.sample_seconds)
        for item in ordered:
            stream = (
                item.spacecraft,
                item.instrument,
                item.detector,
                item.wavelength,
            )
            previous = last_by_stream.get(stream)
            if previous is None or item.start_utc - previous >= cadence:
                sampled.append(item)
                last_by_stream[stream] = item.start_utc
        ordered = sampled
    if len(ordered) > MAX_QUERY_RECORDS:
        raise ValueError(
            f"Search returned {len(ordered)} records; limit is {MAX_QUERY_RECORDS}. "
            "Shorten the time range or set a sample cadence."
        )
    return ordered


def _jsoc_timerange(query: ObservationQueryV1) -> str:
    start = query.start_utc.strftime("%Y.%m.%d_%H:%M:%S_UTC")
    seconds = int((query.end_utc - query.start_utc).total_seconds())
    cadence = f"@{query.sample_seconds}s" if query.sample_seconds else ""
    return f"{start}/{seconds}s{cadence}"


def _search_jsoc(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    import drms

    configurations = {
        "sdo-aia-euv": ("aia.lev1_euv_12s", "image"),
        "sdo-aia-uv": ("aia.lev1_uv_24s", "image"),
        "sdo-hmi-los": ("hmi.m_45s", "magnetogram"),
    }
    series, segment = configurations[query.product_id]
    waves: tuple[int | None, ...] = (
        tuple(query.wavelengths or PRODUCTS[query.product_id].wavelengths)
        if query.product_id != "sdo-hmi-los"
        else (None,)
    )
    client = drms.Client()
    records: list[RemoteObservationV1] = []
    for wave in waves:
        recset = f"{series}[{_jsoc_timerange(query)}]"
        if wave is not None:
            recset += f"[{wave}]"
        keys = "T_REC,QUALITY" + (",WAVELNTH" if wave is not None else "")
        metadata, segments = client.query(recset, key=keys, seg=segment)
        for row, locator in zip(
            metadata.to_dict("records"),
            segments[segment].tolist(),
            strict=False,
        ):
            if int(row.get("QUALITY", 0)) != 0 or not locator:
                continue
            observed = _parse_provider_time(str(row["T_REC"]))
            selected_wave = int(row.get("WAVELNTH") or wave) if wave else None
            suffix = "fits"
            compact = observed.strftime("%Y%m%dT%H%M%SZ")
            label = series.replace(".", "_")
            filename = f"{label}_{compact}"
            if selected_wave is not None:
                filename += f"_{selected_wave}"
            filename += f".{suffix}"
            url = f"{JSOC_BASE_URL.rstrip('/')}/{str(locator).lstrip('/')}"
            detector = "magnetogram" if wave is None else "image"
            records.append(
                RemoteObservationV1(
                    _record_id("jsoc", url),
                    query.product_id,
                    "jsoc",
                    "sdo",
                    "sdo",
                    PRODUCTS[query.product_id].instrument,
                    detector,
                    observed,
                    observed,
                    url,
                    filename,
                    _target_path(
                        mission="sdo",
                        instrument=PRODUCTS[query.product_id].instrument,
                        spacecraft="sdo",
                        observed=observed,
                        filename=filename,
                        detector=detector,
                        wavelength=selected_wave,
                    ),
                    wavelength=selected_wave,
                    level=PRODUCTS[query.product_id].levels[0],
                    format=suffix,
                    metadata={"series": series, "segment": segment, "quality": 0},
                )
            )
    return records


def _search_stereo(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    import astropy.units as u
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    records: list[RemoteObservationV1] = []
    wanted = set(query.wavelengths or PRODUCTS[query.product_id].wavelengths)
    for spacecraft in query.spacecraft:
        source = spacecraft.replace("-", "_").upper()
        attrs: list[Any] = [
            a.Time(query.start_utc, query.end_utc),
            a.Instrument("EUVI"),
            a.Source(source),
        ]
        if query.sample_seconds:
            attrs.append(a.Sample(query.sample_seconds * u.s))
        result = Fido.search(*attrs)
        for table in result:
            for row in table:
                wave = _quantity_wavelength(row["Wavelength"])
                if wanted and wave not in wanted:
                    continue
                observed = _astropy_time(row["Start Time"])
                ended = _astropy_time(row["End Time"])
                fileid = str(row["fileid"])
                filename = _safe_name(fileid, f"stereo_{observed:%Y%m%dT%H%M%S}.fts")
                records.append(
                    RemoteObservationV1(
                        _record_id("ssc", fileid),
                        query.product_id,
                        "ssc",
                        "stereo",
                        spacecraft,
                        "euvi",
                        "euvi",
                        observed,
                        ended,
                        f"{STEREO_BASE_URL.rstrip('/')}/{fileid.lstrip('/')}",
                        filename,
                        _target_path(
                            mission="stereo",
                            instrument="euvi",
                            spacecraft=spacecraft,
                            observed=observed,
                            filename=filename,
                            detector="euvi",
                            wavelength=wave,
                        ),
                        wavelength=wave,
                        level="0.5",
                        format=Path(filename).suffix.lstrip(".") or "fts",
                        size_bytes=_row_size_bytes(row),
                        size_is_estimate=True,
                        metadata={"fileid": fileid},
                    )
                )
    return records


def _search_lasco(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    records: list[RemoteObservationV1] = []
    for detector in query.detectors or ("c2", "c3"):
        result = Fido.search(
            a.Time(query.start_utc, query.end_utc),
            a.Instrument("LASCO"),
            a.Source("SOHO"),
            a.Detector(detector.upper()),
        )
        for table in result:
            for row in table:
                observed = _astropy_time(row["Start Time"])
                ended = _astropy_time(row["End Time"])
                fileid = str(row["fileid"])
                filename = _safe_name(
                    fileid, f"lasco_{detector}_{observed:%Y%m%dT%H%M%S}.fts"
                )
                records.append(
                    RemoteObservationV1(
                        _record_id("sdac", fileid),
                        query.product_id,
                        "sdac",
                        "soho",
                        "soho",
                        "lasco",
                        detector,
                        observed,
                        ended,
                        fileid,
                        filename,
                        _target_path(
                            mission="soho",
                            instrument="lasco",
                            spacecraft="soho",
                            observed=observed,
                            filename=filename,
                            detector=detector,
                        ),
                        level="0.5",
                        format=Path(filename).suffix.lstrip(".") or "fts",
                        size_bytes=_row_size_bytes(row),
                        size_is_estimate=True,
                        metadata={"fileid": fileid},
                    )
                )
    return records


def _search_suvi(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    from .suvi import DEFAULT_BASE_URL, is_suvi_file_in_window, list_remote_links

    records: list[RemoteObservationV1] = []
    waves = query.wavelengths or PRODUCTS[query.product_id].wavelengths
    cursor = query.start_utc.date()
    while cursor <= query.end_utc.date():
        day_start = max(
            query.start_utc,
            dt.datetime.combine(cursor, dt.time.min, tzinfo=dt.UTC),
        )
        day_end = min(
            query.end_utc,
            dt.datetime.combine(cursor, dt.time.max, tzinfo=dt.UTC),
        )
        date_path = cursor.strftime("%Y/%m/%d")
        date_stamp = cursor.strftime("%Y%m%d")
        for spacecraft in query.spacecraft:
            for wave in waves:
                channel = f"{wave:03d}"
                url = (
                    f"{DEFAULT_BASE_URL.rstrip('/')}/{spacecraft}/l2/data/"
                    f"suvi-l2-ci{channel}/{date_path}/"
                )
                for remote in sorted(list_remote_links(url)):
                    if not is_suvi_file_in_window(
                        urllib.parse.urlsplit(remote).path,
                        satellite=spacecraft,
                        channel=channel,
                        date_stamp=date_stamp,
                        start_hms=day_start.strftime("%H%M%S"),
                        end_hms=day_end.strftime("%H%M%S"),
                    ):
                        continue
                    filename = _safe_name(remote, f"suvi_{date_stamp}.fits")
                    observed = _suvi_filename_time(filename)
                    records.append(
                        RemoteObservationV1(
                            _record_id("noaa", remote),
                            query.product_id,
                            "noaa",
                            "goes",
                            spacecraft,
                            "suvi",
                            f"ci{channel}",
                            observed,
                            observed,
                            remote,
                            filename,
                            _target_path(
                                mission="goes",
                                instrument="suvi",
                                spacecraft=spacecraft,
                                observed=observed,
                                filename=filename,
                                wavelength=wave,
                            ),
                            wavelength=wave,
                            level="2",
                            format=Path(filename).suffix.lstrip(".") or "fits",
                        )
                    )
        cursor += dt.timedelta(days=1)
    return records


def _search_soar(query: ObservationQueryV1) -> list[RemoteObservationV1]:
    from .soar import query_eui, unique_rows

    rows = unique_rows(query_eui(_iso(query.start_utc), _iso(query.end_utc)))
    wanted_detectors = set(query.detectors)
    wanted_waves = set(query.wavelengths)
    records: list[RemoteObservationV1] = []
    for row in rows:
        detector = str(row.get("detector") or "unknown").lower()
        level = str(row.get("level") or "")
        wave_value = row.get("wavelength")
        wave = None if wave_value in (None, "") else int(round(float(wave_value)))
        if wanted_detectors and detector not in wanted_detectors:
            continue
        if wanted_waves and wave not in wanted_waves:
            continue
        if query.level and level.lower() != query.level.lower():
            continue
        observed = _parse_provider_time(str(row["begin_time"]))
        ended = _parse_provider_time(str(row.get("end_time") or row["begin_time"]))
        filename = _safe_name(
            str(row["filename"]), f"eui_{observed:%Y%m%dT%H%M%S}.fits"
        )
        data_id = str(row["data_item_id"])
        remote = (
            SOAR_DATA_URL
            + "?"
            + urllib.parse.urlencode(
                {
                    "retrieval_type": "LAST_PRODUCT",
                    "product_type": "SCIENCE",
                    "data_item_id": data_id,
                }
            )
        )
        records.append(
            RemoteObservationV1(
                _record_id("soar", data_id),
                query.product_id,
                "soar",
                "solar-orbiter",
                "solar-orbiter",
                "eui",
                detector,
                observed,
                ended,
                remote,
                filename,
                _target_path(
                    mission="solar-orbiter",
                    instrument="eui",
                    spacecraft="solar-orbiter",
                    observed=observed,
                    filename=filename,
                    detector=detector,
                    wavelength=wave,
                ),
                wavelength=wave,
                level=level,
                format=Path(filename).suffix.lstrip(".") or "fits",
                size_bytes=(
                    None if row.get("filesize") in (None, "") else int(row["filesize"])
                ),
                metadata={
                    "data_item_id": data_id,
                    "descriptor": row.get("descriptor"),
                },
            )
        )
    return records


def _parse_provider_time(value: str) -> dt.datetime:
    source = value.strip()
    is_tai = source.endswith("_TAI")
    rendered = source.removesuffix("_TAI").removesuffix("_UTC")
    if re.match(r"^\d{4}\.\d{2}\.\d{2}", rendered):
        rendered = rendered.replace(".", "-", 2)
    rendered = rendered.replace("_", "T", 1)
    if is_tai:
        from astropy.time import Time

        return _utc(
            Time(rendered, format="isot", scale="tai").utc.to_datetime(timezone=dt.UTC),
            label="provider time",
        )
    return _utc(rendered, label="provider time")


def _astropy_time(value: Any) -> dt.datetime:
    converted = getattr(value, "to_datetime", None)
    if callable(converted):
        result = converted(timezone=dt.UTC)
        return _utc(result, label="astropy time")
    return _utc(str(value), label="astropy time")


def _quantity_wavelength(value: Any) -> int:
    raw = getattr(value, "value", value)
    if hasattr(raw, "__len__") and not isinstance(raw, str):
        values = list(raw)
        raw = sum(float(item) for item in values) / len(values)
    return int(round(float(raw)))


def _row_size_bytes(row: Any) -> int | None:
    try:
        size = row["Size"]
        to_value = getattr(size, "to_value", None)
        if callable(to_value):
            import astropy.units as u

            return int(round(float(to_value(u.byte))))
        return int(round(float(size) * 1024 * 1024))
    except (KeyError, TypeError, ValueError):
        return None


def _suvi_filename_time(filename: str) -> dt.datetime:
    match = re.search(r"_s(\d{8})T(\d{6})Z", filename)
    if not match:
        raise ValueError(f"Could not parse SUVI observation time: {filename}")
    return dt.datetime.strptime(
        f"{match.group(1)}T{match.group(2)}", "%Y%m%dT%H%M%S"
    ).replace(tzinfo=dt.UTC)


def resolve_download_urls(
    records: Sequence[RemoteObservationV1],
) -> dict[str, str]:
    """Resolve provider locators into download URLs without downloading bytes."""

    result = {
        item.record_id: item.remote_locator
        for item in records
        if item.product_id != "soho-lasco"
    }
    lasco = [item for item in records if item.product_id == "soho-lasco"]
    if lasco:
        result.update(_resolve_lasco_urls(lasco))
    return result


def _resolve_lasco_urls(
    records: Sequence[RemoteObservationV1],
) -> dict[str, str]:
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from zeep.helpers import serialize_object

    resolved: dict[str, str] = {}
    by_detector: dict[str, list[RemoteObservationV1]] = {}
    for record in records:
        by_detector.setdefault(record.detector, []).append(record)
    for detector, selected in by_detector.items():
        start = min(item.start_utc for item in selected) - dt.timedelta(minutes=1)
        end = max(item.end_utc for item in selected) + dt.timedelta(minutes=1)
        response = Fido.search(
            a.Time(start, end),
            a.Instrument("LASCO"),
            a.Source("SOHO"),
            a.Detector(detector.upper()),
        )
        wanted = {item.remote_locator: item.record_id for item in selected}
        for table in response:
            indices = [
                index for index, row in enumerate(table) if str(row["fileid"]) in wanted
            ]
            if not indices:
                continue
            subset = table[indices]
            client = table.client
            response_type = client.api.get_type("VSO:VSOGetDataResponse")
            request = client.make_getdatarequest(subset)
            payload = serialize_object(
                response_type(client.api.service.GetData(request))
            )
            for response_item in payload.get("getdataresponseitem") or ():
                data_item = (response_item.get("getdataitem") or {}).get(
                    "dataitem"
                ) or ()
                for item in data_item:
                    url = str(item.get("url") or "")
                    fileids = (item.get("fileiditem") or {}).get("fileid") or ()
                    for fileid in fileids:
                        record_id = wanted.get(str(fileid))
                        if record_id and url:
                            resolved[record_id] = url
    missing = [item.record_id for item in records if item.record_id not in resolved]
    if missing:
        raise RuntimeError(
            "VSO could not resolve LASCO download URL(s): " + ", ".join(missing)
        )
    return resolved


def download_observations(
    records: Sequence[RemoteObservationV1],
    observation_root: str | Path,
    *,
    collection_id: str,
    max_workers: int = 2,
    attempts: int = 3,
    timeout: float = 180,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int, ObservationDownloadItemV1], None] | None = None,
) -> ObservationCollectionV1:
    """Download selected records atomically and return a complete receipt."""

    if not 1 <= int(max_workers) <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    root = Path(observation_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / ".observation-download-index-v1.json"
    history = _read_download_index(index_path)
    urls = resolve_download_urls(records)
    stop = cancelled or (lambda: False)
    results: dict[str, ObservationDownloadItemV1] = {}
    lock = threading.Lock()
    completed = 0

    def run(record: RemoteObservationV1) -> ObservationDownloadItemV1:
        return _download_record(
            record,
            urls[record.record_id],
            root,
            attempts=attempts,
            timeout=timeout,
            cancelled=stop,
            expected_sha256=history.get(record.target_relative_path),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run, item): item for item in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # complete receipts retain every failure
                item = ObservationDownloadItemV1(
                    record.record_id,
                    str(root / record.target_relative_path),
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            with lock:
                results[record.record_id] = item
                completed += 1
                if progress:
                    progress(completed, len(records), item)
    ordered = tuple(results[item.record_id] for item in records)
    updated_history = dict(history)
    by_id = {item.record_id: item for item in records}
    for item in ordered:
        if item.sha256 and item.status in {"downloaded", "exists"}:
            updated_history[by_id[item.record_id].target_relative_path] = item.sha256
    _write_download_index(index_path, updated_history)
    return ObservationCollectionV1(collection_id, ordered)


def _download_record(
    record: RemoteObservationV1,
    url: str,
    root: Path,
    *,
    attempts: int,
    timeout: float,
    cancelled: Callable[[], bool],
    expected_sha256: str | None,
) -> ObservationDownloadItemV1:
    destination = (root / record.target_relative_path).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Download target escaped the observation root") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        existing_size = destination.stat().st_size
        size_matches = (
            record.size_bytes is not None
            and not record.size_is_estimate
            and existing_size == record.size_bytes
        )
        existing_hash = _hash_file(destination) if expected_sha256 else None
        hash_matches = bool(
            expected_sha256
            and existing_hash
            and existing_hash.lower() == expected_sha256.lower()
        )
        if size_matches or hash_matches:
            return ObservationDownloadItemV1(
                record.record_id,
                str(destination),
                "exists",
                existing_size,
                existing_hash or _hash_file(destination),
            )
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            if cancelled():
                raise InterruptedError("Download cancelled")
            request = urllib.request.Request(
                url, headers={"User-Agent": "solar-physics-toolkit"}
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        if cancelled():
                            raise InterruptedError("Download cancelled")
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size <= 0:
                raise RuntimeError("Provider returned an empty file")
            if (
                record.size_bytes is not None
                and not record.size_is_estimate
                and size != record.size_bytes
            ):
                raise RuntimeError(
                    f"Downloaded size {size} does not match expected {record.size_bytes}"
                )
            temporary.replace(destination)
            return ObservationDownloadItemV1(
                record.record_id,
                str(destination),
                "downloaded",
                size,
                digest.hexdigest(),
            )
        except InterruptedError:
            temporary.unlink(missing_ok=True)
            return ObservationDownloadItemV1(
                record.record_id,
                str(destination),
                "cancelled",
                error="Download cancelled",
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
    return ObservationDownloadItemV1(
        record.record_id,
        str(destination),
        "failed",
        error=last_error,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_download_index(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("sha256_by_path", {})
        if not isinstance(values, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in values.items()
            if isinstance(key, str)
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", value)
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _write_download_index(path: Path, values: Mapping[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "sha256_by_path": dict(sorted(values.items())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_search_result(
    path: str | Path,
    query: ObservationQueryV1,
    records: Sequence[RemoteObservationV1],
) -> Path:
    """Persist one portable search result for selection and retries."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query": query.to_dict(),
        "record_count": len(records),
        "total_size_bytes": sum(item.size_bytes or 0 for item in records),
        "unknown_size_count": sum(item.size_bytes is None for item in records),
        "records": [item.to_dict() for item in records],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_remote_records(path: str | Path) -> list[RemoteObservationV1]:
    """Read selected or complete records from a search-result JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("Observation selection must contain a records list")
    return [RemoteObservationV1.from_dict(item) for item in values]


__all__ = [
    "MAX_QUERY_RECORDS",
    "MAX_QUERY_SECONDS",
    "ObservationCollectionV1",
    "ObservationDownloadItemV1",
    "ObservationProductSpec",
    "ObservationQueryV1",
    "PRODUCTS",
    "RemoteObservationV1",
    "download_observations",
    "product_specs",
    "read_remote_records",
    "resolve_download_urls",
    "search_observations",
    "write_search_result",
]
