from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from solar_toolkit.net import observations
from solar_toolkit.net.observations import (
    ObservationQueryV1,
    RemoteObservationV1,
    download_observations,
    read_remote_records,
    search_observations,
    write_search_result,
)

UTC = dt.UTC


def _query(product_id: str = "stereo-euvi") -> ObservationQueryV1:
    return ObservationQueryV1(
        "query-test",
        product_id,
        dt.datetime(2025, 1, 1, tzinfo=UTC),
        dt.datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
    )


def _record(
    locator: str,
    *,
    spacecraft: str = "stereo-a",
    size: int | None = None,
    target: str = "stereo-a/euvi/20250101/171/example.fts",
) -> RemoteObservationV1:
    observed = dt.datetime(2025, 1, 1, tzinfo=UTC)
    if not locator.startswith("https://stereo-ssc.nascom.nasa.gov/"):
        locator = (
            "https://stereo-ssc.nascom.nasa.gov/data/ins_data/" + Path(locator).name
        )
    return RemoteObservationV1(
        "record-" + hashlib.sha256(locator.encode()).hexdigest()[:24],
        "stereo-euvi",
        "ssc",
        "stereo",
        spacecraft,
        "euvi",
        "euvi",
        observed,
        observed,
        locator,
        "example.fts",
        target,
        wavelength=171,
        level="0.5",
        format="fts",
        size_bytes=size,
    )


def test_registry_and_query_contract_cover_all_requested_products() -> None:
    assert tuple(observations.PRODUCTS) == (
        "sdo-aia-euv",
        "sdo-aia-uv",
        "sdo-hmi-los",
        "stereo-euvi",
        "soho-lasco",
        "goes-suvi",
        "solar-orbiter-eui",
    )
    assert observations.PRODUCTS["stereo-euvi"].spacecraft == (
        "stereo-a",
        "stereo-b",
    )
    assert observations.PRODUCTS["soho-lasco"].detectors == ("c2", "c3", "c1")
    assert (
        observations.PRODUCTS["soho-lasco"].url_resolution_capability
        == "vso-getdata-url-file"
    )

    restored = ObservationQueryV1.from_dict(_query().to_dict())
    assert restored == _query()
    assert restored.source_id == "vso-stereo"
    with pytest.raises(ValueError, match="24 hours"):
        ObservationQueryV1(
            "query-long",
            "stereo-euvi",
            dt.datetime(2025, 1, 1, tzinfo=UTC),
            dt.datetime(2025, 1, 2, 0, 0, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="spacecraft"):
        replace(_query(), spacecraft=("stereo-c",))
    with pytest.raises(ValueError, match="wavelength"):
        replace(_query(), wavelengths=(193,))
    with pytest.raises(ValueError, match="level"):
        replace(_query(), level="9")


def test_remote_contract_rejects_path_traversal_and_round_trips(tmp_path: Path) -> None:
    record = _record((tmp_path / "source.fts").as_uri())
    assert RemoteObservationV1.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="safe relative"):
        replace(record, target_relative_path="../escape.fts")
    with pytest.raises(ValueError, match="directory"):
        replace(record, filename="../escape.fts")
    with pytest.raises(ValueError, match="fixed"):
        replace(record, remote_locator="https://example.invalid/example.fts")


def test_lasco_contract_accepts_vso_archive_fileids_only() -> None:
    observations._validate_remote_locator(
        "sdac",
        "/archive/soho/private/data/processed/lasco/level_05/250101/c2/example.fts",
    )
    observations._validate_remote_locator("sdac", "archive/soho/lasco/example.fts")

    for locator in (
        "/etc/passwd",
        "https://example.invalid/example.fts",
        "archive/../escape.fts",
        r"archive\escape.fts",
    ):
        with pytest.raises(ValueError, match="VSO fileid"):
            observations._validate_remote_locator("sdac", locator)


def test_stereo_search_deduplicates_by_spacecraft_and_fileid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one = _record("fileid-one")
    same_file_other_spacecraft = replace(
        one,
        record_id="record-other",
        spacecraft="stereo-b",
        target_relative_path="stereo-b/euvi/20250101/171/example.fts",
    )
    monkeypatch.setattr(
        observations,
        "_search_stereo",
        lambda _query: [one, one, same_file_other_spacecraft],
    )

    result = search_observations(_query())

    assert [(item.spacecraft, item.remote_locator) for item in result] == [
        (
            "stereo-a",
            "https://stereo-ssc.nascom.nasa.gov/data/ins_data/fileid-one",
        ),
        (
            "stereo-b",
            "https://stereo-ssc.nascom.nasa.gov/data/ins_data/fileid-one",
        ),
    ]


def test_stereo_vso_query_uses_secchi_detector_and_nominal_wavelengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import astropy.units as u
    from sunpy.net import Fido

    observed = dt.datetime(2025, 1, 24, 4, 30, tzinfo=UTC)
    rows = [
        {
            "Start Time": observed,
            "End Time": observed + dt.timedelta(seconds=2),
            "Wavelength": [171, 175] * u.AA,
            "fileid": "secchi/L0/a/img/euvi/20250124/171.fts",
            "Size": 8_409_600 * u.byte,
        },
        {
            "Start Time": observed + dt.timedelta(seconds=15),
            "End Time": observed + dt.timedelta(seconds=17),
            "Wavelength": [195, 195] * u.AA,
            "fileid": "secchi/L0/a/img/euvi/20250124/195.fts",
            "Size": 8_409_600 * u.byte,
        },
        {
            "Start Time": observed + dt.timedelta(seconds=30),
            "End Time": observed + dt.timedelta(seconds=32),
            "Wavelength": [304, 304] * u.AA,
            "fileid": "secchi/L0/a/img/euvi/20250124/304.fts",
            "Size": 8_409_600 * u.byte,
        },
    ]
    searches: list[tuple[object, ...]] = []

    def search(*attrs: object) -> list[list[dict[str, object]]]:
        searches.append(attrs)
        return [rows]

    monkeypatch.setattr(Fido, "search", search)
    query = replace(
        _query(),
        start_utc=observed,
        end_utc=observed + dt.timedelta(minutes=30),
        spacecraft=("stereo-a",),
        wavelengths=(171, 304),
    )

    result = observations._search_stereo(query)

    assert [item.wavelength for item in result] == [171, 304]
    values = {type(attr).__name__: getattr(attr, "value", None) for attr in searches[0]}
    assert values["Instrument"] == "SECCHI"
    assert values["Detector"] == "EUVI"
    assert values["Source"] == "STEREO_A"


def test_native_cadence_is_default_and_sampling_is_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _record("first.fts")
    second = replace(
        _record("second.fts"),
        start_utc=first.start_utc + dt.timedelta(seconds=30),
        end_utc=first.end_utc + dt.timedelta(seconds=30),
        filename="second.fts",
        target_relative_path="stereo-a/euvi/20250101/171/second.fts",
    )
    monkeypatch.setattr(
        observations,
        "_search_stereo",
        lambda _query: [first, second],
    )

    assert len(search_observations(_query())) == 2
    assert len(search_observations(replace(_query(), sample_seconds=60))) == 1


def test_search_json_round_trip_and_target_layout(tmp_path: Path) -> None:
    query = _query()
    record = _record("https://example.invalid/example.fts", size=1024)
    target = write_search_result(tmp_path / "search.json", query, [record])
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["record_count"] == 1
    assert payload["total_size_bytes"] == 1024
    assert read_remote_records(target) == [record]
    assert record.target_relative_path.startswith("stereo-a/euvi/20250101/171/")


def test_atomic_download_hash_and_history_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"SIMPLE  =                    T" * 16
    source = tmp_path / "source.fts"
    source.write_bytes(payload)
    record = _record(source.as_uri(), size=len(payload))
    monkeypatch.setattr(
        observations,
        "resolve_download_urls",
        lambda _records: {record.record_id: source.as_uri()},
    )
    root = tmp_path / "observations"

    first = download_observations(
        [record],
        root,
        collection_id="collection-first",
        max_workers=1,
        attempts=1,
    )
    expected = hashlib.sha256(payload).hexdigest()
    downloaded = first.items[0]
    destination = Path(downloaded.local_path)

    assert downloaded.status == "downloaded"
    assert downloaded.sha256 == expected
    assert destination.read_bytes() == payload
    assert not list(root.rglob("*.part"))

    changed_provider_size = replace(record, size_bytes=len(payload) + 99)
    second = download_observations(
        [changed_provider_size],
        root,
        collection_id="collection-second",
        max_workers=1,
        attempts=1,
    )
    assert second.items[0].status == "exists"
    assert second.items[0].sha256 == expected


def test_download_retries_and_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"retry-success"
    calls = 0

    class Response:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.reads = 0

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self.reads += 1
            if self.fail:
                if self.reads == 1:
                    return b"partial"
                raise OSError("temporary network error")
            return payload if self.reads == 1 else b""

    def fake_open(_request: object, *, timeout: float) -> Response:
        nonlocal calls
        assert timeout == 5
        calls += 1
        return Response(calls < 3)

    monkeypatch.setattr(observations.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(observations.time, "sleep", lambda _seconds: None)
    record = _record("https://example.invalid/retry.fts", size=len(payload))

    result = download_observations(
        [record],
        tmp_path / "observations",
        collection_id="collection-retry",
        max_workers=1,
        attempts=3,
        timeout=5,
    )

    assert calls == 3
    assert result.items[0].status == "downloaded"
    assert not list((tmp_path / "observations").rglob("*.part"))


def test_cancelled_download_cleans_current_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b"bytes"

    monkeypatch.setattr(
        observations.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    result = download_observations(
        [_record("https://example.invalid/cancel.fts")],
        tmp_path / "observations",
        collection_id="collection-cancel",
        max_workers=1,
        attempts=1,
        cancelled=cancelled,
    )

    assert result.items[0].status == "cancelled"
    assert not list((tmp_path / "observations").rglob("*.part"))


def test_provider_time_preserves_tai_semantics() -> None:
    parsed = observations._parse_provider_time("2017.01.01_00:00:37_TAI")
    assert parsed == dt.datetime(2017, 1, 1, tzinfo=UTC)
