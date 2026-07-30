"""Explicitly opted-in live archive checks for the observation registry."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from solar_toolkit.net.observations import (
    ObservationQueryV1,
    download_observations,
    search_observations,
)

if os.getenv("SOLAR_TOOLKIT_ONLINE_TESTS") != "1":
    pytest.skip(
        "Set SOLAR_TOOLKIT_ONLINE_TESTS=1 to query and download live archives",
        allow_module_level=True,
    )


def _query(
    query_id: str,
    product_id: str,
    start: str,
    *,
    minutes: int = 1,
    spacecraft: tuple[str, ...] = (),
    detectors: tuple[str, ...] = (),
    wavelengths: tuple[int, ...] = (),
) -> ObservationQueryV1:
    begin = dt.datetime.fromisoformat(start).replace(tzinfo=dt.UTC)
    return ObservationQueryV1(
        query_id,
        product_id,
        begin,
        begin + dt.timedelta(minutes=minutes),
        spacecraft=spacecraft,
        detectors=detectors,
        wavelengths=wavelengths,
    )


LIVE_QUERIES = (
    _query("online-aia-euv", "sdo-aia-euv", "2025-01-01T00:00:00", wavelengths=(171,)),
    _query("online-aia-uv", "sdo-aia-uv", "2025-01-01T00:00:00", wavelengths=(1600,)),
    _query("online-hmi", "sdo-hmi-los", "2025-01-01T00:00:00"),
    _query(
        "online-stereo-ab",
        "stereo-euvi",
        "2012-06-01T00:00:00",
        minutes=10,
        spacecraft=("stereo-a", "stereo-b"),
        wavelengths=(171,),
    ),
    _query(
        "online-lasco-c23",
        "soho-lasco",
        "2025-01-01T00:00:00",
        minutes=30,
        detectors=("c2", "c3"),
    ),
    _query(
        "online-lasco-c1",
        "soho-lasco",
        "1997-05-01T00:00:00",
        minutes=60,
        detectors=("c1",),
    ),
    _query(
        "online-suvi",
        "goes-suvi",
        "2025-01-01T00:00:00",
        spacecraft=("goes16",),
        wavelengths=(171,),
    ),
    _query(
        "online-eui",
        "solar-orbiter-eui",
        "2025-01-01T00:00:00",
        minutes=60,
    ),
)


@pytest.mark.parametrize("query", LIVE_QUERIES, ids=lambda query: query.query_id)
def test_live_provider_search_and_one_file_download(
    query: ObservationQueryV1,
    tmp_path: Path,
) -> None:
    records = search_observations(query)
    assert records, f"{query.query_id} returned no live archive records"

    collection = download_observations(
        records[:1],
        tmp_path / "observations",
        collection_id=f"collection-{query.query_id}",
        max_workers=1,
    )
    item = collection.items[0]
    target = Path(item.local_path)
    assert item.status in {"downloaded", "exists"}
    assert item.bytes_written > 0
    assert item.sha256 and len(item.sha256) == 64
    assert target.is_file()
    assert target.suffix.lower().lstrip(".") in {
        records[0].format.lower(),
        "fits",
        "fts",
    }
