"""Deterministic UTC index and coordinator tests for App 1.0 Phase 3."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from solar_apps.frontends.app_v1.timeline import (
    SQLiteTimelineIndex,
    TimeCoordinator,
    TimelineSample,
    normalize_utc,
)


def _sample(locator: str, timestamp: str | None, status: str = "available"):
    return TimelineSample(
        locator=locator,
        observed_at_utc=timestamp,
        status=status,
    )


def test_sqlite_index_records_statuses_duplicates_and_can_be_rebuilt(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "state" / "time_index.sqlite3")
    count = index.rebuild_source(
        source_id="radio-source",
        module_id="source-map",
        local_source=tmp_path / "radio",
        samples=[
            _sample("b.fits", "2025-01-24T23:59:59Z"),
            _sample("a.fits", "2025-01-24T23:59:59Z"),
            _sample("missing.fits", None, "missing"),
            _sample("broken.fits", None, "unreadable"),
        ],
    )

    assert count == 4
    assert index.schema_version == 1
    assert index.source("radio-source")["record_count"] == 4
    samples = index.samples("radio-source")
    assert [item.status for item in samples] == [
        "duplicate",
        "available",
        "unreadable",
        "missing",
    ]
    assert [
        item.locator for item in index.samples("radio-source", matchable_only=True)
    ] == [
        "a.fits",
        "b.fits",
    ]

    assert (
        index.rebuild_source(
            source_id="radio-source",
            module_id="source-map",
            local_source=tmp_path / "radio",
            samples=[_sample("new.fits", "2025-01-25T00:00:01Z")],
        )
        == 1
    )
    assert [item.locator for item in index.samples("radio-source")] == ["new.fits"]
    assert index.invalidate_source("radio-source") is True
    assert index.invalidate_source("radio-source") is False


def test_coordinator_matches_offsets_duplicates_gaps_and_cross_day(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "time.sqlite3")
    index.rebuild_source(
        source_id="base-source",
        module_id="dart-spectrogram",
        local_source=tmp_path / "dart",
        samples=[
            _sample("before", "2025-01-24T23:59:59Z"),
            _sample("after", "2025-01-25T00:00:01Z"),
        ],
    )
    index.rebuild_source(
        source_id="aia-source",
        module_id="image-viewer",
        local_source=tmp_path / "aia",
        samples=[
            _sample("z-match", "2025-01-25T00:00:00Z"),
            _sample("a-match", "2025-01-25T00:00:00Z"),
        ],
    )
    index.rebuild_source(
        source_id="gap-source",
        module_id="source-map",
        local_source=tmp_path / "gap",
        samples=[_sample("too-far", "2025-01-25T00:00:20Z")],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("base-source")
    coordinator.register_source(
        "aia-source",
        offset_seconds=1.0,
        tolerance_seconds=0.0,
    )
    coordinator.register_source("gap-source", tolerance_seconds=2.0)
    coordinator.set_base_source("base-source")

    first = coordinator.step(1)
    assert first.current_time_utc == datetime(
        2025, 1, 24, 23, 59, 59, tzinfo=timezone.utc
    )
    second = coordinator.step(1)
    assert second.current_time_utc == datetime(
        2025, 1, 25, 0, 0, 1, tzinfo=timezone.utc
    )
    assert second.matched_locators["aia-source"] == "a-match"
    assert second.matched_locators["gap-source"] is None
    assert coordinator.step(-1).current_time_utc == first.current_time_utc


def test_coordinator_persists_and_restores_sync_configuration(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "time.sqlite3")
    index.rebuild_source(
        source_id="radio-source",
        module_id="source-trajectory",
        local_source=tmp_path / "radio",
        samples=[_sample("one", "2025-01-24T04:48:30Z")],
    )
    config = tmp_path / "workspaces" / "preview.timeline.json"
    coordinator = TimeCoordinator(index)
    coordinator.register_source(
        "radio-source",
        offset_seconds=-0.45,
        tolerance_seconds=1.25,
    )
    coordinator.select("2025-01-24T04:48:29.55Z")
    coordinator.save(config)

    restored = TimeCoordinator(SQLiteTimelineIndex(index.path))
    assert restored.load(config) is True
    assert restored.base_source_id == "radio-source"
    assert restored.current_time_utc == normalize_utc("2025-01-24T04:48:29.55Z")
    assert restored.sources[0].offset_seconds == -0.45
    assert restored.sources[0].tolerance_seconds == 1.25
    assert (
        restored.select(restored.current_time_utc).matched_locators["radio-source"]
        == "one"
    )


def test_naive_or_invalid_timestamps_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        normalize_utc("2025-01-24T04:48:30")
    with pytest.raises(ValueError, match="UTC timestamp"):
        normalize_utc("")

    index = SQLiteTimelineIndex(tmp_path / "time.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        with index._connect() as connection:
            connection.execute("""
                INSERT INTO timeline_sources(
                    source_id, module_id, local_source, updated_at_utc, record_count
                ) VALUES ('bad source', 'source-map', 'x', '2025-01-24T00:00:00Z', -1)
                """)
