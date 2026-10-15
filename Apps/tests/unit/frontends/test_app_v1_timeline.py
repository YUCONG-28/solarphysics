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
            _sample("b.fits", "2000-01-01T23:59:59Z"),
            _sample("a.fits", "2000-01-01T23:59:59Z"),
            _sample("missing.fits", None, "missing"),
            _sample("broken.fits", None, "unreadable"),
        ],
    )

    assert count == 4
    assert index.schema_version == 2
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
            samples=[_sample("new.fits", "2000-01-02T00:00:01Z")],
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
            _sample("before", "2000-01-01T23:59:59Z"),
            _sample("after", "2000-01-02T00:00:01Z"),
        ],
    )
    index.rebuild_source(
        source_id="secondary-aia-source",
        module_id="image-viewer",
        local_source=tmp_path / "aia",
        samples=[
            _sample("z-match", "2000-01-02T00:00:00Z"),
            _sample("a-match", "2000-01-02T00:00:00Z"),
        ],
    )
    index.rebuild_source(
        source_id="gap-source",
        module_id="source-map",
        local_source=tmp_path / "gap",
        samples=[_sample("too-far", "2000-01-02T00:00:20Z")],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("base-source")
    coordinator.register_source(
        "secondary-aia-source",
        offset_seconds=1.0,
        tolerance_seconds=0.0,
    )
    coordinator.register_source("gap-source", tolerance_seconds=2.0)
    coordinator.set_base_source("base-source")

    first = coordinator.step(1)
    assert first.base_source_id == "gap-source"
    assert first.output_source_id == "secondary-aia-source"
    assert first.current_time_utc == datetime(2000, 1, 2, tzinfo=timezone.utc)
    second = coordinator.step(1)
    assert second.current_time_utc == first.current_time_utc
    assert first.matched_locators["base-source"] == "before"
    assert second.matched_locators["base-source"] == "after"
    assert (
        coordinator.current_plan.rows[1].match_time.utc
        == "2000-01-02T00:00:01.000000000"
    )
    assert first.matched_locators["secondary-aia-source"] == "a-match"
    assert second.matched_locators["secondary-aia-source"] == "z-match"
    assert second.matched_locators["gap-source"] == "too-far"
    assert (
        coordinator.current_plan.rows[1].selections["gap-source"].within_tolerance
        is False
    )
    assert coordinator.step(-1).current_time_utc == first.current_time_utc


def test_coordinator_persists_and_restores_sync_configuration(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "time.sqlite3")
    index.rebuild_source(
        source_id="radio-source",
        module_id="source-trajectory",
        local_source=tmp_path / "radio",
        samples=[_sample("one", "2000-01-01T04:48:30Z")],
    )
    config = tmp_path / "workspaces" / "preview.timeline.json"
    coordinator = TimeCoordinator(index)
    coordinator.register_source(
        "radio-source",
        offset_seconds=-0.45,
        tolerance_seconds=1.25,
    )
    coordinator.select("2000-01-01T04:48:29.55Z")
    coordinator.save(config)

    restored = TimeCoordinator(SQLiteTimelineIndex(index.path))
    assert restored.load(config) is True
    assert restored.base_source_id == "radio-source"
    assert restored.current_time_utc == normalize_utc("2000-01-01T04:48:30Z")
    assert (
        restored.current_plan.rows[0].match_time.utc == "2000-01-01T04:48:29.550000000"
    )
    assert restored.sources[0].offset_seconds == -0.45
    assert restored.sources[0].tolerance_seconds == 1.25
    assert (
        restored.select(restored.current_time_utc).matched_locators["radio-source"]
        == "one"
    )


def test_naive_or_invalid_timestamps_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        normalize_utc("2000-01-01T04:48:30")
    with pytest.raises(ValueError, match="UTC timestamp"):
        normalize_utc("")

    index = SQLiteTimelineIndex(tmp_path / "time.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        with index._connect() as connection:
            connection.execute(
                """
                INSERT INTO timeline_sources(
                    source_id, module_id, local_source, updated_at_utc, record_count
                ) VALUES ('bad source', 'source-map', 'x', '2000-01-01T00:00:00Z', -1)
                """
            )


def test_submicrosecond_duplicate_browsing_persistence_and_fullest_source(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "precision.sqlite3")
    index.rebuild_source(
        source_id="small",
        module_id="image-composer",
        local_source=tmp_path,
        samples=[_sample("small", "2000-01-01T04:48:30.123456789Z")],
    )
    index.rebuild_source(
        source_id="fullest",
        module_id="image-composer",
        local_source=tmp_path,
        samples=[
            _sample("second", "2000-01-01T04:48:30.123456790Z"),
            _sample("first", "2000-01-01T04:48:30.123456789Z"),
        ],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("small")
    coordinator.register_source("fullest")
    assert coordinator.base_source_id == "small"
    assert coordinator.output_source_id == "fullest"
    assert coordinator.step(1).matched_locators["fullest"] == "first"
    assert coordinator.step(1).matched_locators["fullest"] == "second"
    assert coordinator.current_time_text.endswith(".123456790Z")
    saved = coordinator.to_dict()
    restored = TimeCoordinator(index)
    restored.load_dict(saved)
    assert restored.step(0).matched_locators["fullest"] == "second"
    assert restored.step(-1).matched_locators["fullest"] == "first"
    assert restored.current_plan.rows[1].selections["small"].delta_ns == -1


def test_old_runtime_index_is_invalidated_and_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "old-cache.sqlite3"
    index = SQLiteTimelineIndex(path)
    index.rebuild_source(
        source_id="old",
        module_id="image-composer",
        local_source=tmp_path,
        samples=[_sample("old", "2000-01-01T04:48:30Z")],
    )
    with index._connect() as connection:
        connection.execute("PRAGMA user_version = 1")
    rebuilt = SQLiteTimelineIndex(path)
    assert rebuilt.schema_version == 2
    assert rebuilt.list_sources() == []


def _indexed_source(index, tmp_path, source_id, times):
    index.rebuild_source(
        source_id=source_id,
        module_id="image-composer",
        local_source=tmp_path,
        samples=[
            _sample(f"{source_id}-{number}", value)
            for number, value in enumerate(times)
        ],
    )


def test_window_filters_before_master_selection_and_retains_full_raw_index(tmp_path):
    index = SQLiteTimelineIndex(tmp_path / "window.sqlite3")
    _indexed_source(
        index,
        tmp_path,
        "outside-fullest",
        [
            "2000-01-01T04:47:00Z",
            "2000-01-01T04:47:01Z",
            "2000-01-01T04:47:02Z",
            "2000-01-01T04:48:30Z",
        ],
    )
    _indexed_source(
        index,
        tmp_path,
        "inside-fullest",
        [
            "2000-01-01T04:48:30Z",
            "2000-01-01T04:48:31Z",
            "2000-01-01T04:48:32Z",
        ],
    )
    coordinator = TimeCoordinator(
        index, time_start="2000-01-01T04:48:30Z", time_end="2000-01-01T04:49:00Z"
    )
    coordinator.register_source("outside-fullest")
    coordinator.register_source("inside-fullest")
    plan = coordinator.current_plan
    assert plan.master_source_id == "outside-fullest"
    assert plan.output_source_id == "inside-fullest"
    assert len(plan.rows) == 3
    assert all(
        row.selections["outside-fullest"].frame.frame_id == "outside-fullest-3"
        for row in plan.rows
    )
    summary = plan.summary["sources"]["outside-fullest"]
    assert summary["input_frame_count"] == 4
    assert summary["in_window_frame_count"] == 1
    assert summary["outside_window_frame_count"] == 3
    assert summary["coverage_fraction"] == 1
    assert len(index.samples("outside-fullest")) == 4
    coordinator.set_time_window()
    assert coordinator.base_source_id == "inside-fullest"
    assert coordinator.output_source_id == "outside-fullest"
    assert len(coordinator.current_plan.rows) == 4


def test_window_half_open_nanoseconds_and_observed_clock_ignore_offsets(tmp_path):
    index = SQLiteTimelineIndex(tmp_path / "nanosecond-window.sqlite3")
    _indexed_source(
        index,
        tmp_path,
        "radio",
        [
            "2000-01-01T23:59:59.999999999Z",
            "2000-01-02T00:00:00.000000000Z",
            "2000-01-02T00:00:00.000000001Z",
            "2000-01-02T00:00:00.000000002Z",
        ],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("radio", offset_seconds=-1)
    coordinator.set_time_window(
        "2000-01-02T08:00:00+08:00", "2000-01-02T00:00:00.000000002Z"
    )
    assert len(coordinator.current_plan.rows) == 2
    assert [row.master_frame.frame_id for row in coordinator.current_plan.rows] == [
        "radio-1",
        "radio-2",
    ]
    assert coordinator.step(1).matched_locators["radio"] == "radio-1"
    assert coordinator.step(1).matched_locators["radio"] == "radio-2"
    assert coordinator.current_time_text == "2000-01-02T00:00:00.000000001Z"
    assert (
        coordinator.current_plan.rows[1].match_time.utc
        == "2000-01-01T23:59:59.000000001"
    )
    assert coordinator.time_start == "2000-01-02T00:00:00.000000000Z"


def test_window_cache_changes_even_if_same_frames_and_browsing_does_not_change_bounds(
    tmp_path,
):
    index = SQLiteTimelineIndex(tmp_path / "cache-window.sqlite3")
    _indexed_source(
        index, tmp_path, "radio", ["2000-01-01T04:48:31Z", "2000-01-01T04:48:32Z"]
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("radio")
    coordinator.set_time_window("2000-01-01T04:48:30Z", "2000-01-01T04:49:00Z")
    previous_plan = coordinator.current_plan
    coordinator.step(1)
    coordinator.step(1)
    coordinator.step(-1)
    assert coordinator.current_plan is previous_plan
    assert coordinator.time_start == "2000-01-01T04:48:30.000000000Z"
    coordinator.set_time_window("2000-01-01T04:48:29Z", "2000-01-01T04:49:00Z")
    assert coordinator.current_plan is not previous_plan
    assert coordinator.current_plan.fingerprint != previous_plan.fingerprint
    assert len(coordinator.current_plan.rows) == 2
    assert coordinator.step(0).matched_locators["radio"] == "radio-0"


def test_empty_window_clears_selection_and_reports_missing_secondary(tmp_path):
    index = SQLiteTimelineIndex(tmp_path / "empty-window.sqlite3")
    _indexed_source(
        index, tmp_path, "radio", ["2000-01-01T04:48:31Z", "2000-01-01T04:48:32Z"]
    )
    _indexed_source(index, tmp_path, "aia", ["2000-01-01T04:47:00Z"])
    coordinator = TimeCoordinator(index)
    coordinator.register_source("radio")
    coordinator.register_source("aia")
    coordinator.step(1)
    states = []
    coordinator.subscribe_state(
        lambda: states.append(len(coordinator.current_plan.rows))
    )
    coordinator.set_time_window("2000-01-01T04:48:30Z", "2000-01-01T04:49:00Z")
    assert coordinator.step(0).matched_locators["aia"] is None
    assert len(coordinator.current_plan.rows) == 2
    coordinator.set_time_window("2000-01-01T04:49:00Z", "2000-01-01T04:50:00Z")
    assert states == [2, 0]
    assert coordinator.current_plan.rows == ()
    assert coordinator.current_time_utc is None
    assert coordinator.current_time_text is None
    assert coordinator._cursor_index is None
    with pytest.raises(ValueError, match="No available images inside"):
        coordinator.step(1)
    with pytest.raises(ValueError, match="No available images inside"):
        coordinator.select("2000-01-01T04:48:30Z")


def test_window_configuration_round_trip_and_old_settings_unbounded(tmp_path):
    index = SQLiteTimelineIndex(tmp_path / "saved-window.sqlite3")
    _indexed_source(
        index,
        tmp_path,
        "radio",
        ["2000-01-01T04:48:30.000000001Z", "2000-01-01T04:48:30.000000002Z"],
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("radio")
    coordinator.set_time_window(
        "2000-01-01T04:48:30.000000001Z", "2000-01-01T04:48:30.000000002Z"
    )
    coordinator.step(1)
    saved = coordinator.to_dict()
    restored = TimeCoordinator(index)
    restored.load_dict(saved)
    assert restored.time_start == saved["time_start"]
    assert restored.time_end == saved["time_end"]
    assert restored.step(0).matched_locators["radio"] == "radio-0"
    assert len(restored.current_plan.rows) == 1
    saved.pop("time_start")
    saved.pop("time_end")
    restored.load_dict(saved)
    assert restored.time_start is None and restored.time_end is None
    assert len(restored.current_plan.rows) == 2


def test_invalid_window_does_not_mutate_valid_configuration(tmp_path):
    index = SQLiteTimelineIndex(tmp_path / "invalid-window.sqlite3")
    coordinator = TimeCoordinator(index)
    coordinator.set_time_window("2000-01-01T04:48:30Z", "2000-01-01T04:49:00Z")
    saved = coordinator.to_dict()
    for start, end in (
        ("invalid", None),
        ("2000-01-01T04:49:00Z", "2000-01-01T04:49:00Z"),
    ):
        with pytest.raises(ValueError):
            coordinator.set_time_window(start, end)
        assert coordinator.to_dict() == saved


def test_hierarchical_plan_broadcasts_root_output_and_inherited_parent_identity(
    tmp_path: Path,
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "hierarchy.sqlite3")
    for source_id, seconds in (
        ("root", [0, 10]),
        ("middle", [0, 6, 10]),
        ("output", [0, 1, 2, 4, 5, 10]),
    ):
        _indexed_source(
            index,
            tmp_path,
            source_id,
            [f"2000-01-01T04:48:{second:02d}Z" for second in seconds],
        )
    coordinator = TimeCoordinator(index)
    for source_id in ("output", "root", "middle"):
        coordinator.register_source(source_id)
    plan = coordinator.current_plan
    assert plan.source_order == ("root", "middle", "output")
    selected = coordinator.select("2000-01-01T04:48:04Z")
    assert selected.base_source_id == "root"
    assert selected.output_source_id == "output"
    assert selected.matched_locators == {
        "root": "root-1",
        "middle": "middle-1",
        "output": "output-3",
    }
    assert coordinator.current_time_text == "2000-01-01T04:48:04.000000000Z"
    assert coordinator.step(1).matched_locators["root"] == "root-1"
    assert coordinator.step(-1).matched_locators == selected.matched_locators
    saved = coordinator.to_dict()
    assert saved["algorithm_version"] == 3
    assert saved["master_policy"] == "auto_min_frames_hierarchical"
    assert saved["source_order"] == ["root", "middle", "output"]
    restored = TimeCoordinator(index)
    restored.load_dict(saved)
    assert restored.step(0).to_dict() == selected.to_dict()
    assert restored.current_plan.parent_maps == plan.parent_maps


@pytest.mark.parametrize("strategy", [1, 2])
def test_legacy_strategy_migration_rebuilds_root_and_invalidates_cursor_reference(
    tmp_path: Path, strategy: int
) -> None:
    index = SQLiteTimelineIndex(tmp_path / "migration.sqlite3")
    _indexed_source(index, tmp_path, "root", ["2000-01-01T04:48:00Z"])
    _indexed_source(
        index, tmp_path, "output", ["2000-01-01T04:48:00Z", "2000-01-01T04:48:01Z"]
    )
    coordinator = TimeCoordinator(index)
    coordinator.register_source("root")
    coordinator.register_source("output")
    coordinator.step(1)
    saved = coordinator.to_dict()
    saved.update(
        algorithm_version=strategy,
        master_policy="auto_max_frames",
        base_source_id="output",
        cursor_index=1,
    )
    restored = TimeCoordinator(index)
    restored.load_dict(saved)
    assert restored.base_source_id == "root" and restored.output_source_id == "output"
    assert "Migrated strategy 1/2" in restored.migration_note
    assert restored.step(0).matched_locators["output"] == "output-0"
