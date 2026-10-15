from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from solar_apps.frontends.image_composer.matching import (
    MatchPlanError,
    build_match_plan,
    iter_match_rows,
    nearest_record,
    freeze_match_plan,
    select_plan_frame,
    StaleMatchPlanError,
    FrameMatch,
    ComposerMatchPlan,
)
from solar_apps.frontends.image_composer.models import (
    CanvasSettings,
    ComposerProject,
    FolderSource,
    ImageRecord,
    LayoutSlot,
    MatchSettings,
)

BASE = datetime(2026, 7, 17, 12, 0, 0)


def _folder(
    folder_id: str, root: Path, offsets: list[float], *, clock_offset: float = 0.0
) -> FolderSource:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, seconds in enumerate(offsets, start=1):
        path = root / f"frame{ordinal}.png"
        path.touch()
        records.append(
            ImageRecord(ordinal, path, BASE + timedelta(seconds=seconds), "filename")
        )
    return FolderSource(
        id=folder_id,
        path=root,
        name=folder_id,
        records=records,
        start_index=1,
        end_index=len(records),
        offset_seconds=clock_offset,
    )


def _project(tmp_path: Path, *, strict: bool = True) -> ComposerProject:
    master = _folder("master", tmp_path / "master", [0, 1, 2])
    other = _folder("other", tmp_path / "other", [0.25, 1.25, 2.25])
    return ComposerProject(
        canvas=CanvasSettings(width=100, height=80),
        folders=[master, other],
        slots=[
            LayoutSlot.create("master", 1, x=0, y=0, width=50, height=80, z_index=0),
            LayoutSlot.create("other", 1, x=50, y=0, width=50, height=80, z_index=1),
        ],
        matching=MatchSettings(
            master_folder_id="master",
            mode="time",
            tolerance_seconds=0.5,
            strict=strict,
        ),
    )


def test_offset_and_nearest_match_with_candidate_reuse(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.folders[1].offset_seconds = -0.25

    plan = build_match_plan(project)

    assert [frame.matches["other"].record.ordinal for frame in plan] == [1, 2, 3]
    assert all(frame.matches["other"].delta_seconds == 0 for frame in plan)
    assert all(frame.emitted for frame in plan)

    project.folders[1].records = project.folders[1].records[:1]
    project.folders[1].end_index = 1
    reused = build_match_plan(project)
    assert [frame.matches["other"].record.ordinal for frame in reused] == [1, 1, 1]


def test_nearest_tie_chooses_earlier_then_lower_ordinal(tmp_path: Path) -> None:
    folder = _folder("candidate", tmp_path / "candidate", [-1, 1, -1])
    selected = nearest_record(folder, folder.records, BASE)
    assert selected.ordinal == 1


def test_legacy_strict_does_not_remove_master_frames_and_records_overage(
    tmp_path: Path,
) -> None:
    strict_project = _project(tmp_path / "strict", strict=True)
    strict_project.matching.tolerance_seconds = 0.1
    strict_plan = build_match_plan(strict_project)
    assert all(frame.emitted for frame in strict_plan)
    assert all(frame.matches["master"].over_tolerance for frame in strict_plan)
    assert not any(frame.matches["other"].over_tolerance for frame in strict_plan)
    assert [frame.output_frame_index for frame in strict_plan] == [1, 2, 3]

    relaxed = _project(tmp_path / "relaxed", strict=False)
    relaxed.matching.tolerance_seconds = 0.1
    relaxed_plan = build_match_plan(relaxed)
    assert all(frame.emitted for frame in relaxed_plan)
    assert all(frame.matches["master"].over_tolerance for frame in relaxed_plan)


def test_relative_mode_uses_fullest_real_utc_timeline_and_full_coverage(
    tmp_path: Path,
) -> None:
    master = _folder("master", tmp_path / "master", [0, 1, 2, 3, 4])
    other = _folder("other", tmp_path / "other", [0, 10, 20])
    project = ComposerProject(
        canvas=CanvasSettings(width=100, height=80),
        folders=[master, other],
        slots=[
            LayoutSlot.create("master", 1, x=0, y=0),
            LayoutSlot.create("other", 1, x=0, y=0),
        ],
        matching=MatchSettings(master_folder_id="master", mode="relative", strict=True),
    )

    plan = build_match_plan(project)

    assert len(plan) == 5
    assert set(frame.matches["other"].record.ordinal for frame in plan) == {1, 2, 3}
    assert [frame.matches["other"].corrected_timestamp for frame in plan] == sorted(
        frame.matches["other"].corrected_timestamp for frame in plan
    )
    assert all(frame.emitted for frame in plan)


def test_csv_rows_preserve_all_attempts_and_slots_with_precision_flags(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    project.matching.tolerance_seconds = 0.1
    project.slots.append(LayoutSlot.create("other", 2, x=10, y=10, z_index=2))
    plan = build_match_plan(project)
    rows = list(iter_match_rows(project, plan))

    assert len(rows) == len(plan) * len(project.slots)
    assert {row["emitted"] for row in rows} == {True}
    assert {row["output_frame_index"] for row in rows} == {1, 2, 3}
    assert all(row["skip_reason"] == "" for row in rows)
    assert {row["overflow_policy"] for row in rows} == {"keep_flagged"}
    assert all(row["plan_fingerprint"] == plan.fingerprint for row in rows)


def test_invalid_range_is_rejected_without_silent_clamping(tmp_path: Path) -> None:
    project = _project(tmp_path)
    project.folders[1].end_index = 99
    with pytest.raises(MatchPlanError, match="exceeds"):
        build_match_plan(project)


@pytest.mark.parametrize("mode", ["time", "relative"])
def test_auto_master_full_coverage_real_chronology_and_flags(
    tmp_path: Path, mode: str
) -> None:
    sparse = _folder("a-sparse", tmp_path / "sparse", [3, 0])
    fullest = _folder("z-fullest", tmp_path / "fullest", [3, 1, 2, 0, 4])
    project = ComposerProject(
        canvas=CanvasSettings(100, 80),
        folders=[sparse, fullest],
        slots=[
            LayoutSlot.create(folder.id, 1, x=0, y=0) for folder in (sparse, fullest)
        ],
        matching=MatchSettings(
            master_folder_id=sparse.id, mode=mode, tolerance_seconds=0, strict=True
        ),
    )
    plan = freeze_match_plan(project)
    assert plan.master_folder_id == sparse.id
    assert plan.output_folder_id == fullest.id
    assert plan.source_order == (sparse.id, fullest.id)
    assert len(plan) == 5
    assert [frame.output_record.ordinal for frame in plan] == [4, 2, 3, 1, 5]
    assert all(frame.master_record == frame.matches[sparse.id].record for frame in plan)
    assert set(frame.matches[sparse.id].record.ordinal for frame in plan) == {1, 2}
    assert [frame.matches[sparse.id].record.timestamp for frame in plan] == sorted(
        frame.matches[sparse.id].record.timestamp for frame in plan
    )
    assert all(frame.emitted for frame in plan)
    assert any(frame.matches[sparse.id].over_tolerance for frame in plan)
    assert any(frame.matches[sparse.id].reused for frame in plan)
    # Arbitrary backwards browsing reads the fixed plan, without stateful rematching.
    assert select_plan_frame(plan, plan[-1].output_timestamp_text).attempt_index == 5
    assert select_plan_frame(plan, plan[0].output_timestamp_text).attempt_index == 1


def test_integer_nanosecond_delta_and_stale_preview_rejection(tmp_path: Path) -> None:
    project = _project(tmp_path)
    for folder in project.folders:
        record = folder.records[0]
        fraction = "123456789" if folder.id == "master" else "123456790"
        folder.records = [
            ImageRecord(
                record.ordinal,
                record.path,
                record.timestamp,
                "filename",
                f"2026-07-17T12:00:00.{fraction}Z",
            )
        ]
        folder.end_index = 1
    plan = freeze_match_plan(project)
    assert plan[0].matches["master"].delta_ns == -1
    assert plan[0].matches["other"].delta_ns == 0
    assert plan[0].matches["other"].corrected_timestamp_text.endswith(".123456790Z")
    assert project.sequence_plan["rows"][0]["matches"]["other"] == str(
        project.folders[1].records[0].path.resolve()
    )
    project.folders[1].records[0].path.write_bytes(b"changed")
    with pytest.raises(StaleMatchPlanError, match="stale"):
        build_match_plan(project)


def test_new_file_invalidates_a_frozen_catalog(tmp_path: Path) -> None:
    project = _project(tmp_path)
    freeze_match_plan(project)
    (project.folders[0].path / "new.png").touch()
    with pytest.raises(StaleMatchPlanError, match="stale"):
        build_match_plan(project)


def test_explicit_range_edit_builds_a_new_plan_without_mutating_previous_plan(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    original = freeze_match_plan(project)
    original_dict = original.to_dict()
    project.folders[0].start_index = 2

    edited = freeze_match_plan(project)

    assert edited.fingerprint != original.fingerprint
    assert edited.source_fingerprint == original.source_fingerprint
    assert {row.matches["master"].record.ordinal for row in edited} == {2, 3}
    assert original.to_dict() == original_dict


@pytest.mark.parametrize("change", ["file_contents", "directory_membership"])
def test_selection_edit_cannot_hide_changed_observations(
    tmp_path: Path, change: str,
) -> None:
    project = _project(tmp_path)
    freeze_match_plan(project)
    if change == "file_contents":
        project.folders[0].records[0].path.write_bytes(b"changed observation")
    else:
        (project.folders[0].path / "new.png").touch()
    project.folders[0].start_index = 2

    with pytest.raises(StaleMatchPlanError, match="stale"):
        freeze_match_plan(project)


def test_composer_plan_is_deeply_readonly_and_compatible_with_scaling(
    tmp_path: Path,
) -> None:
    import copy
    from dataclasses import FrozenInstanceError

    project = _project(tmp_path)
    plan = freeze_match_plan(project)
    with pytest.raises(FrozenInstanceError):
        plan.fingerprint = "changed"
    with pytest.raises(TypeError):
        plan[0].matches["other"] = plan[0].matches["master"]
    with pytest.raises(TypeError):
        plan.summary["row_count"] = 0
    scaled = copy.deepcopy(project)
    scaled.canvas.width *= 2
    assert build_match_plan(scaled) is plan


@pytest.mark.parametrize("mode", ["time", "relative"])
def test_observed_window_filters_before_master_selection_and_clock_offsets(
    tmp_path: Path, mode: str
) -> None:
    outside_dense = _folder("a", tmp_path / "a", [-2, -1, 0, 1, 2, 3], clock_offset=100)
    inside_dense = _folder("b", tmp_path / "b", [1.25, 0.75, 1])
    project = ComposerProject(
        folders=[outside_dense, inside_dense],
        slots=[
            LayoutSlot.create(folder.id, 1, x=0, y=0)
            for folder in (outside_dense, inside_dense)
        ],
        matching=MatchSettings(
            mode=mode,
            time_start="2026-07-17T12:00:00.5Z",
            time_end="2026-07-17T12:00:02Z",
        ),
    )
    plan = freeze_match_plan(project)
    assert plan.master_folder_id == "a"
    assert plan.output_folder_id == "b"
    assert len(plan) == 3
    assert [row.output_record.ordinal for row in plan] == [2, 3, 1]
    assert [row.master_record.ordinal for row in plan] == [4, 4, 4]
    assert {row.matches["a"].record.ordinal for row in plan} == {4}
    assert all(row.matches["a"].corrected_timestamp.second == 41 for row in plan)
    assert plan.summary["sources"]["a"]["outside_window_frame_count"] == 5
    assert plan.summary["sources"]["b"]["coverage_fraction"] == 1


def test_window_change_rebuilds_even_with_same_frames_and_preserves_nanosecond_boundaries(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    folder = project.folders[0]
    folder.records = [
        ImageRecord(
            index,
            folder.records[index - 1].path,
            BASE,
            "filename",
            f"2026-07-17T12:00:00.{123456777 + 4 * index:09d}Z",
        )
        for index in range(1, 4)
    ]
    project.matching.time_start = "2026-07-17T12:00:00.123456782Z"
    project.matching.time_end = "2026-07-17T12:00:00.123456789Z"
    first = freeze_match_plan(project)
    assert len(first) == 1 and first[0].master_record.ordinal == 2
    assert "other" not in first[0].matches
    assert first.summary["sources"]["other"]["in_window_frame_count"] == 0
    rows = list(iter_match_rows(project, first))
    assert rows[1]["missing_data"] is True
    assert rows[1]["source_file"] == ""
    project.matching.time_start = "2026-07-17T12:00:00.123456783Z"
    second = freeze_match_plan(project)
    assert len(second) == 1
    assert second.fingerprint != first.fingerprint
    assert second[0].master_record.ordinal == 2


def test_empty_window_plan_has_no_outside_fallback_and_clear_preview_error(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    project.matching.time_start = "2026-07-18T00:00:00Z"
    project.matching.time_end = "2026-07-18T00:01:00Z"
    plan = freeze_match_plan(project)
    assert len(plan) == 0
    assert list(iter_match_rows(project, plan)) == []
    with pytest.raises(MatchPlanError, match="No valid images"):
        select_plan_frame(plan, BASE)


@pytest.mark.parametrize("end", ["2026-07-17T12:00:00Z", "2026-07-17T11:59:59Z"])
def test_invalid_window_is_rejected(tmp_path: Path, end: str) -> None:
    project = _project(tmp_path)
    project.matching.time_start = "2026-07-17T12:00:00Z"
    project.matching.time_end = end
    with pytest.raises(MatchPlanError, match="time_start < time_end"):
        build_match_plan(project)


def test_unreadable_secondary_is_empty_valid_stream(tmp_path: Path) -> None:
    project = _project(tmp_path)
    folder = project.folders[1]
    folder.records = [
        ImageRecord(
            record.ordinal,
            record.path,
            record.timestamp,
            record.time_source,
            readable=False,
        )
        for record in folder.records
    ]
    plan = freeze_match_plan(project)
    assert len(plan) == 3
    assert all("other" not in row.matches for row in plan)
    assert plan.summary["sources"]["other"]["input_frame_count"] == 0


@pytest.mark.parametrize("mode", ["time", "relative"])
def test_hierarchy_inherits_fixed_parent_identity_instead_of_rematching_root(
    tmp_path: Path, mode: str
) -> None:
    root = _folder("root", tmp_path / "root", [0, 10])
    middle = _folder("middle", tmp_path / "middle", [0, 6, 10])
    output = _folder("output", tmp_path / "output", [0, 1, 2, 4, 5, 10])
    project = ComposerProject(
        folders=[root, middle, output],
        slots=[
            LayoutSlot.create(folder.id, 1, x=0, y=0)
            for folder in (root, middle, output)
        ],
        matching=MatchSettings(mode=mode),
    )
    plan = freeze_match_plan(project)
    assert plan.master_folder_id == "root"
    assert plan.output_folder_id == "output"
    assert plan.source_order == ("root", "middle", "output")
    assert [row.output_record.ordinal for row in plan] == [1, 2, 3, 4, 5, 6]
    assert [row.matches["middle"].record.ordinal for row in plan] == [1, 1, 1, 2, 2, 3]
    assert [row.master_record.ordinal for row in plan] == [1, 1, 1, 2, 2, 2]
    inherited = plan[3]
    assert inherited.master_record == root.records[1]
    assert (
        nearest_record(root, root.records, output.records[3].timestamp)
        == root.records[0]
    )
    assert inherited.matches["root"].delta_ns == 6_000_000_000
    edge = inherited.parent_matches["middle"]
    assert edge["parent_frame_id"] == str(root.records[1].path.resolve())
    assert edge["child_frame_id"] == str(middle.records[1].path.resolve())
    assert edge == plan[4].parent_matches["middle"]
    assert plan.to_dict()["algorithm_version"] == 3
    assert plan.to_dict()["rows"][3]["output_locator"] == str(
        output.records[3].path.resolve()
    )
    assert plan.to_dict()["rows"][3]["master_locator"] == str(
        root.records[1].path.resolve()
    )
    with pytest.raises(TypeError):
        plan.parent_maps["middle"][0]["parent_frame_id"] = "changed"
    assert all(
        source["coverage_fraction"] == 1 for source in plan.summary["sources"].values()
    )


def test_output_visible_utc_preserves_original_clock_under_offsets(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    project.folders[1].offset_seconds = 100
    plan = freeze_match_plan(project)
    assert plan.output_folder_id == "other"
    assert plan[1].output_timestamp_text == "2026-07-17T12:00:01.250000000Z"
    assert plan[1].output_corrected_timestamp_text == "2026-07-17T12:01:41.250000000Z"
    assert select_plan_frame(plan, "2026-07-17T12:00:01.25Z") is plan[1]
    assert (
        list(iter_match_rows(project, plan))[2]["output_timestamp"]
        == plan[1].output_timestamp_text
    )


def test_legacy_frame_rows_default_output_identity_to_original_master(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    current = freeze_match_plan(project)
    row = current[0]
    legacy = FrameMatch(
        row.attempt_index,
        row.output_frame_index,
        row.master_record,
        row.master_corrected_timestamp,
        row.matches,
        row.emitted,
        row.skip_reason,
    )
    assert legacy.output_record == legacy.master_record
    assert list(iter_match_rows(project, [legacy]))[0]["output_file"] == str(
        legacy.master_record.path
    )
    plan = ComposerMatchPlan(
        (legacy,), "fingerprint", current.master_folder_id, current.summary, "core"
    )
    assert plan.output_folder_id == current.master_folder_id
    assert plan.to_dict()["rows"][0]["output_locator"] == str(
        legacy.master_record.path.resolve()
    )
