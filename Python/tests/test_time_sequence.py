from __future__ import annotations

import datetime as dt
import itertools
import json
import random
import subprocess
import sys
from decimal import Decimal

import pytest

from solar_toolkit.time import (
    TimeFrame,
    TimeStamp,
    TimeStream,
    build_sequence_match_plan,
    filter_time_frames,
    monotone_one_to_one,
    normalize_time,
    seconds_to_ns,
)


def frame(seconds, name, *, match_seconds=None, payload=None):
    base = normalize_time("2020-01-01T04:48:00")
    stamp = base.shift(seconds)
    match = base.shift(match_seconds) if match_seconds is not None else None
    return TimeFrame(name, stamp, payload, match)


def selected(plan, source):
    return [row.selections[source].sorted_index for row in plan.rows]


def test_nanoseconds_tai_timezone_and_exact_integer_roundtrip():
    a = normalize_time("2020-01-01T04:48:00.123456789Z", source="DATE-OBS")
    b = normalize_time("2020-01-01T12:48:00.123456789+08:00")
    c = normalize_time("2020-01-01T04:48:37.123456789_TAI")
    assert a.tai_ns == b.tai_ns == c.tai_ns
    assert a.utc == "2020-01-01T04:48:00.123456789"
    assert c.scale == "tai"
    later = normalize_time("2020-01-01T04:48:00.123456790Z")
    assert later.tai_ns - a.tai_ns == 1
    assert TimeStamp.from_tai_ns(a.tai_ns).tai_ns == a.tai_ns
    assert TimeStamp.from_tai_ns(a.tai_ns).utc == a.utc
    assert a.shift(Decimal("0.000000001")).tai_ns == a.tai_ns + 1
    assert normalize_time(a, source="copied").source == "copied"
    assert seconds_to_ns("0.0000000005") == 0
    assert seconds_to_ns("0.0000000015") == 2


def test_continuous_clock_preserves_utc_leap_second():
    values = [
        normalize_time(value)
        for value in (
            "2016-12-31T23:59:59",
            "2016-12-31T23:59:60",
            "2017-01-01T00:00:00",
        )
    ]
    assert [
        b.tai_ns - a.tai_ns for a, b in zip(values[:-1], values[1:], strict=True)
    ] == [10**9, 10**9]
    assert values[1].utc == "2016-12-31T23:59:60.000000000"


def test_legacy_datetime_with_second_level_timezone_offset_normalizes_to_utc():
    zone = dt.timezone(dt.timedelta(hours=5, minutes=30, seconds=30))
    value = dt.datetime(2020, 1, 1, 10, 18, 30, 123456, tzinfo=zone)
    stamp = normalize_time(value, scale="UTC")
    assert stamp.utc == "2020-01-01T04:48:00.123456000"


@pytest.mark.parametrize(
    "text,utc",
    [
        ("20200101044830 11", "2020-01-01T04:48:30.011000000"),
        ("20200101044842  2", "2020-01-01T04:48:42.002000000"),
        ("20200101044842 123", "2020-01-01T04:48:42.123000000"),
        ("20200101044837 11_TAI", "2020-01-01T04:48:00.011000000"),
    ],
)
def test_dart_complete_compact_timestamp_with_integer_millisecond_field(text, utc):
    stamp = normalize_time(text, source="DATE-OBS")
    assert stamp.raw == text
    assert stamp.utc == utc
    assert stamp.tai_ns == normalize_time(utc).tai_ns


@pytest.mark.parametrize(
    "value",
    [
        "2020-01-01",
        dt.date(2020, 1, 1),
        "2020-01-01T25:00:00",
        "20200101990000",
        float("nan"),
    ],
)
def test_scientific_normalization_rejects_incomplete_or_invalid_times(value):
    with pytest.raises(ValueError):
        normalize_time(value)


def test_importing_time_remains_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import solar_toolkit.time; "
            "import solar_toolkit.time.precision, solar_toolkit.time.sequence; "
            "assert 'astropy' not in sys.modules; assert 'numpy' not in sys.modules; "
            "assert 'sunpy' not in sys.modules",
        ],
        check=True,
    )


def test_full_coverage_longest_axis_keeps_outside_tolerance_and_empty_stream():
    aa = [frame(t, f"a{i}") for i, t in enumerate([0, 10, 20, 30])]
    bb = [frame(t, f"b{i}") for i, t in enumerate([1, 2, 11])]
    plan = build_sequence_match_plan(
        [
            TimeStream("B", bb, tolerance_seconds=1),
            TimeStream("A", aa),
            TimeStream("empty", []),
        ]
    )
    assert plan.master_source_id == "B" and len(plan.rows) == 4
    assert plan.output_source_id == "A"
    assert plan.source_order == ("B", "A")
    assert {row.output_frame.frame_id for row in plan.rows} == {
        f"a{i}" for i in range(4)
    }
    assert all(row.master_frame is row.selections["B"].frame for row in plan.rows)
    indices = selected(plan, "B")
    assert indices == sorted(indices) and set(indices) == {0, 1, 2}
    assert plan.summary["sources"]["B"]["used_frame_count"] == 3
    assert plan.summary["sources"]["B"]["outside_tolerance_count"] > 0
    assert selected(plan, "empty") == [None] * 4
    assert plan.summary["sources"]["empty"]["missing_count"] == 4
    for row in plan.rows:
        choice = row.selections["B"]
        assert choice.frame is not None
        assert choice.use_count == indices.count(choice.sorted_index)
        assert choice.reused == (choice.use_count > 1)
    json.dumps(plan.to_dict(), allow_nan=False)
    assert plan.to_dict()["algorithm_version"] == 3
    assert (
        plan.to_dict()["algorithm"]
        == "hierarchical_monotone_full_coverage_minimum_absolute_delta"
    )
    with pytest.raises(TypeError):
        plan.rows[0].selections["B"] = plan.rows[0].selections["A"]
    with pytest.raises(TypeError):
        plan.summary["sources"]["B"]["frame_count"] = 99
    detached = plan.to_dict()
    detached["summary"]["sources"]["B"]["frame_count"] = 99
    assert plan.summary["sources"]["B"]["frame_count"] == 3


def test_equal_count_single_frame_and_empty_plan_fastpaths():
    aa = [frame(t, f"a{i}") for i, t in enumerate([0, 1, 2])]
    bb = [frame(t, f"b{i}") for i, t in enumerate([100, 101, 102])]
    plan = build_sequence_match_plan(
        [
            TimeStream("B", bb),
            TimeStream("A", aa),
            TimeStream("one", [frame(1000, "only")]),
        ]
    )
    assert plan.master_source_id == "one"
    assert plan.output_source_id == "B"
    assert plan.source_order == ("one", "A", "B")
    assert selected(plan, "B") == [0, 1, 2]
    assert all(row.selections["A"].within_tolerance is False for row in plan.rows)
    assert selected(plan, "one") == [0, 0, 0]
    for empty in (
        build_sequence_match_plan([]),
        build_sequence_match_plan([TimeStream("empty", [])]),
    ):
        assert empty.rows == ()
        assert empty.master_source_id is None
        assert empty.output_source_id is None
        assert empty.source_order == ()
        assert dict(empty.parent_maps) == {}


def test_duplicate_timestamps_keep_identity_and_equal_cost_advances():
    aa = [frame(0, key) for key in ["a3", "a1", "a0", "a2"]]
    bb = [frame(0, "b1"), frame(0, "b0")]
    plan = build_sequence_match_plan([TimeStream("B", bb), TimeStream("A", aa)])
    assert [row.output_frame.frame_id for row in plan.rows] == ["a0", "a1", "a2", "a3"]
    assert selected(plan, "B") == [0, 0, 0, 1]
    assert [row.selections["B"].original_index for row in plan.rows] == [1, 1, 1, 0]
    assert plan.rows[-1].selections["B"].frame.frame_id == "b1"


def test_offsets_and_scientific_cost_never_replace_observed_utc_order():
    aa = [frame(t, f"a{i}") for i, t in enumerate([0, 10, 20])]
    bb = [frame(9, "b0", match_seconds=1), frame(11, "b1", match_seconds=100)]
    streams = [
        TimeStream("A", aa, offset_seconds=2),
        TimeStream("B", bb, offset_seconds=-1, tolerance_seconds=100),
    ]
    plan = build_sequence_match_plan(streams)
    assert selected(plan, "B") == [0, 0, 1]
    assert plan.rows[0].time == aa[0].time
    assert plan.rows[0].match_time.tai_ns == aa[0].time.tai_ns + 2 * 10**9
    choice = plan.rows[0].selections["B"]
    assert choice.frame is bb[0]
    assert choice.match_time.tai_ns == bb[0].match_time.tai_ns - 10**9
    assert choice.delta_ns == -2 * 10**9
    changed = build_sequence_match_plan(
        [streams[0], TimeStream("B", bb, offset_seconds=0)]
    )
    assert changed.fingerprint != plan.fingerprint
    assert (
        build_sequence_match_plan(list(reversed(streams))).fingerprint
        == plan.fingerprint
    )


def test_tolerance_boundary_flags_only_and_fingerprint_tracks_exact_input():
    aa = [frame(0, "a")]
    bb = [frame("0.000001001", "b")]
    outside = build_sequence_match_plan(
        [TimeStream("A", aa, Decimal("0.000001")), TimeStream("B", bb)]
    )
    inside = build_sequence_match_plan(
        [TimeStream("A", aa, Decimal("0.000001001")), TimeStream("B", bb)]
    )
    assert selected(outside, "A") == selected(inside, "A") == [0]
    assert outside.rows[0].selections["A"].delta_ns == -1001
    assert outside.rows[0].selections["A"].within_tolerance is False
    assert inside.rows[0].selections["A"].within_tolerance is True
    assert outside.fingerprint != inside.fingerprint


def test_global_cost_matches_exhaustive_monotone_full_coverage_assignments():
    rng = random.Random(27)
    for _ in range(20):
        queries = sorted(rng.randrange(12) for _ in range(5))
        references = sorted(rng.randrange(12) for _ in range(3))
        aa = [frame(t, f"a{i}") for i, t in enumerate(queries)]
        bb = [frame(t, f"b{i}") for i, t in enumerate(references)]
        plan = build_sequence_match_plan([TimeStream("A", aa), TimeStream("B", bb)])
        costs = []
        for path in itertools.product(range(3), repeat=5):
            if list(path) == sorted(path) and set(path) == {0, 1, 2}:
                costs.append(
                    sum(
                        abs(t - references[j])
                        for t, j in zip(queries, path, strict=True)
                    )
                )
        actual = sum(abs(row.selections["B"].delta_ns) for row in plan.rows)
        assert actual == min(costs) * 10**9


def _brute_surjective_cost(parent_times, child_times):
    """Independent small-instance oracle; enumerate every admissible assignment."""
    return min(
        sum(
            abs(parent_times[parent] - child)
            for parent, child in zip(path, child_times, strict=True)
        )
        for path in itertools.product(range(len(parent_times)), repeat=len(child_times))
        if list(path) == sorted(path) and set(path) == set(range(len(parent_times)))
    )


def test_every_hierarchy_stage_matches_independent_exhaustive_minimum_cost():
    rng = random.Random(931)
    for _ in range(24):
        values = {
            source: sorted(rng.randrange(12) for _ in range(count))
            for source, count in (("root", 2), ("middle", 3), ("leaf", 4))
        }
        streams = [
            TimeStream(
                source,
                [
                    frame(value, f"{source}:{index}")
                    for index, value in enumerate(values[source])
                ],
            )
            for source in ("leaf", "root", "middle")
        ]
        plan = build_sequence_match_plan(streams)
        assert plan.source_order == ("root", "middle", "leaf")
        for parent, child in (("root", "middle"), ("middle", "leaf")):
            edges = plan.parent_maps[child]
            indices = [edge["parent_sorted_index"] for edge in edges]
            assert len(edges) == len(values[child])
            assert indices == sorted(indices)
            assert set(indices) == set(range(len(values[parent])))
            assert (
                sum(abs(edge["delta_ns"]) for edge in edges)
                == _brute_surjective_cost(values[parent], values[child]) * 10**9
            )
            assert [edge["child_sorted_index"] for edge in edges] == list(
                range(len(values[child]))
            )
            for index, edge in enumerate(edges):
                nearest = (
                    min(abs(value - values[child][index]) for value in values[parent])
                    * 10**9
                )
                assert edge["nearest_distance_ns"] == nearest
                assert edge["is_nearest"] == (abs(edge["delta_ns"]) == nearest)
                assert edge["coverage_forced"] == (abs(edge["delta_ns"]) > nearest)


def test_final_ancestors_follow_adjacent_maps_instead_of_direct_leaf_matching():
    root = [frame(value, f"root:{index}") for index, value in enumerate((0, 100))]
    middle = [frame(value, f"middle:{index}") for index, value in enumerate((0, 1, 2))]
    leaf = [frame(value, f"leaf:{index}") for index, value in enumerate((0, 1, 2, 3))]
    plan = build_sequence_match_plan(
        [
            TimeStream("leaf", leaf),
            TimeStream("root", root),
            TimeStream("middle", middle),
        ]
    )
    direct = build_sequence_match_plan(
        [TimeStream("root", root), TimeStream("leaf", leaf)]
    )
    assert plan.master_source_id == "root"
    assert plan.output_source_id == "leaf"
    assert selected(plan, "root") == [0, 0, 1, 1]
    assert selected(direct, "root") == [0, 0, 0, 1]
    assert selected(plan, "middle") == [0, 1, 2, 2]
    assert [row.output_frame for row in plan.rows] == leaf
    assert (
        sum(abs(edge["delta_ns"]) for edge in plan.parent_maps["middle"]) == 99 * 10**9
    )
    assert (
        plan.summary["parent_map_summary"]["middle"]["total_abs_delta_ns"] == 99 * 10**9
    )
    # The repeated last middle frame must not change its own stage's objective.
    assert (
        sum(abs(row.parent_matches["middle"]["delta_ns"]) for row in plan.rows)
        == 197 * 10**9
    )
    for index, row in enumerate(plan.rows):
        leaf_edge = plan.parent_maps["leaf"][index]
        middle_index = leaf_edge["parent_sorted_index"]
        root_edge = plan.parent_maps["middle"][middle_index]
        assert row.parent_matches["leaf"] == leaf_edge
        assert row.parent_matches["middle"] == root_edge
        assert row.master_frame.frame_id == root_edge["parent_frame_id"]
        assert row.master_frame is row.selections["root"].frame
        assert row.time is row.output_frame.time
        assert (
            row.selections["root"].delta_ns
            == row.master_frame.time.tai_ns - row.time.tai_ns
        )


def test_forced_parent_and_tied_nearest_choices_have_distinct_quality_flags():
    plan = build_sequence_match_plan(
        [
            TimeStream("root", [frame(0, "first"), frame(100, "last")]),
            TimeStream("leaf", [frame(index, f"leaf:{index}") for index in range(3)]),
        ]
    )
    forced = plan.parent_maps["leaf"][-1]
    assert forced["parent_frame_id"] == "last"
    assert forced["delta_ns"] == 98 * 10**9
    assert forced["nearest_distance_ns"] == 2 * 10**9
    assert forced["coverage_forced"] is True
    assert forced["is_nearest"] is False
    assert [edge["coverage_forced"] for edge in plan.parent_maps["leaf"]] == [
        False,
        False,
        True,
    ]
    tied = build_sequence_match_plan(
        [
            TimeStream("root", [frame(0, "first"), frame(10, "last")]),
            TimeStream("leaf", [frame(5, f"leaf:{index}") for index in range(3)]),
        ]
    )
    assert selected(tied, "root") == [0, 0, 1]
    assert all(
        edge["is_nearest"] and not edge["coverage_forced"]
        for edge in tied.parent_maps["leaf"]
    )


def test_hierarchy_equal_counts_duplicate_times_and_original_identity_are_stable():
    plan = build_sequence_match_plan(
        [
            TimeStream("Z", [frame(0, "z1"), frame(0, "z0")]),
            TimeStream("M", [frame(0, "m1"), frame(0, "m0")]),
            TimeStream("A", [frame(0, "a1"), frame(0, "a0")]),
        ]
    )
    assert plan.source_order == ("A", "M", "Z")
    assert plan.master_source_id == "A" and plan.output_source_id == "Z"
    assert [row.master_frame.frame_id for row in plan.rows] == ["a0", "a1"]
    assert [row.output_frame.frame_id for row in plan.rows] == ["z0", "z1"]
    for parent, child in (("A", "M"), ("M", "Z")):
        edges = plan.parent_maps[child]
        assert [edge["parent_sorted_index"] for edge in edges] == [0, 1]
        assert [edge["child_sorted_index"] for edge in edges] == [0, 1]
        assert [edge["parent_original_index"] for edge in edges] == [1, 0]
        assert [edge["child_original_index"] for edge in edges] == [1, 0]
        assert all(edge["parent_source_id"] == parent for edge in edges)
        assert all(edge["child_source_id"] == child for edge in edges)
        assert all(edge["is_nearest"] and not edge["coverage_forced"] for edge in edges)


def test_hierarchy_scientific_clocks_minimize_cost_without_reordering_raw_frames():
    parent = [frame(0, "p0", match_seconds=9), frame(1, "p1", match_seconds=0)]
    child = [
        frame(2, "c0", match_seconds=1),
        frame(3, "c1", match_seconds=8),
        frame(4, "c2", match_seconds=2),
    ]
    plan = build_sequence_match_plan(
        [
            TimeStream("parent", parent, offset_seconds=3),
            TimeStream("child", child, offset_seconds=-2),
        ]
    )
    assert [row.output_frame.frame_id for row in plan.rows] == ["c0", "c1", "c2"]
    assert selected(plan, "parent") == [0, 1, 1]
    child_clocks = [-1, 6, 0]
    parent_clocks = [12, 3]
    edges = plan.parent_maps["child"]
    assert (
        sum(abs(edge["delta_ns"]) for edge in edges)
        == _brute_surjective_cost(parent_clocks, child_clocks) * 10**9
    )
    for index, row in enumerate(plan.rows):
        choice = row.selections["parent"]
        assert row.time == child[index].time
        assert row.match_time.tai_ns == child[index].match_time.tai_ns - 2 * 10**9
        assert choice.delta_ns == choice.match_time.tai_ns - row.match_time.tai_ns
        assert edges[index]["parent_time"] == choice.frame.time.to_dict()
        assert edges[index]["child_time"] == child[index].time.to_dict()
        assert edges[index]["parent_match_time"] == choice.match_time.to_dict()
        assert edges[index]["child_match_time"] == row.match_time.to_dict()
    assert [row.time.tai_ns for row in plan.rows] == sorted(
        row.time.tai_ns for row in plan.rows
    )
    assert [row.selections["parent"].frame.time.tai_ns for row in plan.rows] == sorted(
        row.selections["parent"].frame.time.tai_ns for row in plan.rows
    )


def test_hierarchy_metadata_is_immutable_and_json_detaches_edges():
    plan = build_sequence_match_plan(
        [
            TimeStream("root", [frame(0, "r")]),
            TimeStream("leaf", [frame(0, "c0"), frame(1, "c1")]),
        ]
    )
    with pytest.raises(TypeError):
        plan.parent_maps["leaf"][0]["parent_frame_id"] = "changed"
    with pytest.raises(TypeError):
        plan.rows[0].parent_matches["leaf"]["parent_time"]["utc"] = "changed"
    payload = plan.to_dict()
    assert payload["master_source_id"] == "root"
    assert payload["output_source_id"] == "leaf"
    assert payload["source_order"] == ["root", "leaf"]
    assert payload["rows"][0]["output_frame"]["frame_id"] == "c0"
    payload["parent_maps"]["leaf"][0]["parent_frame_id"] = "changed"
    payload["rows"][0]["parent_matches"]["leaf"]["parent_frame_id"] = "changed again"
    assert plan.parent_maps["leaf"][0]["parent_frame_id"] == "r"
    json.dumps(plan.to_dict(), allow_nan=False)


def test_window_filters_every_source_before_master_count_and_preserves_original_indices():
    streams = [
        TimeStream("outside", [frame(-1, f"x{i}") for i in range(20)]),
        TimeStream(
            "a", [frame(2, "a2"), frame(-1, "old"), frame(0, "a0"), frame(1, "a1")]
        ),
        TimeStream("b", [frame(1, "b1"), frame(0, "b0"), frame(3, "end")]),
    ]
    plan = build_sequence_match_plan(
        streams, time_start="2020-01-01T04:48:00", time_end="2020-01-01T04:48:03"
    )
    assert plan.master_source_id == "b" and len(plan.rows) == 3
    assert plan.output_source_id == "a"
    assert plan.source_order == ("b", "a")
    assert [row.output_frame.frame_id for row in plan.rows] == ["a0", "a1", "a2"]
    assert [row.selections["a"].original_index for row in plan.rows] == [2, 3, 0]
    assert set(selected(plan, "b")) == {0, 1}
    assert selected(plan, "outside") == [None] * 3
    counts = plan.summary["sources"]
    assert counts["outside"]["outside_window_frame_count"] == 20
    assert counts["a"]["in_window_frame_count"] == counts["a"]["used_frame_count"] == 3
    assert counts["b"]["coverage_fraction"] == 1
    assert counts["outside"]["coverage_fraction"] is None


def test_nanosecond_window_members_and_scientific_offsets_use_observed_utc():
    inside = frame("0.000000001", "inside", match_seconds=100)
    outside = frame(-1, "outside", match_seconds=0)
    streams = [TimeStream("a", [outside, inside], offset_seconds=1000)]
    plan = build_sequence_match_plan(
        streams,
        time_start="2020-01-01T12:48:00.000000001+08:00",
        time_end="2020-01-01T04:48:37.000000002_TAI",
    )
    assert [row.master_frame.frame_id for row in plan.rows] == ["inside"]
    assert plan.rows[0].time == inside.time
    assert plan.summary["window"]["end_utc"] == "2020-01-01T04:48:00.000000002"
    pair = frame(
        "0.000000001",
        "pair",
        payload={"actual_times_tai_ns": (inside.time.tai_ns, inside.time.tai_ns + 1)},
    )
    assert (
        filter_time_frames(
            [pair], time_start=inside.time, time_end=inside.time.shift("0.000000001")
        )
        == []
    )


def test_window_crosses_midnight_and_preserves_duplicate_identities():
    clocks = [
        "2020-01-01T23:59:59.999999999",
        "2020-01-02T00:00:00",
        "2020-01-02T00:00:00",
        "2020-01-02T00:00:00.000000001",
    ]
    frames = [
        TimeFrame(str(i), normalize_time(clock)) for i, clock in enumerate(clocks)
    ]
    plan = build_sequence_match_plan(
        [TimeStream("a", list(reversed(frames)))],
        time_start=clocks[0],
        time_end=clocks[-1],
    )
    assert [row.master_frame.frame_id for row in plan.rows] == ["0", "1", "2"]


def test_window_fingerprint_changes_even_when_retained_frames_do_not():
    streams = [TimeStream("a", [frame(1, "a")])]
    first = build_sequence_match_plan(
        streams, time_start="2020-01-01T04:48:00", time_end="2020-01-01T04:48:02"
    )
    second = build_sequence_match_plan(
        streams, time_start="2020-01-01T04:47:59", time_end="2020-01-01T04:48:02"
    )
    assert first.rows == second.rows and first.fingerprint != second.fingerprint
    assert first.to_dict()["version"] == 1
    assert first.to_dict()["window"]["boundary"] == "[start,end)"


@pytest.mark.parametrize(
    "start,end", [("04:48:00", "04:48:00"), ("04:48:01", "04:48:00")]
)
def test_invalid_window_is_rejected_even_with_no_sources(start, end):
    with pytest.raises(ValueError, match="time_start < time_end"):
        build_sequence_match_plan(
            [], time_start="2020-01-01T" + start, time_end="2020-01-01T" + end
        )


def test_filtered_1500_600_48_use_all_in_window_data():
    base = normalize_time("2020-01-01T04:48:00")
    streams = [
        TimeStream(
            key, [TimeFrame(f"{key}:{i}", base.shift(i * step)) for i in range(count)]
        )
        for key, count, step in [
            ("dart", 1500, 0.06),
            ("muser", 600, 0.15),
            ("aia", 48, 1.8),
        ]
    ]
    late = base.shift(100)
    streams.append(
        TimeStream("outside", [TimeFrame(f"old:{i}", late) for i in range(2000)])
    )
    plan = build_sequence_match_plan(streams, time_start=base, time_end=base.shift(90))
    assert plan.master_source_id == "aia" and len(plan.rows) == 1500
    assert plan.output_source_id == "dart"
    assert plan.source_order == ("aia", "muser", "dart")
    for key, count in [("dart", 1500), ("muser", 600), ("aia", 48)]:
        indices = selected(plan, key)
        assert indices == sorted(indices) and set(indices) == set(range(count))
        assert plan.summary["sources"][key]["coverage_fraction"] == 1
        times = [row.selections[key].frame.time.tai_ns for row in plan.rows]
        assert times == sorted(times)
    assert plan.summary["sources"]["outside"]["frame_count"] == 0
    assert set(plan.parent_maps) == {"muser", "dart"}
    assert len(plan.parent_maps["muser"]) == 600
    assert len(plan.parent_maps["dart"]) == 1500
    assert {edge["parent_sorted_index"] for edge in plan.parent_maps["muser"]} == set(
        range(48)
    )
    assert {edge["parent_sorted_index"] for edge in plan.parent_maps["dart"]} == set(
        range(600)
    )


def test_cost_is_python_integer_without_int64_overflow():
    def synthetic(value, name):
        return TimeFrame(name, TimeStamp(str(value), "synthetic", value))

    aa = [synthetic(i, f"a{i}") for i in range(4)]
    bb = [synthetic(10**21, "b0"), synthetic(10**21 + 1, "b1")]
    plan = build_sequence_match_plan([TimeStream("A", aa), TimeStream("B", bb)])
    assert plan.summary["sources"]["B"]["total_abs_delta_ns"] > 2**63
    assert isinstance(plan.rows[0].selections["B"].delta_ns, int)


def test_strict_pairs_maximize_count_then_minimize_cost_without_reuse():
    aa = [frame(t, f"a{i}") for i, t in enumerate([0, ".01"])]
    bb = [frame(t, f"b{i}") for i, t in enumerate([0, ".09"])]
    pairs = monotone_one_to_one(aa, bb, tolerance_seconds=".1")
    assert len(pairs) == 2
    aa = [frame(t, f"a{i}") for i, t in enumerate([0, 1, 3])]
    bb = [frame(t, f"b{i}") for i, t in enumerate([1, 3])]
    pairs = monotone_one_to_one(aa, bb, tolerance_seconds=2)
    assert [pair.left.frame_id for pair in pairs] == ["a1", "a2"]
    assert sum(abs(pair.delta_ns) for pair in pairs) == 0
    assert len({pair.right_original_index for pair in pairs}) == len(pairs)


def test_strict_compatibility_and_fastpath_call_only_eligible_pairs():
    aa = [frame(i, f"a{i}", payload=i) for i in range(8)]
    bb = [frame(i, f"b{i}", payload=i) for i in range(8)]
    calls = []

    def compatible(a, b):
        calls.append((a.frame_id, b.frame_id))
        return a.payload == b.payload

    assert (
        len(monotone_one_to_one(aa, bb, tolerance_seconds=0, compatibility=compatible))
        == 8
    )
    assert len(calls) == 8
    calls.clear()
    far = [frame(100, "far")]
    assert not monotone_one_to_one(
        aa, far, tolerance_seconds=0, compatibility=compatible
    )
    assert not calls


def test_strict_pair_score_matches_exhaustive_monotone_pairs():
    rng = random.Random(41)
    for _ in range(16):
        queries = sorted(rng.randrange(8) for _ in range(4))
        references = sorted(rng.randrange(8) for _ in range(3))
        aa = [frame(t, f"a{i}") for i, t in enumerate(queries)]
        bb = [frame(t, f"b{i}") for i, t in enumerate(references)]
        actual = monotone_one_to_one(aa, bb, tolerance_seconds=2)
        expected = (0, 0)
        for count in range(1, 4):
            for left in itertools.combinations(range(4), count):
                for right in itertools.combinations(range(3), count):
                    deltas = [
                        abs(queries[i] - references[j])
                        for i, j in zip(left, right, strict=True)
                    ]
                    if max(deltas) <= 2:
                        expected = max(expected, (count, -sum(deltas) * 10**9))
        assert (len(actual), -sum(abs(pair.delta_ns) for pair in actual)) == expected


@pytest.mark.parametrize(
    "bad", [-1, Decimal("-0.000000000001"), float("inf"), float("nan"), True]
)
def test_invalid_matching_tolerance_is_rejected(bad):
    with pytest.raises(ValueError):
        build_sequence_match_plan([TimeStream("A", [], bad)])
    with pytest.raises(ValueError):
        monotone_one_to_one([], [], tolerance_seconds=bad)


def test_1500_by_600_plan_covers_original_frames():
    base = normalize_time("2020-01-01T04:48:00")

    def stream(name, count, step):
        values = []
        for i in range(count):
            utc = (
                dt.datetime(2020, 1, 1, 4, 48) + dt.timedelta(seconds=i * step)
            ).isoformat()
            stamp = TimeStamp(utc, utc, base.tai_ns + i * step * 10**9)
            values.append(TimeFrame(f"{name}{i:04}", stamp))
        return TimeStream(name, values, tolerance_seconds=1)

    plan = build_sequence_match_plan([stream("A", 1500, 1), stream("B", 600, 2)])
    assert len(plan.rows) == 1500
    indices = selected(plan, "B")
    assert indices == sorted(indices) and set(indices) == set(range(600))
