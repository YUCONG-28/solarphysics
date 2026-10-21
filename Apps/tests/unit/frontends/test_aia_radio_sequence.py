from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.io import fits

from solar_toolkit.aia.background import (
    scan_aia_folder,
    find_nearest_aia,
    read_aia_background,
)
from solar_toolkit.time import normalize_time
from solar_apps.frontends.radio.aia_radio_composite.sequence import (
    build_image_sequence_plan,
)
from solar_apps.workflows.radio.time_matching_acceptance import validate_plan
from solar_toolkit.time import TimeFrame, TimeStream, build_sequence_match_plan
from dataclasses import replace
import pytest
import json


def _candidate(second):
    stamp = normalize_time(f"2020-01-01T04:48:{second:02d}")
    return SimpleNamespace(obs_time=datetime.fromisoformat(stamp.utc), time_stamp=stamp)


def test_aia_can_be_longest_and_all_sources_are_used_without_time_reversal():
    table = pd.DataFrame(
        [
            {
                "path": f"a{i}.fits",
                "wavelength": 171,
                "obs_time": f"2020-01-01T04:48:{second:02d}",
            }
            for i, second in enumerate([33, 30, 31, 32])
        ]
    )
    plan = build_image_sequence_plan(
        {149: (_candidate(30), _candidate(33))},
        table,
        wavelengths=[171, 304],
        aia_tolerance_seconds=0.1,
        radio_tolerance_seconds=0.1,
    )
    assert plan.master_source_id == "radio:149"
    assert plan.output_source_id == "aia:171"
    assert len(plan.rows) == 4
    selections = [row.selections["radio:149"] for row in plan.rows]
    assert len({item.frame.frame_id for item in selections}) == 2
    assert [item.sorted_index for item in selections] == sorted(
        item.sorted_index for item in selections
    )
    assert any(item.within_tolerance is False for item in selections)
    assert all(row.selections["aia:304"].frame is None for row in plan.rows)


def test_half_open_window_preserves_nanosecond_boundaries():
    candidates = []
    for fraction in ("000000001", "000000002", "000000003"):
        stamp = normalize_time(f"2020-01-01T04:48:30.{fraction}")
        candidates.append(
            SimpleNamespace(
                obs_time=datetime.fromisoformat(stamp.utc), time_stamp=stamp
            )
        )
    plan = build_image_sequence_plan(
        {149: candidates},
        pd.DataFrame(),
        wavelengths=[],
        aia_tolerance_seconds=1,
        radio_tolerance_seconds=1,
        time_start="2020-01-01T04:48:30.000000002",
        time_end="2020-01-01T04:48:30.000000003",
    )
    assert len(plan.rows) == 1
    assert plan.rows[0].time.tai_ns == candidates[1].time_stamp.tai_ns


def test_rebuilt_plan_fingerprint_changes_with_image_version():
    table = pd.DataFrame(
        [
            {
                "path": "a.fits",
                "wavelength": 171,
                "obs_time": "2020-01-01T04:48:30",
                "size_bytes": 10,
                "mtime_ns": 100,
            }
        ]
    )

    def build():
        return build_image_sequence_plan(
            {},
            table,
            wavelengths=[171],
            aia_tolerance_seconds=1,
            radio_tolerance_seconds=1,
        )

    before = build().fingerprint
    table.loc[0, "mtime_ns"] = 101
    assert build().fingerprint != before


def test_aia_header_index_preserves_exact_hdu_and_tai_scale(tmp_path):
    header = fits.Header(
        {
            "T_OBS": "2020-01-01T04:48:37.000000001",
            "TIMESYS": "TAI",
            "WAVELNTH": 171,
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
        }
    )
    later = header.copy()
    later["T_OBS"] = "2020-01-01T04:48:38.000000001"
    path = tmp_path / "wrong_20200101_010000.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((4, 4)), header),
            fits.ImageHDU(np.full((4, 4), 7.0), later),
        ]
    ).writeto(path)
    table = scan_aia_folder(tmp_path)
    assert len(table) == 2
    assert table.iloc[1].tai_ns - table.iloc[0].tai_ns == 1_000_000_000
    assert table.iloc[0].utc == "2020-01-01T04:48:00.000000001"
    nearest = find_nearest_aia(table, "2020-01-01T04:48:01.000000001", max_dt_seconds=0)
    assert nearest.hdu_index == 2
    image = read_aia_background(path, hdu_index=nearest.hdu_index, log_scale=False)
    assert image.obs_time == pd.Timestamp(table.iloc[1].utc)


def test_acceptance_validator_requires_complete_logical_frame_coverage():
    def frames(prefix, n):
        return [
            TimeFrame(f"{prefix}{i}", normalize_time(f"2020-01-01T04:48:{30 + i:02d}"))
            for i in range(n)
        ]

    streams = [
        TimeStream("dart", frames("d", 5)),
        TimeStream("muser", frames("m", 3)),
        TimeStream("aia", frames("a", 1)),
    ]
    report = validate_plan(build_sequence_match_plan(streams), streams)
    assert report["output_rows"] == 5
    assert report["used_independent_frames"] == {"dart": 5, "muser": 3, "aia": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta_ns", 123),
        ("within_tolerance", False),
        ("use_count", 99),
        ("reused", False),
        ("original_index", 1),
    ],
)
def test_acceptance_independently_rejects_incorrect_derived_metadata(field, value):
    frames = [
        TimeFrame(str(i), normalize_time(f"2020-01-01T04:48:{30 + i:02d}"))
        for i in range(3)
    ]
    streams = [TimeStream("main", frames), TimeStream("other", frames[:1], 10)]
    plan = build_sequence_match_plan(streams)
    row = plan.rows[0]
    bad = replace(row.selections["other"], **{field: value})
    changed = replace(row, selections={**row.selections, "other": bad})
    invalid = replace(plan, rows=(changed, *plan.rows[1:]))
    with pytest.raises(AssertionError):
        validate_plan(invalid, streams)


def test_acceptance_rereads_original_hdu_and_rejects_changed_timestamp(tmp_path):
    from solar_toolkit.radio.time_matching import index_radio_frames
    from solar_apps.workflows.radio.time_matching_acceptance import (
        verify_exact_reference,
    )

    header = fits.Header(
        {
            "DATE-OBS": "2020-01-01T04:48:30.000000001",
            "FREQ": 149.0,
            "FREQUNIT": "MHz",
            "POLAR": "RR",
            "BUNIT": "K",
        }
    )
    path = tmp_path / "radio.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((3, 3)), header)]).writeto(
        path
    )
    frame = index_radio_frames([path])[0]
    evidence = verify_exact_reference(frame)
    assert evidence["actual_inputs"][0]["locator"][1] == 1
    assert evidence["observed_tai_ns"] == frame.time.tai_ns
    with pytest.raises(AssertionError, match="times differ"):
        verify_exact_reference(replace(frame, time=frame.time.shift(0.000000001)))


def test_acceptance_checks_all_encoder_references_against_plan(tmp_path):
    from solar_apps.workflows.radio.time_matching_acceptance import (
        validate_encoded_references,
    )

    frame = TimeFrame("a", normalize_time("2020-01-01T04:48:30"))
    plan = build_sequence_match_plan([TimeStream("main", [frame])])
    path = tmp_path / "encoded.jsonl"
    record = {
        "row": 1,
        "master_frame_id": "a",
        "output_frame_id": "a",
        "parent_matches": {},
        "selections": {"main": {"frame_id": "a", "observed_tai_ns": frame.time.tai_ns}},
    }
    path.write_text(json.dumps(record) + "\n")
    assert validate_encoded_references(path, plan) == 1
    record["selections"]["main"]["frame_id"] = "another HDU"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(AssertionError, match="references differ"):
        validate_encoded_references(path, plan)


def test_acceptance_requires_actual_stokes_plane_not_only_reference_value():
    from solar_apps.workflows.radio.time_matching_acceptance import _stokes_i

    payload = {
        "header": {
            "NAXIS": 4,
            "CTYPE4": "STOKES",
            "CRVAL4": 1.0,
            "CRPIX4": 2.0,
            "CDELT4": 1.0,
        },
        "plane_indices": (0, 0),
    }
    assert not _stokes_i(payload)
    payload["plane_indices"] = (1, 0)
    assert _stokes_i(payload)


def test_aia_catalog_rejects_tables_and_date_only_observation_fallback(tmp_path):
    header = fits.Header({"DATE-OBS": "2020-01-01", "WAVELNTH": 171})
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="flux", format="E", array=[1.0])]
    )
    fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
        tmp_path / "aia_20200101.fits"
    )
    fits.PrimaryHDU(np.ones((3, 3)), header).writeto(tmp_path / "image_20200101.fits")
    assert scan_aia_folder(tmp_path).empty


def test_event_window_requires_both_original_polarizations_inside(tmp_path):
    from solar_toolkit.radio.time_matching import index_radio_frames
    from solar_apps.workflows.radio.time_matching_acceptance import build_event_plan

    indexed = {"aia:171": [], "aia:304": []}
    for pol, stamp in [
        ("RR", "2020-01-01T04:48:29.999"),
        ("LL", "2020-01-01T04:48:30.003"),
    ]:
        path = tmp_path / pol / "frame.fits"
        path.parent.mkdir()
        header = fits.Header(
            {
                "DATE-OBS": stamp,
                "FREQ": 149.0,
                "FREQUNIT": "MHz",
                "POLAR": pol,
                "BUNIT": "K",
            }
        )
        fits.PrimaryHDU(np.ones((3, 3)), header).writeto(path)
        indexed[f"dart:149:{pol}"] = index_radio_frames([path])
    plan, streams, audit = build_event_plan(
        indexed,
        frequencies=[149],
        start=normalize_time("2020-01-01T04:48:30"),
        end=normalize_time("2020-01-01T04:48:31"),
        include_muser=False,
    )
    assert len(plan.rows) == 0
    assert audit["149"]["member_times_outside_window"] == 0
    assert audit["149"]["pairs"] == 0
    assert audit["149"]["rr_frames"] == 0
    assert audit["149"]["unpaired_ll"] == 1


def test_acceptance_rejects_forged_adjacent_nearest_and_coverage_flags():
    def frames(label, values):
        return [
            TimeFrame(
                f"{label}{second}", normalize_time(f"2020-01-01T04:48:{second:02d}")
            )
            for second in values
        ]

    streams = [
        TimeStream("few", frames("a", [30, 59])),
        TimeStream("many", frames("b", [50, 51, 52])),
    ]
    plan = build_sequence_match_plan(streams)
    assert validate_plan(plan, streams)["passed"]
    edge = dict(plan.parent_maps["many"][0])
    assert edge["coverage_forced"] is True
    edge.update(is_nearest=True, coverage_forced=False)
    rows = [
        (
            replace(row, parent_matches={"many": edge})
            if row.selections["many"].sorted_index == 0
            else row
        )
        for row in plan.rows
    ]
    invalid = replace(
        plan,
        rows=tuple(rows),
        parent_maps={"many": (edge, *plan.parent_maps["many"][1:])},
    )
    with pytest.raises(AssertionError, match="nearest-distance or forced-coverage"):
        validate_plan(invalid, streams)


def test_acceptance_requires_explicit_window_and_retains_nanoseconds():
    from solar_apps.workflows.radio.time_matching_acceptance import _acceptance_windows

    with pytest.raises(ValueError, match="both required"):
        _acceptance_windows()
    assert _acceptance_windows(
        "2020-01-01T04:48:30.000000001Z", "2020-01-01T04:48:30.000000002Z"
    ) == {"burst": ("2020-01-01T04:48:30.000000001Z", "2020-01-01T04:48:30.000000002Z")}


def _acceptance_inputs(tmp_path, output):
    return [
        "--aia-directory",
        str(tmp_path / "aia"),
        "--dart-directory",
        str(tmp_path / "dart"),
        "--muser-directory",
        str(tmp_path / "muser"),
        "--aia-waves",
        "171",
        "304",
        "--frequencies",
        "149",
        "--fov",
        "0",
        "100",
        "-100",
        "0",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize(
    "bounds",
    [
        ["--time-start", "2020-01-01T04:48:30Z"],
        ["--time-end", "2020-01-01T04:48:31Z"],
        ["--time-start", "2020-01-01T04:48:31Z", "--time-end", "2020-01-01T04:48:30Z"],
        [
            "--time-start",
            "2020-01-01T04:48:30.123456789Z",
            "--time-end",
            "2020-01-01T04:48:30.123456789Z",
        ],
        ["--time-start", "2020-01-01", "--time-end", "2020-01-02"],
    ],
)
def test_acceptance_invalid_cli_window_fails_before_audit_or_output(
    tmp_path, monkeypatch, bounds
):
    from solar_apps.workflows.radio import time_matching_acceptance as acceptance

    audited = []
    monkeypatch.setattr(
        acceptance, "audit_event", lambda *_args, **_kwargs: audited.append(True)
    )
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        acceptance.main([*_acceptance_inputs(tmp_path, output), *bounds])
    assert error.value.code == 2
    assert not audited
    assert not output.exists()


def test_acceptance_explicit_burst_window_runs_both_scenarios_and_preserves_nanos(
    tmp_path, monkeypatch
):
    from solar_apps.workflows.radio import time_matching_acceptance as acceptance

    timestamps = [
        "2020-01-01T04:48:29.999999999Z",
        "2020-01-01T04:48:30.123456789Z",
        "2020-01-01T04:48:30.223456789Z",
        "2020-01-01T04:48:30.323456789Z",
    ]
    indexed = {}
    for source in ("aia:171", "aia:304", "dart:149:RR", "dart:149:LL", "muser:149"):
        indexed[source] = []
        for number, value in enumerate(timestamps):
            path = str(tmp_path / f"{source.replace(':', '-')}-{number}.fits")
            payload = {
                "path": path,
                "paths": (path,),
                "hdu_index": 0,
                "plane_indices": (),
                "shape": (3, 3),
                "header": {"BUNIT": "K"},
                "frequency_mhz": 149.0,
                "polarization": source.rsplit(":", 1)[-1],
            }
            indexed[source].append(
                TimeFrame(f"{source}-{number}", normalize_time(value), payload)
            )
    monkeypatch.setattr(
        acceptance, "audit_event", lambda *_args, **_kwargs: (indexed, [])
    )
    encoded = []

    def render(plan, directory, frequencies, include_muser, **_kwargs):
        assert frequencies == (149.0,)
        assert len(plan.rows) == 2
        assert [row.time.utc + "Z" for row in plan.rows] == timestamps[1:3]
        assert all(
            item.frame is None
            or timestamps[1] <= item.frame.time.utc + "Z" < timestamps[3]
            for row in plan.rows
            for item in row.selections.values()
        )
        encoded.append((directory.name, include_muser, plan.fingerprint))
        return {"decoded_frame_count": len(plan.rows)}

    monkeypatch.setattr(acceptance, "render_video", render)
    output = tmp_path / "output"
    acceptance.main(
        [
            *_acceptance_inputs(tmp_path, output),
            "--time-start",
            "2020-01-01T12:48:30.123456789+08:00",
            "--time-end",
            timestamps[3],
        ]
    )
    assert [name for name, _, _ in encoded] == [
        "burst_aia_dart",
        "burst_aia_dart_muser",
    ]
    overview = json.loads((output / "acceptance.json").read_text())
    expected = {
        "time_start": timestamps[1],
        "time_end": timestamps[3],
        "boundary": "[start,end)",
    }
    assert overview["requested_window"] == expected
    assert list(overview["windows"]) == ["burst"]
    assert len(overview["scenarios"]) == 2
    for scenario in overview["scenarios"]:
        assert scenario["requested_window"] == expected
        assert scenario["start_utc"] == timestamps[1]
        assert scenario["end_utc"] == timestamps[3]
        config = json.loads(
            (output / scenario["scenario"] / "resolved_config.json").read_text()
        )
        saved_plan = json.loads(
            (output / scenario["scenario"] / "matching_plan.json").read_text()
        )
        assert config["window_name"] == "burst"
        assert config["window_source"] == "explicit_utc"
        assert config["requested_window"] == expected
        assert config["matching_fingerprint"] == saved_plan["fingerprint"]


@pytest.mark.parametrize(
    "option",
    [
        ["--fps", "0"],
        ["--frequencies", "nan"],
        ["--frequencies", "149", "149"],
        ["--fov", "0", "0", "-1", "1"],
        ["--pair-tolerance-seconds", "-1"],
        ["--aia-waves", "171", "171"],
    ],
)
def test_acceptance_invalid_render_inputs_fail_before_reading(
    tmp_path, monkeypatch, option
):
    from solar_apps.workflows.radio import time_matching_acceptance as acceptance

    monkeypatch.setattr(
        acceptance, "audit_event", lambda *_args, **_kwargs: pytest.fail("read inputs")
    )
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        acceptance.main(
            [
                *_acceptance_inputs(tmp_path, output),
                "--time-start",
                "2020-01-01T00:00:00",
                "--time-end",
                "2020-01-01T00:00:01",
                *option,
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


def test_acceptance_direct_i_uses_all_in_window_frames_without_pairing():
    from solar_apps.workflows.radio.time_matching_acceptance import build_event_plan

    frames = [
        TimeFrame(
            str(i), normalize_time(f"2020-01-01T00:00:0{i}"), {"polarization": "L+R"}
        )
        for i in range(4)
    ]
    indexed = {"aia:171": frames[:1], "aia:304": frames[:1], "dart:149": frames}
    plan, streams, audit = build_event_plan(
        indexed,
        frequencies=[149],
        start=frames[0].time,
        end=frames[3].time,
        include_muser=False,
        dart_mode="I",
    )
    assert len(plan.rows) == 3
    assert validate_plan(plan, streams)["passed"]
    assert audit["149"] == {
        "mode": "direct_stokes_i",
        "pairing_performed": False,
        "input_frames": 4,
        "in_window_frames": 3,
        "outside_window_frames": 1,
    }


def test_acceptance_existing_output_is_never_modified_on_failure(tmp_path):
    from solar_apps.workflows.radio import time_matching_acceptance as acceptance

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        acceptance.main(
            [
                *_acceptance_inputs(tmp_path, output),
                "--time-start",
                "2020-01-01T00:00:00",
                "--time-end",
                "2020-01-01T00:00:01",
            ]
        )
    assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]
    assert marker.read_text() == "original"
