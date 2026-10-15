from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.radio.centers import parse_datetime_value
from solar_toolkit.radio.time_matching import (
    index_radio_frames,
    materialize_radio_frame,
    pair_radio_frames,
)


def image(path, clock, *, frequency=149.0, unit="sfu", crpix=2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header(
        {
            "DATE-OBS": clock,
            "FREQ": frequency,
            "FREQUNIT": "MHz",
            "BUNIT": unit,
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": crpix,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 10.0,
            "CDELT2": 10.0,
        }
    )
    fits.writeto(path, np.ones((4, 4)), header)
    return path


def test_aware_datetime_converts_before_removing_timezone():
    value = datetime(2020, 1, 1, 12, 48, tzinfo=timezone(timedelta(hours=8)))
    assert parse_datetime_value(value) == datetime(2020, 1, 1, 4, 48)


def test_spectrum_sampling_hdu_does_not_compete_with_image_frames(tmp_path):
    spectrum = fits.PrimaryHDU(
        np.ones((20, 40)),
        fits.Header(
            {"CTYPE1": "FREQ", "CTYPE2": "TIME", "DATE-OBS": "2020-01-01T04:48:30"}
        ),
    )
    path = tmp_path / "spectrum.fits"
    spectrum.writeto(path)
    with pytest.raises(ValueError, match="No image HDU"):
        index_radio_frames([path])
    image_header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "DATE-OBS": "2020-01-01T04:48:31",
            "FREQ": 149.0,
            "FREQUNIT": "MHz",
        }
    )
    mixed = tmp_path / "mixed.fits"
    fits.HDUList([spectrum, fits.ImageHDU(np.full((4, 4), 9.0), image_header)]).writeto(
        mixed
    )
    frames = index_radio_frames([mixed])
    assert len(frames) == 1 and frames[0].payload["hdu_index"] == 1
    assert np.all(materialize_radio_frame(frames[0])[0] == 9.0)


def test_duplicate_paths_are_not_independent_frames_or_synthesis_members(tmp_path):
    path = image(tmp_path / "RR" / "frame.fits", "2020-01-01T04:48:30")
    frames = index_radio_frames([path, path])
    assert len(frames) == 1
    with pytest.raises(ValueError, match="only once"):
        pair_radio_frames(frames, frames, tolerance_seconds=1)


def test_legacy_dart_wrapper_maximizes_pairs_and_rejects_repeated_originals():
    from pathlib import Path

    from solar_toolkit.radio.muser_comparison import FrameRef, pair_dart

    left = [
        FrameRef(t, (Path(f"left-{index}"),)) for index, t in enumerate((0.004, 0.005))
    ]
    right = [
        FrameRef(t, (Path(f"right-{index}"),)) for index, t in enumerate((0.0, 0.004))
    ]
    pairs = pair_dart(left, right, tolerance=0.004)
    assert len(pairs) == 2
    assert {pair.paths for pair in pairs} == {
        (left[0].paths[0], right[0].paths[0]),
        (left[1].paths[0], right[1].paths[0]),
    }
    with pytest.raises(ValueError, match="only once"):
        pair_dart([left[0], left[0]], right)


def test_materialized_time_cards_share_true_utc_without_reinterpreting_tai(tmp_path):
    from solar_toolkit.radio.time_matching import observation_stamp

    path = image(tmp_path / "tai.fits", "2020-01-01T04:48:37.000000001")
    fits.setval(path, "T_OBS", value="2020-01-01T04:48:37.000000001")
    fits.setval(path, "TIMESYS", value="TAI")
    frame = index_radio_frames([path])[0]
    _, header = materialize_radio_frame(frame)
    assert header["TIMESYS"] == "UTC"
    assert header["T_OBS"] == header["DATE-OBS"] == "2020-01-01T04:48:00.000000001"
    assert observation_stamp(header).tai_ns == frame.time.tai_ns
    assert frame.payload["header"]["TIMESYS"] == "TAI"


@pytest.mark.parametrize("suffix", [" TAI", "_tai"])
def test_explicit_time_scale_label_takes_precedence_over_header_default(suffix):
    from solar_toolkit.radio.time_matching import observation_stamp

    stamp = observation_stamp(
        {"T_OBS": "2020-01-01T04:48:37.000000001" + suffix, "TIMESYS": "UTC"}
    )
    assert stamp.utc == "2020-01-01T04:48:00.000000001"


def test_materialized_complete_pair_keeps_sum_polarization_and_midpoint(tmp_path):
    rr = image(tmp_path / "RR" / "r.fits", "2020-01-01T04:48:30.000000001")
    ll = image(tmp_path / "LL" / "l.fits", "2020-01-01T04:48:30.000000003")
    pair = pair_radio_frames(
        index_radio_frames([rr]),
        index_radio_frames([ll]),
        tolerance_seconds="0.000000002",
    )[0]
    pixels, header = materialize_radio_frame(pair)
    assert header["POLAR"] == "L+R"
    assert header["DATE-OBS"] == "2020-01-01T04:48:30.000000002"
    assert header["TIMESYS"] == "UTC"
    np.testing.assert_array_equal(pixels, np.full((4, 4), 2.0))


def test_legacy_sum_normalizes_all_observation_cards_after_tai_pairing(tmp_path):
    from solar_toolkit.radio.centers import iter_radio_images, maybe_make_sum_images
    from solar_toolkit.radio.time_matching import observation_stamp

    paths = [
        image(tmp_path / pol / f"{pol}.fits", f"2020-01-01T04:48:37.00000000{ns}")
        for pol, ns in (("RR", 1), ("LL", 3))
    ]
    for path in paths:
        fits.setval(path, "T_OBS", value=fits.getval(path, "DATE-OBS"))
        fits.setval(path, "TIMESYS", value="TAI")
    combined = maybe_make_sum_images(
        [item for path in paths for item in iter_radio_images(path)]
    )[0]
    assert combined.header["TIMESYS"] == "UTC"
    assert combined.header["DATE-OBS"] == combined.header["T_OBS"]
    assert observation_stamp(combined.header).utc == "2020-01-01T04:48:00.000000002"


def test_pairing_preserves_equivalent_bunit_spelling_compatibility(tmp_path):
    rr = image(tmp_path / "RR" / "r.fits", "2020-01-01T04:48:30", unit="sfu")
    ll = image(tmp_path / "LL" / "l.fits", "2020-01-01T04:48:30", unit=" SFU ")
    assert (
        len(
            pair_radio_frames(
                index_radio_frames([rr]), index_radio_frames([ll]), tolerance_seconds=0
            )
        )
        == 1
    )


def test_header_nanoseconds_and_t_obs_are_preserved(tmp_path):
    path = image(tmp_path / "frame.fits", "2020-01-01T04:48:30.123456789")
    fits.setval(path, "T_OBS", value="2020-01-01T04:48:31.123456790_UTC")
    first = index_radio_frames([path])[0]
    assert first.time.utc.endswith("31.123456790")
    assert first.payload["time_source"] == "T_OBS"
    fits.setval(path, "T_OBS", value="2020-01-01T04:48:31.123456791_UTC")
    second = index_radio_frames([path])[0]
    assert second.time.tai_ns - first.time.tai_ns == 1


def test_maximum_pairing_is_one_to_one_and_crosses_midnight(tmp_path):
    left = [
        image(tmp_path / "LL" / f"{index}.fits", clock)
        for index, clock in enumerate(
            ["2020-01-01T23:59:59.999", "2020-01-02T00:00:00.000"]
        )
    ]
    right = [
        image(tmp_path / "RR" / f"{index}.fits", clock)
        for index, clock in enumerate(
            ["2020-01-01T23:59:59.995", "2020-01-01T23:59:59.999"]
        )
    ]
    pairs = pair_radio_frames(
        index_radio_frames(left), index_radio_frames(right), tolerance_seconds=0.004
    )
    assert len(pairs) == 2
    assert len({frame.payload["paths"][1] for frame in pairs}) == 2
    assert pairs[1].payload["pair_delta_ns"] == -1_000_000


def test_pairing_filters_originals_before_optimization_and_rejects_boundary_pairs(
    tmp_path,
):
    left = [
        image(tmp_path / "LL" / "old.fits", "2020-01-01T04:48:29.990"),
        image(tmp_path / "LL" / "in.fits", "2020-01-01T04:48:30.020"),
    ]
    right = [image(tmp_path / "RR" / "in.fits", "2020-01-01T04:48:30.000")]
    pairs = pair_radio_frames(
        index_radio_frames(left),
        index_radio_frames(right),
        tolerance_seconds=0.1,
        time_start="2020-01-01T04:48:30",
        time_end="2020-01-01T04:48:31",
    )
    assert len(pairs) == 1
    assert pairs[0].payload["paths"][0].endswith("in.fits")
    assert pairs[0].payload["pairing_algorithm_version"] == 2
    start_crossing = index_radio_frames([left[0]])
    assert not pair_radio_frames(
        start_crossing,
        index_radio_frames(right),
        tolerance_seconds=0.1,
        time_start="2020-01-01T04:48:30",
        time_end="2020-01-01T04:48:31",
    )
    end_frame = image(tmp_path / "RR" / "end.fits", "2020-01-01T04:48:31")
    near_end = image(tmp_path / "LL" / "near.fits", "2020-01-01T04:48:30.999999999")
    assert not pair_radio_frames(
        index_radio_frames([near_end]),
        index_radio_frames([end_frame]),
        tolerance_seconds=0.1,
        time_start="2020-01-01T04:48:30",
        time_end="2020-01-01T04:48:31",
    )


def test_legacy_pair_wrapper_retains_original_times_and_cannot_leak_over_window(
    tmp_path,
):
    from solar_toolkit.radio.time_matching import frame_for_item
    from solar_toolkit.time import TimeStream, build_sequence_match_plan

    paths = (
        image(tmp_path / "RR" / "old.fits", "2020-01-01T04:48:29.999999999"),
        image(tmp_path / "LL" / "inside.fits", "2020-01-01T04:48:30.000000001"),
    )
    frame = frame_for_item(paths)
    assert frame.payload is paths
    assert len(frame.member_times) == 2
    assert not build_sequence_match_plan(
        [TimeStream("legacy", [frame])],
        time_start="2020-01-01T04:48:30",
        time_end="2020-01-01T04:48:31",
    ).rows


@pytest.mark.parametrize(
    "changed", [{"unit": "Jy"}, {"frequency": 164.0}, {"crpix": 3.0}]
)
def test_incompatible_polarization_pixels_are_not_combined(tmp_path, changed):
    left = image(tmp_path / "LL" / "left.fits", "2020-01-01T04:48:30")
    right = image(tmp_path / "RR" / "right.fits", "2020-01-01T04:48:30", **changed)
    assert (
        pair_radio_frames(
            index_radio_frames([left]),
            index_radio_frames([right]),
            tolerance_seconds=0.01,
        )
        == []
    )


def test_logical_cube_planes_have_distinct_identity_and_exact_pixels(tmp_path):
    path = tmp_path / "cube.fits"
    header = fits.Header(
        {
            "DATE-OBS": "2020-01-01T04:48:30",
            "CTYPE3": "FREQ",
            "CRVAL3": 149.0,
            "CRPIX3": 1.0,
            "CDELT3": 15.0,
            "CUNIT3": "MHz",
            "CTYPE4": "STOKES",
            "CRVAL4": -1.0,
            "CRPIX4": 1.0,
            "CDELT4": -1.0,
        }
    )
    pixels = np.arange(64.0, dtype=float).reshape(2, 2, 4, 4)
    fits.writeto(path, pixels, header)
    frames = index_radio_frames([path])
    assert len(frames) == 4
    assert len({frame.frame_id for frame in frames}) == 4
    selected = index_radio_frames([path], frequency_mhz=164.0, polarization="LL")
    assert len(selected) == 1
    data, _ = materialize_radio_frame(selected[0])
    np.testing.assert_equal(data, pixels[1, 1])
    assert index_radio_frames([path], frequency_mhz=238.0) == []


def test_unknown_cube_axis_is_rejected_and_snapshot_change_is_detected(tmp_path):
    path = image(tmp_path / "plain.fits", "2020-01-01T04:48:30")
    frame = index_radio_frames([path])[0]
    fits.setval(path, "FREQ", value=164.0)
    with pytest.raises(ValueError, match="changed"):
        materialize_radio_frame(frame)
    cube = tmp_path / "unknown.fits"
    fits.writeto(
        cube, np.ones((2, 4, 4)), fits.Header({"DATE-OBS": "2020-01-01T04:48:30"})
    )
    with pytest.raises(ValueError, match="Unsupported"):
        index_radio_frames([cube])


def test_muser_frequency_table_overrides_singleton_frequency_wcs(tmp_path):
    path = tmp_path / "muser.fits"
    header = fits.Header(
        {
            "DATE-OBS": "2020-01-01T04:48:30.123456789",
            "CTYPE3": "FREQ",
            "CRVAL3": 53_710_000.0,
            "CUNIT3": "Hz",
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CTYPE4": "STOKES",
            "CRVAL4": 1.0,
            "CRPIX4": 1.0,
            "CDELT4": 1.0,
            "BUNIT": "K",
        }
    )
    frequencies = np.arange(21.0, dtype=float) * 15 + 53.71
    pixels = np.arange(336.0).reshape(21, 1, 1, 4, 4)
    fits.HDUList(
        [fits.PrimaryHDU(pixels, header), fits.ImageHDU(frequencies, name="FMHZ")]
    ).writeto(path)
    frames = index_radio_frames([path])
    assert len(frames) == 21
    assert {frame.payload["frequency_mhz"] for frame in frames} == set(frequencies)
    selected = index_radio_frames([path], frequency_mhz=frequencies[6])
    assert len(selected) == 1
    data, plane_header = materialize_radio_frame(selected[0])
    np.testing.assert_equal(data, pixels[6, 0, 0])
    assert plane_header["FREQ"] == frequencies[6]
    assert plane_header["FREQUNIT"] == "MHz"
    assert plane_header["BUNIT"] == "K"


def test_singleton_generic_stokes_i_retains_explicit_rr_ll_directory_identity(tmp_path):
    path = tmp_path / "RR" / "image.fits"
    path.parent.mkdir()
    header = fits.Header(
        {
            "DATE-OBS": "2020-01-01T04:48:30",
            "FREQ": 149.0,
            "CTYPE3": "FREQ",
            "CRVAL3": 149.0,
            "CUNIT3": "MHz",
            "CTYPE4": "STOKES",
            "CRVAL4": 1.0,
            "CRPIX4": 1.0,
            "CDELT4": 1.0,
        }
    )
    fits.writeto(path, np.ones((1, 1, 4, 4)), header)
    assert len(index_radio_frames([path], polarization="RR")) == 1
    assert index_radio_frames([path], polarization="LL") == []


def test_materialized_header_preserves_repeated_comment_history_order(tmp_path):
    path = tmp_path / "comments.fits"
    header = fits.Header(
        {"DATE-OBS": "2020-01-01T04:48:30", "FREQ": 149.0, "WCSAXES": 2}
    )
    header.append(("COMMENT", "first comment"))
    header.append(("HISTORY", "first history"))
    header.append(("COMMENT", "second comment"))
    header.append(("HISTORY", "second history"))
    fits.writeto(path, np.ones((4, 4)), header)
    frame = index_radio_frames([path])[0]
    _, restored = materialize_radio_frame(frame)
    expected = [
        (card.keyword, card.value)
        for card in fits.getheader(path).cards
        if card.keyword in {"COMMENT", "HISTORY"}
    ]
    actual = [
        (card.keyword, card.value)
        for card in restored.cards
        if card.keyword in {"COMMENT", "HISTORY"}
    ]
    assert actual == expected
    assert frame.payload["header"]["COMMENT"] == ["first comment", "second comment"]
    assert restored["TIMESYS"] == "UTC"


def test_reindexed_pixel_version_changes_fixed_plan_fingerprint(tmp_path):
    from solar_toolkit.time import TimeStream, build_sequence_match_plan

    path = image(tmp_path / "version.fits", "2020-01-01T04:48:30")
    before = index_radio_frames([path])
    plan_before = build_sequence_match_plan([TimeStream("radio", before)])
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[0, 0] = 99.0
    after = index_radio_frames([path])
    plan_after = build_sequence_match_plan([TimeStream("radio", after)])
    assert before[0].time == after[0].time
    assert before[0].frame_id != after[0].frame_id
    assert plan_before.fingerprint != plan_after.fingerprint
