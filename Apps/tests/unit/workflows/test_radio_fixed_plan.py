import numpy as np
import pytest
from astropy.io import fits
from solar_toolkit.radio.muser_comparison import index_images

from solar_apps.workflows.radio import _overlay_workflow_core as overlay
from solar_apps.workflows.radio import _source_map_selection as inputs
from solar_apps.workflows.radio import (
    source_map_workflow,  # Initializes the public facade.
)
from solar_apps.workflows.radio.muser_video import build_video_match_plan


def frames(directory, seconds, *, frequency=149, wavelength=None):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, second in enumerate(seconds):
        header = fits.Header(
            {
                "DATE-OBS": f"2025-01-24T04:48:{second:02d}.123456789",
                "FREQ": frequency,
                "FREQUNIT": "MHz",
                "BUNIT": "sfu",
            }
        )
        if wavelength is not None:
            header["WAVELNTH"] = wavelength
        path = directory / f"{index}.fits"
        fits.writeto(path, np.ones((4, 4)), header)
        paths.append(str(path))
    return paths


def test_source_slots_keep_longest_axis_and_transport_actual_master_utc(tmp_path):
    short = frames(tmp_path / "149MHz", [30])
    long = frames(tmp_path / "164MHz", [30, 31, 32], frequency=164)
    cfg = {"multi_band_time_tolerance_seconds": 0.01}
    slots = inputs._build_slots_by_common_time([short, long], cfg)
    assert len(slots) == 3
    assert slots[-1].time_match["master_source_id"] == "band:0000"
    assert slots[-1].time_match["output_source_id"] == "band:0001"
    assert slots[-1].time_match["master_time_utc"].endswith("30.123456789Z")
    assert slots[-1].time_match["output_time_utc"].endswith("32.123456789Z")
    assert slots[-1][0] == short[0]
    assert slots[-1].time_match["selections"]["band:0000"]["within_tolerance"] is False
    assert slots[-1].time_match["selections"]["band:0000"]["reused"] is True


def test_source_map_hierarchy_preserves_parent_for_every_reused_child(tmp_path):
    sources = [
        frames(tmp_path / "root", [30, 33]),
        frames(tmp_path / "middle", [30, 31, 33], frequency=164),
        frames(tmp_path / "output", [30, 31, 32, 32, 33], frequency=190),
    ]
    slots = inputs._build_slots_by_common_time(sources, {})
    assert len(slots) == 5
    metadata = [slot.time_match for slot in slots]
    assert metadata[0]["source_order"] == ["band:0000", "band:0001", "band:0002"]
    by_child = {}
    import json

    for record in metadata:
        record = json.loads(json.dumps(record))
        assert record["algorithm_version"] == 3
        assert record["output_source_id"] == "band:0002"
        assert record["master_source_id"] == "band:0000"
        for child, edge in record["parent_matches"].items():
            previous = by_child.setdefault(
                (child, edge["child_frame_id"]), edge["parent_frame_id"]
            )
            assert previous == edge["parent_frame_id"]
            assert record["selections"][child]["parent_match"] == edge
            assert (
                record["selections"][edge["parent_source_id"]]["frame_id"]
                == edge["parent_frame_id"]
            )
    assert len(by_child) == 3 + 5


@pytest.mark.parametrize("version", [1, 2])
def test_source_map_rejects_previous_strategy_frozen_plans(tmp_path, version):
    source = frames(tmp_path / "radio", [30])
    slot = inputs._build_slots_by_common_time([source], {})[0]
    with pytest.raises(ValueError, match="new hierarchy preview"):
        inputs._restore_source_map_slot(
            slot, dict(slot.time_match, algorithm_version=version)
        )


def test_source_map_enumerates_multiple_hdus_and_restores_exact_pixels(tmp_path):
    path = tmp_path / "multi.fits"
    header = fits.Header({"DATE-OBS": "2025-01-24T04:48:30", "FREQ": 149.0})
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((4, 4)), header),
            fits.ImageHDU(np.full((4, 4), 2.0), header),
        ]
    ).writeto(path)
    slots = inputs._build_slots_by_common_time([[str(path)]], {})
    assert len(slots) == 2
    import json
    import pickle

    for index, slot in enumerate(slots):
        restored = inputs._restore_source_map_slot(
            [str(path)], json.loads(json.dumps(slot.time_match))
        )
        pixels, _ = source_map_workflow.read_fits(restored[0])
        assert np.all(pixels == index + 1)
        assert restored[0].logical_frame.payload["hdu_index"] == index
        assert np.array_equal(
            source_map_workflow.read_fits(pickle.loads(pickle.dumps(slot))[0])[0],
            pixels,
        )


def test_source_map_freq_stokes_cube_exact_pairs_survive_json(tmp_path):
    path = tmp_path / "cube.fits"
    data = np.array(
        [
            [[np.full((4, 4), 1.0), np.full((4, 4), 2.0)]],
            [[np.full((4, 4), 10.0), np.full((4, 4), 20.0)]],
        ]
    ).reshape(2, 2, 4, 4)
    header = fits.Header(
        {
            "DATE-OBS": "2025-01-24T04:48:30.123456789",
            "BUNIT": "sfu",
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
    fits.writeto(path, data, header)
    pairs = inputs._strict_single_band_pairs(
        [str(path)], {"time_tolerance_seconds": 0.001}
    )
    assert len(pairs) == 2
    import json

    for pair, total, frequency in zip(pairs, (11.0, 22.0), (149.0, 164.0)):
        metadata = json.loads(json.dumps(inputs._pair_time_metadata(pair, {})))
        restored = inputs._restore_source_map_slot([list(pair)], metadata)[0]
        values = [source_map_workflow.read_fits(member) for member in restored]
        assert np.all(values[0][0] + values[1][0] == total)
        assert all(value[1]["FREQ"] == frequency for value in values)
        assert restored.logical_frame.time.utc.endswith("30.123456789")


def test_source_map_band_filter_counts_only_selected_frequency_planes(tmp_path):
    root = tmp_path / "bands"
    directory = root / "149MHz" / "RR"
    directory.mkdir(parents=True)
    path = directory / "cube.fits"
    header = fits.Header(
        {
            "DATE-OBS": "2025-01-24T04:48:30",
            "POLAR": "RR",
            "BUNIT": "sfu",
            "CTYPE3": "FREQ",
            "CRVAL3": 149.0,
            "CRPIX3": 1.0,
            "CDELT3": 15.0,
            "CUNIT3": "MHz",
        }
    )
    fits.writeto(path, np.stack([np.ones((4, 4)), np.full((4, 4), 2.0)]), header)
    cfg = dict(
        source_map_workflow.DEFAULT_CONFIG,
        multi_band_root=str(root),
        multi_band_freqs=[149],
        band_dir_pattern="{freq}MHz/{polar}",
        polarization="RR",
        combine_polarizations=False,
        enable_raw_quality_filter=False,
        start_idx=0,
        end_idx=None,
        study_mode="exploratory",
    )
    slots = inputs._build_multi_band_slots(cfg)
    assert len(slots) == 1
    selected = slots[0].time_match["selections"]["band:0000"]
    assert selected["locator"]["frequency_mhz"] == 149.0
    assert selected["locator"]["plane_indices"] == (0,)
    assert (
        cfg["_radio_time_match_plan"]["summary"]["sources"]["band:0000"]["frame_count"]
        == 1
    )


def test_source_map_pairs_frequency_hdus_independently_of_storage_order(tmp_path):
    paths = []
    for pol, frequencies in (("RR", (149.0, 164.0)), ("LL", (164.0, 149.0))):
        directory = tmp_path / "bands" / pol
        directory.mkdir(parents=True)
        headers = [
            fits.Header(
                {
                    "DATE-OBS": "2025-01-24T04:48:30",
                    "POLAR": pol,
                    "FREQ": frequency,
                    "FREQUNIT": "MHz",
                    "BUNIT": "sfu",
                }
            )
            for frequency in frequencies
        ]
        path = directory / "multi.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(np.ones((4, 4)), headers[0]),
                fits.ImageHDU(np.ones((4, 4)), headers[1]),
            ]
        ).writeto(path)
        paths.append(str(path))
    pairs = inputs._match_rr_ll_by_time(paths[:1], paths[1:])
    assert len(pairs) == 2
    for pair in pairs:
        members = pair.logical_frame.payload["pair_members"]
        assert members[0]["frequency_mhz"] == members[1]["frequency_mhz"]
        assert members[0]["hdu_index"] != members[1]["hdu_index"]


def test_source_map_workspace_selection_preserves_exact_cube_plane_and_rejects_stale(
    tmp_path,
):
    path = tmp_path / "cube.fits"
    header = fits.Header(
        {
            "DATE-OBS": "2025-01-24T04:48:30",
            "POLAR": "RR",
            "CTYPE3": "FREQ",
            "CRVAL3": 149.0,
            "CRPIX3": 1.0,
            "CDELT3": 15.0,
            "CUNIT3": "MHz",
        }
    )
    fits.writeto(path, np.stack([np.ones((4, 4)), np.full((4, 4), 2.0)]), header)
    selected = inputs._index_source_items([str(path)], {})[1]
    metadata = inputs._pair_time_metadata(selected, {})
    selection = {
        "mode": "single_band",
        "items": [
            {"run_path": str(path), "paths": [str(path)], "time_match": metadata}
        ],
    }
    cfg = {"combine_polarizations": False, "polarization": "RR"}
    restored = inputs._prepare_single_band_inputs(
        inputs._selected_workspace_files(selection, cfg), cfg
    )
    assert len(restored) == 1
    assert np.all(source_map_workflow.read_fits(restored[0])[0] == 2.0)
    import os

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="Planned radio source changed"):
        inputs._selected_workspace_files(selection, cfg)


def test_radio_filename_integer_millis_and_nanoseconds_do_not_truncate():
    millis = inputs._radio_stamp_from_filename(
        "149MHz_2025124_044829_13.fits", {"date_format": "auto"}
    )
    nano = inputs._radio_stamp_from_filename(
        "149MHz_2025124_044829_123456789.fits", {"date_format": "auto"}
    )
    assert millis.utc == "2025-01-24T04:48:29.013000000"
    assert nano.utc == "2025-01-24T04:48:29.123456789"


def test_source_map_rejects_unresolved_time_axis_without_positional_fallback(tmp_path):
    path = tmp_path / "time_cube.fits"
    fits.writeto(
        path,
        np.ones((2, 4, 4)),
        fits.Header(
            {"DATE-OBS": "2025-01-24T04:48:30", "CTYPE3": "TIME", "FREQ": 149.0}
        ),
    )
    with pytest.raises(ValueError, match="Unsupported.*image axis"):
        inputs._build_slots_by_common_time([[str(path)]], {})


def test_overlay_aia_can_be_longest_and_outside_tolerance_rows_remain(tmp_path):
    radio = tmp_path / "radio"
    frames(radio / "149MHz" / "RR", [30])
    aia = tmp_path / "aia"
    frames(aia, [30, 31, 32], wavelength=171)
    cfg = overlay.Config()
    cfg.selected_bands = ["149MHz"]
    cfg.radio_base_dir = str(radio)
    cfg.aia_base_dir = str(aia)
    cfg.aia_wavelength = "171"
    cfg.aia_file_start_idx = cfg.aia_file_end_idx = None
    cfg.combine_polarizations = False
    cfg.polarization_mode = "RR"
    cfg.overlay_hmi = False
    matched = overlay.build_matched_pairs(cfg)
    assert len(matched) == 3
    slot = matched[-1][2][0][1]
    assert slot.time_match["master_source_id"] == "radio:149MHz"
    assert slot.time_match["output_source_id"] == "aia:171"
    assert overlay._slot_reference_time(slot).second == 32
    assert slot.time_match["selections"]["radio:149MHz"]["within_tolerance"] is False


def test_overlay_filters_cube_frequency_and_preserves_exact_hdu(tmp_path):
    radio = tmp_path / "radio" / "149MHz" / "RR"
    radio.mkdir(parents=True)
    header = fits.Header(
        {
            "DATE-OBS": "2025-01-24T04:48:30.123456789",
            "POLAR": "RR",
            "BUNIT": "sfu",
            "CTYPE3": "FREQ",
            "CRVAL3": 149.0,
            "CRPIX3": 1.0,
            "CDELT3": 15.0,
            "CUNIT3": "MHz",
        }
    )
    data = np.stack([np.ones((4, 4)), np.full((4, 4), 2.0)])
    path = radio / "multi.fits"
    fits.HDUList(
        [fits.PrimaryHDU(data, header), fits.ImageHDU(data * 3, header)]
    ).writeto(path)
    cfg = overlay.Config()
    cfg.selected_bands = ["149MHz"]
    cfg.radio_base_dir = str(tmp_path / "radio")
    cfg.radio_start_idx = cfg.radio_end_idx = None
    cfg.combine_polarizations = False
    cfg.polarization_mode = "RR"
    cfg.use_radec_maps = False
    slots = overlay.build_radio_time_slots_for_overlay(cfg)
    assert len(slots) == 2
    values = []
    for _, slot in slots:
        item = slot["149MHz"][0][0]
        values.append(overlay.extract_radio_2d_data(item, cfg=cfg)[0])
        assert slot.time_match["selections"]["radio:149MHz"]["locator"][
            "plane_indices"
        ] == (0,)
    assert np.all(values[0] == 1.0)
    assert np.all(values[1] == 3.0)


def test_muser_video_does_not_fix_first_dart_as_master_or_drop_outliers(tmp_path):
    muser = index_images(frames(tmp_path / "muser", [30]))
    dart = [index_images(frames(tmp_path / "dart", [30]))]
    aia = [
        index_images(frames(tmp_path / "aia171", [30, 31, 32], wavelength=171)),
        index_images(frames(tmp_path / "aia304", [30], wavelength=304)),
    ]
    plan = build_video_match_plan({"frequencies_mhz": [149]}, muser, dart, aia)
    assert plan.master_source_id == "AIA:304"
    assert plan.output_source_id == "AIA:171"
    assert len(plan.rows) == 3
    assert plan.rows[-1].selections["MUSER:149"].within_tolerance is False
    assert plan.rows[-1].selections["DART:149"].frame.payload is dart[0][0]
    assert plan.rows[-1].time.utc.endswith("32.123456789")


def test_source_map_windows_every_source_before_counting_and_covers_all_inside(
    tmp_path,
):
    first = frames(tmp_path / "first", [28, 29, 30, 31, 32, 33])
    second = frames(tmp_path / "second", [30, 30, 31], frequency=164)
    outside = frames(tmp_path / "outside", [28], frequency=190)
    cfg = {
        "image_time_start": "2025-01-24T04:48:30.123456789Z",
        "image_time_end": "2025-01-24T04:48:32.123456789Z",
    }
    slots = inputs._build_slots_by_common_time([first, second, outside], cfg)
    assert len(slots) == 3
    plan = cfg["_radio_time_match_plan"]
    assert plan["master_source_id"] == "band:0000"
    assert plan["output_source_id"] == "band:0001"
    assert plan["summary"]["sources"]["band:0000"]["outside_window_frame_count"] == 4
    assert plan["summary"]["sources"]["band:0000"]["coverage_fraction"] == 1
    assert all(
        slot.time_match["selections"]["band:0002"]["frame_id"] is None for slot in slots
    )
    assert {
        slot.time_match["selections"]["band:0000"]["locator"]["time_utc"]
        for slot in slots
    } == {"2025-01-24T04:48:30.123456789Z", "2025-01-24T04:48:31.123456789Z"}


def test_source_map_filters_polarizations_before_pairing(tmp_path):
    rr = frames(tmp_path / "RR", [29, 30])
    ll = frames(tmp_path / "LL", [30])
    with fits.open(rr[0], mode="update") as hdus:
        hdus[0].header["DATE-OBS"] = "2025-01-24T04:48:29.999"
    with fits.open(rr[1], mode="update") as hdus:
        hdus[0].header["DATE-OBS"] = "2025-01-24T04:48:30.010"
    with fits.open(ll[0], mode="update") as hdus:
        hdus[0].header["DATE-OBS"] = "2025-01-24T04:48:30.001"
    cfg = {
        "image_time_start": "2025-01-24T04:48:30Z",
        "image_time_end": "2025-01-24T04:48:31Z",
    }
    pairs = inputs._match_rr_ll_by_time(rr, ll, 100.0, cfg, frequency_mhz=149)
    assert len(pairs) == 1
    assert str(pairs[0][0]) == rr[1]
    assert pairs[0].logical_frame.payload["pairing_algorithm_version"] == 2


def test_source_map_saved_selection_rejects_changed_window(tmp_path):
    files = frames(tmp_path / "radio", [30])
    cfg = {
        "image_time_start": "2025-01-24T04:48:30Z",
        "image_time_end": "2025-01-24T04:48:32Z",
    }
    item = inputs._index_source_items(files, cfg)[0]
    metadata = inputs._pair_time_metadata(item, cfg)
    with pytest.raises(ValueError, match="UTC interval changed"):
        inputs._restore_source_map_slot(
            files, metadata, cfg=dict(cfg, image_time_end="2025-01-24T04:48:33Z")
        )


def test_overlay_windows_aia_and_radio_without_using_spectrogram_bounds(tmp_path):
    frames(tmp_path / "radio" / "149MHz" / "RR", [29, 30, 31, 32])
    frames(tmp_path / "aia", [30, 30, 31], wavelength=171)
    cfg = overlay.Config()
    cfg.selected_bands = ["149MHz"]
    cfg.radio_base_dir = str(tmp_path / "radio")
    cfg.aia_base_dir = str(tmp_path / "aia")
    cfg.aia_wavelength = "171"
    cfg.aia_file_start_idx, cfg.aia_file_end_idx = 2, 3
    cfg.radio_start_idx, cfg.radio_end_idx = 2, 3
    cfg.combine_polarizations = False
    cfg.polarization_mode = "RR"
    cfg.overlay_hmi = False
    cfg.image_time_start = "2025-01-24T04:48:30.123456789Z"
    cfg.image_time_end = "2025-01-24T04:48:32.123456789Z"
    cfg.spectrogram_time_start = "2025-01-24T04:48:28Z"
    cfg.spectrogram_time_end = "2025-01-24T04:48:34Z"
    matched = overlay.build_matched_pairs(cfg)
    assert len(matched) == 3
    plan = cfg._time_match_plan
    assert plan["master_source_id"] == "radio:149MHz"
    assert plan["output_source_id"] == "aia:171"
    assert plan["summary"]["sources"]["radio:149MHz"]["outside_window_frame_count"] == 2
    assert all(
        slot.time_match["selections"]["radio:149MHz"]["actual_time_utc"]
        < cfg.image_time_end
        for _, _, groups in matched
        for _, slot in groups
    )


def test_source_map_utc_window_bypasses_legacy_event_indices(tmp_path):
    root = tmp_path / "bands"
    frames(root / "149MHz" / "RR", [29, 30, 31, 32])
    cfg = dict(
        source_map_workflow.DEFAULT_CONFIG,
        multi_band_root=str(root),
        multi_band_freqs=[149],
        band_dir_pattern="{freq}MHz/{polar}",
        polarization="RR",
        combine_polarizations=False,
        enable_raw_quality_filter=False,
        start_idx=1588,
        end_idx=1666,
        study_mode="exploratory",
        image_time_start="2025-01-24T04:48:30Z",
        image_time_end="2025-01-24T04:48:32Z",
    )
    slots = inputs._build_multi_band_slots(cfg)
    assert len(slots) == 2
    assert {slot.time_match["output_time_utc"] for slot in slots} == {
        "2025-01-24T04:48:30.123456789Z",
        "2025-01-24T04:48:31.123456789Z",
    }
    cfg.update(image_window_use_indices=True, start_idx=2, end_idx=3)
    assert len(inputs._build_multi_band_slots(cfg)) == 1


def test_frozen_collection_selection_supplies_image_window(tmp_path):
    import json

    directory = tmp_path / "frozen" / "149MHz" / "RR"
    files = frames(directory, [29, 30, 31])
    (tmp_path / "frozen" / ".frozen-collection-v1.json").write_text(
        json.dumps(
            {
                "schema": "solar-radio-frozen-collection-v1",
                "selection": {
                    "start_utc": "2025-01-24T04:48:30Z",
                    "end_utc": "2025-01-24T04:48:31Z",
                },
            }
        )
    )
    items = inputs._index_source_items(files, {})
    assert len(items) == 1
    assert items[0].logical_frame.time.utc.endswith("30.123456789")


def test_muser_video_explicit_half_open_window_beats_spectrum_bounds(tmp_path):
    muser = index_images(frames(tmp_path / "muser", [28, 29, 30, 31, 32]))
    dart = [index_images(frames(tmp_path / "dart", [30, 30, 31]))]
    aia = [index_images(frames(tmp_path / "aia171", [28, 32], wavelength=171)), []]
    config = {
        "frequencies_mhz": [149],
        "image_time_start": "2025-01-24T04:48:30.123456789Z",
        "image_time_end": "2025-01-24T04:48:32.123456789Z",
    }
    plan = build_video_match_plan(config, muser, dart, aia, bounds=[0.0, 1.0])
    assert len(plan.rows) == 3
    assert plan.master_source_id == "MUSER:149"
    assert plan.output_source_id == "DART:149"
    assert all(row.selections["AIA:171"].frame is None for row in plan.rows)
    assert (
        plan.to_dict()["summary"]["sources"]["MUSER:149"]["outside_window_frame_count"]
        == 3
    )


def test_muser_preview_reuses_exact_plan_row_even_when_sources_are_empty(tmp_path):
    from astropy.time import Time

    from solar_apps.workflows.radio.muser_preview import _select_plan_preview

    muser = index_images(frames(tmp_path / "muser", [30, 31, 32]))
    aia = [index_images(frames(tmp_path / "aia", [30], wavelength=171)), []]
    plan = build_video_match_plan({"frequencies_mhz": [149]}, muser, [[]], aia)
    spectrum_times = Time(
        ["2025-01-24T04:48:30", "2025-01-24T04:48:31.9"], scale="utc"
    ).unix
    source_ids = ["MUSER:149", "DART:149", "AIA:171", "AIA:304"]
    _, refs, score_index, row_index = _select_plan_preview(
        plan, spectrum_times, np.array([1.0, 10.0]), source_ids
    )
    assert row_index == 2
    assert score_index == 1
    for source, ref in zip(source_ids, refs):
        selected = plan.rows[row_index].selections[source].frame
        assert ref is (selected.payload if selected else None)


def test_event_validator_preserves_original_indices_after_window_filtering(tmp_path):
    from solar_toolkit.radio.time_matching import index_radio_frames
    from solar_toolkit.time import normalize_time

    from solar_apps.workflows.radio.time_matching_acceptance import (
        build_event_plan,
        validate_plan,
    )

    indexed = {
        "aia:171": index_radio_frames(
            frames(tmp_path / "aia171", [28, 30, 31], wavelength=171)
        ),
        "aia:304": index_radio_frames(
            frames(tmp_path / "aia304", [28], wavelength=304)
        ),
        "dart:149:RR": index_radio_frames(
            frames(tmp_path / "RR", [30]), polarization="RR"
        ),
        "dart:149:LL": index_radio_frames(
            frames(tmp_path / "LL", [30]), polarization="LL"
        ),
    }
    plan, originals, _ = build_event_plan(
        indexed,
        frequencies=[149],
        start=normalize_time("2025-01-24T04:48:30Z"),
        end=normalize_time("2025-01-24T04:48:32Z"),
        include_muser=False,
    )
    assert plan.master_source_id == "dart:149"
    assert plan.output_source_id == "aia:171"
    assert plan.rows[0].selections["aia:171"].original_index == 1
    assert len(originals[0].frames) == 3
    verification = validate_plan(plan, originals)
    assert verification


def test_muser_video_materializes_aia_with_exact_identity(tmp_path, monkeypatch):
    import json

    import matplotlib.pyplot as plt
    from solar_toolkit.radio import time_matching

    from solar_apps.workflows.radio.muser_video import write_video

    path = frames(tmp_path / "aia", [30], wavelength=171)[0]
    with fits.open(path, mode="update") as hdus:
        hdus[0].header.update(
            {
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CRPIX1": 2.5,
                "CRPIX2": 2.5,
                "CRVAL1": 0.0,
                "CRVAL2": 0.0,
                "CDELT1": 300.0,
                "CDELT2": 300.0,
                "RSUN_OBS": 960.0,
                "RSUN_REF": 695700000.0,
                "DSUN_OBS": 149597870700.0,
                "HGLN_OBS": 0.0,
                "HGLT_OBS": 0.0,
                "TELESCOP": "SDO/AIA",
            }
        )
    reference = index_images([path])[0]
    original_reader = time_matching.materialize_radio_frame
    calls = []

    def read_exact(frame):
        calls.append(frame.frame_id)
        assert frame.frame_id == reference.frame_id
        return original_reader(frame)

    monkeypatch.setattr(time_matching, "materialize_radio_frame", read_exact)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis in axes:
        axis.pcolormesh(np.ones((4, 4)))
    fig.suptitle("Preview")
    output = tmp_path / "video"
    output.mkdir()
    config = {
        "frequencies_mhz": [149],
        "video_fps": 10,
        "fov_arcsec": [-1000, 1000, -1000, 1000],
        "label": "Exact AIA reference",
    }
    try:
        video = write_video(
            fig, config, output, [], [[]], [[reference], []], None, [], ["cyan"]
        )
        assert video.is_file()
        assert calls == [reference.frame_id]
        record = json.loads((output / "video.json").read_text())
        assert record["frames"] == 1
        assert (
            record["frame_records"][0]["time_match"]["selections"]["AIA:171"][
                "frame_id"
            ]
            == reference.frame_id
        )
    finally:
        plt.close(fig)


def test_muser_missing_frequency_is_counted_and_matched_as_its_own_layer(tmp_path):
    """One missing band must not hide usable images in another MUSER band."""
    from solar_toolkit.radio.time_matching import materialize_radio_frame
    from solar_toolkit.time import TimeFrame

    paths = []
    for second, frequencies in [(30, (149, 164)), (31, (149,)), (32, (149,))]:
        hdus = []
        for index, frequency in enumerate(frequencies):
            header = fits.Header(
                {
                    "DATE-OBS": f"2000-01-01T04:48:{second}",
                    "FREQ": frequency,
                    "FREQUNIT": "MHz",
                    "POLAR": "I",
                }
            )
            constructor = fits.PrimaryHDU if index == 0 else fits.ImageHDU
            hdus.append(constructor(np.full((4, 4), frequency), header))
        path = tmp_path / f"frame-{second}.fits"
        fits.HDUList(hdus).writeto(path)
        paths.append(path)
    refs = index_images(paths)
    plan = build_video_match_plan(
        {"frequencies_mhz": [149, 164]}, refs, [[], []], [[], []]
    )
    assert len(plan.rows) == 3
    assert plan.master_source_id == "MUSER:164"
    assert plan.output_source_id == "MUSER:149"
    assert plan.summary["sources"]["MUSER:164"]["frame_count"] == 1
    assert plan.summary["sources"]["MUSER:149"]["frame_count"] == 3
    assert plan.summary["sources"]["MUSER:164"]["extra_uses"] == 2
    for row in plan.rows:
        ref = row.selections["MUSER:164"].frame.payload
        pixels, _ = materialize_radio_frame(
            TimeFrame(ref.frame_id, ref.time, ref.payload)
        )
        assert np.all(pixels == 164)
        assert ref.time.utc.endswith("30.000000000")
    bounded = build_video_match_plan(
        {
            "frequencies_mhz": [149, 164],
            "image_time_start": "2000-01-01T04:48:31Z",
            "image_time_end": "2000-01-01T04:48:33Z",
        },
        refs,
        [[], []],
        [[], []],
    )
    assert len(bounded.rows) == 2
    assert all(row.selections["MUSER:164"].frame is None for row in bounded.rows)
    assert bounded.summary["sources"]["MUSER:149"]["coverage_fraction"] == 1
