"""Sequence contracts for the Radio Composite Figure frontend."""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from solar_apps.frontends.radio.composite_figure import composite_sequence
from solar_apps.frontends.radio.composite_figure.composite_sequence import (
    SEQUENCE_SCHEMA_VERSION,
    CompositeSequenceCancelled,
    SequenceExportOptions,
    common_candidate_time_coverage,
    export_composite_sequences,
    group_candidates_by_frequency,
    prepare_single_panel_render,
    resolve_single_band_frequency_source,
    render_source_map_candidate,
    roi_intersects_source_map,
    select_sequence_candidates,
)
from solar_apps.frontends.radio.source_map import worker as source_map_worker
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandCurve,
    DartNarrowbandResult,
)
from solar_toolkit.radio.roi_lightcurve import RadioRoi
from solar_toolkit.visualization import media


def _candidate(frequency: float, observed: datetime, index: int) -> dict:
    return {
        "id": f"{frequency:g}-{index}",
        "mode": "single_band",
        "paths": [f"{frequency:g}-{index}.fits"],
        "run_path": f"{frequency:g}-{index}.fits",
        "frequencies_mhz": [frequency],
        "observation_time": observed.isoformat(),
    }


def _map_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(output, format="PNG")
    return output.getvalue()


def _map_metadata() -> dict:
    return {
        "schema_version": 1,
        "image": {"width": 200, "height": 100, "sha256": "synthetic"},
        "panels": [
            {
                "id": "radio-0",
                "bbox_normalized": [0.1, 0.1, 0.9, 0.9],
                "xlim_arcsec": [-10.0, 10.0],
                "ylim_arcsec": [-5.0, 5.0],
            }
        ],
    }


def _radio_curve(start: datetime, frequency: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": [
                (start + timedelta(seconds=index)).isoformat() for index in range(3)
            ],
            "freq_mhz": [frequency] * 3,
            "polarization": ["RR"] * 3,
            "raw_sum": [10.0 + index for index in range(3)],
            "valid_pixel_count": [4, 4, 4],
            "quality_flag": ["ok", "ok", "ok"],
            "bunit": ["K", "K", "K"],
        }
    )


def _dart(
    start: datetime,
    frequency: float = 149.0,
    *,
    bandwidth: float = 2.0,
    offset: float = 20.0,
) -> DartNarrowbandResult:
    half_width = bandwidth / 2.0
    return DartNarrowbandResult(
        time_utc=tuple(start + timedelta(seconds=index) for index in range(3)),
        curves=(
            DartNarrowbandCurve(
                center_frequency_mhz=frequency,
                bandwidth_mhz=bandwidth,
                requested_frequency_range_mhz=(
                    frequency - half_width,
                    frequency + half_width,
                ),
                sampled_frequency_range_mhz=(
                    frequency - half_width / 2.0,
                    frequency + half_width / 2.0,
                ),
                channel_count=2,
                stokes_i_db=np.asarray([offset, offset + 1.0, offset + 2.0]),
            ),
        ),
    )


def test_single_band_source_resolves_event_band_and_polarization_levels(
    tmp_path: Path,
) -> None:
    band = tmp_path / "event" / "149MHz"
    rr = band / "RR"
    ll = band / "LL"
    rr.mkdir(parents=True)
    ll.mkdir()
    rr_file = rr / "frame.fits"
    ll_file = ll / "frame.fits"
    rr_file.write_bytes(b"rr")
    ll_file.write_bytes(b"ll")
    manifest = pd.DataFrame(
        {
            "path": [str(rr_file), str(ll_file)],
            "inferred_freq_mhz": [149.0, 149.0],
            "inferred_polarization": ["RR", "LL"],
        }
    )

    assert (
        resolve_single_band_frequency_source(
            tmp_path / "event", manifest, 149.0, polarization="RR+LL"
        )
        == band
    )
    assert (
        resolve_single_band_frequency_source(band, manifest, 149.0, polarization="RR")
        == rr
    )
    assert (
        resolve_single_band_frequency_source(rr, manifest, 149.0, polarization="RR")
        == rr
    )


def test_candidate_grouping_common_intersection_and_stride_use_real_times() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        *[
            _candidate(149.0, start + timedelta(seconds=index), index)
            for index in range(5)
        ],
        *[
            _candidate(164.0, start + timedelta(seconds=index), index)
            for index in range(1, 4)
        ],
    ]

    grouped = group_candidates_by_frequency(candidates, [149.0, 164.0])
    assert common_candidate_time_coverage(grouped) == (
        start + timedelta(seconds=1),
        start + timedelta(seconds=3),
    )
    selected = select_sequence_candidates(
        grouped[149.0],
        start + timedelta(seconds=1),
        start + timedelta(seconds=4),
        stride=2,
    )
    assert [item["observation_time"] for item in selected] == [
        (start + timedelta(seconds=1)).isoformat(),
        (start + timedelta(seconds=3)).isoformat(),
    ]


def test_world_roi_intersection_is_independent_of_png_pixels() -> None:
    inside = RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0)
    outside = RadioRoi.from_box(20.0, 20.0, 30.0, 30.0)

    assert roi_intersects_source_map(_map_metadata(), inside) is True
    assert roi_intersects_source_map(_map_metadata(), outside) is False


def test_sequence_source_map_disables_dynamic_tight_cropping(tmp_path: Path) -> None:
    candidate = _candidate(149.0, datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC), 0)

    normal, _ = prepare_single_panel_render(
        {},
        candidate,
        149.0,
        transform="linear",
        output_directory=tmp_path,
    )
    fixed, _ = prepare_single_panel_render(
        {},
        candidate,
        149.0,
        transform="linear",
        output_directory=tmp_path,
        fixed_canvas=True,
    )

    assert "_artifact_bbox_inches" not in normal
    assert fixed["_artifact_bbox_inches"] is None
    assert fixed["_artifact_pad_inches"] == 0.0
    assert fixed["_artifact_fixed_panel_layout"] is True


def test_source_map_rendering_is_serialized_across_threads(
    tmp_path: Path, monkeypatch
) -> None:
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def fake_prepare(config, candidate, frequency, **kwargs):
        return config, candidate

    def fake_run_job(request):
        nonlocal active, maximum_active
        sequence = int(request["sequence"])
        image_path = tmp_path / f"map-{sequence}.png"
        sidecar_path = tmp_path / f"map-{sequence}.json"
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        image_path.write_bytes(_map_png())
        sidecar_path.write_text("{}", encoding="utf-8")
        with counter_lock:
            active -= 1
        return {"image_path": str(image_path), "sidecar_path": str(sidecar_path)}

    monkeypatch.setattr(composite_sequence, "prepare_single_panel_render", fake_prepare)
    monkeypatch.setattr(composite_sequence, "run_job", fake_run_job)
    monkeypatch.setattr(
        composite_sequence,
        "validate_source_map_artifact",
        lambda *_args: _map_metadata(),
    )
    monkeypatch.setattr(
        composite_sequence,
        "_validate_sequence_source_map_pixels",
        lambda *_args: None,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                render_source_map_candidate,
                {},
                {},
                149.0,
                "linear",
                tmp_path,
                sequence,
            )
            for sequence in (1, 2)
        ]
        for future in futures:
            future.result()

    assert maximum_active == 1


def test_sequence_export_writes_one_validated_video_and_png_set_per_frequency(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    frequencies = (149.0, 164.0)
    grouped = {
        frequency: [
            _candidate(frequency, start + timedelta(seconds=index), index)
            for index in range(3)
        ]
        for frequency in frequencies
    }
    rendered: list[tuple[float, int]] = []
    encoded: dict[str, list[tuple[np.ndarray, tuple[int, int]]]] = {}

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        rendered.append((frequency, sequence))
        return (
            _map_png(),
            _map_metadata(),
            {
                "image_path": str(output_dir / f"map-{sequence}.png"),
                "candidate_id": candidate["id"],
            },
        )

    def fake_writer(frame_source, output_path, fps, **kwargs):
        frames = list(frame_source())
        encoded[Path(output_path).name] = frames
        Path(output_path).write_bytes(b"synthetic-mp4")
        return True

    def fake_probe(path, *, expected_size, expected_frame_count):
        return {
            "codec": "h264",
            "width": expected_size[0],
            "height": expected_size[1],
            "frame_count": expected_frame_count,
            "frame_rate": 10.0,
            "duration": expected_frame_count / 10.0,
        }

    bundle = export_composite_sequences(
        tmp_path,
        source_configs={frequency: {} for frequency in frequencies},
        candidates_by_frequency=grouped,
        radio_curves={
            frequency: _radio_curve(start, frequency) for frequency in frequencies
        },
        dart_result=_dart(start),
        roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0, label="Core"),
        reference_frequency_mhz=149.0,
        reference_time=start + timedelta(seconds=1),
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=2),
        request_signature="a" * 64,
        source_context={"radio_directory": "synthetic"},
        options=SequenceExportOptions(fps=10.0, stride=1, dpi=100),
        render_candidate=fake_render,
        media_writer=fake_writer,
        media_probe=fake_probe,
        generated_at=start,
    )

    assert bundle.metadata["schema_version"] == SEQUENCE_SCHEMA_VERSION
    assert set(bundle.videos) == set(frequencies)
    assert len(rendered) == 6
    assert all(path.is_file() for path in bundle.videos.values())
    assert all(path.parent.name.endswith("mhz") for path in bundle.videos.values())
    assert all(
        "radio-composite-sequence" in path.name for path in bundle.videos.values()
    )
    assert all(
        len(str(path.relative_to(bundle.output_directory))) < 100
        for path in bundle.videos.values()
    )
    assert all(
        len(list(frame_dir.glob("*.png"))) == 3
        for frame_dir in bundle.frame_directories.values()
    )
    assert all(len(frames) == 3 for frames in encoded.values())
    for frequency, frame_dir in bundle.frame_directories.items():
        frame_sizes = {Image.open(path).size for path in frame_dir.glob("*.png")}
        assert len(frame_sizes) == 1
        encoded_frames = encoded[bundle.videos[frequency].name]
        for (encoded_rgb, _size), frame_path in zip(
            encoded_frames, sorted(frame_dir.glob("*.png")), strict=True
        ):
            with Image.open(frame_path) as opened:
                assert np.array_equal(encoded_rgb, np.asarray(opened.convert("RGB")))
        manifest = json.loads(
            (frame_dir.parent / "frame-manifest.json").read_text(encoding="utf-8")
        )
        fixed_bounds = manifest["layout"]["panel_bounds_pixels"]
        fixed_source_map_bbox = manifest["layout"]["source_map_panel_bbox_normalized"]
        assert all(
            record["panel_bounds_pixels"] == fixed_bounds
            for record in manifest["frames"]
        )
        assert all(
            record["source_map_panel_bbox_normalized"] == fixed_source_map_bbox
            for record in manifest["frames"]
        )
        assert manifest["curve_plots"]["marker_free"] is True
        assert (
            frame_dir.parent / manifest["curve_plots"]["radio"]["filename"]
        ).is_file()
        assert (
            frame_dir.parent / manifest["curve_plots"]["dart"]["filename"]
        ).is_file()
        assert all(
            record["time_marker_x_pixels"]["radio_curve"]
            == record["time_marker_x_pixels"]["dart_curve"]
            for record in manifest["frames"]
        )
    assert set(bundle.radio_plot_paths) == set(frequencies)
    assert set(bundle.dart_plot_paths) == set(frequencies)
    with zipfile.ZipFile(bundle.zip_path) as archive:
        mp4_records = [
            item for item in archive.infolist() if item.filename.endswith(".mp4")
        ]
        assert len(mp4_records) == 2
        assert all(item.compress_type == zipfile.ZIP_STORED for item in mp4_records)
        metadata = json.loads(archive.read("radio-composite-metadata.json"))
    assert metadata["time_range"]["interpolation"] == "none"
    assert metadata["time_range"]["default_policy"].startswith("intersection")


def test_sequence_uses_matching_dart_curve_and_csv_for_each_frequency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    frequencies = (149.0, 164.0)
    dart_results = {
        149.0: _dart(start, 149.0, bandwidth=2.0, offset=10.0),
        164.0: _dart(start, 164.0, bandwidth=4.0, offset=30.0),
    }
    combined = DartNarrowbandResult(
        time_utc=dart_results[149.0].time_utc,
        curves=tuple(dart_results[value].curves[0] for value in frequencies),
    )
    template_centers: list[float] = []
    original_template = composite_sequence.build_composite_frame_template

    def capture_template(*args, **kwargs):
        template_centers.append(float(args[2].curves[0].center_frequency_mhz))
        return original_template(*args, **kwargs)

    monkeypatch.setattr(
        composite_sequence,
        "build_composite_frame_template",
        capture_template,
    )

    bundle = export_composite_sequences(
        tmp_path,
        source_configs={frequency: {} for frequency in frequencies},
        candidates_by_frequency={
            frequency: [_candidate(frequency, start, 0)] for frequency in frequencies
        },
        radio_curves={
            frequency: _radio_curve(start, frequency) for frequency in frequencies
        },
        dart_result=combined,
        dart_results_by_frequency=dart_results,
        roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
        reference_frequency_mhz=149.0,
        reference_time=start,
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=1),
        request_signature="9" * 64,
        source_context={"dart_bands_by_frequency": {"149": {}, "164": {}}},
        options=SequenceExportOptions(dpi=72, save_video=False, save_frames=True),
        render_candidate=lambda *_args: (
            _map_png(),
            _map_metadata(),
            {"candidate_id": "synthetic"},
        ),
        generated_at=start,
    )

    assert template_centers == [149.0, 164.0]
    assert set(bundle.dart_csv_paths) == set(frequencies)
    assert all(path.is_file() for path in bundle.dart_csv_paths.values())
    assert (
        bundle.metadata["dart_curves_by_frequency"]["149"]["center_frequency_mhz"]
        == 149.0
    )
    assert (
        bundle.metadata["dart_curves_by_frequency"]["164"]["center_frequency_mhz"]
        == 164.0
    )
    assert bundle.metadata["dart_curves_by_frequency"]["164"]["bandwidth_mhz"] == 4.0
    assert len(pd.read_csv(bundle.dart_csv_path).columns) == 3
    for frequency, dart_path in bundle.dart_csv_paths.items():
        manifest = json.loads(
            (dart_path.parent / "frame-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["dart_curve"]["center_frequency_mhz"] == frequency


def test_sequence_rejects_incomplete_per_frequency_dart_mapping(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="Missing sequence input for 164 MHz"):
        export_composite_sequences(
            tmp_path,
            source_configs={149.0: {}, 164.0: {}},
            candidates_by_frequency={
                149.0: [_candidate(149.0, start, 0)],
                164.0: [_candidate(164.0, start, 0)],
            },
            radio_curves={
                149.0: _radio_curve(start, 149.0),
                164.0: _radio_curve(start, 164.0),
            },
            dart_result=_dart(start),
            dart_results_by_frequency={149.0: _dart(start)},
            roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
            reference_frequency_mhz=149.0,
            reference_time=start,
            polarization="RR",
            time_start=start,
            time_end=start + timedelta(seconds=1),
            request_signature="8" * 64,
            source_context={},
            options=SequenceExportOptions(dpi=72, save_video=False, save_frames=True),
            generated_at=start,
        )


@pytest.mark.parametrize(
    ("save_video", "save_frames", "expected_videos", "expected_pngs"),
    [
        (True, True, 1, 2),
        (True, False, 1, 0),
        (False, True, 0, 2),
    ],
)
def test_sequence_output_modes_are_independent_and_compatible(
    tmp_path: Path,
    save_video: bool,
    save_frames: bool,
    expected_videos: int,
    expected_pngs: int,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]
    writer_calls = 0

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        return _map_png(), _map_metadata(), {"candidate_id": candidate["id"]}

    def fake_writer(frame_source, output_path, fps, **kwargs):
        nonlocal writer_calls
        writer_calls += 1
        assert len(list(frame_source())) == 2
        Path(output_path).write_bytes(b"synthetic-mp4")
        return True

    def fake_probe(path, *, expected_size, expected_frame_count):
        return {
            "codec": "h264",
            "width": expected_size[0],
            "height": expected_size[1],
            "frame_count": expected_frame_count,
            "frame_rate": 10.0,
            "duration": expected_frame_count / 10.0,
        }

    bundle = export_composite_sequences(
        tmp_path,
        source_configs={149.0: {}},
        candidates_by_frequency={149.0: candidates},
        radio_curves={149.0: _radio_curve(start, 149.0)},
        dart_result=_dart(start),
        roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
        reference_frequency_mhz=149.0,
        reference_time=start,
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=1),
        request_signature="d" * 64,
        source_context={},
        options=SequenceExportOptions(
            fps=10.0,
            dpi=72,
            save_video=save_video,
            save_frames=save_frames,
        ),
        render_candidate=fake_render,
        media_writer=fake_writer,
        media_probe=fake_probe,
        generated_at=start,
    )

    assert len(bundle.videos) == expected_videos
    assert writer_calls == expected_videos
    frame_dir = bundle.frame_directories[149.0]
    assert frame_dir.is_dir()
    assert len(list(frame_dir.glob("*.png"))) == expected_pngs
    manifest = json.loads(
        (frame_dir.parent / "frame-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["options"]["save_video"] is save_video
    assert manifest["options"]["save_frames"] is save_frames
    assert (manifest["video"] is not None) is save_video
    assert all(
        (record["filename"] is not None) is save_frames for record in manifest["frames"]
    )


def test_sequence_options_require_at_least_one_output_mode() -> None:
    with pytest.raises(ValueError, match="Enable MP4 video, PNG frames, or both"):
        SequenceExportOptions(save_video=False, save_frames=False)


def test_prefetch_is_bounded_ordered_and_joins_its_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[int] = []
    real_queue = composite_sequence.queue.Queue

    class TrackingQueue(real_queue):
        def put(self, item, *args, **kwargs):
            result = super().put(item, *args, **kwargs)
            observed_sizes.append(self.qsize())
            return result

    monkeypatch.setattr(composite_sequence.queue, "Queue", TrackingQueue)
    frames = ((np.full((2, 2, 3), index, dtype=np.uint8), (2, 2)) for index in range(8))

    values = [
        int(frame[0, 0, 0])
        for frame, _size in composite_sequence._prefetch_frames(
            frames,
            maxsize=2,
        )
    ]

    assert values == list(range(8))
    assert observed_sizes and max(observed_sizes) <= 2
    assert not any(
        thread.name == "radio-composite-frame-prefetch" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_prefetch_propagates_producer_failure_without_leaking_thread() -> None:
    def failing_frames():
        yield np.zeros((2, 2, 3), dtype=np.uint8), (2, 2)
        raise ValueError("synthetic frame failure")

    with pytest.raises(ValueError, match="synthetic frame failure"):
        list(composite_sequence._prefetch_frames(failing_frames(), maxsize=2))

    assert not any(
        thread.name == "radio-composite-frame-prefetch" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_sequence_reuses_matching_curve_template_across_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]
    template_calls = 0
    original_template = composite_sequence.build_composite_frame_template

    def count_template(*args, **kwargs):
        nonlocal template_calls
        template_calls += 1
        return original_template(*args, **kwargs)

    monkeypatch.setattr(
        composite_sequence,
        "build_composite_frame_template",
        count_template,
    )
    common = {
        "source_configs": {149.0: {}},
        "candidates_by_frequency": {149.0: candidates},
        "radio_curves": {149.0: _radio_curve(start, 149.0)},
        "dart_result": _dart(start),
        "roi": RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
        "reference_frequency_mhz": 149.0,
        "reference_time": start,
        "polarization": "RR",
        "time_start": start,
        "time_end": start + timedelta(seconds=1),
        "request_signature": "7" * 64,
        "source_context": {},
        "options": SequenceExportOptions(
            dpi=72,
            save_video=False,
            save_frames=True,
        ),
        "render_candidate": lambda *_args: (
            _map_png(),
            _map_metadata(),
            {"candidate_id": "synthetic"},
        ),
        "generated_at": start,
    }

    first = export_composite_sequences(tmp_path, **common)
    second = export_composite_sequences(
        tmp_path,
        **common,
        curve_templates_by_frequency=first.curve_templates,
    )

    assert template_calls == 1
    assert second.curve_templates[149.0] is first.curve_templates[149.0]


def test_sequence_normalizes_a_same_aspect_source_map_resolution_change(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        output = io.BytesIO()
        size = (200, 100) if sequence == 1 else (600, 300)
        Image.new("RGB", size, "white").save(output, format="PNG")
        metadata = _map_metadata()
        metadata["image"]["width"] = size[0]
        metadata["image"]["height"] = size[1]
        return output.getvalue(), metadata, {"candidate_id": candidate["id"]}

    encoded_frames: list[tuple[np.ndarray, tuple[int, int]]] = []

    def fake_writer(frame_source, output_path, fps, **kwargs):
        encoded_frames.extend(frame_source())
        Path(output_path).write_bytes(b"synthetic-mp4")
        return True

    def fake_probe(path, *, expected_size, expected_frame_count):
        return {
            "codec": "h264",
            "width": expected_size[0],
            "height": expected_size[1],
            "frame_count": expected_frame_count,
            "frame_rate": 10.0,
            "duration": expected_frame_count / 10.0,
        }

    bundle = export_composite_sequences(
        tmp_path,
        source_configs={149.0: {}},
        candidates_by_frequency={149.0: candidates},
        radio_curves={149.0: _radio_curve(start, 149.0)},
        dart_result=_dart(start),
        roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
        reference_frequency_mhz=149.0,
        reference_time=start,
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=1),
        request_signature="e" * 64,
        source_context={},
        options=SequenceExportOptions(dpi=72, save_video=True, save_frames=True),
        render_candidate=fake_render,
        media_writer=fake_writer,
        media_probe=fake_probe,
        generated_at=start,
    )

    manifest = json.loads(
        (bundle.output_directory / "149mhz" / "frame-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["frames"][0]["source_map_resolution_normalized"] is False
    assert manifest["frames"][1]["source_map_resolution_normalized"] is True
    frame_sizes = set()
    for path in (bundle.output_directory / "149mhz" / "frames").glob("*.png"):
        with Image.open(path) as opened:
            frame_sizes.add(opened.size)
    assert frame_sizes == {bundle.curve_templates[149.0].layout.canvas_size_pixels}
    assert len(encoded_frames) == 2
    assert {size for _frame, size in encoded_frames} == frame_sizes


def test_sequence_rejects_a_source_map_aspect_ratio_change(tmp_path: Path) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        output = io.BytesIO()
        size = (200, 100) if sequence == 1 else (201, 100)
        Image.new("RGB", size, "white").save(output, format="PNG")
        metadata = _map_metadata()
        metadata["image"]["width"] = size[0]
        metadata["image"]["height"] = size[1]
        return output.getvalue(), metadata, {"candidate_id": candidate["id"]}

    with pytest.raises(ValueError, match="Source Map frame aspect ratio changed"):
        export_composite_sequences(
            tmp_path,
            source_configs={149.0: {}},
            candidates_by_frequency={149.0: candidates},
            radio_curves={149.0: _radio_curve(start, 149.0)},
            dart_result=_dart(start),
            roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
            reference_frequency_mhz=149.0,
            reference_time=start,
            polarization="RR",
            time_start=start,
            time_end=start + timedelta(seconds=1),
            request_signature="e" * 64,
            source_context={},
            options=SequenceExportOptions(dpi=72, save_video=False, save_frames=True),
            render_candidate=fake_render,
            generated_at=start,
        )


def test_sequence_source_map_pixel_validation_rejects_blank_or_wrong_size(
    tmp_path: Path,
) -> None:
    metadata = {"image": {"width": 200, "height": 100}}
    config = {"fig_size": [2, 1], "dpi": 100}
    valid = tmp_path / "valid.png"
    pixels = np.full((100, 200, 3), 255, dtype=np.uint8)
    pixels[20:80, 40:160] = (80, 20, 10)
    Image.fromarray(pixels, mode="RGB").save(valid)
    composite_sequence._validate_sequence_source_map_pixels(valid, metadata, config)

    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 100), "white").save(blank)
    with pytest.raises(RuntimeError, match="blank"):
        composite_sequence._validate_sequence_source_map_pixels(
            blank,
            metadata,
            config,
        )

    with pytest.raises(ValueError, match="canvas size mismatch"):
        composite_sequence._validate_sequence_source_map_pixels(
            valid,
            {"image": {"width": 100, "height": 50}},
            config,
        )


def test_sequence_rejects_source_map_panel_geometry_drift(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        metadata = _map_metadata()
        if sequence == 2:
            metadata["panels"][0]["bbox_normalized"] = [0.2, 0.1, 0.8, 0.9]
        return _map_png(), metadata, {"candidate_id": candidate["id"]}

    with pytest.raises(ValueError, match="Source Map panel geometry changed"):
        export_composite_sequences(
            tmp_path,
            source_configs={149.0: {}},
            candidates_by_frequency={149.0: candidates},
            radio_curves={149.0: _radio_curve(start, 149.0)},
            dart_result=_dart(start),
            roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
            reference_frequency_mhz=149.0,
            reference_time=start,
            polarization="RR",
            time_start=start,
            time_end=start + timedelta(seconds=1),
            request_signature="f" * 64,
            source_context={},
            options=SequenceExportOptions(dpi=72, save_video=False, save_frames=True),
            render_candidate=fake_render,
            generated_at=start,
        )


def test_canceled_export_does_not_publish_partial_package(tmp_path: Path) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidate = _candidate(149.0, start, 0)

    with pytest.raises(CompositeSequenceCancelled):
        export_composite_sequences(
            tmp_path,
            source_configs={149.0: {}},
            candidates_by_frequency={149.0: [candidate]},
            radio_curves={149.0: _radio_curve(start, 149.0)},
            dart_result=_dart(start),
            roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
            reference_frequency_mhz=149.0,
            reference_time=start,
            polarization="RR",
            time_start=start,
            time_end=start + timedelta(seconds=2),
            request_signature="b" * 64,
            source_context={},
            options=SequenceExportOptions(),
            cancel_check=lambda: True,
            generated_at=start,
        )

    assert not [path for path in tmp_path.iterdir() if not path.name.startswith(".")]


def test_sequence_export_produces_ffprobe_valid_mp4(tmp_path: Path) -> None:
    if media.resolve_ffmpeg() is None or media.resolve_ffprobe() is None:
        pytest.skip("FFmpeg and FFprobe are required for the real media smoke")
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    candidates = [
        _candidate(149.0, start + timedelta(seconds=index), index) for index in range(2)
    ]

    def fake_render(config, candidate, frequency, transform, output_dir, sequence):
        return _map_png(), _map_metadata(), {"candidate_id": candidate["id"]}

    bundle = export_composite_sequences(
        tmp_path,
        source_configs={149.0: {}},
        candidates_by_frequency={149.0: candidates},
        radio_curves={149.0: _radio_curve(start, 149.0)},
        dart_result=_dart(start),
        roi=RadioRoi.from_box(-2.0, -2.0, 2.0, 2.0),
        reference_frequency_mhz=149.0,
        reference_time=start,
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=1),
        request_signature="c" * 64,
        source_context={},
        options=SequenceExportOptions(fps=2.0, dpi=72),
        render_candidate=fake_render,
        generated_at=start,
    )

    probe = media.probe_video(bundle.videos[149.0], expected_frame_count=2)
    assert probe["frame_count"] == 2
    assert probe["width"] % 2 == 0
    assert probe["height"] % 2 == 0


def test_sequence_and_source_map_worker_share_the_matplotlib_lock() -> None:
    assert (
        composite_sequence._FIGURE_RENDER_LOCK is source_map_worker.FIGURE_RENDER_LOCK
    )
