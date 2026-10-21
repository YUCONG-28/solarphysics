import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_apps.workflows.radio import muser_raw_video as video
from solar_apps.workflows.radio.muser_raw_preview import draw_original
from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    normalize_time,
    build_sequence_match_plan,
)


@pytest.mark.parametrize("dual", [False, True])
def test_encode_real_movie_keeps_spectrum_and_scales_and_records_reuse(
    tmp_path, monkeypatch, dual
):
    av = pytest.importorskip("av")
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 200.0,
            "CDELT2": 200.0,
            "BUNIT": "SFU",
        }
    )
    prepared = SimpleNamespace(
        data=np.arange(16).reshape(4, 4) + 1.0,
        header=header,
        metadata={"utc": "2020-01-02T04:14:38.400", "frequency_mhz": 238.71},
    )
    fig = plt.figure(figsize=(8, 10), layout="constrained")
    columns = 3 if dual else 2
    grid = fig.add_gridspec(4, 2 * columns, width_ratios=[1, 0.05] * columns)
    panels = []
    for i in range(2 * columns):
        row, col = divmod(i, columns)
        panels.append(
            draw_original(
                fig.add_subplot(grid[row, 2 * col]),
                fig.add_subplot(grid[row, 2 * col + 1]),
                prepared,
                "test",
                [-2000, 2000, -2000, 2000],
            )
        )
    spec = fig.add_subplot(grid[2, :-1])
    fig.add_subplot(grid[2, -1])
    flux = fig.add_subplot(grid[3, :-1])
    fig.add_subplot(grid[3, -1])
    spec.imshow(np.arange(16).reshape(4, 4))
    spec.axvline(0)
    flux.plot([1, 2, 3])
    flux.axvline(0)
    fig.suptitle("test\nfixed scales\nno interpolation")
    start = Time("2020-01-02T04:14:38").unix
    sources = []
    keys = (
        ["AIA171", "MUSER0", "DART0", "AIA304", "MUSER1", "DART1"]
        if dual
        else ["AIA171", "AIA304", "MUSER", "DART"]
    )
    for key in keys:
        times = [start] if key.startswith("AIA") else [start, start + 0.1, start + 0.2]
        sources.append(
            TimeStream(
                key,
                [
                    TimeFrame(f"{key}-{i}", normalize_time(t, format="unix"))
                    for i, t in enumerate(times)
                ],
                tolerance_seconds=0.05,
            )
        )
    plan = build_sequence_match_plan(sources)
    calls = []

    def pixels(frame):
        calls.append(frame.frame_id)
        xx, yy = np.meshgrid(np.linspace(-300, 300, 4), np.linspace(-300, 300, 4))
        return xx, yy, prepared.data, header

    monkeypatch.setattr(video, "frame_pixels", pixels)
    spec_before = spec.images[0].get_array().copy()
    config = {"event_id": "test", "dart_mode": "I", "video_fps": 10, "video_dpi": 60}
    if dual:
        config["radio_bands"] = [{}, {}]
    video.encode_figure(fig, config, tmp_path, panels, [], plan=plan)
    metadata = json.loads((tmp_path / "video.json").read_text())
    assert metadata["panel_source_ids"] == keys
    assert metadata["frames"] == metadata["decoded_frames"] == 3
    assert metadata["duration_seconds"] == 0.3
    assert len(list((tmp_path / "frames").glob("*.png"))) == 3
    assert calls.count("AIA171-0") == 1
    assert metadata["records"][-1]["selections"]["AIA171"]["within_tolerance"] is False
    assert not (tmp_path / "comparison.partial.mp4").exists()
    np.testing.assert_array_equal(spec.images[0].get_array(), spec_before)
    with av.open(str(tmp_path / "comparison.mp4")) as movie:
        assert len(list(movie.decode(video=0))) == 3
    plt.close(fig)


def test_batch_plan_filters_before_pairing_and_preserves_all_routes(monkeypatch):
    import solar_toolkit.radio.time_matching as matching

    start = Time("2020-01-02T04:14:00").unix

    def indexed(paths, **kwargs):
        label = str(paths[0])
        wave = 171 if label.endswith("171") else 304
        return [
            TimeFrame(
                f"{label}-{i}",
                normalize_time(start + t, format="unix"),
                {"header": {"WAVELNTH": wave}},
            )
            for i, t in enumerate([-1, 0.1, 0.5, 1.1])
        ]

    monkeypatch.setattr(video, "image_paths", lambda path: [str(path)])
    monkeypatch.setattr(matching, "index_radio_frames", indexed)
    config = {
        "display_window_utc": ["2020-01-02T04:14:00", "2020-01-02T04:14:01"],
        "muser_directory": "muser",
        "frequency_mhz": 238.0,
        "dart_directory": "dart",
        "dart_mode": "I",
        "aia_directory": "aia",
    }
    plan = video.build_plan(config)
    assert len(plan.rows) == 2
    for summary in plan.summary["sources"].values():
        assert summary["used_frame_count"] == summary["frame_count"] == 2
        assert summary["coverage_fraction"] == 1
    for row in plan.rows:
        assert start <= Time(row.time.utc).unix <= start + 1


def test_overlapping_colorbar_labels_are_repaired_without_changing_scale():
    from matplotlib.colors import AsinhNorm

    fig, ax = plt.subplots(figsize=(2, 2))
    norm = AsinhNorm(linear_width=1, vmin=-2, vmax=100)
    ax.set_yticks([0, 0.001, 0.002, 0.003])
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    before = (norm.vmin, norm.vmax, norm.linear_width)
    assert video.readable_colorbar_ticks(ax, norm, fig.canvas.get_renderer())
    np.testing.assert_allclose(norm(ax.get_yticks()), np.linspace(0, 1, 5))
    assert (norm.vmin, norm.vmax, norm.linear_width) == before
    plt.close(fig)


def test_two_frequency_plan_indexes_each_requested_plane(monkeypatch):
    import solar_toolkit.radio.time_matching as matching

    start = Time("2020-01-02T04:14:00").unix
    calls = []

    def indexed(paths, **kwargs):
        path = str(paths[0])
        freq = kwargs.get("frequency_mhz")
        calls.append((path, freq))
        wave = 171 if path.endswith("171") else 304
        return [
            TimeFrame(
                f"{path}-{freq}-{i}",
                normalize_time(start + t, format="unix"),
                {"header": {"WAVELNTH": wave}, "frequency_mhz": freq},
            )
            for i, t in enumerate([-0.1, 0.1, 0.5, 1.1])
        ]

    monkeypatch.setattr(video, "image_paths", lambda path: [str(path)])
    monkeypatch.setattr(matching, "index_radio_frames", indexed)
    config = {
        "display_window_utc": ["2020-01-02T04:14:00", "2020-01-02T04:14:01"],
        "muser_directory": "muser",
        "dart_directory": "unused",
        "dart_mode": "I",
        "aia_directory": "aia",
        "radio_bands": [
            {"dart_directory": "dart190", "dart_mhz": 190.0, "muser_mhz": 168.0078125},
            {"dart_directory": "dart285", "dart_mhz": 285.0, "muser_mhz": 277.7734375},
        ],
    }
    plan = video.build_plan(config)
    assert set(plan.rows[0].selections) == {
        "AIA171",
        "AIA304",
        "MUSER0",
        "MUSER1",
        "DART0",
        "DART1",
    }
    for key, freq in [
        ("MUSER0", 168.0078125),
        ("MUSER1", 277.7734375),
        ("DART0", 190.0),
        ("DART1", 285.0),
    ]:
        assert plan.rows[0].selections[key].frame.payload["frequency_mhz"] == freq
    assert ("muser", 168.0078125) in calls and ("muser", 277.7734375) in calls
    assert all(s["coverage_fraction"] == 1 for s in plan.summary["sources"].values())


@pytest.mark.parametrize("source_present", [True, False])
def test_missing_reference_scale_uses_planned_fits_or_keeps_no_data(
    tmp_path, source_present
):
    av = pytest.importorskip("av")
    from solar_toolkit.radio.raw_comparison import prepare_aia
    from solar_toolkit.radio.time_matching import index_radio_frames

    start = Time("2020-01-02T04:14:00").unix
    h = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 200.0,
            "CDELT2": 200.0,
            "BUNIT": "source units",
            "FREQ": 238.0,
        }
    )
    pixels = np.arange(16).reshape(4, 4) + 1.0
    keys = ["AIA171", "AIA304", "MUSER", "DART"]
    streams = []
    for key in keys:
        folder = tmp_path / key
        folder.mkdir()
        offsets = (
            ([20.0] if source_present else []) if key == "AIA304" else [0.0, 0.1, 0.2]
        )
        paths = []
        for i, offset in enumerate(offsets):
            header = h.copy()
            header["DATE-OBS"] = Time(start + offset, format="unix", precision=6).isot
            if key.startswith("AIA"):
                header["WAVELNTH"] = int(key[3:])
            path = folder / f"frame-{i}.fits"
            fits.writeto(path, pixels + i, header)
            paths.append(path)
        streams.append(
            TimeStream(key, index_radio_frames(paths), tolerance_seconds=0.05)
        )
    # Reference matching really misses; the valid later FITS is inside the plan.
    missing = prepare_aia(tmp_path / "AIA304", start, wavelength=304, tolerance=12.0)
    assert missing.data is None
    plan = build_sequence_match_plan(streams)
    prepared = SimpleNamespace(
        data=pixels,
        header=h,
        metadata={"utc": "2020-01-02T04:14:00", "frequency_mhz": 238.0},
    )
    fig = plt.figure(figsize=(8, 10), layout="constrained")
    grid = fig.add_gridspec(4, 4, width_ratios=[1, 0.05, 1, 0.05])
    panels = []
    for i, key in enumerate(keys):
        row, col = divmod(i, 2)
        panels.append(
            draw_original(
                fig.add_subplot(grid[row, 2 * col]),
                fig.add_subplot(grid[row, 2 * col + 1]),
                missing if key == "AIA304" else prepared,
                key,
                [-2000, 2000, -2000, 2000],
            )
        )
    spec = fig.add_subplot(grid[2, :-1])
    fig.add_subplot(grid[2, -1])
    flux = fig.add_subplot(grid[3, :-1])
    fig.add_subplot(grid[3, -1])
    spec.imshow(np.arange(16).reshape(4, 4))
    spec.axvline(0)
    flux.plot([1, 2, 3])
    flux.axvline(0)
    fig.suptitle("test\nfixed spectrum\nno interpolation")
    output = tmp_path / "movie"
    output.mkdir()
    video.encode_figure(
        fig,
        {"event_id": "test", "dart_mode": "I", "frequency_mhz": 238.0, "video_dpi": 60},
        output,
        panels,
        [],
        plan=plan,
    )
    metadata = json.loads((output / "video.json").read_text())
    scale = metadata["image_scales"][1]
    assert panels[1]["status"] == "no_data"  # Preview provenance stays unchanged.
    if source_present:
        assert scale["scale_source"] == "first_valid_planned_frame"
        assert scale["status"] == "ok"
        assert scale["scale_frame_id"] == streams[1].frames[0].frame_id
        assert scale["scale_reference_utc"].startswith("2020-01-02T04:14:20")
        assert fig.axes[2].collections
        assert fig.axes[3].get_visible()
    else:
        assert scale["scale_source"] == "no_available_planned_frame"
        assert scale["display"] is None
        assert not fig.axes[2].collections
        assert any(text.get_text() == "NO DATA" for text in fig.axes[2].texts)
        assert not fig.axes[3].get_visible()
    with av.open(str(output / "comparison.mp4")) as movie:
        assert len(list(movie.decode(video=0))) == len(plan.rows)
    plt.close(fig)
