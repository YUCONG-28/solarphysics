"""Tests for the standalone radio bad-frame review application."""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from matplotlib.colors import to_hex
from PIL import Image

from solar_apps.frontends.radio_bad_frame_review import review as review_module
from solar_apps.frontends.radio_bad_frame_review import (
    BadFrameReviewStore,
    StaleReviewError,
    extract_training_examples,
    final_bad_frame_paths,
    load_bad_frame_review,
)
from solar_apps.frontends.radio_bad_frame_review.server import create_app
from solar_apps.frontends.radio_bad_frame_review import (
    application as review_application,
)
from solar_apps.frontends.radio_bad_frame_review.review import (
    PreviewDisplaySettings,
    _preview_geometry,
    _preview_percentile_limits,
)

APPS_ROOT = Path(__file__).resolve().parents[3]


def _compact_source(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    y, x = np.indices(shape)
    source = 1.0e6 * np.exp(-(((x - 32) ** 2) + ((y - 32) ** 2)) / (2 * 2.4**2))
    return 1.0e3 + 10.0 * x + 5.0 * y + source


def _striped_bad_source(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    data = _compact_source(shape)
    data[:, 6::12] += 1.5e7
    data[5::13, :] += 1.1e7
    return data


def _write_fits(
    path: Path, data: np.ndarray, *, second: int, with_wcs: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = fits.ImageHDU(data=np.asarray(data, dtype=np.float32))
    image.header["DATE-OBS"] = f"2025-05-03T07:20:{second:02d}.000Z"
    if with_wcs:
        image.header.update(
            {
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CRPIX1": 33.0,
                "CRPIX2": 33.0,
                "CRVAL1": 0.0,
                "CRVAL2": 0.0,
                "CDELT1": 156.573805638268,
                "CDELT2": 156.573805638268,
                "BUNIT": "K",
            }
        )
    fits.HDUList([fits.PrimaryHDU(), image]).writeto(path)


def _radio_dataset(
    root: Path, *, include_bad: bool = True, with_wcs: bool = False
) -> tuple[Path, Path | None]:
    folder = root / "149MHz" / "RR"
    bad_path = None
    for index in range(5):
        path = folder / f"149MHz_20250503_0720{index:02d}_000.fits"
        is_bad = include_bad and index == 2
        _write_fits(
            path,
            _striped_bad_source() if is_bad else _compact_source(),
            second=index,
            with_wcs=with_wcs,
        )
        if is_bad:
            bad_path = path
    return root, bad_path


def _create_review(store: BadFrameReviewStore, root: Path) -> dict:
    return store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "start_index": 0,
            "end_index": None,
        }
    )


def test_discovery_scan_preview_and_completed_manifest(tmp_path: Path) -> None:
    root, bad_path = _radio_dataset(tmp_path / "radio")
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])

    discovery = store.discover(root)
    assert discovery["bands"] == [
        {
            "frequency_mhz": 149.0,
            "label": "149 MHz",
            "polarizations": [{"name": "RR", "file_count": 5}],
        }
    ]

    review = _create_review(store, root)
    assert review["status"] == "draft"
    assert review["summary"]["scanned_file_count"] == 5
    assert len(review["candidates"]) == 1
    candidate = review["candidates"][0]
    assert Path(candidate["source_file"]) == bad_path.resolve()
    assert candidate["context_file_ids"][0] is not None
    assert candidate["context_file_ids"][2] is not None

    preview = store.render_candidate_preview(
        review["review_id"], candidate["candidate_id"]
    )
    image = Image.open(io.BytesIO(preview)).convert("RGB")
    assert image.width > 1000
    assert image.height > 400
    assert float(np.asarray(image).std()) > 5.0

    updated = store.update_decisions(
        review["review_id"], {candidate["candidate_id"]: "good"}
    )
    assert updated["summary"]["pending_count"] == 0
    assert updated["summary"]["kept_count"] == 1
    completed = store.finalize(review["review_id"], "completed")
    assert completed["status"] == "completed"
    assert completed["final_bad_files"] == []

    manifest_path = tmp_path / "reviews" / review["review_id"] / "review.json"
    persisted = load_bad_frame_review(manifest_path)
    assert persisted["input_fingerprint"].startswith("sha256:")
    assert final_bad_frame_paths(persisted) == ()
    csv_text = (manifest_path.parent / "candidates.csv").read_text(encoding="utf-8")
    assert "human_decision" in csv_text
    assert ",good,human,ok," in csv_text


def test_preview_wcs_geometry_converts_to_arcsec_and_rejects_rotation() -> None:
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 129.0,
            "CRPIX2": 129.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 156.573805638268,
            "CDELT2": 156.573805638268,
        }
    )

    geometry = _preview_geometry(header, (256, 256))

    assert geometry.extent_arcsec == pytest.approx(
        (-20119.73402451744, 19963.16021887917) * 2
    )
    assert geometry.origin == "lower"

    degree_header = header.copy()
    degree_header["CUNIT1"] = "deg"
    degree_header["CUNIT2"] = "arcmin"
    degree_header["CDELT1"] = header["CDELT1"] / 3600.0
    degree_header["CDELT2"] = header["CDELT2"] / 60.0
    converted = _preview_geometry(degree_header, (256, 256))
    assert converted.extent_arcsec == pytest.approx(geometry.extent_arcsec)

    inverted = header.copy()
    inverted["CDELT2"] = -abs(float(inverted["CDELT2"]))
    inverted_geometry = _preview_geometry(inverted, (256, 256))
    assert inverted_geometry.extent_arcsec[2] > inverted_geometry.extent_arcsec[3]

    rotated = header.copy()
    rotated["PC1_2"] = 0.1
    with pytest.raises(ValueError, match="rotated PC matrix"):
        _preview_geometry(rotated, (256, 256))

    missing_unit = header.copy()
    del missing_unit["CUNIT1"]
    with pytest.raises(ValueError, match="supported angular unit"):
        _preview_geometry(missing_unit, (256, 256))


def test_preview_display_settings_validate_fixed_ranges_and_colormaps() -> None:
    assert PreviewDisplaySettings().to_dict() == {
        "cmap": "coolwarm",
        "transform": "robust_asinh",
        "range_mode": "auto",
        "pmin": 50.0,
        "pmax": 99.7,
        "vmin": None,
        "vmax": None,
    }
    for cmap in (
        "coolwarm",
        "hot",
        "inferno",
        "magma",
        "viridis",
        "plasma",
        "jet",
        "cividis",
    ):
        assert PreviewDisplaySettings(cmap=cmap).cmap == cmap
    assert PreviewDisplaySettings.from_mapping(
        {
            "cmap": "viridis",
            "transform": "linear",
            "range_mode": "fixed",
            "vmin": "10",
            "vmax": "20",
        }
    ).to_dict() == {
        "cmap": "viridis",
        "transform": "linear",
        "range_mode": "fixed",
        "pmin": 50.0,
        "pmax": 99.7,
        "vmin": 10.0,
        "vmax": 20.0,
    }
    with pytest.raises(ValueError, match="cmap must be one of"):
        PreviewDisplaySettings(cmap="not-a-map")
    assert PreviewDisplaySettings(range_mode="fixed").pmin == 50.0
    with pytest.raises(ValueError, match="percentiles must satisfy"):
        PreviewDisplaySettings(pmin=99.7, pmax=50.0)
    with pytest.raises(ValueError, match="legacy absolute range requires both"):
        PreviewDisplaySettings(vmin=10.0)
    with pytest.raises(ValueError, match="less than"):
        PreviewDisplaySettings(range_mode="fixed", vmin=20.0, vmax=10.0)
    values = np.arange(101, dtype=float)
    assert _preview_percentile_limits(values, 50.0, 99.7) == pytest.approx(
        tuple(np.percentile(values, [50.0, 99.7]))
    )


def test_preview_renderer_uses_shared_arcsec_range_and_high_contrast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = _create_review(store, root)
    candidate = review["candidates"][0]
    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        result = original_savefig(figure, *args, **kwargs)
        axes = figure.axes[:3]
        captured["figure_face"] = to_hex(figure.get_facecolor())
        captured["suptitle_color"] = figure._suptitle.get_color()
        captured["suptitle_size"] = figure._suptitle.get_fontsize()
        captured["title_colors"] = [axis.title.get_color() for axis in axes]
        captured["title_sizes"] = [axis.title.get_fontsize() for axis in axes]
        captured["xlabels"] = [axis.get_xlabel() for axis in axes]
        captured["ylabels"] = [axis.get_ylabel() for axis in axes]
        captured["ranges"] = [
            (axis.images[0].norm.vmin, axis.images[0].norm.vmax) for axis in axes
        ]
        captured["extents"] = [tuple(axis.images[0].get_extent()) for axis in axes]
        captured["xlimits"] = [tuple(axis.get_xlim()) for axis in axes]
        captured["ylimits"] = [tuple(axis.get_ylim()) for axis in axes]
        captured["cmaps"] = [axis.images[0].get_cmap().name for axis in axes]
        captured["candidate_border"] = to_hex(axes[1].spines["left"].get_edgecolor())
        captured["colorbar_label"] = figure.axes[3].get_ylabel()
        captured["minimum_tick_size"] = min(
            label.get_fontsize()
            for axis in axes
            for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
        )
        return result

    monkeypatch.setattr(Figure, "savefig", capture)
    preview = store.render_candidate_preview(
        review["review_id"],
        candidate["candidate_id"],
        display=PreviewDisplaySettings(
            cmap="viridis",
            transform="linear",
            range_mode="fixed",
            vmin=0.0,
            vmax=2.0e7,
        ),
    )

    assert Image.open(io.BytesIO(preview)).size[0] > 1000
    assert captured["figure_face"] == "#f4f6f8"
    assert captured["suptitle_color"] == "#13202b"
    assert captured["suptitle_size"] == pytest.approx(12.0)
    assert captured["title_colors"] == ["#13202b"] * 3
    assert captured["title_sizes"] == pytest.approx([10.0] * 3)
    assert captured["xlabels"] == ["HPLN / arcsec"] * 3
    assert captured["ylabels"] == ["HPLT / arcsec"] * 3
    assert captured["ranges"] == [(0.0, 2.0e7)] * 3
    assert len(set(captured["extents"])) == 1
    assert captured["xlimits"] == [(-3000.0, 3000.0)] * 3
    assert captured["ylimits"] == [(-3000.0, 3000.0)] * 3
    assert captured["cmaps"] == ["viridis"] * 3
    assert captured["candidate_border"] == "#007f73"
    assert captured["colorbar_label"] == "Intensity [K]"
    assert captured["minimum_tick_size"] >= 8.5


def test_all_preview_colormaps_render_with_explicit_pixel_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    root, _bad_path = _radio_dataset(tmp_path / "radio")
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = _create_review(store, root)
    candidate = review["candidates"][0]
    observed: dict[str, object] = {}
    original_savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        result = original_savefig(figure, *args, **kwargs)
        axes = figure.axes[:3]
        observed.setdefault("xlabels", [axis.get_xlabel() for axis in axes])
        observed.setdefault("ylabels", [axis.get_ylabel() for axis in axes])
        observed.setdefault(
            "ranges",
            [(axis.images[0].norm.vmin, axis.images[0].norm.vmax) for axis in axes],
        )
        observed.setdefault("colorbar_count", len(figure.axes) - len(axes))
        return result

    monkeypatch.setattr(Figure, "savefig", capture)
    for cmap in (
        "coolwarm",
        "hot",
        "inferno",
        "magma",
        "viridis",
        "plasma",
        "jet",
        "cividis",
    ):
        preview = store.render_candidate_preview(
            review["review_id"],
            candidate["candidate_id"],
            display=PreviewDisplaySettings(cmap=cmap),
        )
        assert Image.open(io.BytesIO(preview)).format == "PNG"

    assert observed["xlabels"] == ["Pixel X — WCS unavailable"] * 3
    assert observed["ylabels"] == ["Pixel Y — WCS unavailable"] * 3
    assert len(set(observed["ranges"])) > 1
    assert all(vmin < vmax for vmin, vmax in observed["ranges"])
    assert observed["ranges"][0][0] == pytest.approx(0.0, abs=1e-12)
    assert observed["colorbar_count"] == 3


def test_fixed_preview_uses_cached_frequency_sample_across_polarizations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    ll_paths = []
    for index in range(5):
        path = root / "149MHz" / "LL" / f"149MHz_20250503_0720{index:02d}_000.fits"
        _write_fits(
            path,
            _compact_source() * (2.0 + index),
            second=index,
            with_wcs=True,
        )
        ll_paths.append(path)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR", "LL"],
            "start_index": 0,
            "end_index": None,
        }
    )
    candidate = review["candidates"][0]
    original_reader = review_module.read_radio_fits_image
    reads: list[Path] = []

    def counted_reader(path):
        resolved = Path(path)
        reads.append(resolved)
        if resolved == ll_paths[-1]:
            raise OSError("synthetic unreadable frame")
        return original_reader(path)

    captured: list[dict[str, object]] = []
    original_savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        result = original_savefig(figure, *args, **kwargs)
        axes = figure.axes[:3]
        captured.append(
            {
                "ranges": [
                    (axis.images[0].norm.vmin, axis.images[0].norm.vmax)
                    for axis in axes
                ],
                "colorbar_count": len(figure.axes) - len(axes),
            }
        )
        return result

    monkeypatch.setattr(review_module, "read_radio_fits_image", counted_reader)
    monkeypatch.setattr(Figure, "savefig", capture)
    first = store.render_candidate_preview(
        review["review_id"],
        candidate["candidate_id"],
        display=PreviewDisplaySettings(
            transform="linear", range_mode="fixed", pmin=50.0, pmax=90.0
        ),
    )
    reads_after_first = len(reads)
    second = store.render_candidate_preview(
        review["review_id"],
        candidate["candidate_id"],
        display=PreviewDisplaySettings(
            transform="linear", range_mode="fixed", pmin=60.0, pmax=99.0
        ),
    )

    assert Image.open(io.BytesIO(first)).format == "PNG"
    assert Image.open(io.BytesIO(second)).format == "PNG"
    assert reads_after_first > 3
    assert len(reads) - reads_after_first == 3
    assert all(len(set(item["ranges"])) == 1 for item in captured)
    assert [item["colorbar_count"] for item in captured] == [1, 1]
    cache_files = list(
        (tmp_path / "reviews" / review["review_id"] / "preview_range_cache").glob(
            "*.npz"
        )
    )
    assert len(cache_files) == 1

    ll_paths[0].touch()
    with pytest.raises(StaleReviewError):
        store.render_candidate_preview(
            review["review_id"],
            candidate["candidate_id"],
            display=PreviewDisplaySettings(
                transform="linear", range_mode="fixed", pmin=50.0, pmax=99.7
            ),
        )


def test_frequency_sample_is_deterministic_bounded_and_frequency_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    for index in range(5):
        _write_fits(
            root / "164MHz" / "RR" / f"164MHz_20250503_0720{index:02d}_000.fits",
            _compact_source() * 20.0,
            second=index,
            with_wcs=True,
        )
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149, 164],
            "polarizations": ["RR"],
            "start_index": 0,
            "end_index": None,
        }
    )
    monkeypatch.setattr(review_module, "PREVIEW_GLOBAL_SAMPLE_LIMIT", 20)
    monkeypatch.setattr(review_module, "PREVIEW_GLOBAL_FILE_SAMPLE_LIMIT", 6)

    first, _unit = store._frequency_preview_sample(
        review, frequency_mhz=149.0, transform="linear"
    )
    cache_dir = tmp_path / "reviews" / review["review_id"] / "preview_range_cache"
    next(cache_dir.glob("149MHz-linear.npz")).unlink()
    repeated, _unit = store._frequency_preview_sample(
        review, frequency_mhz=149.0, transform="linear"
    )
    other, _unit = store._frequency_preview_sample(
        review, frequency_mhz=164.0, transform="linear"
    )

    assert first.size <= 20
    assert np.array_equal(first, repeated)
    assert np.percentile(other, 50) > 10 * np.percentile(first, 50)


def test_fixed_frequency_sample_lock_avoids_duplicate_concurrent_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "start_index": 0,
            "end_index": None,
        }
    )
    original_reader = review_module.read_radio_fits_image
    reads: list[Path] = []

    def counted_reader(path):
        reads.append(Path(path))
        return original_reader(path)

    monkeypatch.setattr(review_module, "read_radio_fits_image", counted_reader)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                store._frequency_preview_sample,
                review,
                frequency_mhz=149.0,
                transform="linear",
            )
            for _index in range(2)
        ]
        results = [future.result()[0] for future in futures]

    frequency_files = [
        item for item in review["files"] if item["frequency_mhz"] == 149.0
    ]
    assert len(reads) == len(frequency_files)
    assert np.array_equal(results[0], results[1])


def test_fixed_frequency_sample_rejects_all_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "start_index": 0,
            "end_index": None,
        }
    )

    def unreadable(_path):
        raise OSError("synthetic unreadable FITS")

    monkeypatch.setattr(review_module, "read_radio_fits_image", unreadable)
    with pytest.raises(ValueError, match="No finite intensity values"):
        store._frequency_preview_sample(review, frequency_mhz=149.0, transform="linear")


def test_fixed_frequency_range_rejects_conflicting_bunit(tmp_path: Path) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio", with_wcs=True)
    for index in range(5):
        path = root / "149MHz" / "LL" / f"149MHz_20250503_0720{index:02d}_000.fits"
        _write_fits(path, _compact_source(), second=index, with_wcs=True)
        with fits.open(path, mode="update") as hdus:
            hdus[1].header["BUNIT"] = "Jy"
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR", "LL"],
            "start_index": 0,
            "end_index": None,
        }
    )

    with pytest.raises(ValueError, match="requires one BUNIT"):
        store.render_candidate_preview(
            review["review_id"],
            review["candidates"][0]["candidate_id"],
            display=PreviewDisplaySettings(
                transform="linear", range_mode="fixed", pmin=50.0, pmax=99.7
            ),
        )


def test_review_requires_all_decisions_and_skip_keeps_automatic_bad(
    tmp_path: Path,
) -> None:
    root, bad_path = _radio_dataset(tmp_path / "radio")
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = _create_review(store, root)

    with pytest.raises(ValueError, match="Every candidate"):
        store.finalize(review["review_id"], "completed")

    skipped = store.finalize(review["review_id"], "skipped")
    assert skipped["status"] == "skipped"
    assert skipped["summary"]["pending_count"] == 0
    assert skipped["summary"]["confirmed_bad_count"] == 0
    assert skipped["summary"]["final_bad_count"] == 1
    assert final_bad_frame_paths(skipped) == (bad_path.resolve(),)
    assert skipped["candidates"][0]["decision_source"] == "automatic_on_skip"
    with pytest.raises(ValueError, match="read-only"):
        store.update_decisions(
            review["review_id"], {review["candidates"][0]["candidate_id"]: "good"}
        )


def test_no_candidates_auto_completes_and_changed_inputs_are_stale(
    tmp_path: Path,
) -> None:
    clean_root, _ = _radio_dataset(tmp_path / "clean", include_bad=False)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    clean_review = _create_review(store, clean_root)
    assert clean_review["status"] == "completed"
    assert clean_review["summary"]["candidate_count"] == 0

    bad_root, bad_path = _radio_dataset(tmp_path / "bad")
    bad_review = _create_review(store, bad_root)
    assert bad_path is not None
    bad_path.write_bytes(bad_path.read_bytes() + b"changed")
    candidate_id = bad_review["candidates"][0]["candidate_id"]
    with pytest.raises(StaleReviewError, match="scan again"):
        store.update_decisions(bad_review["review_id"], {candidate_id: "bad"})


def test_all_frame_review_persists_coverage_and_promotes_explicit_labels(
    tmp_path: Path,
) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio")
    output_root = tmp_path / "reviews"
    store = BadFrameReviewStore(output_root, [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "candidate_strategy": "rules",
            "review_scope": "all_scanned",
        }
    )
    review_id = review["review_id"]
    assert review["status"] == "draft"
    assert review["input"]["review_scope"] == "all_scanned"
    assert store.public_payload(review)["summary"]["remaining_frame_count"] == 5

    page = store.list_frames(review_id, offset=0, limit=2)
    assert page["total"] == 5
    assert [item["ordinal"] for item in page["frames"]] == [1, 2]
    first_frame = page["frames"][0]
    assert first_frame["viewed"] is False
    preview = store.render_frame_preview(review_id, first_frame["file_id"])
    assert Image.open(io.BytesIO(preview)).width > 1000
    assert store.public_payload(review)["summary"]["viewed_frame_count"] == 0

    automatic_candidate = review["candidates"][0]
    store.update_decisions(review_id, {automatic_candidate["candidate_id"]: "good"})
    with pytest.raises(ValueError, match="must be viewed"):
        store.finalize(review_id, "completed")

    all_frames = store.list_frames(review_id, offset=0, limit=100)["frames"]
    clean_frame = next(frame for frame in all_frames if frame["candidate_id"] is None)
    labelled = store.update_frame_label(
        review_id,
        clean_frame["file_id"],
        {
            "quality_label": "bad",
            "event_tags": [],
            "artifact_tags": ["stripe"],
        },
    )
    promoted = next(
        item
        for item in labelled["candidates"]
        if item["file_id"] == clean_frame["file_id"]
    )
    assert promoted["selection_source"] == "manual_full_review"

    for frame in all_frames:
        store.mark_frame_viewed(review_id, frame["file_id"])
    resumed = BadFrameReviewStore(output_root, [tmp_path])
    resumed_page = resumed.list_frames(review_id, offset=0, limit=100)
    assert all(frame["viewed"] for frame in resumed_page["frames"])
    assert resumed_page["first_unviewed_index"] is None

    completed = resumed.finalize(review_id, "completed")
    assert completed["audit"]["coverage_fingerprint"].startswith("sha256:")
    assert final_bad_frame_paths(completed) == (
        Path(promoted["source_file"]).resolve(),
    )
    examples = extract_training_examples(completed)
    assert len(examples) == 2
    assert {item["quality_label"] for item in examples} == {"good", "bad"}
    audit_csv = output_root / review_id / "viewed_frames.csv"
    assert len(audit_csv.read_text(encoding="utf-8").splitlines()) == 6


def test_empty_selected_range_is_rejected(tmp_path: Path) -> None:
    root, _ = _radio_dataset(tmp_path / "radio", include_bad=False)
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    with pytest.raises(ValueError, match="No FITS files"):
        store.create_review(
            {
                "root": str(root),
                "frequencies_mhz": [149],
                "polarizations": ["RR"],
                "start_index": 10,
                "end_index": 20,
            }
        )


def test_api_flow_and_path_boundaries(tmp_path: Path) -> None:
    root, _bad_path = _radio_dataset(tmp_path / "radio")
    output_root = tmp_path / "reviews"
    app = create_app(
        allowed_roots=[tmp_path],
        output_root=output_root,
        stop_on_client_close=False,
    )
    client = app.test_client()

    assert client.get("/api/health").get_json() == {"ok": True}
    config = client.get("/api/config").get_json()["preview_display"]
    assert config["colormaps"] == [
        "coolwarm",
        "hot",
        "inferno",
        "magma",
        "viridis",
        "plasma",
        "jet",
        "cividis",
    ]
    assert config["transforms"] == ["robust_asinh", "linear"]
    assert config["range_modes"] == ["auto", "fixed"]
    assert config["percentile_bounds"] == [0.0, 100.0]
    assert config["percentile_defaults"] == [50.0, 99.7]
    assert config["defaults"] == PreviewDisplaySettings().to_dict()
    assert client.get("/").headers["Cache-Control"] == "no-store, max-age=0"
    assert (
        client.get("/static/app.js").headers["Cache-Control"] == "no-store, max-age=0"
    )
    assert client.get("/api/files", query_string={"path": str(root)}).status_code == 200
    outside = tmp_path.parent / "outside-review-input"
    outside.mkdir(exist_ok=True)
    assert (
        client.get("/api/files", query_string={"path": str(outside)}).status_code == 403
    )

    discovery = client.post("/api/discover", json={"root": str(root)})
    assert discovery.status_code == 200
    response = client.post(
        "/api/reviews",
        json={
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "start_index": 0,
        },
    )
    assert response.status_code == 201
    review = response.get_json()["review"]
    candidate = review["candidates"][0]

    preview = client.get(
        f"/api/reviews/{review['review_id']}/candidates/{candidate['candidate_id']}/preview"
    )
    assert preview.status_code == 200
    assert preview.mimetype == "image/png"
    fixed_preview = client.get(
        f"/api/reviews/{review['review_id']}/candidates/"
        f"{candidate['candidate_id']}/preview",
        query_string={
            "cmap": "viridis",
            "transform": "linear",
            "range_mode": "fixed",
            "vmin": "0",
            "vmax": "20000000",
        },
    )
    assert fixed_preview.status_code == 200
    assert fixed_preview.mimetype == "image/png"
    percentile_fixed_preview = client.get(
        f"/api/reviews/{review['review_id']}/candidates/"
        f"{candidate['candidate_id']}/preview",
        query_string={
            "cmap": "viridis",
            "transform": "linear",
            "range_mode": "fixed",
            "pmin": "50",
            "pmax": "99.7",
        },
    )
    assert percentile_fixed_preview.status_code == 200
    assert percentile_fixed_preview.mimetype == "image/png"
    invalid_cmap = client.get(
        f"/api/reviews/{review['review_id']}/candidates/"
        f"{candidate['candidate_id']}/preview",
        query_string={"cmap": "invalid"},
    )
    assert invalid_cmap.status_code == 400
    assert "cmap must be one of" in invalid_cmap.get_json()["error"]
    invalid_range = client.get(
        f"/api/reviews/{review['review_id']}/candidates/"
        f"{candidate['candidate_id']}/preview",
        query_string={"range_mode": "fixed", "vmin": "10", "vmax": "5"},
    )
    assert invalid_range.status_code == 400
    assert "vmin must be less than vmax" in invalid_range.get_json()["error"]
    invalid_percentiles = client.get(
        f"/api/reviews/{review['review_id']}/candidates/"
        f"{candidate['candidate_id']}/preview",
        query_string={"pmin": "99.7", "pmax": "50"},
    )
    assert invalid_percentiles.status_code == 400
    assert "percentiles must satisfy" in invalid_percentiles.get_json()["error"]
    assert (
        client.get(
            f"/api/reviews/{review['review_id']}/candidates/not-a-candidate/preview"
        ).status_code
        == 404
    )

    patched = client.patch(
        f"/api/reviews/{review['review_id']}",
        json={"decisions": {candidate["candidate_id"]: "bad"}},
    )
    assert patched.status_code == 200
    finalized = client.post(
        f"/api/reviews/{review['review_id']}/finalize",
        json={"mode": "completed"},
    )
    assert finalized.status_code == 200
    assert finalized.get_json()["review"]["summary"]["final_bad_count"] == 1

    manifest = client.get(f"/api/reviews/{review['review_id']}/manifest.json")
    table = client.get(f"/api/reviews/{review['review_id']}/table.csv")
    assert manifest.status_code == 200
    assert table.status_code == 200
    assert json.loads(manifest.data)["status"] == "completed"
    assert b"final_quality" in table.data


def test_open_browser_happens_after_server_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class DummyServer:
        def serve_forever(self) -> None:
            events.append("serve")

        def server_close(self) -> None:
            events.append("close")

        def shutdown(self) -> None:
            events.append("shutdown")

    def fake_make_server(*_args, **_kwargs):
        events.append("bind")
        return DummyServer()

    monkeypatch.setattr("werkzeug.serving.make_server", fake_make_server)
    monkeypatch.setattr(
        review_application.webbrowser,
        "open",
        lambda _url: events.append("browser"),
    )
    result = review_application.main(
        [
            "--allowed-roots",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "reviews"),
            "--open-browser",
        ]
    )
    assert result == 0
    assert events == ["bind", "browser", "serve", "close"]


def test_all_frame_http_endpoints_are_paginated_and_idempotent(tmp_path: Path) -> None:
    root, _ = _radio_dataset(tmp_path / "radio")
    app = create_app(
        allowed_roots=[tmp_path],
        output_root=tmp_path / "reviews",
        stop_on_client_close=False,
    )
    client = app.test_client()
    response = client.post(
        "/api/reviews",
        json={
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "candidate_strategy": "rules",
            "review_scope": "all_scanned",
        },
    )
    assert response.status_code == 201
    review = response.get_json()["review"]
    review_id = review["review_id"]
    page = client.get(
        f"/api/reviews/{review_id}/frames",
        query_string={"offset": 1, "limit": 2},
    )
    assert page.status_code == 200
    frames = page.get_json()["frames"]
    assert [frame["ordinal"] for frame in frames] == [2, 3]
    frame = frames[0]

    preview = client.get(f"/api/reviews/{review_id}/frames/{frame['file_id']}/preview")
    assert preview.status_code == 200
    assert preview.mimetype == "image/png"
    for _ in range(2):
        viewed = client.post(
            f"/api/reviews/{review_id}/frames/{frame['file_id']}/viewed"
        )
        assert viewed.status_code == 200
    assert viewed.get_json()["review"]["summary"]["viewed_frame_count"] == 1

    labelled = client.patch(
        f"/api/reviews/{review_id}/frames/{frame['file_id']}",
        json={
            "label": {
                "quality_label": "degraded",
                "event_tags": [],
                "artifact_tags": ["noise"],
            }
        },
    )
    assert labelled.status_code == 200
    audit = client.get(f"/api/reviews/{review_id}/audit.csv")
    assert audit.status_code == 200
    assert audit.data.count(frame["file_id"].encode()) == 1


def test_frontend_is_independent_and_exposes_review_controls() -> None:
    package = APPS_ROOT / "solar_apps" / "frontends" / "radio_bad_frame_review"
    html = (package / "templates" / "index.html").read_text(encoding="utf-8")
    javascript = (package / "static" / "app.js").read_text(encoding="utf-8")
    stylesheet = (package / "static" / "style.css").read_text(encoding="utf-8")

    assert "Radio Bad Frame Review" in html
    assert 'id="candidate-rows"' in html
    assert 'id="complete-review"' in html
    assert 'id="skip-review"' in html
    assert 'id="review-scope"' in html
    assert 'id="preview-cmap"' in html
    assert 'id="preview-transform"' in html
    assert 'id="preview-range-mode"' in html
    assert 'id="preview-pmin"' in html
    assert 'id="preview-pmax"' in html
    assert 'id="frequency-select-all"' in html
    assert 'id="frequency-clear-all"' in html
    assert 'id="frame-rows"' in html
    assert 'id="scan-progress"' in html
    assert 'API + "/reviews/"' in javascript
    assert "saveDecision" in javascript
    assert "finalizeReview" in javascript
    assert "createClientId" in javascript
    assert "selectedFrameEstimate" in javascript
    assert "markFrameViewed" in javascript
    assert "initializeAllFrameView" in javascript
    assert "previewDisplayParameters" in javascript
    assert "refreshActivePreview" in javascript
    assert (
        "function refreshActivePreview() {\n"
        "  updatePreviewDisplayControls();\n"
        "  if (!state.review) return;" in javascript
    )
    assert "setFrequencySelection" in javascript
    assert "setFrequencySelection(true)" in javascript
    assert "setFrequencySelection(false)" in javascript
    assert 'params.set("pmin", String(pmin))' in javascript
    assert 'params.set("pmax", String(pmax))' in javascript
    assert "Scanning fixed per-frequency intensity range" in javascript
    assert 'window.addEventListener("solar-ui-state-restored"' in javascript
    assert 'review.status === "skipped" ? summary.final_bad_count' in javascript
    assert "[hidden] { display: none !important; }" in stylesheet
    assert "grid-template-columns: 310px" in stylesheet
    assert '#preview-display-help[data-kind="error"]' in stylesheet
    assert "#preview-percentile-range" in stylesheet
    assert "solar_apps.frontends.workbench" not in (package / "server.py").read_text(
        encoding="utf-8"
    )
