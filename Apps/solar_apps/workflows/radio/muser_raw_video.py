"""Export original-image comparison frames and video with explicit spectral input."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from astropy.time import Time
from astropy.wcs import WCS
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from .muser_raw_preview import (
    build_preview,
    compressed_time,
    digest,
    draw_original,
    _json_safe,
    radio_bands,
)


def image_paths(directory):
    return sorted(
        p
        for p in Path(directory).rglob("*")
        if p.is_file()
        and p.name.lower().endswith((".fits", ".fit", ".fts", ".fits.gz"))
    )


def build_plan(config):
    """Restrict every source to the event before pairing and sequence planning."""
    from solar_toolkit.radio.time_matching import index_radio_frames, pair_radio_frames
    from solar_toolkit.time import (
        TimeStream,
        build_sequence_match_plan,
        filter_time_frames,
    )

    start, end = config["display_window_utc"]
    window = dict(time_start=start, time_end=end)

    def indexed(path, **kwargs):
        return filter_time_frames(
            index_radio_frames(image_paths(path), **kwargs), **window
        )

    muser_dir = Path(config["muser_directory"])
    if (muser_dir / "img_fits").is_dir():
        muser_dir /= "img_fits"
    bands = radio_bands(config)
    sources = {}
    for wave in (171, 304):
        sources[f"AIA{wave}"] = [
            f
            for f in indexed(Path(config["aia_directory"]) / str(wave))
            if float(f.payload["header"].get("WAVELNTH", -1)) == wave
        ]
    for i, band in enumerate(bands):
        muser = indexed(muser_dir, frequency_mhz=band["muser_mhz"], polarization="I")
        dart_dir = Path(band["dart_directory"])
        frequency = band["dart_mhz"]
        if config["dart_mode"] == "I":
            dart = indexed(
                dart_dir / "I" if (dart_dir / "I").is_dir() else dart_dir,
                frequency_mhz=frequency,
                polarization="I",
            )
        elif config["dart_mode"] == "RR+LL":
            left = indexed(dart_dir / "LL", frequency_mhz=frequency, polarization="LL")
            right = indexed(dart_dir / "RR", frequency_mhz=frequency, polarization="RR")
            dart = pair_radio_frames(left, right, tolerance_seconds=0.1, **window)
        else:
            raise ValueError("DART mode must be I or RR+LL")
        suffix = str(i) if len(bands) == 2 else ""
        sources["MUSER" + suffix] = muser
        sources["DART" + suffix] = dart
    streams = [
        TimeStream(
            key,
            frames,
            tolerance_seconds=config.get(
                (
                    "aia_tolerance_seconds"
                    if key.startswith("AIA")
                    else "radio_tolerance_seconds"
                ),
                12.0 if key.startswith("AIA") else 0.25,
            ),
        )
        for key, frames in sources.items()
    ]
    plan = build_sequence_match_plan(streams, **window)
    if not plan.rows:
        raise ValueError("No in-window image observations")
    return plan


def frame_pixels(frame):
    """Select native pixels and native WCS using the same stride as the preview."""
    from solar_toolkit.radio.time_matching import materialize_radio_frame
    from solar_toolkit.radio.raw_comparison import _validate_image

    data, header = materialize_radio_frame(frame)
    _validate_image(data, header)
    stride = max(1, int(np.ceil(max(data.shape) / 1024)))
    yy, xx = np.meshgrid(
        np.arange(0, data.shape[0], stride),
        np.arange(0, data.shape[1], stride),
        indexing="ij",
    )
    lon, lat = WCS(header, naxis=2).all_pix2world(xx, yy, 0)
    return (
        ((lon + 180) % 360 - 180) * 3600,
        lat * 3600,
        np.ma.masked_invalid(data[::stride, ::stride]),
        header,
    )


def readable_colorbar_ticks(cax, norm, renderer):
    """Repair overlapping labels without changing the normalization or pixels."""
    labels = [
        label
        for label in cax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    boxes = sorted(
        (label.get_window_extent(renderer) for label in labels), key=lambda b: b.y0
    )
    if not any(a.y1 + 3 > b.y0 for a, b in zip(boxes, boxes[1:])):
        return False
    ticks = norm.inverse(np.linspace(0, 1, 5))
    cax.yaxis.set_major_locator(FixedLocator(ticks))
    cax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.2g}"))
    cax.yaxis.set_minor_locator(NullLocator())
    return True


def _panel_title(key, panel, config):
    if key.startswith("AIA"):
        return f"AIA {key[3:]} Å"
    frequency = panel.get("frequency_mhz")
    if frequency is None:
        index = int(key[-1]) if key[-1].isdigit() else 0
        band = radio_bands(config)[index]
        frequency = band["muser_mhz" if key.startswith("MUSER") else "dart_mhz"]
    if key.startswith("MUSER"):
        return f"MUSER {float(frequency):.7f} MHz | nominal I"
    return f"DART {float(frequency):g} MHz | {config['dart_mode']}"


def _movie_scales(fig, config, source_ids, panels, plan):
    """Keep reference scales or establish one from the first usable planned frame."""
    from solar_toolkit.radio.raw_comparison import PreparedImage
    from solar_toolkit.radio.time_matching import materialize_radio_frame

    scales = []
    for i, (key, panel) in enumerate(zip(source_ids, panels)):
        ax, cax = fig.axes[2 * i : 2 * i + 2]
        metadata = dict(panel)
        if ax.collections:
            metadata["scale_source"] = "reference_preview"
            scales.append(metadata)
            continue
        seen, rejected = set(), []
        fov = [*ax.get_xlim(), *ax.get_ylim()]
        for row in plan.rows:
            selection = row.selections[key]
            frame = selection.frame
            if frame is None or frame.frame_id in seen:
                continue
            seen.add(frame.frame_id)
            try:
                data, header = materialize_radio_frame(frame)
                candidate = dict(
                    status="ok",
                    utc=frame.time.utc,
                    delta_seconds=selection.delta_ns / 1e9,
                    unit=header.get("BUNIT", "source units"),
                    frequency_mhz=frame.payload.get("frequency_mhz"),
                )
                prepared = PreparedImage(data, header, candidate)
                for text in list(ax.texts):
                    text.remove()
                cax.set_visible(True)
                candidate = draw_original(
                    ax,
                    cax,
                    prepared,
                    _panel_title(key, candidate, config),
                    fov,
                    wavelength=int(key[3:]) if key.startswith("AIA") else None,
                )
                if candidate["status"] == "ok":
                    metadata = dict(
                        candidate,
                        scale_source="first_valid_planned_frame",
                        scale_frame_id=frame.frame_id,
                        scale_reference_utc=frame.time.utc,
                    )
                    break
                rejected.append(
                    {"frame_id": frame.frame_id, "reason": candidate["reason"]}
                )
            except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                rejected.append({"frame_id": frame.frame_id, "reason": str(exc)})
        else:
            metadata.update(scale_source="no_available_planned_frame", display=None)
            cax.set_visible(False)
        if rejected:
            metadata["scale_rejected_frames"] = rejected
        scales.append(metadata)
    return scales


def encode_figure(fig, config, output, panels, removed, *, plan):
    """Cache reused panel rasters, keep static spectra and scales, record every row."""
    import av
    from PIL import Image
    from solar_toolkit.radio.time_matching import selection_metadata

    fps = int(config.get("video_fps", 10))
    if fps <= 0:
        raise ValueError("video_fps must be a positive integer")
    output = Path(output)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    dual = len(config.get("radio_bands", [])) == 2
    source_ids = (
        ["AIA171", "MUSER0", "DART0", "AIA304", "MUSER1", "DART1"]
        if dual
        else ["AIA171", "AIA304", "MUSER", "DART"]
    )
    count = len(source_ids)
    if len(panels) != count:
        raise ValueError("Image panel count does not match frequency configuration")
    axes = [fig.axes[2 * i] for i in range(count)]
    spectral_axes = [fig.axes[2 * count + i] for i in (0, 2)]
    cursors = [ax.lines[-1] for ax in spectral_axes]
    movie_panels = _movie_scales(fig, config, source_ids, panels, plan)
    norms = [ax.collections[0].norm if ax.collections else None for ax in axes]
    cmaps = [ax.collections[0].cmap if ax.collections else None for ax in axes]
    titles = [
        _panel_title(key, panel, config) for key, panel in zip(source_ids, movie_panels)
    ]
    fig.set_dpi(int(config.get("video_dpi", 100)))
    fig.canvas.draw()
    repaired_colorbars = [
        i
        for i, key in enumerate(source_ids)
        if not key.startswith("AIA")
        and norms[i] is not None
        and readable_colorbar_ticks(
            fig.axes[2 * i + 1], norms[i], fig.canvas.get_renderer()
        )
    ]
    if repaired_colorbars:
        fig.canvas.draw()
    fig.set_layout_engine(None)
    # Each cached panel includes tick labels; pad enough to erase old UTC titles.
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer).expanded(1.015, 1.035) for ax in axes]
    for ax in axes:
        ax.set_visible(False)
    for cursor in cursors:
        cursor.set_visible(False)
    fig._suptitle.set_visible(False)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    for ax in axes:
        ax.set_visible(True)
    for cursor in cursors:
        cursor.set_visible(True)
    fig._suptitle.set_visible(True)
    width, height = fig.canvas.get_width_height()
    if width % 2 or height % 2:
        raise ValueError("H.264 requires even canvas dimensions")
    current = [None] * count
    regions = [None] * count
    # TimeFrame payloads are dicts: cache by immutable identity instead.
    references = {
        s.frame.frame_id: s.frame
        for row in plan.rows
        for s in row.selections.values()
        if s.frame is not None
    }
    cached = lru_cache(maxsize=8)(lambda identity: frame_pixels(references[identity]))
    partial = output / "comparison.partial.mp4"
    container = av.open(str(partial), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "18", "preset": "fast", "threads": "2"}
    records = []
    started = time.monotonic()
    old_title = fig._suptitle.get_text().split("\n")
    try:
        for number, row in enumerate(plan.rows):
            fig.canvas.restore_region(background)
            record = selection_metadata(row, plan=plan)
            for i, key in enumerate(source_ids):
                selection = row.selections[key]
                frame = selection.frame
                identity = frame.frame_id if frame else None
                cache_key = (identity,)
                ax = axes[i]
                if current[i] != cache_key:
                    for artist in list(ax.collections):
                        artist.remove()
                    for text in list(ax.texts):
                        text.remove()
                    if frame and norms[i] is not None:
                        x, y, data, header = cached(identity)
                        ax.pcolormesh(
                            x,
                            y,
                            data,
                            shading="nearest",
                            norm=norms[i],
                            cmap=cmaps[i],
                            rasterized=True,
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "NO DATA",
                            ha="center",
                            transform=ax.transAxes,
                            fontsize=19,
                        )
                    ax.set_title("")
                    ax.draw(renderer)
                    regions[i] = fig.canvas.copy_from_bbox(boxes[i])
                    current[i] = cache_key
                else:
                    fig.canvas.restore_region(regions[i])
                if frame and norms[i] is not None:
                    delta = selection.delta_ns / 1e9
                    warning = (
                        " [large offset]" if not selection.within_tolerance else ""
                    )
                    ax.set_title(
                        f"{titles[i]}\n{frame.time.utc[11:23]} UTC | Δt {delta:+.3f} s{warning}",
                        fontsize=10,
                        color="#9b2800" if warning else "#222222",
                    )
                else:
                    ax.set_title(titles[i], fontsize=11)
                ax.draw_artist(ax.title)
            target = float(Time(row.time.utc, scale="utc").unix)
            cursor_x = mdates.date2num(
                Time(compressed_time(target, removed), format="unix").to_datetime()
            )
            for ax, cursor in zip(spectral_axes, cursors):
                cursor.set_xdata([cursor_x, cursor_x])
                ax.draw_artist(cursor)
            fig._suptitle.set_text(
                "\n".join(
                    [
                        f"{config['event_id']} | UTC {row.time.utc[:23]} | {number+1}/{len(plan.rows)}",
                        old_title[1],
                        old_title[2] + f" | {fps} fps (sequence playback)",
                    ]
                )
            )
            fig.draw_artist(fig._suptitle)
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            path = frames_dir / f"{number:06d}.png"
            Image.fromarray(rgb).save(path, compress_level=1)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            record.update(
                frame_number=number,
                png=path.name,
                png_sha256=digest(path),
                display_cursor_utc=Time(
                    compressed_time(target, removed), format="unix"
                ).isot,
            )
            records.append(record)
            if number % 25 == 0 or number + 1 == len(plan.rows):
                progress = {
                    "status": "rendering",
                    "frames_done": number + 1,
                    "total_frames": len(plan.rows),
                    "elapsed_seconds": time.monotonic() - started,
                    "utc": row.time.utc,
                }
                (output / "status.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        cached.cache_clear()
    # Decode the complete movie before changing its name to a finished product.
    with av.open(str(partial)) as check:
        decoded = sum(1 for _ in check.decode(video=0))
    if decoded != len(records):
        raise ValueError(f"Decoded {decoded} frames, expected {len(records)}")
    final = output / "comparison.mp4"
    partial.rename(final)
    result = {
        "config": config,
        "panel_source_ids": source_ids,
        "frames": len(records),
        "fps": fps,
        "duration_seconds": len(records) / fps,
        "dimensions": [width, height],
        "decoded_frames": decoded,
        "sha256": digest(final),
        "plan_fingerprint": plan.fingerprint,
        "plan_summary": plan.to_dict()["summary"],
        "image_scales": movie_panels,
        "records": records,
        "time_interpolation": False,
        "colorbar_labels_repaired_panels": repaired_colorbars,
        "removed_spectrum_intervals_unix": removed,
        "playback": "One planned observation per frame; fixed fps, not a uniform physical clock",
        "image_matching": "In-window hierarchical monotone full coverage; reuse originals; label excess offsets",
        "implementation_sha256": {Path(__file__).name: digest(__file__)},
    }
    (output / "video.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
    )
    (output / "status.json").write_text(
        json.dumps(
            {"status": "complete", "frames": decoded, "sha256": result["sha256"]}
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    plan = build_plan(config)
    if args.plan_only:
        Path(args.output).write_text(json.dumps(plan.to_dict(), indent=2))
        return 0
    build_preview(
        config,
        args.output,
        frame_renderer=lambda fig, cfg, out, panels, removed: encode_figure(
            fig, cfg, out, panels, removed, plan=plan
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
