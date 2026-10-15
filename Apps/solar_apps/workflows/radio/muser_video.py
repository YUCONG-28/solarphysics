"""Encode full-interval raw radio/AIA movies from the approved preview figure."""

import json
import time
from functools import lru_cache
from pathlib import Path

import astropy.units as u
import matplotlib.dates as mdates
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.lines import Line2D
from solar_toolkit.radio.muser import hpc_grid, peak_level


def video_source_ids(config):
    """Keep every requested MUSER frequency independent in both plan consumers."""
    frequencies = config["frequencies_mhz"]
    return [
        *(f"MUSER:{frequency:g}" for frequency in frequencies),
        *(f"DART:{frequency:g}" for frequency in frequencies),
        "AIA:171",
        "AIA:304",
    ]


def _muser_band_references(references, frequency):
    """Freeze exact logical planes; a missing band excludes only that observation."""
    from solar_toolkit.radio.muser_comparison import FrameRef

    output = []
    for ref in references:
        if not ref.payload:
            raise ValueError(
                "Rescan MUSER images to obtain exact logical frequency references"
            )
        planes = ref.payload.get("logical_planes", (ref.payload,))
        candidates = [
            plane
            for plane in planes
            if plane.get("frequency_mhz") is not None
            and plane.get("polarization") in {"L+R", "I", "SUM", "UNKNOWN", None}
        ]
        if not candidates:
            continue
        plane = min(
            candidates,
            key=lambda value: (
                abs(value["frequency_mhz"] - frequency),
                value["frame_id"],
            ),
        )
        if abs(plane["frequency_mhz"] - frequency) > 5:
            continue
        payload = dict(ref.payload, **plane)
        payload.pop("logical_planes", None)
        output.append(FrameRef(ref.seconds, ref.paths, stamp=ref.time, payload=payload))
    return output


def build_video_match_plan(config, muser, dart, aia, bounds=None):
    """Freeze separate frequency layers and expand the complete largest layer."""
    from solar_toolkit.time import TimeFrame, TimeStream, build_sequence_match_plan

    groups = [
        (f"MUSER:{frequency:g}", _muser_band_references(muser, frequency), 0.1)
        for frequency in config["frequencies_mhz"]
    ]
    groups += [
        (f"DART:{frequency:g}", values, 0.1)
        for frequency, values in zip(config["frequencies_mhz"], dart)
    ]
    groups += [(f"AIA:{wave}", values, 12.0) for wave, values in zip((171, 304), aia)]
    window = video_image_time_window(config, bounds)
    streams = []
    for source_id, values, tolerance in groups:
        frames = [TimeFrame(ref.frame_id, ref.time, ref) for ref in values]
        streams.append(TimeStream(source_id, frames, tolerance_seconds=tolerance))
    return build_sequence_match_plan(streams, **window)


def video_image_time_window(config, bounds=None):
    """Explicit image bounds take precedence over a legacy spectrum-derived window."""
    from solar_toolkit.time import normalize_time

    if (
        config.get("image_time_start") is not None
        or config.get("image_time_end") is not None
    ):
        return {
            "time_start": config.get("image_time_start"),
            "time_end": config.get("image_time_end"),
        }
    if bounds is None:
        return {"time_start": None, "time_end": None}
    return {
        "time_start": normalize_time(bounds[0], format="unix"),
        "time_end": normalize_time(bounds[1], format="unix"),
    }


def write_video(
    fig, config, output, muser, dart, aia, bounds, removed, colors, *, plan=None
):
    """Keep spectra/background/color limits frozen; only sources, AIA and UTC move."""
    import av
    from solar_toolkit.radio.time_matching import (
        materialize_radio_frame,
        selection_metadata,
    )
    from solar_toolkit.time import TimeFrame

    frequencies = config["frequencies_mhz"]
    fps = float(config.get("video_fps", 10))
    plan = (
        plan
        if plan is not None
        else build_video_match_plan(config, muser, dart, aia, bounds)
    )
    source_ids = video_source_ids(config)
    frames = [
        (
            row,
            [
                (
                    row.selections[source].frame.payload
                    if row.selections[source].frame
                    else None
                )
                for source in source_ids
            ],
        )
        for row in plan.rows
    ]
    skipped = []
    if not frames:
        raise ValueError("No valid video frames")
    if removed:
        raise ValueError(
            "Full-frame video requires the original, uncompressed UTC spectrum axis"
        )
    top = fig.axes[:2]
    spectra = [
        ax
        for ax in fig.axes
        if ax.get_title(loc="left").startswith(("MUSER-L", "Chashan"))
    ]
    cursors = [ax.lines[0] for ax in spectra]
    fov = config["fov_arcsec"]
    aia_references = {ref.frame_id: ref for group in aia for ref in group}

    @lru_cache(maxsize=8)
    def aia_data(frame_id):
        ref = aia_references[frame_id]
        if ref.payload:
            pixels, header = materialize_radio_frame(
                TimeFrame(ref.frame_id, ref.time, ref.payload)
            )
            m = sunpy.map.Map((pixels, header))
        else:
            m = sunpy.map.Map(ref.paths[0])
        m = m.submap(
            SkyCoord(fov[0] * u.arcsec, fov[2] * u.arcsec, frame=m.coordinate_frame),
            top_right=SkyCoord(
                fov[1] * u.arcsec, fov[3] * u.arcsec, frame=m.coordinate_frame
            ),
        )
        m = m.resample(
            u.Quantity([min(512, m.data.shape[1]), min(512, m.data.shape[0])], u.pix)
        )
        x, y = hpc_grid(m.wcs.to_header(), m.data.shape)
        data = np.log10(np.maximum(m.data.astype(float), 0) + 1)
        return x, y, data, float(m.rsun_obs.to_value(u.arcsec))

    aia_clims = [
        ax.collections[0].get_clim() if ax.collections else (0.0, 1.0) for ax in top
    ]
    # Freeze geometry and the static spectral raster once, then redraw top panels.
    fig.set_dpi(120)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    for ax in top:
        ax.set_visible(False)
    for c in cursors:
        c.set_visible(False)
    fig._suptitle.set_visible(False)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    for ax in top:
        ax.set_visible(True)
    for c in cursors:
        c.set_visible(True)
    fig._suptitle.set_visible(True)
    width, height = fig.canvas.get_width_height()
    path = Path(output) / "comparison.mp4"
    tmp = Path(output) / "comparison.partial.mp4"
    container = av.open(str(tmp), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    records = []
    started = time.time()
    try:
        for number, (row, refs) in enumerate(frames):
            target = float(Time(row.time.utc, scale="utc").unix)
            overlays = []
            pairs = []
            for k, freq in enumerate(frequencies):
                ref = refs[k]
                mf = float("nan")
                if ref is not None:
                    mf = ref.payload["frequency_mhz"]
                    data, hdr = materialize_radio_frame(
                        TimeFrame(ref.frame_id, ref.time, ref.payload)
                    )
                    x, y = hpc_grid(hdr, data.shape)
                    overlays.append((x, y, data, peak_level(data), colors[k], "-"))
                ref = refs[len(frequencies) + k]
                if ref is not None:
                    data, hdr = materialize_radio_frame(
                        TimeFrame(ref.frame_id, ref.time, ref.payload)
                    )
                    x, y = hpc_grid(hdr, data.shape)
                    overlays.append((x, y, data, peak_level(data), colors[k], "--"))
                pairs.append([mf, freq])
            fig.canvas.restore_region(background)
            for k, (ax, wave, ref) in enumerate(zip(top, [171, 304], refs[-2:])):
                ax.clear()
                ax.set_facecolor("black")
                radius = None
                if ref is not None:
                    x, y, data, radius = aia_data(ref.frame_id)
                    ax.pcolormesh(
                        x,
                        y,
                        data,
                        shading="nearest",
                        cmap=f"sdoaia{wave}",
                        vmin=aia_clims[k][0],
                        vmax=aia_clims[k][1],
                        rasterized=True,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"AIA {wave}: no observation",
                        transform=ax.transAxes,
                        ha="center",
                        color="white",
                    )
                for xx, yy, z, level, color, style in overlays:
                    ax.contour(
                        xx,
                        yy,
                        z,
                        levels=[level],
                        colors=[color],
                        linewidths=1.8,
                        linestyles=style,
                    )
                import matplotlib.pyplot as plt

                if radius is not None:
                    ax.add_patch(
                        plt.Circle(
                            (0, 0), radius, fill=False, color="white", alpha=0.4, lw=0.7
                        )
                    )
                ax.set(
                    xlim=fov[:2],
                    ylim=fov[2:],
                    aspect="equal",
                    xlabel="Solar X [arcsec]",
                    ylabel="Solar Y [arcsec]" if k == 0 else "",
                )
                ax.set_title(
                    f"AIA {wave} Å | {ref.time.utc[11:] if ref is not None else 'no data'} UTC",
                    fontsize=12,
                )
                handles = [
                    Line2D(
                        [],
                        [],
                        color=colors[j],
                        lw=2,
                        label=f"M {mf:.2f} / D {df:g} MHz",
                    )
                    for j, (mf, df) in enumerate(pairs)
                ]
                handles += [
                    Line2D([], [], color="white", lw=2, ls=s, label=n)
                    for s, n in [("-", "MUSER I"), ("--", "DART RR+LL")]
                ]
                legend = ax.legend(
                    handles=handles,
                    loc=config.get("legend_location", "lower left"),
                    fontsize=8,
                    facecolor="#111111",
                    framealpha=0.85,
                )
                for t in legend.get_texts():
                    t.set_color("white")
                fig.draw_artist(ax)
            cursor_time = target - sum(
                np.clip(target - a, 0, b - a) for a, b in removed
            )
            xx = mdates.date2num(Time(cursor_time, format="unix").to_datetime())
            for ax, c in zip(spectra, cursors):
                c.set_xdata([xx, xx])
                ax.draw_artist(c)
            title = f"{config['label']} | {row.time.utc} UTC\nRaw 90% peak contours · total intensity · fixed FOV (outside contours clipped)\nSpectra: relative background [dB]; no interpolation"
            if removed:
                title += "; missing UTC intervals removed"
            fig._suptitle.set_text(title)
            fig.draw_artist(fig._suptitle)
            frame = av.VideoFrame.from_ndarray(
                np.asarray(fig.canvas.buffer_rgba()).copy(), format="rgba"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            if number in {0, len(frames) // 2, len(frames) - 1}:
                from PIL import Image

                Image.frombuffer(
                    "RGBA",
                    (width, height),
                    bytes(fig.canvas.buffer_rgba()),
                    "raw",
                    "RGBA",
                    0,
                    1,
                ).save(Path(output) / f"frame-{number:04d}.png")
            records.append(
                {
                    "frame": number,
                    "utc": row.time.utc,
                    "time_tai_ns": row.time.tai_ns,
                    "time_match": selection_metadata(row, plan=plan),
                    "inputs": [str(p) for r in refs if r is not None for p in r.paths],
                }
            )
            if number % 10 == 0:
                (Path(output) / "video-status.json").write_text(
                    json.dumps(
                        {
                            "phase": "encoding",
                            "frames": number + 1,
                            "total": len(frames),
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                )
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        tmp.rename(path)
    except BaseException:
        container.close()
        raise
    meta = {
        "config": config,
        "fps": fps,
        "frames": len(frames),
        "duration_seconds": len(frames) / fps,
        "playback": "one frame per longest-route observation; constant display fps, original UTC in every frame",
        "time_match_plan": plan.to_dict(),
        "first_utc": records[0]["utc"],
        "last_utc": records[-1]["utc"],
        "frame_records": records,
        "skipped": skipped,
        "aia_color_limits": aia_clims,
        "spectra": "fixed approved-preview P20 backgrounds and color limits throughout video",
        "removed_intervals_utc": [
            Time(x, format="unix").isot.tolist() for x in removed
        ],
    }
    from .muser_preview import _digest

    meta["video_sha256"] = _digest(path)
    meta["implementation_sha256"] = _digest(__file__)
    (Path(output) / "video.json").write_text(json.dumps(meta, indent=2))
    (Path(output) / "video-status.json").write_text(
        json.dumps({"phase": "complete", "frames": len(frames)})
    )
    return path
