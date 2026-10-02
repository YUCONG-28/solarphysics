"""Encode full-interval raw radio/AIA movies from a fixed preview figure."""

from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import sunpy.map
from solar_toolkit.radio.muser import read_muser_images, hpc_grid, peak_level
from solar_toolkit.radio.muser_comparison import nearest_index, load_dart_sum
from solar_toolkit.radio.centers import parse_observation_time


def write_video(fig, config, output, muser, dart, aia, bounds, removed, colors):
    """Keep spectra/background/color limits frozen; only sources, AIA and UTC move."""
    import av

    groups = [muser, *dart, *aia]
    axes = [[r.seconds for r in g] for g in groups]
    frequencies = config["frequencies_mhz"]
    fps = float(config.get("video_fps", 10))
    frames = []
    skipped = []
    for first in dart[0]:
        target = first.seconds
        if not bounds[0] <= target <= bounds[1]:
            continue
        if any(a <= target < b for a, b in removed):
            skipped.append(
                {
                    "utc": Time(target, format="unix").isot,
                    "reason": "removed spectrum interval",
                }
            )
            continue
        indices = [
            nearest_index(ts, target, 0.1 if k <= len(dart) else 12)
            for k, ts in enumerate(axes)
        ]
        if any(x is None for x in indices):
            skipped.append(
                {
                    "utc": Time(target, format="unix").isot,
                    "reason": "incomplete radio/AIA pairing",
                }
            )
            continue
        refs = [g[i] for g, i in zip(groups, indices)]
        pol_times = [
            float(Time(parse_observation_time(p, fits.getheader(p))).unix)
            for ref in refs[1 : 1 + len(dart)]
            for p in ref.paths
        ]
        if max(abs(t - target) for t in pol_times) > 0.100001:
            skipped.append(
                {
                    "utc": Time(target, format="unix").isot,
                    "reason": "polarization tolerance",
                }
            )
            continue
        frames.append((target, refs))
    if not frames:
        raise ValueError("No valid video frames")
    top = fig.axes[:2]
    spectra = [
        ax
        for ax in fig.axes
        if ax.get_title(loc="left").startswith(("MUSER-L", "Chashan"))
    ]
    cursors = [ax.lines[0] for ax in spectra]
    fov = config["fov_arcsec"]

    @lru_cache(maxsize=8)
    def aia_data(path):
        m = sunpy.map.Map(path)
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

    aia_clims = [ax.collections[0].get_clim() for ax in top]
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
        for number, (target, refs) in enumerate(frames):
            planes = read_muser_images(refs[0].paths[0])
            overlays = []
            pairs = []
            for k, freq in enumerate(frequencies):
                mf, data, hdr, beam = min(planes, key=lambda p: abs(p[0] - freq))
                if abs(mf - freq) > 5:
                    raise ValueError("MUSER frequency match exceeds 5 MHz")
                ds = load_dart_sum(refs[k + 1])
                pairs.append([mf, freq])
                for name, z, h, style in [
                    ("MUSER", data, hdr, "-"),
                    ("DART", ds.image, ds.header, "--"),
                ]:
                    x, y = hpc_grid(h, z.shape)
                    overlays.append((x, y, z, peak_level(z), colors[k], style))
            fig.canvas.restore_region(background)
            for k, (ax, wave, ref) in enumerate(zip(top, [171, 304], refs[-2:])):
                ax.clear()
                ax.set_facecolor("black")
                x, y, data, radius = aia_data(str(ref.paths[0]))
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
                    f'AIA {wave} Å | {Time(ref.seconds,format="unix").isot[11:]} UTC',
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
            title = f'{config["label"]} | {Time(target,format="unix").isot} UTC\nRaw 90% peak contours · total intensity · fixed FOV (outside contours clipped)\nSpectra: relative background [dB]; no interpolation'
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
                    "utc": Time(target, format="unix").isot,
                    "delta_seconds": [r.seconds - target for r in refs],
                    "inputs": [str(p) for r in refs for p in r.paths],
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
        "playback": "one frame per valid first-selected DART frequency observation; constant display fps, actual UTC in every frame",
        "first_utc": records[0]["utc"],
        "last_utc": records[-1]["utc"],
        "frame_records": records,
        "skipped": skipped,
        "aia_color_limits": aia_clims,
        "spectra": "fixed preview P20 backgrounds and color limits throughout video",
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
