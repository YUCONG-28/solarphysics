"""Raw-contour MUSER/DART comparison with explicitly supplied input paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from solar_toolkit.aia.background import scan_aia_folder
from solar_toolkit.radio.dart_spectrogram import (
    discover_dart_spectrogram_files,
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)
from solar_toolkit.radio.muser import (
    hpc_grid,
    peak_level,
    read_muser_spectrum,
)
from solar_toolkit.radio.muser_comparison import (
    FrameRef,
    index_images,
)

COLORS = ("#00e5ff", "#ff9b35", "#65ff6a", "#ff65c8", "#b69cff", "#ffe047")


def _requested_fov(config):
    value = config.get("fov_arcsec")
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    if a.shape != (4,) or not np.all(np.isfinite(a)) or a[0] >= a[1] or a[2] >= a[3]:
        raise ValueError("fov_arcsec requires increasing [xmin, xmax, ymin, ymax]")
    return a.tolist()


def _requested_frequencies(config):
    """Require an explicit nonempty set of finite positive frequencies."""
    frequencies = tuple(float(value) for value in config.get("frequencies_mhz", ()))
    if (
        not frequencies
        or len(set(frequencies)) != len(frequencies)
        or not np.isfinite(frequencies).all()
        or min(frequencies) <= 0
    ):
        raise ValueError(
            "frequencies_mhz requires distinct finite positive frequencies"
        )
    return frequencies


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _segments(axis):
    """Separate gaps instead of connecting them through a pcolormesh cell."""
    delta = np.diff(axis)
    if len(delta) == 0:
        return []
    breaks = np.flatnonzero(delta > 1.5 * np.median(delta)) + 1
    return [s for s in np.split(np.arange(len(axis)), breaks) if len(s) >= 2]


def _spectrum(ax, time, freq, data, bounds, target, title, label, color_limits=None):
    import matplotlib.dates as mdates

    vals = data[np.isfinite(data)]
    if not vals.size:
        raise ValueError(f"No finite spectrum values: {title}")
    lo, hi = np.percentile(vals, [2, 99.5]) if color_limits is None else color_limits
    artist = None
    x = mdates.date2num(Time(time, format="unix").to_datetime())
    for ti in _segments(np.asarray(time)):
        for fi in _segments(np.asarray(freq)):
            artist = ax.pcolormesh(
                x[ti],
                freq[fi],
                data[np.ix_(fi, ti)],
                shading="nearest",
                cmap="turbo",
                vmin=lo,
                vmax=hi,
                rasterized=True,
            )
    if artist is None:
        raise ValueError(
            "Spectrum requires at least two contiguous samples on both axes"
        )
    ax.set_xlim(mdates.date2num(Time(bounds, format="unix").to_datetime()))
    ax.set_ylim(freq[0], freq[-1])
    ax.axvline(
        mdates.date2num(Time(target, format="unix").to_datetime()),
        color="white",
        lw=1.3,
        ls="--",
    )
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_ylabel("Frequency [MHz]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    return artist, {"color_limits": [float(lo), float(hi)], "unit": label}


def build_preview(config, output):
    config = dict(config, label=config.get("label", "Radio/AIA comparison"))
    import matplotlib

    matplotlib.use("Agg")
    import astropy.units as u
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    if config.get("make_video"):
        config["remove_missing_times"] = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requested_fov = _requested_fov(config)
    frequencies = _requested_frequencies(config)
    colors = tuple(COLORS[i % len(COLORS)] for i in range(len(frequencies)))
    if config.get("remove_missing_times") and not config.get("cso_files"):
        raise ValueError("Missing-time compaction requires CSO source inputs")
    mdir = Path(config["muser_directory"])
    specpaths = list(mdir.glob("*spec.PWR_I*.fits"))
    if len(specpaths) != 1:
        raise ValueError("Select one MUSER event directory with one PWR_I spectrum")
    spec = read_muser_spectrum(specpaths[0])
    bounds = [float(spec.time.unix[0]), float(spec.time.unix[-1])]
    from solar_toolkit.time import normalize_time

    from .muser_video import (
        build_video_match_plan,
        video_image_time_window,
        video_source_ids,
    )

    window = video_image_time_window(config, [spec.time[0], spec.time[-1]])
    # Keep the user's exact image window separate from the spectral plotting axis.
    config["image_time_start"] = (
        normalize_time(window["time_start"]).utc
        if window["time_start"] is not None
        else None
    )
    config["image_time_end"] = (
        normalize_time(window["time_end"]).utc
        if window["time_end"] is not None
        else None
    )
    window = video_image_time_window(config)
    image_bounds = [
        float(Time(config[key], scale="utc").unix)
        if config[key] is not None
        else bounds[index]
        for index, key in enumerate(("image_time_start", "image_time_end"))
    ]
    if image_bounds[0] >= image_bounds[1]:
        image_bounds = list(
            bounds
        )  # finite score/display fallback; image eligibility remains open
    for index, key in enumerate(("spectrogram_time_start", "spectrogram_time_end")):
        if config.get(key) is not None:
            bounds[index] = float(
                Time(normalize_time(config[key]).utc, scale="utc").unix
            )
    print("Indexing MUSER and DART FITS headers", flush=True)
    mi = index_images((mdir / "img_fits").glob("*.fits"))
    dart = []
    for freq in frequencies:
        d = Path(config["dart_directory"]) / f"{freq:g}MHz"
        pairs = _pair_dart_in_window(d, freq, window)
        dart.append(pairs)
    aia = []
    for wave in (171, 304):
        table = scan_aia_folder(
            Path(config["aia_directory"]) / str(wave), recursive=True
        )
        table = table.loc[np.isclose(table.wavelength.astype(float), wave)]
        aia.append(index_images(table.path))
    trange = tuple(Time(image_bounds, format="unix").isot)
    curves = extract_dart_narrowband_lightcurves(
        config["spectrum_directory"],
        frequencies,
        bandwidth_mhz=2.0,
        time_range_utc=trange,
    )
    st = np.array([t.timestamp() for t in curves.time_utc])
    score = np.mean([c.stokes_i_db for c in curves.curves], axis=0)
    config["frequencies_mhz"] = list(frequencies)
    plan = build_video_match_plan(config, mi, dart, aia)
    source_ids = video_source_ids(config)
    target, refs, score_index, preview_row = _select_plan_preview(
        plan, st, score, source_ids
    )
    from solar_toolkit.radio.time_matching import selection_metadata

    preview_match = selection_metadata(plan.rows[preview_row], plan=plan)
    print("Selected UTC", Time(target, format="unix").isot, flush=True)
    overlays, overlay_meta = [], []
    inputs = {specpaths[0]}
    matched_frequencies = []
    from solar_toolkit.radio.time_matching import materialize_radio_frame
    from solar_toolkit.time import TimeFrame

    for k, freq in enumerate(frequencies):
        band_images = []
        mf, beam = float("nan"), None
        ref = refs[k]
        if ref is not None:
            mf = ref.payload["frequency_mhz"]
            image, header = materialize_radio_frame(
                TimeFrame(ref.frame_id, ref.time, ref.payload)
            )
            beam = {key: header.get(key) for key in ("BMAJ", "BMIN", "BPA")}
            band_images.append(("MUSER", image, header, mf, "-"))
            inputs.update(ref.paths)
        matched_frequencies.append(mf)
        ref = refs[len(frequencies) + k]
        if ref is not None:
            data, header = materialize_radio_frame(
                TimeFrame(ref.frame_id, ref.time, ref.payload)
            )
            band_images.append(("DART", data, header, freq, "--"))
            inputs.update(ref.paths)
        for instrument, data, hdr, f, style in band_images:
            x, y = hpc_grid(hdr, data.shape)
            level = peak_level(data)
            overlays.append((x, y, data, level, colors[k], style))
            source_id = f"{instrument}:{freq:g}"
            overlay_meta.append(
                {
                    "instrument": instrument,
                    "frequency_mhz": f,
                    "peak": level / 0.9,
                    "level": level,
                    "unit": hdr.get("BUNIT"),
                    "beam": beam if instrument == "MUSER" else None,
                    "delta_seconds": plan.rows[preview_row]
                    .selections[source_id]
                    .delta_ns
                    / 1e9,
                    "time_match": preview_match["selections"][source_id],
                }
            )
    maps = []
    aia_meta = []
    for wave, ref in zip((171, 304), refs[-2:]):
        if ref is None:
            maps.append(None)
            aia_meta.append(
                {
                    "wave_angstrom": wave,
                    "status": "no_data",
                    "path": None,
                    "utc": None,
                    "delta_seconds": None,
                }
            )
            continue
        from solar_toolkit.radio.time_matching import materialize_radio_frame
        from solar_toolkit.time import TimeFrame

        pixels, header = materialize_radio_frame(
            TimeFrame(ref.frame_id, ref.time, ref.payload)
        )
        m = sunpy.map.Map((pixels, header))
        if str(m.meta.get("CTYPE1", "")).upper() != "HPLN-TAN":
            raise ValueError("AIA must have helioprojective coordinates")
        if requested_fov is not None:
            xmin, xmax, ymin, ymax = requested_fov
            m = m.submap(
                SkyCoord(xmin * u.arcsec, ymin * u.arcsec, frame=m.coordinate_frame),
                top_right=SkyCoord(
                    xmax * u.arcsec, ymax * u.arcsec, frame=m.coordinate_frame
                ),
            )
        m = m.resample(
            u.Quantity([min(1024, m.data.shape[1]), min(1024, m.data.shape[0])], u.pix)
        )
        maps.append(m)
        inputs.add(ref.paths[0])
        aia_meta.append(
            {
                "wave_angstrom": wave,
                "path": str(ref.paths[0]),
                "utc": ref.time.utc + "Z",
                "delta_seconds": plan.rows[preview_row]
                .selections[f"AIA:{wave}"]
                .delta_ns
                / 1e9,
                "time_match": preview_match["selections"][f"AIA:{wave}"],
            }
        )
    # Determine a common FOV from actual interpolated contour vertices and solar limb.
    radius = max(
        (float(m.rsun_obs.to_value(u.arcsec)) for m in maps if m is not None),
        default=960.0,
    )
    limits = [-radius, radius, -radius, radius]
    scratch, ax = plt.subplots()
    distant = []
    outside_view = []
    for index, (x, y, data, level, _, _) in enumerate(overlays):
        cs = ax.contour(x, y, data, levels=[level])
        segments = [v for v in cs.allsegs[0] if len(v)]
        if requested_fov is not None and segments:
            vertices_all = np.concatenate(segments)
            xmin, xmax, ymin, ymax = requested_fov
            if (
                vertices_all[:, 0].max() < xmin
                or vertices_all[:, 0].min() > xmax
                or vertices_all[:, 1].max() < ymin
                or vertices_all[:, 1].min() > ymax
            ):
                outside_view.append(
                    f"{overlay_meta[index]['instrument']} {overlay_meta[index]['frequency_mhz']:.2f} MHz"
                )
        for vertices in cs.allsegs[0]:
            if len(vertices):
                if np.max(np.hypot(vertices[:, 0], vertices[:, 1])) > 2 * radius:
                    name = f"{overlay_meta[index]['instrument']} {overlay_meta[index]['frequency_mhz']:.2f} MHz"
                    if name not in distant:
                        distant.append(name)
                limits = [
                    min(limits[0], vertices[:, 0].min()),
                    max(limits[1], vertices[:, 0].max()),
                    min(limits[2], vertices[:, 1].min()),
                    max(limits[3], vertices[:, 1].max()),
                ]
    plt.close(scratch)
    pad = 0.1 * max(limits[1] - limits[0], limits[3] - limits[2])
    limits = [limits[0] - pad, limits[1] + pad, limits[2] - pad, limits[3] + pad]
    if requested_fov is not None:
        limits = requested_fov
    print("Rendering", flush=True)
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.035], height_ratios=[2.2, 1, 1])
    for k, (wave, m) in enumerate(zip((171, 304), maps)):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor("black")
        if m is not None:
            x, y = hpc_grid(m.wcs.to_header(), m.data.shape)
            values = np.log10(np.maximum(m.data.astype(float), 0) + 1)
            finite = values[np.isfinite(values)]
            ax.pcolormesh(
                x,
                y,
                values,
                shading="nearest",
                cmap=f"sdoaia{wave}",
                vmin=np.percentile(finite, 1),
                vmax=np.percentile(finite, 99.7),
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
        for xx, yy, data, level, color, style in overlays:
            ax.contour(
                xx,
                yy,
                data,
                levels=[level],
                colors=[color],
                linewidths=1.8,
                linestyles=style,
            )
        ax.add_patch(
            plt.Circle((0, 0), radius, fill=False, color="white", alpha=0.4, lw=0.7)
        )
        ax.set(
            xlim=limits[:2], ylim=limits[2:], aspect="equal", xlabel="Solar X [arcsec]"
        )
        ax.set_ylabel("Solar Y [arcsec]" if k == 0 else "")
        utc = aia_meta[k]["utc"]
        ax.set_title(
            f"AIA {wave} Å  |  {utc[11:] + ' UTC' if utc else 'no data'}", fontsize=12
        )
        handles = [
            Line2D(
                [],
                [],
                color=colors[j],
                lw=2,
                label=f"M {matched_frequencies[j]:.2f} / D {f:g} MHz",
            )
            for j, f in enumerate(frequencies)
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
    cso_meta = None
    if config.get("cso_files"):
        from solar_toolkit.radio.cso_window import read_cso_total_window

        shared_freq = config.get("spectrum_frequency_bounds_mhz", [90.0, 300.0])
        stime, sfreq, sdata, cso_meta = read_cso_total_window(
            config["cso_files"], bounds, shared_freq
        )
        if not config.get("relative_background_db"):
            with np.errstate(divide="ignore", invalid="ignore"):
                sdata = np.where(sdata > 0, np.log10(sdata), np.nan)
        stitle, slabel = (
            "Chashan (CSO) total intensity R+L",
            "log10(SFU; source header)",
        )
    else:
        dart_files = discover_dart_spectrogram_files(config["spectrum_directory"])
        dart_freq = np.asarray(fits.getdata(dart_files.frequency)).ravel()
        shared_freq = (
            max(float(spec.frequency_mhz[0]), float(dart_freq.min())),
            min(float(spec.frequency_mhz[-1]), float(dart_freq.max())),
        )
        dw = read_dart_spectrogram_window(
            dart_files,
            time_range_utc=trange,
            frequency_range_mhz=shared_freq,
            max_time_samples=1000000,
            max_frequency_samples=1000000,
        )
        stime, sfreq, sdata = (
            [t.timestamp() for t in dw.time_utc],
            dw.frequency_mhz,
            dw.stokes_i_db,
        )
        stitle, slabel = "DART Stokes I", "Stokes I [dB]"
    low, high = shared_freq
    fi = (spec.frequency_mhz >= low) & (spec.frequency_mhz <= high)
    ti = (spec.time.unix >= bounds[0]) & (spec.time.unix <= bounds[1])
    data = spec.data[np.ix_(fi, ti)]
    original_bounds = list(bounds)
    mtime = spec.time.unix[ti]
    removed_intervals = []
    time_formatter = None
    if config.get("remove_missing_times"):
        missing = ~np.any(np.isfinite(sdata) & (sdata > 0), axis=0)
        cadence = float(cso_meta["time_bin_seconds"])
        runs = np.split(
            np.flatnonzero(missing),
            np.flatnonzero(np.diff(np.flatnonzero(missing)) > 1) + 1,
        )
        removed_intervals = [
            [
                max(bounds[0], float(stime[a[0]] - cadence / 2)),
                min(bounds[1], float(stime[a[-1]] + cadence / 2)),
            ]
            for a in runs
            if len(a)
        ]

        def compress(t):
            out = np.asarray(t, dtype=float).copy()
            for start, end in removed_intervals:
                out -= np.clip(np.asarray(t) - start, 0, end - start)
            return out

        def restore(t):
            out = t
            elapsed = 0.0
            for start, end in removed_intervals:
                if t >= start - elapsed - 1e-6:
                    out += end - start
                elapsed += end - start
            return out

        keep = np.ones(len(mtime), dtype=bool)
        for start, end in removed_intervals:
            keep &= ~((mtime >= start) & (mtime < end))
        data = data[:, keep]
        mtime = compress(mtime[keep])
        stime = compress(np.asarray(stime)[~missing])
        sdata = sdata[:, ~missing]
        bounds = compress(bounds).tolist()
        target = float(compress(target))
        time_formatter = FuncFormatter(
            lambda x, pos: Time(
                restore(Time(mdates.num2date(x)).unix), format="unix"
            ).strftime("%H:%M:%S")
        )
    display_meta = None
    mlabel = "10 log10(power) [arbitrary reference]"
    mbar = "Power [dB, arb.]"
    mlimits = slimits = None
    if config.get("relative_background_db"):
        if not config.get("cso_files"):
            raise ValueError("Relative mode requires linear CSO source inputs")
        from solar_toolkit.radio.spectrum_display import relative_background_db

        data, mmode = relative_background_db(data, mtime)
        sdata, smode = relative_background_db(
            sdata, stime, config.get("display_gap_seconds", 0.8)
        )
        mlabel = mbar = slabel = "Enhancement above background [dB]"
        mlimits = [0.0, float(np.nanpercentile(data, 99.5))]
        slimits = [0.0, float(np.nanpercentile(sdata, 99.5))]
        display_meta = {
            "MUSER": mmode,
            "CSO": smode,
            "color_limit_rule": "0 to positive-data 99.5th percentile, independently for each panel",
            "absolute_flux_equivalence": False,
        }
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.where(data > 0, 10 * np.log10(data), np.nan)
    ax = fig.add_subplot(gs[1, :2])
    art, mm = _spectrum(
        ax,
        mtime,
        spec.frequency_mhz[fi],
        data,
        bounds,
        target,
        "MUSER-L total-power I",
        mlabel,
        mlimits,
    )
    ax.set_ylim(shared_freq)
    if time_formatter is not None:
        ax.xaxis.set_major_formatter(time_formatter)
    fig.colorbar(art, cax=fig.add_subplot(gs[1, 2]), label=mbar)
    ax = fig.add_subplot(gs[2, :2])
    art, dm = _spectrum(
        ax, stime, sfreq, sdata, bounds, target, stitle, slabel, slimits
    )
    ax.set_ylim(shared_freq)
    if time_formatter is not None:
        ax.xaxis.set_major_formatter(time_formatter)
    ax.set_xlabel(
        "UTC (missing intervals removed; time axis compressed)"
        if removed_intervals
        else "Time [UTC]"
    )
    fig.colorbar(art, cax=fig.add_subplot(gs[2, 2]), label=slabel)
    subtitle = (
        "Raw 90% peak contours · total intensity · independent spectrum color scales"
    )
    if display_meta:
        subtitle += (
            "\nSpectra: relative background [dB]; missing intervals removed from BOTH spectra; no interpolation"
            if config.get("remove_missing_times")
            else "\nSpectra: relative background [dB]"
        )
    if outside_view:
        subtitle += "\nContours outside selected view: " + ", ".join(outside_view)
    elif distant:
        description = (
            "Distant contours outside selected view"
            if requested_fov is not None
            else "Distant contours retained (>2 solar radii)"
        )
        subtitle += "\n" + description + ": " + ", ".join(distant)
    fig.suptitle(
        f"{config['label']}  |  {plan.rows[preview_row].time.utc} UTC\n" + subtitle,
        fontsize=13,
    )
    png = output / "preview.png"
    fig.savefig(png, dpi=180)
    if config.get("make_video"):
        from .muser_video import write_video

        write_video(
            fig,
            config,
            output,
            mi,
            dart,
            aia,
            original_bounds,
            removed_intervals,
            colors,
            plan=plan,
        )
    plt.close(fig)
    # Full hashes of selected imaging files; huge DART spectral originals have stat provenance.
    spectral_inputs = [
        {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(Path(config["spectrum_directory"]).glob("*.fits"))
    ]
    meta = {
        "config": config,
        "reference_utc": plan.rows[preview_row].time.utc + "Z",
        "time_bounds_utc": Time(original_bounds, format="unix").isot.tolist(),
        "removed_time_intervals_utc": [
            Time(x, format="unix").isot.tolist() for x in removed_intervals
        ],
        "removed_seconds": sum(b - a for a, b in removed_intervals),
        "time_axis_compressed": bool(removed_intervals),
        "fov_arcsec": limits,
        "frequency_bounds_mhz": [float(low), float(high)],
        "contour_fraction": 0.9,
        "selection": {
            "bandwidth_mhz": 2,
            "frequencies_mhz": frequencies,
            "score_mean_source_db": float(score[score_index])
            if score_index is not None
            else None,
            "radio_tolerance_s": 0.1,
            "aia_tolerance_s": 12,
        },
        "overlays": overlay_meta,
        "aia": aia_meta,
        "spectra": [mm, dm],
        "coordinate_policy": "Observed angular HPC WCS; no shifts or radius rescaling",
        "dart_polarization_policy": "Existing toolkit parent LL/RR labels override generic StokesI header; RR+LL",
        "inputs": [{"path": str(p), "sha256": _digest(p)} for p in sorted(inputs)],
        "spectrum_display": display_meta,
        "source_units": {"MUSER": spec.unit, "CSO": "SolarFluxUnit(SFU)"},
        "cso_spectrum": cso_meta,
        "selection_score_source": "Immutable image-plan row nearest the highest finite in-window DART original Stokes I dB sample; middle plan row when no score exists",
        "dart_spectrum_inputs": spectral_inputs,
        "preview_sha256": _digest(png),
        "time_match_plan": plan.to_dict(),
        "preview_row_index": preview_row,
        "time_match": preview_match,
        "image_window_utc": plan.to_dict()["window"],
        "quality_notes": {
            "distant_contours_retained": distant,
            "contours_outside_view": outside_view,
        },
        "implementation_sha256": {
            str(p.name): _digest(p)
            for p in [
                Path(__file__),
                Path(__import__("solar_toolkit.radio.muser", fromlist=["x"]).__file__),
                Path(
                    __import__(
                        "solar_toolkit.radio.muser_comparison", fromlist=["x"]
                    ).__file__
                ),
                Path(
                    __import__(
                        "solar_toolkit.radio.cso_window", fromlist=["x"]
                    ).__file__
                ),
                Path(
                    __import__(
                        "solar_toolkit.radio.spectrum_display", fromlist=["x"]
                    ).__file__
                ),
            ]
        },
    }
    (output / "preview.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(png, flush=True)
    return png


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_preview(json.loads(args.config.read_text()), args.output)
    return 0


def _pair_dart_in_window(directory, frequency, window):
    from solar_toolkit.radio.time_matching import index_radio_frames, pair_radio_frames

    rr = index_radio_frames(
        (directory / "RR").glob("*.fits"), frequency_mhz=frequency, polarization="RR"
    )
    ll = index_radio_frames(
        (directory / "LL").glob("*.fits"), frequency_mhz=frequency, polarization="LL"
    )
    paired = pair_radio_frames(rr, ll, tolerance_seconds=0.1, **window)
    return [
        FrameRef(
            float(Time(frame.time.utc, scale="utc").unix),
            tuple(Path(path) for path in frame.payload["paths"]),
            stamp=frame.time,
            payload=dict(frame.payload, frame_id=frame.frame_id),
        )
        for frame in paired
    ]


def _select_plan_preview(plan, spectrum_times, score, source_ids):
    """Select a representative immutable plan row; never rematch its sources."""
    from solar_toolkit.time import normalize_time

    if not plan.rows:
        raise ValueError("No valid image observations in the selected UTC interval")
    samples = [
        (index, normalize_time(float(value), format="unix"))
        for index, value in enumerate(spectrum_times)
        if np.isfinite(value) and np.isfinite(score[index])
    ]
    window = plan.to_dict()["window"]
    start = normalize_time(window["start_utc"]).tai_ns if window["start_utc"] else None
    end = normalize_time(window["end_utc"]).tai_ns if window["end_utc"] else None
    samples = [
        (index, stamp)
        for index, stamp in samples
        if (start is None or stamp.tai_ns >= start)
        and (end is None or stamp.tai_ns < end)
    ]
    score_index, peak = (
        min(samples, key=lambda item: (-score[item[0]], item[0]))
        if samples
        else (None, None)
    )
    row_index = (
        min(
            range(len(plan.rows)),
            key=lambda index: (abs(plan.rows[index].time.tai_ns - peak.tai_ns), index),
        )
        if peak is not None
        else len(plan.rows) // 2
    )
    row = plan.rows[row_index]
    refs = [
        row.selections[source].frame.payload if row.selections[source].frame else None
        for source in source_ids
    ]
    return float(Time(row.time.utc, scale="utc").unix), refs, score_index, row_index


if __name__ == "__main__":
    raise SystemExit(main())
