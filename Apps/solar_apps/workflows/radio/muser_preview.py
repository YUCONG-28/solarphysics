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
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
    discover_dart_spectrogram_files,
)
from solar_toolkit.radio.muser import (
    read_muser_spectrum,
    read_muser_images,
    peak_level,
    hpc_grid,
)
from solar_toolkit.radio.muser_comparison import (
    FrameRef,
    index_images,
    pair_dart,
    load_dart_sum,
    select_peak_frame,
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requested_fov = _requested_fov(config)
    frequencies = _requested_frequencies(config)
    colors = tuple(COLORS[i % len(COLORS)] for i in range(len(frequencies)))
    config = dict(
        config,
        frequencies_mhz=frequencies,
        label=config.get("label", "Radio/AIA comparison"),
    )
    if config.get("remove_missing_times") and not config.get("cso_files"):
        raise ValueError("Missing-time compaction requires CSO source inputs")
    mdir = Path(config["muser_directory"])
    specpaths = list(mdir.glob("*spec.PWR_I*.fits"))
    if len(specpaths) != 1:
        raise ValueError("Select one MUSER input directory with one PWR_I spectrum")
    spec = read_muser_spectrum(specpaths[0])
    bounds = [float(spec.time.unix[0]), float(spec.time.unix[-1])]
    print("Indexing MUSER and DART FITS headers", flush=True)
    mi = index_images((mdir / "img_fits").glob("*.fits"))
    if not mi:
        raise ValueError("No MUSER image frames")
    dart = []
    for freq in frequencies:
        d = Path(config["dart_directory"]) / f"{freq:g}MHz"
        pairs = pair_dart(
            index_images((d / "LL").glob("*.fits")),
            index_images((d / "RR").glob("*.fits")),
        )
        if not pairs:
            raise ValueError(f"No DART LL/RR pairs for {freq:g} MHz")
        dart.append(pairs)
    for group in [mi, *dart]:
        bounds[0] = max(bounds[0], group[0].seconds)
        bounds[1] = min(bounds[1], group[-1].seconds)
    if bounds[0] >= bounds[1]:
        raise ValueError("No shared time coverage")
    aia = []
    for wave in (171, 304):
        table = scan_aia_folder(
            Path(config["aia_directory"]) / str(wave), recursive=True
        )
        table = table.loc[np.isclose(table.wavelength.astype(float), wave)]
        aia.append(
            [
                FrameRef(float(Time(t.to_pydatetime()).unix), (Path(p),))
                for p, t in zip(table.path, table.obs_time)
            ]
        )
    trange = tuple(Time(bounds, format="unix").isot)
    curves = extract_dart_narrowband_lightcurves(
        config["spectrum_directory"],
        frequencies,
        bandwidth_mhz=2.0,
        time_range_utc=trange,
    )
    st = np.array([t.timestamp() for t in curves.time_utc])
    score = np.mean([c.stokes_i_db for c in curves.curves], axis=0)
    bounds = [max(bounds[0], st[0]), min(bounds[1], st[-1])]
    target, refs, score_index = select_peak_frame(st, score, mi, dart, aia, bounds)
    print("Selected UTC", Time(target, format="unix").isot, flush=True)
    planes = read_muser_images(refs[0].paths[0])
    overlays, overlay_meta = [], []
    inputs = {specpaths[0], refs[0].paths[0]}
    for k, freq in enumerate(frequencies):
        mf, image, header, beam = min(planes, key=lambda p: abs(p[0] - freq))
        if abs(mf - freq) > 5:
            raise ValueError(f"No MUSER channel within 5 MHz of {freq}")
        ds = load_dart_sum(refs[k + 1])
        for instrument, data, hdr, f, style in [
            ("MUSER", image, header, mf, "-"),
            ("DART", ds.image, ds.header, freq, "--"),
        ]:
            x, y = hpc_grid(hdr, data.shape)
            level = peak_level(data)
            overlays.append((x, y, data, level, colors[k], style))
            overlay_meta.append(
                {
                    "instrument": instrument,
                    "frequency_mhz": f,
                    "peak": level / 0.9,
                    "level": level,
                    "unit": hdr.get("BUNIT"),
                    "beam": beam if instrument == "MUSER" else None,
                    "delta_seconds": refs[0 if instrument == "MUSER" else k + 1].seconds
                    - target,
                }
            )
        inputs.update(refs[k + 1].paths)
    maps = []
    aia_meta = []
    for wave, ref in zip((171, 304), refs[-2:]):
        m = sunpy.map.Map(ref.paths[0])
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
                "utc": Time(ref.seconds, format="unix").isot,
                "delta_seconds": ref.seconds - target,
            }
        )
    # Determine a common FOV from actual interpolated contour vertices and solar limb.
    radius = max(float(m.rsun_obs.to_value(u.arcsec)) for m in maps)
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
                    f'{overlay_meta[index]["instrument"]} {overlay_meta[index]["frequency_mhz"]:.2f} MHz'
                )
        for vertices in cs.allsegs[0]:
            if len(vertices):
                if np.max(np.hypot(vertices[:, 0], vertices[:, 1])) > 2 * radius:
                    name = f'{overlay_meta[index]["instrument"]} {overlay_meta[index]["frequency_mhz"]:.2f} MHz'
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
        ax.set_title(f'AIA {wave} Å  |  {aia_meta[k]["utc"][11:]} UTC', fontsize=12)
        handles = [
            Line2D(
                [],
                [],
                color=colors[j],
                lw=2,
                label=f'M {overlay_meta[2*j]["frequency_mhz"]:.2f} / D {f:g} MHz',
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
        missing_cso_columns = ~np.any(np.isfinite(sdata), axis=0)
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
    original_target = target
    mtime = spec.time.unix[ti]
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
    removed_intervals = []
    time_formatter = None
    if config.get("remove_missing_times"):
        missing = missing_cso_columns
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
        f'{config["label"]}  |  {Time(original_target, format="unix").isot} UTC\n'
        + subtitle,
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
        )
    plt.close(fig)
    # Full hashes of selected imaging files; huge DART spectral originals have stat provenance.
    spectral_inputs = [
        {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(Path(config["spectrum_directory"]).glob("*.fits"))
    ]
    meta = {
        "config": config,
        "reference_utc": Time(original_target, format="unix").isot,
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
            "score_mean_source_db": float(score[score_index]),
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
        "source_units": {
            "MUSER": spec.unit,
            "CSO": cso_meta["unit"] if cso_meta is not None else None,
        },
        "cso_spectrum": cso_meta,
        "selection_score_source": "DART original Stokes I dB; unchanged selection rule",
        "dart_spectrum_inputs": spectral_inputs,
        "preview_sha256": _digest(png),
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


if __name__ == "__main__":
    raise SystemExit(main())
