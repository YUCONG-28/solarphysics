"""Compose four- or six-panel original-image comparisons with CSO diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import AsinhNorm, LogNorm
import numpy as np
from astropy.time import Time
from astropy.wcs import WCS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_intervals(edges, valid):
    """Return exact empty-bin intervals, without altering the physical clock."""
    valid = np.asarray(valid, dtype=bool)
    if len(edges) != len(valid) + 1:
        raise ValueError("One more edge than validity bins is required")
    changes = np.diff(np.r_[False, ~valid, False].astype(int))
    return [
        (float(edges[a]), float(edges[b]))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    ]


def display_time_selection(edges, valid, remove_internal=False):
    """Remove only wholly empty bins bounded by observations; retain endpoints."""
    valid = np.asarray(valid, dtype=bool)
    intervals = missing_intervals(edges, valid)
    keep = np.ones(len(valid), dtype=bool)
    observed = np.flatnonzero(valid)
    removed = []
    if remove_internal and observed.size:
        first, last = observed[[0, -1]]
        keep[first : last + 1] = valid[first : last + 1]
        removed = [
            (a, b) for a, b in intervals if a >= edges[first] and b <= edges[last + 1]
        ]
    return keep, removed


def display_bin_slice(edges, bounds):
    """Select whole native display bins intersecting a requested time window."""
    edges, bounds = np.asarray(edges, dtype=float), np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[1] <= bounds[0]
        or bounds[0] < edges[0]
        or bounds[1] > edges[-1]
    ):
        raise ValueError(
            "Display window must be increasing and inside the extraction window"
        )
    first = max(0, int(np.searchsorted(edges, bounds[0], side="right")) - 1)
    stop = min(len(edges) - 1, int(np.searchsorted(edges, bounds[1], side="left")))
    return slice(first, stop)


def compressed_time(seconds, removed):
    """Map actual UTC seconds onto a display axis without inserting samples."""
    seconds = np.asarray(seconds, dtype=float)
    result = seconds.copy()
    for start, end in removed:
        result -= np.clip(seconds - start, 0.0, end - start)
    return result


def actual_time(display_seconds, removed):
    """Invert retained display times; a deleted boundary maps to its right edge."""
    display_seconds = np.asarray(display_seconds, dtype=float)
    result = display_seconds.copy()
    elapsed = 0.0
    for start, end in removed:
        duration = end - start
        result += np.where(display_seconds >= start - elapsed, duration, 0.0)
        elapsed += duration
    return result


def mark_missing(ax, intervals, window_seconds):
    for start, end in intervals:
        a, b = mdates.date2num(Time([start, end], format="unix").to_datetime())
        ax.axvspan(a, b, color="#eeeeee", zorder=0)
        if end - start >= window_seconds * 0.035:
            ax.text(
                (a + b) / 2,
                0.52,
                "NO DATA",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )


def draw_original(ax, cax, prepared, title, fov, wavelength=None):
    """Display sampled original pixels at their native WCS; never shift sources."""
    metadata = dict(prepared.metadata)
    ax.set(
        xlim=fov[:2],
        ylim=fov[2:],
        aspect="equal",
        xlabel="Solar X [arcsec]",
        ylabel="Solar Y [arcsec]",
    )
    ax.set_xticks(np.linspace(fov[0], fov[1], 5))
    ax.set_yticks(np.linspace(fov[2], fov[3], 5))
    ax.tick_params(labelsize=9)
    if prepared.data is None:
        ax.set_facecolor("#eeeeee")
        ax.text(
            0.5,
            0.55,
            "NO DATA",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=19,
            color="#555555",
        )
        ax.text(
            0.5,
            0.43,
            metadata.get("reason", "Unavailable"),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            wrap=True,
        )
        ax.set_title(title, fontsize=12)
        cax.set_visible(False)
        return metadata
    data = np.asarray(prepared.data)
    # Decimation selects existing AIA pixels for a compact preview; radio maps
    # are normally below this cap and are displayed pixel-for-pixel.
    stride = max(1, int(np.ceil(max(data.shape) / 1024)))
    yy, xx = np.meshgrid(
        np.arange(0, data.shape[0], stride),
        np.arange(0, data.shape[1], stride),
        indexing="ij",
    )
    lon, lat = WCS(prepared.header, naxis=2).all_pix2world(xx, yy, 0)
    x, y = ((lon + 180) % 360 - 180) * 3600, lat * 3600
    view = data[::stride, ::stride].astype(float)
    mask = (
        (x >= fov[0])
        & (x <= fov[1])
        & (y >= fov[2])
        & (y <= fov[3])
        & np.isfinite(view)
    )
    if wavelength is not None:
        mask &= view > 0
    finite = view[mask]
    if not finite.size:
        ax.text(0.5, 0.5, "NO DATA IN VIEW", ha="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        cax.set_visible(False)
        metadata.update(
            status="no_data", reason="No finite pixels inside requested field"
        )
        return metadata
    percentiles = [1, 99.7]
    lo, hi = map(float, np.percentile(finite, percentiles))
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.01, 1e-12)
    if wavelength is None:
        norm = AsinhNorm(linear_width=max((hi - lo) / 50, 1e-12), vmin=lo, vmax=hi)
        cmap, stretch = "inferno", "asinh"
    else:
        from sunpy.visualization.colormaps import cmlist

        norm = LogNorm(vmin=max(lo, np.finfo(float).tiny), vmax=hi)
        cmap, stretch = cmlist[f"sdoaia{wavelength}"].copy(), "log"
        cmap.set_bad("black")
    ax.set_facecolor("#dddddd")
    artist = ax.pcolormesh(
        x,
        y,
        np.ma.masked_invalid(view),
        shading="nearest",
        norm=norm,
        cmap=cmap,
        rasterized=True,
    )
    colorbar = ax.figure.colorbar(artist, cax=cax)
    colorbar.set_label(
        metadata.get("unit") or prepared.header.get("BUNIT") or "source units",
        fontsize=9,
    )
    colorbar.ax.tick_params(labelsize=8)
    if wavelength is None:
        ticks = norm.inverse(np.linspace(0, 1, 5))
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{v:.2g}" for v in ticks])
    utc = metadata.get("utc", "")
    delta = metadata.get("delta_seconds", 0.0)
    ax.set_title(f"{title}\n{utc[11:]} UTC  |  Δt {delta:+.3f} s", fontsize=11)
    metadata["display"] = {
        "stretch": stretch,
        "percentiles": percentiles,
        "vmin": lo,
        "vmax": hi,
        "native_pixel_stride": stride,
        "fov_arcsec": fov,
        "spatial_interpolation": False,
    }
    return metadata


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def radio_bands(config):
    """Resolve explicit positive frequencies without supplying an event preset."""
    if "radio_bands" in config:
        bands = config["radio_bands"]
    else:
        frequency = config["frequency_mhz"]
        bands = [
            {
                "dart_mhz": frequency,
                "muser_mhz": frequency,
                "dart_directory": config["dart_directory"],
            }
        ]
    if not isinstance(bands, list) or len(bands) not in (1, 2):
        raise ValueError("Expected one or two explicit radio frequency pairs")
    for band in bands:
        for key in ("dart_mhz", "muser_mhz"):
            value = float(band[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Radio frequencies must be finite and positive")
        if not band["dart_directory"]:
            raise ValueError("Every band requires a DART input directory")
    return bands


def build_preview(config, output, *, frame_renderer=None):
    from solar_toolkit.radio.raw_comparison import (
        prepare_muser,
        prepare_dart,
        prepare_aia,
    )
    from solar_toolkit.radio import cso_comparison

    bands = radio_bands(config)
    target = float(Time(config["reference_utc"], scale="utc").unix)
    half_window = float(config.get("window_seconds", 300.0))
    bounds = [target - half_window, target + half_window]
    fov = list(config.get("fov_arcsec", [-1200, 1200, -1200, 1200]))
    if (
        len(fov) != 4
        or not np.all(np.isfinite(fov))
        or fov[0] >= fov[1]
        or fov[2] >= fov[3]
    ):
        raise ValueError("Expected increasing finite four-element FOV")
    half_band = float(config.get("half_bandwidth_mhz", 1.0))
    if (
        not np.isfinite([half_window, half_band]).all()
        or half_window <= 0
        or half_band <= 0
    ):
        raise ValueError("Window and bandwidth must be finite and positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    print("Matching original image observation times", flush=True)
    aia = [
        prepare_aia(
            Path(config["aia_directory"]) / str(w),
            target,
            wavelength=w,
            tolerance=config.get("aia_tolerance_seconds", 12.0),
        )
        for w in (171, 304)
    ]
    frequencies = [float(b["dart_mhz"]) for b in bands]
    musers = [
        prepare_muser(
            config["muser_directory"],
            target,
            frequency_mhz=float(b["muser_mhz"]),
            tolerance=config.get("radio_tolerance_seconds", 0.25),
        )
        for b in bands
    ]
    darts = [
        prepare_dart(
            b["dart_directory"],
            target,
            mode=config["dart_mode"],
            frequency_mhz=float(b["dart_mhz"]),
            tolerance=config.get("radio_tolerance_seconds", 0.25),
        )
        for b in bands
    ]
    print(
        "Extracting full-frequency CSO spectrum and original SFU band means", flush=True
    )
    cso = cso_comparison.read_cso_comparison(
        config["cso_files"], bounds, bands_mhz=frequencies, half_width_mhz=half_band
    )
    normalized = cso_comparison.normalize_spectrum_db(cso.spectrum_sfu)
    values, normalization = normalized
    requested_display_bounds = (
        Time(config["display_window_utc"], scale="utc").unix
        if "display_window_utc" in config
        else bounds
    )
    window_slice = display_bin_slice(cso.time_edges, requested_display_bounds)
    normalization["background_window_utc"] = Time(bounds, format="unix").isot.tolist()
    dual = len(bands) == 2
    fig = plt.figure(figsize=(20, 16) if dual else (14, 18), layout="constrained")
    columns = 3 if dual else 2
    grid = fig.add_gridspec(
        4,
        2 * columns,
        width_ratios=[1, 0.035] * columns,
        height_ratios=[1, 1, 0.65, 0.43],
        wspace=0.10,
        hspace=0.10,
    )
    m_titles = [
        f"MUSER {m.metadata.get('frequency_mhz', b['muser_mhz']):.7f} MHz | nominal I"
        for m, b in zip(musers, bands)
    ]
    d_titles = [f"DART {f:g} MHz | {config['dart_mode']}" for f in frequencies]
    if dual:
        images = [aia[0], musers[0], darts[0], aia[1], musers[1], darts[1]]
        titles = [
            "AIA 171 Å",
            m_titles[0],
            d_titles[0],
            "AIA 304 Å",
            m_titles[1],
            d_titles[1],
        ]
        wavelengths = [171, None, None, 304, None, None]
    else:
        images = [*aia, musers[0], darts[0]]
        titles = ["AIA 171 Å", "AIA 304 Å", m_titles[0], d_titles[0]]
        wavelengths = [171, 304, None, None]
    panels = []
    for n, prepared in enumerate(images):
        row, col = divmod(n, columns)
        panels.append(
            draw_original(
                fig.add_subplot(grid[row, 2 * col]),
                fig.add_subplot(grid[row, 2 * col + 1]),
                prepared,
                titles[n],
                fov,
                wavelength=wavelengths[n],
            )
        )
    spec_ax = fig.add_subplot(grid[2, :-1])
    spec_cax = fig.add_subplot(grid[2, -1])
    flux_ax = fig.add_subplot(grid[3, :-1], sharex=spec_ax)
    fig.add_subplot(grid[3, -1]).set_visible(False)
    # Normalize using the full extraction window, then crop the display only.
    # Keep the original CSO bin edges and values: no rebinning or interpolation.
    edges = cso.time_edges[window_slice.start : window_slice.stop + 1]
    centers = (edges[:-1] + edges[1:]) / 2
    valid_spec = np.any(cso.spectrum_counts[:, window_slice] > 0, axis=0)
    keep, removed = display_time_selection(
        edges, valid_spec, config.get("remove_internal_missing_times", False)
    )
    kept_edges = np.r_[edges[:-1][keep], edges[-1]]
    display_edges = compressed_time(kept_edges, removed)
    xedges = mdates.date2num(Time(display_edges, format="unix").to_datetime())
    xcenters = mdates.date2num(
        Time(compressed_time(centers[keep], removed), format="unix").to_datetime()
    )
    display_window = float(display_edges[-1] - display_edges[0])
    missing_spec = missing_intervals(edges, valid_spec)
    visible_missing_spec = [
        tuple(compressed_time(v, removed)) for v in missing_spec if v not in removed
    ]
    mark_missing(spec_ax, visible_missing_spec, display_window)
    displayed_values = values[:, window_slice][:, keep]
    if np.any(np.isfinite(displayed_values)):
        artist = spec_ax.pcolormesh(
            xedges,
            cso.frequency_edges,
            np.ma.masked_invalid(displayed_values),
            shading="flat",
            cmap="turbo",
            vmin=normalization["vmin"],
            vmax=normalization["vmax"],
            rasterized=True,
        )
        fig.colorbar(artist, cax=spec_cax, label="Above per-frequency P20 [dB]")
    else:
        spec_ax.text(0.5, 0.5, "NO DATA", transform=spec_ax.transAxes, ha="center")
        spec_cax.set_visible(False)
    colors = ["#0072b2", "#d55e00"] if dual else ["#c87900"]
    for frequency, line_color in zip(frequencies, colors):
        spec_ax.axhline(frequency, color=line_color, lw=0.9, ls="--", alpha=0.9)
        spec_ax.text(
            0.995,
            frequency,
            f" {frequency:g} MHz ",
            color="black",
            fontsize=9,
            transform=spec_ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1},
        )
    spec_ax.set(
        ylim=(
            [cso.frequency_edges[0], cso.frequency_edges[-1]]
            if len(cso.frequency_edges)
            else [0.0, 625.0]
        ),
        ylabel="Frequency [MHz]",
    )
    spec_ax.set_title(
        "Chashan (CSO) R+L | full frequency range | original-window P20 background",
        loc="left",
        fontsize=11,
    )
    missing_flux_by_band = {}
    for i, (frequency, line_color) in enumerate(zip(frequencies, colors)):
        flux_values = cso.lightcurves_sfu[i, window_slice]
        flux_counts = cso.lightcurve_counts[i, window_slice]
        missing_flux = missing_intervals(edges, flux_counts > 0)
        missing_flux_by_band[str(frequency)] = [
            Time(v, format="unix").isot.tolist() for v in missing_flux
        ]
        visible_missing_flux = [
            tuple(compressed_time(v, removed)) for v in missing_flux
        ]
        visible_missing_flux = [(a, b) for a, b in visible_missing_flux if b > a]
        mark_missing(flux_ax, visible_missing_flux, display_window)
        flux_ax.plot(
            xcenters,
            flux_values[keep],
            color=line_color,
            lw=0.9,
            label=f"{frequency:g} ± {half_band:g} MHz | native R+L mean",
        )
    flux_ax.set_title(
        "Chashan selected-band flux | no background subtraction",
        loc="left",
        fontsize=12,
    )
    flux_ax.set(
        ylabel="Flux density [SFU]",
        xlabel="Time [UTC; internal gaps removed]" if removed else "Time [UTC]",
    )
    flux_ax.legend(loc="upper right", fontsize=9)
    flux_ax.grid(axis="y", color="#dddddd", lw=0.5)
    target_x = mdates.date2num(
        Time(compressed_time(target, removed), format="unix").to_datetime()
    )
    for ax in [spec_ax, flux_ax]:
        ax.set_xlim(xedges[0], xedges[-1])
        cursor = ax.axvline(target_x, color="#222222", lw=1.0, ls="--")
        if ax is spec_ax:
            cursor.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.4, foreground="white"),
                    path_effects.Normal(),
                ]
            )
        tick_seconds = np.linspace(display_edges[0], display_edges[-1], 7)
        ax.set_xticks(mdates.date2num(Time(tick_seconds, format="unix").to_datetime()))
        ax.set_xticklabels(
            [
                dt.strftime("%H:%M:%S")
                for dt in Time(
                    actual_time(tick_seconds, removed), format="unix"
                ).to_datetime()
            ]
        )
        ax.tick_params(labelsize=9)
    spec_ax.tick_params(labelbottom=False)
    removed_seconds = sum(b - a for a, b in removed)
    clock_note = (
        f"Internal missing times removed ({removed_seconds:.1f} s); UTC axis compressed"
        if removed
        else f"Full ±{half_window/60:g} min UTC window"
    )
    if "display_window_utc" in config:
        clock_note = (
            f"Event interval · internal gaps removed ({removed_seconds:.1f} s)"
            if removed
            else "Event interval"
        )
    fov_note = (
        f"fixed ±{fov[1]:g} arcsec"
        if fov[0] == fov[2] == -fov[1] and fov[1] == fov[3]
        else f"X [{fov[0]:g}, {fov[1]:g}], Y [{fov[2]:g}, {fov[3]:g}] arcsec"
    )
    fig.suptitle(
        f"{config['event_id']} | Reference {config['reference_utc']} UTC\n"
        f"Original images · independent intensity scales · {fov_note}\n"
        f"{clock_note} · no time interpolation",
        fontsize=13,
    )
    png = output / "preview.png"
    print("Rendering preview", flush=True)
    fig.savefig(png, dpi=160, bbox_inches="tight", pad_inches=0.15)
    if frame_renderer is not None:
        frame_renderer(fig, config, output, panels, removed)
    plt.close(fig)
    csv_path = output / "flux.csv"
    csv_displayed = np.zeros(len(cso.time_centers), dtype=bool)
    csv_displayed[window_slice] = keep
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "utc",
                "frequency_mhz",
                "half_bandwidth_mhz",
                "flux_sfu",
                "valid_samples",
                "status",
                "displayed",
            ]
        )
        for i, frequency in enumerate(frequencies):
            for utc, flux, count, displayed in zip(
                Time(cso.time_centers, format="unix").isot,
                cso.lightcurves_sfu[i],
                cso.lightcurve_counts[i],
                csv_displayed,
            ):
                writer.writerow(
                    [
                        utc,
                        frequency,
                        half_band,
                        format(float(flux), ".12g") if np.isfinite(flux) else "",
                        int(count),
                        "ok" if count else "no_data",
                        int(displayed),
                    ]
                )
    inputs = {}
    for panel in panels:
        for path in panel.get("paths", []):
            if path not in inputs:
                p = Path(path)
                inputs[path] = {"sha256": digest(p), "size": p.stat().st_size}
    from solar_toolkit.radio import (
        raw_comparison,
        muser,
        muser_comparison,
        cso as cso_reader,
    )

    modules = [
        Path(__file__),
        Path(raw_comparison.__file__),
        Path(cso_comparison.__file__),
        Path(muser.__file__),
        Path(muser_comparison.__file__),
        Path(cso_reader.__file__),
    ]
    result = {
        "config": config,
        "reference_utc": Time(target, format="unix").isot,
        "window_utc": Time(bounds, format="unix").isot.tolist(),
        "panels": panels,
        "requested_display_window_utc": Time(
            requested_display_bounds, format="unix"
        ).isot.tolist(),
        "display_window_utc": Time(edges[[0, -1]], format="unix").isot.tolist(),
        "display_window_selection": "Whole existing 0.1 s bins intersecting requested interval",
        "cso": cso.metadata,
        "spectrum_display": normalization,
        "frequency_marker_mhz": frequencies if dual else frequencies[0],
        "lightcurve_band_mhz": (
            [[f - half_band, f + half_band] for f in frequencies]
            if dual
            else [frequencies[0] - half_band, frequencies[0] + half_band]
        ),
        "missing_flux_by_band_utc": missing_flux_by_band,
        "image_layout": "3 columns x 2 rows" if dual else "2 columns x 2 rows",
        "missing_spectrum_intervals_utc": [
            Time(v, format="unix").isot.tolist() for v in missing_spec
        ],
        "missing_flux_intervals_utc": [
            Time(v, format="unix").isot.tolist() for v in missing_flux
        ],
        "image_inputs": inputs,
        "implementation_sha256": {p.name: digest(p) for p in modules},
        "outputs": {"preview.png": digest(png), "flux.csv": digest(csv_path)},
        "time_interpolation": False,
        "time_compression": bool(removed),
        "removed_internal_intervals_utc": [
            Time(v, format="unix").isot.tolist() for v in removed
        ],
        "removed_internal_seconds": removed_seconds,
        "displayed_time_bins": int(keep.sum()),
        "window_time_bins": len(keep),
        "source_time_bins": len(cso.time_centers),
        "csv_time_axis": "Original UTC, including missing bins; displayed flag records plot selection",
        "muser_polarization": "Nominal I per product metadata; upstream synthesis not independently verified",
    }
    (output / "preview.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    print(png, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    build_preview(json.loads(Path(args.config).read_text()), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
