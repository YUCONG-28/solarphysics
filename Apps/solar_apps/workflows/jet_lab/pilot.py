"""Prepare native-pixel samples on the configured scientific compute host.

Run with the existing selected interpreter: ``-m ...pilot --config FILE``.
The private config specifies paths and compute host; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import socket

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import sunpy.map

from solar_toolkit.map.jet_annotations import (
    JetDocument,
    file_sha256,
    json_safe,
    observation_info,
    pixel_hpc,
    save_session,
)
from solar_toolkit.map.jet_registration import registered_difference
from solar_toolkit.map.jet_extraction import (
    SegmentationParameters,
    axis_metrics,
    component_at,
    gaussian_width,
    segment,
    skeleton_paths,
)


def dump(path, obj):
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False)
    )


def synthetic_images():
    """Fixed-seed engineering fixtures, with known 2-D axes where meaningful."""
    yy, xx = np.mgrid[:160, :200]
    rng = np.random.default_rng(1207)
    centre = 75 + 22 * np.sin((xx - 30) / 75)
    horizontal = np.exp(-0.5 * ((yy - 80) / 4) ** 2) * ((xx > 20) & (xx < 180))
    curved = np.exp(-0.5 * ((yy - centre) / 4) ** 2) * ((xx > 20) & (xx < 180))
    branch = np.exp(-0.5 * ((xx - 100) / 4) ** 2) * ((yy > 80) & (yy < 140))
    crossing = np.exp(-0.5 * ((xx - 100) / 4) ** 2) * ((yy > 20) & (yy < 140))
    missing = 5 + 100 * curved
    missing[:, 95:100] = np.nan
    double = np.exp(-0.5 * ((yy - 73) / 3) ** 2) + np.exp(-0.5 * ((yy - 87) / 3) ** 2)
    return {
        "straight": 5 + 100 * horizontal,
        "curved": 5 + 100 * curved,
        "branched": 5 + 100 * np.maximum(horizontal, branch),
        "crossing": 5 + 100 * np.maximum(horizontal, crossing),
        "low_snr": 5 + curved + rng.normal(0, 3, xx.shape),
        "missing": missing,
        "multi_peak": 5 + 100 * double * ((xx > 20) & (xx < 180)),
    }


def synthetic_qa(out):
    destination = out / "synthetic"
    destination.mkdir(exist_ok=True)
    images = synthetic_images()
    summary = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
    for ax, (name, data) in zip(axes.ravel(), images.items()):
        header = fits.Header(
            dict(
                CTYPE1="HPLN-TAN",
                CTYPE2="HPLT-TAN",
                CUNIT1="arcsec",
                CUNIT2="arcsec",
                CRPIX1=100,
                CRPIX2=80,
                CRVAL1=850,
                CRVAL2=-170,
                CDELT1=0.6,
                CDELT2=0.6,
                EXPTIME=2,
                DATE_OBS="2000-01-01T12:00:30",
                DSUN_OBS=149600000000.0,
                HGLN_OBS=0,
                HGLT_OBS=-5.0,
                RSUN_REF=695700000,
                TELESCOP="SDO/AIA",
                INSTRUME="AIA",
                WAVELNTH=171,
                WAVEUNIT="angstrom",
                BUNIT="DN",
                SYNTHET=True,
                JETTEST=True,
            )
        )
        path = destination / (name + ".fits")
        fits.writeto(path, 2 * data, header, overwrite=False)
        ax.imshow(data, origin="lower", cmap="gray", vmin=0, vmax=110)
        ax.set_title(name)
        ax.set_xlabel("native pixel x")
        ax.set_ylabel("native pixel y")
        for threshold, colour in zip((80, 85, 90), ("cyan", "orange", "magenta")):
            labels, cut = segment(data, SegmentationParameters(value=threshold))
            # A known synthetic start is a fixture, not an event identity inference.
            start = [25, 73.5 if name in {"curved", "missing"} else 80]
            try:
                selected = component_at(labels, start)
                skel, paths, info = skeleton_paths(selected, start)
                if not paths:
                    raise ValueError(info["reason"])
                path_xy = paths[0]
                ax.plot(*path_xy.T, color=colour, lw=1, label=f"p{threshold}")
                doc = JetDocument(path)
                result = axis_metrics(pixel_hpc(doc.map, path_xy))
                summary.append(
                    {
                        "case": name,
                        "percentile": threshold,
                        "candidate_count": len(paths),
                        "status": "pending",
                        **result,
                    }
                )
                if name == "curved" and threshold == 85:
                    doc.select(start)
                    doc.trace(start)
                    save_session(
                        out / "export_example",
                        [doc],
                        sample_id="synthetic_curved",
                        role="automatic_unconfirmed_example",
                    )
            except ValueError as exc:
                summary.append(
                    {"case": name, "percentile": threshold, "status": str(exc)}
                )
        if ax.lines:
            ax.legend(fontsize=8)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(
        0,
        0.8,
        "Synthetic engineering fixtures\nAxes are pending candidates.\nNo event identity or 3-D claims.\nMissing pixels remain missing.",
        va="top",
    )
    fig.savefig(out / "synthetic_extraction.png", dpi=160)
    plt.close(fig)
    dump(out / "synthetic_sensitivity.json", summary)
    x = np.arange(-20, 20.25, 0.25)
    profiles = {
        "single": 8 + 40 * np.exp(-0.5 * (x / 3) ** 2),
        "multiple": 8
        + 40 * np.exp(-0.5 * ((x - 6) / 2) ** 2)
        + 40 * np.exp(-0.5 * ((x + 6) / 2) ** 2),
        "truncated": 8 + 40 * np.exp(-0.5 * ((x - 18) / 3) ** 2),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    fits_report = {}
    for ax, (name, y) in zip(axes, profiles.items()):
        result = gaussian_width(x, y)
        fits_report[name] = result
        ax.plot(x, y)
        ax.set_title(name + "\n" + result["status"])
        ax.set_xlabel("normal offset / native pixel")
    fig.savefig(out / "width_validity.png", dpi=160)
    plt.close(fig)
    dump(out / "width_validity.json", fits_report)


def native_cutout(source, bounds, destination):
    smap = sunpy.map.Map(source)
    # Native submap preserves WCS, parity, pointing and sampling; no display resampling.
    bl = SkyCoord(
        bounds[0] * u.arcsec, bounds[2] * u.arcsec, frame=smap.coordinate_frame
    )
    tr = SkyCoord(
        bounds[1] * u.arcsec, bounds[3] * u.arcsec, frame=smap.coordinate_frame
    )
    crop = smap.submap(bl, top_right=tr)
    origin = smap.world_to_pixel(crop.pixel_to_world(0 * u.pix, 0 * u.pix))
    header = crop.fits_header.copy()
    header.remove("BLANK", ignore_missing=True)
    header["JXOFF"] = float(origin.x.value)
    header["JYOFF"] = float(origin.y.value)
    header["JPARHASH"] = file_sha256(source)
    from solar_toolkit.map.euvi_preprocessing import intensity_array, prepare_euvi_map

    if str(crop.meta.get("detector", "")).strip().upper() == "EUVI":
        result = prepare_euvi_map(crop)
        data, unit, action = result.data, result.unit, result.audit["exposure_action"]
        header["EUPYVER"] = result.audit["version"]
        header["EUPYSTAT"] = result.audit["status"]
        Path(str(destination) + ".preprocessing.json").write_text(
            json.dumps(dict(result.audit, source_sha256=file_sha256(source)), indent=2)
        )
    else:
        data, unit, action, _ = intensity_array(crop)
    header["JETNORM"] = action != "unchanged_requires_review"
    if unit:
        header["BUNIT"] = unit
    fits.writeto(destination, data, header)
    from .cutout_cache import identity, record

    sidecars = (
        [".preprocessing.json"]
        if str(crop.meta.get("detector", "")).strip().upper() == "EUVI"
        else []
    )
    record(destination, identity(smap, file_sha256(source), bounds), sidecars)
    return sunpy.map.Map(destination)


def prepare(config):
    sample_times = config.get("sample_times_utc")
    if not isinstance(sample_times, list) or not sample_times:
        raise ValueError("Supply a nonempty sample_times_utc list")
    targets = [(str(value), Time(value).unix) for value in sample_times]
    if (
        config.get("compute_host")
        and socket.gethostname().lower() != config["compute_host"].lower()
    ):
        raise ValueError(
            "Run this scientific preparation on the configured compute host"
        )
    out = Path(config["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "resolved_config.json", config)
    dump(
        out / "environment.json",
        {
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "sunpy": sunpy.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    synthetic_qa(out)
    inventory = list(csv.DictReader(open(config["inventory"], newline="")))
    # Recheck selected frame times below against real FITS; inventory locates candidates.
    pairs = []
    audits = []
    failures = []
    sample_dir = out / "samples"
    sample_dir.mkdir()
    bands = config.get("bands") or sorted({int(r["band"]) for r in inventory})
    for band in bands:
        euvi = [
            r for r in inventory if r["instrument"] == "EUVI" and int(r["band"]) == band
        ]
        aia = [
            r for r in inventory if r["instrument"] == "AIA" and int(r["band"]) == band
        ]
        if not euvi or not aia:
            failures.append({"band": band, "reason": "missing_inventory"})
            continue
        for target_utc, target in targets:
            e = min(euvi, key=lambda r: abs(float(r["unix"]) - target))
            a = min(
                aia, key=lambda r: abs(float(r["solar_unix"]) - float(e["solar_unix"]))
            )
            pair = {
                "id": f"{band}_{target_utc.replace(':', '').replace('-', '')}",
                "split": config.get("sample_roles", {}).get(target_utc, "sample"),
                "views": [],
                "pairing": "nearest_solar_emission_time_no_interpolation",
            }
            for camera, row, rows in (("AIA", a, aia), ("EUVI", e, euvi)):
                try:
                    source = Path(row["path"])
                    output = sample_dir / (pair["id"] + "_" + camera + ".fits")
                    smap = native_cutout(source, config["roi_arcsec"][camera], output)
                    info = observation_info(smap)
                    if abs(Time(info["midpoint_utc"]).unix - float(row["unix"])) > 0.01:
                        raise ValueError("inventory_midpoint_disagrees_with_FITS")
                    entry = {
                        "instrument": camera,
                        "image": output.relative_to(out).as_posix(),
                        "image_sha256": file_sha256(output),
                        "parent": str(source),
                        "info": info,
                        "difference": None,
                    }
                    previous = [
                        r for r in rows if float(r["unix"]) < float(row["unix"]) - 1
                    ]
                    if previous:
                        before = max(previous, key=lambda r: float(r["unix"]))
                        difference, audit = registered_difference(
                            smap,
                            Path(before["path"]),
                            entry["image_sha256"],
                            output.with_stem(output.stem + "_difference"),
                        )
                        audits.append(
                            {"pair": pair["id"], "instrument": camera, **audit}
                        )
                        if difference:
                            entry.update(
                                difference=difference.relative_to(out).as_posix(),
                                difference_sha256=file_sha256(difference),
                            )
                    pair["views"].append(entry)
                except (OSError, ValueError) as exc:
                    failures.append(
                        {"pair": pair["id"], "instrument": camera, "reason": str(exc)}
                    )
            if len(pair["views"]) == 2:
                pair["emission_time_difference_s"] = float(a["solar_unix"]) - float(
                    e["solar_unix"]
                )
                pairs.append(pair)
    manifest = {
        "schema": "solarphysics.jet_lab.samples",
        "version": 1,
        "pairs": pairs,
        "annotations": "not_yet_supplied",
        "registration_audit": "registration_audit.json",
    }
    dump(out / "registration_audit.json", audits)
    dump(out / "unresolved.json", failures)
    dump(out / "paired_samples.json", manifest)
    if pairs:
        fig, axes = plt.subplots(
            len(pairs),
            2,
            figsize=(10, 3.8 * len(pairs)),
            squeeze=False,
            layout="constrained",
        )
        for row, pair in zip(axes, pairs):
            for ax, view in zip(row, pair["views"]):
                smap = sunpy.map.Map(out / view["image"])
                data = np.asarray(smap.data, float)
                lo, hi = np.nanpercentile(data, [1, 99.5])
                display = np.arcsinh(np.clip((data - lo) / (hi - lo), 0, 1) * 10)
                ax.imshow(display, origin="lower", cmap="gray")
                ax.set_title(
                    f'{pair["id"]} / {pair["split"]}\n{view["instrument"]} {view["info"]["midpoint_utc"]}'
                )
                ax.set_xlabel("native pixel x")
                ax.set_ylabel("native pixel y")
        fig.savefig(out / "paired_samples_overview.png", dpi=140)
        plt.close(fig)
    checksums = {
        p.relative_to(out).as_posix(): file_sha256(p)
        for p in out.rglob("*")
        if p.is_file()
    }
    dump(
        out / "PREPARATION_COMPLETE.json",
        {"version": 1, "pair_count": len(pairs), "sha256": checksums},
    )
    print(
        json.dumps(
            {"output": str(out), "pair_count": len(pairs), "failures": len(failures)}
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    prepare(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
