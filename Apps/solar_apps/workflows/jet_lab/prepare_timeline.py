"""Prepare native, hashed timeline crops on the configured compute host."""

import argparse
import json
import socket
from pathlib import Path
import shutil

import pandas as pd
from astropy.time import Time
import sunpy.map

from .pilot import native_cutout
from solar_toolkit.map.jet_annotations import file_sha256, observation_info
from solar_toolkit.map.jet_registration import registered_difference


def prepare(config):
    if socket.gethostname().lower() != config["compute_host"].lower():
        raise ValueError("Run scientific preparation on the configured compute host")
    out = Path(config["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    inv = pd.read_csv(config["inventory"])
    existing = {}
    for p in Path(config["reuse_samples"]).glob("*.fits"):
        if "difference" in p.name:
            continue
        from .cutout_cache import verify

        try:
            checked = verify(p)
        except (OSError, ValueError, KeyError):
            continue
        existing[checked["identity"]["key"]] = p
    frames = []
    audit = []
    used = {Path(config["inventory"])}
    for instrument in ["AIA", "EUVI"]:
        for band in [171, 304]:
            prev = None
            rows = inv[(inv.instrument == instrument) & (inv.band == band)].sort_values(
                "unix"
            )
            for row in rows.itertuples():
                if (
                    not Time(config["start_utc"]).unix
                    <= row.unix
                    <= Time(config["end_utc"]).unix
                ):
                    continue
                src = Path(row.path)
                used.add(src)
                m = sunpy.map.Map(src)
                info = observation_info(m)
                if abs(Time(info["midpoint_utc"]).unix - row.unix) > 0.02:
                    raise ValueError("Inventory/header time mismatch")
                parent_hash = file_sha256(src)
                dest = out / "frames" / f"{instrument}_{band}_{parent_hash[:16]}.fits"
                from .cutout_cache import identity, verify

                expected = identity(m, parent_hash, config["roi_arcsec"][instrument])
                cached = existing.get(expected["key"])
                if cached:
                    checked = verify(cached, expected)
                    shutil.copy2(cached, dest)
                    for suffix in [".cutout.json", *checked["sidecars"]]:
                        shutil.copy2(
                            Path(str(cached) + suffix), Path(str(dest) + suffix)
                        )
                else:
                    native_cutout(src, config["roi_arcsec"][instrument], dest)
                crop = sunpy.map.Map(dest)
                cinfo = observation_info(crop)
                digest = file_sha256(dest)
                record = {
                    "instrument": instrument,
                    "band": band,
                    "image": dest.relative_to(out).as_posix(),
                    "image_sha256": digest,
                    "midpoint_utc": cinfo["midpoint_utc"],
                    "dsun_m": cinfo["dsun_m"],
                    "parent_sha256": parent_hash,
                }
                if prev:
                    delta = dest.with_name(dest.stem + "_difference.fits")
                    result, details = registered_difference(crop, prev, digest, delta)
                    audit.append({"image": record["image"], **details})
                    if result:
                        record.update(
                            difference=delta.relative_to(out).as_posix(),
                            difference_sha256=file_sha256(delta),
                        )
                frames.append(record)
                prev = src
    (out / "timeline.json").write_text(
        json.dumps(
            {"schema": "solarphysics.jet_lab.timeline", "version": 1, "frames": frames},
            ensure_ascii=False,
            indent=2,
        )
    )
    (out / "registration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    (out / "input_manifest.json").write_text(
        json.dumps(
            [{"path": str(p), "sha256": file_sha256(p)} for p in sorted(used)], indent=2
        )
    )
    (out / "resolved_config.json").write_text(json.dumps(config, indent=2))
    files = {
        p.relative_to(out).as_posix(): file_sha256(p)
        for p in out.rglob("*")
        if p.is_file()
    }
    (out / "COMPLETE.json").write_text(
        json.dumps({"version": 1, "sha256": files}, indent=2)
    )
    print(f"Prepared {len(frames)} native frames: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    prepare(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
