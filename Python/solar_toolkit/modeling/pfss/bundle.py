"""Portable arrays plus JSON, with a completion manifest written last."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_member(root, value):
    member = Path(value)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Unsafe artifact path in manifest")
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()) or (root / member).is_symlink():
        raise ValueError("Artifact escapes run directory")
    return target


def save_bundle(directory, records, projections, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("Completed run directories are immutable")
    arrays, lines = {}, []
    for index, (record, projected) in enumerate(zip(records, projections, strict=True)):
        arrays[f"xyz_{index}"] = np.asarray(record["xyz_carrington_rsun"], float)
        arrays[f"hpc_{index}"] = np.asarray(projected["hpc_arcsec"], float)
        lines.append(
            {
                k: v
                for k, v in record.items()
                if k not in {"xyz_carrington_rsun", "hpc_arcsec"}
            }
        )
    np.savez_compressed(directory / "fieldlines.npz", **arrays)
    payload = {
        "schema_version": 1,
        "frame": "heliographic_carrington",
        "length_unit": "R_sun",
        "angle_unit": "arcsec",
        "time_assumption": "frozen Carrington pattern",
        "lines": lines,
        **metadata,
    }
    (directory / "metadata.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    complete_manifest(directory)


def complete_manifest(directory):
    directory = Path(directory)
    files = [
        {
            "path": p.relative_to(directory).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": file_hash(p),
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "manifest.json.tmp"}
    ]
    manifest = {"schema_version": 1, "status": "complete", "files": files}
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(directory / "manifest.json")


def load_bundle(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete PFSS bundle")
    names = set()
    for item in manifest["files"]:
        path = _safe_member(root, item["path"])
        if (
            item["path"] in names
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_hash(path) != item["sha256"]
        ):
            raise ValueError("Incomplete synchronization or artifact checksum mismatch")
        names.add(item["path"])
    if not {"metadata.json", "fieldlines.npz"}.issubset(names):
        raise ValueError("Required PFSS bundle artifacts missing")
    meta = json.loads((root / "metadata.json").read_text())
    if (
        meta.get("schema_version") != 1
        or meta.get("length_unit") != "R_sun"
        or meta.get("frame") != "heliographic_carrington"
    ):
        raise ValueError("Unsupported PFSS coordinate metadata")
    records = []
    with np.load(root / "fieldlines.npz", allow_pickle=False) as arrays:
        for i, line in enumerate(meta["lines"]):
            xyz, xy = arrays[f"xyz_{i}"], arrays[f"hpc_{i}"]
            if xyz.ndim != 2 or xyz.shape[1:] != (3,) or xy.shape != (len(xyz), 2):
                raise ValueError("Malformed fieldline arrays")
            if line.get("trace_valid") and (len(xyz) < 2 or not np.isfinite(xyz).all()):
                raise ValueError("Valid fieldline contains missing coordinates")
            records.append({**line, "xyz_carrington_rsun": xyz, "hpc_arcsec": xy})
    meta["_verified_files"] = sorted(names)
    return meta, records


__all__ = ["file_hash", "save_bundle", "complete_manifest", "load_bundle"]
