"""Versioned jet annotations with native-pixel WCS and reproducible exports."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.io import fits
from astropy.time import Time

from .jet_extraction import (
    SegmentationParameters,
    axis_metrics,
    component_at,
    cross_sections,
    retain_component,
    segment,
    skeleton_paths,
    validate_axis,
)

SCHEMA_VERSION = 1


__all__ = [
    "SCHEMA_VERSION",
    "file_sha256",
    "json_safe",
    "pixel_hpc",
    "observation_info",
    "intensity_per_second",
    "JetDocument",
    "save_session",
    "load_session",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def pixel_hpc(smap, xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not len(xy):
        return np.empty((0, 2))
    world = smap.pixel_to_world(xy[:, 0] * u.pix, xy[:, 1] * u.pix)
    if not hasattr(world, "Tx"):
        raise ValueError("Jet Lab v1 requires a helioprojective FITS WCS")
    return np.c_[world.Tx.to_value(u.arcsec), world.Ty.to_value(u.arcsec)]


def observation_info(smap):
    meta = smap.meta
    exposure = float(meta.get("exptime", 0))
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("FITS requires a positive EXPTIME")
    date = meta.get("date-obs") or meta.get("date_obs")
    if not date:
        raise ValueError("FITS observation time is missing")
    midpoint = (
        Time(str(date).rstrip("Z"), scale=str(meta.get("timesys") or "utc").lower())
        + exposure / 2 * u.s
    )
    observer = smap.observer_coordinate
    return {
        "instrument": str(smap.instrument),
        "wavelength": str(smap.wavelength),
        "observatory": str(smap.observatory),
        "detector": str(smap.detector),
        "observer_hgln_deg": float(observer.lon.to_value(u.deg)),
        "observer_hglt_deg": float(observer.lat.to_value(u.deg)),
        "exposure_s": exposure,
        "midpoint_utc": midpoint.utc.isot,
        "dsun_m": float(smap.dsun.to_value(u.m)),
        "parent_sha256": str(meta.get("jparhash", "")),
        "parent_pixel_offset_xy": [
            float(meta.get("jxoff", 0)),
            float(meta.get("jyoff", 0)),
        ],
        "unit": str(smap.unit or "unspecified"),
    }


def intensity_per_second(smap):
    """Do not divide rate-calibrated EUVI or prepared cutouts a second time."""
    data = np.array(smap.data, float, copy=True)
    unit = str(smap.meta.get("bunit", "")).lower().replace(" ", "")
    is_rate = (
        bool(smap.meta.get("jetnorm", False))
        or "/s" in unit
        or "s-1" in unit
        or "s**-1" in unit
    )
    if not is_rate:
        data /= observation_info(smap)["exposure_s"]
    data.setflags(write=False)
    return data


class JetDocument:
    """One independently selected observer/image; histories never cross views."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.map = sunpy.map.Map(self.path)
        self.info = observation_info(self.map)
        self.raw = intensity_per_second(self.map)
        self.sha256 = file_sha256(self.path)
        # Also asserts HPC support before enabling measurement.
        pixel_hpc(self.map, [[0, 0]])
        self.difference = None
        self.difference_path = None
        self.difference_sha256 = None
        self.state = {
            "parameters": asdict(SegmentationParameters()),
            "roi": None,
            "source": "intensity",
            "seed": None,
            "inner": None,
            "axis": [],
            "axis_status": "pending",
            "branch_index": 0,
            "original_skeleton": [],
            "automatic_axis": [],
            "control_points": [],
            "extension": [],
            "tiepoints": [],
        }
        self.labels = np.zeros(self.raw.shape, dtype=int)
        self.selected = None
        self.paths = []
        self.selection_reason = "select_component"
        self.threshold = None
        self.history = []
        self._undo = []
        self.dirty = False
        self.resegment()

    @property
    def data(self):
        if self.state["source"] == "difference":
            if self.difference is None:
                raise ValueError("No verified registered difference has been loaded")
            return self.difference
        return self.raw

    def load_difference(self, path):
        candidate = sunpy.map.Map(path)
        if not candidate.meta.get("jetreg", False):
            raise ValueError("Difference requires JETREG=T and a registration audit")
        if candidate.data.shape != self.raw.shape:
            raise ValueError("Registered difference shape does not match the original")
        height, width = self.raw.shape
        probes = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1],
                [width / 2, height / 2],
            ]
        )
        world = candidate.pixel_to_world(probes[:, 0] * u.pix, probes[:, 1] * u.pix)
        px, py = self.map.world_to_pixel(world)
        if np.max(np.abs(np.c_[px.value, py.value] - probes)) > 0.01:
            raise ValueError("Difference WCS differs by more than 0.01 native pixels")
        parent = candidate.meta.get("jorighsh", "")
        if parent != self.sha256:
            raise ValueError("Difference original-image hash does not match")
        self.difference = np.array(candidate.data, dtype=float)
        self.difference.setflags(write=False)
        self.difference_path = Path(path).resolve()
        self.difference_sha256 = file_sha256(path)

    def checkpoint(self, action):
        self._undo.append(deepcopy(self.state))
        self._undo = self._undo[-50:]
        self.history.append(
            {
                "action": action,
                "utc": datetime.now(timezone.utc).isoformat(),
                "before": deepcopy(self.state),
            }
        )
        self.dirty = True

    def undo(self):
        if not self._undo:
            return False
        self.state = self._undo.pop()
        self.history.append(
            {"action": "undo", "utc": datetime.now(timezone.utc).isoformat()}
        )
        self.resegment(restore=True)
        self.dirty = True
        return True

    def clear_axis(self):
        self.state.update(
            axis=[],
            axis_status="pending",
            original_skeleton=[],
            automatic_axis=[],
            control_points=[],
            extension=[],
        )
        self.paths = []

    def resegment(self, restore=False):
        labels, self.threshold = segment(
            self.data,
            SegmentationParameters(**self.state["parameters"]),
            self.state["roi"],
        )
        if restore:
            try:
                self.selected = (
                    component_at(labels, self.state["seed"])
                    if self.state["seed"]
                    else None
                )
            except ValueError:
                self.selected = None
            self.selection_reason = (
                "restored" if self.selected is not None else "select_component"
            )
        else:
            self.selected, self.selection_reason = retain_component(
                self.labels, self.selected, labels, self.state["seed"]
            )
            self.clear_axis()
            if self.selected is None:
                self.state["seed"] = None
        self.labels = labels
        if restore and self.selected is not None and self.state["inner"] is not None:
            _, self.paths, _ = skeleton_paths(self.selected, self.state["inner"])
        if self.state["axis"]:
            if self.selected is None:
                raise ValueError("Saved axis has no selected component")
            validate_axis(self.state["axis"], self.selected)

    def parameters(self, **changes):
        parameters = {**self.state["parameters"], **changes}
        SegmentationParameters(**parameters).validate()
        self.checkpoint("segmentation_parameters")
        self.state["parameters"] = parameters
        self.resegment()

    def set_roi(self, vertices):
        self.checkpoint("roi")
        self.state["roi"] = vertices
        self.resegment()

    def set_source(self, source):
        if source not in {"intensity", "difference"} or (
            source == "difference" and self.difference is None
        ):
            raise ValueError("Load a verified difference before selecting it")
        self.checkpoint("measurement_source")
        self.state["source"] = source
        self.state["seed"] = None
        self.selected = None
        self.resegment()

    def select(self, xy):
        selected = component_at(self.labels, xy)
        self.checkpoint("select_component")
        self.selected = selected
        self.state["seed"] = list(map(float, xy))
        self.selection_reason = "component_selected"
        self.clear_axis()

    def trace(self, xy):
        if self.selected is None:
            raise ValueError("Select the target component first")
        component_at(self.selected.astype(int), xy)
        skel, paths, info = skeleton_paths(self.selected, xy)
        self.checkpoint("inner_endpoint")
        self.state["inner"] = list(map(float, xy))
        self.clear_axis()
        self.paths = paths
        self.state["original_skeleton"] = np.argwhere(skel)[:, ::-1].tolist()
        if paths:
            self._set_branch(0)
        return info

    def _set_branch(self, index):
        path = self.paths[index]
        self.state.update(
            axis=path.tolist(),
            automatic_axis=path.tolist(),
            control_points=path.tolist(),
            branch_index=index,
            axis_status="pending",
        )

    def choose_branch(self, index):
        if not 0 <= index < len(self.paths):
            raise ValueError("No such branch")
        self.checkpoint("choose_branch")
        self._set_branch(index)

    def confirm(self):
        if not self.state["axis"]:
            raise ValueError("No axis to confirm")
        self.checkpoint("confirm_branch")
        self.state["axis_status"] = (
            "manual_corrected"
            if self.state["axis"] != self.state["automatic_axis"]
            else "automatic_confirmed"
        )

    def edit(self, controls):
        if self.selected is None:
            raise ValueError("Select a mask before editing an axis")
        dense = validate_axis(controls, self.selected)
        self.checkpoint("edit_control_points")
        self.state.update(
            control_points=np.asarray(controls).tolist(),
            axis=dense.tolist(),
            axis_status="manual_corrected",
        )

    def reverse(self):
        if not self.state["axis"]:
            raise ValueError("No axis to reverse")
        self.checkpoint("reverse_axis")
        for key in ("axis", "control_points", "automatic_axis"):
            self.state[key] = self.state[key][::-1]

    def add_tiepoint(self, xy, number, status, feature):
        if status not in {"possible", "manual_confirmed", "uncertain"}:
            raise ValueError("Unknown identity status")
        if any(p["number"] == number for p in self.state["tiepoints"]):
            raise ValueError(
                "This feature number already exists in this view; undo or remove it first"
            )
        component_at(np.ones(self.raw.shape, dtype=int), xy)
        self.checkpoint("tiepoint")
        self.state["tiepoints"].append(
            {
                "number": int(number),
                "pixel_xy": list(map(float, xy)),
                "hpc_arcsec": pixel_hpc(self.map, [xy])[0].tolist(),
                "identity_status": status,
                "feature": feature,
                "verified_3d": False,
            }
        )

    def measurements(self):
        axis = self.state["axis"]
        result = axis_metrics(pixel_hpc(self.map, axis))
        result.update(
            axis_status=self.state["axis_status"], projected_only=True, widths=[]
        )
        if not axis:
            return result
        for row in cross_sections(self.raw, self.selected, axis):
            edges = pixel_hpc(self.map, row["mask_edge_pixels"])
            row["mask_width_arcsec"] = (
                float(np.linalg.norm(edges[1] - edges[0]))
                if row["mask_width_pixel"]
                else None
            )
            g = row["gaussian"]
            if g["status"] == "valid":
                centre = np.array(row["axis_pixel"]) + g["centre_pixel"] * np.array(
                    row["normal_pixel"]
                )
                half = g["fwhm_pixel"] / 2 * np.array(row["normal_pixel"])
                edges = pixel_hpc(self.map, [centre - half, centre + half])
                g["fwhm_arcsec"] = float(np.linalg.norm(edges[1] - edges[0]))
            result["widths"].append(row)
        return result


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_session(directory, documents, *, sample_id="", role="annotation"):
    """New bundle only; COMPLETE.json is last. No overwriting earlier annotation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    bundle = {
        "schema": "solarphysics.jet_lab",
        "version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "role": role,
        "pixel_origin": 0,
        "pixel_convention": "x=column,y=row; original stored array; pixel centres",
        "geometry_status": "two_dimensional_annotation_not_verified_3d",
        "views": [],
    }
    for index, doc in enumerate(documents):
        if doc is None:
            continue
        view = {
            "view": index,
            "image": os.path.relpath(doc.path, directory),
            "sha256": doc.sha256,
            "observation": doc.info,
            "difference": (
                os.path.relpath(doc.difference_path, directory)
                if doc.difference_path
                else None
            ),
            "difference_sha256": doc.difference_sha256,
            "state": doc.state,
            "history": doc.history,
            "measurements": doc.measurements(),
        }
        bundle["views"].append(view)
        rows = []
        for kind, points in (
            ("axis", doc.state["axis"]),
            ("automatic_axis", doc.state["automatic_axis"]),
            ("skeleton", doc.state["original_skeleton"]),
            ("extension", doc.state["extension"]),
        ):
            for i, (pixel, hpc) in enumerate(
                zip(points, pixel_hpc(doc.map, points), strict=True)
            ):
                offset = doc.info["parent_pixel_offset_xy"]
                rows.append(
                    dict(
                        kind=kind,
                        point_id=i,
                        x_pixel=pixel[0],
                        y_pixel=pixel[1],
                        parent_x_pixel=pixel[0] + offset[0],
                        parent_y_pixel=pixel[1] + offset[1],
                        tx_arcsec=hpc[0],
                        ty_arcsec=hpc[1],
                        status=doc.state["axis_status"],
                    )
                )
        _write_csv(
            directory / f"view_{index}_axis.csv",
            [
                "kind",
                "point_id",
                "x_pixel",
                "y_pixel",
                "parent_x_pixel",
                "parent_y_pixel",
                "tx_arcsec",
                "ty_arcsec",
                "status",
            ],
            rows,
        )
        ties = [
            dict(
                number=p["number"],
                x_pixel=p["pixel_xy"][0],
                y_pixel=p["pixel_xy"][1],
                tx_arcsec=p["hpc_arcsec"][0],
                ty_arcsec=p["hpc_arcsec"][1],
                identity_status=p["identity_status"],
                feature=p["feature"],
                verified_3d=False,
            )
            for p in doc.state["tiepoints"]
        ]
        _write_csv(
            directory / f"view_{index}_tiepoints.csv",
            [
                "number",
                "x_pixel",
                "y_pixel",
                "tx_arcsec",
                "ty_arcsec",
                "identity_status",
                "feature",
                "verified_3d",
            ],
            ties,
        )
        header = doc.map.fits_header.copy()
        header.remove("BLANK", ignore_missing=True)
        header["BUNIT"] = "1"
        header["JLSCHEMA"] = SCHEMA_VERSION
        header["JLSRCHSH"] = doc.sha256
        header["JLMASK"] = "0=background,1=selected,255=missing"
        mask = np.zeros(doc.raw.shape, dtype=np.uint8)
        if doc.selected is not None:
            mask[doc.selected] = 1
        mask[~np.isfinite(doc.data) | ~np.isfinite(doc.raw)] = 255
        fits.PrimaryHDU(mask, header).writeto(directory / f"view_{index}_mask.fits")
    (directory / "annotations.json").write_text(
        json.dumps(json_safe(bundle), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    checksums = {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()}
    (directory / "COMPLETE.json").write_text(
        json.dumps({"version": 1, "sha256": checksums}, indent=2)
    )
    for doc in documents:
        if doc is not None:
            doc.dirty = False
    return directory / "annotations.json"


def load_session(path, validate_path=lambda p: Path(p)):
    path = Path(validate_path(path))
    complete = path.parent / "COMPLETE.json"
    if not complete.is_file():
        raise ValueError("Annotation bundle is incomplete or still syncing")
    for name, digest in json.loads(complete.read_text())["sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid bundle member")
        if file_sha256(validate_path(path.parent / name)) != digest:
            raise ValueError(f"Annotation checksum mismatch: {name}")
    bundle = json.loads(path.read_text())
    if (
        bundle.get("version") != SCHEMA_VERSION
        or bundle.get("schema") != "solarphysics.jet_lab"
    ):
        raise ValueError("Unsupported annotation version")
    docs = [None, None]
    for view in bundle["views"]:
        doc = JetDocument(validate_path(path.parent / view["image"]))
        if doc.sha256 != view["sha256"]:
            raise ValueError("Original image hash changed")
        if view["difference"]:
            doc.load_difference(validate_path(path.parent / view["difference"]))
            if doc.difference_sha256 != view["difference_sha256"]:
                raise ValueError("Difference image hash changed")
        doc.state, doc.history = view["state"], view["history"]
        doc.resegment(restore=True)
        docs[view["view"]] = doc
    return docs, bundle
