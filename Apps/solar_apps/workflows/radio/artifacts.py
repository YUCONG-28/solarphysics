"""Source-map artifact, unit, image geometry, and ROI contracts.

This module is workflow-owned so scientific rendering never depends on a web,
Streamlit, or Qt frontend.  Frontends may re-export these APIs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_SCHEMA_VERSION = 1
ROI_SCHEMA_VERSION = 1
COORDINATE_SYSTEM = "HPLN/HPLT arcsec"
POWER_OF_TEN_TICKS = "power_of_ten"
SCIENTIFIC_OFFSET_TICKS = "scientific_offset"
_TICK_NOTATIONS = {POWER_OF_TEN_TICKS, SCIENTIFIC_OFFSET_TICKS}


@dataclass(frozen=True)
class UnitResolution:
    unit: str
    source: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_colorbar_unit(
    headers: Sequence[Mapping[str, Any] | None], override: Any = None
) -> UnitResolution:
    explicit = str(override or "").strip()
    if explicit:
        return UnitResolution(explicit, "config_override")
    values: list[str | None] = []
    for header in headers:
        raw = header.get("BUNIT") if header is not None else None
        values.append(" ".join(str(raw).split()) if raw not in (None, "") else None)
    present = [value for value in values if value]
    if values and len(present) == len(values):
        if len({value.casefold() for value in present}) == 1:
            return UnitResolution(present[0], "fits_bunit")
    message = (
        "Contributing FITS files have missing or inconsistent BUNIT values; "
        "using a.u. for the colorbar."
        if present
        else "FITS BUNIT is unavailable; using a.u. for the colorbar."
    )
    return UnitResolution("a.u.", "fallback", (message,))


def tick_notation_for_transform(transform: str) -> str:
    if transform == "linear":
        return SCIENTIFIC_OFFSET_TICKS
    if transform == "log10":
        return POWER_OF_TEN_TICKS
    raise ValueError(f"Unsupported colorbar transform: {transform}")


def colorbar_label(
    resolution: UnitResolution,
    *,
    transform: str = "linear",
    tick_notation: str | None = None,
) -> str:
    if transform == "linear":
        return f"Intensity [{resolution.unit}]"
    if transform != "log10":
        raise ValueError(f"Unsupported colorbar transform: {transform}")
    if tick_notation == POWER_OF_TEN_TICKS:
        return f"Intensity [{resolution.unit}]"
    if resolution.unit == "a.u.":
        return "log10 Intensity [a.u.]"
    return f"log10(I / 1 {resolution.unit})"


def apply_colorbar_tick_notation(colorbar: Any, *, transform: str) -> str:
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    notation = tick_notation_for_transform(transform)
    if notation == POWER_OF_TEN_TICKS:
        colorbar.formatter = FuncFormatter(
            lambda value, _position: rf"$10^{{{_format_exponent(value)}}}$"
        )
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        colorbar.formatter = formatter
    colorbar.update_ticks()
    return notation


def sidecar_path_for(image_path: str | Path) -> Path:
    return Path(image_path).with_suffix(".source-map.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure_artifact(
    fig: Any,
    image_path: str | Path,
    *,
    dpi: int,
    radio_axes: Sequence[Any],
    panel_metadata: Sequence[Mapping[str, Any]],
    mode: str,
    polarization: str,
    source_files: Sequence[str | Path],
    write_sidecar: bool,
    warnings: Sequence[str] = (),
    display: Mapping[str, Any] | None = None,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.1,
) -> tuple[Path, Path | None]:
    """Save PNG and optional schema-1 sidecar atomically.

    ``display`` is optional for backward compatibility.  It contains only
    material spatial-radio display parameters and must not contain UI theme.
    """

    if len(radio_axes) != len(panel_metadata):
        raise ValueError("radio_axes and panel_metadata must have equal lengths")
    target = Path(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-", suffix=target.suffix, dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        if bbox_inches == "tight":
            canvas = fig.canvas
            if not callable(getattr(canvas, "get_renderer", None)):
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                canvas = FigureCanvasAgg(fig)
            canvas.draw()
            boxes = _normalized_panel_boxes(
                fig,
                canvas.get_renderer(),
                radio_axes,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
            fig.savefig(
                temporary,
                dpi=int(dpi),
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
            )
        else:
            boxes = _save_fixed_canvas_png(
                fig,
                temporary,
                dpi=int(dpi),
                radio_axes=radio_axes,
                pad_inches=pad_inches,
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if not write_sidecar:
        return target, None

    from PIL import Image

    with Image.open(target) as image:
        width, height = image.size
    panels = []
    for axis, box, supplied in zip(radio_axes, boxes, panel_metadata):
        record = dict(supplied)
        record.update(
            {
                "bbox_normalized": [float(value) for value in box],
                "xlim_arcsec": [float(value) for value in axis.get_xlim()],
                "ylim_arcsec": [float(value) for value in axis.get_ylim()],
            }
        )
        panels.append(record)
    payload: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": {
            "filename": target.name,
            "width": int(width),
            "height": int(height),
            "sha256": sha256_file(target),
        },
        "mode": str(mode),
        "polarization": str(polarization),
        "source_files": [str(Path(path)) for path in source_files],
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item))),
        "panels": panels,
    }
    if display is not None:
        display_payload = _display_payload(display)
        payload["display"] = display_payload
    sidecar = sidecar_path_for(target)
    _atomic_write_json(sidecar, payload)
    return target, sidecar


def _save_fixed_canvas_png(
    fig: Any,
    target: Path,
    *,
    dpi: int,
    radio_axes: Sequence[Any],
    pad_inches: float,
) -> list[list[float]]:
    """Render a fixed-canvas artifact once at its final DPI.

    Re-running ``Figure.savefig`` after measuring the panel boxes can invoke a
    second layout/aspect pass.  Under threaded frontend rendering that allowed
    the saved pixels to diverge from the sidecar even though the outer PNG size
    stayed constant.  The fixed-canvas path therefore measures and writes the
    exact same Agg buffer.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    from PIL import Image

    requested_dpi = int(dpi)
    if requested_dpi < 1:
        raise ValueError("Artifact DPI must be positive")
    original_dpi = float(fig.dpi)
    original_canvas = fig.canvas
    canvas = FigureCanvasAgg(fig)
    try:
        fig.set_dpi(requested_dpi)
        canvas.draw()
        boxes = _normalized_panel_boxes(
            fig,
            canvas.get_renderer(),
            radio_axes,
            bbox_inches=None,
            pad_inches=pad_inches,
        )
        width, height = canvas.get_width_height()
        expected_width = int(round(float(fig.get_figwidth()) * requested_dpi))
        expected_height = int(round(float(fig.get_figheight()) * requested_dpi))
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                "Fixed-canvas renderer returned an unexpected pixel size: "
                f"expected {expected_width}x{expected_height}, got {width}x{height}"
            )
        # ``buffer_rgba`` is a live view owned by the Agg renderer.  Copy it
        # before restoring the Figure DPI/canvas so a frontend worker cannot
        # publish pixels from the subsequently resized (usually 100-DPI)
        # renderer.  This is especially important for streamed sequence jobs.
        rgba = np.frombuffer(bytes(canvas.buffer_rgba()), dtype=np.uint8).reshape(
            height, width, 4
        )
        Image.fromarray(rgba, mode="RGBA").save(
            target,
            format="PNG",
            dpi=(requested_dpi, requested_dpi),
        )
        return boxes
    finally:
        fig.set_dpi(original_dpi)
        if canvas is not original_canvas:
            fig.set_canvas(original_canvas)


def validate_source_map_artifact(
    image_path: str | Path, sidecar_path: str | Path | None = None
) -> dict[str, Any]:
    image = Path(image_path).resolve(strict=True)
    sidecar = (
        Path(sidecar_path).resolve(strict=True)
        if sidecar_path is not None
        else sidecar_path_for(image).resolve(strict=True)
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported source-map sidecar schema")
    if payload.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError("Sidecar coordinate system must be HPLN/HPLT arcsec")
    image_record = payload.get("image")
    if not isinstance(image_record, dict):
        raise ValueError("Sidecar image record is missing")
    if image_record.get("filename") != image.name:
        raise ValueError("Sidecar image filename does not match the selected PNG")
    from PIL import Image

    with Image.open(image) as opened:
        actual_size = list(opened.size)
    if actual_size != [image_record.get("width"), image_record.get("height")]:
        raise ValueError("Sidecar image dimensions do not match the selected PNG")
    if image_record.get("sha256") != sha256_file(image):
        raise ValueError("Sidecar image SHA-256 does not match the selected PNG")
    if "display" in payload:
        payload["display"] = _display_payload(payload["display"])
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("Sidecar contains no radio panel mappings")
    seen: set[str] = set()
    for panel in panels:
        _validate_panel(panel, seen)
    payload["_image_path"] = str(image)
    payload["_sidecar_path"] = str(sidecar)
    return payload


def data_to_image_pixel(
    metadata: Mapping[str, Any], panel_id: str, x_arcsec: float, y_arcsec: float
) -> tuple[float, float]:
    panel = _panel_by_id(metadata, panel_id)
    width, height = float(metadata["image"]["width"]), float(
        metadata["image"]["height"]
    )
    left, top, right, bottom = panel["bbox_normalized"]
    x0, x1 = panel["xlim_arcsec"]
    y0, y1 = panel["ylim_arcsec"]
    return (
        (left + (float(x_arcsec) - x0) / (x1 - x0) * (right - left)) * width,
        (bottom - (float(y_arcsec) - y0) / (y1 - y0) * (bottom - top)) * height,
    )


def image_pixel_to_data(
    metadata: Mapping[str, Any], panel_id: str, x_pixel: float, y_pixel: float
) -> tuple[float, float]:
    panel = _panel_by_id(metadata, panel_id)
    width, height = float(metadata["image"]["width"]), float(
        metadata["image"]["height"]
    )
    left, top, right, bottom = panel["bbox_normalized"]
    x0, x1 = panel["xlim_arcsec"]
    y0, y1 = panel["ylim_arcsec"]
    xf = (float(x_pixel) / width - left) / (right - left)
    yf = (bottom - float(y_pixel) / height) / (bottom - top)
    return x0 + xf * (x1 - x0), y0 + yf * (y1 - y0)


def validate_roi_set(payload: Any, *, expected_image_sha256: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("ROI set must be a JSON object")
    if payload.get("schema_version") != ROI_SCHEMA_VERSION:
        raise ValueError("Unsupported ROI set schema")
    if payload.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError("ROI coordinate system must be HPLN/HPLT arcsec")
    if payload.get("image_sha256") != expected_image_sha256:
        raise ValueError("ROI set does not match the source image")
    rois = payload.get("rois")
    if not isinstance(rois, list):
        raise ValueError("rois must be an array")
    names: set[str] = set()
    normalized = []
    for index, raw in enumerate(rois):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ROI #{index + 1} must be an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("Every ROI requires a name")
        key = name.casefold()
        if key in names:
            raise ValueError(f"ROI names must be unique: {name}")
        names.add(key)
        roi_type = str(raw.get("type") or "").lower()
        geometry = raw.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError(f"ROI {name} geometry must be an object")
        if roi_type == "rectangle":
            left, bottom, right, top = (
                _finite(geometry.get(item))
                for item in ("left", "bottom", "right", "top")
            )
            left, right = sorted((left, right))
            bottom, top = sorted((bottom, top))
            if left == right or bottom == top:
                raise ValueError(f"ROI {name} rectangle must have positive area")
            cleaned: dict[str, Any] = {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            }
        elif roi_type == "lasso":
            raw_points = geometry.get("points")
            if not isinstance(raw_points, list):
                raise ValueError(f"ROI {name} lasso points must be an array")
            points = [
                [_finite(point[0]), _finite(point[1])]
                for point in raw_points
                if isinstance(point, (list, tuple)) and len(point) == 2
            ]
            if (
                len(points) != len(raw_points)
                or len({tuple(item) for item in points}) < 3
            ):
                raise ValueError(
                    f"ROI {name} lasso requires at least three unique points"
                )
            cleaned = {"points": points}
        else:
            raise ValueError(f"ROI {name} type must be rectangle or lasso")
        style = raw.get("style") if isinstance(raw.get("style"), Mapping) else {}
        normalized.append(
            {
                "id": str(raw.get("id") or f"roi-{index + 1}"),
                "name": name,
                "type": roi_type,
                "geometry": cleaned,
                "visible": bool(raw.get("visible", True)),
                "style": {
                    "color": str(style.get("color") or "#00d4ff"),
                    "line_width": min(
                        max(_finite(style.get("line_width", 3)), 1.0), 12.0
                    ),
                    "show_label": bool(style.get("show_label", True)),
                },
            }
        )
    return {
        "schema_version": ROI_SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "image_sha256": expected_image_sha256,
        "rois": normalized,
    }


def _display_payload(display: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(display, Mapping):
        raise ValueError("sidecar display must be an object")
    forbidden = {"theme", "ui_theme", "theme_mode"}.intersection(display)
    if forbidden:
        raise ValueError("UI theme must not be stored in source-map sidecars")
    from .spatial_display import SpatialRadioDisplay

    return SpatialRadioDisplay.from_mapping(display).sidecar_payload()


def _normalized_panel_boxes(
    fig: Any,
    renderer: Any,
    axes: Sequence[Any],
    *,
    bbox_inches: str | None,
    pad_inches: float,
) -> list[list[float]]:
    if bbox_inches != "tight":
        crop_x0 = crop_y0 = 0.0
        crop_width, crop_height = fig.get_size_inches()
    else:
        tight = fig.get_tightbbox(renderer)
        crop_x0, crop_y0 = float(tight.x0) - pad_inches, float(tight.y0) - pad_inches
        crop_width = float(tight.width) + 2.0 * pad_inches
        crop_height = float(tight.height) + 2.0 * pad_inches
    boxes = []
    for axis in axes:
        extent = axis.get_window_extent(renderer)
        x0, x1 = float(extent.x0) / fig.dpi, float(extent.x1) / fig.dpi
        y0, y1 = float(extent.y0) / fig.dpi, float(extent.y1) / fig.dpi
        boxes.append(
            [
                _clamp01((x0 - crop_x0) / crop_width),
                _clamp01(1.0 - (y1 - crop_y0) / crop_height),
                _clamp01((x1 - crop_x0) / crop_width),
                _clamp01(1.0 - (y0 - crop_y0) / crop_height),
            ]
        )
    return boxes


def _validate_panel(panel: Any, seen: set[str]) -> None:
    if not isinstance(panel, Mapping):
        raise ValueError("Every sidecar panel must be an object")
    panel_id = str(panel.get("id") or "")
    if not panel_id or panel_id in seen:
        raise ValueError("Sidecar panel IDs must be present and unique")
    seen.add(panel_id)
    notation = panel.get("tick_notation")
    if notation is not None and notation not in _TICK_NOTATIONS:
        raise ValueError(f"Panel {panel_id} has an invalid tick_notation")
    bbox = panel.get("bbox_normalized")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Panel {panel_id} has an invalid normalized bbox")
    left, top, right, bottom = [_finite(value) for value in bbox]
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"Panel {panel_id} normalized bbox is outside the image")
    for key in ("xlim_arcsec", "ylim_arcsec"):
        limits = panel.get(key)
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError(f"Panel {panel_id} is missing {key}")
        first, second = [_finite(value) for value in limits]
        if first == second:
            raise ValueError(f"Panel {panel_id} {key} must span a nonzero range")


def _panel_by_id(metadata: Mapping[str, Any], panel_id: str) -> Mapping[str, Any]:
    for panel in metadata.get("panels", []):
        if panel.get("id") == panel_id:
            return panel
    raise KeyError(f"Unknown panel ID: {panel_id}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}-",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _format_exponent(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return ""
    if abs(numeric) < 5e-13:
        numeric = 0.0
    return f"{numeric:.6g}"


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("ROI and panel coordinates must be finite")
    return number


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
