"""Manual drift-rate selection and overlay helpers.

English: This module handles user-selected type-III drift-rate lines and the
lightweight HTML/plotting helpers used to review them.

中文：本模块处理人工选取的 III 型漂移率线段，以及用于检查结果的轻量
HTML/绘图辅助逻辑。
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import threading
import warnings
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from solar_toolkit.radio.io import DRIFT_RATE_DIAGNOSTIC_FIELDS
from solar_toolkit.radio.io import drift_output_path as _drift_output_path
from solar_toolkit.radio.io import parse_datetime_value as _parse_datetime_value
from solar_toolkit.radio.spectrogram import (
    _date_num_to_datetime,
    _spectrogram_display_data_extent,
    _spectrogram_time_locator,
)
from solar_apps.workflows.common.image_naming import configured_scientific_image_path

__all__ = [
    "DriftRateResult",
    "assert_spectrogram_mapping_not_flipped",
    "calculate_drift_rate_from_line",
    "get_or_load_drift_rate_results",
    "launch_drift_selection_server",
    "load_drift_selection_json",
    "overlay_drift_rate_results",
    "render_spectrogram_selection_preview",
    "save_drift_rate_diagnostics_once",
    "save_drift_selection_json",
]

_DRIFT_RATE_RESULTS_CACHE = {}
_DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS = set()


@dataclass
class DriftRateResult:
    label: str
    mode: str
    t_start: datetime.datetime
    t_end: datetime.datetime
    f_start_mhz: float
    f_end_mhz: float
    drift_rate_mhz_s: float
    abs_drift_rate_mhz_s: float
    duration_s: float
    bandwidth_mhz: float
    color: str = "white"
    quality_flag: str = "ok"
    warning: str = ""


@dataclass(frozen=True)
class _DriftRateCalculationProfile:
    """Compatibility policy for constructing one drift-rate result."""

    sort_endpoints_by_time: bool
    zero_duration_policy: str
    zero_duration_tolerance_s: float
    default_label: str
    default_mode: str
    default_on_falsy: bool
    warn_positive_drift: bool
    strict_frequency_fields: bool


_CANONICAL_DRIFT_RATE_PROFILE = _DriftRateCalculationProfile(
    sort_endpoints_by_time=True,
    zero_duration_policy="quality_flag",
    zero_duration_tolerance_s=0.0,
    default_label="drift_001",
    default_mode="manual",
    default_on_falsy=True,
    warn_positive_drift=True,
    strict_frequency_fields=False,
)

_CSO_DRIFT_RATE_PROFILE = _DriftRateCalculationProfile(
    sort_endpoints_by_time=False,
    zero_duration_policy="raise",
    zero_duration_tolerance_s=1e-9,
    default_label="drift",
    default_mode="manual_endpoint",
    default_on_falsy=False,
    warn_positive_drift=False,
    strict_frequency_fields=True,
)

_UNSET_DRIFT_TIME = object()


def _datetime_iso_ms(value: datetime.datetime) -> str:
    return value.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _drift_line_time(line: dict, key: str) -> datetime.datetime:
    parsed = _parse_datetime_value(line.get(key))
    if parsed is None:
        raise ValueError(f"Invalid drift-rate time field {key}: {line.get(key)!r}")
    return parsed


def _calculate_drift_rate_from_line(
    line: dict,
    *,
    profile: _DriftRateCalculationProfile = _CANONICAL_DRIFT_RATE_PROFILE,
    t_start=_UNSET_DRIFT_TIME,
    t_end=_UNSET_DRIFT_TIME,
) -> DriftRateResult:
    """Use the canonical exact clock under the retained endpoint profile."""
    from solar_toolkit.radio.drift_rate import (
        _calculate_drift_rate_from_line as calculate,
    )

    kwargs = {"profile": profile}
    if t_start is not _UNSET_DRIFT_TIME:
        kwargs["t_start"] = t_start
    if t_end is not _UNSET_DRIFT_TIME:
        kwargs["t_end"] = t_end
    result = calculate(line, **kwargs)
    return DriftRateResult(**vars(result))


def calculate_drift_rate_from_line(line: dict) -> DriftRateResult:
    """Calculate df/dt with the canonical sorted-endpoint semantics."""
    return _calculate_drift_rate_from_line(line)


def _mark_drift_range_warnings(results, cache):
    x_start, x_end = cache.display_time_nums
    f_min = float(np.nanmin(cache.freq))
    f_max = float(np.nanmax(cache.freq))
    for result in results:
        t1 = mdates.date2num(result.t_start)
        t2 = mdates.date2num(result.t_end)
        out = (
            max(t1, t2) < x_start
            or min(t1, t2) > x_end
            or max(result.f_start_mhz, result.f_end_mhz) < f_min
            or min(result.f_start_mhz, result.f_end_mhz) > f_max
        )
        if out and "out_of_range" not in result.warning:
            result.warning = ";".join(filter(None, [result.warning, "out_of_range"]))
    return results


def _spectrogram_coord_from_pixel(
    metadata: dict, x_pixel: float, y_pixel: float
) -> dict:
    bbox = metadata["axes_bbox_px"]
    left = float(bbox["left"])
    right = float(bbox["right"])
    top = float(bbox["top"])
    bottom = float(bbox["bottom"])
    if x_pixel < left or x_pixel > right or y_pixel < top or y_pixel > bottom:
        raise ValueError("click outside spectrogram axes")
    x_frac = (float(x_pixel) - left) / max(right - left, 1e-12)
    y_frac = (float(y_pixel) - top) / max(bottom - top, 1e-12)
    x_num = float(metadata["x_start_num"]) + x_frac * (
        float(metadata["x_end_num"]) - float(metadata["x_start_num"])
    )
    f_max = float(metadata["f_max_mhz"])
    f_min = float(metadata["f_min_mhz"])
    freq = f_max - y_frac * (f_max - f_min)
    dt = _date_num_to_datetime(x_num)
    return {
        "time_num": x_num,
        "time_iso": _datetime_iso_ms(dt),
        "frequency_mhz": float(freq),
    }


def assert_spectrogram_mapping_not_flipped(metadata):
    bbox = metadata["axes_bbox_px"]
    top_left = _spectrogram_coord_from_pixel(metadata, bbox["left"], bbox["top"])
    bottom_left = _spectrogram_coord_from_pixel(metadata, bbox["left"], bbox["bottom"])
    if top_left["frequency_mhz"] <= bottom_left["frequency_mhz"]:
        raise AssertionError("Spectrogram selector mapping is flipped")


def save_drift_selection_json(path, lines, cache, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_file": cache.source_file,
        "source_files": cache.source_files or [cache.source_file],
        "created_at": _datetime_iso_ms(datetime.datetime.now()),
        "spectrogram_time_start": _datetime_iso_ms(
            _date_num_to_datetime(cache.display_time_nums[0])
        ),
        "spectrogram_time_end": _datetime_iso_ms(
            _date_num_to_datetime(cache.display_time_nums[1])
        ),
        "spectrogram_f_start": float(cfg.get("spectrogram_f_start", np.nan)),
        "spectrogram_f_end": float(cfg.get("spectrogram_f_end", np.nan)),
        "lines": list(lines or []),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_drift_selection_json(path) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift-rate selection JSON does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    return list(payload.get("lines", []) or [])


def _load_drift_selection_payload(path) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift-rate selection JSON does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"lines": payload}
    return dict(payload)


def render_spectrogram_selection_preview(cache, cfg) -> tuple[str, dict]:
    x_start, x_end = cache.display_time_nums
    configured_name = cfg.get(
        "drift_rate_selection_preview_png",
        "drift_rate_selection_preview",
    )
    png_path = str(
        configured_scientific_image_path(
            configured_name,
            output_dir=cfg.get("output_dir") or os.getcwd(),
            sequence=1,
            start_time=_date_num_to_datetime(x_start),
            end_time=_date_num_to_datetime(x_end),
            instrument="radio",
            product=str(configured_name or "drift_rate_selection_preview"),
            generated_at=datetime.datetime.now(datetime.timezone.utc),
        )
    )
    metadata_path = _drift_output_path(cfg, "drift_rate_selection_metadata_json")
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    display_data, extent, f_min, f_max = _spectrogram_display_data_extent(cache)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=int(cfg.get("dpi", 150)))
    ax.imshow(
        display_data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cache.cmap,
        vmin=cache.vmin,
        vmax=cache.vmax,
    )
    ax.set_ylim(f_min, f_max)
    ax.set_title(cache.title)
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Frequency (MHz)")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(cfg.get("spectrogram_xtick_format", "%H:%M:%S"))
    )
    span_seconds = float((x_end - x_start) * 86400.0)
    if np.isfinite(span_seconds) and span_seconds > 0:
        ax.xaxis.set_major_locator(_spectrogram_time_locator(cfg, span_seconds))
    ax.set_xlim(x_start, x_end)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    bbox = ax.get_window_extent()
    metadata = {
        "png_path": png_path,
        "fig_width_px": int(width_px),
        "fig_height_px": int(height_px),
        "axes_bbox_px": {
            "left": float(bbox.x0),
            "right": float(bbox.x1),
            "top": float(height_px - bbox.y1),
            "bottom": float(height_px - bbox.y0),
        },
        "x_start_num": float(x_start),
        "x_end_num": float(x_end),
        "x_start_iso": _datetime_iso_ms(_date_num_to_datetime(x_start)),
        "x_end_iso": _datetime_iso_ms(_date_num_to_datetime(x_end)),
        "f_min_mhz": f_min,
        "f_max_mhz": f_max,
        "x_start_unix_ms": int(
            _date_num_to_datetime(x_start)
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
            * 1000
        ),
        "x_end_unix_ms": int(
            _date_num_to_datetime(x_end)
            .replace(tzinfo=datetime.timezone.utc)
            .timestamp()
            * 1000
        ),
        "freq_axis_order": "increasing_up",
        "source_file": cache.source_file,
    }
    assert_spectrogram_mapping_not_flipped(metadata)
    if cfg.get("draw_coordinate_corner_debug", False):
        ax.text(
            0.01,
            0.98,
            f"top: {f_max:.1f} MHz\nleft: {metadata['x_start_iso']}",
            transform=ax.transAxes,
            color="white",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )
        ax.text(
            0.99,
            0.02,
            f"bottom: {f_min:.1f} MHz\nright: {metadata['x_end_iso']}",
            transform=ax.transAxes,
            color="white",
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
        )
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches=None)
    plt.close(fig)
    saved = plt.imread(png_path)
    saved_height, saved_width = saved.shape[:2]
    if (
        saved_width != metadata["fig_width_px"]
        or saved_height != metadata["fig_height_px"]
    ):
        raise RuntimeError(
            "Selection preview PNG size does not match metadata: "
            f"png={saved_width}x{saved_height}, "
            f"metadata={metadata['fig_width_px']}x{metadata['fig_height_px']}"
        )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return png_path, metadata


def _drift_selection_html(metadata, interactive):
    metadata_json = json.dumps(metadata)
    interactive_json = json.dumps(interactive)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Drift-rate selection</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #202124; color: #f1f3f4; }}
.bar {{ display: flex; gap: 8px; align-items: center; padding: 10px 14px; background: #111; position: sticky; top: 0; z-index: 3; }}
button {{ padding: 7px 10px; border: 1px solid #555; background: #2d2f31; color: white; cursor: pointer; border-radius: 4px; }}
button:hover {{ background: #3a3d40; }}
#status {{ margin-left: auto; color: #d7e3fc; }}
#wrap {{ position: relative; display: inline-block; margin: 14px; }}
#spec {{ display: block; max-width: calc(100vw - 28px); height: auto; }}
#overlay {{ position: absolute; left: 0; top: 0; pointer-events: auto; }}
#hint {{ padding: 0 14px 12px; color: #c9d1d9; }}
</style>
</head>
<body>
<div class="bar">
  <button id="undo">Undo last point</button>
  <button id="delete">Delete last line</button>
  <button id="clear">Clear all</button>
  <button id="check">Check mapping</button>
  <button id="save">Save</button>
  <button id="finish">Save & Continue</button>
  <span id="status">Move over the spectrum</span>
</div>
<div id="wrap">
  <img id="spec" src="/preview.png" alt="spectrogram">
  <canvas id="overlay"></canvas>
</div>
<div id="hint">Click two points for each drift-rate line. Mapping: top = high frequency, bottom = low frequency. Click outside the data axes is ignored.</div>
<script>
const metadata = {metadata_json};
const interactive = {interactive_json};
const img = document.getElementById('spec');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
let points = [];
let lines = [];
let currentMouse = null;
const colors = interactive.line_color_cycle || ['white','cyan','lime','yellow','magenta','orange'];

function scaleInfo() {{
  const sx = img.clientWidth / metadata.fig_width_px;
  const sy = img.clientHeight / metadata.fig_height_px;
  return {{sx, sy}};
}}
function resizeCanvas() {{
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
  draw();
}}
function eventPixel(ev) {{
  const rect = img.getBoundingClientRect();
  const s = scaleInfo();
  return {{x: (ev.clientX - rect.left) / s.sx, y: (ev.clientY - rect.top) / s.sy,
           cx: ev.clientX - rect.left, cy: ev.clientY - rect.top}};
}}
function inAxes(p) {{
  const b = metadata.axes_bbox_px;
  return p.x >= b.left && p.x <= b.right && p.y >= b.top && p.y <= b.bottom;
}}
function mapCoord(p) {{
  const b = metadata.axes_bbox_px;
  const xf = (p.x - b.left) / (b.right - b.left);
  const yf = (p.y - b.top) / (b.bottom - b.top);
  const xnum = metadata.x_start_num + xf * (metadata.x_end_num - metadata.x_start_num);
  const f = metadata.f_max_mhz - yf * (metadata.f_max_mhz - metadata.f_min_mhz);
  const unix_ms = metadata.x_start_unix_ms + xf * (metadata.x_end_unix_ms - metadata.x_start_unix_ms);
  const d = new Date(unix_ms);
  const iso = d.toISOString().replace('Z','');
  return {{time_num: xnum, time_iso: iso, frequency_mhz: f}};
}}
function fmtTime(iso) {{
  const t = iso.split('T')[1] || iso.split(' ')[1] || iso;
  return t.substring(0, 12);
}}
function drift(a,b) {{
  const dt = (b.time_num - a.time_num) * 86400.0;
  return (b.frequency_mhz - a.frequency_mhz) / dt;
}}
function drawPoint(p, color) {{
  const s = scaleInfo();
  ctx.beginPath();
  ctx.arc(p.x * s.sx, p.y * s.sy, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#000';
  ctx.stroke();
}}
function drawLineObj(line, idx) {{
  const s = scaleInfo();
  const a = line._p1, b = line._p2;
  ctx.strokeStyle = line.color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(a.x * s.sx, a.y * s.sy);
  ctx.lineTo(b.x * s.sx, b.y * s.sy);
  ctx.stroke();
  drawPoint(a, line.color);
  drawPoint(b, line.color);
  const mx = (a.x + b.x) * 0.5 * s.sx + 8;
  const my = (a.y + b.y) * 0.5 * s.sy - 8 - idx * 4;
  ctx.font = '13px Arial';
  const text = `${{line.label}} df/dt=${{line.rate.toFixed(2)}} MHz/s`;
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(mx - 4, my - 15, w, 19);
  ctx.fillStyle = line.color;
  ctx.fillText(text, mx, my);
}}
function drawPreviewLine() {{
  if (!interactive.show_preview_line) return;
  if (points.length !== 1 || !currentMouse) return;
  if (!inAxes(currentMouse)) return;
  const s = scaleInfo();
  const a = points[0];
  const b = currentMouse;
  const color = colors[lines.length % colors.length];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.0;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(a.x * s.sx, a.y * s.sy);
  ctx.lineTo(b.x * s.sx, b.y * s.sy);
  ctx.stroke();
  ctx.setLineDash([]);
  drawPoint(a, color);
  drawPoint(b, color);
  const ca = a.coord;
  const cb = mapCoord(b);
  const dt = (cb.time_num - ca.time_num) * 86400.0;
  if (Math.abs(dt) > 1e-9) {{
    const rate = (cb.frequency_mhz - ca.frequency_mhz) / dt;
    const mx = (a.x + b.x) * 0.5 * s.sx + 8;
    const my = (a.y + b.y) * 0.5 * s.sy - 8;
    const text = `preview df/dt=${{rate.toFixed(2)}} MHz/s`;
    ctx.font = '13px Arial';
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(mx - 4, my - 15, w, 19);
    ctx.fillStyle = color;
    ctx.fillText(text, mx, my);
  }}
  ctx.restore();
}}
function drawCrosshair(p) {{
  if (!interactive.show_crosshair || !p || !inAxes(p)) return;
  const s = scaleInfo();
  const b = metadata.axes_bbox_px;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(b.left * s.sx, p.y * s.sy);
  ctx.lineTo(b.right * s.sx, p.y * s.sy);
  ctx.moveTo(p.x * s.sx, b.top * s.sy);
  ctx.lineTo(p.x * s.sx, b.bottom * s.sy);
  ctx.stroke();
  ctx.restore();
}}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  lines.forEach(drawLineObj);
  points.forEach(p => drawPoint(p, colors[lines.length % colors.length]));
  drawPreviewLine();
  drawCrosshair(currentMouse);
}}
function addPoint(p) {{
  const coord = mapCoord(p);
  p.coord = coord;
  points.push(p);
  if (points.length === 1) {{
    statusEl.textContent = 'Start point fixed. Move mouse to preview line; click again to set end point.';
  }}
  if (points.length === 2) {{
    if (!interactive.allow_multiple_lines) {{
      lines = [];
    }}
    const idx = lines.length + 1;
    const color = colors[(idx - 1) % colors.length];
    const rate = drift(points[0].coord, points[1].coord);
    const label = 'drift_' + String(idx).padStart(3, '0');
    lines.push({{
      label, color, rate,
      t_start: points[0].coord.time_iso,
      f_start_mhz: points[0].coord.frequency_mhz,
      t_end: points[1].coord.time_iso,
      f_end_mhz: points[1].coord.frequency_mhz,
      note: '',
      _p1: points[0],
      _p2: points[1]
    }});
    points = [];
    statusEl.textContent = `${{label}} saved. Move to the next start point or Save & Continue.`;
  }}
  draw();
}}
img.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
canvas.addEventListener('mousemove', ev => {{
  const p = eventPixel(ev);
  if (!inAxes(p)) {{
    currentMouse = null;
    statusEl.textContent = 'outside axes';
    draw();
    return;
  }}
  currentMouse = p;
  const c = mapCoord(p);
  statusEl.textContent = `Time: ${{fmtTime(c.time_iso)}}   Frequency: ${{c.frequency_mhz.toFixed(1)}} MHz`;
  draw();
}});
canvas.addEventListener('mouseleave', ev => {{
  currentMouse = null;
  draw();
}});
canvas.addEventListener('click', ev => {{
  const p = eventPixel(ev);
  if (!inAxes(p)) {{
    statusEl.textContent = 'Click ignored: outside axes';
    return;
  }}
  currentMouse = p;
  addPoint(p);
  draw();
}});
document.getElementById('undo').onclick = () => {{ if (points.length > 0) points.pop(); currentMouse = null; draw(); }};
document.getElementById('delete').onclick = () => {{ lines.pop(); draw(); }};
document.getElementById('clear').onclick = () => {{ points = []; lines = []; currentMouse = null; draw(); }};
document.getElementById('check').onclick = () => {{
  const b = metadata.axes_bbox_px;
  const top = mapCoord({{x:b.left, y:b.top}});
  const bottom = mapCoord({{x:b.left, y:b.bottom}});
  statusEl.textContent = `Mapping check: top=${{top.frequency_mhz.toFixed(1)}} MHz, bottom=${{bottom.frequency_mhz.toFixed(1)}} MHz`;
}};
async function post(path) {{
  const payload = lines.map(l => ({{
    label: l.label, t_start: l.t_start, f_start_mhz: l.f_start_mhz,
    t_end: l.t_end, f_end_mhz: l.f_end_mhz, color: l.color, note: l.note || ''
  }}));
  const resp = await fetch(path, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{lines: payload}})}});
  statusEl.textContent = await resp.text();
}}
document.getElementById('save').onclick = () => post('/api/save');
document.getElementById('finish').onclick = () => post('/api/finish');
resizeCanvas();
</script>
</body>
</html>"""


def launch_drift_selection_server(cache, cfg) -> list[dict]:
    preview_path, metadata = render_spectrogram_selection_preview(cache, cfg)
    selection_path = _drift_output_path(cfg, "drift_rate_selection_json")
    os.makedirs(os.path.dirname(selection_path) or ".", exist_ok=True)
    interactive = dict(cfg.get("drift_rate_interactive", {}) or {})
    host = str(interactive.get("host", "127.0.0.1"))
    requested_port = int(interactive.get("port", 8050))
    auto_increment = bool(interactive.get("auto_increment_port", True))
    max_tries = max(1, int(interactive.get("max_port_tries", 20) or 20))
    done_event = threading.Event()
    state = {"lines": []}

    class DriftSelectionHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, status, content, content_type="text/plain; charset=utf-8"):
            body = content.encode("utf-8") if isinstance(content, str) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(
                    200,
                    _drift_selection_html(metadata, interactive),
                    "text/html; charset=utf-8",
                )
            elif path == "/preview.png":
                with open(preview_path, "rb") as handle:
                    self._send(200, handle.read(), "image/png")
            elif path == "/metadata.json":
                self._send(200, json.dumps(metadata), "application/json")
            else:
                self._send(404, "not found")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            lines = payload.get("lines", [])
            state["lines"] = lines
            save_drift_selection_json(selection_path, lines, cache, cfg)
            if path == "/api/finish":
                done_event.set()
                self._send(200, f"Saved {len(lines)} line(s). You can close this tab.")
            elif path == "/api/save":
                self._send(200, f"Saved {len(lines)} line(s).")
            else:
                self._send(404, "not found")

    server = None
    last_error = None
    port_range = range(requested_port, requested_port + max_tries)
    for candidate_port in port_range:
        try:
            server = ThreadingHTTPServer((host, candidate_port), DriftSelectionHandler)
            port = candidate_port
            break
        except OSError as exc:
            last_error = exc
            if not auto_increment:
                break
    if server is None:
        end_port = requested_port + max_tries - 1
        raise OSError(
            f"Cannot start drift selection server. Ports "
            f"{requested_port}-{end_port} are unavailable."
        ) from last_error
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{port}"
    print("=" * 70)
    print("[Drift selection] Interactive endpoint selector is running")
    print(f"[Drift selection] URL: {url}")
    print(f"[Drift selection] Preview PNG: {preview_path}")
    print(
        "[Drift selection] Metadata JSON: "
        f"{_drift_output_path(cfg, 'drift_rate_selection_metadata_json')}"
    )
    print(f"[Drift selection] Selection JSON: {selection_path}")
    print("[Drift selection] Click two points for each drift-rate line.")
    print("[Drift selection] Click 'Save & Continue' to return to Python.")
    print("=" * 70)
    if interactive.get("auto_open_browser", True):
        opened = webbrowser.open(url)
        if not opened:
            print(f"[Drift selection] Browser did not open; copy this URL: {url}")
    if interactive.get("block_until_done", True):
        timeout = float(interactive.get("selection_timeout_seconds", 0) or 0)
        finished = done_event.wait(timeout if timeout > 0 else None)
        if not finished:
            server.shutdown()
            raise TimeoutError("Drift-rate selection timed out")
    server.shutdown()
    thread.join(timeout=5)
    if not state["lines"] and os.path.exists(selection_path):
        state["lines"] = load_drift_selection_json(selection_path)
    return list(state["lines"] or [])


def get_or_load_drift_rate_results(
    cache, cfg, launch_func=None
) -> list[DriftRateResult]:
    if not cfg.get("enable_drift_rate_overlay", False):
        return []
    launch_func = launch_func or launch_drift_selection_server
    mode = str(cfg.get("drift_rate_mode", "off") or "off").lower()
    if mode == "off":
        return []
    selection_path = _drift_output_path(cfg, "drift_rate_selection_json")
    if cfg.get("_drift_selection_cli_path"):
        selection_path = cfg["_drift_selection_cli_path"]
    interactive = dict(cfg.get("drift_rate_interactive", {}) or {})
    launch_policy = str(interactive.get("launch_policy", "cli_only") or "cli_only")
    cache_key = (mode, os.path.abspath(selection_path), launch_policy)
    if cache_key in _DRIFT_RATE_RESULTS_CACHE:
        return _DRIFT_RATE_RESULTS_CACHE[cache_key]
    selection_exists = os.path.exists(selection_path)

    def _load_selection_payload():
        payload = _load_drift_selection_payload(selection_path)
        source_file = payload.get("source_file")
        if source_file and os.path.abspath(str(source_file)) != os.path.abspath(
            cache.source_file
        ):
            warnings.warn(
                "Drift-rate selection source_file differs from current "
                f"spectrogram_file_path: {source_file}",
                stacklevel=2,
            )
        return list(payload.get("lines", []) or [])

    if mode == "interactive_manual":
        if cfg.get("_select_drift_now", False):
            lines = launch_func(cache, cfg)
        elif launch_policy == "always":
            lines = launch_func(cache, cfg)
        elif launch_policy == "auto_if_missing" and not selection_exists:
            print(
                "[Drift selection] selection JSON not found; "
                "starting interactive selector..."
            )
            lines = launch_func(cache, cfg)
        elif selection_exists:
            lines = _load_selection_payload()
        else:
            hint = (
                "No drift-rate selection JSON found. Run:\n"
                "  python run_radio_burst_pipeline.py "
                "--select-drift --drift-port 8050\n"
                "or set drift_rate.interactive.launch_policy='auto_if_missing'."
            )
            if interactive.get("print_usage_hint", True):
                print(f"[Drift selection] {hint}")
            warnings.warn(
                f"No drift-rate selection JSON found: {selection_path}",
                stacklevel=2,
            )
            return []
    elif mode == "manual_json":
        if not selection_exists:
            warnings.warn(
                "No drift-rate selection JSON found for manual_json mode. Run: "
                "python run_radio_burst_pipeline.py --select-drift "
                "--drift-port 8050",
                stacklevel=2,
            )
            return []
        lines = _load_selection_payload()
    elif mode in {"auto_peak", "auto_ridge"}:
        warnings.warn(
            f"drift_rate_mode={mode!r} is reserved for future implementation.",
            stacklevel=2,
        )
        return []
    else:
        return []
    results = [calculate_drift_rate_from_line(line) for line in lines]
    results = _mark_drift_range_warnings(results, cache)
    _DRIFT_RATE_RESULTS_CACHE[cache_key] = results
    return results


def overlay_drift_rate_results(ax, results, cfg):
    if not results:
        return
    color_cycle = cfg.get("drift_rate_interactive", {}).get(
        "line_color_cycle", ["white", "cyan", "lime", "yellow", "magenta", "orange"]
    )
    line_width = float(cfg.get("drift_rate_line_width", 2.2))
    endpoint_marker = cfg.get("drift_rate_endpoint_marker", "o")
    endpoint_size = float(cfg.get("drift_rate_endpoint_size", 30))
    for idx, result in enumerate(results):
        color = result.color or color_cycle[idx % len(color_cycle)]
        x1 = mdates.date2num(result.t_start)
        x2 = mdates.date2num(result.t_end)
        if cfg.get("draw_drift_rate_lines", True):
            ax.plot(
                [x1, x2],
                [result.f_start_mhz, result.f_end_mhz],
                color=color,
                linewidth=line_width,
                alpha=0.95,
                clip_on=True,
                zorder=4,
            )
        if cfg.get("draw_drift_rate_endpoints", True):
            ax.scatter(
                [x1, x2],
                [result.f_start_mhz, result.f_end_mhz],
                marker=endpoint_marker,
                s=endpoint_size,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                clip_on=True,
                zorder=5,
            )
        if cfg.get("draw_drift_rate_label", True):
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (result.f_start_mhz + result.f_end_mhz)
            label = cfg.get(
                "drift_rate_label_format", "{label}: df/dt={drift_rate:.2f} MHz/s"
            ).format(
                label=result.label,
                drift_rate=result.drift_rate_mhz_s,
                abs_drift_rate=result.abs_drift_rate_mhz_s,
            )
            if "out_of_range" in result.warning:
                label = f"{label} (out_of_range)"
            ax.annotate(
                label,
                xy=(xm, ym),
                xytext=(8, 8 + 5 * (idx % 3)),
                textcoords="offset points",
                color=color,
                fontsize=max(cfg.get("annotation_fontsize", 20) - 10, 8),
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
                clip_on=True,
                zorder=6,
            )


def save_drift_rate_diagnostics_once(results, cfg, source_file):
    if not results:
        return
    csv_path = _drift_output_path(cfg, "drift_rate_diagnostics_csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            header = []
        if header and header != DRIFT_RATE_DIAGNOSTIC_FIELDS:
            shutil.copy2(csv_path, f"{csv_path}.bak")
            os.remove(csv_path)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DRIFT_RATE_DIAGNOSTIC_FIELDS, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        for result in results:
            key = (
                source_file,
                result.label,
                result.t_start.isoformat(),
                result.t_end.isoformat(),
            )
            if key in _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS:
                continue
            _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS.add(key)
            writer.writerow(
                {
                    "source_file": source_file,
                    "label": result.label,
                    "mode": result.mode,
                    "t_start": _datetime_iso_ms(result.t_start),
                    "t_end": _datetime_iso_ms(result.t_end),
                    "f_start_mhz": result.f_start_mhz,
                    "f_end_mhz": result.f_end_mhz,
                    "duration_s": result.duration_s,
                    "bandwidth_mhz": result.bandwidth_mhz,
                    "drift_rate_mhz_s": result.drift_rate_mhz_s,
                    "abs_drift_rate_mhz_s": result.abs_drift_rate_mhz_s,
                    "color": result.color,
                    "quality_flag": result.quality_flag,
                    "warning": result.warning,
                }
            )
