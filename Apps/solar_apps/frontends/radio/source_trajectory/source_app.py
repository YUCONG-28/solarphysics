"""Streamlit frontend for radio-source trajectory playback."""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import io
import json
import os
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import AllowedRootPolicyError
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
    render_native_path_input,
    resolve_streamlit_allowed_roots,
)
from solar_apps.ui.media import read_asset_text
from solar_apps.ui.state import (
    bind_streamlit_fields,
    frontend_path_memory,
    frontend_state_store,
    save_streamlit_fields,
)
from solar_apps.ui.theme import render_streamlit_theme, streamlit_theme_css
from solar_toolkit.aia.background import (
    AiaBackground,
    find_nearest_aia,
    read_aia_background,
    scan_aia_folder,
)
from solar_toolkit.radio.trajectory import (
    FRAME_MODE_ALL,
    FRAME_MODE_LABELS,
    FRAME_MODE_TAIL,
    filter_centers,
    filter_time_range,
    frame_times,
    load_centers_table,
    select_visible_centers,
    summarize_motion,
)
from solar_apps.workflows.visualization.radio_source_trajectory import (
    FACET_BY_OPTIONS,
    MARKER_SYMBOL_OPTIONS,
    PLOT_LAYOUTS,
    apply_aia_colormap_to_uint8,
    build_trajectory_figure,
    frequency_marker_key,
    marker_symbol_for_frequency,
    normalize_marker_symbol_by_frequency,
    resolve_theme_palette,
)
from solar_apps.workflows.visualization.radio_source_video import (
    VideoExportOptions,
    export_radio_source_video,
)

__all__ = [
    "APP_CLI_SETTING_MAP",
    "DEFAULT_APP_SETTINGS",
    "THEME_MODES",
    "aia_background_to_png_image",
    "build_parser",
    "build_preload_signature",
    "build_preloaded_playback_html",
    "build_preloaded_playback_payload",
    "bundled_browser_media_scripts",
    "coalesce_frame_times",
    "default_frequency_marker_symbols",
    "default_settings_path",
    "load_app_settings",
    "main",
    "playback_interval_seconds",
    "preload_settings_changed",
    "resolve_app_settings",
    "resolve_effective_aia_max_pixels",
    "save_app_settings",
    "should_build_preloaded_payload",
]

THEME_MODES = ("light", "dark", "auto")
SCREEN_FIT_MODES = ("auto", "landscape", "portrait")
PLAYBACK_RENDERERS = ("preloaded", "streamlit")
VIDEO_DEFAULT_WIDTH = 1280
VIDEO_DEFAULT_HEIGHT = 720
VIDEO_DEFAULT_FPS = 6.0
VIDEO_OUTPUT_FORMATS = ("mp4", "gif", "webm")
BROWSER_RECORDING_FORMATS = ("mp4", "webm")
VIDEO_QUALITY_OPTIONS = ("low", "high")
DEFAULT_MARKER_SIZE = 8
DEFAULT_TRAIL_MIN_OPACITY = 0.25
DEFAULT_PLAYBACK_AIA_MAX_PIXELS = 384
DEFAULT_PLAYBACK_MIN_STEP_SEC = 0.25
DEFAULT_MARKER_SYMBOL_CYCLE = MARKER_SYMBOL_OPTIONS
PRELOADED_FACET_SPACING = {"x": 0.075, "y": 0.13}

DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "centers": "radio_centers.csv",
    "time_start": "",
    "time_end": "",
    "aia_dir": "",
    "aia_pattern": "*.fits",
    "frame_mode": FRAME_MODE_TAIL,
    "tail_n": 5,
    "plot_layout": "overlay",
    "facet_by": "freq_mhz",
    "theme_mode": "auto",
    "screen_fit": "auto",
    "use_aia": False,
    "max_pixels": 1024,
    "percentile_limits": [1.0, 99.7],
    "log_scale": True,
    "max_aia_dt_sec": 3600.0,
    "wcs_mode": "header",
    "compare_lr": True,
    "compare_tolerance_sec": 1.0,
    "draw_lines": True,
    "marker_size": DEFAULT_MARKER_SIZE,
    "frequency_marker_symbols": {},
    "trail_min_opacity": DEFAULT_TRAIL_MIN_OPACITY,
    "link_facet_views": False,
    "fps": 2.0,
    "video_width": VIDEO_DEFAULT_WIDTH,
    "video_height": VIDEO_DEFAULT_HEIGHT,
    "video_fps": VIDEO_DEFAULT_FPS,
    "video_output_format": "mp4",
    "video_browser_format": "webm",
    "video_quality": "high",
    "video_output_path": "",
    "video_include_aia": True,
    "playback_renderer": "preloaded",
    "playback_aia_max_pixels": DEFAULT_PLAYBACK_AIA_MAX_PIXELS,
    "playback_min_step_sec": DEFAULT_PLAYBACK_MIN_STEP_SEC,
    "selected_freqs": [],
    "selected_pols": [],
    "selected_methods": [],
    "show_debug_tables": False,
}

TRAJECTORY_UI_FIELD_KEYS = (
    "centers_path",
    "time_start",
    "time_end",
    "aia_dir",
    "screen_fit",
    "use_aia",
    "aia_background_max_side",
    "percentile_limits",
    "log_scale",
    "playback_aia_preview_max_side",
    "playback_fps",
    "frame_mode",
    "frames_shown_per_trace",
    "center_marker_size",
    "plot_layout",
    "facet_by",
    "aia_pattern",
    "playback_renderer",
    "draw_lines",
    "max_aia_dt_sec",
    "wcs_mode",
    "compare_lr",
    "compare_tolerance_sec",
    "playback_min_step_sec",
    "minimum_historical_point_opacity",
    "link_facet_views",
    "video_output_format",
    "video_browser_format",
    "video_quality",
    "video_output_path",
    "video_width",
    "video_height",
    "video_fps",
    "video_include_aia",
    "show_debug_tables",
    "selected_freqs",
    "selected_pols",
    "selected_methods",
)

APP_CLI_SETTING_MAP = {
    "centers": "centers",
    "time_start": "time_start",
    "time_end": "time_end",
    "aia_dir": "aia_dir",
    "aia_pattern": "aia_pattern",
    "frame_mode": "frame_mode",
    "tail_n": "tail_n",
    "plot_layout": "plot_layout",
    "facet_by": "facet_by",
    "theme_mode": "theme_mode",
    "screen_fit": "screen_fit",
}


def build_parser() -> argparse.ArgumentParser:
    """Build a lightweight help parser for direct ``python --help`` use."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch the Streamlit radio-source trajectory app. "
            "Use the public frontend source-trajectory command."
        )
    )
    parser.add_argument(
        "--centers",
        default=None,
        help="Default center CSV/XLSX path shown in the sidebar.",
    )
    parser.add_argument(
        "--time-start",
        default=None,
        help="Inclusive default time-window start in ISO-8601 UTC form.",
    )
    parser.add_argument(
        "--time-end",
        default=None,
        help="Inclusive default time-window end in ISO-8601 UTC form.",
    )
    parser.add_argument(
        "--aia-dir",
        default=None,
        help="Default AIA FITS folder shown in the sidebar.",
    )
    parser.add_argument(
        "--aia-pattern",
        default=None,
        help="Default AIA FITS glob pattern.",
    )
    parser.add_argument(
        "--frame-mode",
        choices=list(FRAME_MODE_LABELS),
        default=None,
        help="Default trajectory display mode: current, tail, or all.",
    )
    parser.add_argument(
        "--tail-n",
        type=int,
        default=None,
        help="Default tail length for tail display mode.",
    )
    parser.add_argument(
        "--plot-layout",
        choices=PLOT_LAYOUTS,
        default=None,
        help="Default trajectory layout: overlay or facets.",
    )
    parser.add_argument(
        "--facet-by",
        choices=FACET_BY_OPTIONS,
        default=None,
        help="Default faceting dimension for facets layout.",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help="Local JSON settings file used to remember paths and UI choices.",
    )
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to this app.",
    )
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="Ignore the saved settings file and rebuild defaults for this run.",
    )
    parser.add_argument(
        "--theme-mode",
        choices=THEME_MODES,
        default=None,
        help="Default app theme mode.",
    )
    parser.add_argument(
        "--screen-fit",
        choices=SCREEN_FIT_MODES,
        default=None,
        help="Default screen layout mode.",
    )
    return parser


def default_settings_path() -> Path:
    """Return the default local settings path for this single-user app."""

    override = os.environ.get("SOLAR_TOOLKIT_RADIO_SOURCE_APP_SETTINGS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".solar_toolkit" / "radio_source_trajectory_app.json"


def load_app_settings(
    settings_file: str | Path | None = None,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    """Load remembered UI settings, falling back to defaults on any issue."""

    if reset:
        return _coerce_app_settings({})
    path = (
        Path(settings_file).expanduser() if settings_file else default_settings_path()
    )
    if not path.exists():
        return _coerce_app_settings({})
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _coerce_app_settings({})
    if not isinstance(loaded, dict):
        return _coerce_app_settings({})
    return _coerce_app_settings(loaded)


def save_app_settings(
    settings_file: str | Path | None,
    settings: dict[str, Any],
) -> Path:
    """Persist known app settings to local JSON and return the file path."""

    path = (
        Path(settings_file).expanduser() if settings_file else default_settings_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _coerce_app_settings(settings)
    path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def resolve_app_settings(
    args: argparse.Namespace,
    stored_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge defaults, local memory, and explicit CLI values."""

    resolved = _coerce_app_settings(stored_settings or {})
    for arg_name, setting_name in APP_CLI_SETTING_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            resolved[setting_name] = value
    return _coerce_app_settings(resolved)


def coalesce_frame_times(
    times: list[Any], *, min_step_sec: float = 0.0
) -> list[pd.Timestamp]:
    """Reduce dense playback frames while preserving the first and last frame."""

    normalized = [
        pd.Timestamp(value) for value in times if pd.notna(pd.Timestamp(value))
    ]
    if not normalized:
        return []
    ordered = sorted(normalized)
    step = max(0.0, float(min_step_sec))
    if step <= 0:
        return ordered

    result = [ordered[0]]
    last_kept = ordered[0]
    for timestamp in ordered[1:]:
        if (timestamp - last_kept).total_seconds() >= step:
            result.append(timestamp)
            last_kept = timestamp
    if result[-1] != ordered[-1]:
        result.append(ordered[-1])
    return result


def resolve_effective_aia_max_pixels(
    *,
    max_pixels: int,
    playback_aia_max_pixels: int,
    playing: bool,
) -> int:
    """Use a small AIA heatmap during playback and full resolution while paused."""

    full = min(2048, max(256, int(max_pixels)))
    preview = min(1024, max(128, int(playback_aia_max_pixels)))
    return min(full, preview) if playing else full


def playback_interval_seconds(fps: float, *, playing: bool) -> float | None:
    """Return Streamlit fragment interval in seconds for playback."""

    if not playing:
        return None
    return max(0.05, 1.0 / max(0.2, float(fps)))


def default_frequency_marker_symbols(
    freqs: list[float],
    saved: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return stable frequency marker symbols, with saved choices preferred."""

    saved_symbols = normalize_marker_symbol_by_frequency(saved)
    result: dict[str, str] = {}
    for index, freq in enumerate(freqs):
        key = frequency_marker_key(freq)
        default_symbol = DEFAULT_MARKER_SYMBOL_CYCLE[
            index % len(DEFAULT_MARKER_SYMBOL_CYCLE)
        ]
        result[key] = saved_symbols.get(key, default_symbol)
    return result


def slider_with_step_buttons(
    st,
    label: str,
    min_value,
    max_value,
    value,
    step,
    *,
    key: str,
    help: str | None = None,
):
    """Render a Streamlit slider with adjacent fine-adjust buttons."""

    integer_slider = all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in (min_value, max_value, value, step)
    )

    def normalize(raw_value):
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            numeric = float(value)
        numeric = min(float(max_value), max(float(min_value), numeric))
        return int(round(numeric)) if integer_slider else numeric

    state_key = f"{key}_value"
    st.session_state[state_key] = normalize(st.session_state.get(state_key, value))
    minus_col, slider_col, plus_col = st.columns([0.14, 0.72, 0.14])
    with minus_col:
        if st.button(
            "-",
            key=f"{key}_minus",
            width="stretch",
            help=f"Decrease {label}",
        ):
            st.session_state[state_key] = normalize(
                float(st.session_state[state_key]) - float(step)
            )
    with plus_col:
        if st.button(
            "+",
            key=f"{key}_plus",
            width="stretch",
            help=f"Increase {label}",
        ):
            st.session_state[state_key] = normalize(
                float(st.session_state[state_key]) + float(step)
            )
    with slider_col:
        return st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=state_key,
            help=help,
        )


def aia_background_to_png_image(background: AiaBackground) -> dict[str, object]:
    """Convert a scaled AIA background into a browser-friendly PNG image."""

    from PIL import Image

    image = Image.fromarray(
        apply_aia_colormap_to_uint8(background.z, background.wavelength)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )
    return {
        "source": source,
        "x0": float(np.nanmin(background.x_arcsec)),
        "x1": float(np.nanmax(background.x_arcsec)),
        "y0": float(np.nanmin(background.y_arcsec)),
        "y1": float(np.nanmax(background.y_arcsec)),
        "label": str(background.label),
        "path": str(background.path),
    }


def build_preloaded_playback_payload(
    centers: pd.DataFrame,
    times: list[Any],
    *,
    frame_mode: str,
    tail_n: int,
    aia_table: pd.DataFrame | None = None,
    use_aia: bool = False,
    max_aia_dt_sec: float = 3600.0,
    playback_aia_max_pixels: int = DEFAULT_PLAYBACK_AIA_MAX_PIXELS,
    percentile_limits: tuple[float, float] = (1.0, 99.7),
    log_scale: bool = True,
    wcs_mode: str = "header",
    theme_mode: str = "auto",
    screen_fit: str = "auto",
    draw_lines: bool = True,
    fps: float = 2.0,
    plot_layout: str = "overlay",
    facet_by: str = "freq_mhz",
    marker_size: int = DEFAULT_MARKER_SIZE,
    marker_symbol_by_freq: dict[str, str] | None = None,
    trail_min_opacity: float = DEFAULT_TRAIL_MIN_OPACITY,
    link_views: bool = False,
    background_loader=None,
) -> dict[str, object]:
    """Build compact data for browser-side playback without per-frame figures."""

    normalized_times = [pd.Timestamp(value) for value in times]
    resolved_plot_layout = _choice_value(PLOT_LAYOUTS, plot_layout, "overlay")
    resolved_facet_by = _choice_value(FACET_BY_OPTIONS, facet_by, "freq_mhz")
    groups, trace_times_ns, facets = _build_preloaded_trace_groups(
        centers,
        facet_by=resolved_facet_by,
        plot_layout=resolved_plot_layout,
        marker_symbol_by_freq=marker_symbol_by_freq,
    )
    backgrounds: dict[str, dict[str, object]] = {}
    background_key_by_path: dict[str, str] = {}
    frames: list[dict[str, object]] = []
    background_loader = background_loader or _default_preloaded_background_loader

    for frame_time in normalized_times:
        background_key = None
        if use_aia and aia_table is not None and not aia_table.empty:
            nearest = find_nearest_aia(
                aia_table,
                frame_time,
                max_dt_seconds=float(max_aia_dt_sec),
            )
            if nearest.status == "matched" and nearest.path:
                background_key = background_key_by_path.get(nearest.path)
                if background_key is None:
                    background_key = f"bg{len(background_key_by_path)}"
                    background_key_by_path[nearest.path] = background_key
                    background = background_loader(
                        nearest.path,
                        max_pixels=int(playback_aia_max_pixels),
                        percentile_limits=tuple(percentile_limits),
                        log_scale=bool(log_scale),
                        wcs_mode=wcs_mode,
                    )
                    image = aia_background_to_png_image(background)
                    image["key"] = background_key
                    image["delta_seconds"] = float(nearest.delta_seconds or 0.0)
                    backgrounds[background_key] = image
        frames.append(
            {
                "time": frame_time.isoformat(),
                "groups": _preloaded_frame_groups(
                    trace_times_ns,
                    frame_time,
                    mode=frame_mode,
                    tail_n=int(tail_n),
                ),
                "background": background_key,
            }
        )

    axis = _preloaded_axis_extent(groups, backgrounds)
    return {
        "version": 1,
        "frames": frames,
        "traces": groups,
        "backgrounds": backgrounds,
        "layout": {
            "height": _preloaded_height(
                screen_fit,
                plot_layout=resolved_plot_layout,
                facet_count=len(facets),
            ),
            "title": "Radio source trajectory",
            "theme_mode": str(theme_mode or "auto"),
            "theme": _resolve_preloaded_theme(theme_mode),
            "themes": {
                "light": _resolve_preloaded_theme("light"),
                "dark": _resolve_preloaded_theme("dark"),
            },
            "axis": axis,
            "plot_layout": resolved_plot_layout,
            "facet_by": resolved_facet_by,
            "facets": facets,
            "facet_spacing": dict(PRELOADED_FACET_SPACING),
        },
        "config": {
            "fps": float(fps),
            "draw_lines": bool(draw_lines),
            "marker_size": max(1, int(marker_size)),
            "trail_min_opacity": _coerce_unit_interval(
                trail_min_opacity,
                DEFAULT_TRAIL_MIN_OPACITY,
            ),
            "link_views": bool(link_views),
        },
        "stats": {
            "frame_count": len(frames),
            "trace_count": len(groups),
            "background_count": len(backgrounds),
        },
    }


@lru_cache(maxsize=1)
def bundled_browser_media_scripts() -> tuple[str, str]:
    """Return the bundled Mediabunny and shared recorder scripts."""

    return (
        read_asset_text("mediabunny-1.50.8.cjs"),
        read_asset_text("browser_media.js"),
    )


def _escape_inline_script(source: str) -> str:
    return str(source).replace("</script", "<\\/script")


def build_preloaded_playback_html(
    payload: dict[str, object],
    *,
    plotly_js: str | None = None,
    mediabunny_js: str | None = None,
    browser_media_js: str | None = None,
) -> str:
    """Return standalone HTML for the browser-side preloaded player."""

    if plotly_js is None:
        from plotly.offline import get_plotlyjs

        plotly_js = get_plotlyjs()
    if mediabunny_js is None or browser_media_js is None:
        bundled_mediabunny, bundled_browser_media = bundled_browser_media_scripts()
        mediabunny_js = mediabunny_js or bundled_mediabunny
        browser_media_js = browser_media_js or bundled_browser_media
    plotly_js = _escape_inline_script(plotly_js)
    mediabunny_js = _escape_inline_script(mediabunny_js)
    browser_media_js = _escape_inline_script(browser_media_js)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  margin: 0;
  padding: 0;
  font-family: Arial, sans-serif;
  background: transparent;
}}
.radio-player {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.radio-toolbar {{
  display: grid;
  grid-template-columns: 72px 72px 1fr 96px 96px 96px 96px 96px 150px;
  gap: 8px;
  align-items: center;
}}
.radio-toolbar button,
.radio-toolbar select,
.radio-toolbar a {{
  min-height: 34px;
  border: 1px solid #9ca3af;
  border-radius: 6px;
  background: #f8fafc;
  color: #111827;
  text-align: center;
  text-decoration: none;
  line-height: 32px;
  box-sizing: border-box;
}}
.radio-toolbar button:disabled,
.radio-toolbar a[aria-disabled="true"] {{
  opacity: 0.55;
  cursor: not-allowed;
}}
.radio-toolbar input[type="range"] {{
  width: 100%;
}}
.radio-status {{
  color: #475569;
  font-size: 13px;
  min-width: 0;
  overflow-wrap: anywhere;
  white-space: normal;
}}
#radio-preloaded-plot {{
  width: 100%;
  height: {int(payload.get("layout", {}).get("height", 760))}px;
}}
</style>
<script>{plotly_js}</script>
<script>{mediabunny_js}</script>
<script>{browser_media_js}</script>
</head>
<body>
<div class="radio-player">
  <div class="radio-toolbar">
    <button id="radio-play" type="button" title="Play or pause smooth playback." aria-label="Play or pause smooth playback">Play</button>
    <button id="radio-prev" type="button" title="Step back by one playback frame." aria-label="Previous playback frame">Prev</button>
    <input id="radio-progress" type="range" min="0" max="0" value="0" step="1" title="Jump to a playback frame." aria-label="Playback frame index">
    <select id="radio-speed" title="Playback speed multiplier." aria-label="Playback speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <button id="radio-reset-view" type="button" title="Reset all plot axes to the initial data range." aria-label="Reset all plot axes">Reset</button>
    <button id="radio-sync-view" type="button" title="Synchronize all facet plots to the last adjusted view." aria-label="Synchronize facet plot views">Sync</button>
    <button id="radio-link-views" type="button" title="Toggle live synchronization between facet plot views." aria-label="Toggle linked facet views">Link</button>
    <button id="radio-record" type="button" title="Record the selected frame range." aria-label="Record selected frame range">Record</button>
    <a id="radio-download" href="#" download="radio-source-trajectory.webm" aria-disabled="true" title="Download the completed browser recording." aria-label="Download browser recording">Download</a>
    <div id="radio-status" class="radio-status"></div>
  </div>
  <div id="radio-preloaded-plot"></div>
</div>
<script>
const radioPayload = {payload_json};
const plotEl = document.getElementById("radio-preloaded-plot");
const playEl = document.getElementById("radio-play");
const prevEl = document.getElementById("radio-prev");
const progressEl = document.getElementById("radio-progress");
const speedEl = document.getElementById("radio-speed");
const statusEl = document.getElementById("radio-status");
const recordEl = document.getElementById("radio-record");
const downloadEl = document.getElementById("radio-download");
const resetViewEl = document.getElementById("radio-reset-view");
const syncViewEl = document.getElementById("radio-sync-view");
const linkViewsEl = document.getElementById("radio-link-views");
const prefersDarkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
let frameIndex = 0;
let timer = null;
let plotInitialized = false;
let lastRelayoutAxis = 0;
let relayoutGuard = false;
let linkViews = Boolean(radioPayload.config.link_views);
let recordingSession = null;
let recordingActive = false;
let recordingCancelRequested = false;
let recordingDownloadUrl = null;

function activeTheme() {{
  const themes = radioPayload.layout.themes || {{}};
  const mode = radioPayload.layout.theme_mode || "light";
  if (mode === "dark") return themes.dark || radioPayload.layout.theme;
  if (mode === "auto" && prefersDarkQuery && prefersDarkQuery.matches) {{
    return themes.dark || radioPayload.layout.theme;
  }}
  return themes.light || radioPayload.layout.theme;
}}

function facetAxisName(prefix, facetIndex) {{
  const suffix = facetIndex <= 0 ? "" : String(facetIndex + 1);
  return prefix + suffix;
}}

function facetGrid() {{
  const facets = radioPayload.layout.facets || [];
  const enabled = radioPayload.layout.plot_layout === "facets" && facets.length > 0;
  const columns = enabled ? Math.min(3, Math.max(1, facets.length)) : 1;
  const rows = enabled ? Math.ceil(facets.length / columns) : 1;
  return {{enabled, facets, columns, rows}};
}}

function facetSpacing() {{
  const spacing = radioPayload.layout.facet_spacing || {{}};
  return {{
    x: Math.min(0.18, Math.max(0.02, Number(spacing.x || 0.075))),
    y: Math.min(0.22, Math.max(0.04, Number(spacing.y || 0.13)))
  }};
}}

function facetDomain(index, grid) {{
  if (!grid.enabled) return {{x: [0, 1], y: [0, 1]}};
  const spacing = facetSpacing();
  const row = Math.floor(index / grid.columns);
  const col = index % grid.columns;
  const xWidth = (1 - spacing.x * Math.max(0, grid.columns - 1)) / grid.columns;
  const yHeight = (1 - spacing.y * Math.max(0, grid.rows - 1)) / grid.rows;
  const x0 = col * (xWidth + spacing.x);
  const x1 = x0 + xWidth;
  const y1 = 1 - row * (yHeight + spacing.y);
  const y0 = y1 - yHeight;
  return {{x: [x0, x1], y: [y0, y1]}};
}}

function aspectCorrectRange(x0, x1, y0, y1, domain) {{
  const width = Math.max(1, Number(plotEl.clientWidth || 960));
  const height = Math.max(1, Number(radioPayload.layout.height || 760));
  const xDomain = domain && domain.x ? domain.x : [0, 1];
  const yDomain = domain && domain.y ? domain.y : [0, 1];
  const pixelWidth = Math.max(1, width * Math.max(0.01, xDomain[1] - xDomain[0]));
  const pixelHeight = Math.max(1, height * Math.max(0.01, yDomain[1] - yDomain[0]));
  const pixelAspect = pixelWidth / pixelHeight;
  const xSpan = Math.max(1e-9, Number(x1) - Number(x0));
  const ySpan = Math.max(1e-9, Number(y1) - Number(y0));
  const xMid = (Number(x0) + Number(x1)) / 2;
  const yMid = (Number(y0) + Number(y1)) / 2;
  if (!Number.isFinite(pixelAspect) || pixelAspect <= 0) {{
    return {{x: [x0, x1], y: [y0, y1]}};
  }}
  if ((xSpan / ySpan) > pixelAspect) {{
    const targetYSpan = xSpan / pixelAspect;
    return {{x: [x0, x1], y: [yMid - targetYSpan / 2, yMid + targetYSpan / 2]}};
  }}
  const targetXSpan = ySpan * pixelAspect;
  return {{x: [xMid - targetXSpan / 2, xMid + targetXSpan / 2], y: [y0, y1]}};
}}

function aspectCorrectedAxis(domain) {{
  const axis = radioPayload.layout.axis || {{x0: -1, x1: 1, y0: -1, y1: 1}};
  return aspectCorrectRange(axis.x0, axis.x1, axis.y0, axis.y1, domain);
}}

function subplotCount() {{
  const grid = facetGrid();
  return grid.enabled ? grid.facets.length : 1;
}}

function relayoutAllAxes(xRange, yRange) {{
  const update = {{}};
  const grid = facetGrid();
  for (let i = 0; i < subplotCount(); i += 1) {{
    const corrected = aspectCorrectRange(
      xRange[0],
      xRange[1],
      yRange[0],
      yRange[1],
      facetDomain(i, grid)
    );
    update[`${{facetAxisName("xaxis", i)}}.range`] = corrected.x;
    update[`${{facetAxisName("yaxis", i)}}.range`] = corrected.y;
  }}
  relayoutGuard = true;
  return Plotly.relayout(plotEl, update).finally(() => {{
    relayoutGuard = false;
  }});
}}

function resetView() {{
  const axis = radioPayload.layout.axis;
  return relayoutAllAxes([axis.x0, axis.x1], [axis.y0, axis.y1]);
}}

function currentRanges(facetIndex) {{
  const layout = plotEl._fullLayout || {{}};
  const xAxis = layout[facetAxisName("xaxis", facetIndex)] || {{}};
  const yAxis = layout[facetAxisName("yaxis", facetIndex)] || {{}};
  return {{
    x: xAxis.range || [radioPayload.layout.axis.x0, radioPayload.layout.axis.x1],
    y: yAxis.range || [radioPayload.layout.axis.y0, radioPayload.layout.axis.y1]
  }};
}}

function syncViews() {{
  const ranges = currentRanges(lastRelayoutAxis);
  return relayoutAllAxes(ranges.x, ranges.y);
}}

function relayoutFacetIndex(eventData) {{
  for (const key of Object.keys(eventData || {{}})) {{
    const match = key.match(/^[xy]axis(\\d*)\\./);
    if (match) {{
      return match[1] ? Number(match[1]) - 1 : 0;
    }}
  }}
  return lastRelayoutAxis;
}}

function updateLinkButton() {{
  linkViewsEl.textContent = linkViews ? "Link On" : "Link Off";
}}

function buildLayout(frame) {{
  const theme = activeTheme();
  const axis = radioPayload.layout.axis;
  const grid = facetGrid();
  const bg = frame.background ? radioPayload.backgrounds[frame.background] : null;
  const images = [];
  const layout = {{
    title: `${{radioPayload.layout.title}} | ${{frame.time}}`,
    height: radioPayload.layout.height,
    margin: {{l: 44, r: 22, t: grid.enabled ? 78 : 54, b: 42}},
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: {{color: theme.font_color}},
    images,
    annotations: [{{
      text: `Radio source time: ${{frame.time}}`,
      showarrow: false,
      xref: "paper",
      yref: "paper",
      x: 1,
      y: 1.08,
      xanchor: "right",
      yanchor: "bottom",
      font: {{color: theme.font_color, size: 13}}
    }}],
    legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0}},
    uirevision: "radio-preloaded"
  }};
  const count = grid.enabled ? grid.facets.length : 1;
  for (let i = 0; i < count; i += 1) {{
    const xaxis = facetAxisName("xaxis", i);
    const yaxis = facetAxisName("yaxis", i);
    const xref = facetAxisName("x", i);
    const yref = facetAxisName("y", i);
    const domain = facetDomain(i, grid);
    const corrected = aspectCorrectedAxis(domain);
    layout[xaxis] = {{
      title: "HPLN / arcsec",
      range: corrected.x,
      domain: domain.x,
      gridcolor: theme.grid_color,
      zerolinecolor: theme.grid_color
    }};
    layout[yaxis] = {{
      title: "HPLT / arcsec",
      range: corrected.y,
      domain: domain.y,
      gridcolor: theme.grid_color,
      zerolinecolor: theme.grid_color,
      scaleanchor: xref,
      scaleratio: 1
    }};
    if (bg) {{
      images.push({{
        source: bg.source,
        xref,
        yref,
        x: bg.x0,
        y: bg.y1,
        sizex: bg.x1 - bg.x0,
        sizey: bg.y1 - bg.y0,
        sizing: "stretch",
        opacity: 1,
        layer: "below"
      }});
    }}
    if (grid.enabled) {{
      layout.annotations.push({{
        text: grid.facets[i].label,
        showarrow: false,
        xref: "paper",
        yref: "paper",
        x: (domain.x[0] + domain.x[1]) / 2,
        y: Math.min(1.04, domain.y[1] + 0.025),
        font: {{color: theme.font_color, size: 13}}
      }});
    }}
  }}
  return layout;
}}

function frameData(frame) {{
  const grid = facetGrid();
  return frame.groups.map((group) => {{
    const trace = radioPayload.traces[group.trace];
    const facetIndex = grid.enabled ? Number(trace.facet || 0) : 0;
    const count = Math.max(0, group.end - group.start);
    const item = {{
      type: "scattergl",
      mode: radioPayload.config.draw_lines ? "lines+markers" : "markers",
      name: trace.name,
      x: trace.x.slice(group.start, group.end),
      y: trace.y.slice(group.start, group.end),
      marker: {{
        size: Math.max(1, Number(radioPayload.config.marker_size || 8)),
        symbol: trace.marker_symbol || "circle",
        opacity: opacityForPoints(count)
      }},
      line: {{width: 2}},
      hovertemplate: trace.name + "<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>"
    }};
    if (grid.enabled) {{
      item.xaxis = facetAxisName("x", facetIndex);
      item.yaxis = facetAxisName("y", facetIndex);
    }}
    return item;
  }});
}}

function opacityForPoints(count) {{
  if (count <= 0) return [];
  if (count === 1) return [1.0];
  const minimum = Math.min(1, Math.max(0, Number(radioPayload.config.trail_min_opacity || 0.25)));
  const step = (1.0 - minimum) / Math.max(1, count - 1);
  return Array.from({{length: count}}, (_, index) => minimum + step * index);
}}

function renderFrame(index) {{
  frameIndex = Math.max(0, Math.min(index, radioPayload.frames.length - 1));
  const frame = radioPayload.frames[frameIndex];
  progressEl.value = String(frameIndex);
  statusEl.textContent = `${{frameIndex + 1}} / ${{radioPayload.frames.length}}  ${{frame.time}}`;
  const data = frameData(frame);
  const layout = buildLayout(frame);
  const config = {{
    responsive: true,
    displaylogo: false
  }};
  if (!plotInitialized) {{
    plotInitialized = true;
    return Plotly.newPlot(plotEl, data, layout, config);
  }} else {{
    return Plotly.react(plotEl, data, layout, config);
  }}
}}

function intervalMs() {{
  const baseFps = Math.max(0.2, Number(radioPayload.config.fps || 2));
  const speed = Math.max(0.1, Number(speedEl.value || 1));
  return Math.max(25, 1000 / (baseFps * speed));
}}

function stopPlayback() {{
  if (timer !== null) {{
    clearInterval(timer);
    timer = null;
  }}
  playEl.textContent = "Play";
}}

function startPlayback() {{
  stopPlayback();
  playEl.textContent = "Pause";
  timer = setInterval(() => {{
    renderFrame((frameIndex + 1) % radioPayload.frames.length);
  }}, intervalMs());
}}

async function blobToImage(blob) {{
  const url = URL.createObjectURL(blob);
  try {{
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  }} finally {{
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}
}}

async function dataUrlToImage(dataUrl) {{
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  if (window.createImageBitmap) {{
    return createImageBitmap(blob);
  }}
  return blobToImage(blob);
}}

function drawRecordingFrame(ctx, image, width, height) {{
  ctx.fillStyle = activeTheme().paper_bgcolor || "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const sourceWidth = Math.max(1, Number(image.naturalWidth || image.width || width));
  const sourceHeight = Math.max(1, Number(image.naturalHeight || image.height || height));
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const drawWidth = Math.max(1, Math.round(sourceWidth * scale));
  const drawHeight = Math.max(1, Math.round(sourceHeight * scale));
  const offsetX = Math.floor((width - drawWidth) / 2);
  const offsetY = Math.floor((height - drawHeight) / 2);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
}}

function recordingOptions() {{
  const configured = radioPayload.config.recording || {{}};
  const total = radioPayload.frames.length;
  const startFrame = Math.max(0, Math.min(total, Math.floor(Number(configured.start_frame || 0))));
  const requestedEnd = Number.isFinite(Number(configured.end_frame))
    ? Math.floor(Number(configured.end_frame))
    : total;
  const endFrame = Math.max(startFrame, Math.min(total, requestedEnd));
  const normalizedFormat = SolarToolkitMedia.normalizeOutputFormat(configured.format, "webm");
  const format = normalizedFormat === "mp4" ? "mp4" : "webm";
  const size = SolarToolkitMedia.normalizeEvenSize(
    configured.width || plotEl.clientWidth || 960,
    configured.height || radioPayload.layout.height || 760,
    320,
    240
  );
  return {{
    startFrame,
    endFrame,
    frameCount: Math.max(0, endFrame - startFrame),
    fps: Math.max(0.2, Number(configured.fps || radioPayload.config.fps || 2)),
    quality: SolarToolkitMedia.normalizeQuality(configured.quality, "high"),
    format,
    width: size.width,
    height: size.height
  }};
}}

function setRecordingControls(active) {{
  for (const control of [playEl, prevEl, progressEl, speedEl, resetViewEl, syncViewEl, linkViewsEl]) {{
    control.disabled = active;
  }}
}}

function clearRecordingDownload() {{
  if (recordingDownloadUrl) URL.revokeObjectURL(recordingDownloadUrl);
  recordingDownloadUrl = null;
  downloadEl.removeAttribute("href");
  downloadEl.setAttribute("aria-disabled", "true");
  downloadEl.textContent = "Download";
}}

async function cancelRangeRecording() {{
  if (!recordingActive) return;
  recordingCancelRequested = true;
  recordEl.disabled = true;
  recordEl.textContent = "Canceling...";
  if (recordingSession) {{
    try {{
      await recordingSession.cancel();
    }} catch (_cancelError) {{}}
  }}
}}

async function recordSelectedRange() {{
  if (recordingActive) {{
    await cancelRangeRecording();
    return;
  }}
  if (!window.SolarToolkitMedia || !window.Mediabunny) {{
    statusEl.textContent = "Deterministic browser recording is unavailable.";
    return;
  }}
  const options = recordingOptions();
  if (options.frameCount <= 0) {{
    statusEl.textContent = "The selected recording range is empty.";
    return;
  }}

  stopPlayback();
  recordingActive = true;
  const originalFrame = frameIndex;
  const canvas = document.createElement("canvas");
  canvas.width = options.width;
  canvas.height = options.height;
  const ctx = canvas.getContext("2d", {{alpha: false}});
  if (!ctx) {{
    statusEl.textContent = "Could not create the recording canvas.";
    return;
  }}

  clearRecordingDownload();
  recordingCancelRequested = false;
  setRecordingControls(true);
  recordEl.textContent = "Cancel";
  let finalMessage = "Recording canceled.";
  try {{
    recordingSession = await SolarToolkitMedia.createCanvasRecorder({{
      canvas,
      format: options.format,
      quality: options.quality,
      fps: options.fps,
      targetMode: "buffer",
      contentHint: "detail"
    }});
    for (let index = options.startFrame; index < options.endFrame; index += 1) {{
      if (recordingCancelRequested) throw new DOMException("Recording canceled.", "AbortError");
      await renderFrame(index);
      const dataUrl = await Plotly.toImage(plotEl, {{
        format: "png",
        width: Math.max(320, Math.round(plotEl.clientWidth || 960)),
        height: Math.max(240, Number(radioPayload.layout.height || 760)),
        scale: 1
      }});
      const image = await dataUrlToImage(dataUrl);
      try {{
        drawRecordingFrame(ctx, image, options.width, options.height);
        const offset = index - options.startFrame;
        await recordingSession.addFrame(offset / options.fps, 1 / options.fps, {{
          keyFrame: offset === 0
        }});
        statusEl.textContent = `Encoding ${{offset + 1}} / ${{options.frameCount}}`;
      }} finally {{
        if (image && typeof image.close === "function") image.close();
      }}
    }}
    const result = await recordingSession.finalize();
    if (recordingCancelRequested) throw new DOMException("Recording canceled.", "AbortError");
    if (!(result.buffer instanceof ArrayBuffer) || result.buffer.byteLength === 0) {{
      throw new Error("The browser encoder produced an empty recording.");
    }}
    const mimeType = options.format === "mp4" ? "video/mp4" : "video/webm";
    const blob = new Blob([result.buffer], {{type: mimeType}});
    recordingDownloadUrl = URL.createObjectURL(blob);
    downloadEl.href = recordingDownloadUrl;
    downloadEl.download = `radio-source-trajectory-${{options.startFrame + 1}}-${{options.endFrame}}.${{options.format}}`;
    downloadEl.textContent = options.format.toUpperCase();
    downloadEl.setAttribute("aria-disabled", "false");
    finalMessage = `${{options.format.toUpperCase()}} ready: ${{options.frameCount}} frames, ${{result.codec}}, ${{(blob.size / 1024 / 1024).toFixed(2)}} MB`;
  }} catch (error) {{
    if (!recordingCancelRequested) {{
      const detail = error && error.message ? error.message : String(error);
      finalMessage = `Browser recording failed: ${{detail}}`;
    }}
  }} finally {{
    const session = recordingSession;
    recordingSession = null;
    if (session && !["canceled", "finalized"].includes(session.state)) {{
      try {{
        await session.cancel();
      }} catch (_cleanupError) {{}}
    }}
    recordingActive = false;
    recordingCancelRequested = false;
    recordEl.disabled = false;
    recordEl.textContent = "Record";
    setRecordingControls(false);
    try {{
      await renderFrame(originalFrame);
    }} finally {{
      statusEl.textContent = finalMessage;
    }}
  }}
}}

playEl.addEventListener("click", () => {{
  if (timer === null) startPlayback();
  else stopPlayback();
}});
prevEl.addEventListener("click", () => renderFrame(frameIndex - 1));
progressEl.addEventListener("input", () => renderFrame(Number(progressEl.value)));
speedEl.addEventListener("change", () => {{
  if (timer !== null) startPlayback();
}});
recordEl.addEventListener("click", () => void recordSelectedRange());
downloadEl.addEventListener("click", (event) => {{
  if (downloadEl.getAttribute("aria-disabled") === "true") event.preventDefault();
}});
window.addEventListener("beforeunload", () => {{
  if (recordingSession) void recordingSession.cancel();
  if (recordingDownloadUrl) URL.revokeObjectURL(recordingDownloadUrl);
}});
resetViewEl.addEventListener("click", () => resetView());
syncViewEl.addEventListener("click", () => syncViews());
linkViewsEl.addEventListener("click", () => {{
  linkViews = !linkViews;
  updateLinkButton();
}});
if (prefersDarkQuery) {{
  prefersDarkQuery.addEventListener("change", () => renderFrame(frameIndex));
}}

progressEl.max = String(Math.max(0, radioPayload.frames.length - 1));
updateLinkButton();
renderFrame(0).then(() => {{
  plotEl.on("plotly_relayout", (eventData) => {{
    if (relayoutGuard) return;
    lastRelayoutAxis = relayoutFacetIndex(eventData);
    if (linkViews && facetGrid().enabled) {{
      syncViews();
    }}
  }});
}});
</script>
</body>
</html>"""


def _default_preloaded_background_loader(path: str, **kwargs) -> AiaBackground:
    return read_aia_background(path, **kwargs)


def _build_preloaded_trace_groups(
    centers: pd.DataFrame,
    *,
    facet_by: str = "freq_mhz",
    plot_layout: str = "overlay",
    marker_symbol_by_freq: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], list[list[int]], list[dict[str, object]]]:
    if centers is None or centers.empty:
        return [], [], []
    groups: list[dict[str, object]] = []
    group_times_ns: list[list[int]] = []
    resolved_facet_by = _choice_value(FACET_BY_OPTIONS, facet_by, "freq_mhz")
    resolved_plot_layout = _choice_value(PLOT_LAYOUTS, plot_layout, "overlay")
    marker_symbols = normalize_marker_symbol_by_frequency(marker_symbol_by_freq)
    facet_values = _preloaded_facet_values(centers, resolved_facet_by)
    facet_lookup = {str(value): index for index, value in enumerate(facet_values)}
    facets = [
        {
            "index": index,
            "value": _jsonable_facet_value(value),
            "label": _format_preloaded_facet_label(resolved_facet_by, value),
        }
        for index, value in enumerate(facet_values)
    ]
    group_columns = ["freq_mhz", "polarization", "center_method"]
    for (freq, pol, method), group in centers.groupby(group_columns, sort=True):
        ordered = group.sort_values("obs_time")
        times = [pd.Timestamp(value) for value in ordered["obs_time"].tolist()]
        facet_value = (
            ordered.iloc[0][resolved_facet_by]
            if resolved_plot_layout == "facets" and resolved_facet_by in ordered.columns
            else None
        )
        groups.append(
            {
                "name": f"{float(freq):.3g} MHz | {pol} | {method}",
                "freq_mhz": float(freq),
                "polarization": str(pol),
                "center_method": str(method),
                "marker_symbol": marker_symbol_for_frequency(freq, marker_symbols),
                "facet": int(facet_lookup.get(str(facet_value), 0)),
                "facet_value": _jsonable_facet_value(facet_value),
                "x": [float(value) for value in ordered["center_x_arcsec"].tolist()],
                "y": [float(value) for value in ordered["center_y_arcsec"].tolist()],
                "time": [value.isoformat() for value in times],
            }
        )
        group_times_ns.append([int(value.value) for value in times])
    return groups, group_times_ns, facets if resolved_plot_layout == "facets" else []


def _preloaded_facet_values(centers: pd.DataFrame, facet_by: str) -> list[object]:
    if facet_by not in centers.columns:
        return []
    values = centers[facet_by].dropna().unique().tolist()
    if facet_by == "freq_mhz":
        return sorted(values, key=lambda value: float(value))
    return sorted(values, key=lambda value: str(value))


def _jsonable_facet_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _format_preloaded_facet_label(facet_by: str, value: object) -> str:
    if facet_by == "freq_mhz":
        return f"{float(value):.3g} MHz"
    if facet_by == "polarization":
        return f"Polarization: {value}"
    if facet_by == "center_method":
        return f"Method: {value}"
    return str(value)


def _preloaded_frame_groups(
    trace_times_ns: list[list[int]],
    frame_time: pd.Timestamp,
    *,
    mode: str,
    tail_n: int,
) -> list[dict[str, int]]:
    frame_ns = int(pd.Timestamp(frame_time).value)
    resolved_mode = str(mode or FRAME_MODE_ALL).lower()
    result: list[dict[str, int]] = []
    for trace_index, times_ns in enumerate(trace_times_ns):
        end = bisect.bisect_right(times_ns, frame_ns)
        if end <= 0:
            continue
        if resolved_mode == "current":
            start = end - 1
        elif resolved_mode == "tail":
            start = max(0, end - max(1, int(tail_n)))
        else:
            start = 0
        result.append({"trace": trace_index, "start": int(start), "end": int(end)})
    return result


def _preloaded_axis_extent(
    traces: list[dict[str, object]],
    backgrounds: dict[str, dict[str, object]],
) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for trace in traces:
        xs.extend(float(value) for value in trace.get("x", []))
        ys.extend(float(value) for value in trace.get("y", []))
    for background in backgrounds.values():
        xs.extend([float(background["x0"]), float(background["x1"])])
        ys.extend([float(background["y0"]), float(background["y1"])])
    if not xs or not ys:
        return {"x0": -1.0, "x1": 1.0, "y0": -1.0, "y1": 1.0}
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_pad = max(1.0, (x1 - x0) * 0.04)
    y_pad = max(1.0, (y1 - y0) * 0.04)
    return {
        "x0": float(x0 - x_pad),
        "x1": float(x1 + x_pad),
        "y0": float(y0 - y_pad),
        "y1": float(y1 + y_pad),
    }


def _resolve_preloaded_theme(theme_mode: str) -> dict[str, str]:
    return resolve_theme_palette(theme_mode)


def _preloaded_height(
    screen_fit: str,
    *,
    plot_layout: str = "overlay",
    facet_count: int = 0,
) -> int:
    mode = str(screen_fit or "auto").lower()
    if mode == "portrait":
        base_height = 920
        row_height = 330
        extra = 180
    elif mode == "landscape":
        base_height = 640
        row_height = 250
        extra = 150
    else:
        base_height = 760
        row_height = 290
        extra = 170
    if str(plot_layout or "overlay") == "facets" and int(facet_count) > 0:
        rows = int(np.ceil(max(1, int(facet_count)) / 3))
        return max(base_height, rows * row_height + extra)
    return base_height


def _preloaded_payload_size_mb(payload: dict[str, object]) -> float:
    raw = json.dumps(payload, ensure_ascii=False)
    return len(raw.encode("utf-8")) / 1024 / 1024


def build_preload_signature(values: dict[str, Any]) -> str:
    """Build a deterministic signature for settings that affect preloaded playback."""

    raw = json.dumps(
        _jsonable_signature_value(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def should_build_preloaded_payload(
    current_signature: str,
    applied_signature: str | None,
    *,
    apply_clicked: bool,
    has_payload: bool,
) -> bool:
    """Return whether this rerun should construct the heavy preloaded payload."""

    if apply_clicked:
        return True
    if not has_payload and not applied_signature:
        return True
    return bool(
        applied_signature
        and current_signature == applied_signature
        and not bool(has_payload)
    )


def preload_settings_changed(
    current_signature: str,
    applied_signature: str | None,
    *,
    has_payload: bool,
) -> bool:
    """Return whether visible settings differ from the currently shown payload."""

    return bool(
        has_payload and applied_signature and current_signature != applied_signature
    )


def _jsonable_signature_value(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable_signature_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_signature_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _advance_playback_index_if_due(
    session_state,
    *,
    frame_count: int,
    interval_sec: float,
) -> None:
    """Advance playback at most once per interval."""

    if frame_count <= 0:
        return
    now = time.monotonic()
    last_tick = float(session_state.get("radio_source_last_tick", 0.0))
    if now - last_tick < max(0.01, float(interval_sec)) * 0.8:
        return
    session_state.radio_source_frame_idx = (
        int(session_state.radio_source_frame_idx) + 1
    ) % int(frame_count)
    session_state.radio_source_last_tick = now


def _run_streamlit_app(argv: list[str] | None = None) -> None:
    """Run the Streamlit app. The app reads existing center tables only."""

    args = build_parser().parse_args(argv)
    settings_file = (
        Path(args.settings_file).expanduser()
        if args.settings_file
        else default_settings_path()
    )
    layout = RuntimeLayout.discover()
    ui_store = frontend_state_store("source-trajectory", layout=layout)
    ui_snapshot = ui_store.load(default={})
    import_legacy = not bool(ui_snapshot.get("legacy_imported"))
    settings = resolve_app_settings(
        args,
        (
            load_app_settings(settings_file, reset=bool(args.reset_settings))
            if import_legacy
            else {}
        ),
    )
    explicit_aia_dir = getattr(args, "aia_dir", None) is not None

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise SystemExit(
            "Streamlit is required for this frontend. Install with: "
            'python -m pip install -e ".[app]"'
        ) from exc

    st.set_page_config(
        page_title="Radio Source Trajectory",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_root = layout.local_root
    try:
        allowed_roots = resolve_streamlit_allowed_roots(args.allowed_roots)
    except AllowedRootPolicyError as exc:
        allowed_roots = ()
        st.error(f"Path configuration error: {exc}")
    if not allowed_roots:
        st.error(
            "No valid allowed roots are configured. Path browsing and local path "
            "operations are disabled."
        )
    path_policy = PathAccessPolicy.create(
        allowed_roots,
        protected_output_roots=(layout.outputs_dir,),
        base_directory=local_root,
    )
    if import_legacy:
        ui_store.update({"theme": str(settings["theme_mode"])})
    theme_mode = render_streamlit_theme(
        st,
        frontend_id="source-trajectory",
        state_store=ui_store,
        path_memory=frontend_path_memory(path_policy.output_roots, layout=layout),
    )
    bind_streamlit_fields(
        st,
        ui_store,
        frontend_id="source-trajectory",
        field_keys=TRAJECTORY_UI_FIELD_KEYS,
    )
    ui_store.update({"legacy_imported": True})
    settings["theme_mode"] = theme_mode

    @st.cache_data(show_spinner=False)
    def _cached_load_centers(path_text: str, token: tuple[str, int | None, int | None]):
        del token
        return load_centers_table(path_text)

    @st.cache_data(show_spinner=False)
    def _cached_filter_time_range(raw: pd.DataFrame, start: str, end: str):
        return filter_time_range(
            raw,
            start=start.strip() or None,
            end=end.strip() or None,
        )

    @st.cache_data(show_spinner=False)
    def _cached_filter_centers(
        centers_df: pd.DataFrame,
        freqs: tuple[float, ...],
        polarizations: tuple[str, ...],
        methods: tuple[str, ...],
    ):
        return filter_centers(
            centers_df,
            freqs=list(freqs),
            polarizations=list(polarizations),
            center_methods=list(methods),
        )

    @st.cache_data(show_spinner=False)
    def _cached_scan_aia_folder(
        aia_dir_text: str,
        pattern: str,
        token: tuple[str, int | None, int | None],
    ):
        del token
        return scan_aia_folder(aia_dir_text, pattern=pattern)

    @st.cache_data(show_spinner=False)
    def _cached_read_aia_background(
        path_text: str,
        token: tuple[str, int | None, int | None],
        max_pixels_value: int,
        percentile_low: float,
        percentile_high: float,
        use_log_scale: bool,
        selected_wcs_mode: str,
    ):
        del token
        return read_aia_background(
            path_text,
            max_pixels=int(max_pixels_value),
            percentile_limits=(float(percentile_low), float(percentile_high)),
            log_scale=bool(use_log_scale),
            wcs_mode=selected_wcs_mode,
        )

    @st.cache_data(show_spinner=False)
    def _cached_preloaded_playback_payload(
        centers_df: pd.DataFrame,
        frame_time_iso_values: tuple[str, ...],
        display_mode: str,
        display_tail_n: int,
        aia_table_df: pd.DataFrame,
        display_use_aia: bool,
        aia_token: tuple[str, int | None, int | None],
        max_aia_delta_sec: float,
        preview_aia_max_pixels: int,
        percentile_low: float,
        percentile_high: float,
        use_log_scale: bool,
        selected_wcs_mode: str,
        selected_theme_mode: str,
        selected_screen_fit: str,
        selected_draw_lines: bool,
        selected_fps: float,
        selected_plot_layout: str,
        selected_facet_by: str,
        selected_marker_size: int,
        selected_marker_symbol_by_freq: dict[str, str],
        selected_trail_min_opacity: float,
        selected_link_views: bool,
    ):
        del aia_token
        return build_preloaded_playback_payload(
            centers_df,
            [pd.Timestamp(value) for value in frame_time_iso_values],
            frame_mode=display_mode,
            tail_n=int(display_tail_n),
            aia_table=aia_table_df,
            use_aia=bool(display_use_aia),
            max_aia_dt_sec=float(max_aia_delta_sec),
            playback_aia_max_pixels=int(preview_aia_max_pixels),
            percentile_limits=(float(percentile_low), float(percentile_high)),
            log_scale=bool(use_log_scale),
            wcs_mode=selected_wcs_mode,
            theme_mode=selected_theme_mode,
            screen_fit=selected_screen_fit,
            draw_lines=bool(selected_draw_lines),
            fps=float(selected_fps),
            plot_layout=selected_plot_layout,
            facet_by=selected_facet_by,
            marker_size=int(selected_marker_size),
            marker_symbol_by_freq=selected_marker_symbol_by_freq,
            trail_min_opacity=float(selected_trail_min_opacity),
            link_views=bool(selected_link_views),
        )

    @st.cache_data(show_spinner=False)
    def _cached_summarize_motion(centers_df: pd.DataFrame):
        return summarize_motion(centers_df)

    @st.cache_data(show_spinner=False)
    def _cached_visible_centers(
        centers_df: pd.DataFrame,
        frame_time_iso: str,
        display_mode: str,
        display_tail_n: int,
    ):
        return select_visible_centers(
            centers_df,
            pd.Timestamp(frame_time_iso),
            mode=display_mode,
            tail_n=int(display_tail_n),
        )

    st.title("Radio Source Center Trajectory with AIA Background")
    apply_preload_clicked = False

    with st.sidebar:
        st.header("Common Parameters")
        centers_path = render_native_path_input(
            st,
            "Center table CSV/XLSX",
            key="centers_path",
            initial_value=str(settings["centers"]),
            roots=path_policy.input_roots,
            kind="file",
            extensions=(".csv", ".xlsx", ".xls"),
            help_text="Path to the radio-source center table generated from FITS files.",
            frontend_id="source-trajectory",
            operation="load-centers",
            state_store=ui_store,
            stacked=True,
        )
        time_start = st.text_input(
            "Time window start",
            key="time_start",
            value=str(settings["time_start"]),
            help="Inclusive start time in ISO-8601 UTC form.",
        )
        time_end = st.text_input(
            "Time window end",
            key="time_end",
            value=str(settings["time_end"]),
            help="Inclusive end time in ISO-8601 UTC form.",
        )
        aia_dir = render_native_path_input(
            st,
            "AIA FITS folder",
            key="aia_dir",
            initial_value=str(settings["aia_dir"]),
            roots=path_policy.input_roots,
            kind="directory",
            help_text="Folder containing AIA FITS images for the background layer.",
            frontend_id="source-trajectory",
            operation="load-aia",
            state_store=ui_store,
            stacked=True,
        )
        screen_fit = st.radio(
            "Screen fit",
            SCREEN_FIT_MODES,
            key="screen_fit",
            index=_choice_index(SCREEN_FIT_MODES, settings["screen_fit"]),
            horizontal=True,
            help="Optimize plot height and legend placement for the current screen.",
        )

        st.header("Image Display")
        use_aia_default = bool(settings["use_aia"]) or (
            explicit_aia_dir and bool(aia_dir)
        )
        use_aia = st.checkbox(
            "Show nearest AIA background",
            key="use_aia",
            value=use_aia_default,
            help="Overlay the closest AIA image within the allowed time difference.",
        )
        max_pixels = slider_with_step_buttons(
            st,
            "AIA background max side / px",
            256,
            2048,
            value=int(settings["max_pixels"]),
            step=128,
            key="aia_background_max_side",
            help="Resolution used for paused/manual AIA viewing.",
        )
        percentile_limits = st.slider(
            "AIA display percentile clip",
            0.0,
            100.0,
            tuple(float(value) for value in settings["percentile_limits"]),
            key="percentile_limits",
            step=0.1,
            help="Lower and upper percentile limits used before AIA color mapping.",
        )
        log_scale = st.checkbox(
            "Use log10 AIA intensity",
            key="log_scale",
            value=bool(settings["log_scale"]),
            help="Apply log scaling after percentile clipping to reveal faint structure.",
        )
        playback_aia_max_pixels = slider_with_step_buttons(
            st,
            "Playback AIA preview max side / px",
            128,
            1024,
            value=int(settings["playback_aia_max_pixels"]),
            step=64,
            key="playback_aia_preview_max_side",
            help="Lower-resolution AIA preview used for smooth browser playback.",
        )
        fps = slider_with_step_buttons(
            st,
            "Playback FPS",
            min_value=0.2,
            max_value=20.0,
            value=float(settings["fps"]),
            step=0.2,
            key="playback_fps",
            help="Target playback frame rate for the browser player and fallback controls.",
        )
        mode_label_to_value = {label: key for key, label in FRAME_MODE_LABELS.items()}
        mode_values = list(FRAME_MODE_LABELS)
        default_mode = (
            str(settings["frame_mode"])
            if settings["frame_mode"] in mode_values
            else FRAME_MODE_TAIL
        )
        selected_mode_label = st.selectbox(
            "Trajectory range",
            list(mode_label_to_value),
            key="frame_mode",
            index=mode_values.index(default_mode),
            help="Choose whether each frame shows current points, a fixed trail, or all prior points.",
        )
        frame_mode = mode_label_to_value[selected_mode_label]
        tail_n = slider_with_step_buttons(
            st,
            "Frames shown per trace, including current",
            min_value=1,
            max_value=max(200, int(settings["tail_n"])),
            value=max(1, int(settings["tail_n"])),
            step=1,
            key="frames_shown_per_trace",
            help="For trail mode, each frequency/polarization/method trace keeps this many recent points.",
        )
        marker_size = slider_with_step_buttons(
            st,
            "Center marker size",
            min_value=2,
            max_value=24,
            value=int(settings["marker_size"]),
            step=1,
            key="center_marker_size",
            help="Marker size for radio-source center points in plots and exported video.",
        )
        plot_layout = st.radio(
            "Trajectory layout",
            PLOT_LAYOUTS,
            key="plot_layout",
            index=_choice_index(PLOT_LAYOUTS, settings["plot_layout"]),
            horizontal=True,
            help="Overlay all traces in one plot or split them into facet plots.",
        )
        facet_by = st.selectbox(
            "Facet split dimension",
            FACET_BY_OPTIONS,
            key="facet_by",
            index=_choice_index(FACET_BY_OPTIONS, settings["facet_by"]),
            disabled=str(plot_layout) != "facets",
            help="When using facets, choose whether to split by frequency, polarization, or center method.",
        )

        with st.expander("Settings", expanded=False):
            uploaded = st.file_uploader(
                "Upload center table",
                type=["csv", "xlsx", "xls"],
                help="Optional uploaded table; if present, it overrides the path above for this session.",
            )
            aia_pattern = st.text_input(
                "AIA file pattern",
                key="aia_pattern",
                value=str(settings["aia_pattern"]),
                help="Glob pattern used when scanning the AIA FITS folder.",
            )
            playback_renderer = st.radio(
                "Playback renderer",
                PLAYBACK_RENDERERS,
                key="playback_renderer",
                index=_choice_index(PLAYBACK_RENDERERS, settings["playback_renderer"]),
                horizontal=True,
                help="Use preloaded browser playback for smooth motion, or Streamlit fallback for debugging.",
            )
            draw_lines = st.checkbox(
                "Connect centers within each trace",
                key="draw_lines",
                value=bool(settings["draw_lines"]),
                help="Draw a line through points that share frequency, polarization, and method.",
            )
            max_aia_dt_sec = st.number_input(
                "Max AIA-radio time gap / s",
                key="max_aia_dt_sec",
                min_value=0.0,
                value=float(settings["max_aia_dt_sec"]),
                step=10.0,
                help="Nearest AIA images beyond this time gap are not shown.",
            )
            wcs_mode = st.selectbox(
                "AIA coordinate mode",
                ["header", "sunpy"],
                key="wcs_mode",
                index=_choice_index(("header", "sunpy"), settings["wcs_mode"]),
                help="Use fast FITS-header coordinates or validate with SunPy first.",
            )
            compare_lr = st.checkbox(
                "Show LCP-RCP links and difference table",
                key="compare_lr",
                value=bool(settings["compare_lr"]),
                help="Draw separation links and table rows for paired LCP/RCP centers.",
            )
            compare_tolerance_sec = st.number_input(
                "LCP/RCP match tolerance / s",
                key="compare_tolerance_sec",
                min_value=0.0,
                value=float(settings["compare_tolerance_sec"]),
                step=0.1,
                help="Maximum time difference allowed when pairing LCP and RCP centers.",
            )
            playback_min_step_sec = st.number_input(
                "Playback time step / s",
                key="playback_min_step_sec",
                min_value=0.0,
                value=float(settings["playback_min_step_sec"]),
                step=0.05,
                help="Coalesce dense radio frames for smoother playback; set 0 to keep every frame.",
            )
            trail_min_opacity = slider_with_step_buttons(
                st,
                "Minimum historical-point opacity",
                min_value=0.05,
                max_value=1.0,
                value=float(settings["trail_min_opacity"]),
                step=0.05,
                key="minimum_historical_point_opacity",
                help="Oldest visible trail points use this opacity; current points remain fully opaque.",
            )
            link_facet_views = st.checkbox(
                "Link facet views",
                key="link_facet_views",
                value=bool(settings["link_facet_views"]),
                disabled=str(plot_layout) != "facets",
                help="Keep facet zoom and pan ranges synchronized.",
            )
            st.subheader("Video Export")
            video_output_format = st.selectbox(
                "Backend format",
                VIDEO_OUTPUT_FORMATS,
                key="video_output_format",
                index=_choice_index(
                    VIDEO_OUTPUT_FORMATS, settings["video_output_format"]
                ),
                format_func=lambda value: value.upper(),
                help="Format used by the reproducible backend export.",
            )
            video_browser_format = st.selectbox(
                "Browser format",
                BROWSER_RECORDING_FORMATS,
                key="video_browser_format",
                index=_choice_index(
                    BROWSER_RECORDING_FORMATS, settings["video_browser_format"]
                ),
                format_func=lambda value: value.upper(),
                help="Format downloaded by deterministic recording in the browser player.",
            )
            video_quality = st.selectbox(
                "Video quality",
                VIDEO_QUALITY_OPTIONS,
                key="video_quality",
                index=_choice_index(VIDEO_QUALITY_OPTIONS, settings["video_quality"]),
                help="Shared encoding quality for browser recording and backend export.",
            )
            video_width = st.number_input(
                "Video width / px",
                key="video_width",
                min_value=320,
                max_value=3840,
                value=int(settings["video_width"]),
                step=160,
                help="Width used by browser recording and backend export.",
            )
            video_height = st.number_input(
                "Video height / px",
                key="video_height",
                min_value=240,
                max_value=2160,
                value=int(settings["video_height"]),
                step=90,
                help="Height used by browser recording and backend export.",
            )
            video_fps = st.number_input(
                "Video FPS",
                key="video_fps",
                min_value=0.2,
                max_value=60.0,
                value=float(settings["video_fps"]),
                step=0.5,
                help="Frame rate used by browser recording and backend export.",
            )
            video_include_aia = st.checkbox(
                "Include AIA background in exported video",
                key="video_include_aia",
                value=bool(settings["video_include_aia"]),
                help="Include the AIA background layer in backend video export when AIA is enabled.",
            )
            configured_video_path = str(settings["video_output_path"]).strip()
            default_video_path = configured_video_path or str(
                layout.outputs_dir
                / f"radio_source_trajectory_export.{video_output_format}"
            )
            video_output_path = render_native_path_input(
                st,
                "Video output path",
                key="video_output_path",
                initial_value=default_video_path,
                roots=path_policy.output_roots,
                kind="save_file",
                extensions=(f".{video_output_format}",),
                default_suffix=f".{video_output_format}",
                help_text="Full backend output path; its extension follows Backend format.",
                frontend_id="source-trajectory",
                operation="export-video",
                state_store=ui_store,
                stacked=True,
            )
            show_debug_tables = st.checkbox(
                "Show debug tables",
                key="show_debug_tables",
                value=bool(settings["show_debug_tables"]),
                help="Show current center rows and LCP/RCP difference tables when playback is paused.",
            )
            if st.button(
                "Clear cache",
                width="stretch",
                help="Clear cached AIA scans, background reads, and filtered tables.",
            ):
                st.cache_data.clear()
            st.caption("Latest UI choices are saved in the private Local state.")

    _apply_theme_css(st, str(theme_mode))

    try:
        if uploaded is not None:
            raw_centers = load_centers_table(uploaded)
        else:
            centers_path = str(path_policy.input_file(centers_path))
            raw_centers = _cached_load_centers(
                centers_path,
                _file_mtime_token(centers_path),
            )
    except Exception as exc:
        st.error(f"Unable to read the center table: {exc}")
        st.stop()

    try:
        centers = _cached_filter_time_range(raw_centers, time_start, time_end)
    except Exception as exc:
        st.error(f"Invalid time window: {exc}")
        st.stop()

    if centers.empty:
        st.warning("No valid radio-source centers were found in the time window.")
        st.stop()

    with st.sidebar:
        st.header("Radio Trajectory")
        freqs_all = sorted(centers["freq_mhz"].dropna().unique().tolist())
        pols_all = sorted(
            centers["polarization"].dropna().astype(str).unique().tolist()
        )
        methods_all = sorted(
            centers["center_method"].dropna().astype(str).unique().tolist()
        )
        selected_freqs = st.multiselect(
            "Radio frequencies / MHz",
            freqs_all,
            key="selected_freqs",
            default=(
                _saved_float_selection(settings["selected_freqs"], freqs_all)
                or freqs_all
            ),
            help="Choose which radio frequency bands to display.",
        )
        frequency_marker_symbols = default_frequency_marker_symbols(
            [float(freq) for freq in freqs_all],
            settings["frequency_marker_symbols"],
        )
        with st.expander("Frequency marker symbols", expanded=False):
            for freq in freqs_all:
                marker_key = frequency_marker_key(freq)
                current_symbol = frequency_marker_symbols.get(marker_key, "circle")
                frequency_marker_symbols[marker_key] = st.selectbox(
                    f"{float(freq):.3g} MHz marker",
                    MARKER_SYMBOL_OPTIONS,
                    index=_choice_index(MARKER_SYMBOL_OPTIONS, current_symbol),
                    key=f"radio_marker_symbol_{marker_key}",
                    help="Marker symbol used for this radio frequency in playback and video export.",
                )
        default_pols = _saved_str_selection(settings["selected_pols"], pols_all) or (
            ["L+R"] if "L+R" in pols_all else pols_all
        )
        selected_pols = st.multiselect(
            "Polarization",
            pols_all,
            key="selected_pols",
            default=default_pols,
            help="Choose polarization products to display.",
        )
        with st.expander("Settings: Filters and Tables", expanded=False):
            selected_methods = st.multiselect(
                "Center method",
                methods_all,
                key="selected_methods",
                default=_saved_str_selection(settings["selected_methods"], methods_all)
                or methods_all,
                help="Choose which center-detection methods to display.",
            )
        apply_preload_clicked = st.button(
            "Apply & Preload",
            width="stretch",
            help="Build or rebuild the smooth browser player after all parameters are set.",
        )

    selected = _cached_filter_centers(
        centers,
        tuple(float(freq) for freq in selected_freqs),
        tuple(str(pol) for pol in selected_pols),
        tuple(str(method) for method in selected_methods),
    )
    _persist_current_settings(
        settings_file,
        {
            "centers": centers_path,
            "time_start": time_start,
            "time_end": time_end,
            "aia_dir": aia_dir,
            "aia_pattern": aia_pattern,
            "frame_mode": frame_mode,
            "tail_n": int(tail_n),
            "plot_layout": str(plot_layout),
            "facet_by": str(facet_by),
            "theme_mode": theme_mode,
            "screen_fit": screen_fit,
            "use_aia": bool(use_aia),
            "max_pixels": int(max_pixels),
            "percentile_limits": [
                float(percentile_limits[0]),
                float(percentile_limits[1]),
            ],
            "log_scale": bool(log_scale),
            "max_aia_dt_sec": float(max_aia_dt_sec),
            "wcs_mode": wcs_mode,
            "compare_lr": bool(compare_lr),
            "compare_tolerance_sec": float(compare_tolerance_sec),
            "draw_lines": bool(draw_lines),
            "marker_size": int(marker_size),
            "frequency_marker_symbols": frequency_marker_symbols,
            "trail_min_opacity": float(trail_min_opacity),
            "link_facet_views": bool(link_facet_views),
            "fps": float(fps),
            "video_width": int(video_width),
            "video_height": int(video_height),
            "video_fps": float(video_fps),
            "video_output_format": str(video_output_format),
            "video_browser_format": str(video_browser_format),
            "video_quality": str(video_quality),
            "video_output_path": str(video_output_path),
            "video_include_aia": bool(video_include_aia),
            "playback_renderer": str(playback_renderer),
            "playback_aia_max_pixels": int(playback_aia_max_pixels),
            "playback_min_step_sec": float(playback_min_step_sec),
            "selected_freqs": [float(freq) for freq in selected_freqs],
            "selected_pols": list(selected_pols),
            "selected_methods": list(selected_methods),
            "show_debug_tables": bool(show_debug_tables),
        },
        st=st,
    )
    save_streamlit_fields(st, ui_store, TRAJECTORY_UI_FIELD_KEYS)

    if selected.empty:
        st.warning("No radio-source centers match the current filters.")
        st.stop()

    raw_times = frame_times(selected)
    times = coalesce_frame_times(
        raw_times,
        min_step_sec=float(playback_min_step_sec),
    )
    if not times:
        st.warning("No valid time frames are available.")
        st.stop()

    if "radio_source_frame_idx" not in st.session_state:
        st.session_state.radio_source_frame_idx = 0
    if "radio_source_playing" not in st.session_state:
        st.session_state.radio_source_playing = False
    if "radio_source_last_tick" not in st.session_state:
        st.session_state.radio_source_last_tick = time.monotonic()
    st.session_state.radio_source_frame_idx = int(
        max(0, min(st.session_state.radio_source_frame_idx, len(times) - 1))
    )

    aia_table = pd.DataFrame()
    if use_aia and aia_dir:
        try:
            aia_dir = str(path_policy.input_directory(aia_dir))
            aia_table = _cached_scan_aia_folder(
                aia_dir,
                aia_pattern,
                _directory_mtime_token(aia_dir),
            )
        except Exception as exc:
            st.warning(f"AIA folder scan failed: {exc}")

    motion_summary = _cached_summarize_motion(selected)

    video_frame_count = len(times)
    current_start = min(
        max(1, int(st.session_state.get("video_start_frame", 1))),
        video_frame_count,
    )
    current_end = min(
        max(
            current_start,
            int(st.session_state.get("video_end_frame", video_frame_count)),
        ),
        video_frame_count,
    )
    st.session_state.video_start_frame = current_start
    st.session_state.video_end_frame = current_end
    st.session_state.video_start_frame_input = min(
        max(
            1,
            int(st.session_state.get("video_start_frame_input", current_start)),
        ),
        video_frame_count,
    )
    st.session_state.video_end_frame_input = min(
        max(
            current_start,
            int(st.session_state.get("video_end_frame_input", current_end)),
        ),
        video_frame_count,
    )

    with st.expander("Video Export", expanded=False):
        st.caption(
            "Browser recording and backend export use the same frame range, "
            "size, frame rate, and quality."
        )
        range_start_col, range_end_col = st.columns(2)
        with range_start_col:
            video_start_frame = int(
                st.number_input(
                    "Start frame",
                    min_value=1,
                    max_value=video_frame_count,
                    step=1,
                    key="video_start_frame_input",
                )
            )
        st.session_state.video_end_frame_input = min(
            max(video_start_frame, int(st.session_state.video_end_frame_input)),
            video_frame_count,
        )
        with range_end_col:
            video_end_frame = int(
                st.number_input(
                    "End frame",
                    min_value=video_start_frame,
                    max_value=video_frame_count,
                    step=1,
                    key="video_end_frame_input",
                )
            )
        st.session_state.video_start_frame = video_start_frame
        st.session_state.video_end_frame = video_end_frame
        if st.button(
            "Export Video",
            width="stretch",
            help="Render the selected range with the current filters, layout, theme, and timeline.",
        ):
            try:
                validated_video_output = path_policy.save_file(
                    video_output_path,
                    default_suffix=f".{video_output_format}",
                )
                with st.spinner(
                    f"Exporting {str(video_output_format).upper()} video..."
                ):
                    exported = export_radio_source_video(
                        selected,
                        times,
                        frame_mode=frame_mode,
                        tail_n=int(tail_n),
                        plot_layout=str(plot_layout),
                        facet_by=str(facet_by),
                        aia_table=aia_table if bool(use_aia and aia_dir) else None,
                        options=VideoExportOptions(
                            out_path=validated_video_output,
                            output_format=str(video_output_format),
                            quality=str(video_quality),
                            fps=float(video_fps),
                            width=int(video_width),
                            height=int(video_height),
                            theme_mode=str(theme_mode),
                            draw_lines=bool(draw_lines),
                            marker_size=int(marker_size),
                            marker_symbol_by_freq=frequency_marker_symbols,
                            trail_min_opacity=float(trail_min_opacity),
                            include_aia=bool(video_include_aia and use_aia),
                            max_aia_dt_sec=float(max_aia_dt_sec),
                            aia_max_pixels=int(playback_aia_max_pixels),
                            percentile_limits=(
                                float(percentile_limits[0]),
                                float(percentile_limits[1]),
                            ),
                            log_scale=bool(log_scale),
                            wcs_mode=wcs_mode,
                            start_frame=video_start_frame - 1,
                            end_frame=video_end_frame,
                        ),
                    )
                st.success(f"Video exported: {exported}")
            except Exception as exc:
                st.error(f"Video export failed: {exc}")

    playback_interval = playback_interval_seconds(
        float(fps),
        playing=bool(st.session_state.radio_source_playing),
    )
    fragment_decorator = getattr(st, "fragment", None)

    def _render_playback_panel() -> None:
        playing = bool(st.session_state.radio_source_playing)
        if playing:
            _advance_playback_index_if_due(
                st.session_state,
                frame_count=len(times),
                interval_sec=playback_interval or 0.5,
            )

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 5])
        with c1:
            if st.button(
                "Previous Frame",
                width="stretch",
                disabled=playing,
                help="Move one frame backward while playback is paused.",
            ):
                st.session_state.radio_source_frame_idx = max(
                    0,
                    st.session_state.radio_source_frame_idx - 1,
                )
        with c2:
            if st.button(
                "Next Frame",
                width="stretch",
                disabled=playing,
                help="Move one frame forward while playback is paused.",
            ):
                st.session_state.radio_source_frame_idx = min(
                    len(times) - 1,
                    st.session_state.radio_source_frame_idx + 1,
                )
        with c3:
            button_label = "Pause" if playing else "Play"
            if st.button(
                button_label,
                width="stretch",
                help="Start or pause Streamlit fallback playback.",
            ):
                st.session_state.radio_source_playing = not playing
                st.session_state.radio_source_last_tick = time.monotonic()
                st.rerun()
        with c4:
            if st.button(
                "First Frame",
                width="stretch",
                help="Return to the first playback frame.",
            ):
                st.session_state.radio_source_frame_idx = 0
                st.session_state.radio_source_playing = False
                st.session_state.radio_source_last_tick = time.monotonic()
                st.rerun()

        selected_index = st.slider(
            "Time frame index",
            min_value=0,
            max_value=len(times) - 1,
            value=int(st.session_state.radio_source_frame_idx),
            step=1,
            help="Select the frame displayed by the Streamlit fallback renderer.",
        )
        st.session_state.radio_source_frame_idx = int(selected_index)
        frame_time = pd.Timestamp(times[st.session_state.radio_source_frame_idx])

        aia_background = None
        title_extra = ""
        if use_aia and aia_dir:
            try:
                nearest = find_nearest_aia(
                    aia_table,
                    frame_time,
                    max_dt_seconds=float(max_aia_dt_sec),
                )
                if nearest.status == "matched" and nearest.path:
                    effective_max_pixels = resolve_effective_aia_max_pixels(
                        max_pixels=int(max_pixels),
                        playback_aia_max_pixels=int(playback_aia_max_pixels),
                        playing=playing,
                    )
                    aia_background = _cached_read_aia_background(
                        nearest.path,
                        _file_mtime_token(nearest.path),
                        int(effective_max_pixels),
                        float(percentile_limits[0]),
                        float(percentile_limits[1]),
                        bool(log_scale),
                        wcs_mode,
                    )
                    title_extra = f"AIA dt={nearest.delta_seconds:.1f}s"
                    preview_note = (
                        f", playback preview {effective_max_pixels}px"
                        if playing and effective_max_pixels < int(max_pixels)
                        else ""
                    )
                    st.caption(
                        f"AIA background: {aia_background.label}, "
                        f"time offset from radio frame {nearest.delta_seconds:.1f} s"
                        f"{preview_note}"
                    )
                else:
                    st.info(f"AIA background not shown: {nearest.status}")
            except Exception as exc:
                st.warning(f"AIA background read failed: {exc}")

        visible = _cached_visible_centers(
            selected,
            frame_time.isoformat(),
            frame_mode,
            int(tail_n),
        )
        fig, compare_df = build_trajectory_figure(
            visible,
            frame_time,
            aia_background=aia_background,
            draw_lines=bool(draw_lines),
            compare_lr=bool(compare_lr),
            compare_tolerance_sec=float(compare_tolerance_sec),
            title_extra=title_extra,
            theme_mode=str(theme_mode),
            screen_fit=str(screen_fit),
            use_webgl=True,
            plot_layout=str(plot_layout),
            facet_by=str(facet_by),
            marker_size=int(marker_size),
            marker_symbol_by_freq=frequency_marker_symbols,
            trail_min_opacity=float(trail_min_opacity),
            sync_axes=bool(link_facet_views),
        )
        st.plotly_chart(fig, width="stretch")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Loaded centers", len(raw_centers))
        m2.metric("Window centers", len(centers))
        m3.metric("Filtered centers", len(selected))
        m4.metric("Visible centers", len(visible))
        m5.metric("Playback frames", len(times))
        if len(raw_times) != len(times):
            st.caption(
                f"Raw time frames {len(raw_times)}, playback frames {len(times)}"
            )

        if show_debug_tables:
            if playing:
                st.info(
                    "Table rendering is paused during playback; pause to inspect center rows and LCP-RCP differences."
                )
            else:
                with st.expander("Currently visible radio centers", expanded=False):
                    st.dataframe(
                        visible.sort_values(["freq_mhz", "polarization", "obs_time"]),
                        width="stretch",
                        hide_index=True,
                    )
                if compare_lr:
                    with st.expander("LCP-RCP position differences", expanded=True):
                        if compare_df.empty:
                            st.info(
                                "No pairable LCP/RCP centers are visible in the current range."
                            )
                        else:
                            st.dataframe(
                                compare_df.sort_values(["freq_mhz", "obs_time"]),
                                width="stretch",
                                hide_index=True,
                            )

    rendered_preloaded = False
    if str(playback_renderer) == "preloaded":
        frame_time_iso_values = tuple(
            pd.Timestamp(value).isoformat() for value in times
        )
        directory_token = (
            _directory_mtime_token(aia_dir) if aia_dir else ("", None, None)
        )
        preload_signature = build_preload_signature(
            {
                "centers_path": centers_path,
                "centers_file": (
                    _file_mtime_token(centers_path)
                    if uploaded is None
                    else ("uploaded", getattr(uploaded, "name", ""))
                ),
                "time_start": time_start,
                "time_end": time_end,
                "frame_times": frame_time_iso_values,
                "selected_freqs": [float(freq) for freq in selected_freqs],
                "selected_pols": [str(pol) for pol in selected_pols],
                "selected_methods": [str(method) for method in selected_methods],
                "frame_mode": frame_mode,
                "tail_n": int(tail_n),
                "aia_dir": aia_dir,
                "aia_pattern": aia_pattern,
                "aia_dir_token": directory_token,
                "use_aia": bool(use_aia and aia_dir),
                "max_aia_dt_sec": float(max_aia_dt_sec),
                "playback_aia_max_pixels": int(playback_aia_max_pixels),
                "percentile_limits": [
                    float(percentile_limits[0]),
                    float(percentile_limits[1]),
                ],
                "log_scale": bool(log_scale),
                "wcs_mode": wcs_mode,
                "theme_mode": str(theme_mode),
                "screen_fit": str(screen_fit),
                "draw_lines": bool(draw_lines),
                "fps": float(fps),
                "plot_layout": str(plot_layout),
                "facet_by": str(facet_by),
                "marker_size": int(marker_size),
                "frequency_marker_symbols": frequency_marker_symbols,
                "trail_min_opacity": float(trail_min_opacity),
                "link_facet_views": bool(link_facet_views),
            }
        )
        applied_signature = st.session_state.get("radio_source_preload_signature")
        session_payload = st.session_state.get("radio_source_preload_payload")
        has_payload = isinstance(session_payload, dict)
        if preload_settings_changed(
            preload_signature,
            applied_signature,
            has_payload=has_payload,
        ):
            st.info(
                "Settings changed. Click Apply & Preload to rebuild smooth playback."
            )
        if should_build_preloaded_payload(
            preload_signature,
            applied_signature,
            apply_clicked=bool(apply_preload_clicked),
            has_payload=has_payload,
        ):
            try:
                with st.spinner("Preparing smooth preloaded playback..."):
                    session_payload = _cached_preloaded_playback_payload(
                        selected,
                        frame_time_iso_values,
                        frame_mode,
                        int(tail_n),
                        aia_table,
                        bool(use_aia and aia_dir),
                        directory_token,
                        float(max_aia_dt_sec),
                        int(playback_aia_max_pixels),
                        float(percentile_limits[0]),
                        float(percentile_limits[1]),
                        bool(log_scale),
                        wcs_mode,
                        str(theme_mode),
                        str(screen_fit),
                        bool(draw_lines),
                        float(fps),
                        str(plot_layout),
                        str(facet_by),
                        int(marker_size),
                        frequency_marker_symbols,
                        float(trail_min_opacity),
                        bool(link_facet_views),
                    )
                st.session_state.radio_source_preload_payload = session_payload
                st.session_state.radio_source_preload_signature = preload_signature
                applied_signature = preload_signature
                has_payload = True
            except Exception as exc:
                st.warning(
                    "Preloaded playback failed; falling back to Streamlit renderer: "
                    f"{exc}"
                )
                session_payload = (
                    st.session_state.get("radio_source_preload_payload")
                    if isinstance(
                        st.session_state.get("radio_source_preload_payload"),
                        dict,
                    )
                    else None
                )
                has_payload = isinstance(session_payload, dict)
        elif not has_payload:
            st.info(
                "Set parameters, then click Apply & Preload to build smooth playback."
            )

        if isinstance(session_payload, dict):
            player_payload = dict(session_payload)
            player_config = dict(player_payload.get("config", {}))
            player_config["recording"] = {
                "format": str(video_browser_format),
                "quality": str(video_quality),
                "fps": float(video_fps),
                "width": int(video_width),
                "height": int(video_height),
                "start_frame": int(video_start_frame) - 1,
                "end_frame": int(video_end_frame),
            }
            player_payload["config"] = player_config
            html = build_preloaded_playback_html(player_payload)
            component_height = (
                int(session_payload.get("layout", {}).get("height", 760)) + 112
            )
            if hasattr(st, "iframe"):
                st.iframe(html, height=component_height, width="stretch")
            else:  # pragma: no cover - compatibility with Streamlit < 1.58.
                from streamlit.components.v1 import html as render_component_html

                render_component_html(html, height=component_height, scrolling=False)
            stats = dict(session_payload.get("stats", {}))
            stats["payload_size_mb"] = round(
                _preloaded_payload_size_mb(session_payload),
                2,
            )
            stale_note = (
                " Showing the last applied preload."
                if applied_signature and preload_signature != applied_signature
                else ""
            )
            st.caption(
                "Preloaded playback: "
                f"{stats.get('frame_count', 0)} frames, "
                f"{stats.get('background_count', 0)} AIA backgrounds, "
                f"{stats.get('payload_size_mb', 0):.2f} MB payload."
                f"{stale_note}"
            )
            with st.expander("Preload Stats", expanded=False):
                st.json(stats)
            rendered_preloaded = True

    if not rendered_preloaded:
        if fragment_decorator is not None:
            fragment_decorator(run_every=playback_interval)(_render_playback_panel)()
        else:  # pragma: no cover - kept for older Streamlit runtimes.
            _render_playback_panel()

    with st.expander("Motion Summary", expanded=True):
        if motion_summary.empty:
            st.info(
                "No trajectories are available to summarize under the current filters."
            )
        else:
            st.dataframe(
                motion_summary.sort_values(
                    ["freq_mhz", "polarization", "center_method"]
                ),
                width="stretch",
                hide_index=True,
            )


def _coerce_app_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(DEFAULT_APP_SETTINGS)
    for key, value in dict(settings).items():
        if key in result:
            result[key] = value
    result["theme_mode"] = _choice_value(THEME_MODES, result["theme_mode"], "auto")
    result["screen_fit"] = _choice_value(SCREEN_FIT_MODES, result["screen_fit"], "auto")
    result["plot_layout"] = _choice_value(
        PLOT_LAYOUTS,
        result["plot_layout"],
        "overlay",
    )
    result["facet_by"] = _choice_value(
        FACET_BY_OPTIONS,
        result["facet_by"],
        "freq_mhz",
    )
    result["playback_renderer"] = _choice_value(
        PLAYBACK_RENDERERS,
        result["playback_renderer"],
        "preloaded",
    )
    result["frame_mode"] = _choice_value(
        tuple(FRAME_MODE_LABELS),
        result["frame_mode"],
        FRAME_MODE_TAIL,
    )
    result["wcs_mode"] = _choice_value(
        ("header", "sunpy"),
        result["wcs_mode"],
        "header",
    )
    result["tail_n"] = max(1, _safe_int(result["tail_n"], 5))
    result["marker_size"] = min(
        24,
        max(2, _safe_int(result["marker_size"], DEFAULT_MARKER_SIZE)),
    )
    result["frequency_marker_symbols"] = normalize_marker_symbol_by_frequency(
        result["frequency_marker_symbols"]
    )
    result["trail_min_opacity"] = _coerce_unit_interval(
        result["trail_min_opacity"],
        DEFAULT_TRAIL_MIN_OPACITY,
    )
    result["max_pixels"] = min(2048, max(256, _safe_int(result["max_pixels"], 1024)))
    result["playback_aia_max_pixels"] = min(
        1024,
        max(
            128,
            _safe_int(
                result["playback_aia_max_pixels"],
                DEFAULT_PLAYBACK_AIA_MAX_PIXELS,
            ),
        ),
    )
    result["playback_min_step_sec"] = max(
        0.0,
        _safe_float(
            result["playback_min_step_sec"],
            DEFAULT_PLAYBACK_MIN_STEP_SEC,
        ),
    )
    result["max_aia_dt_sec"] = max(0.0, _safe_float(result["max_aia_dt_sec"], 3600.0))
    result["compare_tolerance_sec"] = max(
        0.0,
        _safe_float(result["compare_tolerance_sec"], 1.0),
    )
    result["fps"] = min(20.0, max(0.2, _safe_float(result["fps"], 2.0)))
    result["video_width"] = min(
        3840,
        max(320, _safe_int(result["video_width"], VIDEO_DEFAULT_WIDTH)),
    )
    result["video_height"] = min(
        2160,
        max(240, _safe_int(result["video_height"], VIDEO_DEFAULT_HEIGHT)),
    )
    result["video_fps"] = min(
        60.0,
        max(0.2, _safe_float(result["video_fps"], VIDEO_DEFAULT_FPS)),
    )
    result["video_output_format"] = _choice_value(
        VIDEO_OUTPUT_FORMATS,
        result["video_output_format"],
        "mp4",
    )
    result["video_browser_format"] = _choice_value(
        BROWSER_RECORDING_FORMATS,
        result["video_browser_format"],
        "webm",
    )
    result["video_quality"] = _choice_value(
        VIDEO_QUALITY_OPTIONS,
        result["video_quality"],
        "high",
    )
    result["video_output_path"] = str(result["video_output_path"] or "").strip()
    result["percentile_limits"] = _coerce_percentile_limits(result["percentile_limits"])
    for key in (
        "use_aia",
        "log_scale",
        "compare_lr",
        "draw_lines",
        "show_debug_tables",
        "video_include_aia",
        "link_facet_views",
    ):
        result[key] = bool(result[key])
    for key in ("centers", "time_start", "time_end", "aia_dir", "aia_pattern"):
        result[key] = "" if result[key] is None else str(result[key])
    for key in ("selected_freqs", "selected_pols", "selected_methods"):
        value = result[key]
        result[key] = list(value) if isinstance(value, list | tuple) else []
    return result


def _choice_index(options: tuple[str, ...], value: object) -> int:
    text = str(value)
    return options.index(text) if text in options else 0


def _choice_value(options: tuple[str, ...], value: object, default: str) -> str:
    text = str(value)
    return text if text in options else default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_unit_interval(value: object, default: float) -> float:
    return min(1.0, max(0.0, _safe_float(value, default)))


def _coerce_percentile_limits(value: object) -> list[float]:
    if isinstance(value, list | tuple) and len(value) >= 2:
        low = _safe_float(value[0], 1.0)
        high = _safe_float(value[1], 99.7)
    else:
        low, high = 1.0, 99.7
    low = min(100.0, max(0.0, low))
    high = min(100.0, max(0.0, high))
    if low > high:
        low, high = high, low
    return [low, high]


def _saved_float_selection(saved: object, available: list[float]) -> list[float]:
    if not isinstance(saved, list | tuple):
        return []
    saved_values: set[float] = set()
    for value in saved:
        try:
            saved_values.add(float(value))
        except (TypeError, ValueError):
            continue
    return [float(value) for value in available if float(value) in saved_values]


def _saved_str_selection(saved: object, available: list[str]) -> list[str]:
    if not isinstance(saved, list | tuple):
        return []
    saved_values = {str(value) for value in saved}
    return [value for value in available if str(value) in saved_values]


def _file_mtime_token(path_text: str) -> tuple[str, int | None, int | None]:
    path = Path(path_text).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _directory_mtime_token(path_text: str) -> tuple[str, int | None, int | None]:
    path = Path(path_text).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), int(stat.st_mtime_ns), None)


def _persist_current_settings(
    settings_file: Path,
    settings: dict[str, Any],
    *,
    st,
) -> None:
    # Kept as a compatibility seam for callers and tests. The current frontend
    # imports this legacy file once, then persists only whitelisted UI fields in
    # Local/state through ``save_streamlit_fields``.
    del settings_file, settings, st


def _apply_theme_css(st, theme_mode: str) -> None:
    st.markdown(
        f"<style>{streamlit_theme_css(theme_mode)}</style>",
        unsafe_allow_html=True,
    )


def _theme_css(
    *,
    app_bg: str,
    sidebar_bg: str,
    text: str,
    border: str,
    input_bg: str,
    scoped: bool = False,
) -> str:
    prefix = "" if not scoped else "\n"
    return f"""
{prefix}:root {{
    --radio-theme-app-bg: {app_bg};
    --radio-theme-sidebar-bg: {sidebar_bg};
    --radio-theme-text: {text};
    --radio-theme-border: {border};
    --radio-theme-input-bg: {input_bg};
}}
{prefix}.stApp {{
    background: {app_bg};
    color: {text};
}}
{prefix}[data-testid="stHeader"] {{
    background: {app_bg};
    color: {text};
    border-bottom: 1px solid {border};
}}
{prefix}[data-testid="stToolbar"] {{
    color: {text};
}}
{prefix}[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {border};
}}
{prefix}[data-testid="stSidebar"] * {{
    color: {text};
}}
{prefix}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
{prefix}[data-testid="stSidebar"] label,
{prefix}[data-testid="stSidebar"] span {{
    color: {text};
}}
{prefix}[data-baseweb="input"],
{prefix}[data-baseweb="select"],
{prefix}[data-baseweb="textarea"] {{
    background-color: {input_bg};
    border-color: {border};
    color: {text};
}}
{prefix}[data-baseweb="input"] input,
{prefix}[data-baseweb="select"] input,
{prefix}[data-baseweb="textarea"] textarea,
{prefix}[data-testid="stTextInput"] input,
{prefix}[data-testid="stNumberInput"] input,
{prefix}textarea {{
    background-color: {input_bg};
    color: {text};
    -webkit-text-fill-color: {text};
}}
{prefix}[data-baseweb="select"] > div,
{prefix}[data-baseweb="input"] > div,
{prefix}[data-baseweb="textarea"] > div {{
    background-color: {input_bg};
    border-color: {border};
    color: {text};
}}
{prefix}[data-testid="stButton"] button,
{prefix}[data-testid="stDownloadButton"] button {{
    background-color: {input_bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
}}
{prefix}[data-testid="stButton"] button:hover,
{prefix}[data-testid="stDownloadButton"] button:hover {{
    border-color: #ef4444;
    color: {text};
}}
{prefix}[data-testid="stSlider"] [data-baseweb="slider"] > div {{
    color: {text};
}}
{prefix}[data-testid="stAlert"] {{
    background-color: color-mix(in srgb, {input_bg} 88%, #60a5fa 12%);
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
}}
{prefix}.streamlit-expanderHeader,
{prefix}[data-testid="stExpander"] {{
    background-color: {input_bg};
    color: {text};
    border-color: {border};
}}
{prefix}[data-testid="stExpander"] details {{
    border-color: {border};
}}
{prefix}div[data-testid="stMetric"] {{
    border: 1px solid {border};
    border-radius: 8px;
    background-color: {input_bg};
    color: {text};
    padding: 0.35rem 0.55rem;
}}
{prefix}iframe {{
    background: {app_bg};
}}
"""


def main(argv: list[str] | None = None) -> int:
    """Run the Streamlit app and return zero after normal completion."""

    _run_streamlit_app(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
