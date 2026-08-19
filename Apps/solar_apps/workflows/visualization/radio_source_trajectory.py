"""Plotly visualization for radio-source center trajectories.

English: Build an interactive radio-source trajectory figure with optional AIA
backgrounds, tail/current/all frame modes, LCP-RCP comparison, and HTML export.

中文：生成交互式射电源中心轨迹图，支持 AIA 背景、当前/尾迹/全轨迹显示、
LCP-RCP 对比以及单文件 HTML 导出。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from solar_toolkit.aia.background import AiaBackground
from solar_toolkit.radio.trajectory import make_lr_compare_table

__all__ = [
    "FACET_BY_OPTIONS",
    "MARKER_SYMBOL_OPTIONS",
    "PLOT_LAYOUT_FACETS",
    "PLOT_LAYOUT_OVERLAY",
    "PLOT_LAYOUTS",
    "add_lr_compare_segments",
    "aia_colormap_name",
    "aia_plotly_colorscale",
    "apply_aia_colormap_to_uint8",
    "build_trajectory_figure",
    "export_trajectory_html",
    "frequency_marker_key",
    "marker_symbol_for_frequency",
    "normalize_marker_symbol_by_frequency",
    "resolve_theme_palette",
]

PLOT_LAYOUT_OVERLAY = "overlay"
PLOT_LAYOUT_FACETS = "facets"
PLOT_LAYOUTS = (PLOT_LAYOUT_OVERLAY, PLOT_LAYOUT_FACETS)
FACET_BY_OPTIONS = ("freq_mhz", "polarization", "center_method")
FACET_HORIZONTAL_SPACING = 0.075
FACET_VERTICAL_SPACING = 0.14
MARKER_SYMBOL_OPTIONS = ("circle", "x", "cross", "triangle-up", "square", "diamond")
AIA_COLORMAPS = {
    "94": "sdoaia94",
    "131": "sdoaia131",
    "171": "sdoaia171",
    "193": "sdoaia193",
    "211": "sdoaia211",
    "304": "sdoaia304",
    "335": "sdoaia335",
    "1600": "sdoaia1600",
    "1700": "sdoaia1700",
}


def _go():
    import plotly.graph_objects as go

    return go


def build_trajectory_figure(
    visible: pd.DataFrame,
    frame_time,
    *,
    aia_background: AiaBackground | None = None,
    draw_lines: bool = True,
    compare_lr: bool = False,
    compare_tolerance_sec: float = 1.0,
    title_extra: str = "",
    height: int = 760,
    theme_mode: str = "auto",
    screen_fit: str = "auto",
    use_webgl: bool = True,
    image_display_options: dict[str, object] | None = None,
    plot_layout: str = PLOT_LAYOUT_OVERLAY,
    facet_by: str = "freq_mhz",
    marker_size: int = 9,
    marker_symbol_by_freq: dict[str, str] | None = None,
    trail_min_opacity: float = 0.25,
    sync_axes: bool = False,
):
    """Build a Plotly figure for one radio-source playback frame."""

    go = _go()
    theme = resolve_theme_palette(theme_mode)
    layout = _resolve_screen_layout(screen_fit, height=height)
    if image_display_options:
        explicit_height = image_display_options.get("height")
        if explicit_height is not None:
            layout["height"] = int(explicit_height)

    resolved_layout = _normalize_plot_layout(plot_layout)
    resolved_facet_by = _normalize_facet_by(facet_by)
    facet_values = (
        _facet_values(visible, resolved_facet_by)
        if resolved_layout == PLOT_LAYOUT_FACETS
        else []
    )
    use_facets = resolved_layout == PLOT_LAYOUT_FACETS and bool(facet_values)
    if use_facets:
        layout["height"] = _facet_height(
            screen_fit,
            base_height=int(layout["height"]),
            facet_count=len(facet_values),
        )

    if use_facets:
        from plotly.subplots import make_subplots

        cols = min(3, max(1, len(facet_values)))
        rows = int(math.ceil(len(facet_values) / cols))
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                _format_facet_label(resolved_facet_by, value) for value in facet_values
            ],
            horizontal_spacing=FACET_HORIZONTAL_SPACING,
            vertical_spacing=FACET_VERTICAL_SPACING,
        )
    else:
        fig = go.Figure()
        facet_values = []

    axis_ranges = _axis_ranges(visible, aia_background)
    if not use_facets and aia_background is not None:
        _add_aia_trace(fig, go, aia_background)
    elif use_facets and aia_background is not None:
        for facet_index, _facet_value in enumerate(facet_values):
            row, col = _facet_row_col(facet_index, len(facet_values))
            _add_aia_trace(fig, go, aia_background, row=row, col=col)

    mode = "lines+markers" if draw_lines else "markers"
    scatter_class = go.Scattergl if use_webgl else go.Scatter
    if visible is not None and not visible.empty:
        marker_symbols = normalize_marker_symbol_by_frequency(marker_symbol_by_freq)
        for (freq, pol, method), group in visible.groupby(
            ["freq_mhz", "polarization", "center_method"],
            sort=True,
        ):
            sorted_group = group.sort_values("obs_time")
            customdata = [
                [pd.Timestamp(value).isoformat()]
                for value in sorted_group["obs_time"].tolist()
            ]
            trace = scatter_class(
                x=sorted_group["center_x_arcsec"],
                y=sorted_group["center_y_arcsec"],
                mode=mode,
                marker={
                    "size": max(1, int(marker_size)),
                    "symbol": marker_symbol_for_frequency(freq, marker_symbols),
                    "opacity": _trail_opacity(
                        len(sorted_group),
                        min_opacity=trail_min_opacity,
                    ),
                },
                line={"width": 2},
                name=f"{float(freq):.3g} MHz | {pol} | {method}",
                customdata=customdata,
                hovertemplate=(
                    "time=%{customdata[0]}"
                    f"<br>freq={float(freq):.3g} MHz"
                    f"<br>pol={pol}"
                    f"<br>method={method}"
                    '<br>x=%{x:.2f}"'
                    '<br>y=%{y:.2f}"'
                    "<extra></extra>"
                ),
            )
            if use_facets:
                facet_value = sorted_group.iloc[0][resolved_facet_by]
                facet_index = _facet_index(facet_values, facet_value)
                row, col = _facet_row_col(facet_index, len(facet_values))
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

    compare_df = pd.DataFrame()
    if compare_lr and visible is not None and not visible.empty:
        compare_df = make_lr_compare_table(
            visible,
            tolerance_sec=float(compare_tolerance_sec),
        )
        if use_facets and not compare_df.empty:
            for facet_index, facet_value in enumerate(facet_values):
                subset = compare_df[
                    compare_df[resolved_facet_by].astype(str) == str(facet_value)
                ]
                row, col = _facet_row_col(facet_index, len(facet_values))
                add_lr_compare_segments(fig, subset, row=row, col=col)
        else:
            add_lr_compare_segments(fig, compare_df)

    frame_time_iso = pd.Timestamp(frame_time).isoformat()
    title = "Radio source trajectory | " f"display time: {frame_time_iso}"
    if title_extra:
        title = f"{title} | {title_extra}"
    fig.update_layout(
        title=title,
        xaxis_title="HPLN / arcsec",
        yaxis_title="HPLT / arcsec",
        template=theme["template"],
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font={"color": theme["font_color"]},
        height=int(layout["height"]),
        legend=layout["legend"],
        margin=layout["margin"],
    )
    fig.add_annotation(
        text=f"Radio source time: {frame_time_iso}",
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.08,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font={"color": theme["font_color"], "size": 12},
    )
    fig.update_xaxes(
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
        range=axis_ranges["x"],
        title_text="HPLN / arcsec",
    )
    fig.update_yaxes(
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
        range=axis_ranges["y"],
        title_text="HPLT / arcsec",
    )
    if use_facets:
        for facet_index, _facet_value in enumerate(facet_values):
            row, col = _facet_row_col(facet_index, len(facet_values))
            x_anchor = "x" if facet_index == 0 else f"x{facet_index + 1}"
            fig.update_yaxes(
                scaleanchor=x_anchor,
                scaleratio=1,
                constrain="domain",
                row=row,
                col=col,
            )
    else:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    if use_facets and bool(sync_axes):
        for facet_index in range(1, len(facet_values)):
            row, col = _facet_row_col(facet_index, len(facet_values))
            fig.update_xaxes(matches="x", row=row, col=col)
            fig.update_yaxes(matches="y", row=row, col=col)
    return fig, compare_df


def frequency_marker_key(freq: object) -> str:
    """Return a stable compact key for frequency-specific marker settings."""

    try:
        value = float(freq)
    except (TypeError, ValueError):
        return str(freq)
    if not np.isfinite(value):
        return str(freq)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def normalize_marker_symbol_by_frequency(
    marker_symbol_by_freq: dict[str, str] | None,
) -> dict[str, str]:
    """Return sanitized Plotly marker symbols keyed by compact frequency text."""

    if not isinstance(marker_symbol_by_freq, dict):
        return {}
    normalized: dict[str, str] = {}
    for freq, symbol in marker_symbol_by_freq.items():
        marker_symbol = str(symbol)
        if marker_symbol not in MARKER_SYMBOL_OPTIONS:
            marker_symbol = "circle"
        normalized[frequency_marker_key(freq)] = marker_symbol
    return normalized


def marker_symbol_for_frequency(
    freq: object,
    marker_symbol_by_freq: dict[str, str] | None,
) -> str:
    """Return the configured Plotly marker symbol for one frequency."""

    normalized = normalize_marker_symbol_by_frequency(marker_symbol_by_freq)
    return normalized.get(frequency_marker_key(freq), "circle")


def resolve_theme_palette(theme_mode: str) -> dict[str, str]:
    """Return Plotly layout colors for light/dark/auto modes."""

    mode = str(theme_mode or "auto").strip().lower()
    if mode == "dark":
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "#0f172a",
            "plot_bgcolor": "#111827",
            "font_color": "#e5e7eb",
            "grid_color": "#334155",
        }
    return {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f8fafc",
        "font_color": "#111827",
        "grid_color": "#d1d5db",
    }


def aia_colormap_name(wavelength) -> str | None:
    """Return the registered SunPy/Matplotlib colormap name for an AIA band."""

    text = str(wavelength or "").strip()
    if not text:
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        digits = "".join(character for character in text if character.isdigit())
        if digits:
            text = str(int(digits))
    return AIA_COLORMAPS.get(text)


def aia_plotly_colorscale(wavelength) -> tuple[tuple[float, str], ...] | str:
    """Return a Plotly colorscale using SunPy's wavelength-specific AIA palette."""

    cmap = _load_aia_colormap(wavelength)
    if cmap is None:
        return "gray"
    samples = 16
    colorscale: list[tuple[float, str]] = []
    for index in range(samples):
        value = index / max(1, samples - 1)
        red, green, blue, _alpha = cmap(value)
        colorscale.append(
            (
                float(value),
                "rgb("
                f"{int(round(red * 255))},"
                f"{int(round(green * 255))},"
                f"{int(round(blue * 255))}"
                ")",
            )
        )
    return tuple(colorscale)


def apply_aia_colormap_to_uint8(z: np.ndarray, wavelength) -> np.ndarray:
    """Scale AIA data to uint8 and apply the matching SDO AIA colormap if known."""

    scaled = _scale_to_uint8(z)
    cmap = _load_aia_colormap(wavelength)
    if cmap is None:
        return scaled
    rgba = cmap(scaled.astype(float) / 255.0)
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)


def _resolve_theme(theme_mode: str) -> dict[str, str]:
    return resolve_theme_palette(theme_mode)


def _resolve_screen_layout(screen_fit: str, *, height: int) -> dict[str, object]:
    """Return a stable Plotly layout for auto, landscape, or portrait screens."""

    mode = str(screen_fit or "auto").strip().lower()
    if mode == "portrait":
        return {
            "height": max(920, int(height)),
            "legend": {
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
            "margin": {"l": 42, "r": 170, "t": 80, "b": 42},
        }
    if mode == "landscape":
        return {
            "height": max(560, min(720, int(height))),
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            "margin": {"l": 36, "r": 28, "t": 82, "b": 36},
        }
    return {
        "height": int(height),
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "margin": {"l": 30, "r": 30, "t": 85, "b": 30},
    }


def _facet_height(screen_fit: str, *, base_height: int, facet_count: int) -> int:
    rows = int(math.ceil(max(1, int(facet_count)) / 3))
    mode = str(screen_fit or "auto").strip().lower()
    if mode == "portrait":
        return max(int(base_height), rows * 330 + 180)
    if mode == "landscape":
        return max(int(base_height), rows * 250 + 150)
    return max(int(base_height), rows * 290 + 170)


def add_lr_compare_segments(
    fig,
    compare_df: pd.DataFrame,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add dotted LCP-RCP separation segments to an existing figure."""

    if compare_df is None or compare_df.empty:
        return
    go = _go()
    scatter_class = (
        go.Scattergl
        if any(trace.type == "scattergl" for trace in fig.data)
        else go.Scatter
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    hover: list[str] = []
    for _, row in compare_df.iterrows():
        xs.extend([row["L_x_arcsec"], row["R_x_arcsec"], None])
        ys.extend([row["L_y_arcsec"], row["R_y_arcsec"], None])
        text = (
            f"{float(row['freq_mhz']):.3g} MHz"
            f"<br>{pd.Timestamp(row['obs_time']).isoformat()}"
            f"<br>|L-R|={float(row['distance_arcsec']):.2f} arcsec"
        )
        hover.extend([text, text, ""])
    trace = scatter_class(
        x=xs,
        y=ys,
        mode="lines",
        name="LCP-RCP separation",
        line={"width": 1, "dash": "dot"},
        hovertext=hover,
        hoverinfo="text",
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _add_aia_trace(
    fig, go, aia_background: AiaBackground, *, row=None, col=None
) -> None:
    trace = go.Heatmap(
        z=aia_background.z,
        x=aia_background.x_arcsec,
        y=aia_background.y_arcsec,
        colorscale=aia_plotly_colorscale(aia_background.wavelength),
        showscale=False,
        name="AIA",
        hovertemplate='x=%{x:.1f}"<br>y=%{y:.1f}"<extra>AIA</extra>',
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _normalize_plot_layout(value: str) -> str:
    text = str(value or PLOT_LAYOUT_OVERLAY).strip().lower()
    return text if text in PLOT_LAYOUTS else PLOT_LAYOUT_OVERLAY


def _normalize_facet_by(value: str) -> str:
    text = str(value or "freq_mhz").strip().lower()
    return text if text in FACET_BY_OPTIONS else "freq_mhz"


def _facet_values(visible: pd.DataFrame | None, facet_by: str) -> list[object]:
    if visible is None or visible.empty or facet_by not in visible.columns:
        return []
    values = visible[facet_by].dropna().unique().tolist()
    if facet_by == "freq_mhz":
        return sorted(values, key=lambda item: float(item))
    return sorted(values, key=lambda item: str(item))


def _facet_index(facet_values: list[object], value: object) -> int:
    value_text = str(value)
    for index, candidate in enumerate(facet_values):
        if str(candidate) == value_text:
            return index
    return 0


def _facet_row_col(index: int, facet_count: int) -> tuple[int, int]:
    cols = min(3, max(1, int(facet_count)))
    return index // cols + 1, index % cols + 1


def _format_facet_label(facet_by: str, value: object) -> str:
    if facet_by == "freq_mhz":
        return f"{float(value):.3g} MHz"
    if facet_by == "polarization":
        return f"Polarization: {value}"
    if facet_by == "center_method":
        return f"Method: {value}"
    return str(value)


def _axis_ranges(
    visible: pd.DataFrame | None,
    aia_background: AiaBackground | None,
) -> dict[str, list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    if visible is not None and not visible.empty:
        xs.extend(float(value) for value in visible["center_x_arcsec"].tolist())
        ys.extend(float(value) for value in visible["center_y_arcsec"].tolist())
    if aia_background is not None:
        xs.extend(float(value) for value in aia_background.x_arcsec.tolist())
        ys.extend(float(value) for value in aia_background.y_arcsec.tolist())
    if not xs or not ys:
        return {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_pad = max(1.0, (x1 - x0) * 0.05)
    y_pad = max(1.0, (y1 - y0) * 0.05)
    return {"x": [x0 - x_pad, x1 + x_pad], "y": [y0 - y_pad, y1 + y_pad]}


def _trail_opacity(count: int, *, min_opacity: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    low = min(1.0, max(0.0, float(min_opacity)))
    step = (1.0 - low) / float(count - 1)
    return [low + step * index for index in range(count)]


def _load_aia_colormap(wavelength):
    name = aia_colormap_name(wavelength)
    if not name:
        return None
    try:
        import sunpy.visualization.colormaps  # noqa: F401
        from matplotlib import pyplot as plt

        return plt.get_cmap(name)
    except Exception:
        return None


def _scale_to_uint8(z: np.ndarray) -> np.ndarray:
    array = np.asarray(z, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(np.nanmin(array[finite]))
    hi = float(np.nanmax(array[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def export_trajectory_html(
    fig,
    out: str | Path,
    *,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write a Plotly trajectory figure to a standalone HTML file."""

    output_path = Path(out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )
    return output_path
