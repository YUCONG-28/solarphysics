"""Source Map semantic theme shared by every application frontend."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Literal

ThemeMode = Literal["auto", "light", "dark"]
THEME_MODES: tuple[ThemeMode, ...] = ("auto", "light", "dark")


def normalize_theme_mode(value: object) -> ThemeMode:
    """Return a supported mode, defaulting invalid or missing values to Auto."""

    candidate = str(value or "auto").strip().casefold()
    return candidate if candidate in THEME_MODES else "auto"  # type: ignore[return-value]


def _asset_text(name: str) -> str:
    return (
        files("solar_apps.ui.theme_assets").joinpath(name).read_text(encoding="utf-8")
    )


def register_theme_assets(app: Any, *, url_prefix: str = "/assets/solar-theme") -> None:
    """Expose the immutable shared CSS and controller from a Flask application."""

    from flask import Response

    endpoint = f"solar_theme_{len(app.url_map._rules)}"
    app.add_url_rule(
        f"{url_prefix}.css",
        endpoint=f"{endpoint}_css",
        view_func=lambda: Response(
            _asset_text("theme.css"), content_type="text/css; charset=utf-8"
        ),
    )
    app.add_url_rule(
        f"{url_prefix}.js",
        endpoint=f"{endpoint}_js",
        view_func=lambda: Response(
            _asset_text("theme.js"),
            content_type="application/javascript; charset=utf-8",
        ),
    )
    app.add_url_rule(
        f"{url_prefix}-state.js",
        endpoint=f"{endpoint}_state_js",
        view_func=lambda: Response(
            _asset_text("state.js"),
            content_type="application/javascript; charset=utf-8",
        ),
    )


def streamlit_theme_css(mode: object = "auto") -> str:
    """Return semantic CSS for Streamlit without altering scientific canvases."""

    selected = normalize_theme_mode(mode)
    css = _asset_text("theme.css")
    streamlit_rules = """
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--solar-bg);
  color: var(--solar-text);
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
  background: var(--solar-surface);
  color: var(--solar-text);
  border-color: var(--solar-border);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
  color: var(--solar-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="textarea"] > div,
[data-testid="stAppViewContainer"] [data-baseweb="input"] > div,
[data-testid="stAppViewContainer"] [data-baseweb="textarea"] > div {
  background: var(--solar-surface-raised);
  border-color: var(--solar-border) !important;
  color: var(--solar-text) !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: transparent !important;
  color: var(--solar-text) !important;
  -webkit-text-fill-color: var(--solar-text) !important;
}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-tertiary"],
button[kind="secondary"],
button[kind="tertiary"] {
  background: var(--solar-surface-raised) !important;
  border-color: var(--solar-border) !important;
  color: var(--solar-text) !important;
}
[data-testid="stBaseButton-secondary"] :where(p, span),
[data-testid="stBaseButton-tertiary"] :where(p, span),
button[kind="secondary"] :where(p, span),
button[kind="tertiary"] :where(p, span) {
  color: inherit !important;
}
button[kind="primary"] {
  background: var(--solar-accent) !important;
  border-color: var(--solar-accent) !important;
  color: #ffffff !important;
}
button[kind="primary"] :where(p, span),
[data-testid="stBaseButton-primary"] :where(p, span) {
  color: #ffffff !important;
}
button[kind="primary"]:hover {
  background: var(--solar-accent-hover) !important;
  border-color: var(--solar-accent-hover) !important;
}
[data-baseweb="select"] > div {
  background: var(--solar-surface-raised) !important;
  border-color: var(--solar-border) !important;
  color: var(--solar-text) !important;
}
[data-baseweb="select"] input {
  color: var(--solar-text) !important;
  -webkit-text-fill-color: var(--solar-text) !important;
}
[data-baseweb="select"]:focus-within > div {
  border-color: var(--solar-focus) !important;
  box-shadow: 0 0 0 1px var(--solar-focus) !important;
}
[data-testid="stSelectbox"] div[role="group"] {
  background: var(--solar-surface-raised) !important;
  border-color: var(--solar-border) !important;
  color: var(--solar-text) !important;
}
[data-testid="stSelectbox"] div[role="group"]:has(input:focus) {
  border-color: var(--solar-focus) !important;
  box-shadow: 0 0 0 1px var(--solar-focus) !important;
}
[data-testid="stCheckbox"] label > div:first-of-type {
  border-color: var(--solar-border-strong) !important;
}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {
  background: var(--solar-accent) !important;
  border-color: var(--solar-accent) !important;
}
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
  background: var(--solar-accent) !important;
}
[data-testid="stSlider"] [role="group"] > div > div:has(input[type="range"]) {
  background: var(--solar-accent) !important;
}
[data-testid="stAlert"] {
  color: var(--solar-text);
  border-color: var(--solar-border);
}
[data-testid="stAlert"] :where(p, span) {
  color: inherit !important;
}
"""
    if selected == "auto":
        return css + streamlit_rules
    tokens = {
        "light": {
            "bg": "#f4f7fb",
            "surface": "#ffffff",
            "muted_surface": "#edf2f8",
            "raised": "#ffffff",
            "text": "#172033",
            "muted": "#5d6b80",
            "border": "#c9d4e3",
            "strong": "#9cabc0",
            "accent": "#2563eb",
            "hover": "#1d4ed8",
            "focus": "#60a5fa",
        },
        "dark": {
            "bg": "#0b1120",
            "surface": "#111a2c",
            "muted_surface": "#182338",
            "raised": "#162137",
            "text": "#e7edf7",
            "muted": "#a9b5c8",
            "border": "#33435d",
            "strong": "#53647e",
            "accent": "#60a5fa",
            "hover": "#93c5fd",
            "focus": "#93c5fd",
        },
    }[selected]
    explicit = f"""
html:root:root:root {{
  color-scheme: {selected};
  --solar-bg: {tokens['bg']};
  --solar-surface: {tokens['surface']};
  --solar-surface-muted: {tokens['muted_surface']};
  --solar-surface-raised: {tokens['raised']};
  --solar-text: {tokens['text']};
  --solar-text-muted: {tokens['muted']};
  --solar-border: {tokens['border']};
  --solar-border-strong: {tokens['strong']};
  --solar-accent: {tokens['accent']};
  --solar-accent-hover: {tokens['hover']};
  --solar-focus: {tokens['focus']};
}}
"""
    return css + explicit + streamlit_rules


def render_streamlit_theme(
    st: Any,
    *,
    frontend_id: str,
    state_store: Any | None = None,
    path_memory: Any | None = None,
) -> ThemeMode:
    """Render persistent theme/reset controls and inject the shared Streamlit skin."""

    saved = state_store.load(default={}) if state_store is not None else {}
    default = normalize_theme_mode(
        saved.get("theme") if isinstance(saved, dict) else None
    )
    key = f"{frontend_id}_theme_mode"
    if key not in st.session_state:
        st.session_state[key] = default
    with st.sidebar:
        selected = normalize_theme_mode(
            st.selectbox(
                "Theme",
                options=list(THEME_MODES),
                index=list(THEME_MODES).index(
                    normalize_theme_mode(st.session_state[key])
                ),
                key=key,
                format_func=lambda value: value.title(),
            )
        )
        if state_store is not None:
            state_store.update({"theme": selected})
        if st.button("Reset UI State", key=f"{frontend_id}_reset_ui_state"):
            remembered_fields = (
                saved.get("fields", {}) if isinstance(saved, dict) else {}
            )
            if state_store is not None:
                state_store.save(
                    {"theme": "auto", "fields": {}, "legacy_imported": True}
                )
            if path_memory is not None:
                path_memory.reset(frontend=frontend_id)
            if isinstance(remembered_fields, dict):
                for field in remembered_fields:
                    st.session_state.pop(str(field), None)
            for item in tuple(st.session_state):
                if item.startswith(f"{frontend_id}_"):
                    del st.session_state[item]
            st.rerun()
    st.markdown(
        f"<style>{streamlit_theme_css(selected)}</style>", unsafe_allow_html=True
    )
    return selected


def apply_plotly_chrome(figure: Any, mode: object) -> Any:
    """Theme Plotly chrome without changing traces, colorscales, or z ranges."""

    selected = normalize_theme_mode(mode)
    if selected == "auto":
        return figure
    dark = selected == "dark"
    figure.update_layout(
        paper_bgcolor="#0b1120" if dark else "#ffffff",
        plot_bgcolor="#111a2c" if dark else "#f4f7fb",
        font={"color": "#e7edf7" if dark else "#172033"},
    )
    grid = "#33435d" if dark else "#c9d4e3"
    figure.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    figure.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return figure


__all__ = [
    "THEME_MODES",
    "ThemeMode",
    "apply_plotly_chrome",
    "normalize_theme_mode",
    "register_theme_assets",
    "render_streamlit_theme",
    "streamlit_theme_css",
]
