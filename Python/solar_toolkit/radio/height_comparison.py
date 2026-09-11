"""Gaussian projected-height versus Newkirk radial-height diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import parse_datetime_value, truthy
from .newkirk import (
    effective_density_factor,
    newkirk_assumption_label,
    newkirk_radius_from_frequency_mhz,
    plasma_density_from_frequency_mhz,
)

__all__ = [
    "DEFAULT_SELECTED_MODELS",
    "HEIGHT_COLUMNS",
    "build_gaussian_newkirk_height_summary_table",
    "build_gaussian_newkirk_height_table",
    "classify_source_type",
    "compute_gaussian_projected_height",
    "model_label",
]

DEFAULT_SELECTED_MODELS = [
    {"multiplier": 1.0, "harmonic": 1},
    {"multiplier": 1.0, "harmonic": 2},
    {"multiplier": 2.0, "harmonic": 1},
    {"multiplier": 2.0, "harmonic": 2},
    {"multiplier": 4.0, "harmonic": 1},
    {"multiplier": 4.0, "harmonic": 2},
]


def compute_gaussian_projected_height(x_arcsec, y_arcsec, solar_radius_arcsec):
    radius = _float_or_nan(solar_radius_arcsec)
    if not np.isfinite(radius) or radius <= 0:
        return {
            "gaussian_rho_rsun": np.nan,
            "gaussian_height_rsun": np.nan,
            "height_valid": False,
            "height_invalid_reason": "invalid_solar_radius_arcsec",
            "gaussian_projected_height_valid": False,
            "gaussian_projected_height_reason": "gaussian_fit_or_coordinate_invalid",
        }

    x = _float_or_nan(x_arcsec)
    y = _float_or_nan(y_arcsec)
    if not np.isfinite(x) or not np.isfinite(y):
        return {
            "gaussian_rho_rsun": np.nan,
            "gaussian_height_rsun": np.nan,
            "height_valid": False,
            "height_invalid_reason": "nonfinite_projected_coordinate",
            "gaussian_projected_height_valid": False,
            "gaussian_projected_height_reason": "gaussian_fit_or_coordinate_invalid",
        }

    rho = float(np.hypot(x, y) / radius)
    height = rho - 1.0
    if rho < 1.0:
        return {
            "gaussian_rho_rsun": rho,
            "gaussian_height_rsun": height,
            "height_valid": False,
            "height_invalid_reason": "inside_disk_projected_distance_only",
            "gaussian_projected_height_valid": False,
            "gaussian_projected_height_reason": "projected_inside_limb_or_bad_fit",
        }
    return {
        "gaussian_rho_rsun": rho,
        "gaussian_height_rsun": height,
        "height_valid": True,
        "height_invalid_reason": "ok",
        "gaussian_projected_height_valid": True,
        "gaussian_projected_height_reason": "valid_projected_height",
    }


def classify_source_type(row, config):
    """Classify a row without implying any Newkirk 2D source position."""
    for key in ("source_type", "burst_type", "type"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    time_value = parse_datetime_value(row.get("time"))
    freq = _row_frequency(row)
    if _matches_window_and_frequency(
        time_value,
        freq,
        config.get("TYPEIII_TIME_WINDOWS", []),
        config.get("TYPEIII_FREQ_RANGE"),
    ):
        return "typeIII"
    if _matches_window_and_frequency(
        time_value,
        freq,
        config.get("SPIKE_TIME_WINDOWS", []),
        config.get("SPIKE_FREQ_RANGE"),
    ):
        return "spike"
    return "unknown"


def build_gaussian_newkirk_height_table(gaussian_df, config, geometry_df=None):
    """Compare projected heights, optionally attaching independent 3D geometry.

    Geometry rows use source_id, height_rsun, geometry_valid, method, evidence_id;
    each source must have at most one solution. Compare ambiguous scenarios in
    separate calls. The original projected columns retain their definitions.
    """
    cfg = dict(config or {})
    df = pd.DataFrame(gaussian_df).copy()
    if df.empty:
        return pd.DataFrame(columns=HEIGHT_COLUMNS)
    if cfg.get("drift_selections"):
        from .frequency_priority_diagnostics import (
            apply_frequency_priority_drift_matching,
        )

        df = apply_frequency_priority_drift_matching(
            df,
            pd.DataFrame(cfg.get("drift_selections") or []),
            cfg,
        )

    solar_radius = _solar_radius_arcsec(cfg)
    rows = []
    for _, row in df.iterrows():
        freq = _row_frequency(row)
        drift_match = {
            "drift_label": _text_or_empty(row.get("drift_label")),
            "source_type": _text_or_empty(row.get("source_type")),
            "warning": _text_or_empty(row.get("drift_match_warning")),
        }
        if not drift_match["drift_label"] and not cfg.get("drift_selections"):
            drift_match.update(_match_drift_selection(row, freq, cfg))
        source_type = classify_source_type(row, cfg)
        if drift_match["source_type"] and source_type == "unknown":
            source_type = drift_match["source_type"]
        projected = compute_gaussian_projected_height(
            row.get("center_x_arcsec"),
            row.get("center_y_arcsec"),
            solar_radius,
        )
        gaussian_fit_success = _gaussian_fit_success(row)
        if not gaussian_fit_success and projected["height_invalid_reason"] == "ok":
            projected["height_valid"] = False
            projected["height_invalid_reason"] = "invalid_gaussian_fit"
            projected["gaussian_projected_height_valid"] = False
            projected["gaussian_projected_height_reason"] = (
                "gaussian_fit_or_coordinate_invalid"
            )

        for model in _selected_models(cfg):
            multiplier = float(
                model.get("multiplier", model.get("newkirk_multiplier", 1.0))
            )
            harmonic = model.get("harmonic", 1)
            effective_factor = effective_density_factor(multiplier, harmonic)
            density = _safe_physics_value(
                plasma_density_from_frequency_mhz, freq, harmonic=harmonic
            )
            radius = _safe_physics_value(
                newkirk_radius_from_frequency_mhz,
                freq,
                multiplier=multiplier,
                harmonic=harmonic,
            )
            newkirk_height = radius - 1.0 if np.isfinite(radius) else np.nan
            reason = projected["height_invalid_reason"]
            height_valid = bool(projected["height_valid"])
            if not np.isfinite(radius):
                height_valid = False
                reason = _append_reason(reason, "invalid_newkirk_inversion")

            residual = (
                projected["gaussian_height_rsun"] - newkirk_height
                if np.isfinite(projected["gaussian_height_rsun"])
                and np.isfinite(newkirk_height)
                else np.nan
            )
            ratio = (
                projected["gaussian_height_rsun"] / newkirk_height
                if np.isfinite(projected["gaussian_height_rsun"])
                and np.isfinite(newkirk_height)
                and newkirk_height != 0
                else np.nan
            )
            rows.append(
                {
                    "source_id": row.get("source_id", ""),
                    "time": row.get("time", ""),
                    "frequency_mhz": freq,
                    "source_type": source_type,
                    "drift_label": drift_match["drift_label"],
                    "drift_match_warning": drift_match.get("warning", ""),
                    "gaussian_x_arcsec": _float_or_nan(row.get("center_x_arcsec")),
                    "gaussian_y_arcsec": _float_or_nan(row.get("center_y_arcsec")),
                    "solar_radius_arcsec": solar_radius,
                    "gaussian_rho_rsun": projected["gaussian_rho_rsun"],
                    "gaussian_height_rsun": projected["gaussian_height_rsun"],
                    "gaussian_projected_height_valid": projected[
                        "gaussian_projected_height_valid"
                    ],
                    "gaussian_projected_height_reason": projected[
                        "gaussian_projected_height_reason"
                    ],
                    "newkirk_multiplier": multiplier,
                    "harmonic": harmonic,
                    "density_multiplier": multiplier,
                    "emission_harmonic": harmonic,
                    "effective_density_factor": effective_factor,
                    "newkirk_assumption_label": newkirk_assumption_label(
                        multiplier, harmonic
                    ),
                    "electron_density_cm3": density,
                    "newkirk_radius_rsun": radius,
                    "newkirk_height_rsun": newkirk_height,
                    "height_residual_rsun": residual,
                    "height_residual_arcsec": (
                        residual * solar_radius if np.isfinite(residual) else np.nan
                    ),
                    "height_ratio_gauss_to_newkirk": ratio,
                    "gaussian_fit_success": gaussian_fit_success,
                    "gaussian_quality_flag": str(row.get("quality_flag", "")),
                    "height_valid": height_valid,
                    "height_invalid_reason": reason,
                }
            )

    out = pd.DataFrame(rows)
    for column in HEIGHT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    if geometry_df is None:
        return out[HEIGHT_COLUMNS]
    geometry = pd.DataFrame(geometry_df).copy()
    required = {"source_id", "height_rsun", "geometry_valid", "method", "evidence_id"}
    if not required.issubset(geometry.columns) or "source_id" not in df:
        raise ValueError(
            "Independent geometry requires source_id, height_rsun, geometry_valid, method, evidence_id"
        )
    geometry = geometry.rename(
        columns={
            "height_rsun": "source_height_3d_rsun",
            "method": "geometry_method",
            "evidence_id": "geometry_evidence_id",
        }
    )
    out = out.merge(geometry, on="source_id", how="left", validate="many_to_one")
    out["height_3d_valid"] = (
        out.gaussian_fit_success
        & out.geometry_valid.fillna(False).map(truthy)
        & np.isfinite(out.source_height_3d_rsun)
        & np.isfinite(out.newkirk_height_rsun)
    )
    out["height_residual_3d_rsun"] = (
        out.source_height_3d_rsun - out.newkirk_height_rsun
    ).where(out.height_3d_valid)
    # New model-constrained geometry must not silently become independent data.
    # Legacy geometry tables without an evidence flag retain their old contract.
    if "independent_geometry_valid" in out:
        out["height_3d_independent_valid"] = (
            out.height_3d_valid
            & out.independent_geometry_valid.fillna(False).map(truthy)
        )
    else:
        out["height_3d_independent_valid"] = (
            out.height_3d_valid
            & ~out.geometry_method.astype(str).str.contains(
                "pfss|conditional", case=False, regex=True
            )
        )
    out["height_3d_independent_valid"] &= ~out.geometry_method.astype(str).str.contains(
        "pfss|conditional", case=False, regex=True
    )
    out["height_3d_evidence_class"] = np.where(
        out.height_3d_independent_valid, "independent", "conditional_or_unverified"
    )
    return out


def build_gaussian_newkirk_height_summary_table(height_df, config=None):
    """Summarize valid projected Gaussian heights against Newkirk radial heights."""
    cfg = dict(config or {})
    df = pd.DataFrame(height_df).copy()
    columns = _height_summary_columns()
    if df.empty:
        return pd.DataFrame(columns=columns)

    freqs = _summary_frequencies(df, cfg)
    reference = str(cfg.get("reference_newkirk_assumption") or "2xH2")
    unique = df.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    ).copy()
    if "gaussian_projected_height_valid" not in unique.columns:
        unique["gaussian_projected_height_valid"] = (
            pd.to_numeric(unique.get("gaussian_height_rsun"), errors="coerce") >= 0
        )

    rows = []
    for freq in freqs:
        group = unique[
            pd.to_numeric(unique.get("frequency_mhz"), errors="coerce").eq(float(freq))
        ]
        valid_mask = group["gaussian_projected_height_valid"].map(truthy)
        valid_heights = pd.to_numeric(
            group.loc[valid_mask, "gaussian_height_rsun"], errors="coerce"
        ).dropna()
        row = {
            "frequency_mhz": float(freq),
            "gaussian_valid_count": int(len(valid_heights)),
            "gaussian_invalid_count": int(len(group) - len(valid_heights)),
            "gaussian_projected_height_median_rsun": np.nan,
            "gaussian_projected_height_q25_rsun": np.nan,
            "gaussian_projected_height_q75_rsun": np.nan,
            "abs_delta_reference_rsun": np.nan,
            "relative_delta_reference": np.nan,
        }
        if not valid_heights.empty:
            row["gaussian_projected_height_median_rsun"] = float(valid_heights.median())
            row["gaussian_projected_height_q25_rsun"] = float(
                valid_heights.quantile(0.25)
            )
            row["gaussian_projected_height_q75_rsun"] = float(
                valid_heights.quantile(0.75)
            )

        for multiplier, harmonic in _canonical_assumptions():
            key = _assumption_key(multiplier, harmonic)
            row[f"newkirk_height_rsun_{key}"] = _height_for_assumption(
                df, freq, multiplier, harmonic
            )

        median = row["gaussian_projected_height_median_rsun"]
        ref_height = row.get(f"newkirk_height_rsun_{reference}", np.nan)
        if np.isfinite(median) and np.isfinite(ref_height):
            delta = abs(float(median) - float(ref_height))
            row["abs_delta_reference_rsun"] = delta
            row["relative_delta_reference"] = (
                delta / abs(float(ref_height)) if float(ref_height) != 0 else np.nan
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[columns]


def model_label(multiplier, harmonic) -> str:
    return f"{float(multiplier):g}× Newkirk, s={float(harmonic):g}"


def _canonical_assumptions():
    return [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (4.0, 1.0), (4.0, 2.0)]


def _assumption_key(multiplier, harmonic):
    return f"{float(multiplier):g}xH{float(harmonic):g}"


def _summary_frequencies(df, config):
    freqs = config.get("comparison_frequency_mhz")
    if freqs:
        return [float(freq) for freq in freqs]
    values = pd.to_numeric(df.get("frequency_mhz"), errors="coerce").dropna().unique()
    return [float(freq) for freq in sorted(values)]


def _height_for_assumption(df, freq, multiplier, harmonic):
    data = df[
        pd.to_numeric(df.get("frequency_mhz"), errors="coerce").eq(float(freq))
        & pd.to_numeric(df.get("newkirk_multiplier"), errors="coerce").eq(
            float(multiplier)
        )
        & pd.to_numeric(df.get("harmonic"), errors="coerce").eq(float(harmonic))
    ]
    values = pd.to_numeric(data.get("newkirk_height_rsun"), errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _height_summary_columns():
    columns = [
        "frequency_mhz",
        "gaussian_valid_count",
        "gaussian_invalid_count",
        "gaussian_projected_height_median_rsun",
        "gaussian_projected_height_q25_rsun",
        "gaussian_projected_height_q75_rsun",
    ]
    columns.extend(
        f"newkirk_height_rsun_{_assumption_key(multiplier, harmonic)}"
        for multiplier, harmonic in _canonical_assumptions()
    )
    columns.extend(["abs_delta_reference_rsun", "relative_delta_reference"])
    return columns


def _selected_models(config):
    models = config.get("selected_models") or DEFAULT_SELECTED_MODELS
    normalized = []
    for model in models:
        if not isinstance(model, dict):
            continue
        normalized.append(
            {
                "multiplier": float(
                    model.get("multiplier", model.get("newkirk_multiplier", 1.0))
                ),
                "harmonic": model.get("harmonic", 1),
            }
        )
    return normalized or list(DEFAULT_SELECTED_MODELS)


def _solar_radius_arcsec(config) -> float:
    value = config.get("solar_radius_arcsec")
    if value is None or not np.isfinite(_float_or_nan(value)) or float(value) <= 0:
        return 959.63
    return float(value)


def _matches_window_and_frequency(time_value, freq, windows, freq_range) -> bool:
    if not windows and freq_range is None:
        return False
    time_ok = not windows or any(
        _time_in_window(time_value, window) for window in windows
    )
    freq_ok = _frequency_in_range(freq, freq_range)
    return bool(time_ok and freq_ok)


def _time_in_window(value, window) -> bool:
    if value is None or not window:
        return False
    if isinstance(window, dict):
        start = parse_datetime_value(window.get("start"))
        end = parse_datetime_value(window.get("end"))
    else:
        start = parse_datetime_value(window[0]) if len(window) >= 1 else None
        end = parse_datetime_value(window[1]) if len(window) >= 2 else None
    if start is None or end is None:
        return False
    if end < start:
        start, end = end, start
    return start <= value <= end


def _frequency_in_range(freq, freq_range) -> bool:
    if freq_range is None:
        return True
    if not np.isfinite(freq):
        return False
    lo, hi = map(float, freq_range)
    if lo > hi:
        lo, hi = hi, lo
    return lo <= float(freq) <= hi


def _row_frequency(row) -> float:
    for key in ("frequency_mhz", "freq_mhz", "freq"):
        value = _float_or_nan(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def _gaussian_fit_success(row) -> bool:
    flag = str(row.get("quality_flag", "")).strip().lower()
    overlay = row.get("overlay_valid", True)
    trajectory = row.get("trajectory_valid", True)
    return bool(flag in {"ok", ""} and truthy(overlay) and truthy(trajectory))


def _match_drift_selection(row, frequency_mhz: float, config: dict) -> dict:
    selections = config.get("drift_selections") or []
    if isinstance(selections, pd.DataFrame):
        selections = selections.to_dict("records")
    if not selections:
        return {"drift_label": "", "source_type": ""}
    row_time = parse_datetime_value(row.get("time"))
    if row_time is None or not np.isfinite(frequency_mhz):
        return {"drift_label": "", "source_type": ""}
    time_tol = float(config.get("drift_time_tolerance_s", 1.0) or 0.0)
    raw_freq_tol = config.get("drift_frequency_tolerance_mhz", 5.0)
    if isinstance(raw_freq_tol, str):
        from .frequency_priority_diagnostics import (
            resolve_comparison_frequencies,
            resolve_drift_frequency_tolerance,
        )

        freq_tol = resolve_drift_frequency_tolerance(
            config, resolve_comparison_frequencies(config)
        )
    else:
        freq_tol = float(raw_freq_tol or 0.0)
    best = None
    best_delta = np.inf
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        expected = _frequency_on_drift_selection(row_time, selection, time_tol)
        if expected is None or not np.isfinite(expected):
            continue
        delta = abs(float(frequency_mhz) - expected)
        if delta <= freq_tol and delta < best_delta:
            best = selection
            best_delta = delta
    if best is None:
        return {"drift_label": "", "source_type": ""}
    return {
        "drift_label": _text_or_empty(best.get("label")),
        "source_type": _source_type_from_selection(best),
    }


def _frequency_on_drift_selection(row_time, selection: dict, time_tolerance_s: float):
    t_start = parse_datetime_value(selection.get("t_start"))
    t_end = parse_datetime_value(selection.get("t_end"))
    f_start = _float_or_nan(selection.get("f_start_mhz"))
    f_end = _float_or_nan(selection.get("f_end_mhz"))
    if (
        t_start is None
        or t_end is None
        or not np.isfinite(f_start)
        or not np.isfinite(f_end)
    ):
        return None
    if t_end < t_start:
        t_start, t_end = t_end, t_start
        f_start, f_end = f_end, f_start
    total = (t_end - t_start).total_seconds()
    offset = (row_time - t_start).total_seconds()
    if offset < -time_tolerance_s or offset > total + time_tolerance_s:
        return None
    if abs(total) <= 1e-12:
        return 0.5 * (f_start + f_end)
    fraction = min(1.0, max(0.0, offset / total))
    return f_start + fraction * (f_end - f_start)


def _source_type_from_selection(selection: dict) -> str:
    for key in ("source_type", "burst_type", "type"):
        value = _text_or_empty(selection.get(key))
        if value:
            return value
    return ""


def _text_or_empty(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_physics_value(func, *args, **kwargs) -> float:
    try:
        return _float_or_nan(func(*args, **kwargs))
    except Exception:
        return np.nan


def _append_reason(existing: str, reason: str) -> str:
    if not existing or existing == "ok":
        return reason
    if reason in str(existing).split(";"):
        return str(existing)
    return f"{existing};{reason}"


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


HEIGHT_COLUMNS = [
    "time",
    "frequency_mhz",
    "source_type",
    "drift_label",
    "drift_match_warning",
    "gaussian_x_arcsec",
    "gaussian_y_arcsec",
    "solar_radius_arcsec",
    "gaussian_rho_rsun",
    "gaussian_height_rsun",
    "gaussian_projected_height_valid",
    "gaussian_projected_height_reason",
    "newkirk_multiplier",
    "harmonic",
    "density_multiplier",
    "emission_harmonic",
    "effective_density_factor",
    "newkirk_assumption_label",
    "electron_density_cm3",
    "newkirk_radius_rsun",
    "newkirk_height_rsun",
    "height_residual_rsun",
    "height_residual_arcsec",
    "height_ratio_gauss_to_newkirk",
    "gaussian_fit_success",
    "gaussian_quality_flag",
    "height_valid",
    "height_invalid_reason",
]
