"""Gaussian + spectrogram + drift-rate + Newkirk extrapolation pipeline.

The module-level imports stay light so CLI discovery and documentation checks
can run without importing NumPy/Pandas/Matplotlib. Heavy dependencies are loaded
inside the helpers that actually need scientific arrays or figures.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["main", "parse_radio_time_value", "run_pipeline"]

from solar_toolkit.radio.config import (
    load_drift_selection_product_config,
    load_newkirk_height_comparison_config,
    load_radio_diagnostic_presentation_config,
    load_radio_output_config,
    load_radio_user_config,
)
from solar_apps.workflows.radio.entrypoint_utils import (
    apply_output_overrides,
    apply_pipeline_output_overrides,
    build_legacy_config,
    load_workspace_config_overrides,
    parse_known_common_args,
    resolve_analysis_dir,
)
from .configs import DEFAULT_CONFIG_NAME
from solar_apps.workflows.common.image_naming import configured_radio_image_path
from solar_toolkit.radio.provenance import write_radio_provenance


def _pd():
    """Load Pandas at runtime, after the pipeline is committed to running."""
    import pandas as pd

    return pd


def _np():
    """Load NumPy lazily to keep entrypoint imports independent of BLAS."""
    import numpy as np

    return np


def _plt():
    """Load Matplotlib with the non-interactive backend used for saved figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mdates():
    """Load Matplotlib date helpers only for plotting/time-axis conversion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates

    return mdates


def _parse_args(argv=None):
    """Parse the shared radio pipeline CLI surface."""
    return parse_known_common_args(
        "Run the full radio burst Gaussian, drift-rate, and Newkirk-height pipeline.",
        default_config=DEFAULT_CONFIG_NAME,
        include_pipeline_outputs=True,
        argv=argv,
    )


def _run_pipeline(argv=None, *, config_name: str | None = None):
    pd = _pd()
    # Source-map generation and all downstream products share the package-owned
    # workflow, preserving the historical scientific decision path.
    from . import source_map_workflow

    args = _parse_args(argv)
    # Load the event config in layers: legacy source-map settings, output naming,
    # Newkirk diagnostics, drift-selection products, and presentation toggles.
    selected_config = config_name or args.config
    user_config, newkirk_cfg = load_radio_user_config(selected_config)
    output_cfg = load_radio_output_config(selected_config)
    newkirk_height_cfg = load_newkirk_height_comparison_config(selected_config)
    drift_product_cfg = load_drift_selection_product_config(selected_config)
    presentation_cfg = load_radio_diagnostic_presentation_config(selected_config)
    workspace_overrides = load_workspace_config_overrides(args)
    for target, section in (
        (newkirk_cfg, "newkirk"),
        (newkirk_height_cfg, "newkirk_height_comparison"),
        (drift_product_cfg, "drift_selection_products"),
        (presentation_cfg, "diagnostic_presentation"),
        (output_cfg, "output"),
    ):
        values = workspace_overrides.get(section)
        if isinstance(values, dict):
            target.update(values)

    user_config = apply_output_overrides(user_config, args)
    output_cfg = apply_pipeline_output_overrides(
        output_cfg,
        newkirk_cfg,
        drift_product_cfg,
        presentation_cfg,
        args,
    )
    cfg = build_legacy_config(user_config, source_map_workflow)
    cfg["enable_gaussian_overlay"] = True
    cfg["save_gaussian_diagnostics"] = True
    if user_config.get("drift_rate", {}).get("enabled", False):
        cfg["enable_spectrogram_panel"] = True
        cfg["enable_drift_rate_overlay"] = True

    source_map_workflow._run_source_map_config(cfg, argv=[])

    # Downstream stages consume the Gaussian diagnostics table rather than
    # re-fitting radio images, preserving the legacy scientific decision path.
    analysis_dir = resolve_analysis_dir(cfg)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_radio_provenance(
        analysis_dir,
        cfg,
        newkirk_config=newkirk_cfg,
        config_source=selected_config,
        cli_overrides=vars(args),
    )

    gaussian_csv = analysis_dir / cfg.get(
        "gaussian_diagnostics_csv", "radio_gaussian_fit_diagnostics.csv"
    )
    if not gaussian_csv.exists():
        raise FileNotFoundError(f"Gaussian diagnostics CSV not found: {gaussian_csv}")

    gaussian_df = pd.read_csv(gaussian_csv)
    valid_df = _valid_gaussian_centers(gaussian_df)
    valid_csv = analysis_dir / output_cfg.get(
        "valid_centers_csv", "radio_gaussian_valid_centers.csv"
    )
    valid_df.to_csv(valid_csv, index=False)

    if newkirk_cfg.get("enabled", True):
        newkirk_df = _build_gaussian_newkirk_table(valid_df, newkirk_cfg)
        newkirk_csv = analysis_dir / newkirk_cfg.get(
            "output_csv", "radio_gaussian_newkirk_extrapolated.csv"
        )
        newkirk_df.to_csv(newkirk_csv, index=False)
    else:
        newkirk_df = pd.DataFrame()
        newkirk_csv = None

    drift_result = _load_or_create_drift_diagnostics(
        cfg, drift_product_cfg, return_cache=True
    )
    if isinstance(drift_result, tuple):
        drift_df, spectrogram_cache = drift_result
    else:
        drift_df = drift_result
        spectrogram_cache = None
    if newkirk_cfg.get("enabled", True) and not drift_df.empty:
        drift_speed_df = _build_drift_newkirk_table(drift_df, newkirk_cfg)
    else:
        drift_speed_df = pd.DataFrame()
    drift_speed_csv = analysis_dir / newkirk_cfg.get(
        "drift_speed_csv", "radio_drift_newkirk_speed.csv"
    )
    drift_speed_df.to_csv(drift_speed_csv, index=False)

    # Plotting is deliberately kept after table generation so CSV/JSON products
    # remain available even if a later diagnostic figure fails.
    batch_generated_at = datetime.now(timezone.utc)
    _plot_gaussian_center_trajectory(
        valid_df,
        configured_radio_image_path(
            analysis_dir,
            "source_trajectory",
            valid_df,
            sequence=1,
            product="source_trajectory",
            generated_at=batch_generated_at,
        ),
    )
    _plot_gaussian_newkirk_height_time(
        newkirk_df,
        configured_radio_image_path(
            analysis_dir,
            "newkirk_height_time",
            newkirk_df,
            sequence=2,
            product="newkirk_height_time",
            generated_at=batch_generated_at,
        ),
    )
    _plot_drift_speed_comparison(
        drift_speed_df,
        configured_radio_image_path(
            analysis_dir,
            "newkirk_drift_speed_comparison",
            drift_speed_df,
            sequence=3,
            product="newkirk_drift_speed_comparison",
            generated_at=batch_generated_at,
        ),
    )
    if newkirk_height_cfg.get("enable", True):
        _run_newkirk_height_comparison(
            gaussian_df,
            analysis_dir,
            newkirk_height_cfg,
            newkirk_cfg,
            drift_df,
            presentation_cfg,
            cfg,
            spectrogram_cache,
            batch_generated_at,
        )
    print("[Pipeline] outputs:")
    print(f"  Gaussian diagnostics: {gaussian_csv}")
    print(f"  Valid Gaussian centers: {valid_csv}")
    if newkirk_csv is not None:
        print(f"  Gaussian Newkirk extrapolation: {newkirk_csv}")
    print(f"  Drift Newkirk speeds: {drift_speed_csv}")


def _run_newkirk_height_comparison(
    gaussian_df: pd.DataFrame,
    analysis_dir: Path,
    height_cfg: dict,
    newkirk_cfg: dict,
    drift_df: pd.DataFrame | None = None,
    presentation_cfg: dict | None = None,
    pipeline_cfg: dict | None = None,
    spectrogram_cache=None,
    batch_generated_at=None,
) -> None:
    from solar_toolkit.radio.height_comparison import (
        build_gaussian_newkirk_height_summary_table,
        build_gaussian_newkirk_height_table,
    )
    from solar_toolkit.radio.height_plots import (
        plot_gaussian_vs_newkirk_height_frequency,
        plot_gaussian_vs_newkirk_height_time,
        plot_height_residual_vs_frequency,
    )
    from solar_toolkit.radio.io import summarize_invalid_reasons

    pd = _pd()
    cfg = dict(height_cfg or {})
    batch_generated_at = batch_generated_at or datetime.now(timezone.utc)
    presentation = dict(presentation_cfg or {})
    for key in (
        "comparison_frequency_mhz",
        "drift_source_type_map",
        "drift_time_tolerance_s",
        "drift_frequency_tolerance_mhz",
        "max_adaptive_frequency_tolerance_mhz",
        "min_adaptive_frequency_tolerance_mhz",
        "reference_newkirk_assumption",
    ):
        if key in presentation and key not in cfg:
            cfg[key] = presentation[key]
    if cfg.get("solar_radius_arcsec") is None:
        cfg["solar_radius_arcsec"] = float(
            newkirk_cfg.get("solar_radius_arcsec", 959.63)
        )
    if drift_df is not None and not drift_df.empty:
        cfg["drift_selections"] = drift_df.to_dict("records")

    print("[Newkirk Height] enabled")
    height_df = build_gaussian_newkirk_height_table(gaussian_df, cfg)
    raw_table_path = analysis_dir / cfg.get(
        "raw_output_table_name", "gaussian_newkirk_height_rows.csv"
    )
    height_df.to_csv(raw_table_path, index=False)
    print(f"[Newkirk Height] raw comparison rows saved: {raw_table_path}")
    height_summary_df = build_gaussian_newkirk_height_summary_table(height_df, cfg)
    table_path = analysis_dir / cfg.get(
        "output_table_name", "gaussian_newkirk_height_comparison_table.csv"
    )
    height_summary_df.to_csv(table_path, index=False)
    print(f"[HeightComparison] summary table saved: {table_path}")

    skipped_rows = (
        int((~height_df["height_valid"].map(_truthy)).sum())
        if not height_df.empty
        else 0
    )
    skipped_reasons = summarize_invalid_reasons(
        height_df, "height_valid", "height_invalid_reason"
    )
    print(f"[Newkirk Height] invalid height rows: {skipped_rows}")
    print(f"[Newkirk Height] invalid height reasons: {skipped_reasons}")

    if cfg.get("plot_height_frequency", True):
        path = configured_radio_image_path(
            analysis_dir,
            cfg.get("height_frequency_plot_name", "newkirk_height_frequency"),
            height_df,
            sequence=4,
            product="newkirk_height_frequency",
            generated_at=batch_generated_at,
        )
        result = plot_gaussian_vs_newkirk_height_frequency(height_df, path, cfg)
        if result.get("status") == "saved":
            print(f"[Newkirk Height] frequency plot saved: {path}")
        else:
            print(
                f"[Newkirk Height] frequency plot skipped: {result.get('reason', 'unknown')}"
            )
    if cfg.get("plot_height_time", True):
        path = configured_radio_image_path(
            analysis_dir,
            cfg.get("height_time_plot_name", "newkirk_height_time"),
            height_df,
            sequence=5,
            product="newkirk_height_time",
            generated_at=batch_generated_at,
        )
        result = plot_gaussian_vs_newkirk_height_time(height_df, path, cfg)
        if result.get("status") == "saved":
            print(f"[Newkirk Height] time plot saved: {path}")
        else:
            print(
                f"[Newkirk Height] time plot skipped: {result.get('reason', 'unknown')}"
            )
    if cfg.get("plot_residual_frequency", True):
        path = configured_radio_image_path(
            analysis_dir,
            cfg.get(
                "height_residual_plot_name",
                "newkirk_height_residual_frequency",
            ),
            height_df,
            sequence=6,
            product="newkirk_height_residual_frequency",
            generated_at=batch_generated_at,
        )
        result = plot_height_residual_vs_frequency(height_df, path, cfg)
        if result.get("status") == "saved":
            print(f"[Newkirk Height] residual plot saved: {path}")
            if result.get("summary_csv"):
                print(
                    f"[Newkirk Height] residual summary saved: {result['summary_csv']}"
                )
        else:
            print(
                f"[Newkirk Height] residual plot skipped: {result.get('reason', 'unknown')}"
            )
    if presentation.get("enable", True):
        _run_frequency_priority_diagnostics(
            height_df,
            gaussian_df,
            drift_df if drift_df is not None else pd.DataFrame(),
            analysis_dir,
            presentation,
            pipeline_cfg or {},
            spectrogram_cache=spectrogram_cache,
            batch_generated_at=batch_generated_at,
        )


def _run_frequency_priority_diagnostics(
    height_df: pd.DataFrame,
    gaussian_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    analysis_dir: Path,
    presentation_cfg: dict,
    pipeline_cfg: dict,
    spectrogram_cache=None,
    batch_generated_at=None,
) -> None:
    from solar_toolkit.radio.frequency_priority_diagnostics import (
        build_frequency_priority_summary,
        build_selected_band_newkirk_height_speed_table,
        plot_drift_frequency_band_matching,
        plot_event_gaussian_newkirk_height_comparison,
        plot_event_newkirk_speed_frequency,
        plot_frequency_priority_summary,
        plot_gaussian_center_by_frequency_facets,
        plot_gaussian_center_trajectory_by_frequency,
        plot_height_time_by_frequency_facets,
        save_frequency_priority_summary_csv,
        save_newkirk_physical_consistency_report,
        write_frequency_priority_dashboard,
    )
    from solar_toolkit.radio.height_comparison import (
        build_gaussian_newkirk_height_summary_table,
    )

    cfg = dict(presentation_cfg or {})
    batch_generated_at = batch_generated_at or datetime.now(timezone.utc)
    print("[Frequency Priority] enabled")
    summary = build_frequency_priority_summary(height_df, gaussian_df, drift_df, cfg)
    csv_path = analysis_dir / cfg.get(
        "summary_csv_name", "radio_newkirk_frequency_priority_summary.csv"
    )
    save_frequency_priority_summary_csv(summary, csv_path)
    print(f"[Frequency Priority] summary CSV saved: {csv_path}")
    selected_band_path = analysis_dir / cfg.get(
        "selected_band_newkirk_table_name",
        "event_selected_band_newkirk_table.csv",
    )
    selected_band_table = build_selected_band_newkirk_height_speed_table(drift_df, cfg)
    selected_band_table.to_csv(selected_band_path, index=False)
    print(
        f"[Frequency Priority] selected-band Newkirk table saved: {selected_band_path}"
    )
    height_summary_table = build_gaussian_newkirk_height_summary_table(height_df, cfg)
    report_path = analysis_dir / cfg.get(
        "physical_consistency_report_name",
        "newkirk_physical_consistency_report.md",
    )
    report_result = save_newkirk_physical_consistency_report(
        selected_band_table, height_summary_table, report_path, cfg
    )
    print(f"[PhysicalCheck] report saved: {report_result['path']}")

    if cfg.get("enable_event_height_comparison", True):
        path = configured_radio_image_path(
            analysis_dir,
            cfg.get("event_height_comparison_name", "newkirk_height_comparison"),
            height_df,
            sequence=7,
            product="newkirk_height_comparison",
            generated_at=batch_generated_at,
        )
        result = plot_event_gaussian_newkirk_height_comparison(height_df, path, cfg)
        if result.get("status") == "saved":
            print(f"[Frequency Priority] event height comparison saved: {path}")
        else:
            print(
                "[Frequency Priority] event height comparison skipped: "
                f"{result.get('reason', 'unknown')}"
            )

    if cfg.get("enable_event_speed_frequency", True):
        path = configured_radio_image_path(
            analysis_dir,
            cfg.get("event_speed_frequency_name", "newkirk_speed_frequency_scatter"),
            selected_band_table,
            sequence=8,
            product="newkirk_speed_frequency_scatter",
            generated_at=batch_generated_at,
        )
        result = plot_event_newkirk_speed_frequency(selected_band_table, path, cfg)
        if result.get("status") == "saved":
            print(f"[Frequency Priority] event speed-frequency plot saved: {path}")
        else:
            print(
                "[Frequency Priority] event speed-frequency plot skipped: "
                f"{result.get('reason', 'unknown')}"
            )

    debug_outputs = []
    if cfg.get("enable_static_summary", True):
        debug_outputs.append(
            (
                "summary panel",
                configured_radio_image_path(
                    analysis_dir,
                    cfg.get(
                        "summary_panel_name",
                        "newkirk_frequency_priority_summary",
                    ),
                    height_df,
                    sequence=9,
                    product="newkirk_frequency_priority_summary",
                    generated_at=batch_generated_at,
                ),
                lambda path: plot_frequency_priority_summary(
                    height_df, gaussian_df, drift_df, path, cfg
                ),
            )
        )
    if cfg.get("enable_debug_center_facets", False):
        debug_outputs.append(
            (
                "center facets",
                configured_radio_image_path(
                    analysis_dir,
                    cfg.get("center_facets_name", "source_center_frequency_facets"),
                    gaussian_df,
                    sequence=10,
                    product="source_center_frequency_facets",
                    generated_at=batch_generated_at,
                ),
                lambda path: plot_gaussian_center_by_frequency_facets(
                    gaussian_df, path, cfg
                ),
            )
        )
    if cfg.get("enable_debug_height_time_facets", False):
        debug_outputs.append(
            (
                "height-time facets",
                configured_radio_image_path(
                    analysis_dir,
                    cfg.get(
                        "height_time_facets_name",
                        "height_time_frequency_facets",
                    ),
                    height_df,
                    sequence=11,
                    product="height_time_frequency_facets",
                    generated_at=batch_generated_at,
                ),
                lambda path: plot_height_time_by_frequency_facets(height_df, path, cfg),
            )
        )
    for label, path, func in debug_outputs:
        result = func(path)
        if result.get("status") == "saved":
            print(f"[Frequency Priority] {label} saved: {path}")
        else:
            print(
                f"[Frequency Priority] {label} skipped: "
                f"{result.get('reason', 'unknown')}"
            )
    if cfg.get("enable_debug_drift_band_matching", False):
        _run_drift_band_matching_plot(
            drift_df,
            analysis_dir,
            cfg,
            pipeline_cfg,
            plot_drift_frequency_band_matching,
            spectrogram_cache=spectrogram_cache,
            batch_generated_at=batch_generated_at,
        )
    if cfg.get("enable_debug_trajectory_by_frequency", False):
        trajectory_result = plot_gaussian_center_trajectory_by_frequency(
            gaussian_df, analysis_dir, cfg
        )
        if trajectory_result.get("status") == "saved":
            print(
                "[Frequency Priority] time-colored trajectories saved: "
                f"{len(trajectory_result.get('paths', []))} files"
            )
        else:
            print(
                "[Frequency Priority] time-colored trajectories skipped: "
                f"{trajectory_result.get('reason', 'unknown')}"
            )

    if cfg.get("enable_html_dashboard", True):
        path = analysis_dir / cfg.get(
            "dashboard_name", "radio_newkirk_frequency_priority_dashboard.html"
        )
        result = write_frequency_priority_dashboard(
            height_df, gaussian_df, drift_df, path, cfg
        )
        if result.get("status") == "saved":
            print(f"[Frequency Priority] dashboard saved: {path}")


def _run_drift_band_matching_plot(
    drift_df: pd.DataFrame,
    analysis_dir: Path,
    cfg: dict,
    pipeline_cfg: dict,
    plot_func,
    spectrogram_cache=None,
    batch_generated_at=None,
) -> None:
    if drift_df.empty:
        print("[Frequency Priority] drift band matching skipped: no_drift_rows")
        return
    cache = spectrogram_cache
    if cache is None:
        try:
            from solar_toolkit.radio.spectrogram import build_spectrogram_cache

            cache = build_spectrogram_cache(pipeline_cfg)
        except Exception as exc:
            print(f"[Frequency Priority] drift band matching skipped: {exc}")
            return
    if cache is None:
        print(
            "[Frequency Priority] drift band matching skipped: missing_spectrogram_cache"
        )
        return
    path = configured_radio_image_path(
        analysis_dir,
        cfg.get("drift_band_matching_name", "drift_frequency_band_matching"),
        drift_df,
        sequence=12,
        product="drift_frequency_band_matching",
        generated_at=batch_generated_at or datetime.now(timezone.utc),
    )
    result = plot_func(
        cache.data, cache.time_datetimes, cache.freq, drift_df, path, cfg
    )
    if result.get("status") == "saved":
        print(f"[Frequency Priority] drift band matching saved: {path}")
    else:
        print(
            "[Frequency Priority] drift band matching skipped: "
            f"{result.get('reason', 'unknown')}"
        )


def _valid_gaussian_centers(df: pd.DataFrame) -> pd.DataFrame:
    from solar_toolkit.radio.quicklook import filter_valid_gaussian_centers

    return filter_valid_gaussian_centers(df)


def _build_gaussian_newkirk_table(
    valid_df: pd.DataFrame, newkirk_cfg: dict
) -> pd.DataFrame:
    from solar_toolkit.radio.newkirk import (
        attach_newkirk_height_to_gaussian,
    )

    pd = _pd()
    frames = []
    for multiplier in newkirk_cfg.get("multipliers", [1]):
        for harmonic in newkirk_cfg.get("harmonics", [1]):
            frames.append(
                attach_newkirk_height_to_gaussian(
                    valid_df,
                    multiplier=multiplier,
                    harmonic=harmonic,
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_or_create_drift_diagnostics(
    cfg: dict, product_cfg: dict | None = None, *, return_cache: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, object | None]:
    from . import source_map_workflow
    from solar_toolkit.radio.drift_rate import (
        get_or_load_drift_rate_results,
        save_drift_rate_diagnostics_once,
    )
    from solar_toolkit.radio.spectrogram import build_spectrogram_cache

    pd = _pd()
    csv_path = Path(
        source_map_workflow._drift_output_path(cfg, "drift_rate_diagnostics_csv")
    )
    if csv_path.exists():
        drift_df = pd.read_csv(csv_path)
        cache = _save_drift_selection_products_from_cache(
            cfg, product_cfg, drift_df, csv_path.parent
        )
        return (drift_df, cache) if return_cache else drift_df
    if not cfg.get("enable_drift_rate_overlay", False):
        drift_df = pd.DataFrame()
        return (drift_df, None) if return_cache else drift_df
    cache = build_spectrogram_cache(cfg)
    if cache is None:
        drift_df = pd.DataFrame()
        return (drift_df, None) if return_cache else drift_df
    results = get_or_load_drift_rate_results(cache, cfg)
    save_drift_rate_diagnostics_once(results, cfg, cache.source_file)
    drift_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    _save_drift_selection_products(cache, product_cfg, drift_df, csv_path.parent)
    return (drift_df, cache) if return_cache else drift_df


def _save_drift_selection_products_from_cache(
    cfg: dict,
    product_cfg: dict | None,
    drift_df: pd.DataFrame,
    analysis_dir: Path,
) -> object | None:
    if not product_cfg or not product_cfg.get("enable", True) or drift_df.empty:
        return None
    try:
        from solar_toolkit.radio.spectrogram import build_spectrogram_cache

        cache = build_spectrogram_cache(cfg)
    except Exception as exc:
        print(f"[Drift selection products] skipped: {exc}")
        return None
    if cache is None:
        print("[Drift selection products] skipped: missing_spectrogram_cache")
        return None
    _save_drift_selection_products(cache, product_cfg, drift_df, analysis_dir)
    return cache


def _save_drift_selection_products(
    cache,
    product_cfg: dict | None,
    drift_df: pd.DataFrame,
    analysis_dir: Path,
) -> None:
    if not product_cfg or not product_cfg.get("enable", True) or drift_df.empty:
        return
    from solar_toolkit.radio.drift_products import save_drift_selection_artifacts

    out_dir = Path(analysis_dir) / product_cfg.get("output_subdir", "drift_selection")
    preview_cfg = dict(product_cfg)
    preview_cfg.update(
        {
            "cmap": cache.cmap,
            "vmin": cache.vmin,
            "vmax": cache.vmax,
            "colorbar_label": cache.cbar_label,
        }
    )
    try:
        result = save_drift_selection_artifacts(
            cache.data,
            cache.time_datetimes,
            cache.freq,
            drift_df,
            out_dir,
            source_file=cache.source_file,
            config=preview_cfg,
        )
    except Exception as exc:
        print(f"[Drift selection products] skipped: {exc}")
        return
    if result.get("status") == "saved":
        print(f"[Drift selection products] saved: {out_dir}")
    else:
        print(f"[Drift selection products] skipped: {result.get('reason', 'unknown')}")


def _build_drift_newkirk_table(
    drift_df: pd.DataFrame, newkirk_cfg: dict
) -> pd.DataFrame:
    from .physical_diagnostics_cli import (
        build_drift_newkirk_table,
    )

    return build_drift_newkirk_table(drift_df, newkirk_cfg)


def parse_radio_time_value(value):
    pd = _pd()
    digits = "".join(re.findall(r"\d", str(value)))
    if len(digits) < 14:
        return pd.NaT

    base = digits[:14]
    suffix = digits[14:]
    if suffix:
        if len(suffix) <= 3:
            microsecond = suffix.zfill(3) + "000"
        else:
            microsecond = suffix[:6].ljust(6, "0")
    else:
        microsecond = "000000"

    return pd.to_datetime(base + microsecond, format="%Y%m%d%H%M%S%f", errors="coerce")


def _plot_gaussian_center_trajectory(df: pd.DataFrame, path: Path) -> None:
    from solar_toolkit.radio.quicklook import plot_gaussian_center_trajectory

    plot_gaussian_center_trajectory(df, path)


def _plot_gaussian_center_trajectory_time_colored(df: pd.DataFrame, path: Path) -> None:
    pd = _pd()
    plt = _plt()
    mdates = _mdates()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    if not df.empty:
        data = df.copy()
        data["time_dt"] = data["time"].map(parse_radio_time_value)
        data["freq_num"] = pd.to_numeric(data.get("freq"), errors="coerce")
        data["center_x_num"] = pd.to_numeric(data["center_x_arcsec"], errors="coerce")
        data["center_y_num"] = pd.to_numeric(data["center_y_arcsec"], errors="coerce")
        data = data.dropna(
            subset=["time_dt", "freq_num", "center_x_num", "center_y_num"]
        ).sort_values("time_dt")
        if not data.empty:
            time_nums = mdates.date2num(data["time_dt"])
            ax.plot(
                data["center_x_num"],
                data["center_y_num"],
                color="0.55",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
            sc = ax.scatter(
                data["center_x_num"],
                data["center_y_num"],
                c=time_nums,
                cmap="plasma",
                s=34,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )
            for _, row in data.iterrows():
                ax.annotate(
                    f"{row['freq_num']:.0f}",
                    (row["center_x_num"], row["center_y_num"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                )
            if len(data) >= 2:
                start = data.iloc[-2]
                end = data.iloc[-1]
                ax.annotate(
                    "",
                    xy=(end["center_x_num"], end["center_y_num"]),
                    xytext=(start["center_x_num"], start["center_y_num"]),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
                )
            cbar = fig.colorbar(sc, ax=ax, label="Time (UT)")
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Gaussian center trajectory colored by time")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_gaussian_newkirk_height_time(df: pd.DataFrame, path: Path) -> None:
    pd = _pd()
    np = _np()
    plt = _plt()
    mdates = _mdates()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    if not df.empty:
        data = df.copy()
        data["time_dt"] = data["time"].map(parse_radio_time_value)
        data["newkirk_height_rsun_num"] = pd.to_numeric(
            data["newkirk_height_rsun"], errors="coerce"
        )
        data = data[
            data["time_dt"].notna() & np.isfinite(data["newkirk_height_rsun_num"])
        ]
        for (multiplier, harmonic), group in data.groupby(
            ["newkirk_multiplier", "newkirk_harmonic"]
        ):
            group = group.sort_values("time_dt")
            ax.plot(
                group["time_dt"],
                group["newkirk_height_rsun_num"],
                marker="o",
                linewidth=1.2,
                markersize=3,
                label=f"{multiplier:g}x H{harmonic:g}",
            )
        if not data.empty:
            ax.legend(fontsize=8, ncol=2)
            ax.set_xlim(data["time_dt"].min(), data["time_dt"].max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Newkirk height (Rsun above photosphere)")
    ax.set_title("Gaussian Newkirk height evolution")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_drift_speed_comparison(df: pd.DataFrame, path: Path) -> None:
    from solar_toolkit.radio.frequency_priority_diagnostics import (
        format_newkirk_case_label,
    )

    pd = _pd()
    np = _np()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=180)
    if not df.empty:
        data = df.copy()
        data["newkirk_case"] = (
            data["newkirk_multiplier"].map(lambda v: f"{float(v):g}x")
            + "H"
            + data["newkirk_harmonic"].map(lambda v: f"{float(v):g}")
        )
        case_labels = {
            row["newkirk_case"]: format_newkirk_case_label(
                row["newkirk_multiplier"], row["newkirk_harmonic"], compact=True
            )
            for _, row in data.drop_duplicates(subset=["newkirk_case"]).iterrows()
        }
        drift_rates = (
            data.assign(
                drift_rate_num=pd.to_numeric(
                    data.get("drift_rate_mhz_s"), errors="coerce"
                )
            )
            .drop_duplicates(subset=["label"])
            .set_index("label")["drift_rate_num"]
            .to_dict()
        )
        speed_col = (
            "newkirk_speed_km_s"
            if "newkirk_speed_km_s" in data.columns
            else "speed_km_s"
        )
        data["speed_km_s_num"] = pd.to_numeric(data[speed_col], errors="coerce")
        if "newkirk_speed_c" in data.columns:
            data["speed_c_num"] = pd.to_numeric(
                data["newkirk_speed_c"], errors="coerce"
            )
        else:
            data["speed_c_num"] = data["speed_km_s_num"] / 299792.458
        heatmap = data.pivot_table(
            index="label",
            columns="newkirk_case",
            values="speed_km_s_num",
            aggfunc="mean",
        )
        heatmap_c = data.pivot_table(
            index="label",
            columns="newkirk_case",
            values="speed_c_num",
            aggfunc="mean",
        )
        desired_cols = ["1xH1", "1xH2", "2xH1", "2xH2", "4xH1", "4xH2"]
        existing_cols = [col for col in desired_cols if col in heatmap.columns]
        extra_cols = [col for col in heatmap.columns if col not in existing_cols]
        heatmap = heatmap.reindex(columns=existing_cols + extra_cols)
        heatmap_c = heatmap_c.reindex(index=heatmap.index, columns=heatmap.columns)
        values = heatmap.to_numpy(dtype=float)
        c_values = heatmap_c.to_numpy(dtype=float)
        if values.size:
            finite_mean = np.nanmean(values) if np.isfinite(values).any() else np.nan
            im = ax.imshow(values, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(heatmap.columns)))
            ax.set_xticklabels(
                [case_labels.get(col, col) for col in heatmap.columns],
                rotation=40,
                ha="right",
                fontsize=8,
            )
            ax.set_yticks(np.arange(len(heatmap.index)))
            ax.set_yticklabels(
                [
                    (
                        f"{label}\n{drift_rates[label]:.2f} MHz/s"
                        if label in drift_rates and np.isfinite(drift_rates[label])
                        else label
                    )
                    for label in heatmap.index
                ]
            )
            for y_idx in range(values.shape[0]):
                for x_idx in range(values.shape[1]):
                    value = values[y_idx, x_idx]
                    c_value = c_values[y_idx, x_idx] if c_values.size else np.nan
                    if np.isfinite(value):
                        ax.text(
                            x_idx,
                            y_idx,
                            (
                                f"{value:.0f}\n{c_value:.2f}c"
                                if np.isfinite(c_value)
                                else f"{value:.0f}"
                            ),
                            ha="center",
                            va="center",
                            color="white" if value > finite_mean else "black",
                            fontsize=7,
                        )
            fig.colorbar(im, ax=ax, label="Newkirk-inferred exciter speed (km/s)")
    ax.set_xlabel("Density / emission assumption")
    ax.set_ylabel("Drift label")
    ax.set_title("Drift-rate-derived Newkirk exciter speed comparison")
    fig.text(
        0.5,
        0.02,
        "Note: 1xH2 and 4xH1 are degenerate because the inferred height depends on N*s^2.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _truthy(value) -> bool:
    from solar_toolkit.radio.io import truthy

    return truthy(value)


def run_pipeline(argv=None, *, config_name: str | None = None) -> int:
    """Run the complete package-owned radio pipeline."""

    result = _run_pipeline(argv, config_name=config_name)
    return result if isinstance(result, int) else 0


def main(config_name: str | None = None, argv=None) -> int:
    """Run the complete pipeline and return a process status code."""

    forwarded = list(sys.argv[1:] if argv is None else argv)
    if config_name is not None:
        forwarded.extend(["--config", config_name])
    return run_pipeline(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
