"""Radio pipeline diagnostic tables and derived product orchestration.

Scientific imports stay inside calls so command discovery remains lightweight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def configured_radio_image_path(
    output_dir: str | Path,
    configured_name: str,
    data,
    *,
    sequence: int,
    product: str,
    generated_at: datetime,
    frequency_mhz: float | None = None,
    polarization: str | Sequence[str] | None = None,
) -> Path:
    """Load the shared naming adapter only when a scientific product is made."""
    from solar_apps.workflows.common.image_naming import (
        configured_radio_image_path as build_path,
    )

    return build_path(
        output_dir,
        configured_name,
        data,
        sequence=sequence,
        product=product,
        generated_at=generated_at,
        frequency_mhz=frequency_mhz,
        polarization=polarization,
    )


def write_radio_provenance(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    newkirk_config: Mapping[str, Any] | None = None,
    config_source: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    filename: str = "radio_run_provenance.json",
) -> Path:
    """Load the scientific I/O adapter after command discovery has finished."""
    from solar_toolkit.radio.provenance import write_radio_provenance as write_record

    return write_record(
        output_dir,
        config,
        newkirk_config=newkirk_config,
        config_source=config_source,
        cli_overrides=cli_overrides,
        filename=filename,
    )


def _pd():
    """Load Pandas at runtime, after the pipeline is committed to running."""
    import pandas as pd

    return pd


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


def _truthy(value) -> bool:
    from solar_toolkit.radio.io import truthy

    return truthy(value)
