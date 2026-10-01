"""Gaussian + spectrogram + drift-rate + Newkirk extrapolation pipeline.

The module-level imports stay light so CLI discovery and documentation checks
can run without importing NumPy/Pandas/Matplotlib. Heavy dependencies are loaded
inside the helpers that actually need scientific arrays or figures.
"""

from __future__ import annotations

import re  # noqa: F401 - retained module namespace
import sys
from datetime import datetime, timezone
from pathlib import Path  # noqa: F401 - retained module namespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401 - retained annotation namespace

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


from ._pipeline_products import (  # noqa: F401
    configured_radio_image_path,
    write_radio_provenance,
    _build_drift_newkirk_table,
    _build_gaussian_newkirk_table,
    _load_or_create_drift_diagnostics,
    _pd,
    _run_drift_band_matching_plot,
    _run_frequency_priority_diagnostics,
    _run_newkirk_height_comparison,
    _save_drift_selection_products,
    _save_drift_selection_products_from_cache,
    _truthy,
    _valid_gaussian_centers,
)

from ._pipeline_plotting import (  # noqa: F401
    _mdates,
    _np,
    _plot_drift_speed_comparison,
    _plot_gaussian_center_trajectory,
    _plot_gaussian_center_trajectory_time_colored,
    _plot_gaussian_newkirk_height_time,
    _plt,
    parse_radio_time_value,
)


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

    source_map_workflow._run_source_map_config(cfg, argv=argv)

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
