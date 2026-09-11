"""Independent diagnostics for existing Gaussian and drift tables.

This service deliberately consumes persisted tables.  It never runs imaging,
spectrogram generation, or another upstream action.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solar_apps.workflows.common.image_naming import configured_radio_image_path
from solar_toolkit.radio.physical_diagnostics import build_drift_newkirk_table

from .configs import DEFAULT_CONFIG_NAME

__all__ = [
    "build_drift_newkirk_table",
    "build_parser",
    "main",
    "run_physical_diagnostics",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze existing Gaussian and/or drift tables without upstream work."
    )
    parser.add_argument("--gaussian-csv")
    parser.add_argument("--drift-csv")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-dir", default="physical_diagnostics")
    parser.add_argument(
        "--workspace-config-json",
        help="Structured Radio Workspace overrides encoded as a JSON object.",
    )
    return parser


def run_physical_diagnostics(
    *,
    gaussian_csv: str | Path | None = None,
    drift_csv: str | Path | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
    output_dir: str | Path,
    workspace_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate physical tables, summaries, reports, figures, and a dashboard."""

    import pandas as pd

    from solar_toolkit.radio.config import (
        load_radio_diagnostic_presentation_config,
        load_radio_user_config,
    )
    from solar_toolkit.radio.frequency_priority_diagnostics import (
        build_frequency_priority_summary,
        build_selected_band_newkirk_height_speed_table,
        plot_frequency_priority_summary,
        save_frequency_priority_summary_csv,
        save_newkirk_physical_consistency_report,
        write_frequency_priority_dashboard,
    )
    from solar_toolkit.radio.height_comparison import (
        build_gaussian_newkirk_height_summary_table,
        build_gaussian_newkirk_height_table,
    )
    from .quicklook import build_quicklook_config

    if not gaussian_csv and not drift_csv:
        raise ValueError("Provide --gaussian-csv, --drift-csv, or both.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overrides = dict(workspace_config or {})

    _user, newkirk_config = load_radio_user_config(config_name)
    newkirk_config.update(_mapping_section(overrides, "newkirk"))
    height_config = build_quicklook_config(config_name)
    height_config.update(_mapping_section(overrides, "newkirk_height_comparison"))
    presentation = load_radio_diagnostic_presentation_config(config_name)
    presentation.update(_mapping_section(overrides, "diagnostic_presentation"))

    gaussian_df = _read_optional_table(gaussian_csv)
    drift_df = _read_optional_table(drift_csv)
    if not drift_df.empty:
        height_config["drift_selections"] = drift_df.to_dict("records")
    height_df = (
        build_gaussian_newkirk_height_table(gaussian_df, height_config)
        if not gaussian_df.empty
        else pd.DataFrame()
    )
    height_path = output / "gaussian_newkirk_height_rows.csv"
    height_df.to_csv(height_path, index=False)

    drift_speed_df = build_drift_newkirk_table(drift_df, newkirk_config)
    drift_speed_path = output / "radio_drift_newkirk_speed.csv"
    drift_speed_df.to_csv(drift_speed_path, index=False)

    selected_band = build_selected_band_newkirk_height_speed_table(
        drift_df, presentation
    )
    selected_band_path = output / "event_selected_band_newkirk_table.csv"
    selected_band.to_csv(selected_band_path, index=False)
    height_summary = build_gaussian_newkirk_height_summary_table(
        height_df, presentation
    )
    height_summary_path = output / "gaussian_newkirk_height_summary.csv"
    height_summary.to_csv(height_summary_path, index=False)

    summary = build_frequency_priority_summary(
        height_df, gaussian_df, drift_df, presentation
    )
    summary_path = output / "radio_newkirk_frequency_priority_summary.csv"
    save_frequency_priority_summary_csv(summary, summary_path)
    report_path = output / "newkirk_physical_consistency_report.md"
    save_newkirk_physical_consistency_report(
        selected_band, height_summary, report_path, presentation
    )

    artifacts: dict[str, str] = {
        "height_rows_csv": str(height_path),
        "drift_speed_csv": str(drift_speed_path),
        "selected_band_csv": str(selected_band_path),
        "height_summary_csv": str(height_summary_path),
        "frequency_summary_csv": str(summary_path),
        "physical_consistency_report": str(report_path),
    }
    if presentation.get("enable_static_summary", True):
        naming_frame = (
            height_df
            if not height_df.empty
            else gaussian_df if not gaussian_df.empty else drift_df
        )
        figure_path = configured_radio_image_path(
            output,
            presentation.get(
                "summary_panel_name",
                "newkirk_frequency_priority_summary",
            ),
            naming_frame,
            sequence=1,
            product="newkirk_frequency_priority_summary",
            generated_at=datetime.now(timezone.utc),
        )
        result = plot_frequency_priority_summary(
            height_df, gaussian_df, drift_df, figure_path, presentation
        )
        if result.get("status") == "saved":
            artifacts["frequency_summary_plot"] = str(figure_path)
    if presentation.get("enable_html_dashboard", True):
        dashboard_path = output / "radio_newkirk_frequency_priority_dashboard.html"
        result = write_frequency_priority_dashboard(
            height_df, gaussian_df, drift_df, dashboard_path, presentation
        )
        if result.get("status") == "saved":
            artifacts["dashboard"] = str(dashboard_path)
    return {
        "gaussian_input": str(Path(gaussian_csv)) if gaussian_csv else None,
        "drift_input": str(Path(drift_csv)) if drift_csv else None,
        "artifacts": artifacts,
    }


def _read_optional_table(value: str | Path | None):
    import pandas as pd

    if value in (None, ""):
        return pd.DataFrame()
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostics table not found: {path}")
    # Preserve compact UTC timestamps exactly (17 digits exceed float precision).
    return pd.read_csv(path, dtype={"time": str})


def _mapping_section(value: dict[str, Any], key: str) -> dict[str, Any]:
    section = value.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"Workspace configuration section {key!r} must be an object")
    return dict(section)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_config = (
        json.loads(args.workspace_config_json) if args.workspace_config_json else {}
    )
    if not isinstance(workspace_config, dict):
        raise TypeError("--workspace-config-json must contain a JSON object")
    result = run_physical_diagnostics(
        gaussian_csv=args.gaussian_csv,
        drift_csv=args.drift_csv,
        config_name=args.config,
        output_dir=args.output_dir,
        workspace_config=workspace_config,
    )
    print(f"Physical diagnostics output: {Path(args.output_dir).resolve()}")
    print(f"Artifacts: {len(result['artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
