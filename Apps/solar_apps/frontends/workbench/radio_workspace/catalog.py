"""Built-in fused module catalog for the Radio Workspace."""

from __future__ import annotations

from typing import Any

from .contracts import SCHEMA_VERSION, RadioActionSpec, RadioModuleSpec

_PATH_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "path": {"path_kind": "directory", "extensions": []},
    "root": {"path_kind": "directory", "extensions": []},
    "single_file_path": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "radio_dir": {"path_kind": "directory", "extensions": []},
    "radio_root": {"path_kind": "directory", "extensions": []},
    "spectrogram_file": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "roi_json": {"path_kind": "file", "extensions": [".json"]},
    "primary_file": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "adjacent_file": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "spectrogram_image": {
        "path_kind": "file",
        "extensions": [".png", ".jpg", ".jpeg"],
    },
    "spectrogram_metadata": {"path_kind": "file", "extensions": [".json"]},
    "file_path": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "use_drift_selection": {"path_kind": "file", "extensions": [".json"]},
    "gaussian_csv": {"path_kind": "file", "extensions": [".csv"]},
    "drift_csv": {"path_kind": "file", "extensions": [".csv"]},
    "config_file": {"path_kind": "file", "extensions": [".json"]},
    "config": {"path_kind": "file", "extensions": [".json"]},
    "aia_dir": {"path_kind": "directory", "extensions": []},
    "hmi_dir": {"path_kind": "directory", "extensions": []},
    "center_csv": {"path_kind": "file", "extensions": [".csv"]},
    "aia_fits": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "tb_data": {"path_kind": "file", "extensions": [".npy"]},
    "radio_file": {
        "path_kind": "file",
        "extensions": [".fits", ".fit", ".fts"],
    },
    "centers": {
        "path_kind": "file",
        "extensions": [".csv", ".xlsx", ".xls"],
    },
}


def _field(
    name: str,
    label: str,
    field_type: str = "text",
    *,
    cli_flag: str | None = None,
    required: bool = False,
    path: bool = False,
    default: Any = None,
    choices: tuple[str, ...] = (),
    artifact_types: tuple[str, ...] = (),
    help_text: str = "",
    config_path: str | None = None,
    hidden: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "label": label,
        "type": field_type,
        "required": required,
        "path": path,
    }
    if cli_flag:
        payload["cli_flag"] = cli_flag
    if default is not None:
        payload["default"] = default
    if choices:
        payload["choices"] = list(choices)
    if artifact_types:
        payload["artifact_types"] = list(artifact_types)
    if help_text:
        payload["help"] = help_text
    if config_path:
        payload["config_path"] = config_path
    if hidden:
        payload["hidden"] = True
    if path:
        try:
            metadata = _PATH_FIELD_METADATA[name]
        except KeyError as exc:
            raise ValueError(
                f"Missing explicit path metadata for field {name!r}"
            ) from exc
        payload.update(metadata)
        payload["allow_multiple"] = False
    return payload


_ADVANCED_ARGUMENTS = _field(
    "arguments",
    "Additional arguments",
    "argv",
    help_text="Optional argument tokens. Shell command text is not accepted.",
)


MODULES: tuple[RadioModuleSpec, ...] = (
    RadioModuleSpec(
        id="data-configuration",
        title="Data & Configuration",
        group="Core",
        description=(
            "Choose local data, event settings, frequency, polarization, and time "
            "filters, or inspect raw-data quality without running later modules."
        ),
        default_enabled=False,
        default_collapsed=True,
        accepts_artifacts=("radio-fits",),
        produces_artifacts=("file-selection", "quality-table"),
        actions=(
            RadioActionSpec(
                id="browse-data",
                title="Browse Data Files",
                description="Inspect allowed local folders without loading FITS data.",
                preview_adapter="file-browser",
                produces_artifacts=("file-selection",),
                input_schema=(_field("path", "Folder", "path", path=True),),
            ),
            RadioActionSpec(
                id="raw-quality",
                title="Raw Quality Diagnostics",
                description=(
                    "Run file-level and time-frequency quality checks for selected "
                    "radio FITS data."
                ),
                command_module="solar_apps.workflows.radio.raw_quality_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                produces_artifacts=("quality-table", "diagnostic-image"),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _field(
                        "root",
                        "Radio data folder",
                        "path",
                        cli_flag="--root",
                        path=True,
                    ),
                    _field("freqs", "Frequencies (MHz)", cli_flag="--freqs"),
                    _field(
                        "polarizations", "Polarizations", cli_flag="--polarizations"
                    ),
                    _field(
                        "start_idx", "Start index", "number", cli_flag="--start-idx"
                    ),
                    _field("end_idx", "End index", "number", cli_flag="--end-idx"),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="imaging-localization",
        title="Imaging & Source Localization",
        group="Analysis",
        description=(
            "Create source maps, extract polarization-aware centers, and run shared "
            "single- or multi-source Gaussian diagnostics."
        ),
        accepts_artifacts=("radio-fits",),
        produces_artifacts=("source-map", "center-table", "gaussian-table"),
        actions=(
            RadioActionSpec(
                id="inspect-source-map",
                title="Inspect Source Map",
                description="Create radio source maps with configured overlays.",
                command_module="solar_apps.workflows.radio.source_map_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                preview_adapter="source-map-selection",
                run_required_fields=("selected_source_map_json",),
                default_config={
                    "features": {
                        "gaussian_overlay": False,
                        "save_gaussian_diagnostics": False,
                        "spectrogram_panel": False,
                    }
                },
                accepts_artifacts=("radio-fits",),
                produces_artifacts=("source-map",),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _field(
                        "mode",
                        "Input mode",
                        "select",
                        default="multi_band",
                        choices=("multi_band", "single_band"),
                        config_path="mode",
                    ),
                    _field(
                        "polarization",
                        "Polarization mode",
                        "select",
                        default="RR+LL",
                        choices=("RR+LL", "RR", "LL"),
                        config_path="data.polarization",
                    ),
                    _field(
                        "combine_polarizations",
                        "Combine RR and LL",
                        "checkbox",
                        default=True,
                        config_path="data.combine_polarizations",
                        hidden=True,
                    ),
                    _field(
                        "single_file_path",
                        "Single radio FITS",
                        "path",
                        path=True,
                        artifact_types=("radio-fits",),
                        config_path="data.single_file_path",
                    ),
                    _field(
                        "radio_dir",
                        "Multi-band radio folder",
                        "path",
                        path=True,
                        config_path="data.multi_band_root",
                    ),
                    _field(
                        "selected_source_map_json",
                        "Selected source-map preview",
                        "json",
                        config_path="data.selected_source_map_json",
                        hidden=True,
                        help_text=(
                            "Managed by Source Map Preview. Run uses only this "
                            "explicitly selected file or time slot."
                        ),
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="extract-centers",
                title="Extract Centers",
                description=(
                    "Extract threshold or contour centers with optional LCP/RCP pairing."
                ),
                command_module="solar_apps.workflows.radio.centers_application",
                output_flag="--out",
                output_filename="radio_centers.csv",
                produces_artifacts=("center-table",),
                input_schema=(
                    _field(
                        "radio_dir",
                        "Radio data folder",
                        "path",
                        cli_flag="--radio-dir",
                        required=True,
                        path=True,
                    ),
                    _field(
                        "pattern",
                        "FITS pattern",
                        cli_flag="--pattern",
                        default="*.fits",
                    ),
                    _field(
                        "recursive",
                        "Search subfolders",
                        "checkbox",
                        cli_flag="--recursive",
                    ),
                    _field("freqs", "Frequencies (MHz)", cli_flag="--freqs"),
                    _field(
                        "polarizations", "Polarizations", cli_flag="--polarizations"
                    ),
                    _field("time_start", "Start time", cli_flag="--time-start"),
                    _field("time_end", "End time", cli_flag="--time-end"),
                    _field(
                        "threshold",
                        "Threshold",
                        "number",
                        cli_flag="--threshold",
                        default=0.95,
                    ),
                    _field(
                        "make_sum",
                        "Create L+R centers",
                        "checkbox",
                        cli_flag="--make-sum",
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="fit-gaussian",
                title="Fit Gaussian Sources",
                description=(
                    "Run the canonical source-map Gaussian fit and diagnostic outputs."
                ),
                command_module="solar_apps.workflows.radio.source_map_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                default_config={
                    "features": {
                        "gaussian_overlay": True,
                        "save_gaussian_diagnostics": True,
                        "spectrogram_panel": False,
                    }
                },
                accepts_artifacts=("radio-fits",),
                produces_artifacts=("gaussian-table", "diagnostic-image"),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _field(
                        "mode",
                        "Input mode",
                        "select",
                        default="multi_band",
                        choices=("multi_band", "single_band"),
                        config_path="mode",
                    ),
                    _field(
                        "single_file_path",
                        "Single radio FITS",
                        "path",
                        path=True,
                        artifact_types=("radio-fits",),
                        config_path="data.single_file_path",
                    ),
                    _field(
                        "radio_dir",
                        "Multi-band radio folder",
                        "path",
                        path=True,
                        config_path="data.multi_band_root",
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="rrll-percentile-comparison",
                title="RR/LL Percentile Comparison",
                description=(
                    "Compare fixed percentile display ranges for selected RR/LL data."
                ),
                command_module=(
                    "solar_apps.workflows.radio.rrll_percentile_preview_comparison"
                ),
                output_flag="--output-dir",
                section="advanced",
                risk_level="advanced",
                accepts_artifacts=("cso-data",),
                produces_artifacts=("preview-image", "provenance-json"),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        required=True,
                        path=True,
                    ),
                    _field(
                        "radio_root",
                        "Multi-band radio root",
                        "path",
                        cli_flag="--radio-root",
                        required=True,
                        path=True,
                    ),
                    _field(
                        "spectrogram_file",
                        "Spectrogram FITS",
                        "path",
                        cli_flag="--spectrogram-file",
                        required=True,
                        path=True,
                        artifact_types=("cso-data",),
                    ),
                    _field("run_stem", "Run stem", cli_flag="--run-stem"),
                    _field("run_tag", "Run tag", cli_flag="--run-tag"),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="roi-light-curves",
        title="ROI & Light Curves",
        group="Analysis",
        description=(
            "Select a box or lasso ROI and extract raw sum, mean, or peak light curves."
        ),
        accepts_artifacts=("radio-fits", "roi-selection"),
        produces_artifacts=("roi-selection", "lightcurve-table", "lightcurve-image"),
        actions=(
            RadioActionSpec(
                id="select-roi",
                title="Select ROI",
                description="Select and save a box or lasso ROI on a reference image.",
                command_module="solar_apps.workflows.radio.roi_selection_cli",
                output_flag="--output-dir",
                preview_adapter="roi-selection",
                produces_artifacts=("roi-selection",),
                run_required_fields=("roi_json_payload",),
                input_schema=(
                    _field(
                        "radio_dir",
                        "Radio data folder",
                        "path",
                        required=True,
                        path=True,
                    ),
                    _field(
                        "pattern",
                        "FITS pattern",
                        default="*.fits",
                    ),
                    _field(
                        "recursive",
                        "Search subfolders",
                        "checkbox",
                        default=True,
                    ),
                    _field(
                        "roi_mode",
                        "Selection mode",
                        "select",
                        default="box",
                        choices=("box", "lasso"),
                    ),
                    _field(
                        "roi_json_payload",
                        "ROI selection JSON",
                        "json",
                        cli_flag="--roi-json-payload",
                        help_text=(
                            "Preview the reference image and drag a box or lasso; "
                            "the same-page plot writes the selection here."
                        ),
                    ),
                    _field(
                        "selected_files_json",
                        "Selected reference files",
                        "json",
                        hidden=True,
                        help_text=(
                            "Managed by the same-page candidate file selector. "
                            "This selection changes the reference preview only."
                        ),
                    ),
                ),
            ),
            RadioActionSpec(
                id="extract-light-curves",
                title="Extract Light Curves",
                description=(
                    "Extract ROI statistics and write tables, reference images, and plots."
                ),
                command_module="solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_application",
                output_flag="--out-dir",
                accepts_artifacts=("roi-selection",),
                produces_artifacts=("lightcurve-table", "lightcurve-image"),
                run_required_any_fields=("roi_bounds", "roi_json"),
                input_schema=(
                    _field(
                        "radio_dir",
                        "Radio data folder",
                        "path",
                        cli_flag="--radio-dir",
                        required=True,
                        path=True,
                    ),
                    _field("roi_bounds", "ROI bounds", cli_flag="--roi-bounds"),
                    _field(
                        "roi_json",
                        "ROI JSON",
                        "path",
                        cli_flag="--roi-json",
                        path=True,
                        artifact_types=("roi-selection",),
                    ),
                    _field(
                        "pattern",
                        "FITS pattern",
                        cli_flag="--pattern",
                        default="*.fits",
                    ),
                    _field(
                        "no_recursive",
                        "Do not search subfolders",
                        "checkbox",
                        cli_flag="--no-recursive",
                    ),
                    _field("freqs", "Frequencies (MHz)", cli_flag="--freqs"),
                    _field(
                        "polarization",
                        "Polarization",
                        "select",
                        cli_flag="--polarization",
                        default="L+R",
                        choices=("L+R", "LCP", "RCP", "all"),
                    ),
                    _field(
                        "metric",
                        "Statistic",
                        "select",
                        cli_flag="--metric",
                        default="raw_sum",
                        choices=("raw_sum", "raw_mean", "raw_peak"),
                    ),
                    _field(
                        "time_start",
                        "Start time (inclusive)",
                        cli_flag="--time-start",
                    ),
                    _field(
                        "time_end",
                        "End time (inclusive)",
                        cli_flag="--time-end",
                    ),
                    _field(
                        "pair_time_tolerance_sec",
                        "LCP/RCP pair tolerance (seconds)",
                        "number",
                        cli_flag="--pair-time-tolerance-sec",
                        default=0.5,
                    ),
                    _field(
                        "selected_products",
                        "Products",
                        "multiselect",
                        cli_flag="--selected-products",
                        required=True,
                        default=[
                            "csv",
                            "json",
                            "reference_png",
                            "lightcurve_png",
                            "lightcurve_detail_png",
                            "lightcurve_normalized_png",
                        ],
                        choices=(
                            "csv",
                            "json",
                            "reference_png",
                            "lightcurve_png",
                            "lightcurve_detail_png",
                            "lightcurve_normalized_png",
                        ),
                        help_text=(
                            "Choose one or more exports. Workspace defaults include "
                            "the detailed and normalized plots; the CLI default remains "
                            "the four compatibility products."
                        ),
                    ),
                    _field(
                        "detail_frequency_mhz",
                        "Detail plot frequency (MHz)",
                        "number",
                        cli_flag="--detail-frequency-mhz",
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="spectrogram-drift",
        title="Spectrogram & Drift",
        group="Analysis",
        description=(
            "Build LL, RR, total-intensity, and polarization-ratio spectra and apply "
            "an explicit drift selection."
        ),
        accepts_artifacts=("cso-data", "drift-selection"),
        produces_artifacts=(
            "spectrogram",
            "spectrogram-metadata",
            "drift-selection",
            "drift-table",
        ),
        actions=(
            RadioActionSpec(
                id="rebuild-spectrogram-coverage",
                title="Rebuild Spectrogram Coverage",
                description=(
                    "Explicitly combine a primary CSO FITS file with selected "
                    "adjacent FITS files for a same-page Preview. Real gaps longer "
                    "than one second remain empty; this never runs an upstream action."
                ),
                preview_adapter="spectrogram-coverage",
                accepts_artifacts=("cso-data",),
                produces_artifacts=(),
                input_schema=(
                    _field(
                        "primary_file",
                        "Primary CSO FITS file",
                        "path",
                        required=True,
                        path=True,
                        artifact_types=("cso-data",),
                    ),
                    _field(
                        "adjacent_file",
                        "Adjacent CSO FITS artifact or file",
                        "path",
                        path=True,
                        artifact_types=("cso-data",),
                        help_text=(
                            "Optional adjacent FITS input. This field supports a "
                            "workspace cso-data artifact without copying it."
                        ),
                    ),
                    _field(
                        "adjacent_files_json",
                        "Adjacent CSO FITS files",
                        "json",
                        help_text=(
                            "Optional JSON array of allowed local FITS paths. Each "
                            "path is validated before any FITS data is read."
                        ),
                    ),
                    _field(
                        "frequency_start",
                        "Minimum frequency (MHz)",
                        "number",
                        default=80.0,
                    ),
                    _field(
                        "frequency_end",
                        "Maximum frequency (MHz)",
                        "number",
                        default=340.0,
                    ),
                    _field(
                        "polarization",
                        "Spectrum",
                        "select",
                        default="sum",
                        choices=("LL", "RR", "sum", "ratio"),
                    ),
                    _field(
                        "rebin_time",
                        "Target time samples",
                        "number",
                        default=1000,
                    ),
                    _field(
                        "rebin_frequency",
                        "Target frequency samples",
                        "number",
                        default=700,
                    ),
                    _field(
                        "use_log10",
                        "Use log10 intensity",
                        "checkbox",
                        default=True,
                    ),
                    _field(
                        "cmap",
                        "Color map",
                        "select",
                        default="jet",
                        choices=(
                            "jet",
                            "viridis",
                            "plasma",
                            "inferno",
                            "magma",
                            "cividis",
                        ),
                    ),
                    _field("vmin", "Display minimum", "number"),
                    _field("vmax", "Display maximum", "number"),
                ),
            ),
            RadioActionSpec(
                id="select-drift-lines",
                title="Select Drift Lines",
                description=(
                    "Choose two endpoints per drift line on an existing spectrogram "
                    "preview without starting a spectrum pipeline."
                ),
                command_module="solar_apps.workflows.radio.drift_selection_cli",
                output_flag="--output-dir",
                preview_adapter="drift-selection",
                accepts_artifacts=("spectrogram", "spectrogram-metadata"),
                produces_artifacts=("drift-selection", "drift-table"),
                run_required_fields=("drift_lines_json",),
                input_schema=(
                    _field(
                        "spectrogram_image",
                        "Spectrogram preview image",
                        "path",
                        required=True,
                        path=True,
                        artifact_types=("spectrogram",),
                    ),
                    _field(
                        "spectrogram_metadata",
                        "Spectrogram metadata JSON",
                        "path",
                        required=True,
                        path=True,
                        artifact_types=("spectrogram-metadata",),
                    ),
                    _field(
                        "drift_lines_json",
                        "Selected drift lines JSON",
                        "json",
                        cli_flag="--drift-lines-json",
                        help_text=(
                            "Preview the spectrogram and click two endpoints for "
                            "each line; the same-page plot writes them here."
                        ),
                    ),
                ),
            ),
            RadioActionSpec(
                id="dynamic-spectrum-drift",
                title="Dynamic Spectrum & Drift",
                description=(
                    "Run CSO spectra without opening a second browser and optionally "
                    "reuse an existing drift-selection JSON file."
                ),
                command_module="solar_apps.workflows.radio.cso_workflow",
                output_flag="--output-dir",
                fixed_arguments=(
                    "--no-drift-browser",
                    "--drift-launch-policy",
                    "cli_only",
                    "--export-drift-preview",
                ),
                blocked_arguments=(
                    "--select-drift",
                    "--drift-port",
                    "--drift-launch-policy",
                ),
                accepts_artifacts=("cso-data", "drift-selection"),
                produces_artifacts=(
                    "spectrogram",
                    "spectrogram-metadata",
                    "drift-table",
                ),
                input_schema=(
                    _field(
                        "file_path",
                        "CSO FITS file",
                        "path",
                        cli_flag="--file-path",
                        required=True,
                        path=True,
                        artifact_types=("cso-data",),
                    ),
                    _field("time_start", "Start time (UTC)", cli_flag="--time-start"),
                    _field("time_end", "End time (UTC)", cli_flag="--time-end"),
                    _field(
                        "frequency_start",
                        "Minimum frequency (MHz)",
                        "number",
                        cli_flag="--frequency-start",
                    ),
                    _field(
                        "frequency_end",
                        "Maximum frequency (MHz)",
                        "number",
                        cli_flag="--frequency-end",
                    ),
                    _field(
                        "rebin_time",
                        "Target time samples",
                        "number",
                        cli_flag="--rebin-time",
                    ),
                    _field(
                        "rebin_frequency",
                        "Target frequency samples",
                        "number",
                        cli_flag="--rebin-frequency",
                    ),
                    _field(
                        "max_workers",
                        "Maximum workers",
                        "number",
                        cli_flag="--max-workers",
                    ),
                    _field(
                        "plot_ll",
                        "Include LL intensity",
                        "checkbox",
                        cli_flag="--plot-ll",
                    ),
                    _field(
                        "plot_rr",
                        "Include RR intensity",
                        "checkbox",
                        cli_flag="--plot-rr",
                    ),
                    _field(
                        "no_plot_sum",
                        "Exclude total intensity",
                        "checkbox",
                        cli_flag="--no-plot-sum",
                    ),
                    _field(
                        "no_plot_ratio",
                        "Exclude polarization ratio",
                        "checkbox",
                        cli_flag="--no-plot-ratio",
                    ),
                    _field(
                        "use_drift_selection",
                        "Drift selection JSON",
                        "path",
                        cli_flag="--use-drift-selection",
                        path=True,
                        artifact_types=("drift-selection",),
                    ),
                    _field(
                        "disable_drift",
                        "Disable drift overlay",
                        "checkbox",
                        cli_flag="--disable-drift",
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="cso-legacy-mode",
                title="CSO Legacy Mode",
                description="Run the compatibility CSO workflow with explicit arguments.",
                command_module="solar_apps.workflows.radio.cso_workflow",
                output_flag="--output-dir",
                fixed_arguments=(
                    "--no-drift-browser",
                    "--drift-launch-policy",
                    "cli_only",
                    "--export-drift-preview",
                ),
                blocked_arguments=(
                    "--select-drift",
                    "--drift-port",
                    "--drift-launch-policy",
                ),
                section="advanced",
                risk_level="advanced",
                accepts_artifacts=("cso-data", "drift-selection"),
                produces_artifacts=(
                    "spectrogram",
                    "spectrogram-metadata",
                    "drift-table",
                ),
                input_schema=(
                    _field(
                        "file_path",
                        "CSO FITS file",
                        "path",
                        cli_flag="--file-path",
                        required=True,
                        path=True,
                        artifact_types=("cso-data",),
                    ),
                    _field("time_start", "Start time (UTC)", cli_flag="--time-start"),
                    _field("time_end", "End time (UTC)", cli_flag="--time-end"),
                    _field(
                        "frequency_start",
                        "Minimum frequency (MHz)",
                        "number",
                        cli_flag="--frequency-start",
                    ),
                    _field(
                        "frequency_end",
                        "Maximum frequency (MHz)",
                        "number",
                        cli_flag="--frequency-end",
                    ),
                    _field(
                        "use_drift_selection",
                        "Drift selection JSON",
                        "path",
                        cli_flag="--use-drift-selection",
                        path=True,
                        artifact_types=("drift-selection",),
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="physical-diagnostics",
        title="Physical Diagnostics",
        group="Analysis",
        description=(
            "Compare Gaussian centers with Newkirk heights and drift-derived speeds."
        ),
        accepts_artifacts=("gaussian-table", "drift-table"),
        produces_artifacts=("physics-table", "physics-dashboard"),
        actions=(
            RadioActionSpec(
                id="analyze-existing-tables",
                title="Analyze Existing Tables",
                description=(
                    "Build Newkirk heights, drift speeds, frequency summaries, a "
                    "consistency report, and a dashboard from saved tables only."
                ),
                command_module="solar_apps.workflows.radio.physical_diagnostics_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                accepts_artifacts=("gaussian-table", "drift-table"),
                run_required_any_fields=("gaussian_csv", "drift_csv"),
                produces_artifacts=(
                    "physics-table",
                    "physics-report",
                    "physics-dashboard",
                    "diagnostic-image",
                ),
                input_schema=(
                    _field(
                        "gaussian_csv",
                        "Existing Gaussian table",
                        "path",
                        cli_flag="--gaussian-csv",
                        path=True,
                        artifact_types=("gaussian-table",),
                    ),
                    _field(
                        "drift_csv",
                        "Existing drift table",
                        "path",
                        cli_flag="--drift-csv",
                        path=True,
                        artifact_types=("drift-table",),
                    ),
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="gaussian-newkirk-quicklook",
                title="Gaussian/Newkirk Quicklook",
                description="Create center, height, residual, and trajectory quicklooks.",
                command_module="solar_apps.workflows.radio.quicklook",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                accepts_artifacts=("gaussian-table",),
                produces_artifacts=("physics-table", "diagnostic-image"),
                input_schema=(
                    _field(
                        "gaussian_csv",
                        "Gaussian results",
                        "path",
                        cli_flag="--gaussian-csv",
                        path=True,
                        artifact_types=("gaussian-table",),
                    ),
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="legacy-full-pipeline",
                title="Legacy Full Pipeline",
                description=(
                    "Compatibility orchestration for the existing complete radio pipeline."
                ),
                command_module="solar_apps.workflows.radio.pipeline_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                section="advanced",
                risk_level="advanced",
                produces_artifacts=(
                    "gaussian-table",
                    "physics-table",
                    "physics-dashboard",
                ),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="context-overlays",
        title="Context & Overlays",
        group="Context",
        description=(
            "Overlay radio contours, centers, and Gaussian fits on AIA, HMI, or DEM context."
        ),
        accepts_artifacts=(
            "radio-fits",
            "aia-fits",
            "center-table",
            "gaussian-table",
            "dem-map",
        ),
        produces_artifacts=(
            "overlay-image",
            "overlay-animation",
            "overlay-metadata",
        ),
        actions=(
            RadioActionSpec(
                id="aia-radio-hmi-overlay",
                title="AIA/Radio/HMI Overlay",
                description="Generate time-matched multi-instrument overlays.",
                command_module="solar_apps.workflows.radio.overlay_cli",
                output_flag="--output-dir",
                config_json_flag="--workspace-config-json",
                produces_artifacts=("overlay-image", "overlay-animation"),
                input_schema=(
                    _field(
                        "config",
                        "Configuration JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _field(
                        "config_file",
                        "Config JSON",
                        "path",
                        cli_flag="--config-file",
                        path=True,
                    ),
                    _field(
                        "overlay_section",
                        "Overlay section",
                        cli_flag="--overlay-section",
                    ),
                    _field(
                        "radio_dir",
                        "Radio data folder",
                        "path",
                        path=True,
                        config_path="paths.radio_base_dir",
                    ),
                    _field(
                        "aia_dir",
                        "AIA data folder",
                        "path",
                        path=True,
                        config_path="paths.aia_base_dir",
                    ),
                    _field(
                        "hmi_dir",
                        "HMI data folder",
                        "path",
                        path=True,
                        config_path="paths.hmi_base_dir",
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="existing-fit-overlay",
                title="Overlay Existing Centers and Fits",
                description=(
                    "Render existing center and/or Gaussian tables over an optional "
                    "AIA FITS background without rerunning source localization."
                ),
                command_module="solar_apps.workflows.radio.existing_fit_overlay_cli",
                output_flag="--output-dir",
                accepts_artifacts=("center-table", "gaussian-table", "aia-fits"),
                produces_artifacts=("overlay-metadata", "overlay-image"),
                run_required_any_fields=("center_csv", "gaussian_csv"),
                input_schema=(
                    _field(
                        "center_csv",
                        "Existing center CSV",
                        "path",
                        cli_flag="--center-csv",
                        path=True,
                        artifact_types=("center-table",),
                        help_text="Threshold or contour center-table product.",
                    ),
                    _field(
                        "gaussian_csv",
                        "Existing Gaussian CSV",
                        "path",
                        cli_flag="--gaussian-csv",
                        path=True,
                        artifact_types=("gaussian-table",),
                        help_text="Persisted Gaussian diagnostics product.",
                    ),
                    _field(
                        "aia_fits",
                        "AIA background FITS",
                        "path",
                        cli_flag="--aia-fits",
                        path=True,
                        artifact_types=("aia-fits",),
                    ),
                    _field(
                        "width",
                        "Image width (px)",
                        "number",
                        cli_flag="--width",
                        default=1200,
                    ),
                    _field(
                        "height",
                        "Image height (px)",
                        "number",
                        cli_flag="--height",
                        default=900,
                    ),
                    _field(
                        "marker_size",
                        "Marker size",
                        "number",
                        cli_flag="--marker-size",
                        default=10,
                    ),
                    _field(
                        "theme",
                        "Theme",
                        "select",
                        cli_flag="--theme",
                        default="light",
                        choices=("auto", "light", "dark"),
                    ),
                    _field(
                        "markers_only",
                        "Markers only",
                        "checkbox",
                        cli_flag="--markers-only",
                    ),
                ),
            ),
            RadioActionSpec(
                id="dem-radio-overlay",
                title="DEM/Radio Overlay",
                description=(
                    "Compare a DEM brightness-temperature map with one explicit or "
                    "time-matched radio FITS image."
                ),
                command_module="solar_apps.workflows.xray_dem.dem_radio_cli",
                output_flag="--output-dir",
                section="adjacent",
                risk_level="advanced",
                accepts_artifacts=("aia-fits", "radio-fits", "dem-map"),
                produces_artifacts=("overlay-metadata", "overlay-image"),
                run_required_fields=("aia_fits", "tb_data"),
                run_required_any_fields=("radio_file", "radio_dir"),
                input_schema=(
                    _field(
                        "aia_fits",
                        "AIA reference FITS",
                        "path",
                        cli_flag="--aia-fits",
                        required=True,
                        path=True,
                        artifact_types=("aia-fits",),
                    ),
                    _field(
                        "tb_data",
                        "DEM brightness-temperature NPY",
                        "path",
                        cli_flag="--tb-data",
                        required=True,
                        path=True,
                        artifact_types=("dem-map",),
                    ),
                    _field(
                        "radio_file",
                        "Radio FITS",
                        "path",
                        cli_flag="--radio-file",
                        path=True,
                        artifact_types=("radio-fits",),
                        help_text="Use one explicit radio image.",
                    ),
                    _field(
                        "radio_dir",
                        "Radio FITS folder",
                        "path",
                        cli_flag="--radio-dir",
                        path=True,
                        help_text="Or select the closest image from this folder.",
                    ),
                    _field(
                        "radio_pattern",
                        "Radio filename pattern",
                        cli_flag="--radio-pattern",
                        default="*.fits",
                    ),
                    _field(
                        "time_match_level",
                        "Time match level",
                        "select",
                        cli_flag="--time-match-level",
                        default="minute",
                        choices=("minute", "hour", "any"),
                    ),
                    _field(
                        "display_mode",
                        "Display extent",
                        "select",
                        cli_flag="--display-mode",
                        default="custom",
                        choices=("full", "solar_disk", "custom"),
                    ),
                    _field(
                        "tb_pixel_size",
                        "Tb pixel size (arcsec)",
                        "number",
                        cli_flag="--tb-pixel-size",
                        default=3.0,
                    ),
                    _field(
                        "tb_xmin",
                        "Tb minimum X (arcsec)",
                        "number",
                        cli_flag="--tb-xmin",
                        default=-1150.0,
                    ),
                    _field(
                        "tb_xmax",
                        "Tb maximum X (arcsec)",
                        "number",
                        cli_flag="--tb-xmax",
                        default=1150.0,
                    ),
                    _field(
                        "tb_ymin",
                        "Tb minimum Y (arcsec)",
                        "number",
                        cli_flag="--tb-ymin",
                        default=-1150.0,
                    ),
                    _field(
                        "tb_ymax",
                        "Tb maximum Y (arcsec)",
                        "number",
                        cli_flag="--tb-ymax",
                        default=1150.0,
                    ),
                    _field(
                        "display_xmin",
                        "Display minimum X (arcsec)",
                        "number",
                        cli_flag="--display-xmin",
                        default=-1600.0,
                    ),
                    _field(
                        "display_xmax",
                        "Display maximum X (arcsec)",
                        "number",
                        cli_flag="--display-xmax",
                        default=1600.0,
                    ),
                    _field(
                        "display_ymin",
                        "Display minimum Y (arcsec)",
                        "number",
                        cli_flag="--display-ymin",
                        default=-1600.0,
                    ),
                    _field(
                        "display_ymax",
                        "Display maximum Y (arcsec)",
                        "number",
                        cli_flag="--display-ymax",
                        default=1600.0,
                    ),
                    _field(
                        "radio_smooth_sigma",
                        "Radio smoothing sigma",
                        "number",
                        cli_flag="--radio-smooth-sigma",
                        default=1.5,
                    ),
                    _field(
                        "percentile_low",
                        "Tb lower percentile",
                        "number",
                        cli_flag="--percentile-low",
                        default=1.0,
                    ),
                    _field(
                        "percentile_high",
                        "Tb upper percentile",
                        "number",
                        cli_flag="--percentile-high",
                        default=99.0,
                    ),
                    _field(
                        "dpi",
                        "Output DPI",
                        "number",
                        cli_flag="--dpi",
                        default=300,
                    ),
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="trajectory-media",
        title="Trajectory & Media",
        group="Context",
        description=(
            "Filter center tables, compare L/R motion, render AIA-backed trajectories, "
            "and prepare browser-native media."
        ),
        accepts_artifacts=("center-table",),
        produces_artifacts=("trajectory-html", "trajectory-video"),
        actions=(
            RadioActionSpec(
                id="trajectory-export",
                title="Export Trajectory",
                description="Export a selected trajectory frame as standalone Plotly HTML.",
                command_module="solar_apps.workflows.radio.trajectory_cli",
                output_flag="--out",
                output_filename="radio_trajectory.html",
                accepts_artifacts=("center-table",),
                produces_artifacts=("trajectory-html",),
                input_schema=(
                    _field(
                        "centers",
                        "Center table",
                        "path",
                        cli_flag="--centers",
                        required=True,
                        path=True,
                        artifact_types=("center-table",),
                    ),
                    _field(
                        "aia_dir", "AIA folder", "path", cli_flag="--aia-dir", path=True
                    ),
                    _field("freqs", "Frequencies (MHz)", cli_flag="--freqs"),
                    _field(
                        "polarizations", "Polarizations", cli_flag="--polarizations"
                    ),
                    _field(
                        "compare_lr", "Compare L/R", "checkbox", cli_flag="--compare-lr"
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
            RadioActionSpec(
                id="trajectory-media",
                title="Interactive Trajectory & Media",
                description=(
                    "Use the same-page trajectory player and local media encoder, "
                    "or export reproducible MP4, GIF, or WebM media."
                ),
                command_module="solar_apps.workflows.radio.trajectory_media_cli",
                output_flag="--output-dir",
                preview_adapter="trajectory-media",
                accepts_artifacts=("center-table",),
                produces_artifacts=("trajectory-video",),
                input_schema=(
                    _field(
                        "centers",
                        "Center table",
                        "path",
                        cli_flag="--centers",
                        required=True,
                        path=True,
                        artifact_types=("center-table",),
                    ),
                    _field(
                        "aia_dir",
                        "AIA background folder",
                        "path",
                        cli_flag="--aia-dir",
                        path=True,
                    ),
                    _field(
                        "use_aia",
                        "Use nearest AIA background",
                        "checkbox",
                        cli_flag="--use-aia",
                    ),
                    _field(
                        "aia_pattern",
                        "AIA FITS pattern",
                        cli_flag="--aia-pattern",
                        default="*.fits",
                    ),
                    _field(
                        "max_aia_dt_sec",
                        "Maximum AIA time gap (s)",
                        "number",
                        cli_flag="--max-aia-dt-sec",
                        default=3600,
                    ),
                    _field(
                        "aia_max_pixels",
                        "AIA preview max side (px)",
                        "number",
                        cli_flag="--aia-max-pixels",
                        default=384,
                    ),
                    _field("freqs", "Frequencies (MHz)", cli_flag="--freqs"),
                    _field(
                        "polarizations",
                        "Polarizations",
                        cli_flag="--polarizations",
                    ),
                    _field(
                        "center_methods",
                        "Center methods",
                        cli_flag="--center-methods",
                    ),
                    _field(
                        "frame_mode",
                        "Frame mode",
                        "select",
                        cli_flag="--frame-mode",
                        default="tail",
                        choices=("current", "tail", "all"),
                    ),
                    _field(
                        "tail_n",
                        "Tail frames",
                        "number",
                        cli_flag="--tail-n",
                        default=5,
                    ),
                    _field(
                        "plot_layout",
                        "Plot layout",
                        "select",
                        cli_flag="--plot-layout",
                        default="overlay",
                        choices=("overlay", "facets"),
                    ),
                    _field(
                        "facet_by",
                        "Facet by",
                        "select",
                        cli_flag="--facet-by",
                        default="freq_mhz",
                        choices=("freq_mhz", "polarization", "center_method"),
                    ),
                    _field(
                        "format",
                        "Export format",
                        "select",
                        cli_flag="--format",
                        default="mp4",
                        choices=("mp4", "gif", "webm"),
                    ),
                    _field(
                        "fps",
                        "Frames per second",
                        "number",
                        cli_flag="--fps",
                        default=6,
                    ),
                    _field(
                        "theme",
                        "Theme",
                        "select",
                        cli_flag="--theme",
                        default="auto",
                        choices=("auto", "light", "dark"),
                    ),
                    _field(
                        "marker_size",
                        "Marker size",
                        "number",
                        cli_flag="--marker-size",
                        default=9,
                    ),
                    _ADVANCED_ARGUMENTS,
                ),
            ),
        ),
    ),
    RadioModuleSpec(
        id="runs-results",
        title="Runs & Results",
        group="Core",
        description=(
            "Inspect the queue, logs, provenance, reusable artifacts, previews, and downloads."
        ),
        default_enabled=True,
        default_collapsed=True,
        always_available=True,
        accepts_artifacts=("*",),
        actions=(
            RadioActionSpec(
                id="inspect-runs",
                title="Inspect Runs",
                description="Review persisted run state and incremental logs.",
                preview_adapter="run-index",
            ),
            RadioActionSpec(
                id="inspect-artifacts",
                title="Inspect Artifacts",
                description="Preview or download safe artifacts from previous runs.",
                preview_adapter="artifact-index",
            ),
        ),
    ),
)


MODULES_BY_ID = {item.id: item for item in MODULES}


PRESETS: dict[str, dict[str, Any]] = {
    "source-localization": {
        "id": "source-localization",
        "title": "Source Localization",
        "module_ids": [
            "data-configuration",
            "imaging-localization",
            "trajectory-media",
            "runs-results",
        ],
    },
    "roi-study": {
        "id": "roi-study",
        "title": "ROI Study",
        "module_ids": [
            "data-configuration",
            "roi-light-curves",
            "runs-results",
        ],
    },
    "burst-physics": {
        "id": "burst-physics",
        "title": "Burst Physics",
        "module_ids": [
            "imaging-localization",
            "spectrogram-drift",
            "physical-diagnostics",
            "runs-results",
        ],
    },
    "multi-instrument-context": {
        "id": "multi-instrument-context",
        "title": "Multi-Instrument Context",
        "module_ids": [
            "imaging-localization",
            "context-overlays",
            "runs-results",
        ],
    },
    "full-analysis": {
        "id": "full-analysis",
        "title": "Full Analysis",
        "module_ids": [item.id for item in MODULES],
    },
}


EVENT_PRESETS: dict[str, dict[str, Any]] = {}


def get_module(module_id: str) -> RadioModuleSpec:
    try:
        return MODULES_BY_ID[module_id]
    except KeyError as exc:
        raise KeyError(f"Unknown radio module: {module_id}") from exc


def get_action(module_id: str, action_id: str) -> RadioActionSpec:
    return get_module(module_id).get_action(action_id)


def catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "modules": [item.to_dict() for item in MODULES],
        "groups": ["Core", "Analysis", "Context", "Advanced"],
    }


def presets_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "presets": [dict(item) for item in PRESETS.values()],
        "event_presets": [
            {"id": item["id"], "title": item["title"]}
            for item in EVENT_PRESETS.values()
        ],
    }


__all__ = [
    "EVENT_PRESETS",
    "MODULES",
    "MODULES_BY_ID",
    "PRESETS",
    "catalog_payload",
    "get_action",
    "get_module",
    "presets_payload",
]
