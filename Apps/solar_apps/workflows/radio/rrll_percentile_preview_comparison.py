"""Render RR+LL percentile-comparison source-map previews for 2025-01-24.

This is a focused batch runner for the 2025-01-24 nine-band RR+LL source maps:
it computes one fixed per-band log10 color range per percentile pair over the
full event window, then renders only the first, middle, and last slots.
"""

from __future__ import annotations

import argparse
import copy
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from . import source_map_workflow as workflow
from solar_toolkit.radio.config import load_radio_user_config
from solar_toolkit.radio.io import write_json_file
from solar_toolkit.radio.provenance import write_radio_provenance

__all__ = [
    "build_parser",
    "compute_fixed_band_ranges",
    "main",
    "resolve_available_run_tag",
    "run_percentile_preview_comparison",
]

CONFIG_NAME = (
    "solar_apps.workflows.radio.configs."
    "radio_20250124_center_pm2min_9band_raw_rrll_full_config"
)
DEFAULT_OUTPUT_DIR = Path("outputs/radio/2025-01-24")
DEFAULT_RUN_STEM = "rrll_spec_percentile_compare_20260712"
PERCENTILE_GROUPS: tuple[tuple[float, float], ...] = (
    (99.0, 99.99),
    (95.0, 99.99),
    (90.0, 99.99),
    (50.0, 99.99),
)
PREVIEW_SLOTS: tuple[tuple[str, int], ...] = (
    ("first", 0),
    ("middle", 294),
    ("last", 587),
)
SPECTROGRAM_FILE = "data/radio/2025-01-24/spectrogram.fits"
SPECTROGRAM_START = "2025-01-24T04:47:43"
SPECTROGRAM_END = "2025-01-24T04:50:35"


def _percentile_token(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _analysis_subdir(
    run_tag: str, percentiles: tuple[float, float], position: str
) -> str:
    low, high = percentiles
    return (
        "radio_source_maps_9band_"
        f"{run_tag}_p{_percentile_token(low)}_{_percentile_token(high)}"
        f"_preview_{position}"
    )


def _candidate_subdirs(run_tag: str) -> list[str]:
    return [
        _analysis_subdir(run_tag, percentiles, position)
        for percentiles in PERCENTILE_GROUPS
        for position, _slot_idx in PREVIEW_SLOTS
    ]


def resolve_available_run_tag(
    output_dir: Path, run_stem: str = DEFAULT_RUN_STEM
) -> str:
    """Return the first run tag whose 12 output directories do not exist."""

    for run_number in range(1, 100):
        run_tag = f"{run_stem}_r{run_number:02d}"
        if not any(
            (output_dir / subdir).exists() for subdir in _candidate_subdirs(run_tag)
        ):
            return run_tag
    raise RuntimeError(f"No available run tag found for {run_stem!r}")


def _base_user_config(
    *,
    radio_root: str | Path | None = None,
    spectrogram_file: str | Path | None = None,
) -> tuple[dict, dict]:
    user_config, newkirk_config = load_radio_user_config(CONFIG_NAME)
    user_config = copy.deepcopy(user_config)
    if radio_root is not None:
        user_config.setdefault("data", {})["multi_band_root"] = str(radio_root)
    user_config.setdefault("features", {}).update(
        {
            "gaussian_overlay": False,
            "spectrogram_panel": True,
            "save_gaussian_diagnostics": False,
            "save_background_products": False,
            "save_individual_pols": False,
        }
    )
    user_config.setdefault("display", {}).update(
        {
            "color_range_mode": "fixed_per_band",
            "use_per_band_colormap": True,
            "per_band_range_method": "fixed_percentile",
        }
    )
    user_config.setdefault("spectrogram", {}).update(
        {
            "file_path": str(spectrogram_file or SPECTROGRAM_FILE),
            "time_display_mode": "user",
            "time_start": SPECTROGRAM_START,
            "time_end": SPECTROGRAM_END,
            "f_start": 80.0,
            "f_end": 340.0,
            "polarization": "sum",
            "vmin": 2.5,
            "vmax": 4.5,
            "use_log10": True,
            "cmap": "jet",
            "colorbar_label": r"log$_{10}$ intensity",
        }
    )
    user_config.setdefault("drift_rate", {}).update({"enabled": False, "mode": "off"})
    user_config.setdefault("output", {}).update(
        {
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "show_plot": False,
            "save_plot": True,
            "dpi": 180,
        }
    )
    return user_config, newkirk_config


def _flat_config(user_config: dict) -> dict:
    return workflow._migrate_config(
        workflow.build_config(user_config, workflow.DEFAULT_CONFIG)
    )


def _collect_band_log_values(cfg: dict, freq: float) -> np.ndarray:
    root = cfg["multi_band_root"]
    pattern = cfg["band_dir_pattern"]
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx")
    rr_dir = os.path.join(root, pattern.format(freq=freq, polar=cfg["rr_dir_suffix"]))
    ll_dir = os.path.join(root, pattern.format(freq=freq, polar=cfg["ll_dir_suffix"]))
    study_mode = cfg.get("study_mode")
    rr_files = workflow._sorted_fits_for_band(
        rr_dir, start_idx, end_idx, study_mode=study_mode
    )
    ll_files = workflow._sorted_fits_for_band(
        ll_dir, start_idx, end_idx, study_mode=study_mode
    )
    rr_files = workflow._filter_bad_radio_files(
        rr_files, freq, cfg["rr_dir_suffix"], cfg, drop_bad=True
    )
    ll_files = workflow._filter_bad_radio_files(
        ll_files, freq, cfg["ll_dir_suffix"], cfg, drop_bad=True
    )
    tolerance_ms = float(cfg.get("time_tolerance_seconds", 1.0)) * 1000.0
    matched_pairs = workflow._match_rr_ll_by_time(rr_files, ll_files, tolerance_ms, cfg)
    if not matched_pairs:
        raise RuntimeError(f"No RR/LL matches for {freq} MHz")

    chunks: list[np.ndarray] = []
    for rr_path, ll_path in matched_pairs:
        rr_data, _rr_header = workflow.read_fits(rr_path)
        ll_data, _ll_header = workflow.read_fits(ll_path)
        combined = workflow._combine_polarization_data(rr_data, ll_data, cfg)
        valid = combined[np.isfinite(combined) & (combined > 0)]
        if valid.size:
            chunks.append(np.log10(valid).astype(np.float32, copy=False))
    if not chunks:
        raise RuntimeError(f"No finite positive RR+LL values for {freq} MHz")
    return np.concatenate(chunks)


def compute_fixed_band_ranges(
    cfg: dict,
    percentile_groups: Iterable[tuple[float, float]] = PERCENTILE_GROUPS,
) -> dict[tuple[float, float], dict[str, list[float]]]:
    """Compute fixed per-band log10 ranges for all requested percentile groups."""

    groups = tuple((float(low), float(high)) for low, high in percentile_groups)
    ranges = {
        group: {"fixed_band_vmins": [], "fixed_band_vmaxs": []} for group in groups
    }
    for freq in cfg["multi_band_freqs"]:
        values = _collect_band_log_values(cfg, freq)
        needed = sorted({percentile for group in groups for percentile in group})
        computed = {
            percentile: float(np.percentile(values, percentile))
            for percentile in needed
        }
        print(
            f"{freq} MHz fixed ranges from {values.size} samples: "
            + ", ".join(
                f"{low:g}-{high:g}%=[{computed[low]:.6f}, {computed[high]:.6f}]"
                for low, high in groups
            )
        )
        for low, high in groups:
            ranges[(low, high)]["fixed_band_vmins"].append(computed[low])
            ranges[(low, high)]["fixed_band_vmaxs"].append(computed[high])
    return ranges


def _slot_time(slot: Sequence[object], cfg: dict):
    if not slot:
        return None
    return workflow._radio_item_datetime(slot[0], cfg)


def _provenance_config(
    user_config: dict,
    *,
    analysis_subdir: str,
    percentiles: tuple[float, float],
    fixed_ranges: dict[str, list[float]],
    position: str,
    slot_idx: int,
    slot_time,
) -> dict:
    config = copy.deepcopy(user_config)
    config["output"]["analysis_subdir"] = analysis_subdir
    config["display"]["per_band_percentiles"] = list(percentiles)
    config["display"]["fixed_band_vmins"] = fixed_ranges["fixed_band_vmins"]
    config["display"]["fixed_band_vmaxs"] = fixed_ranges["fixed_band_vmaxs"]
    config["display"]["fixed_band_percentile_vmins"] = fixed_ranges["fixed_band_vmins"]
    config["display"]["fixed_band_percentile_vmaxs"] = fixed_ranges["fixed_band_vmaxs"]
    config["preview_selection"] = {
        "position": position,
        "slot_index": slot_idx,
        "slot_time": (
            slot_time.isoformat(timespec="milliseconds") if slot_time else None
        ),
    }
    return config


def _render_one_preview(
    *,
    output_dir: Path,
    user_config: dict,
    newkirk_config: dict,
    base_cfg: dict,
    slots: list,
    run_tag: str,
    percentiles: tuple[float, float],
    fixed_ranges: dict[str, list[float]],
    position: str,
    slot_idx: int,
) -> Path:
    analysis_subdir = _analysis_subdir(run_tag, percentiles, position)
    cfg = dict(base_cfg)
    cfg.update(
        {
            "analysis_subdir": analysis_subdir,
            "output_dir": str(output_dir),
            "per_band_percentiles": list(percentiles),
            "fixed_band_vmins": list(fixed_ranges["fixed_band_vmins"]),
            "fixed_band_vmaxs": list(fixed_ranges["fixed_band_vmaxs"]),
            "color_range_mode": "fixed_per_band",
            "save_plot": True,
            "show_plot": False,
            "dpi": 180,
        }
    )
    slot = slots[slot_idx]
    slot_time = _slot_time(slot, cfg)
    output_path = Path(
        workflow.plot_multi_band_slot(
            slot_idx,
            slot,
            str(output_dir),
            cfg,
            vmin=None,
            vmax=None,
        )
    )
    analysis_dir = output_dir / analysis_subdir
    provenance_config = _provenance_config(
        user_config,
        analysis_subdir=analysis_subdir,
        percentiles=percentiles,
        fixed_ranges=fixed_ranges,
        position=position,
        slot_idx=slot_idx,
        slot_time=slot_time,
    )
    write_radio_provenance(
        analysis_dir,
        provenance_config,
        newkirk_config=newkirk_config,
        config_source=CONFIG_NAME,
        cli_overrides={"analysis_subdir": analysis_subdir},
    )
    write_json_file(
        analysis_dir / "percentile_preview_metadata.json",
        {
            "run_tag": run_tag,
            "analysis_subdir": analysis_subdir,
            "percentiles": list(percentiles),
            "frequencies_mhz": list(cfg["multi_band_freqs"]),
            "fixed_band_vmins": fixed_ranges["fixed_band_vmins"],
            "fixed_band_vmaxs": fixed_ranges["fixed_band_vmaxs"],
            "preview_position": position,
            "slot_index": slot_idx,
            "slot_time": (
                slot_time.isoformat(timespec="milliseconds") if slot_time else None
            ),
            "output_png": str(output_path),
            "spectrogram_file": str(
                base_cfg.get("spectrogram_file_path", SPECTROGRAM_FILE)
            ),
            "spectrogram_time_start": SPECTROGRAM_START,
            "spectrogram_time_end": SPECTROGRAM_END,
            "first_slot_spectrogram_note": (
                "The first preview time is outside the displayed spectrogram range; "
                "the figure should show the out-of-range note instead of a vertical line."
                if position == "first"
                else None
            ),
        },
    )
    return output_path


def run_percentile_preview_comparison(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_stem: str = DEFAULT_RUN_STEM,
    run_tag: str | None = None,
    radio_root: str | Path | None = None,
    spectrogram_file: str | Path | None = None,
) -> dict:
    """Render the 12 comparison preview PNGs and return a run summary."""

    output_dir = Path(output_dir)
    resolved_tag = run_tag or resolve_available_run_tag(output_dir, run_stem)
    if any(
        (output_dir / subdir).exists() for subdir in _candidate_subdirs(resolved_tag)
    ):
        raise FileExistsError(
            f"At least one target directory already exists for {resolved_tag}"
        )

    user_config, newkirk_config = _base_user_config(
        radio_root=radio_root,
        spectrogram_file=spectrogram_file,
    )
    user_config["output"]["output_dir"] = str(output_dir)
    base_cfg = _flat_config(user_config)
    slots = workflow._build_multi_band_slots(base_cfg)
    if len(slots) <= max(slot_idx for _position, slot_idx in PREVIEW_SLOTS):
        raise RuntimeError(f"Expected at least 588 slots, got {len(slots)}")

    print("Computing full-window fixed per-band color ranges...")
    range_map = compute_fixed_band_ranges(base_cfg, PERCENTILE_GROUPS)

    print("Loading spectrogram cache once for all preview renders...")
    workflow._SPECTROGRAM_CACHE = workflow.build_spectrogram_cache(base_cfg)
    if workflow._SPECTROGRAM_CACHE is None:
        raise RuntimeError("Spectrogram cache could not be built")

    matplotlib.use("Agg")
    outputs = []
    for percentiles in PERCENTILE_GROUPS:
        for position, slot_idx in PREVIEW_SLOTS:
            output_path = _render_one_preview(
                output_dir=output_dir,
                user_config=user_config,
                newkirk_config=newkirk_config,
                base_cfg=base_cfg,
                slots=slots,
                run_tag=resolved_tag,
                percentiles=percentiles,
                fixed_ranges=range_map[percentiles],
                position=position,
                slot_idx=slot_idx,
            )
            outputs.append(
                {
                    "percentiles": list(percentiles),
                    "position": position,
                    "slot_index": slot_idx,
                    "path": str(output_path),
                }
            )
            print(f"Rendered {percentiles} {position}: {output_path}")

    summary = {
        "run_tag": resolved_tag,
        "output_dir": str(output_dir),
        "config_source": CONFIG_NAME,
        "percentile_groups": [list(item) for item in PERCENTILE_GROUPS],
        "preview_slots": [
            {"position": position, "slot_index": slot_idx}
            for position, slot_idx in PREVIEW_SLOTS
        ],
        "frequencies_mhz": list(base_cfg["multi_band_freqs"]),
        "radio_root": str(base_cfg["multi_band_root"]),
        "spectrogram_file": str(base_cfg["spectrogram_file_path"]),
        "fixed_ranges": {
            f"{low:g}-{high:g}": range_map[(low, high)]
            for low, high in PERCENTILE_GROUPS
        },
        "outputs": outputs,
    }
    write_json_file(
        output_dir / f"radio_source_maps_9band_{resolved_tag}_summary.json", summary
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render RR+LL percentile comparison preview source maps."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-stem", default=DEFAULT_RUN_STEM)
    parser.add_argument("--run-tag")
    parser.add_argument(
        "--radio-root",
        help=(
            "Explicit multi-band radio root. Radio Workspace requires this "
            "allowed-root-validated override; the legacy CLI keeps its event default."
        ),
    )
    parser.add_argument(
        "--spectrogram-file",
        help=(
            "Explicit spectrogram FITS file. Radio Workspace requires this "
            "allowed-root-validated override; the legacy CLI keeps its event default."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_percentile_preview_comparison(
        output_dir=Path(args.output_dir),
        run_stem=args.run_stem,
        run_tag=args.run_tag,
        radio_root=args.radio_root,
        spectrogram_file=args.spectrogram_file,
    )
    print(f"Done. Run tag: {summary['run_tag']}")
    print(f"Generated {len(summary['outputs'])} preview PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
