"""Command-line entry point for the spike-topping Type III visual simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from PIL import Image

from .athena_io import read_bridge_hdf5, write_bridge_hdf5
from .config import MHDConfig, RunConfig, TimeCalibrationConfig, profile_config
from .events import EventBundle
from .physics.fields import MHDFieldSeries
from .physics.jet import JetResult, diagnose_jet, reconnection_flux_rate
from .physics.radio import RadioResult, synthesize_radio_proxy
from .physics.rmhd import MHDResult, solve_rmhd
from .storage import read_rmhd_hdf5, write_rmhd_hdf5
from .visualization.animations import (
    AnimationFormat,
    normalize_animation_formats,
    require_mp4_backend,
    save_animations,
    validate_animation_formats,
)
from .visualization.figures import save_static_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a normalized 2-D reduced-MHD current-sheet simulation and "
            "generate Type III radio-proxy figures and animations."
        )
    )
    parser.add_argument(
        "--profile",
        choices=(
            "quick",
            "standard",
            "cuda-coarse",
            "cuda-medium",
            "cuda-fine",
            "cuda-medium-event",
            "cuda-fine-event",
            "cuda-fine-control",
            "rmhd-coarse",
            "rmhd-medium",
            "rmhd-fine",
            "rmhd-medium-event",
            "rmhd-fine-event",
            "rmhd-fine-control",
        ),
        default="standard",
        help="Numerical resolution and run-length profile.",
    )
    parser.add_argument(
        "--rmhd-engine",
        choices=("numpy", "torch"),
        default="numpy",
        help="Reduced-MHD numerical array/FFT engine.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device for the torch RMHD engine.",
    )
    parser.add_argument(
        "--precision",
        choices=("float64", "float32"),
        default="float64",
        help="RMHD arithmetic precision; formal results require float64.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Atomically record restart state every N steps (0 disables it).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the output directory checkpoint after hash validation.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "simulate", "render"),
        default="all",
        help="Run the full workflow, scientific simulation only, or re-render HDF5.",
    )
    parser.add_argument(
        "--render-profile",
        choices=(
            "legacy",
            "preview",
            "presentation-4k",
            "scientific-preview",
            "scientific-4k",
        ),
        default="legacy",
        help="Static-figure and video resolution/quality profile.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Atomically update simulation/README.md after strict validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help="Seed for deterministic phenomenological spike components.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: package outputs/).",
    )
    parser.add_argument(
        "--animation-format",
        choices=("none", "gif", "mp4", "both"),
        default=None,
        help=(
            "Animation export after the simulation: none, gif, mp4, or both "
            "(default: gif)."
        ),
    )
    parser.add_argument(
        "--lorentz-convention",
        choices=("physical", "legacy"),
        default="physical",
        help=(
            "Lorentz bracket convention. 'physical' uses [j, psi]; "
            "'legacy' is retained only for diagnostic comparison."
        ),
    )
    parser.add_argument(
        "--spike-coupling",
        choices=("jet", "uniform"),
        default="jet",
        help="Choose jet-conditioned or uniform onset-window spike times.",
    )
    parser.add_argument(
        "--mhd-backend",
        choices=("rmhd", "athena", "amrvac", "athenak"),
        default="rmhd",
        help=(
            "MHD source. The compatibility default is rmhd; formal Athena "
            "or AMRVAC results require an explicit schema-v3/v4/v5 bridge."
        ),
    )
    parser.add_argument(
        "--mhd-dataset",
        type=Path,
        help="Common schema-v3/v4/v5 bridge HDF5 for Athena or AMRVAC.",
    )
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        help=(
            "Validated zero-perturbation control run required by scientific "
            "event/control animation profiles."
        ),
    )
    parser.add_argument(
        "--athena-dataset",
        type=Path,
        help="Compatibility alias for --mhd-dataset with the Athena backend.",
    )
    parser.add_argument(
        "--time-calibration",
        choices=("proxy", "alfven", "event"),
        default="proxy",
        help="Proxy, explicit Alfvén-time, or reviewed event-window calibration.",
    )
    parser.add_argument(
        "--event-bundle",
        type=Path,
        help="Sanitized EventBundle JSON required by event time calibration.",
    )
    parser.add_argument("--length-scale-mm", type=float)
    parser.add_argument("--magnetic-field-gauss", type=float)
    parser.add_argument("--electron-density-cm3", type=float)
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --animation-format none. "
            "Cannot be combined with --animation-format."
        ),
    )
    return parser


MHDData = MHDResult | MHDFieldSeries


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _total_energy(mhd: MHDData) -> np.ndarray:
    if isinstance(mhd, MHDFieldSeries):
        return mhd.total_energy
    return mhd.magnetic_energy + mhd.kinetic_energy


def _resolve_animation_formats(
    animation_format: str | None,
    skip_animations: bool,
) -> tuple[AnimationFormat, ...]:
    if skip_animations and animation_format is not None:
        raise ValueError(
            "--skip-animations cannot be combined with --animation-format."
        )
    selection = "none" if skip_animations else (animation_format or "gif")
    return normalize_animation_formats(selection)


def _write_data(
    output_dir: Path,
    config: RunConfig,
    mhd: MHDData,
    jet: JetResult,
    radio: RadioResult,
    elapsed_s: float,
    animation_formats: tuple[AnimationFormat, ...],
    event_bundle: EventBundle | None,
    *,
    stage: str,
    render_profile: str,
    control_run_id: str | None = None,
) -> list[Path]:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = data_dir / "mhd_snapshots.npz"
    arrays: dict[str, object] = {
        "x": mhd.grid.x,
        "y": mhd.grid.y,
        "time": mhd.times,
        "radio_time_s": radio.times_s,
        "radio_frequency_mhz": radio.frequencies_mhz,
        "radio_intensity": radio.intensity.astype(np.float32),
        "radio_ridge_frequency_mhz": radio.ridge_frequency_mhz,
        "radio_injection_activity": radio.injection_activity,
        "radio_jet_activity": radio.jet_activity,
        "radio_conditioned_reconnection_activity": (
            radio.conditioned_reconnection_activity
        ),
        "spike_catalog": radio.spike_catalog,
        "jet_positive_speed": jet.positive_speed,
        "jet_negative_speed": jet.negative_speed,
        "jet_bidirectional_speed": jet.bidirectional_speed,
        "jet_activity_mhd": jet.jet_activity,
        "global_jet_speed_mhd": jet.global_jet_speed,
        "global_jet_activity_mhd": jet.global_jet_activity,
        "reconnection_activity_mhd": jet.reconnection_activity,
        "reconnection_flux_rate_mhd": reconnection_flux_rate(mhd),
        "mhd_source": np.asarray(
            getattr(mhd, "source", "rmhd"),
            dtype="U16",
        ),
    }
    if isinstance(mhd, MHDFieldSeries):
        arrays.update(
            internal_energy=mhd.internal_energy,
            total_energy=mhd.total_energy,
            flux_difference=mhd.flux_difference,
            xpoint_electric_field=mhd.xpoint_electric_field,
            divergence_normalized_rms=mhd.divergence_normalized_rms,
        )
    elif config.profile in {"quick", "standard"}:
        arrays.update(
            psi=mhd.psi.astype(np.float32),
            omega=mhd.omega.astype(np.float32),
        )
    data_paths: list[Path] = []
    if config.profile in {"quick", "standard"}:
        np.savez_compressed(snapshot_path, **arrays)
        data_paths.append(snapshot_path)
    else:
        radio_path = data_dir / "radio_proxy.npz"
        radio_arrays = {
            key: value
            for key, value in arrays.items()
            if key.startswith(("radio_", "spike_", "jet_", "global_", "reconnection_"))
        }
        np.savez_compressed(radio_path, **radio_arrays)
        data_paths.append(radio_path)
    if isinstance(mhd, MHDFieldSeries):
        bridge_path = data_dir / "mhd_bridge.h5"
        if stage != "render" or not bridge_path.is_file():
            write_bridge_hdf5(mhd, bridge_path)
        data_paths.append(bridge_path)
    else:
        hdf5_path = data_dir / "rmhd_fields.h5"
        if stage != "render" or not hdf5_path.is_file():
            write_rmhd_hdf5(
                mhd,
                config.mhd,
                hdf5_path,
                metadata={
                    "profile": config.profile,
                    "seed": config.seed,
                    "stage": stage,
                    "render_profile": render_profile,
                    "rng_scheme": "seed-sequence-v1",
                },
            )
        data_paths.append(hdf5_path)

    diagnostics_path = data_dir / "diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        headings = [
            "time_normalized",
            "magnetic_energy",
            "kinetic_energy",
            "total_energy",
            "max_abs_current",
            "max_speed",
            "reconnection_rate_or_proxy",
            "reconnection_flux_rate",
        ]
        columns: list[np.ndarray] = [
            mhd.times,
            mhd.magnetic_energy,
            mhd.kinetic_energy,
            _total_energy(mhd),
            mhd.max_current,
            mhd.max_speed,
            mhd.reconnection_proxy,
            reconnection_flux_rate(mhd),
        ]
        if isinstance(mhd, MHDResult):
            headings.extend(
                [
                    "flux_difference",
                    "xpoint_electric_field",
                    "island_width_proxy",
                    "ohmic_dissipation",
                    "viscous_dissipation",
                    "energy_budget_residual",
                ]
            )
            columns.extend(
                [
                    mhd.flux_difference,
                    mhd.xpoint_electric_field,
                    mhd.island_width_proxy,
                    mhd.ohmic_dissipation,
                    mhd.viscous_dissipation,
                    mhd.energy_budget_residual,
                ]
            )
        writer.writerow(headings)
        for row in zip(*columns, strict=True):
            writer.writerow([f"{float(value):.12g}" for value in row])

    total_energy = _total_energy(mhd)
    mhd_source = getattr(mhd, "source", "rmhd")
    full_mhd = isinstance(mhd, MHDFieldSeries)
    metadata = {
        "schema_version": 5 if full_mhd else 6,
        "mhd_backend": (
            "amrvac"
            if full_mhd and mhd_source == "mpi-amrvac"
            else ("athena" if full_mhd else "rmhd")
        ),
        "mhd_source": mhd_source,
        "model_boundary": {
            "self_consistent": (
                "2.5-D compressible resistive-viscous full MHD"
                if full_mhd
                else "2-D incompressible reduced resistive MHD"
            ),
            "proxy": "kinematic electron beam and phenomenological radio intensity",
            "not_included": [
                "PIC or Boris test particles",
                "kinetic radio emission",
                "radio-wave propagation",
            ],
        },
        "config": config.to_dict(),
        "runtime": {
            "elapsed_s": float(elapsed_s),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "imageio": _package_version("imageio"),
            "imageio_ffmpeg": _package_version("imageio-ffmpeg"),
            "pillow": Image.__version__,
            "execution_backend": getattr(mhd, "execution_backend", mhd_source),
            "execution_device": getattr(mhd, "execution_device", "cpu"),
            "execution_precision": getattr(mhd, "execution_precision", "float64"),
            "peak_device_memory_bytes": int(
                getattr(mhd, "peak_device_memory_bytes", 0)
            ),
        },
        "exports": {
            "animation_formats": list(animation_formats),
            "stage": stage,
            "render_profile": render_profile,
            "control_run_id": control_run_id,
        },
        "event_constraint": (
            None
            if event_bundle is None
            else {
                "event_id": event_bundle.event_id,
                "bundle_sha256": event_bundle.to_dict()["bundle_sha256"],
                "core_start_utc": event_bundle.core_start_utc,
                "core_end_utc": event_bundle.core_end_utc,
                "confirmed_drift_ids": [
                    drift.drift_id
                    for drift in event_bundle.drifts
                    if drift.status == "confirmed"
                ],
                "candidate_drift_ids": [
                    drift.drift_id
                    for drift in event_bundle.drifts
                    if drift.status == "candidate"
                ],
            }
        ),
        "diagnostics": {
            "snapshot_count": len(mhd.times),
            "divergence_normalized_rms": mhd.divergence_rms,
            "total_energy_drift_fraction": float(
                (total_energy[-1] - total_energy[0]) / total_energy[0]
            ),
            "final_max_speed": float(mhd.max_speed[-1]),
            "reconnection_definition": (
                "abs(d/dt(psi_O-psi_X)); Ez(X) cross-check"
            ),
            "base_density_cm3": radio.base_density_cm3,
            "beam_gamma": radio.beam_gamma,
            "jet_onset_time_normalized": jet.onset_time_normalized,
            "event_status": radio.event_status,
            "event_count": len(radio.spike_catalog),
            "jet_coincidence_fraction": radio.jet_coincidence_fraction,
            "minimum_topping_margin_mhz": (
                None
                if radio.topping_margin_mhz.size == 0
                else float(np.min(radio.topping_margin_mhz))
            ),
            "energy_budget_max_abs_fraction": (
                None
                if not isinstance(mhd, MHDResult)
                or mhd.energy_budget_residual.size == 0
                else float(
                    np.max(np.abs(mhd.energy_budget_residual))
                    / max(float(total_energy[0]), 1.0e-15)
                )
            ),
            "flux_difference_final": (
                None
                if not isinstance(mhd, MHDResult)
                or mhd.flux_difference.size == 0
                else float(mhd.flux_difference[-1])
            ),
            "xpoint_electric_field_final": (
                None
                if not isinstance(mhd, MHDResult)
                or mhd.xpoint_electric_field.size == 0
                else float(mhd.xpoint_electric_field[-1])
            ),
            "rng_scheme": "seed-sequence-v1",
        },
        "activity": {
            "time_mapping": (
                "[tau_J, tau_end] compressed into onset window"
                if config.time_calibration.mode == "proxy"
                else (
                    "reviewed event window"
                    if config.time_calibration.mode == "event"
                    else "t_seconds = tau * L0 / v_A"
                )
            ),
            "mapping_role": "onset conditioning only",
            "jet_onset_radio_s": radio.jet_onset_radio_s,
            "jet_activity": radio.jet_activity.tolist(),
            "conditioned_reconnection_activity": (
                radio.conditioned_reconnection_activity.tolist()
            ),
            "jet_spike_lag_s": radio.jet_spike_lag_s.tolist(),
            "topping_margin_mhz": radio.topping_margin_mhz.tolist(),
        },
    }
    metadata_path = data_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return [*data_paths, diagnostics_path, metadata_path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, paths: list[Path]) -> Path:
    manifest_path = output_dir / "SHA256SUMS.txt"
    candidates = {
        path.resolve()
        for path in paths
        if path.is_file()
        and path.resolve() != manifest_path.resolve()
        and path.name != "validation_report.json"
    }
    candidates.update(
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.resolve() != manifest_path.resolve()
        and path.name != "validation_report.json"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in sorted(candidates)
    ]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return manifest_path


def run(
    config: RunConfig,
    output_dir: Path,
    animation_formats: tuple[AnimationFormat, ...] = ("gif",),
    *,
    mhd_result: MHDData | None = None,
    control_result: MHDResult | None = None,
    control_mhd_config: MHDConfig | None = None,
    control_run_id: str | None = None,
    event_bundle: EventBundle | None = None,
    rmhd_engine: str = "numpy",
    device: str = "auto",
    precision: str = "float64",
    stage: str = "all",
    render_profile: str = "legacy",
    update_readme: bool = False,
    checkpoint_every: int = 0,
    resume: bool = False,
) -> list[Path]:
    if stage not in {"all", "simulate", "render"}:
        raise ValueError("stage must be all, simulate, or render.")
    if rmhd_engine not in {"numpy", "torch"}:
        raise ValueError("rmhd_engine must be numpy or torch.")
    if precision not in {"float64", "float32"}:
        raise ValueError("precision must be float64 or float32.")
    if update_readme and stage != "all":
        raise ValueError("--update-readme requires --stage all.")
    if update_readme and config.profile in {"quick", "standard"}:
        raise ValueError("--update-readme requires a CUDA convergence profile.")
    validate_animation_formats(animation_formats)
    effective_animation_formats = () if stage == "simulate" else animation_formats
    if "mp4" in effective_animation_formats:
        require_mp4_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "data" / "rmhd_checkpoint.npz"
    last_reported = -1

    def progress(step: int, total: int) -> None:
        nonlocal last_reported
        percent = int(100 * step / total)
        bucket = percent // 20
        if bucket > last_reported:
            last_reported = bucket
            print(f"MHD progress: {min(bucket * 20, 100)}%")

    started = perf_counter()
    prior_metadata: dict[str, object] | None = None
    prior_metadata_path = output_dir / "data" / "run_metadata.json"
    if stage == "render" and prior_metadata_path.is_file():
        prior_metadata = json.loads(
            prior_metadata_path.read_text(encoding="utf-8")
        )
    if mhd_result is not None:
        mhd = mhd_result
    elif rmhd_engine == "torch":
        from .physics.rmhd_torch import solve_rmhd_torch

        mhd = solve_rmhd_torch(
            config.mhd,
            device=device,
            precision=precision,
            progress=progress,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    else:
        if device not in {"auto", "cpu"}:
            raise ValueError("The NumPy RMHD engine supports only the CPU device.")
        if precision != "float64":
            raise ValueError("The NumPy reference engine is fixed to float64.")
        mhd = solve_rmhd(
            config.mhd,
            progress=progress,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    jet = diagnose_jet(mhd, config.mhd, config.jet)
    radio = synthesize_radio_proxy(
        mhd,
        config.radio,
        config.seed,
        jet_result=jet,
        jet_config=config.jet,
        spike_coupling=config.spike_coupling,
        time_calibration=config.time_calibration,
    )
    figures: list[Path] = []
    animations: list[Path] = []
    if stage != "simulate":
        static_render_profile = (
            "presentation-4k"
            if render_profile == "scientific-4k"
            else ("preview" if render_profile == "scientific-preview" else render_profile)
        )
        figures = save_static_figures(
            mhd,
            jet,
            radio,
            config.mhd,
            config.jet,
            output_dir / "figures",
            render_profile=static_render_profile,
        )
        if render_profile.startswith("scientific-"):
            if not isinstance(mhd, MHDResult):
                raise ValueError("Scientific event/control animations require RMHD.")
            if control_result is None:
                raise ValueError(
                    "Scientific event/control animations require --control-run-dir."
                )
            from .visualization.scientific_animations import (
                save_scientific_animations,
            )

            control_config = control_mhd_config or config.mhd
            control_jet = diagnose_jet(control_result, control_config, config.jet)
            control_radio = synthesize_radio_proxy(
                control_result,
                config.radio,
                config.seed,
                jet_result=control_jet,
                jet_config=config.jet,
                spike_coupling=config.spike_coupling,
                time_calibration=config.time_calibration,
            )
            animations = save_scientific_animations(
                mhd,
                control_result,
                jet,
                control_jet,
                radio,
                control_radio,
                config.mhd,
                config.jet,
                output_dir / "animations",
                effective_animation_formats,
                render_profile=render_profile,
            )
        else:
            animations = save_animations(
                mhd,
                radio,
                output_dir / "animations",
                formats=effective_animation_formats,
                render_profile=render_profile,
            )
    elapsed = perf_counter() - started
    data_paths = _write_data(
        output_dir,
        config,
        mhd,
        jet,
        radio,
        elapsed,
        effective_animation_formats,
        event_bundle,
        stage=stage,
        render_profile=render_profile,
        control_run_id=control_run_id,
    )
    if prior_metadata is not None:
        current_metadata_path = output_dir / "data" / "run_metadata.json"
        current_metadata = json.loads(
            current_metadata_path.read_text(encoding="utf-8")
        )
        prior_runtime = prior_metadata.get("runtime", {})
        if isinstance(prior_runtime, dict):
            current_metadata["runtime"]["simulation_elapsed_s"] = (
                prior_runtime.get(
                    "simulation_elapsed_s",
                    prior_runtime.get("elapsed_s"),
                )
            )
        current_metadata["runtime"]["render_elapsed_s"] = float(elapsed)
        current_metadata_path.write_text(
            json.dumps(current_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    all_paths = figures + animations + data_paths
    manifest_path = _write_manifest(output_dir, all_paths)
    all_paths.append(manifest_path)
    if update_readme:
        from .reporting import update_project_readme

        readme_path = update_project_readme(output_dir)
        all_paths.append(readme_path)

    total_energy = _total_energy(mhd)
    energy_drift = (total_energy[-1] - total_energy[0]) / total_energy[0]
    print(f"Completed {config.profile!r} profile in {elapsed:.2f} s")
    print(f"Snapshots: {len(mhd.times)}")
    print(f"Normalized divergence RMS: {mhd.divergence_rms:.3e}")
    print(f"Total-energy drift: {energy_drift:+.3%}")
    print(f"Final max speed: {mhd.max_speed[-1]:.4f}")
    print(f"Spike events: {len(radio.spike_catalog)} ({radio.event_status})")
    print(f"Output: {output_dir.name}/")
    return all_paths


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        animation_formats = _resolve_animation_formats(
            args.animation_format,
            args.skip_animations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    event_bundle: EventBundle | None = None
    if args.time_calibration == "event":
        if args.event_bundle is None:
            parser.error("--time-calibration event requires --event-bundle.")
        try:
            event_bundle = EventBundle.from_dict(
                json.loads(args.event_bundle.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"Unable to load EventBundle: {exc}")
    elif args.event_bundle is not None:
        parser.error("--event-bundle is valid only with event time calibration.")
    if args.stage != "simulate" and "mp4" in animation_formats:
        try:
            require_mp4_backend()
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        calibration = TimeCalibrationConfig(
            mode=args.time_calibration,
            length_scale_mm=args.length_scale_mm,
            magnetic_field_gauss=args.magnetic_field_gauss,
            electron_density_cm3=args.electron_density_cm3,
        )
    except ValueError as exc:
        parser.error(str(exc))
    config = profile_config(
        args.profile,
        args.seed,
        lorentz_convention=args.lorentz_convention,
        spike_coupling=args.spike_coupling,
        time_calibration=calibration,
    )
    if event_bundle is not None:
        event_time_samples = max(
            2,
            round(event_bundle.duration_s / event_bundle.cadence_s) + 1,
        )
        event_low, event_high = event_bundle.frequency_range_mhz
        config = replace(
            config,
            radio=replace(
                config.radio,
                duration_s=event_bundle.duration_s,
                time_samples=event_time_samples,
                min_frequency_mhz=event_low,
                max_frequency_mhz=event_high,
            ),
        )
    mhd_result: MHDData | None = None
    control_result: MHDResult | None = None
    control_mhd_config = None
    control_run_id: str | None = None
    if args.athena_dataset is not None and args.mhd_dataset is not None:
        parser.error("--athena-dataset conflicts with --mhd-dataset.")
    dataset = args.mhd_dataset or args.athena_dataset
    if args.mhd_backend in {"athena", "amrvac", "athenak"}:
        if dataset is None:
            parser.error(
                f"--mhd-backend {args.mhd_backend} requires --mhd-dataset."
            )
        if args.athena_dataset is not None and args.mhd_backend != "athena":
            parser.error("--athena-dataset is valid only with the Athena backend.")
        try:
            mhd_result = read_bridge_hdf5(dataset.resolve())
        except (OSError, ValueError) as exc:
            parser.error(f"Unable to load MHD dataset: {exc}")
        expected_source = {
            "athena": "athena-c",
            "amrvac": "mpi-amrvac",
            "athenak": "athenak",
        }[args.mhd_backend]
        if mhd_result.source != expected_source:
            parser.error(
                f"Dataset source {mhd_result.source!r} does not match "
                f"--mhd-backend {args.mhd_backend}."
            )
        config = replace(
            config,
            mhd=replace(
                config.mhd,
                nx=len(mhd_result.grid.x),
                ny=len(mhd_result.grid.y),
                lx=mhd_result.geometry.lx,
                ly=mhd_result.geometry.ly,
                sheet_half_width=mhd_result.geometry.sheet_half_width,
                resistivity=mhd_result.resistivity,
                viscosity=mhd_result.viscosity,
            ),
        )
    elif dataset is not None:
        if args.stage != "render":
            parser.error(
                "An RMHD --mhd-dataset is accepted only with --stage render."
            )
        try:
            (
                mhd_result,
                stored_mhd_config,
                stored_metadata,
            ) = read_rmhd_hdf5(dataset.resolve(), lazy=True)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(f"Unable to load RMHD HDF5 dataset: {exc}")
        stored_profile = str(stored_metadata.get("profile", args.profile))
        stored_seed = int(stored_metadata.get("seed", args.seed))
        try:
            config = profile_config(
                stored_profile,
                stored_seed,
                lorentz_convention=stored_mhd_config.lorentz_convention,
                spike_coupling=args.spike_coupling,
                time_calibration=calibration,
            )
        except ValueError as exc:
            parser.error(f"Invalid RMHD HDF5 metadata: {exc}")
        config = replace(config, mhd=stored_mhd_config)
    elif args.stage == "render":
        parser.error("--stage render requires --mhd-dataset.")
    if args.render_profile.startswith("scientific-") and args.stage != "simulate":
        if args.control_run_dir is None:
            parser.error(
                "Scientific render profiles require --control-run-dir."
            )
        control_dir = args.control_run_dir.resolve()
        control_path = control_dir / "data" / "rmhd_fields.h5"
        try:
            control_result, control_mhd_config, _ = read_rmhd_hdf5(
                control_path, lazy=True
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(f"Unable to load control RMHD HDF5 dataset: {exc}")
        control_run_id = control_dir.name
    elif args.control_run_dir is not None:
        parser.error("--control-run-dir requires a scientific render profile.")
    default_output = Path(__file__).resolve().parent / "outputs"
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.profile.startswith(("cuda-", "rmhd-")):
        output_dir = (
            default_output / "runs" / f"{args.profile}_seed{args.seed}"
        ).resolve()
    else:
        output_dir = default_output.resolve()
    run(
        config,
        output_dir,
        animation_formats,
        mhd_result=mhd_result,
        control_result=control_result,
        control_mhd_config=control_mhd_config,
        control_run_id=control_run_id,
        event_bundle=event_bundle,
        rmhd_engine=args.rmhd_engine,
        device=args.device,
        precision=args.precision,
        stage=args.stage,
        render_profile=args.render_profile,
        update_readme=args.update_readme,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
