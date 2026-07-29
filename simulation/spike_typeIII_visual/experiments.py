"""Run the documented convergence and control experiment suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import JetConfig, MHDConfig, RadioConfig
from .physics.jet import JetResult, diagnose_jet, find_sustained_onset
from .physics.radio import RadioResult, synthesize_radio_proxy
from .physics.rmhd import MHDResult, solve_rmhd

SEED = 20260726


def _mhd_metrics(result: MHDResult) -> dict[str, float]:
    total = result.magnetic_energy + result.kinetic_energy
    return {
        "final_total_energy": float(total[-1]),
        "energy_drift_fraction": float((total[-1] - total[0]) / total[0]),
        "peak_current": float(np.max(result.max_current)),
        "peak_speed": float(np.max(result.max_speed)),
        "divergence_normalized_rms": float(result.divergence_rms),
    }


def _radio_metrics(radio: RadioResult) -> dict[str, Any]:
    return {
        "event_status": radio.event_status,
        "event_count": len(radio.spike_catalog),
        "jet_coincidence_fraction": radio.jet_coincidence_fraction,
        "minimum_topping_margin_mhz": (
            None
            if radio.topping_margin_mhz.size == 0
            else float(np.min(radio.topping_margin_mhz))
        ),
        "mean_jet_lag_s": (
            None
            if radio.jet_spike_lag_s.size == 0
            else float(np.nanmean(radio.jet_spike_lag_s))
        ),
    }


def _run_mhd_case(config: MHDConfig) -> tuple[MHDResult, float]:
    started = perf_counter()
    result = solve_rmhd(config)
    return result, perf_counter() - started


def _case_record(
    name: str,
    config: MHDConfig,
    result: MHDResult,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "config": {
            "nx": config.nx,
            "ny": config.ny,
            "dt": config.dt,
            "steps": config.steps,
            "snapshot_stride": config.snapshot_stride,
            "lorentz_convention": config.lorentz_convention,
            "perturbation_amplitude": config.perturbation_amplitude,
        },
        "elapsed_s": float(elapsed_s),
        "metrics": _mhd_metrics(result),
    }


def _metric_distance(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    names = ("final_total_energy", "peak_current", "peak_speed")
    differences = [
        abs(first[name] - second[name]) / max(abs(second[name]), 1.0e-12)
        for name in names
    ]
    return float(max(differences))


def _convergence_record(cases: list[dict[str, Any]]) -> dict[str, Any]:
    coarse_to_middle = _metric_distance(
        cases[0]["metrics"],
        cases[1]["metrics"],
    )
    middle_to_fine = _metric_distance(
        cases[1]["metrics"],
        cases[2]["metrics"],
    )
    return {
        "coarse_to_middle_max_relative_difference": coarse_to_middle,
        "middle_to_fine_max_relative_difference": middle_to_fine,
        "converged": bool(middle_to_fine < coarse_to_middle and middle_to_fine < 0.10),
        "criterion": (
            "fine difference < coarse difference and fine core-scalar difference < 10%"
        ),
    }


def _shifted_jet(
    jet: JetResult,
    mhd: MHDResult,
    config: JetConfig,
) -> JetResult:
    shifted = np.roll(jet.jet_activity, max(1, len(jet.jet_activity) // 3))
    onset_index = find_sustained_onset(
        shifted,
        config.jet_threshold,
        config.consecutive_snapshots,
    )
    return replace(
        jet,
        jet_activity=shifted,
        onset_index=onset_index,
        onset_time_normalized=(
            None if onset_index is None else float(mhd.times[onset_index])
        ),
    )


def _radio_case(
    name: str,
    mhd: MHDResult,
    mhd_config: MHDConfig,
    radio_config: RadioConfig,
    jet_config: JetConfig,
    seed: int,
    coupling: str,
    jet_override: JetResult | None = None,
) -> dict[str, Any]:
    jet = jet_override or diagnose_jet(mhd, mhd_config, jet_config)
    radio = synthesize_radio_proxy(
        mhd,
        radio_config,
        seed,
        jet_result=jet,
        jet_config=jet_config,
        spike_coupling=coupling,
    )
    return {
        "name": name,
        "seed": seed,
        "coupling": coupling,
        "jet_threshold": jet_config.jet_threshold,
        "reconnection_threshold": jet_config.reconnection_threshold,
        "jet_onset_time_normalized": jet.onset_time_normalized,
        "metrics": _radio_metrics(radio),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "name",
                "nx",
                "dt",
                "steps",
                "elapsed_s",
                "energy_drift_fraction",
                "peak_current",
                "peak_speed",
                "divergence_normalized_rms",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["name"],
                    record["config"]["nx"],
                    record["config"]["dt"],
                    record["config"]["steps"],
                    f"{record['elapsed_s']:.12g}",
                    f"{record['metrics']['energy_drift_fraction']:.12g}",
                    f"{record['metrics']['peak_current']:.12g}",
                    f"{record['metrics']['peak_speed']:.12g}",
                    (f"{record['metrics']['divergence_normalized_rms']:.12g}"),
                ]
            )


def _plot_convergence(
    spatial: list[dict[str, Any]],
    temporal: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0].plot(
        [case["config"]["nx"] for case in spatial],
        [case["metrics"]["peak_speed"] for case in spatial],
        marker="o",
        color="#0B2545",
        linewidth=2.4,
    )
    axes[0].set(
        title="Spatial refinement at fixed normalized end time",
        xlabel="Grid size N",
        ylabel="Peak speed",
    )
    axes[1].plot(
        [case["config"]["dt"] for case in temporal],
        [case["metrics"]["peak_speed"] for case in temporal],
        marker="o",
        color="#2A7F8E",
        linewidth=2.4,
    )
    axes[1].invert_xaxis()
    axes[1].set(
        title="Temporal refinement at fixed 96×96 grid",
        xlabel="Time step",
        ylabel="Peak speed",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        "Convergence is assessed from independent spatial and temporal sequences",
        color="#0B2545",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def _plot_controls(records: list[dict[str, Any]], path: Path) -> None:
    labels = [record["name"] for record in records]
    counts = [record["metrics"]["event_count"] for record in records]
    coincidence = [
        0.0
        if record["metrics"]["jet_coincidence_fraction"] is None
        else record["metrics"]["jet_coincidence_fraction"]
        for record in records
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0].bar(x, counts, color="#2A7F8E")
    axes[0].set(ylabel="Spike count", title="Event yield by control")
    axes[1].bar(x, coincidence, color="#D97706")
    axes[1].set(
        ylabel=r"$C_{\mathrm{jet}}$",
        ylim=(0.0, 1.05),
        title="Jet/reconnection coincidence",
    )
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        "Controls separate strict topping from jet-conditioned coincidence",
        color="#0B2545",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def _plot_spectrum_comparison(
    conditioned: RadioResult,
    uniform: RadioResult,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    for axis, radio, title in (
        (axes[0], conditioned, "Jet-conditioned: no valid candidates"),
        (axes[1], uniform, "Uniform control: geometric topping only"),
    ):
        axis.pcolormesh(
            radio.times_s,
            radio.frequencies_mhz,
            radio.intensity,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axis.plot(
            radio.times_s,
            radio.ridge_frequency_mhz,
            color="#67E8F9",
            linewidth=1.6,
        )
        if radio.spike_catalog.size:
            axis.scatter(
                radio.spike_catalog[:, 0],
                radio.spike_catalog[:, 1],
                facecolors="none",
                edgecolors="#FDE68A",
                s=42,
                linewidths=1.0,
            )
        axis.set(
            title=title,
            xlabel="Time (s)",
            ylabel="Frequency (MHz)",
            xlim=(0.0, radio.times_s[-1]),
            ylim=(
                radio.frequencies_mhz.min(),
                radio.frequencies_mhz.max(),
            ),
        )
    fig.suptitle(
        "Strict topping does not by itself establish jet coincidence",
        color="#0B2545",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def _plot_sign_comparison(
    physical: MHDResult,
    legacy: MHDResult,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), constrained_layout=True)
    axes[0].plot(
        physical.times,
        physical.magnetic_energy + physical.kinetic_energy,
        color="#0B2545",
        linewidth=2.4,
        label="physical [j, ψ]",
    )
    axes[0].plot(
        legacy.times,
        legacy.magnetic_energy + legacy.kinetic_energy,
        color="#D97706",
        linewidth=2.4,
        label="legacy [ψ, j]",
    )
    axes[0].set(
        title="Finite-dissipation total energy",
        xlabel="Normalized MHD time",
        ylabel="Total energy",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        physical.times,
        physical.max_speed,
        color="#2A7F8E",
        linewidth=2.4,
        label="physical",
    )
    axes[1].plot(
        legacy.times,
        legacy.max_speed,
        color="#C9302C",
        linewidth=2.4,
        label="legacy",
    )
    axes[1].set(
        title="The sign choice changes the flow solution",
        xlabel="Normalized MHD time",
        ylabel="Peak speed",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        "The legacy sign is retained only as a diagnostic comparison",
        color="#0B2545",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_suite(output_dir: Path) -> dict[str, Any]:
    """Execute all documented resolution and scientific-control cases."""

    output_dir.mkdir(parents=True, exist_ok=True)
    base_kwargs = {
        "lorentz_convention": "physical",
        "snapshot_stride": 10,
    }
    spatial_configs = [
        MHDConfig(
            nx=48,
            ny=48,
            dt=0.010,
            steps=200,
            snapshot_stride=5,
            lorentz_convention="physical",
        ),
        MHDConfig(nx=96, ny=96, dt=0.005, steps=400, **base_kwargs),
        MHDConfig(
            nx=192,
            ny=192,
            dt=0.0025,
            steps=800,
            snapshot_stride=20,
            lorentz_convention="physical",
        ),
    ]
    spatial: list[dict[str, Any]] = []
    spatial_results: list[MHDResult] = []
    for index, config in enumerate(spatial_configs):
        result, elapsed = _run_mhd_case(config)
        spatial_results.append(result)
        spatial.append(_case_record(f"spatial_{index + 1}", config, result, elapsed))

    temporal_configs = [
        spatial_configs[1],
        MHDConfig(nx=96, ny=96, dt=0.0025, steps=800, **base_kwargs),
        MHDConfig(nx=96, ny=96, dt=0.00125, steps=1600, **base_kwargs),
    ]
    temporal: list[dict[str, Any]] = [spatial[1]]
    for index, config in enumerate(temporal_configs[1:], start=2):
        result, elapsed = _run_mhd_case(config)
        temporal.append(_case_record(f"temporal_{index}", config, result, elapsed))

    base_config = spatial_configs[1]
    base_result = spatial_results[1]
    radio_config = RadioConfig()
    jet_config = JetConfig()
    base_jet = diagnose_jet(base_result, base_config, jet_config)

    legacy_config = replace(base_config, lorentz_convention="legacy")
    legacy_result, legacy_elapsed = _run_mhd_case(legacy_config)
    no_perturb_config = replace(base_config, perturbation_amplitude=0.0)
    no_perturb_result, no_perturb_elapsed = _run_mhd_case(no_perturb_config)

    controls = [
        _radio_case(
            "physical_jet",
            base_result,
            base_config,
            radio_config,
            jet_config,
            SEED,
            "jet",
            base_jet,
        ),
        _radio_case(
            "uniform_spikes",
            base_result,
            base_config,
            radio_config,
            jet_config,
            SEED,
            "uniform",
            base_jet,
        ),
        _radio_case(
            "shuffled_jet",
            base_result,
            base_config,
            radio_config,
            jet_config,
            SEED,
            "jet",
            _shifted_jet(base_jet, base_result, jet_config),
        ),
        _radio_case(
            "no_perturbation",
            no_perturb_result,
            no_perturb_config,
            radio_config,
            jet_config,
            SEED,
            "jet",
        ),
    ]
    for threshold in (0.4, 0.6, 0.8):
        threshold_config = replace(
            jet_config,
            jet_threshold=threshold,
            reconnection_threshold=threshold,
        )
        controls.append(
            _radio_case(
                f"threshold_{threshold:.1f}",
                base_result,
                base_config,
                radio_config,
                threshold_config,
                SEED,
                "jet",
            )
        )
    seed_records = [
        _radio_case(
            f"seed_{SEED + offset}",
            base_result,
            base_config,
            radio_config,
            jet_config,
            SEED + offset,
            "jet",
            base_jet,
        )
        for offset in range(10)
    ]

    mhd_controls = [
        _case_record(
            "physical_sign",
            base_config,
            base_result,
            spatial[1]["elapsed_s"],
        ),
        _case_record(
            "legacy_sign",
            legacy_config,
            legacy_result,
            legacy_elapsed,
        ),
        _case_record(
            "no_perturbation",
            no_perturb_config,
            no_perturb_result,
            no_perturb_elapsed,
        ),
    ]
    summary = {
        "schema_version": 1,
        "seed": SEED,
        "spatial_cases": spatial,
        "spatial_convergence": _convergence_record(spatial),
        "temporal_cases": temporal,
        "temporal_convergence": _convergence_record(temporal),
        "mhd_controls": mhd_controls,
        "radio_controls": controls,
        "radio_seed_reproducibility": seed_records,
    }

    summary_path = output_dir / "science_suite.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "convergence_cases.csv"
    _write_csv(csv_path, spatial + temporal[1:])
    convergence_path = output_dir / "convergence_summary.png"
    controls_path = output_dir / "control_summary.png"
    spectrum_path = output_dir / "spectrum_control_comparison.png"
    sign_path = output_dir / "lorentz_sign_comparison.png"
    _plot_convergence(spatial, temporal, convergence_path)
    _plot_controls(controls, controls_path)
    conditioned_radio = synthesize_radio_proxy(
        base_result,
        radio_config,
        SEED,
        jet_result=base_jet,
        jet_config=jet_config,
        spike_coupling="jet",
    )
    uniform_radio = synthesize_radio_proxy(
        base_result,
        radio_config,
        SEED,
        jet_result=base_jet,
        jet_config=jet_config,
        spike_coupling="uniform",
    )
    _plot_spectrum_comparison(conditioned_radio, uniform_radio, spectrum_path)
    _plot_sign_comparison(base_result, legacy_result, sign_path)

    files = [
        summary_path,
        csv_path,
        convergence_path,
        controls_path,
        spectrum_path,
        sign_path,
    ]
    manifest = output_dir / "SHA256SUMS.txt"
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(files)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "outputs"
            / "experiments"
            / "science_suite_seed20260726"
        ),
    )
    args = parser.parse_args(argv)
    summary = run_suite(args.output_dir)
    print(
        json.dumps(
            {
                "spatial_convergence": summary["spatial_convergence"],
                "temporal_convergence": summary["temporal_convergence"],
                "output": args.output_dir.name,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
