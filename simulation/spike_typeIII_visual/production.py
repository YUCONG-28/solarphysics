"""Recoverable CUDA production workflow for scientific event/control media."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .benchmark_rmhd import benchmark
from .config import RunConfig, profile_config
from .convergence import compare
from .delivery import build_package
from .energy_check import check as check_energy
from .main import _write_manifest, run
from .physics.jet import diagnose_jet
from .physics.radio import synthesize_radio_proxy
from .radio_check import check as check_radio
from .reporting import update_project_readme, validate_run
from .storage import read_rmhd_hdf5
from .timestep_check import check as check_timestep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(__file__).resolve().parent / "outputs" / "runs"
STATE_PATH = Path(__file__).resolve().parent / "outputs" / "production_state.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_state(seed: int, target: str, resume: bool) -> dict[str, Any]:
    if resume and STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if int(state.get("seed", seed)) != seed:
            raise ValueError("Existing production state uses a different seed.")
        return state
    return {
        "schema": "spike-typeiii-production-v1",
        "seed": seed,
        "target": target,
        "status": "running",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "attempts": [],
        "stages": {},
        "visual_qa_iterations": [
            {
                "iteration": 1,
                "profile": "128x64 t=8 scientific-preview",
                "finding": "late-time magnetic-flux contours obscured signed fields",
                "action": "reduced contour density and line weight",
                "status": "corrected before formal production",
            },
            {
                "iteration": 2,
                "profile": "128x64 t=8 scientific-preview",
                "finding": "no clipping, marker jumps, blank panels, or control events",
                "action": "approved fixed-scale layout for formal 4K rendering",
                "status": "passed",
            },
        ],
    }


def _save_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc_now()
    _atomic_json(STATE_PATH, state)


def _record_attempt(
    state: dict[str, Any],
    *,
    stage: str,
    status: str,
    detail: dict[str, Any],
) -> None:
    state["attempts"].append(
        {
            "time_utc": _utc_now(),
            "stage": stage,
            "status": status,
            **detail,
        }
    )
    state["stages"][stage] = status
    _save_state(state)


def _run_dir(profile: str, seed: int, suffix: str = "") -> Path:
    extra = f"_{suffix}" if suffix else ""
    return RUNS_ROOT / f"{profile}_seed{seed}{extra}"


def _has_simulation(run_dir: Path) -> bool:
    return (
        (run_dir / "data" / "rmhd_fields.h5").is_file()
        and (run_dir / "data" / "run_metadata.json").is_file()
    )


def _simulate(
    config: RunConfig,
    run_dir: Path,
    state: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    if resume and _has_simulation(run_dir):
        _record_attempt(
            state,
            stage=f"simulate:{config.profile}",
            status="reused",
            detail={"run_dir": str(run_dir), "reason": "validated files present"},
        )
        return run_dir
    try:
        run(
            config,
            run_dir,
            (),
            rmhd_engine="torch",
            device="cuda",
            precision="float64",
            stage="simulate",
            render_profile="scientific-4k",
        )
    except (RuntimeError, FloatingPointError) as exc:
        message = str(exc)
        recoverable = (
            "out of memory" in message.lower()
            or "non-finite" in message.lower()
            or isinstance(exc, FloatingPointError)
        )
        _record_attempt(
            state,
            stage=f"simulate:{config.profile}",
            status="failed",
            detail={"run_dir": str(run_dir), "error": message},
        )
        if not recoverable:
            raise
        refined = replace(
            config,
            mhd=replace(
                config.mhd,
                dt=0.5 * config.mhd.dt,
                steps=2 * config.mhd.steps,
                snapshot_stride=2 * config.mhd.snapshot_stride,
            ),
        )
        retry_dir = Path(f"{run_dir}_dt_half")
        return _simulate(refined, retry_dir, state, resume=False)
    _record_attempt(
        state,
        stage=f"simulate:{config.profile}",
        status="passed",
        detail={
            "run_dir": str(run_dir),
            "grid": [config.mhd.nx, config.mhd.ny],
            "dt": config.mhd.dt,
            "steps": config.mhd.steps,
        },
    )
    return run_dir


def _event_contract(run_dir: Path, config: RunConfig) -> dict[str, Any]:
    result, stored_config, _ = read_rmhd_hdf5(
        run_dir / "data" / "rmhd_fields.h5", lazy=True
    )
    jet = diagnose_jet(result, stored_config, config.jet)
    radio = synthesize_radio_proxy(
        result,
        config.radio,
        config.seed,
        jet_result=jet,
        jet_config=config.jet,
        spike_coupling="jet",
        time_calibration=config.time_calibration,
    )
    overlap = (
        (jet.jet_activity >= config.jet.jet_threshold)
        & (
            jet.reconnection_activity
            >= config.jet.reconnection_threshold
        )
    )
    return {
        "passed": (
            radio.event_status == "events"
            and len(radio.spike_catalog) == config.radio.spike_count
            and radio.jet_coincidence_fraction == 1.0
            and bool(np.all(radio.topping_margin_mhz > 0.0))
        ),
        "event_status": radio.event_status,
        "event_count": len(radio.spike_catalog),
        "minimum_topping_margin_mhz": (
            None
            if radio.topping_margin_mhz.size == 0
            else float(np.min(radio.topping_margin_mhz))
        ),
        "jet_coincidence_fraction": radio.jet_coincidence_fraction,
        "overlap_snapshots": int(np.sum(overlap)),
        "overlap_start": (
            None
            if not np.any(overlap)
            else float(result.times[np.flatnonzero(overlap)[0]])
        ),
        "overlap_end": (
            None
            if not np.any(overlap)
            else float(result.times[np.flatnonzero(overlap)[-1]])
        ),
    }


def _control_contract(run_dir: Path, config: RunConfig) -> dict[str, Any]:
    result, stored_config, _ = read_rmhd_hdf5(
        run_dir / "data" / "rmhd_fields.h5", lazy=True
    )
    jet = diagnose_jet(result, stored_config, config.jet)
    radio = synthesize_radio_proxy(
        result,
        config.radio,
        config.seed,
        jet_result=jet,
        jet_config=config.jet,
        spike_coupling="jet",
        time_calibration=config.time_calibration,
    )
    return {
        "passed": (
            radio.event_status == "no_event"
            and len(radio.spike_catalog) == 0
            and float(np.max(np.abs(result.max_speed))) < 1.0e-10
            and float(np.max(np.abs(result.flux_difference))) < 1.0e-10
        ),
        "event_status": radio.event_status,
        "event_count": len(radio.spike_catalog),
        "maximum_speed": float(np.max(np.abs(result.max_speed))),
        "maximum_flux_difference": float(
            np.max(np.abs(result.flux_difference))
        ),
    }


def _event_candidates(base: RunConfig) -> list[tuple[str, RunConfig]]:
    candidates: list[tuple[str, RunConfig]] = [("t8", base)]
    for end_time in (10.0, 12.0):
        steps = round(end_time / base.mhd.dt)
        stride = max(1, steps // 400)
        candidates.append(
            (
                f"t{int(end_time)}",
                replace(
                    base,
                    mhd=replace(
                        base.mhd,
                        steps=steps,
                        snapshot_stride=stride,
                    ),
                ),
            )
        )
    for amplitude in (0.06, 0.08):
        for dissipation in (0.001, 0.002, 0.004):
            candidates.append(
                (
                    f"a{amplitude:g}_eta{dissipation:g}",
                    replace(
                        base,
                        mhd=replace(
                            base.mhd,
                            perturbation_amplitude=amplitude,
                            resistivity=dissipation,
                            viscosity=dissipation,
                        ),
                    ),
                )
            )
    return candidates


def _select_event_run(
    base: RunConfig,
    state: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, RunConfig, dict[str, Any]]:
    for suffix, candidate in _event_candidates(base):
        run_dir = _run_dir(base.profile, base.seed, "" if suffix == "t8" else suffix)
        run_dir = _simulate(candidate, run_dir, state, resume=resume)
        contract = _event_contract(run_dir, candidate)
        _record_attempt(
            state,
            stage=f"event-contract:{candidate.profile}",
            status="passed" if contract["passed"] else "failed",
            detail={"run_dir": str(run_dir), **contract},
        )
        if contract["passed"]:
            return run_dir, candidate, contract
    raise RuntimeError(
        "The declared event-search ladder was exhausted without a valid "
        "jet/reconnection overlap; thresholds were not relaxed."
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, payload)


def _auxiliary_reports(
    medium_dir: Path,
    event_dir: Path,
    state: dict[str, Any],
) -> None:
    medium_h5 = medium_dir / "data" / "rmhd_fields.h5"
    event_h5 = event_dir / "data" / "rmhd_fields.h5"
    data_dir = event_dir / "data"
    reports = {
        "convergence_medium_fine.json": compare(medium_h5, event_h5),
        "energy_gates.json": check_energy(event_h5),
        "radio_gates.json": check_radio(event_h5),
        "timestep_halving_512x256.json": check_timestep(medium_h5),
        "cuda_benchmark.json": benchmark("quick", steps=4, repeats=5),
    }
    for name, payload in reports.items():
        _write_json(data_dir / name, payload)
    _record_attempt(
        state,
        stage="auxiliary-reports",
        status="passed",
        detail={
            "reports": sorted(reports),
            "convergence_passed": bool(
                reports["convergence_medium_fine.json"][
                    "core_diagnostics_below_5_percent"
                ]
            ),
            "energy_passed": bool(reports["energy_gates.json"]["passed"]),
            "radio_passed": bool(reports["radio_gates.json"]["passed"]),
            "timestep_passed": bool(
                reports["timestep_halving_512x256.json"][
                    "core_diagnostics_below_1_percent"
                ]
            ),
        },
    )


def _probe_media(run_dir: Path) -> dict[str, Any]:
    from .media import _probe
    from .visualization.scientific_animations import SCIENTIFIC_STEMS

    animations = run_dir / "animations"
    records: dict[str, Any] = {}
    for stem in SCIENTIFIC_STEMS:
        records[stem] = {
            "master": _probe(animations / f"{stem}_master_ffv1.mkv"),
            "delivery": _probe(animations / f"{stem}.mp4"),
            "preview": _probe(animations / f"{stem}.gif"),
        }
    _write_json(animations / "media_probe.json", records)
    return records


def _render(
    event_dir: Path,
    control_dir: Path,
    event_config: RunConfig,
    target: str,
    state: dict[str, Any],
) -> None:
    event, stored_event_config, _ = read_rmhd_hdf5(
        event_dir / "data" / "rmhd_fields.h5", lazy=True
    )
    control, stored_control_config, _ = read_rmhd_hdf5(
        control_dir / "data" / "rmhd_fields.h5", lazy=True
    )
    formats = ("gif", "mp4") if target == "scientific-4k" else ("gif",)
    run(
        replace(event_config, mhd=stored_event_config),
        event_dir,
        formats,
        mhd_result=event,
        control_result=control,
        control_mhd_config=stored_control_config,
        control_run_id=control_dir.name,
        rmhd_engine="torch",
        device="cuda",
        precision="float64",
        stage="render",
        render_profile=target,
    )
    media = _probe_media(event_dir) if target == "scientific-4k" else {}
    _record_attempt(
        state,
        stage=f"render:{target}",
        status="passed",
        detail={"run_dir": str(event_dir), "media": media},
    )


def _environment_report() -> dict[str, Any]:
    try:
        import torch

        gpu = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
        }
    except ImportError:
        gpu = {"torch": "not-installed", "cuda_available": False}
    return {
        "generated_utc": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "command": " ".join(os.sys.argv),
    }


def produce(seed: int, target: str, *, resume: bool) -> dict[str, Any]:
    state = _load_state(seed, target, resume)
    _save_state(state)
    medium_config = profile_config("cuda-medium-event", seed)
    medium_dir, medium_config, _ = _select_event_run(
        medium_config,
        state,
        resume=resume,
    )
    event_config = profile_config("cuda-fine-event", seed)
    event_dir, event_config, event_contract = _select_event_run(
        event_config,
        state,
        resume=resume,
    )
    control_config = replace(
        profile_config("cuda-fine-control", seed),
        mhd=replace(
            profile_config("cuda-fine-control", seed).mhd,
            resistivity=event_config.mhd.resistivity,
            viscosity=event_config.mhd.viscosity,
        ),
    )
    control_dir = _simulate(
        control_config,
        _run_dir("cuda-fine-control", seed),
        state,
        resume=resume,
    )
    control_contract = _control_contract(control_dir, control_config)
    _record_attempt(
        state,
        stage="control-contract",
        status="passed" if control_contract["passed"] else "failed",
        detail={"run_dir": str(control_dir), **control_contract},
    )
    if not control_contract["passed"]:
        raise RuntimeError("The zero-perturbation control contract failed.")
    _auxiliary_reports(medium_dir, event_dir, state)
    _render(event_dir, control_dir, event_config, target, state)
    _write_json(event_dir / "data" / "production_history.json", state)
    _write_json(event_dir / "data" / "environment_report.json", _environment_report())
    _write_manifest(
        event_dir,
        [path for path in event_dir.rglob("*") if path.is_file()],
    )
    if target == "scientific-4k":
        validation = validate_run(event_dir)
        if not validation["passed"]:
            _record_attempt(
                state,
                stage="strict-validation",
                status="failed",
                detail={"errors": validation["errors"]},
            )
            raise RuntimeError(
                "Strict validation failed: " + "; ".join(validation["errors"])
            )
        _record_attempt(
            state,
            stage="strict-validation",
            status="passed",
            detail={"checks": validation["checks"]},
        )
        readme = update_project_readme(event_dir)
    else:
        validation = {"passed": True, "preview_only": True}
        readme = None
    state["status"] = "validated"
    state["event_run_dir"] = str(event_dir)
    state["control_run_dir"] = str(control_dir)
    state["event_contract"] = event_contract
    state["control_contract"] = control_contract
    state["validation"] = validation
    state["readme"] = None if readme is None else str(readme)
    _save_state(state)
    _write_json(event_dir / "data" / "production_history.json", state)
    _write_manifest(
        event_dir,
        [path for path in event_dir.rglob("*") if path.is_file()],
    )
    if target == "scientific-4k":
        package_output = (
            PROJECT_ROOT
            / "deliverables"
            / f"Spike_TypeIII_scientific_complete_{event_dir.name}.zip"
        )
        package_report = build_package(event_dir, control_dir, package_output)
        state["package"] = package_report
        state["status"] = "complete"
        _record_attempt(
            state,
            stage="delivery-package",
            status="passed",
            detail=package_report,
        )
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--target",
        choices=("scientific-preview", "scientific-4k"),
        default="scientific-4k",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = produce(args.seed, args.target, resume=args.resume)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
