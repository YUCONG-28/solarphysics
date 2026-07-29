"""Isolated build, run, ingest, and benchmark workflow for MPI-AMRVAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .amrvac_io import ingest_amrvac, read_dat_v5_header

PACKAGE_ROOT = Path(__file__).resolve().parent
SIMULATION_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
AMRVAC_VENDOR = SIMULATION_ROOT / "amrvac" / "amrvac"
CASE_SOURCE = SIMULATION_ROOT / "amrvac" / "spike_topping_solar_jet"
LOCAL_ROOT = WORKSPACE_ROOT / "Local" / "amrvac"


@dataclass(frozen=True)
class BuildSpec:
    """Reproducible isolated AMRVAC build choices."""

    jobs: int = max(1, min(4, os.cpu_count() or 1))
    rebuild: bool = False

    def __post_init__(self) -> None:
        if self.jobs < 1:
            raise ValueError("AMRVAC build jobs must be positive.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint() -> str:
    """Hash the vendor snapshot and project case without using Git metadata."""

    digest = hashlib.sha256()
    ignored_parts = {".git", "__pycache__", ".pytest_cache", "autom4te.cache"}
    ignored_suffixes = {
        ".o",
        ".mod",
        ".pyc",
        ".dat",
        ".vtu",
        ".log",
        ".a",
    }
    for root in (AMRVAC_VENDOR, CASE_SOURCE):
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                not path.is_file()
                or any(part in ignored_parts for part in relative.parts)
                or path.suffix.lower() in ignored_suffixes
                or path.name == "amrvac"
            ):
                continue
            digest.update(root.name.encode("utf-8"))
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _sanitize(value: str) -> str:
    value = re.sub(r"/Users/[^/\s]+", "<USER_HOME>", value)
    value = re.sub(r"/home/[^/\s]+", "<USER_HOME>", value)
    value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", value)
    value = re.sub(
        r"(?im)^(?:host(?:name)?|node)\s*[:=]\s*\S+",
        "host=<REDACTED>",
        value,
    )
    return value


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _copy_vendor(destination: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".DS_Store",
        "__pycache__",
        ".pytest_cache",
        "*.o",
        "*.mod",
        "*.a",
        "*.dat",
        "*.vtu",
        "*.log",
        "amrvac",
    )
    shutil.copytree(AMRVAC_VENDOR, destination, ignore=ignore)


def build(spec: BuildSpec) -> Path:
    """Build the project case from a copied, read-only vendor snapshot."""

    if not AMRVAC_VENDOR.is_dir() or not CASE_SOURCE.is_dir():
        raise FileNotFoundError("AMRVAC vendor snapshot or project case is missing.")
    fingerprint = _tree_fingerprint()
    build_root = LOCAL_ROOT / "build" / fingerprint[:16]
    vendor = build_root / "vendor"
    case = build_root / "case"
    binary = case / "amrvac"
    manifest_path = build_root / "build_manifest.json"
    if not spec.rebuild and binary.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_content_hash") == fingerprint:
            return binary
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)
    _copy_vendor(vendor)
    shutil.copytree(CASE_SOURCE, case)
    env = os.environ.copy()
    env["AMRVAC_DIR"] = str(vendor)
    setup = _run([str(vendor / "setup.pl"), "-d=2", "-arch=default"], cwd=case, env=env)
    compiled = _run(["make", "-j", str(spec.jobs)], cwd=case, env=env)
    if not binary.is_file():
        raise RuntimeError("AMRVAC build completed without producing the binary.")
    sidecar_path = case / "bridge_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["source_content_hash"] = fingerprint
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "solver": "mpi-amrvac",
        "source_content_hash": fingerprint,
        "binary_sha256": _sha256(binary),
        "jobs": spec.jobs,
        "system": platform.system(),
        "architecture": platform.machine(),
        "setup_log": _sanitize(setup.stdout + setup.stderr).splitlines(),
        "build_log_tail": _sanitize(compiled.stdout + compiled.stderr).splitlines()[-80:],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return binary


def doctor(binary: Path | None = None) -> dict[str, object]:
    """Report portable prerequisites without exposing host or user identity."""

    result: dict[str, object] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "vendor_snapshot": AMRVAC_VENDOR.is_dir(),
        "project_case": CASE_SOURCE.is_dir(),
        "mpif90": shutil.which("mpif90") is not None,
        "mpirun": shutil.which("mpirun") is not None,
        "make": shutil.which("make") is not None,
        "perl": shutil.which("perl") is not None,
        "vendor_policy": "read-only-copy-on-build",
    }
    if binary is not None:
        result["binary_exists"] = Path(binary).is_file()
        result["binary_sha256"] = _sha256(binary) if Path(binary).is_file() else None
    return result


def run(
    binary: Path,
    *,
    run_id: str,
    ranks: int = 1,
    overwrite: bool = False,
) -> tuple[Path, float]:
    """Run the fixed no-driver development smoke under ignored Local storage."""

    if ranks < 1:
        raise ValueError("AMRVAC ranks must be positive.")
    binary = Path(binary).resolve()
    case = binary.parent
    run_dir = LOCAL_ROOT / "runs" / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"AMRVAC run {run_id!r} already exists.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    for name in ("amrvac.par", "bridge_sidecar.json"):
        shutil.copy2(case / name, run_dir / name)
    command = [str(binary), "-i", "amrvac.par"]
    if ranks > 1:
        launcher = shutil.which("mpirun")
        if launcher is None:
            raise RuntimeError("AMRVAC MPI execution requires mpirun.")
        command = [launcher, "-np", str(ranks), *command]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    started = perf_counter()
    completed = _run(command, cwd=run_dir, env=env)
    elapsed = perf_counter() - started
    (run_dir / "amrvac.stdout.log").write_text(
        _sanitize(completed.stdout),
        encoding="utf-8",
    )
    (run_dir / "amrvac.stderr.log").write_text(
        _sanitize(completed.stderr),
        encoding="utf-8",
    )
    dat_paths = sorted(run_dir.glob("*.dat"))
    if not dat_paths:
        raise RuntimeError("AMRVAC smoke did not produce a dat-v5 snapshot.")
    header = read_dat_v5_header(dat_paths[-1])
    manifest = {
        "schema_version": 1,
        "profile": "solar-jet-dev-static",
        "grid_base": [64, 128],
        "maximum_amr_level": 2,
        "time_limit": 0.2,
        "ranks": ranks,
        "wall_time_s": elapsed,
        "native_format": "dat-v5",
        "final_iteration": header.iteration,
        "final_time": header.time,
        "binary_sha256": _sha256(binary),
        "parameter_sha256": _sha256(run_dir / "amrvac.par"),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir, elapsed


def ingest(run_dir: Path, output: Path) -> dict[str, object]:
    """Convert dat-v5 snapshots using the mandatory background-field sidecar."""

    paths = sorted(Path(run_dir).glob("*.dat"))
    series = ingest_amrvac(
        paths,
        sidecar_path=Path(run_dir) / "bridge_sidecar.json",
        output_path=output,
    )
    return {
        "snapshots": len(series.times),
        "analysis_grid": [len(series.grid.x), len(series.grid.y)],
        "minimum_density": float(series.rho.min()),
        "minimum_pressure": float(series.pressure.min()),
        "max_divergence_normalized": series.divergence_rms,
        "status": "development-smoke-only",
    }


def benchmark(
    binary: Path,
    *,
    ranks: list[int],
    repeats: int,
) -> Path:
    """Benchmark only the fixed development smoke; no science claim is made."""

    rows: list[dict[str, object]] = []
    for rank_count in ranks:
        for repeat in range(1, repeats + 1):
            run_dir, elapsed = run(
                binary,
                run_id=f"benchmark_dev_r{rank_count}_n{repeat}",
                ranks=rank_count,
                overwrite=True,
            )
            rows.append(
                {
                    "ranks": rank_count,
                    "repeat": repeat,
                    "wall_time_s": elapsed,
                    "run_hash": _sha256(run_dir / "run_manifest.json")[:16],
                }
            )
    output = LOCAL_ROOT / "benchmarks" / "solar_jet_dev.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": 1, "runs": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--binary", type=Path)
    build_parser = commands.add_parser("build")
    build_parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    build_parser.add_argument("--rebuild", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--ranks", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    ingest_parser = commands.add_parser("ingest")
    ingest_parser.add_argument("--run-dir", type=Path, required=True)
    ingest_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--binary", type=Path, required=True)
    benchmark_parser.add_argument("--ranks", nargs="+", type=int, default=[1])
    benchmark_parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(args.binary), indent=2))
    elif args.command == "build":
        binary = build(BuildSpec(args.jobs, args.rebuild))
        print(f"Built isolated AMRVAC binary: {binary.parent.parent.name}/case/amrvac")
    elif args.command == "run":
        run_dir, elapsed = run(
            args.binary,
            run_id=args.run_id,
            ranks=args.ranks,
            overwrite=args.overwrite,
        )
        print(f"Completed private AMRVAC smoke {run_dir.name!r} in {elapsed:.3f} s")
    elif args.command == "ingest":
        print(json.dumps(ingest(args.run_dir, args.output), indent=2))
    elif args.command == "benchmark":
        print(
            benchmark(
                args.binary,
                ranks=args.ranks,
                repeats=args.repeats,
            ).name
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
