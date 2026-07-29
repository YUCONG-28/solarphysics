"""Build, run, ingest, and benchmark the private Athena C backend."""

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

import h5py
import numpy as np

from .athena_io import (
    discover_binary_dumps,
    ingest_run_directory,
    initial_balance_metrics,
    initial_binary_metrics,
    read_dump_series,
)
from .physics.normalization import PhysicalNormalization

PACKAGE_ROOT = Path(__file__).resolve().parent
SIMULATION_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
ATHENA_SOURCE = SIMULATION_ROOT / "fluxrope_demo" / "athena4.2"
ATHENA_INPUTS = {
    "spike_topping_jet": (
        ATHENA_SOURCE / "tst" / "2D-mhd" / "athinput.spike_topping_jet"
    ),
    "spike_topping_solar_jet": (
        ATHENA_SOURCE / "tst" / "2D-mhd" / "athinput.spike_topping_solar_jet"
    ),
}
ATHENA_INPUT = ATHENA_INPUTS["spike_topping_jet"]
LOCAL_ROOT = WORKSPACE_ROOT / "Local" / "athena"

PROFILES: dict[str, dict[str, float | int | str]] = {
    "smoke": {
        "problem": "spike_topping_jet",
        "nx": 128,
        "ny": 64,
        "tlim": 0.20,
    },
    "coarse": {
        "problem": "spike_topping_jet",
        "nx": 256,
        "ny": 128,
        "tlim": 2.0,
    },
    "standard": {
        "problem": "spike_topping_jet",
        "nx": 512,
        "ny": 256,
        "tlim": 2.0,
    },
    "fine": {
        "problem": "spike_topping_jet",
        "nx": 1024,
        "ny": 512,
        "tlim": 2.0,
    },
    "jet-smoke": {
        "problem": "spike_topping_solar_jet",
        "nx": 128,
        "ny": 256,
        "tlim": 7.54,
        "physical_duration_s": 120.0,
    },
    "jet-static": {
        "problem": "spike_topping_solar_jet",
        "nx": 256,
        "ny": 512,
        "tlim": 37.7,
        "physical_duration_s": 600.0,
    },
    "jet-coarse": {
        "problem": "spike_topping_solar_jet",
        "nx": 256,
        "ny": 512,
        "tlim": 37.7,
        "physical_duration_s": 600.0,
    },
    "jet-standard": {
        "problem": "spike_topping_solar_jet",
        "nx": 512,
        "ny": 1024,
        "tlim": 37.7,
        "physical_duration_s": 600.0,
    },
    "jet-fine": {
        "problem": "spike_topping_solar_jet",
        "nx": 1024,
        "ny": 2048,
        "tlim": 37.7,
        "physical_duration_s": 600.0,
    },
}
OVERRIDE_PATTERN = re.compile(
    r"^(?:problem|time|domain1|output[1-4])/[A-Za-z0-9_]+=[A-Za-z0-9_.+-]+$"
)


@dataclass(frozen=True)
class BuildSpec:
    """Reproducible Athena build choices without machine identifiers."""

    mpi: bool = False
    performance: bool = False
    flux: str = "hlld"
    problem: str = "spike_topping_jet"
    jobs: int = max(1, min(8, os.cpu_count() or 1))
    rebuild: bool = False

    def __post_init__(self) -> None:
        if self.flux not in {"hlld", "roe"}:
            raise ValueError("Athena flux must be 'hlld' or 'roe'.")
        if self.problem not in ATHENA_INPUTS:
            raise ValueError(f"Unknown Athena problem {self.problem!r}.")
        if self.jobs < 1:
            raise ValueError("Athena build jobs must be positive.")

    @property
    def name(self) -> str:
        kind = "performance" if self.performance else "reference"
        parallel = "mpi" if self.mpi else "serial"
        suffix = "" if self.flux == "hlld" else f"_{self.flux}"
        problem = "solar_jet" if self.problem.endswith("solar_jet") else "harris"
        return f"{problem}_{kind}_{parallel}{suffix}"

    @property
    def optimization(self) -> str:
        if not self.performance:
            return "-O3"
        native = "-mcpu=native" if platform.system() == "Darwin" else "-march=native"
        return f"-O3 {native}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    """Hash portable source inputs while ignoring in-place build products."""

    ignored_names = {
        ".git",
        "athena",
        "Makefile",
        "Makeoptions",
        "Makedepend",
        "config.h",
        "defs.h",
        "problem.c",
        "config.log",
        "config.status",
        "autom4te.cache",
    }
    ignored_suffixes = {".o", ".bin", ".vtk", ".hst", ".rst", ".pyc"}
    digest = hashlib.sha256()
    for path in sorted(ATHENA_SOURCE.rglob("*")):
        relative = path.relative_to(ATHENA_SOURCE)
        if (
            not path.is_file()
            or any(part in ignored_names for part in relative.parts)
            or path.suffix in ignored_suffixes
        ):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


_PRIVATE_PATTERNS = (
    (re.compile(r"/Users/[^/\s]+"), "<USER_HOME>"),
    (re.compile(r"/home/[^/\s]+"), "<USER_HOME>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"(?im)^(?:host(?:name)?|node)\s*[:=]\s*\S+"), "host=<REDACTED>"),
)


def _sanitize_text(value: str) -> str:
    """Remove personal locators before any command output is persisted."""

    sanitized = value
    for pattern, replacement in _PRIVATE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _load_case_config(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML case configs require PyYAML; JSON is also supported."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("Athena case config must contain one mapping.")
    serialized = json.dumps(data, ensure_ascii=False)
    if re.search(r"(?:/Users/|/home/|[A-Za-z]:[\\/]|@)", serialized):
        raise ValueError("Athena case config contains a private locator.")
    return data


def _normalization_from_case(data: object) -> PhysicalNormalization:
    if not isinstance(data, dict):
        raise TypeError("normalization must be a mapping.")
    required = (
        "length_mm",
        "magnetic_field_gauss",
        "electron_density_cm3",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing normalization values: {', '.join(missing)}")
    return PhysicalNormalization.from_solar_units(
        length_mm=float(data["length_mm"]),
        magnetic_field_gauss=float(data["magnetic_field_gauss"]),
        electron_density_cm3=float(data["electron_density_cm3"]),
        mean_mass_per_electron=float(data.get("mean_mass_per_electron", 1.2)),
    )


def _safe_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _copy_source(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(
        ".DS_Store",
        "*.o",
        "athena",
        "Makefile",
        "Makeoptions",
        "Makedepend",
        "config.h",
        "defs.h",
        "problem.c",
        "config.log",
        "config.status",
        "autom4te.cache",
        "*.bin",
        "*.vtk",
        "*.hst",
        "*.rst",
    )
    shutil.copytree(ATHENA_SOURCE, destination, ignore=ignore)


def build_athena(spec: BuildSpec) -> Path:
    """Create an isolated Athena build under ignored private runtime storage."""

    input_path = ATHENA_INPUTS[spec.problem]
    if not ATHENA_SOURCE.exists() or not input_path.exists():
        raise FileNotFoundError("Athena C source or Spike-Topping input is missing.")
    build_dir = LOCAL_ROOT / "build" / spec.name
    binary = build_dir / "bin" / "athena"
    build_manifest = build_dir / "build_manifest.json"
    source_fingerprint = _source_fingerprint()
    expected_manifest = {
        "schema_version": 2,
        "source_fingerprint": source_fingerprint,
        "mpi": spec.mpi,
        "performance": spec.performance,
        "flux": spec.flux,
        "problem": spec.problem,
        "optimization": spec.optimization,
    }
    if not spec.rebuild and binary.exists() and build_manifest.exists():
        existing = json.loads(build_manifest.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected_manifest.items()):
            return binary
    _copy_source(build_dir)
    configure = [
        "./configure",
        f"--with-problem={spec.problem}",
        "--with-gas=mhd",
        "--with-eos=adiabatic",
        "--with-order=3",
        f"--with-flux={spec.flux}",
        "--with-integrator=ctu",
        "--enable-resistivity",
        "--enable-viscosity",
    ]
    if spec.mpi:
        configure.append("--enable-mpi")
    env = os.environ.copy()
    env["OPT"] = spec.optimization
    if spec.mpi:
        compiler = shutil.which("mpicc")
        if compiler is None:
            raise RuntimeError(
                "MPI build requires mpicc in solar_simulation. Activate that "
                "environment before running the build command."
            )
        env["CC"] = compiler
        if platform.system() == "Darwin":
            system_clang = shutil.which("clang")
            if system_clang is None:
                raise RuntimeError("The macOS MPI build requires Apple Clang.")
            env["OMPI_CC"] = system_clang
    _safe_run(configure, cwd=build_dir, env=env)
    if spec.performance:
        makeoptions = build_dir / "Makeoptions"
        content = makeoptions.read_text(encoding="utf-8")
        content = content.replace(
            "OPT = -O3\n",
            f"OPT = {spec.optimization}\n",
            1,
        )
        makeoptions.write_text(content, encoding="utf-8")
    _safe_run(["make", "all", "-j", str(spec.jobs)], cwd=build_dir, env=env)
    if not binary.exists():
        raise RuntimeError("Athena build completed without producing bin/athena.")
    report = _safe_run([str(binary), "-c"], cwd=build_dir, capture=True).stdout
    required = ("MHD", "CTU", spec.flux, "resistivity", "viscosity")
    missing = [token for token in required if token.lower() not in report.lower()]
    if missing:
        raise RuntimeError(f"Built Athena configuration omits: {', '.join(missing)}")
    build_manifest.write_text(
        json.dumps(
            {
                **expected_manifest,
                "binary_sha256": _sha256(binary),
                "configuration": _sanitize_text(report).splitlines(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return binary


def doctor(binary: Path | None = None) -> dict[str, object]:
    """Inspect the environment and optionally verify one Athena binary."""

    imports: dict[str, str] = {}
    for module_name in (
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "imageio",
        "imageio_ffmpeg",
        "pyvista",
        "vtk",
    ):
        try:
            module = __import__(module_name)
            imports[module_name] = str(getattr(module, "__version__", "available"))
        except ImportError:
            imports[module_name] = "missing"
    result: dict[str, object] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "packages": imports,
        "mpicc": shutil.which("mpicc") is not None,
        "mpirun": shutil.which("mpirun") is not None,
        "athena_source": ATHENA_SOURCE.exists(),
        "input_files": {
            name: path.exists() for name, path in ATHENA_INPUTS.items()
        },
    }
    if binary is not None:
        binary = binary.resolve()
        completed = _safe_run([str(binary), "-c"], cwd=binary.parent, capture=True)
        result["athena_configuration"] = completed.stdout.strip().splitlines()
    return result


def _run_command(binary: Path, ranks: int) -> list[str]:
    if ranks == 1:
        return [str(binary)]
    launcher = shutil.which("mpirun")
    if launcher is None:
        raise RuntimeError("MPI execution requires mpirun in solar_simulation.")
    return [launcher, "-np", str(ranks), str(binary)]


def run_athena(
    binary: Path,
    *,
    profile: str,
    run_id: str,
    ranks: int = 1,
    overwrite: bool = False,
    overrides: list[str] | None = None,
    measure_resources: bool = False,
    case_config: Path | None = None,
) -> tuple[Path, float]:
    """Run a fixed profile and retain all raw products only under Local."""

    if profile not in PROFILES:
        raise ValueError(f"Unknown Athena profile {profile!r}.")
    if ranks < 1:
        raise ValueError("ranks must be positive.")
    binary = binary.resolve()
    run_dir = LOCAL_ROOT / "runs" / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Private run {run_id!r} already exists; choose a new run id."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    settings = PROFILES[profile]
    problem = str(settings["problem"])
    input_source = ATHENA_INPUTS[problem]
    input_copy = run_dir / input_source.name
    shutil.copy2(input_source, input_copy)
    case_overrides: list[str] = []
    normalization: dict[str, float] | None = None
    if case_config is not None:
        case_data = _load_case_config(case_config)
        configured_problem = case_data.get("problem", problem)
        if configured_problem != problem:
            raise ValueError(
                f"Case problem {configured_problem!r} does not match profile "
                f"problem {problem!r}."
            )
        for section, values in case_data.get("athena", {}).items():
            if not isinstance(values, dict):
                raise TypeError("Each Athena case section must be a mapping.")
            case_overrides.extend(
                f"{section}/{key}={value}" for key, value in values.items()
            )
        if "normalization" in case_data:
            normalization = _normalization_from_case(
                case_data["normalization"]
            ).to_metadata()
    command = _run_command(binary, ranks) + [
        "-i",
        input_copy.name,
        f"domain1/Nx1={settings['nx']}",
        f"domain1/Nx2={settings['ny']}",
        f"time/tlim={settings['tlim']}",
        f"job/problem_id={run_id}",
    ]
    if ranks > 1:
        command.append(f"domain1/AutoWithNProc={ranks}")
    safe_overrides = [*case_overrides, *(overrides or [])]
    if profile == "jet-static":
        safe_overrides = [
            value
            for value in safe_overrides
            if not value.startswith("problem/drive_enabled=")
        ]
        safe_overrides.append("problem/drive_enabled=0")
    invalid = [value for value in safe_overrides if not OVERRIDE_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"Invalid Athena parameter overrides: {invalid}")
    command.extend(safe_overrides)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if measure_resources and Path("/usr/bin/time").exists():
        time_flag = "-l" if platform.system() == "Darwin" else "-v"
        command = ["/usr/bin/time", time_flag, *command]
    started = perf_counter()
    completed = _safe_run(command, cwd=run_dir, env=env, capture=True)
    elapsed = perf_counter() - started
    (run_dir / "athena.stdout.log").write_text(
        _sanitize_text(completed.stdout),
        encoding="utf-8",
    )
    (run_dir / "athena.stderr.log").write_text(
        _sanitize_text(completed.stderr),
        encoding="utf-8",
    )
    cycle_matches = re.findall(r"cycle=(\d+)", completed.stdout)
    cycles = int(cycle_matches[-1]) if cycle_matches else 0
    peak_match = re.search(
        r"^\s*(\d+)\s+maximum resident set size",
        completed.stderr,
        flags=re.MULTILINE,
    )
    peak_memory_bytes = int(peak_match.group(1)) if peak_match else None
    effective: dict[str, float | int] = {
        "domain1/Nx1": int(settings["nx"]),
        "domain1/Nx2": int(settings["ny"]),
        "time/tlim": float(settings["tlim"]),
    }
    for override in safe_overrides:
        key, value = override.split("=", 1)
        if key in effective:
            effective[key] = int(value) if key.startswith("domain1/Nx") else float(value)
    cell_updates = (
        int(effective["domain1/Nx1"])
        * int(effective["domain1/Nx2"])
        * cycles
    )
    manifest = {
        "schema_version": 1,
        "profile": profile,
        "problem": problem,
        "grid": [
            effective["domain1/Nx1"],
            effective["domain1/Nx2"],
        ],
        "time_limit": effective["time/tlim"],
        "ranks": ranks,
        "wall_time_s": elapsed,
        "cycles": cycles,
        "cell_updates": cell_updates,
        "cell_updates_per_s": cell_updates / elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "binary_sha256": _sha256(binary),
        "input_sha256": _sha256(input_copy),
        "source_fingerprint": _source_fingerprint(),
        "parameter_overrides": safe_overrides,
        "normalization": normalization,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir, elapsed


def ingest(run_dir: Path, output_path: Path) -> dict[str, float]:
    """Convert raw Athena binary dumps to the schema-v4 HDF5 bridge."""

    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    problem = str(manifest.get("problem", "spike_topping_jet"))
    input_paths = sorted(run_dir.glob("athinput.*"))
    input_text = (
        input_paths[0].read_text(encoding="utf-8") if input_paths else ""
    )

    def parameter(name: str, default: float) -> float:
        value = default
        match = re.search(
            rf"(?m)^\s*{re.escape(name)}\s*=\s*([0-9.eE+-]+)",
            input_text,
        )
        if match:
            value = float(match.group(1))
        for override in manifest.get("parameter_overrides", []):
            if str(override).startswith(f"problem/{name}="):
                value = float(str(override).split("=", 1)[1])
        return value

    paths = discover_binary_dumps(run_dir)
    if not paths:
        raise FileNotFoundError("No numbered Athena .bin dumps were found.")
    initial_index = min(int(path.name.rsplit(".", 2)[-2]) for path in paths)
    initial_paths = [
        path
        for path in paths
        if int(path.name.rsplit(".", 2)[-2]) == initial_index
    ]
    binary_metrics = initial_binary_metrics(read_dump_series(initial_paths)[0])
    if problem == "spike_topping_solar_jet":
        normalization = manifest.get("normalization")
        unit_metadata = (
            {"system": "SI-reference", **normalization}
            if isinstance(normalization, dict)
            else {"system": "dimensionless", "status": "normalization-missing"}
        )
        series = ingest_run_directory(
            run_dir,
            output_path,
            resistivity=parameter("eta_background", 1.0e-5),
            viscosity=parameter("nu_iso", 1.0e-5),
            geometry_kind="open_solar_jet",
            boundary_mode="open",
            diagnostic_centers_y=(2.0,),
            diagnostic_x_points=(0.0,),
            unit_metadata=unit_metadata,
            provenance_metadata={
                "solver": "athena-c-4.2-project-patch",
                "source_content_hash": str(
                    manifest.get("source_fingerprint", "unknown")
                ),
                "native_format": "Athena C primitive BIN",
                "native_format_version": 4,
                "analysis_grid": list(manifest.get("grid", [])),
                "projection_method": "native fixed grid",
                "magnetic_storage": "total field",
                "energy_convention": "three-component total field",
                "resistivity_model": {
                    "kind": "current-triggered",
                    "background": parameter("eta_background", 1.0e-5),
                    "anomalous": parameter("eta_anomalous", 2.0e-4),
                    "threshold": parameter("current_threshold", 5.0),
                },
            },
            diagnostic_metadata={
                "density_floor": parameter("density_floor", 1.0e-10),
                "pressure_floor": parameter("pressure_floor", 1.0e-10),
                "resistivity_model": {
                    "background": parameter("eta_background", 1.0e-5),
                    "anomalous": parameter("eta_anomalous", 2.0e-4),
                    "threshold": parameter("current_threshold", 5.0),
                },
            },
        )
        with h5py.File(output_path, "r") as handle:
            diagnostics = handle["development_diagnostics"]
            maximum_mach = float(np.max(diagnostics["maximum_mach"][...]))
            density_floor_count = int(
                np.sum(diagnostics["density_floor_count"][...])
            )
            pressure_floor_count = int(
                np.sum(diagnostics["pressure_floor_count"][...])
            )
        return {
            "minimum_density": float(np.min(series.rho[0])),
            "minimum_pressure": float(np.min(series.pressure[0])),
            "density_floor_count": density_floor_count,
            "pressure_floor_count": pressure_floor_count,
            "maximum_mach": maximum_mach,
            "ct_divergence_normalized_rms": float(
                series.divergence_normalized_rms[0]
            ),
            "binary_precision_bytes": binary_metrics[
                "binary_precision_bytes"
            ],
        }
    series = ingest_run_directory(run_dir, output_path)
    return {**initial_balance_metrics(series), **binary_metrics}


def benchmark(
    binary: Path,
    *,
    profile: str,
    ranks: list[int],
    repeats: int,
) -> Path:
    """Measure fixed-case wall time without publishing machine identity."""

    if repeats < 1:
        raise ValueError("repeats must be positive.")
    rows: list[dict[str, float | int | str]] = []
    for rank_count in ranks:
        for repeat in range(repeats):
            run_id = f"benchmark_{profile}_r{rank_count}_n{repeat + 1}"
            run_dir, elapsed = run_athena(
                binary,
                profile=profile,
                run_id=run_id,
                ranks=rank_count,
                overwrite=True,
                measure_resources=True,
            )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "profile": profile,
                    "ranks": rank_count,
                    "repeat": repeat + 1,
                    "wall_time_s": elapsed,
                    "cell_updates_per_s": manifest["cell_updates_per_s"],
                    "peak_memory_bytes": manifest["peak_memory_bytes"],
                    "io_fraction": None,
                    "io_fraction_note": (
                        "Athena C does not expose phase timing; no value is "
                        "invented from total wall time."
                    ),
                    "run_hash": _sha256(run_dir / "run_manifest.json")[:16],
                }
            )
    output = LOCAL_ROOT / "benchmarks" / f"{profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": 1, "runs": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--binary", type=Path)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--mpi", action="store_true")
    build_parser.add_argument("--performance", action="store_true")
    build_parser.add_argument("--flux", choices=("hlld", "roe"), default="hlld")
    build_parser.add_argument(
        "--problem",
        choices=tuple(ATHENA_INPUTS),
        default="spike_topping_jet",
    )
    build_parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    build_parser.add_argument("--rebuild", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--ranks", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--case-config", type=Path)
    run_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION/KEY=VALUE",
    )

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--run-dir", type=Path, required=True)
    ingest_parser.add_argument("--output", type=Path, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--binary", type=Path, required=True)
    benchmark_parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="smoke",
    )
    benchmark_parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    benchmark_parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(args.binary), indent=2, ensure_ascii=False))
    elif args.command == "build":
        binary = build_athena(
            BuildSpec(
                args.mpi,
                args.performance,
                args.flux,
                args.problem,
                args.jobs,
                args.rebuild,
            )
        )
        print(f"Built private Athena binary: {binary.parent.parent.name}/{binary.name}")
    elif args.command == "run":
        run_dir, elapsed = run_athena(
            args.binary,
            profile=args.profile,
            run_id=args.run_id,
            ranks=args.ranks,
            overwrite=args.overwrite,
            overrides=args.override,
            case_config=args.case_config,
        )
        print(f"Completed private run {run_dir.name!r} in {elapsed:.3f} s")
    elif args.command == "ingest":
        metrics = ingest(args.run_dir, args.output)
        print(json.dumps(metrics, indent=2))
    elif args.command == "benchmark":
        output = benchmark(
            args.binary,
            profile=args.profile,
            ranks=args.ranks,
            repeats=args.repeats,
        )
        print(f"Private benchmark report: {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
