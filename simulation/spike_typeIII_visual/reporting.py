"""Validation-gated, deterministic project README generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _verify_manifest(run_dir: Path) -> tuple[bool, list[str]]:
    manifest = run_dir / "SHA256SUMS.txt"
    errors: list[str] = []
    if not manifest.is_file():
        return False, ["缺少 SHA256SUMS.txt"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"无法解析校验行：{line}")
            continue
        target = run_dir / relative
        if not target.is_file():
            errors.append(f"缺少文件：{relative}")
        elif _sha256(target) != expected:
            errors.append(f"校验失败：{relative}")
    return not errors, errors


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for formal media validation.")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_packets,codec_name",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_packets"]),
        "codec": stream["codec_name"],
    }


def validate_run(run_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    """Validate one candidate authoritative CUDA run without guessing values."""

    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "data" / "run_metadata.json"
    hdf5_path = run_dir / "data" / "rmhd_fields.h5"
    diagnostics_path = run_dir / "data" / "diagnostics.csv"
    errors: list[str] = []
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    diagnostics = metadata["diagnostics"]
    runtime = metadata["runtime"]
    exports = metadata["exports"]
    profile = str(config["profile"])

    checks: dict[str, bool] = {
        "cuda_profile": profile
        in {
            "cuda-coarse",
            "cuda-medium",
            "cuda-fine",
            "cuda-medium-event",
            "cuda-fine-event",
            "cuda-fine-control",
        },
        "physical_lorentz": config["mhd"]["lorentz_convention"] == "physical",
        "torch_cuda_float64": (
            runtime["execution_backend"] == "torch"
            and runtime["execution_device"] == "cuda"
            and runtime["execution_precision"] == "float64"
        ),
        "authoritative_hdf5": hdf5_path.is_file(),
        "diagnostics_csv": diagnostics_path.is_file(),
        "snapshot_count": int(diagnostics["snapshot_count"]) == 401,
        "divergence": float(diagnostics["divergence_normalized_rms"]) < 1.0e-10,
        "strict_topping": (
            diagnostics["event_status"] == "no_event"
            or (
                int(diagnostics["event_count"]) > 0
                and diagnostics["minimum_topping_margin_mhz"] is not None
                and float(diagnostics["minimum_topping_margin_mhz"]) > 0.0
            )
        ),
        "event_control_contract": (
            (
                diagnostics["event_status"] == "events"
                and int(diagnostics["event_count"]) == 12
                and float(diagnostics["jet_coincidence_fraction"]) == 1.0
            )
            if profile.endswith("-event")
            else (
                diagnostics["event_status"] == "no_event"
                if profile.endswith("-control")
                else True
            )
        ),
        "presentation_render": exports["render_profile"]
        in {"presentation-4k", "scientific-4k"},
        "radio_proxy_data": (run_dir / "data" / "radio_proxy.npz").is_file(),
    }
    budget_limit = (
        2.0e-4
        if profile in {"cuda-fine", "cuda-fine-event", "cuda-fine-control"}
        else 1.0e-3
    )
    budget_value = diagnostics.get("energy_budget_max_abs_fraction")
    checks["energy_budget"] = (
        budget_value is not None and float(budget_value) < budget_limit
    )
    manifest_ok, manifest_errors = _verify_manifest(run_dir)
    checks["sha256_manifest"] = manifest_ok
    errors.extend(manifest_errors)
    for name, passed in checks.items():
        if not passed:
            errors.append(f"未通过：{name}")

    rows = 0
    if diagnostics_path.is_file():
        with diagnostics_path.open(encoding="utf-8", newline="") as stream:
            rows = sum(1 for _ in csv.DictReader(stream))
    checks["diagnostic_rows_match"] = rows == int(diagnostics["snapshot_count"])
    if not checks["diagnostic_rows_match"]:
        errors.append("未通过：diagnostic_rows_match")

    media: dict[str, dict[str, Any]] = {}
    animation_dir = run_dir / "animations"
    expected_stems = (
        (
            "causal_chain",
            "reconnection_topology",
            "bidirectional_outflow",
            "radio_event_control",
        )
        if exports.get("render_profile") == "scientific-4k"
        else ("tearing", "jet", "electron_beam", "typeIII")
    )
    try:
        for stem in expected_stems:
            delivery = animation_dir / f"{stem}.mp4"
            master = animation_dir / f"{stem}_master_ffv1.mkv"
            if not delivery.is_file() or not master.is_file():
                raise FileNotFoundError(stem)
            delivery_probe = _probe_video(delivery)
            master_probe = _probe_video(master)
            media[stem] = {
                "delivery": delivery_probe,
                "master": master_probe,
            }
        checks["four_4k_videos"] = all(
            entry["delivery"]["width"] == 3840
            and entry["delivery"]["height"] == 2160
            and abs(entry["delivery"]["fps"] - 30.0) < 1.0e-6
            and entry["delivery"]["frames"] >= 401
            and entry["master"]["codec"] == "ffv1"
            and entry["master"]["frames"] >= 401
            for entry in media.values()
        )
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
        checks["four_4k_videos"] = False
    if not checks["four_4k_videos"]:
        errors.append("未通过：four_4k_videos")

    try:
        from PIL import Image

        pngs = sorted((run_dir / "figures").glob("*.png"))
        checks["static_4k"] = bool(pngs) and all(
            Image.open(path).size == (3840, 2160) for path in pngs
        )
    except (ImportError, OSError):
        checks["static_4k"] = False
    if not checks["static_4k"]:
        errors.append("未通过：static_4k")
    try:
        from PIL import Image

        previews = [
            Image.open(animation_dir / f"{stem}.gif")
            for stem in expected_stems
        ]
        checks["gif_previews"] = all(
            image.size == (960, 540)
            and int(getattr(image, "n_frames", 1)) >= 30
            for image in previews
        )
    except (ImportError, OSError):
        checks["gif_previews"] = False
    if not checks["gif_previews"]:
        errors.append("未通过：gif_previews")

    required_reports = {
        "energy_gates": "energy_gates.json",
        "radio_gates": "radio_gates.json",
        "timestep_halving": "timestep_halving_512x256.json",
        "medium_fine_convergence": "convergence_medium_fine.json",
        "cuda_benchmark": "cuda_benchmark.json",
    }
    auxiliary: dict[str, Any] = {}
    for check_name, filename in required_reports.items():
        path = run_dir / "data" / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            auxiliary[check_name] = payload
            if check_name in {"energy_gates", "radio_gates"}:
                passed = bool(payload["passed"])
            elif check_name == "timestep_halving":
                passed = bool(payload["core_diagnostics_below_1_percent"])
            elif check_name == "medium_fine_convergence":
                passed = bool(payload["core_diagnostics_below_5_percent"])
            else:
                passed = (
                    int(payload["repeats"]) >= 5
                    and float(payload["psi_relative_l2"]) < 1.0e-9
                    and payload["precision"] == "float64"
                    and payload["tf32"] is False
                    and payload["amp"] is False
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            passed = False
        checks[check_name] = passed
        if not passed:
            errors.append(f"未通过：{check_name}")

    report = {
        "schema": "spike-typeiii-validation-v1",
        "run_id": run_dir.name,
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "measured": {
            "profile": profile,
            "grid": [config["mhd"]["nx"], config["mhd"]["ny"]],
            "steps": config["mhd"]["steps"],
            "dt": config["mhd"]["dt"],
            "snapshots": diagnostics["snapshot_count"],
            "elapsed_s": runtime.get(
                "simulation_elapsed_s",
                runtime["elapsed_s"],
            ),
            "render_elapsed_s": runtime.get("render_elapsed_s"),
            "peak_device_memory_bytes": runtime["peak_device_memory_bytes"],
            "divergence_normalized_rms": diagnostics[
                "divergence_normalized_rms"
            ],
            "energy_budget_max_abs_fraction": budget_value,
            "total_energy_drift_fraction": diagnostics[
                "total_energy_drift_fraction"
            ],
            "final_max_speed": diagnostics["final_max_speed"],
            "event_count": diagnostics["event_count"],
            "minimum_topping_margin_mhz": diagnostics[
                "minimum_topping_margin_mhz"
            ],
            "media": media,
            "auxiliary_reports": auxiliary,
        },
    }
    if write_report:
        _atomic_text(
            run_dir / "data" / "validation_report.json",
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    return report


def _fmt(value: Any, notation: str = ".6g") -> str:
    if value is None:
        return "无事件"
    return format(float(value), notation)


def _readme_text(
    run_dir: Path,
    metadata: dict[str, Any],
    report: dict[str, Any],
) -> str:
    config = metadata["config"]
    radio = config["radio"]
    measured = report["measured"]
    auxiliary = measured["auxiliary_reports"]
    benchmark = auxiliary["cuda_benchmark"]
    convergence = auxiliary["medium_fine_convergence"]
    timestep = auxiliary["timestep_halving"]
    energy = auxiliary["energy_gates"]
    core_convergence_max = max(
        convergence["relative_changes"][name]
        for name in (
            "peak_max_speed",
            "final_flux_difference",
            "peak_reconnection_flux_rate",
            "peak_reconnection_time_fraction",
            "final_island_width_proxy",
        )
    )
    timestep_max = max(
        timestep["relative_changes"][name]
        for name in (
            "peak_max_speed",
            "final_flux_difference",
            "peak_reconnection_flux_rate",
        )
    )
    run_rel = f"spike_typeIII_visual/outputs/runs/{run_dir.name}"
    passed = "通过" if report["passed"] else "未通过"
    return f"""# Spike-Topping Type III 高保真模拟

## 1. 项目目标与模型边界

当前权威运行：`{run_dir.name}`。本项目用二维不可压缩电阻约化 MHD 自洽演化磁岛、电流与双向喷流；电子束、Type III 主脊和 spike 是运动学／现象学代理，不是 PIC、动理学辐射或完整射电传播计算。

## 2. WSL2 与环境启用

在 Windows PowerShell 进入 Ubuntu：

```powershell
wsl -d Ubuntu-24.04
```

在 WSL 中初始化 Miniforge，并按任务选择环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate torch-cuda       # PyTorch CUDA RMHD 与媒体
conda activate solar_simulation # CPU full-MHD 桥接与测试
```

CUDA 只使用 Windows 主机 NVIDIA 驱动。WSL 内不要安装 `nvidia-driver` 或 `cuda-drivers`；只有编译 AthenaK 时才需要 `cuda-toolkit-12-6`、CMake、Ninja、Ubuntu GCC/GFortran/OpenMPI。

## 3. 源码包校验与 WSL 工作目录

Windows 保留 ZIP 和 SHA-256；大型第三方源码只解压到 WSL ext4：

```bash
mkdir -p ~/solarphysics-runtime
cd /mnt/d/solarphysics/simulation
sha256sum -c Spike_Topping_TypeIII_Windows_WSL2_20260728.zip.sha256
rsync -a --exclude outputs spike_typeIII_visual/ ~/solarphysics-runtime/spike_typeIII_visual/
```

## 4. 运行、渲染与验证

```bash
cd ~/solarphysics-runtime
python -m spike_typeIII_visual.main --profile quick --rmhd-engine numpy --device cpu --stage all
python -m spike_typeIII_visual.main --profile cuda-coarse --rmhd-engine torch --device cuda --precision float64 --stage simulate
python -m spike_typeIII_visual.production --seed {config["seed"]} --target scientific-4k --resume
python -m spike_typeIII_visual.reporting validate --run-dir {run_rel}

# Athena C：doctor、构建、smoke、600 s 候选运行、摄取与基准
python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build --mpi --performance --flux hlld --problem spike_topping_solar_jet --jobs 16
ATHENA_BIN="$(find ~/Local/athena/build -path '*/bin/athena' -type f | head -n 1)"
python -m spike_typeIII_visual.athena run --binary "$ATHENA_BIN" --profile jet-smoke --run-id athena_jet_smoke
python -m spike_typeIII_visual.athena run --binary "$ATHENA_BIN" --profile jet-standard --run-id athena_jet_standard
python -m spike_typeIII_visual.athena ingest --run-dir ~/Local/athena/runs/athena_jet_smoke --output ~/Local/athena/runs/athena_jet_smoke/bridge.h5
python -m spike_typeIII_visual.athena benchmark --binary "$ATHENA_BIN" --profile jet-smoke --ranks 1 --repeats 5

# MPI-AMRVAC：doctor、构建、smoke、摄取与基准
python -m spike_typeIII_visual.amrvac doctor
python -m spike_typeIII_visual.amrvac build --jobs 16
AMRVAC_BIN="$(find ~/Local/amrvac/build -path '*/case/amrvac' -type f | head -n 1)"
python -m spike_typeIII_visual.amrvac run --binary "$AMRVAC_BIN" --run-id amrvac_jet_smoke
python -m spike_typeIII_visual.amrvac ingest --run-dir ~/Local/amrvac/runs/amrvac_jet_smoke --output ~/Local/amrvac/runs/amrvac_jet_smoke/bridge.h5
python -m spike_typeIII_visual.amrvac benchmark --binary "$AMRVAC_BIN" --ranks 1 --repeats 5

# AthenaK：doctor、固定提交的 CUDA 构建、运行、摄取清单与基准
python -m spike_typeIII_visual.athenak doctor
python -m spike_typeIII_visual.athenak build
python -m spike_typeIII_visual.athenak run --help
python -m spike_typeIII_visual.athenak ingest --help
python -m spike_typeIII_visual.athenak benchmark --help
```

Athena C、MPI-AMRVAC 与 AthenaK 必须先通过各自 doctor、静态平衡、divB、restart、floor 和守恒预算门槛，才能标记为 production。

## 5. 配置档位

| 档位 | 网格 | dt | 步数 | 快照 | 目的 |
|---|---:|---:|---:|---:|---|
| quick | 48×48 | 0.005 | 80 | 41 | 功能 smoke |
| standard | 96×96 | 0.005 | 400 | 41 | 兼容演示 |
| cuda-coarse | 256×128 | 0.005 | 400 | 401 | CUDA 基线 |
| cuda-medium | 512×256 | 0.0025 | 800 | 401 | 收敛 |
| cuda-fine | 1024×512 | 0.00125 | 1600 | 401 | 正式高分辨率 |
| cuda-medium-event | 512×256 | 0.0025 | 3200 | 401 | 长时事件收敛 |
| cuda-fine-event | 1024×512 | 0.00125 | 6400 | 401 | 长时事件正式运行 |
| cuda-fine-control | 1024×512 | 0.00125 | 6400 | 401 | 零扰动 no-event 控制 |

## 6. 参数调整

- 网格加倍时应按 CFL 同步减小 `dt`；保持终止时间与输出时刻一致后再比较。
- 电流片半宽、扰动幅度、电阻率 `eta`、黏性 `nu` 分别控制片层尺度、撕裂种子和耗散；磁 Prandtl 数为 `Pm=nu/eta`。
- 射电代理参数包括密度标高 `{radio["density_scale_height_mm"]} Mm`、束速 `{radio["beam_speed_fraction_c"]}c`、起始频率 `{radio["start_frequency_mhz"]} MHz`、spike 数 `{radio["spike_count"]}` 和种子 `{config["seed"]}`。
- `scientific-4k` 使用 3840×2160、30 fps、401 个真实快照和全序列固定色标，不做 AI 或光流插帧。
- `production --resume` 会记录失败原因，并按减小时间步、延长演化和预声明参数扫描顺序恢复；不会放宽事件或 topping 门槛。

## 7. 理论推导

取磁通函数和流函数

```text
B = (-dψ/dy, dψ/dx),   v = (-dφ/dy, dφ/dx)
j = -∇²ψ,              ω = ∇²φ
```

泊松括号 `[a,b]=a_x b_y-a_y b_x`。物理 Lorentz 号约定下：

```text
∂tψ + [φ,ψ] = η∇²ψ
∂tω + [φ,ω] = [j,ψ] + ν∇²ω
```

总能量 `E=1/2∫(|B|²+|v|²)dA` 满足

```text
dE/dt = -η∫j²dA - ν∫ω²dA.
```

空间导数用 FFT 伪谱法，`ω_hat=-k²φ_hat` 作 Poisson 反演，非线性项使用 2/3 去混叠，时间推进为 RK4。完整 MHD 后端另解质量、动量、总能量与感应方程，并要求静力平衡 `dp/dy=-ρg`。

射电代理采用指数密度 `n_e(h)=n_0 exp(-h/H)`、电子束高度 `h=v_b t` 与等离子体频率

```text
f_p ≈ 8980 sqrt(n_e) Hz.
```

标准重联率使用 `R_ψ=|d(ψ_O-ψ_X)/dt|`。每个 spike 是时频二维高斯；严格 topping 判据要求其中心频率减去同一时刻主脊频率的 margin 大于 0。零扰动控制必须写入 `no_event`。

## 8. 输出

- `{run_rel}/data/rmhd_fields.h5`：float64、按时间分块的权威 `psi/omega` 与诊断。
- `{run_rel}/data/diagnostics.csv`：能量、X/O 点磁通差、X 点电场、岛宽、耗散与喷流。
- `{run_rel}/data/radio_proxy.npz`：事件动态谱、主脊、门槛活动和固定种子的 spike catalog。
- `{run_rel}/data/run_metadata.json`：参数、软件、运行设备和代理边界。
- `{run_rel}/data/validation_report.json`：逐项验收结果。
- `{run_rel}/figures`：PNG；4K 档同时输出 PDF/SVG。
- `{run_rel}/animations`：`causal_chain`、`reconnection_topology`、`bidirectional_outflow` 和 `radio_event_control` 的 MP4、GIF 与本地 FFV1 母版。
- `{run_rel}/SHA256SUMS.txt`：产物校验和。
- `deliverables/Spike_TypeIII_scientific_complete_{run_dir.name}.zip`：包含源码、event/control float64 数据、媒体、PPT、环境和验证信息的科研完整包；FFV1 可由 HDF5 重建，因此不在传递包中重复存放。

## 9. 本次实测

| 项目 | 实测值 |
|---|---:|
| 网格 / 步数 / dt | {measured["grid"][0]}×{measured["grid"][1]} / {measured["steps"]} / {measured["dt"]} |
| 快照数 | {measured["snapshots"]} |
| 耗时 | {_fmt(measured["elapsed_s"])} s |
| 4K 渲染耗时 | {_fmt(measured["render_elapsed_s"])} s |
| 峰值设备显存 | {_fmt(measured["peak_device_memory_bytes"] / 2**30)} GiB |
| divB 归一化 RMS | {_fmt(measured["divergence_normalized_rms"], ".3e")} |
| 能量预算最大残差 / E0 | {_fmt(measured["energy_budget_max_abs_fraction"], ".3e")} |
| 总能量漂移 | {_fmt(measured["total_energy_drift_fraction"], ".3e")} |
| 最大速度（末快照） | {_fmt(measured["final_max_speed"])} |
| spike 数 / 最小 topping margin | {measured["event_count"]} / {_fmt(measured["minimum_topping_margin_mhz"])} MHz |
| CPU/CUDA 短时 ψ 相对 L2 | {_fmt(benchmark["psi_relative_l2"], ".3e")} |
| medium→fine 核心诊断最大变化 | {_fmt(core_convergence_max, ".3e")} |
| 固定网格时间步减半最大变化 | {_fmt(timestep_max, ".3e")} |
| 理想能量交换残差 / E0 | {_fmt(energy["ideal_exchange_residual_fraction"], ".3e")} |

## 10. 验收与当前限制

本次严格验收：**{passed}**。验证报告是唯一判定依据；未通过的运行不会覆盖本 README。当前权威后端为 `{metadata["mhd_backend"]}`，主要限制是二维不可压缩 RMHD 以及代理射电层；Athena C、MPI-AMRVAC 和 AthenaK 的 development smoke 不属于本次正式验收。只有在 full-MHD 全部门槛通过后，其结果才可替代 RMHD 权威结果。

### 后续模拟推荐

1. 完成网格、时间步、电阻率和黏性的系统收敛扫描。
2. 加入引导场并扩展到可压缩 full-MHD。
3. 在通过验证的 MHD 场上加入测试粒子电子。
4. 使用观测约束密度模型，并逐步加入射电传播和仪器响应。
"""


def update_project_readme(
    run_dir: Path,
    output: Path | None = None,
) -> Path:
    """Update the sole root README only after every formal gate passes."""

    run_dir = Path(run_dir).resolve()
    report = validate_run(run_dir, write_report=True)
    if not report["passed"]:
        raise RuntimeError(
            "Formal validation failed; README was not changed: "
            + "; ".join(report["errors"])
        )
    metadata = json.loads(
        (run_dir / "data" / "run_metadata.json").read_text(encoding="utf-8")
    )
    if output is not None:
        target = Path(output).resolve()
    elif os.environ.get("SPIKE_TYPEIII_README_PATH"):
        target = Path(os.environ["SPIKE_TYPEIII_README_PATH"]).resolve()
    elif Path("/mnt/d/solarphysics/simulation").is_dir():
        target = Path("/mnt/d/solarphysics/simulation/README.md")
    else:
        target = Path(__file__).resolve().parents[1] / "README.md"
    _atomic_text(target, _readme_text(run_dir, metadata, report))
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "readme"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "readme":
            command.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        report = validate_run(args.run_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    path = update_project_readme(args.run_dir, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
