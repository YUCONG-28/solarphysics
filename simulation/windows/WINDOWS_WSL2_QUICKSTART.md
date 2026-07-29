# Windows / WSL2 运行指南

本包面向 Windows 11 + WSL2 Ubuntu。Athena 4.2 与当前 MPI-AMRVAC 都是
Linux/Unix 编译工作流，因此不建议直接在原生 PowerShell 中编译。RTX 4060
本阶段不参与 MHD 求解，只可用于后续视频编码。

## 1. Windows 侧准备

以管理员身份打开 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

重启后打开 Ubuntu，完成 Linux 用户初始化。建议把压缩包解压到 WSL2 的
Linux 文件系统，例如 `~/solarphysics/`，不要长期放在 `/mnt/c/` 下运行，
否则大量小文件编译和 HDF5 I/O 会明显变慢。

## 2. WSL2 编译依赖

在 Ubuntu 终端执行：

```bash
sudo apt update
sudo apt install -y \
  build-essential gfortran make perl pkg-config \
  openmpi-bin libopenmpi-dev \
  curl ca-certificates unzip
```

安装 Linux x86-64 版 Miniforge 后重新打开终端，确认 `conda` 或 `mamba`
可用。不要把 Mac 的 Conda 环境目录复制到 Windows。

## 3. 创建最小 Python 环境

进入解压后的包根目录：

```bash
cd ~/solarphysics/Spike_Topping_TypeIII_Windows_WSL2_20260728
bash windows/setup_environment.sh
conda activate solar_simulation
```

脚本默认使用 `simulation/environment.solar-simulation.yml`。若需要严格复现
2026-07-28 的 Linux 包版本，可改用：

```bash
conda create -n solar_simulation \
  --file simulation/locks/solar-simulation-linux-64.txt
```

显式锁定清单依赖在线 conda-forge 包仍可访问；一般迁移优先使用 YAML。

## 4. 校验压缩包与 Python 层

```bash
bash windows/verify_bundle.sh
conda activate solar_simulation
cd simulation
python -m ruff check spike_typeIII_visual
python -m pytest spike_typeIII_visual/tests -q
```

Mac 开发机基线为 38 项 pytest 通过。VTK/NumPy 的弃用警告不等于测试失败。

## 5. 双后端开发 smoke

默认使用 8 个编译任务、单进程求解：

```bash
cd ~/solarphysics/Spike_Topping_TypeIII_Windows_WSL2_20260728
conda activate solar_simulation
JOBS=8 RANKS=1 bash windows/run_development_smokes.sh
```

脚本将：

1. 构建 Athena 4.2 solar-jet reference 版本；
2. 运行 \(128\times256,\ t_{\rm end}=0.2\) 无驱动 smoke；
3. 生成 Athena schema-v5 bridge；
4. 隔离复制并构建 MPI-AMRVAC；
5. 运行基础 \(64\times128\)、最高 AMR level 2、\(t_{\rm end}=0.2\) smoke；
6. 生成 AMRVAC schema-v5 bridge。

所有构建、原始 DAT/BIN、日志和 HDF5 都写到包根目录的 `Local/`，不会写入
源码目录。

## 6. i7-13700H / 32 GB 建议

- 编译：`JOBS=8`；若系统响应正常可试 `JOBS=12`。
- 首次 smoke：`RANKS=1`，先排除 MPI 配置问题。
- MPI 基准：依次测试 1、2、4、6、8、12 ranks，每档重复 3 次。
- 固定 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
- 同时运行的任务数乘以 ranks 不要超过可用物理核心。
- 600 s、standard/fine 和热传导计算当前禁止直接启动；必须先修复静态平衡、
  AMRVAC `divB` 和 restart 验收。

## 7. 当前可信状态

本包只完成开发级链路验证：

- Athena 4.2 和 MPI-AMRVAC 均可构建、短时运行和输出 schema v5；
- Athena CT 无散检查通过；
- 无驱动静态 Mach、AMRVAC Powell `divB` 和 Athena restart 尚未通过门槛；
- 没有新的 600 s 科学结果，也没有新的事件反演结论。

完整理论、公式、方法和限制见 `simulation/README.md`，实测审计见
`simulation/RESULTS.md`。
