# Spike-Topping Type III Windows 交付包摘要

## 交付目标

本包用于在 Windows 11 的 WSL2 Ubuntu 环境继续开发“伴随全局日冕 jet 的
Spike-Topping Type III”前向模拟。项目采用互补双后端：

- Athena 4.2：固定网格、HLLD/CTU/CT，作为参考求解器；
- MPI-AMRVAC：AMR、2.5D 与后续非绝热扩展，作为独立交叉验证后端。

射电部分仍是由重联活动与全局 jet 活动联合约束的现象学代理。无联合候选时
必须输出 `no_event`，不得自动降低阈值。

## 已实现

- 统一 2.5D solar-jet 参数与几何诊断；
- Athena 闭式静水平衡、异常电阻 `CASE=2`、底边矢势延拓；
- Athena 2.5D 各向异性热传导的 \(B_z^2\) 投影修复；
- MPI-AMRVAC 隔离复制构建和独立 solar-jet case；
- dat-v5 header/tree/block 读取、三分量压强恢复和 `B0field` 重建；
- AMR 到统一分析网格的体积守恒投影；
- schema v5 HDF5、公共 `--mhd-backend` / `--mhd-dataset` 接口；
- 开发诊断：positivity、floor、Mach、边界通量、局部电阻、全局 jet；
- 38 项 Python 测试和双后端 0.2 开发 smoke。

## 尚未通过

- 无驱动静态最大 Mach 数未达到 \(10^{-3}\)；
- MPI-AMRVAC Powell 归一化 `divB` 未达到 \(10^{-6}\)；
- Athena restart 差异未达到 \(10^{-8}\)；
- 尚未执行 WSL2 MPI 性能实测、热传导正式案例或 600 s 长时模拟。

因此本包不包含新的科学结论，也不把开发 smoke 当作真实 jet 或事件反演。

## 包内目录

```text
Spike_Topping_TypeIII_Windows_WSL2_20260728/
├── README_FIRST.md
├── PACKAGE_SUMMARY.md
├── SHA256SUMS.txt
├── windows/
│   ├── setup_environment.sh
│   ├── verify_bundle.sh
│   └── run_development_smokes.sh
├── simulation/
│   ├── README.md
│   ├── readme.pdf
│   ├── RESULTS.md
│   ├── environment.solar-simulation.yml
│   ├── locks/
│   ├── configs/
│   ├── spike_typeIII_visual/
│   ├── fluxrope_demo/athena4.2/
│   └── amrvac/
└── docs/
    └── Spike_Topping_TypeIII_dual_backend_teacher_report.pptx
```

## 有意排除

- `Local/` 中的 Mac 构建、BIN、DAT、HDF5、restart 和日志；
- 历史模拟 `outputs/`；
- `.git`、缓存、对象文件和可执行文件；
- 原始 2025-01-24 观测数据；
- 用户名、主机名、IP、邮箱和个人绝对路径；
- Mac Conda 环境目录。

## 建议起点

解压到 WSL2 Linux 文件系统后：

```bash
bash windows/verify_bundle.sh
bash windows/setup_environment.sh
conda activate solar_simulation
JOBS=8 RANKS=1 bash windows/run_development_smokes.sh
```

先复现开发 smoke，再修复静态平衡、AMRVAC `divB` 和 restart。通过这些门槛
后，才进入 MPI 基准、驱动 jet、热传导和 600 s 阶段。
