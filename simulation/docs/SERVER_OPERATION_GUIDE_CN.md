# Spike-Topping Type III 服务器全程操作指导书

> 适用范围：GridView 提交界面、Slurm CPU 分区、二维 RMHD 科学生产流程。
> 状态边界：当前服务器任务继续使用提交时的源码和环境。本地新版只用于**下一次**任务；严禁向正在运行的目录覆盖源码、环境或配置。本文不包含账户、IP、主机名或个人目录。

## 1. 模型边界与运行原则

本流程求解二维、不可压缩、电阻性约化 MHD 双 Harris 电流片，并以局部双向重联 outflow 和重联活动联合约束射电 spike。电子束传播、Type III 主脊和 spike 辐射是现象学代理。结果可用于数值方法、条件同期性和收敛研究，但不能表述为 2.5D 真实日冕 jet、粒子自洽加速或 2025-01-24 事件反演。

正式结果必须使用 `float64`。若联合门槛没有候选，`no_event` 是合法科学结果；不得降低 jet 或 reconnection 门槛制造事件。

Python RMHD 是单进程程序。PyTorch/FFT 可以在一个进程内使用多个 CPU 线程，但**不能**把同一条 Python 命令用 MPI 启动多份。禁止：

```bash
# 错误：会产生 64 份互不通信、相互覆盖输出的独立模拟
mpirun -np 64 python -m spike_typeIII_visual.main ...
```

Athena/AMRVAC 后端另有 MPI 工作流；本指导书中的 RMHD 脚本不使用 MPI。

## 2. 当前正在运行任务

只读记录下列私有信息，不写进公开报告：

- job ID、提交时间、分区和资源；
- 提交时的源码 SHA-256 或 Git tree 状态；
- Python、NumPy、PyTorch 和 FFmpeg 版本；
- 输出目录及最近一个合法 checkpoint 的时间和步数。

可以从 GridView 的 **Job**、**Report** 页面以及调度器只读查询：

```bash
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,Partition,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,ExitCode
tail -n 80 "$JOB_LOG"
```

成功后，先用原版本自带 validator 和 `SHA256SUMS.txt` 验证，再下载 `report-lite`。失败且存在合法 checkpoint 时，只能用原服务器源码恢复；没有 checkpoint 时保留日志，待本地新版完整上传后重新开始。当前任务和新版任务必须使用不同目录。

## 3. 上传清单

从本地 `simulation/` 上传：

- `spike_typeIII_visual/` 源码，但排除 `outputs/`、缓存和 checkpoint；
- `server/gridview/` 的 11 个模板；
- 最小环境文件（若项目已有无 `prefix` 的 YAML）；
- `README.md`、本指导书和必要的小型配置。

禁止上传：

- 2.62 GB Windows 原始 ZIP、历史 HDF5/BIN/VTK、4K 视频；
- `Local/`、`.git/`、Conda 环境目录、编译树、缓存；
- 原始事件私有数据、截图、邮件、个人绝对路径；
- `server_env.sh`（它只在服务器上创建）。

上传后在项目目录保存清单：

```bash
find simulation/spike_typeIII_visual simulation/server/gridview -type f -print0 \
  | sort -z | xargs -0 sha256sum > source_upload_SHA256SUMS.txt
```

## 4. 环境建立

在服务器用户空间创建最小环境，不安装整个桌面 SolarPhysics 环境：

```bash
conda create -n solar_simulation -y python=3.12 numpy scipy matplotlib \
  h5py pillow imageio pytest ruff
conda activate solar_simulation
python -m pip install imageio-ffmpeg
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ffmpeg -version
```

如果集群提供经过维护的 PyTorch module，优先遵循管理员文档，不能混用相互冲突的 module 与 Conda CUDA 库。本 CPU 流程不需要 CUDA。

生成个人环境文件：

```bash
cd simulation
python -m spike_typeIII_visual.server render-scripts \
  --output-dir server_scripts
cd server_scripts
cp 00_server_env.example server_env.sh
```

只在服务器编辑 `server_env.sh`：

```bash
export PROJECT_ROOT="${HOME}/spike_typeiii_project"
export RESULT_ROOT="${SCRATCH:-${HOME}/scratch}/spike_typeiii_results"
export CONDA_ENV_NAME="solar_simulation"
export SEED="20260726"
```

`PROJECT_ROOT` 指向包含 `simulation/` 的上传目录；`RESULT_ROOT` 必须是新结果根目录。不要把真实值复制进 README、PPT 或打包文件。

## 5. GridView 字段填写

截图所示界面的字段按下表填写。`core/node` 表示单个 Python 进程可用的 CPU 核数，`node` 固定为 1。

| 阶段 | 分区 | job name 示例 | core/node | node | runtime |
|---|---|---:|---:|---:|---:|
| doctor、NumPy/Torch quick | debug | `st3_qcheck` | 4 | 1 | 1 h |
| 分区基准 | AMD7742、E74809 各一次 | `st3_bench` | 8 | 1 | 1 h |
| medium event | 基准胜出分区 | `st3_medium` | 16 | 1 | 6 h |
| fine event | 基准胜出分区 | `st3_fevent` | 16 | 1 | 12 h |
| fine control | 基准胜出分区 | `st3_fctrl` | 16 | 1 | 12 h |
| 科学检查 | 同上 | `st3_checks` | 8 | 1 | 4 h |
| 4K 渲染 | 同上 | `st3_render` | 16 | 1 | 8 h |
| 验证和打包 | debug | `st3_pack` | 2 | 1 | 1 h |

分区用途：

- `debug`：环境、CLI 和小网格 smoke，不跑正式 fine；
- `AMD7742`、`E74809`：分别做三次相同基准，以 wall time 中位数选分区；
- 不根据处理器名称猜性能。只有实测提升超过 20%，且 quick 数值诊断一致，才固定较快分区。

GridView 的 **script file content** 中粘贴对应编号脚本的全部内容。脚本没有 `#SBATCH` 行，因为资源由网页字段提供。`Work list` 保持站点默认，除非管理员另有规定。

## 6. 严格执行顺序

每一步成功后才进行下一步：

1. `01_doctor.sh`：检查 Python、库、FFmpeg、磁盘和源码入口；
2. `02_quick_numpy.sh`：NumPy 参考 smoke；
3. `03_quick_torch_cpu.sh`：Torch CPU smoke；
4. `04_partition_benchmark.sh`：分别在 AMD7742、E74809 提交，取三次中位数；
5. `05_medium_event.sh`：先确认事件候选与稳定性；
6. `06_fine_event.sh`；
7. `07_fine_control.sh`：相同网格和时长、扰动幅度为零；
8. `08_science_checks.sh`：validator、时间步和能量门槛；
9. `09_render_4k.sh`：只读取 HDF5，逐帧生成图像和视频；
10. `10_validate_and_package.sh`：最终摘要、隐私和校验。

不要并发运行 event 与 control 造成共享磁盘争用；不要让两个任务写入同一目录。4K 渲染不应与正式求解同时执行。

## 7. 参数和可调范围

| profile | 网格 | `dt` | steps | stride | 物理时长 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| quick | 48×48 | 0.005 | 80 | 2 | 0.4 | 接口 smoke |
| rmhd-medium-event | 512×256 | 0.0025 | 3200 | 8 | 8.0 | 中等事件 |
| rmhd-fine-event | 1024×512 | 0.00125 | 6400 | 16 | 8.0 | 正式事件 |
| rmhd-fine-control | 1024×512 | 0.00125 | 6400 | 16 | 8.0 | 无扰动对照 |

共同基线：扰动幅度 event 为 0.04、control 为 0；电阻率和黏性均为 0.002；固定 seed 为 `20260726`；射电网格 1441×1024；spike 数 12；jet 与 reconnection 门槛均 0.6，连续 3 个快照；spike 必须位于起始时间窗并严格高于同期 Type III 主脊。

可直接调整且不改变科学方程：checkpoint 间隔、输出目录、线程数、动画格式、渲染分辨率。修改下列任一项后必须重新做 medium/fine 空间收敛、时间步检查和 event/control 对照：网格、`dt`、steps、snapshot stride、扰动、电阻率、黏性、初值、边界、jet/reconnection 门槛、连续快照数、射电采样、spike 数或时间标定。`float32` 只允许性能探索，不能替代 `float64` 正式结果。

CLI 示例：

```bash
python -m spike_typeIII_visual.main \
  --profile rmhd-medium-event \
  --rmhd-engine torch --device cpu --precision float64 \
  --stage simulate --animation-format none \
  --checkpoint-every 200 --output-dir "$RUN_DIR"
```

历史 `cuda-medium-event` 等名称仍可读取，其数值配置与对应 `rmhd-*` 名称相同；新 CPU 结果使用 `rmhd-*` 以免误解。

## 8. checkpoint 恢复

checkpoint 位于 `RUN_DIR/data/rmhd_checkpoint.npz`，包含当前步、频谱可恢复状态、已保存快照/标量诊断和配置 SHA-256。写入先落到同目录临时文件，再原子替换。恢复时配置、engine 或精度哈希不一致会被拒绝。

确认任务已停止、目录没有其他写进程，再把原命令增加：

```bash
--resume --checkpoint-every 400
```

不要修改 profile、seed、engine、precision、时间步或耗散参数后强行恢复。若 checkpoint 损坏，保留日志并从新目录重跑。

## 9. 故障处理

- **超时/节点中断**：检查最后 checkpoint；用相同源码和配置加 `--resume`。
- **磁盘不足**：停止新渲染，保留 checkpoint 和 HDF5；删除可再生的临时帧前先核对目标，不能删除权威 HDF5。
- **NaN/Inf**：保留日志和最后合法 checkpoint；检查时间步和精度，不降低科学门槛。
- **环境丢失**：重建同版本 Conda 环境，运行 doctor；不要在运行目录原位升级库。
- **任务显示运行但日志不增长**：结合 `sacct` 的 CPU/MaxRSS 和文件时间判断；不要仅因 FFT 长时间无输出就强杀。
- **validator 失败**：按报告定位缺失文件、摘要、尺寸或科学门槛；不手改 JSON 伪造通过。

## 10. 验证、隐私与下载

最终运行：

```bash
python -m spike_typeIII_visual.server validate --run-root "$RUN_DIR"
(cd "$RUN_DIR" && sha256sum -c SHA256SUMS.txt)
```

三档下载：

- `report-lite`：JSON/CSV/NPZ、图、验证报告和摘要；不含 HDF5、原始日志或 checkpoint，优先下载到 Mac；
- `presentation`：在 report-lite 上增加 GIF/MP4/PPT；
- `research-complete`：增加权威 HDF5，体积最大，仅科研归档时下载。

```bash
python -m spike_typeIII_visual.server package \
  --run-root "$RUN_DIR" --tier report-lite \
  --output "$RESULT_ROOT/report-lite_seed20260726.tar.gz"
sha256sum "$RESULT_ROOT/report-lite_seed20260726.tar.gz"
```

打包器会拒绝文本中的个人 home、Windows 用户目录和 IPv4 地址；包内成员清除 uid/gid/用户名，并附 `PACKAGE_SHA256SUMS.txt`。不要下载 Conda 环境、缓存、原始日志或服务器私有配置。

Mac 端只解压 `report-lite`，核对 SHA-256，读取 JSON/CSV/NPZ 和 PNG；不要在 Mac 上重跑 fine 或 4K。需要重新排版时下载 presentation；只有诊断需要逐快照场时才下载 research-complete。

## 11. 结果判读

至少报告：无散诊断、能量预算残差、event/control 差异、medium→fine 核心指标变化、时间步敏感性、jet/reconnection 联合候选、topping 正频偏和 seed。若收敛未达标，写“尚未收敛”；若 `no_event`，写“当前门槛下无联合候选”。服务器 CPU 运行与历史 Windows CUDA 结果允许出现浮点归约级差异，但事件分类、趋势和验收结论必须一致。

## 12. Windows CUDA 历史基线审计

导入档案 SHA-256 为
`985ba6aa614245e99515c953a6c9f38a7c51d1e99d57fd380ce07a556ad0f525`，
ZIP 完整性检查通过。其 `cuda-fine-event` 基线为 1024×512、`dt=0.00125`、
6400 步、401 个快照；历史报告记录 CUDA 求解约 187.95 s、4K 渲染约
2348.02 s、归一化无散误差 \(3.87\times10^{-14}\)、能量预算残差
\(3.14\times10^{-8}\)、12 个严格 topping spikes、jet coincidence 为 1，
最小正频偏约 8.816 MHz。总能量降低约 6.69%，应解释为包含显式耗散的变化，
不能写成理想守恒。

这些数值只用于跨设备回归基线，不能预先当作服务器 CPU 的性能或科学验收
结果。服务器必须重新生成 metadata、validator 报告和 SHA-256；只有分类、
趋势与门槛一致后才可称跨平台可复现。
