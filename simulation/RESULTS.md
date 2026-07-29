# Athena Jet 条件 Spike-Topping Type III：科学基线结果

## 1. 结论先行

- 正式 MHD 后端已由 Python RMHD 提升为 Athena C 二维可压缩电阻—黏性
  full-MHD；RMHD 保留为快速回归和机制对照。
- Athena 正式构建启用了 double、CTU、HLLD、三阶重构、Ohmic
  resistivity、viscosity 和 constrained transport。
- \(512\times256\) standard 主案例通过初态压力平衡、CT 无散、能量、
  空间收敛、无扰动对照、媒体和摘要校验。
- 固定 seed `20260726` 得到 7 个严格 Spike-Topping 事件：
  \(C_{\mathrm{jet}}=1\)，最小高频余量为 \(7.6186\ \mathrm{MHz}\)。
- 这里的结论是“full-MHD jet 条件化的现象学射电事件成立”，不等于 spike
  由 MHD 自洽产生；电子束、射电强度和 spike 仍是后处理代理。
- Apple Silicon Mac 已实测。WSL2、i7-13700H、RTX 4060 和 AthenaK
  均未实测，不写成已实现加速。

正式结果位于
`spike_typeIII_visual/outputs/runs/athena_physical_jet_seed20260726/`。
历史 RMHD 输出没有被覆盖。

## 2. 环境、构建与隐私

| 项目 | 已验证值 |
|---|---|
| 平台类别 | Apple Silicon Mac（Darwin arm64） |
| Python | 3.14.6 |
| NumPy／SciPy | 2.5.1／1.18.0 |
| Matplotlib／h5py | 3.11.1／3.16.0 |
| PyVista／VTK | 0.48.4／9.6.2 |
| ImageIO／imageio-ffmpeg | 2.37.0／0.6.0 |
| FFmpeg | 8.1.2 |
| pytest／Ruff | 9.1.1／0.16.0 |
| Open MPI | 5.0.10 |

可复现环境文件：

- `environment.solar-simulation.yml`：无本机 `prefix` 的最小环境；
- `locks/solar-simulation-osx-arm64.txt`：Apple Silicon 显式锁定；
- `locks/solar-simulation-linux-64.txt`：WSL2/Linux 显式锁定，尚未在目标电脑实测。

Athena 编译树、BIN/VTK/HST、运行日志和基准数据均保存在被 Git 忽略的
`Local/athena/`。公开结果只包含审查后的 HDF5、图、CSV、元数据和媒体。
持久化文件不记录用户名、主机名、设备标识或个人绝对路径。

## 3. 正式 Athena 模型

### 3.1 计算域和初态

二维周期域为

\[
L_x=4\pi,\qquad L_y=2\pi ,
\]

两条 Harris 片层位于 \(y=\pm L_y/4\)。主参数为

| 参数 | 数值 |
|---|---:|
| \(B_0,\rho_0\) | 1, 1 |
| 片层半宽 \(a\) | 0.20 |
| 扰动幅度／宽度 | 0.04／0.45 |
| \(\gamma\) | \(5/3\) |
| \(\beta\) | 1 |
| \(\eta,\nu\) | 0.002, 0.002 |

平衡场与压力为

\[
B_x=B_0\left[
\tanh\frac{y+L_y/4}{a}
-\tanh\frac{y-L_y/4}{a}-1
\right],
\qquad
p=p_{\rm bg}+\frac{B_0^2-B_x^2}{2}.
\]

面心磁场由离散矢势差分写入，因此 constrained transport 初态无散。实测
double 初态全压平衡最大残差为
\(2.220446049250313\times10^{-16}\)，归一化离散
\(\nabla\cdot\boldsymbol B\) 为
\(7.1156186491755\times10^{-16}\)。

### 3.2 重联和 jet 判据

正式重联指标为

\[
R(t)=\left|\frac{\mathrm d}{\mathrm dt}
(\psi_O-\psi_X)\right|,
\]

并用 X 点电场

\[
E_z(X)=-(v_xB_y-v_yB_x)_X+\eta J_z(X)
\]

交叉检查。jet 诊断限定在每个 X 点两侧出流窗口，分别取正、负 \(v_x\)
的 95% 分位，再以两者较小值定义双向出流强度。事件要求
\(q_J\ge0.6\)、\(q_R\ge0.6\)，且连续 3 个输出时刻成立。

## 4. 主案例结果

| 指标 | 实测值 | 验收 |
|---|---:|---|
| 网格 | \(512\times256\) | standard |
| MHD 结束时间 | 2 | 计划值 |
| BIN 快照数 | 41 | 通过 |
| 最大 CT 无散 RMS | \(6.6283\times10^{-14}\) | \(<10^{-12}\) 量级目标的双精度诊断；通过 |
| 总能量漂移 | \(1.5337\times10^{-10}\) | 绝对值 \(<1\%\)；通过 |
| 最终最大速度 | 0.228880 | 记录值 |
| jet onset | 0.851905 | 连续 3 输出条件 |
| \(\max q_R\) | 0.640826 | 达到 0.6 |
| 峰值 \(R\) | 0.00477206 | 局地磁通差指标 |
| 峰值 \(|E_z(X)|\) | 0.0105094 | 独立交叉检查 |
| 事件状态／数量 | `events`／7 | 通过 |
| \(C_{\mathrm{jet}}\) | 1 | 通过 |
| 最小 topping 余量 | 7.61862 MHz | \(>0\)，通过 |

jet 条件窗口从 \([\tau_J,\tau_{\rm end}]\) 线性压缩到
\([0.08,0.75]\ \mathrm{s}\)，只用于 onset 条件射电采样。这是明确标注的
`proxy` 标定，不是由隐藏物理尺度恢复的观测时间。

## 5. 空间收敛

| 网格 | jet onset | 最终最大速度 | 峰值 \(R\) | 最终磁通差 | 峰值 \(|E_z|\) |
|---:|---:|---:|---:|---:|---:|
| \(256\times128\) | 0.854152 | 0.224799 | 0.00600086 | 0.0824410 | 0.0108979 |
| \(512\times256\) | 0.851905 | 0.228880 | 0.00477206 | 0.0812958 | 0.0105094 |
| \(1024\times512\) | 0.851257 | 0.229942 | 0.00446031 | 0.0809949 | 0.0104557 |

峰值 \(R\) 的相邻分辨率相对差异由约 20.5% 降至约 6.5%；其余核心标量
也呈差异递减。\(512\rightarrow1024\) 的目标差异低于 10%，onset 改变量
小于一个输出间隔，因此按预注册标准接受 \(512\times256\) 为科学基线。

## 6. 对照、等价性与可复现性

- 无扰动 \(512\times256\) 对照：无持续 jet/reconnection onset，
  峰值 \(R=0\)，`no_event`。
- serial 与 2-rank MPI smoke：桥接场数组逐项一致，事件分类一致。
- `-O3` reference 与 `-O3 -mcpu=native` performance smoke：场数组逐项一致。
- Roe smoke 仅作求解器敏感性对照，不进入正式结果。
- 同配置 `animation-format none` 重跑：10 个关键场、射电数组和 catalog
  均满足 `np.array_equal`。

这些对照排除了“通过降低阈值制造正结果”和“由媒体编码改变科学数组”两种
解释。无联合候选时，代码仍合法返回 `(0,5)` 的空 catalog。

## 7. Mac MPI 基准

smoke 案例每个 ranks 重复 3 次：

| MPI ranks | 中位 wall time | 中位 cell-updates/s | 峰值内存范围 |
|---:|---:|---:|---:|
| 1 | 1.1487 s | 49,922 | 20.6–21.2 MiB |
| 2 | 1.1646 s | 49,239 | 18.6–18.7 MiB |
| 4 | 1.1551 s | 49,644 | 17.2–17.3 MiB |
| 8 | 1.1958 s | 47,953 | 16.4–16.4 MiB |

没有任何配置比 1 rank 快 20%，因此 Mac 当前不采用 MPI 作为默认正式运行方式。
Athena C 没有暴露分阶段 I/O 计时，报告中的 I/O 占比为 `null` 并附原因，
没有用总 wall time 反推虚假数值。

WSL2 的 1、2、4、6、8、12 ranks 尚未运行；i7-13700H 的推荐 ranks 必须
在目标电脑实测后决定。

## 8. 媒体与完整性

四类 GIF 和 MP4 均已生成：

| 动画 | GIF 帧数 | MP4 帧数 | 尺寸／帧率 |
|---|---:|---:|---|
| tearing | 41 | 41 | 960×540／10 fps |
| jet | 41 | 41 | 960×540／10 fps |
| electron_beam | 40 | 40 | 960×540／10 fps |
| typeIII | 60 | 60 | 960×540／10 fps |

MP4 为 H.264、`yuv420p`。validator、可解码性和
`shasum -a 256 -c SHA256SUMS.txt` 全部通过。权威清单自身 SHA-256 为：

```text
b490c0cd7575d3a3aa3dadb9ae184d59197fdd5f032e1b31f0eccf7aebd34b5f
```

环境与汇报文件的本轮摘要为：

| 文件 | SHA-256 |
|---|---|
| `environment.solar-simulation.yml` | `2e1616b95e2c549b63203ddcf424860d836a3cf82d728ba01a930995df0d3ca7` |
| `locks/solar-simulation-osx-arm64.txt` | `d06fbd46b8558a887a9f778f029cb14f1d28182cfcd72b5c7009d1db8205ac63` |
| `locks/solar-simulation-linux-64.txt` | `28372ee5fdbd5c83638799d83ec1fc725e6b2782e4ac31e177198d288b88e99c` |
| `readme.pdf` | `f4ff9e2c9a6b9014a1c1802ee32529099081dccb42217b886036005bf6781fbf` |
| `Spike_Topping_TypeIII_teacher_report.pptx` | `34d166e55b652c2334ed265ca546bf959cff38c6d58d4ed6f94ade9dbefb3ece` |

## 9. 局限与下一步

- full-MHD 不包含测试粒子、动理学辐射、吸收、散射或真实事件反演。
- \(R\) 和 \(E_z\) 是局地重联诊断，spike 仍由条件化代理生成。
- `proxy` 时间压缩不具有事件定标含义。Alfvén 标定必须显式给出
  \(L_0\)、\(B_0\)、\(n_{e0}\)。
- 先在 WSL2 完成 CPU/MPI 科学等价和性能基准，再迁移 AthenaK。
- RTX 4060 当前只适合作为未来 NVENC 或 AthenaK 路线；Athena C 不宣称
  CUDA 求解器加速。
- AthenaK 只有在 \(256\times128\) 核心诊断与 Athena C 相差低于 5% 后，
  才能生成 GPU 正式结果。

## 10. 本轮范围

本轮没有修改历史根目录结果，没有创建 Git 提交，也没有推送远端。

## 11. v4 2.5D solar-jet 前置 smoke（未通过）

本节是下一阶段的失败前置检查，不改变第 1–10 节 v3 科学基线，也不进入教师
PPT。运行采用绝热 `spike_topping_solar_jet`、\(128\times256\)、120 s
物理时长和公开配置 `configs/athena_2p5d_solar_jet.yaml`。原始 BIN、HST、
restart、日志和 bridge 只保存在忽略的私有运行树。

构建检查通过：

- Athena C double precision；
- CTU、HLLD、三阶重构；
- Ohmic resistivity、viscosity；
- \(N_z=1\) 且保留 \(v_z,B_z\)；
- schema v4 BIN→HDF5 和逐快照读取。

120 s smoke 的关键指标为：

| 指标 | 实测值 | 判定 |
|---|---:|---|
| wall time | 38.379 s | 仅性能记录 |
| bridge 快照数 | 26 | 通过结构检查 |
| 初态 CT 归一化 \(\nabla\cdot B\) | \(2.09\times10^{-16}\) | 通过 |
| 全序列最大 CT 归一化 \(\nabla\cdot B\) | \(1.06\times10^{-14}\) | 通过 \(<10^{-12}\) |
| 质量相对变化 | \(-4.88\%\) | 未通过静态稳定性目标 |
| 总能量相对变化 | \(-3.86\%\) | 未通过完整预算目标 |
| 最小压强 | \(1.0\times10^{-20}\)（code unit） | **触及数值下限，失败** |
| 最大速度 | \(0.1093\)（code unit） | 仅记录 |
| 最大 Mach 数 | 不可信 | 压强下限使声速失真 |

注意：该档位在 \(t=120\) s 才开始底边驱动，因此上述失败首先说明初始分层、
重力离散和开放边界尚未形成足够稳定的静态基线，不能解释为物理 jet。按预注册
流程，本轮停止 600 s static/coarse/standard/fine、Sobol 扫描、非绝热和事件
拟合，未通过调参制造正结果。

下一次运行前必须先完成：

1. well-balanced 重力源与离散静水平衡的一致性测试；
2. 顶边／底边热力学 ghost 状态和波反射单独基准；
3. 关闭驱动、关闭异常电阻的 600 s static 验收；
4. 压强／密度下限触发位置与时间的诊断输出；
5. 质量、能量的边界通量闭合预算。

私有审计摘要（不含路径）：

| 文件角色 | SHA-256 |
|---|---|
| smoke run manifest | `47c9f86399b608ebbe5312e1e4fa51208607101130416c75ecdace2a10a269e9` |
| schema v4 bridge | `57dc7b713a73621c8d7983f1c4e306ebfd6b7ace401836eb267b1dae108f213d` |
| 增量方法 `readme.pdf` | `f7cc32ee742c5ac12e20c39fe6ba41b9364b6f1944f620ef2caaccaad20d94d6` |

## 12. Athena 4.2 权威源码迁移审计

本节只记录源码位置迁移和低成本结构验证，不修改第 1–11 节的科学结果、
事件判定或性能结论。唯一权威 Athena C 源码现为
`fluxrope_demo/athena4.2/`，其定义为 Athena 4.2 基础版加项目最小兼容
补丁。

迁移后摘要：

| 文件角色 | SHA-256 |
|---|---|
| 二维双 Harris 问题 | `99c0a56d937e84978b3301652a747a9218ce2313993ddbdfd9b174c69368565a` |
| 2.5D solar-jet 问题 | `a4a87cbfca3bccf6615ed447f7dfa06f21f6624b8a9689cbf82f551f4739446a` |
| 二维输入 | `e2e357753f86c0ec1f7916be309c498dce6a9290b931a810426c1d6876e7cba2` |
| 2.5D 输入 | `9cdd2c82249fc279cd00d03f342541a14bce7aa3656d1ddf248da91b4ccb2c4a` |
| Athena 4.2 粘性兼容修复 | `846cff956801276675d6c8b89cf0faed2150414b45c45151c314a1f488f9dbb4` |
| 迁移后 `readme.pdf` | `b066e23156775e5429c0530b8a18993761083876c04b2f7885a9eadac7a948d8` |

三个独立构建均通过：fluxrope 使用二阶 HLLD/CTU；两个 Spike-Topping
问题使用 double、三阶 HLLD/CTU，并启用 resistivity 和 viscosity。
三者均执行 \(32\times32\)、\(t=0.001\) 的临时 smoke。fluxrope 产生可读
HST/VTK；两个 Spike-Topping BIN 均成功转为 schema v4 HDF5。初态检查为：

| smoke | 最小密度 | 最小压强 | CT 归一化 \(\nabla\cdot B\) |
|---|---:|---:|---:|
| 双 Harris | 1.0000 | 0.4601 | \(1.60\times10^{-16}\) |
| 2.5D solar-jet | 0.4404 | 0.4404 | \(3.10\times10^{-17}\) |

这些极短运行只证明迁移后的编译、输入和 BIN 桥接链路有效，不替代 v3 正式
结果，也不改变 v4 长时 smoke 未通过的结论。

## 13. Athena 4.2 + MPI-AMRVAC 双后端开发审计

本节只记录 2026-07-28 执行的开发级构建、短时无驱动 smoke、原生格式解析和
公共 bridge 验证。没有运行 600 s，没有进行参数扫描、事件拟合或射电科学
分析；以下数据不得作为 jet 或 Spike-Topping 的新科学结论。

### 13.1 软件链路

| 项目 | Athena 4.2 | MPI-AMRVAC |
|---|---|---|
| 源码策略 | 权威树加项目补丁，隔离构建 | 旧版 vendor 快照只读复制构建 |
| 开发网格 | \(128\times256\) | \(64\times128\)，AMR level 2 |
| 时间 | 0.2 | 0.2 |
| 原生科学格式 | double primitive BIN | double dat-v5 |
| 可视化交叉格式 | 未用于本次验收 | float32 VTU |
| 公共桥接 | schema v5 HDF5 | schema v5 HDF5 |
| wall time | 0.740 s | 2.519 s |
| 状态 | 运行成功 | 运行成功 |

AMRVAC DAT-v5 读取器已通过 header/tree/block、错误版本拒绝、2.5D
\(m_3,B_3\) 压强恢复、`B0field` 总场重建和 AMR coverage 检查。legacy
DAT 与 VTU 仅作为格式 fixture，未进入本次科学模型。

### 13.2 数值质量

| 指标 | Athena 4.2 | MPI-AMRVAC | 门槛与判定 |
|---|---:|---:|---|
| 最小密度 | 0.4262 | 0.4262 | 均为正，通过 |
| 最小压强 | 0.4261 | 0.4262 | 均为正，通过 |
| 最大 Mach 数 | 0.1074 | 0.1204 | \(<10^{-3}\)，两者未通过 |
| 最大归一化 `divB` | \(2.10\times10^{-15}\) | \(2.16\times10^{-2}\) | Athena 通过；AMRVAC 未通过 \(10^{-6}\) |
| 场体积平均能量变化 | \(-5.07\times10^{-5}\) | \(-2.02\times10^{-4}\) | 仅开发记录 |
| restart 最大相对差异量级 | \(2.24\times10^{-5}\) | 未执行 | Athena 未通过 \(10^{-8}\) |

共同初态投影到同一 \(128\times256\) 网格后：

| 场 | 最大相对差异 |
|---|---:|
| \(\rho\) | \(1.53\times10^{-5}\) |
| \(p\) | \(1.15\times10^{-3}\) |
| \(B_x\) | \(2.04\times10^{-2}\) |
| \(B_y\) | \(7.90\times10^{-3}\) |
| \(B_z\) | 0 |

这些差异主要反映 Athena 面心 CT 与 AMRVAC cell-centered 静态背景场的离散
方式不同。由于无驱动静态 Mach 数和 AMRVAC `divB` 未通过，未继续比较演化
逐像素差异，也未启动热传导或长时计算。

### 13.3 测试与摘要

- Ruff：通过；
- pytest：38 项通过，4 个来自 VTK/NumPy 接口的弃用警告；
- Athena 闭式静水平衡与高精度积分：误差 \(<10^{-12}\)；
- AddressSanitizer：定位并修复 `get_eta_user` 外层 ghost 越界；
- Athena CT：通过；
- AMRVAC positivity、DAT-v5 provenance：通过；
- AMRVAC Powell `divB`、双后端静态 Mach、Athena restart：未通过；
- 未覆盖历史输出、历史教师 PPT 或 v3 正式结果。

私有运行只保存在忽略的 `Local/`。公开报告仅记录内容摘要：

| 文件角色 | SHA-256 |
|---|---|
| Athena smoke manifest | `763b3eeb2da7663cf97ef0c8e97d47d09b67c0541927faea271927c1d2ec2d2c` |
| Athena schema-v5 bridge | `e62b5440762f9a526057d0057c69160f552ab2f999f73c086628f7778009a59a` |
| AMRVAC smoke manifest | `903b3685de18630e7d930a29841bedee4a148241c5a5756b0b483cce46e376bd` |
| AMRVAC schema-v5 bridge | `14b187ca3b9b9b858a5ddf38d6857a826642d355fd74fbe1c6283946cb19e8a8` |
| DAT-v5／AMR 读取实现 | `54ce015e19d91d002389dc90ce2903322a41b4df309e392d4863e3f8c3ce21e8` |
| AMRVAC 隔离工作流 | `a8a50852486aef5eeba084d2cff35c641db5d2a132d82f9e001bfad3db6faa88` |
| 双后端统一配置 | `f62ade5abb4b7b7b225e60776e3d4c21c9d9c8cd7d81f35ff5e29f825e9ec76c` |

下一步必须先完成 well-balanced 静态基准、AMRVAC 更严格的无散方案／投影
诊断和可复现 restart，再考虑驱动、热传导、600 s 或 2025-01-24 事件约束。

两个 bridge 均包含 `development_diagnostics`：逐快照最小密度／压强、floor
计数、最大 Mach 数、外向边界质量／能量通量、局部电阻范围、全局 jet 速度
95% 分位数及归一化活动。开放边界通量当前仅是统一分析网格上的诊断量；在加入
源项和求解器原生面通量前，不作为闭合能量预算。

### 13.4 文档与教师汇报交付

本轮同步更新现有 README 并生成 59 页 `readme.pdf`。另建 13 页双后端教师
汇报，历史教师 PPT 保持不变。新汇报中的 Athena v3 图均标注为历史已验证
基线；Athena 4.2 与 AMRVAC 的 0.2 smoke 只标注为开发验证，未产生或更新
射电科学结果。

| 文件角色 | SHA-256 |
|---|---|
| 增量方法 `readme.pdf` | `cb1cbb30d7157ef0a6541c788eb999467f20a41de497daa8fa3b442452c325b8` |
| 双后端教师汇报 | `734477bafd9f95bc48580148d497f851c10bb49dc31efebd69f9fa78b20817d1` |

PDF 已逐页渲染检查；PPT 已渲染全部 13 页并检查裁切、溢出、空占位符、讲者
备注来源和隐藏个人路径。上述文件均未发现用户名、主机名、IP 或个人绝对路径。

## 14. 服务器 CPU 与 WSL2 CUDA 正式 RMHD 对照

本节记录 2026-07-29 完成的二维 RMHD 科学生产结果。两套正式计算使用相同
源码、seed `20260726`、\(1024\times512\) 网格、\(\Delta t=0.00125\)、
6400 步、401 个快照、\(\eta=\nu=0.002\) 和 float64。CPU 使用 Torch CPU，
CUDA 使用 Torch CUDA；CUDA 端关闭 TF32 和 AMP。

### 14.1 科学结果

| 指标 | 服务器 CPU | WSL2 CUDA | 一致性 |
|---|---:|---:|---:|
| 归一化 `divB` RMS | \(3.7366\times10^{-14}\) | \(3.8705\times10^{-14}\) | 绝对差 \(1.34\times10^{-15}\) |
| 能量预算最大绝对残差 | \(3.1364924\times10^{-8}\) | \(3.1364925\times10^{-8}\) | 绝对差 \(3.81\times10^{-16}\) |
| 总能量相对漂移 | \(-0.06686929036960404\) | \(-0.06686929036960398\) | 绝对差 \(5.55\times10^{-17}\) |
| 最终最大速度 | 0.06929671401219835 | 0.06929671401219827 | 绝对差 \(8.33\times10^{-17}\) |
| event 状态／数量 | `events`／12 | `events`／12 | 完全一致 |
| control 状态／数量 | `no_event`／0 | `no_event`／0 | 完全一致 |
| 最小 topping 余量 | 8.8164496888 MHz | 8.8164496888 MHz | 完全一致 |
| jet coincidence | 1.0 | 1.0 | 完全一致 |

CPU 正式结果还通过以下数值门槛：

- medium→fine 核心量最大相对差异 \(1.7066\times10^{-5}\)；
- 时间步减半最大变化 \(7.6522\times10^{-13}\)；
- 401 个快照均为有限值；
- 16/16 正式验证项通过；
- 64 个交付文件由 manifest 完整覆盖；
- 隐私扫描通过。

总能量下降约 6.69%包含显式电阻和黏性耗散，不等同于离散能量预算不闭合；
后者的最大绝对残差为 \(3.14\times10^{-8}\)。

### 14.2 性能

| 阶段 | 服务器 CPU | WSL2 CUDA | CUDA 加速 |
|---|---:|---:|---:|
| 正式求解 | 976.85 s | 187.95 s | \(5.20\times\) |
| 4K 渲染 | 4173.70 s | 2348.02 s | \(1.78\times\) |
| 求解加渲染 | 5150.55 s | 2535.97 s | \(2.03\times\) |

CUDA 的加速对象是 Torch RMHD，不是 Athena 4.2 或 MPI-AMRVAC。渲染包含
Matplotlib、HDF5 读取和视频编码，因此总流程加速低于纯求解加速。

服务器短基准在两个 CPU 分区各重复 3 次：

| 分区 | 三次 wall time | 中位时间 |
|---|---|---:|
| AMD7742 | 4.711、1.682、1.542 s | 1.682 s |
| E74809 | 5.530、3.174、3.018 s | 3.174 s |

AMD7742 的中位时间比 E74809 短约 47.0%，且两分区科学数组保持双精度一致，
因此正式 medium/fine 任务选择 AMD7742。Python RMHD 未使用 MPI 域分解，
不得用 `mpirun` 重复启动相同 Python 任务。

### 14.3 媒体、交付和结论边界

CPU 正式交付包括：

- 13 张 \(3840\times2160\) PNG；
- 4 类 H.264／`yuv420p`／4K／30 fps／401 帧 MP4；
- 4 类 \(960\times540\)／134 帧 GIF；
- 4 类 401 帧 FFV1 无损母版；
- `report-lite`、`presentation` 和 `research-complete` 三档归档；
- SHA-256 校验清单。

正式 CPU/CUDA 对照支持以下结论：在当前二维 RMHD 与严格 jet/reconnection
条件下，12 个起始高频 topping 事件具有数值可复现性，且硬件后端不改变
事件分类。该结论不应扩展为自洽 2.5D 日冕 jet、真实等离子体辐射或
2025-01-24 事件反演。

完整教师汇报见 `Spike_Topping_TypeIII_complete_teacher_report.pptx`，
逐页讲稿见 `Spike_Topping_TypeIII_complete_speaker_script_CN.md`。

另生成简约的山东大学青岛校区实景版
`Spike_Topping_TypeIII_SDU_Qingdao_teacher_report.pptx`。该版本保持 23 页科学
内容和全部结果不变，仅重新设计视觉层级；封面、附录转场和结尾页使用山东大学
（青岛）官网公开校园照片。PPT 已通过逐页渲染、溢出、备注来源和隐私路径检查，
SHA-256 为
`ac872ecf880164663b7c0c1df65b68bc93bbff712e514bf1c5f3705bff1fcec5`。

另生成太阳物理深色主题版
`Spike_Topping_TypeIII_solar_physics_teacher_report.pptx`。该版本继续复用同一套
23 页科学内容、正式 CPU/CUDA 结果与结论边界，视觉采用深空蓝黑、日冕橙红和
EUV 青色；封面、附录转场及结尾页使用 NASA/SDO 的真实 AIA 193 Å 与 131 Å
观测图像。PPT 已通过 23 页逐页渲染、溢出检测、23 个 `[Sources]` 备注块和
隐私路径扫描，SHA-256 为
`cda5fe31abfd9da0987a2faa83fc8725309c4ed384ff6aec38c6ffac14fd76af`。

另生成山东大学青岛与太阳物理融合版
`Spike_Topping_TypeIII_SDU_Qingdao_solar_physics_teacher_report.pptx`。该版本
保持同一套 23 页科学内容、CPU/CUDA 正式指标和结论边界，以山东大学红标识学校
身份，以日冕橙与 EUV 青标识太阳物理主题；封面、附录转场与结尾同时使用山东大学
青岛校区官网实景照片和 NASA/SDO 真实观测图像。PPT 已通过两套渲染、溢出检测、
23 个 `[Sources]` 备注块及隐私路径扫描，SHA-256 为
`1fa3973de396d2acfb39a800be1f435c48ec46927dfee96f1632dfc63cc331eb`。
