# 伴随重联喷流的 Spike-Topping Type III 数值模拟

## 理论推导、研究思路、源码映射、动画导出

> - 主题：伴随重联喷流（reconnection jet）发生的 Spike-Topping Type III
> - 已验证基线：Athena C 二维可压缩电阻—黏性 full-MHD + jet 条件射电代理
> - 下一阶段模型：Athena 4.2 + MPI-AMRVAC 双后端 2.5D 分层开放场日冕 jet
> - 正式 RMHD 结果：Python 二维不可压缩电阻 RMHD 已完成服务器 CPU 与
>   WSL2 CUDA 双精度对照，用于机制研究、严格 event/control 和性能验证
> - 实现状态：v3 双 Harris、严格 topping 和 Mac 科学基线已验证；v5
>   双后端接口、DAT-v5 读取、三分量桥接及无驱动开发 smoke 已完成，但
>   静态 Mach 数、AMRVAC `divB` 和 restart 等物理门槛未通过
> - 平台边界：Apple Silicon Mac、服务器 x86_64 CPU 与 WSL2 RTX 4060
>   已实测；AthenaK GPU 仍未实测
> - 本轮数值结果、对照实验和摘要见 [`RESULTS.md`](RESULTS.md)

---

## 目录

- [1–3：研究问题、模型层级与归一化](#1-目的)
- [4A：Athena full-MHD 方程与离散构造](#4a-athena-正式-full-mhd-方程与离散构造)
- [4–9：RMHD 回归、数值方法、jet 与射电代理](#4-rmhd-回归模型的详细推导)
- [10–14：代码映射、运行、验证、性能与环境](#10-源码结构与公式映射)
- [15–17：局限、参考资料与既有过程记录](#15-局限与下一步研究)
- [18：面向 2025-01-24 事件的 2.5D 日冕 Jet](#18-下一阶段面向-2025-01-24-事件的-25d-日冕-jet)
- [19：Athena 4.2 + MPI-AMRVAC 双后端增强](#19-athena-42--mpi-amrvac-双后端增强)
- [20：服务器 RMHD 科学生产工作流](#20-服务器-rmhd-科学生产工作流)
- [21：Git 版本控制与本地数据边界](#21-git-版本控制与本地数据边界)

---

## 1. 目的

1. 给出“磁重联—双向喷流—电子注入—Type III—高频 spike”的研究思路；
2. 从基本方程开始，逐步推导当前数值模型所用公式；
3. 将公式映射到 Athena C 与 Python 源码，区分正式求解、回归模型和射电代理；
4. 给出不依赖完整 `solarphysics_env_latest` 的 `solar_simulation` 环境与双平台路线。

**本文参考陈耀《等离子体物理学基础》（科学出版社，2019，
ISBN 978-7-03-061388-2）**

---

## 2. 核心科学问题与研究思路

### 2.1 核心问题

> 当二维电流片发生撕裂和磁重联、并形成双向喷流时，如何构造一个可检验的Spike-Topping Type III 动态谱代理，并定量判断 spike 是否与 jet 同期？

“Spike-Topping Type III”须满足

$$
t_k\in\mathcal{W}_{\mathrm{onset}},
\qquad
\Delta f_k
=f_k-f_{\mathrm{III}}(t_k)>0 .
\tag{2.1}
$$

式中，\(t_k\) 为第 \(k\) 个 spike 的中心时刻，\(f_k\) 为其中心频率，\(f_{\mathrm{III}}(t_k)\) 为同一时刻 Type III 主脊频率，\(\mathcal{W}_{\mathrm{onset}}\) 为起始时间窗。时间单位为秒，频率单位为 MHz。

### 2.2 物理机制

```text
                  物理机制

周期双 Harris 电流片
        │
        ▼
撕裂模与磁岛形成
        │
        ▼
电流增强和磁重联活动代理
        │
        ├──────────────┐
        ▼              ▼
双向重联喷流       电子注入活动代理
        │              │
        └──────┬───────┘
               ▼
          电子束向外传播
               │
               ▼
      指数密度 → 等离子体频率
               │
               ▼
          Type III 负频漂主脊
               │
               ▼
   起始窗内、主脊高频侧的窄带 spikes
```

其中二维 full-MHD 是正式自洽时间演化，Python RMHD 是回归对照；电子束、
日冕密度、射电强度和 spike 均为后处理代理。它们用于提出和检验机制假设，
不等同于测试粒子、PIC、波粒相互作用或射电辐射转移计算。

上图中的“双向重联喷流”由 Athena 速度场直接诊断，但“jet 触发电子注入并
产生 spike”的联系仍属于代理。源码现已在 X 点两侧出流窗口计算正、负
\(v_x\) 分位数、连续 onset 和同期统计。该诊断不能替代完整三维喷流分割；
本轮主案例结果以 [`RESULTS.md`](RESULTS.md) 为准。

### 2.3 当前基线与目标模型

必须区分以下两层：

| 层级 | 当前源码状态 | 本课题中的作用 |
|---|---|---|
| Athena C full-MHD、CT 双 Harris | **正式后端，已实现** | 提供密度、压力、速度、磁场和能量的自洽演化 |
| Python RMHD／FFT／RK4 | 已保留 | 快速回归和机制对照，不作为新正式结果 |
| \(R=\left|\mathrm d(\psi_O-\psi_X)/\mathrm dt\right|\) | **已实现** | 正式局地重联指标 |
| \(E_z(X)=-(v_xB_y-v_yB_x)_X+\eta J_z(X)\) | **已实现** | 对 \(R\) 的独立交叉检查 |
| X 点两侧正、负 \(v_x\) 分位数 | **已实现** | 给出双向 jet 活动 \(q_J\) 和 onset |
| MHDFieldSeries 与 BIN→HDF5 | **已实现** | 统一 Athena／RMHD 后处理接口 |
| spike 起始窗、严格正频偏与 jet 条件抽样 | **已实现** | 保证 topping 几何定义和同期性 |
| schema v3 validator、GIF／MP4 | **已实现** | 动态产物、摘要、隐私和媒体验收 |
| 2.5D 三分量 BIN→schema v4 HDF5 | **已实现并通过单元测试** | 保留 \(v_z,B_z,\boldsymbol J,\boldsymbol\omega\)，v3 读取时零填充 |
| `EventBundle` 与 `event` 时间标定 | **已实现并通过单元测试** | 只携带审查后的 UTC、频段、ROI 和逻辑数据 ID |
| `spike_topping_solar_jet` 绝热基线 | **已实现；仅构建／初始化验证** | 分层、开放场、导引场、底边驱动、重力和异常电阻 |
| 场向热传导／辐射／背景加热 | **计划中** | 必须分级验收后才能用于正式 v4 结果 |
| 600 s v4 coarse／standard／fine | **尚未运行** | 不得引用为结果或更新教师 PPT |
| Athena C MPI | Mac 小案例已实测 | 无超过 20% 加速，暂不作为默认 |
| WSL2／RTX 4060／AthenaK | **尚未实测** | CPU MPI 与 GPU 迁移的后续路线 |

因此，当前源码可以分别检验“严格高频 topping”和“spike 是否伴随 jet”。
若预设阈值下没有候选时段，程序返回可复现的零事件结果，而不放宽条件。正式
研究命令必须显式选择 `--mhd-backend athena`；旧脚本的 CLI 默认仍为
`rmhd`，以保持兼容。

### 2.4 可证伪假设

- **H1：重联—喷流假设。** 电流片局地电流增强时，片层附近出现明显的双向 \(v_x\) 分量。
- **H2：喷流—注入假设。** 电子注入概率随重联活动和 jet 活动的共同增强而增大。
- **H3：Type III 映射假设。** 向低密度区传播的电子束产生严格下降的 \(f_{\mathrm{III}}(t)\)。
- **H4：topping 假设。** 起始窗内的 spike 满足 \(f_k>f_{\mathrm{III}}(t_k)\)。
- **H5：伴随性假设。** spike 时刻的 jet 活动和重联活动同时超过预先定义的阈值。

H1、H3、H4 和 H5 现均由数组、目录量和校验器直接检查。H2 仍是现象学映射
假设：代码检验“条件是否满足”，不把相关性解释为微观因果。

---

## 3. 模型层级、符号与归一化

### 3.1 模型边界

正式模型包含：

- 二维、可压缩、周期边界的电阻—黏性 full-MHD；
- 离散矢势构造的双 Harris 面心磁场和确定性撕裂扰动；
- Athena C 的 CTU、HLLD、三阶重构和 constrained transport；
- BIN 科学数据、VTK 可视化交叉检查和分块 LZF HDF5 桥接；
- O/X 点磁通差变化率、X 点电场和双向出流窗口诊断；
- 匀速电子束、指数日冕密度和基频等离子体频率映射；
- Type III-like 高斯主脊及起始窗内的二维高斯 spikes；
- 13 张静态图和可选 GIF／MP4 动画。

Python RMHD 仍包含 Fourier 伪谱、严格 \(2/3\) 去混叠和固定步长 RK4，
用于快速回归；第 4–6 节保留其详细推导，不能与 Athena 正式离散混为一谈。

当前模型不包含：

- 三维几何、可压缩效应、热传导、重力或完整能量方程；
- 测试粒子轨道、碰撞、朗道增长、束流—等离子体波耦合；
- 基频／谐波辐射效率、吸收、散射和传播；
- 由 full-MHD 自动涌现的 spike；
- 观测仪器响应和绝对辐射通量标定。

### 3.2 主要符号

| 符号 | 含义 | 单位或状态 |
|---|---|---|
| \(\rho,p,E\) | Athena 质量密度、气体压强和总能量密度 | full-MHD 无量纲 |
| \(\boldsymbol{B}\) | 磁场 | MHD 无量纲 |
| \(\boldsymbol{v}\) | 速度 | MHD 无量纲 |
| \(\psi\) | 面内磁通函数／O-X 点磁通诊断 | MHD 无量纲 |
| \(J_z\) | 面外电流密度 \(\partial_xB_y-\partial_yB_x\) | MHD 无量纲 |
| \(\phi\) | RMHD 回归模型的流函数 | RMHD 无量纲 |
| \(j\) | RMHD 源码电流变量，\(j=-\nabla^2\psi\) | RMHD 无量纲 |
| \(\omega\) | RMHD 垂直涡度，\(\omega=\nabla^2\phi\) | RMHD 无量纲 |
| \(\eta\) | 归一化 Ohmic 电阻系数 | MHD 无量纲 |
| \(\nu\) | 归一化黏性系数 | MHD 无量纲 |
| \([a,b]\) | Poisson 括号 | 由 \(a,b\) 决定 |
| \(\tau\) | MHD 求解器时间 | 无量纲 |
| \(t\) | 射电代理时间 | s |
| \(v_b=\beta c\) | 电子束速度 | m s\(^{-1}\) |
| \(h\) | 电子束高度 | Mm |
| \(H\) | 指数密度标高 | Mm |
| \(n_e\) | 电子数密度 | cm\(^{-3}\) |
| \(f_{\mathrm{pe}}\) | 电子等离子体频率 | Hz |
| \(f_{\mathrm{III}}\) | Type III 主脊频率 | MHz |
| \(I(f,t)\) | 合成动态谱强度 | 归一化 |

本文用粗体表示矢量、普通斜体表示标量、正体下标表示物理标签。例如\(f_{\mathrm{pe}}\) 中的 \(\mathrm{pe}\) 表示 plasma electron。

### 3.3 无量纲化

设特征长度、磁场、密度和 Alfvén 速度分别为
\(L_0\)、\(B_0\)、\(\rho_0\) 和

$$
v_{\mathrm{A}0}
=\frac{B_0}{\sqrt{\mu_0\rho_0}} .
\tag{3.1}
$$

Alfvén 时间为

$$
t_{\mathrm{A}0}=\frac{L_0}{v_{\mathrm{A}0}} .
\tag{3.2}
$$

以下用星号表示有量纲量，无星号表示源码使用的无量纲量：

$$
\boldsymbol{x}
=\frac{\boldsymbol{x}^{*}}{L_0},
\qquad
\tau=\frac{t^{*}}{t_{\mathrm{A}0}},
\qquad
\boldsymbol{v}
=\frac{\boldsymbol{v}^{*}}{v_{\mathrm{A}0}},
\qquad
\boldsymbol{B}
=\frac{\boldsymbol{B}^{*}}{B_0}.
\tag{3.3}
$$

将这些定义代入有量纲 MHD 方程后，无星号量满足无量纲方程。因而`max_speed=0.2` 不能直接解释为 \(0.2c\)；只有给出
\(L_0\)、\(B_0\) 和 \(\rho_0\) 后，RMHD 时间、速度和能量才可转换为物理量。

射电后处理另用秒、Mm、MHz 和 cm\(^{-3}\)。RMHD 时间 \(\tau\) 与射电时间\(t\) 之间目前采用线性重采样，它不是由量纲恢复自动得到的物理映射。

---

## 4A. Athena 正式 full-MHD 方程与离散构造

本节是正式科学后端；后续第 4–6 节是保留的 RMHD 回归模型。

### 4A.1 从守恒律到 Athena 演化变量

忽略重力、热传导和显式辐射损失，采用 \(\mu_0=1\) 的归一化。质量守恒为

$$
\frac{\partial\rho}{\partial\tau}
+\boldsymbol{\nabla}\cdot(\rho\boldsymbol{v})=0 .
\tag{4A.1}
$$

动量守恒为

$$
\frac{\partial(\rho\boldsymbol{v})}{\partial\tau}
+\boldsymbol{\nabla}\cdot
\left[
\rho\boldsymbol{v}\boldsymbol{v}
+\left(p+\frac{B^2}{2}\right)\boldsymbol{I}
-\boldsymbol{B}\boldsymbol{B}
\right]
=\boldsymbol{\nabla}\cdot\boldsymbol{\Pi}.
\tag{4A.2}
$$

式中，\(\rho\boldsymbol{v}\boldsymbol{v}\) 和
\(\boldsymbol{B}\boldsymbol{B}\) 是并矢，\(\boldsymbol I\) 是单位张量，
\(\boldsymbol\Pi\) 是黏性应力张量。源码启用 Athena 的 isotropic viscosity，
归一化运动黏度为 \(\nu=0.002\)。

总能量密度由内能、动能和磁能相加：

$$
E
=\frac{p}{\gamma-1}
+\frac{\rho v^2}{2}
+\frac{B^2}{2}.
\tag{4A.3}
$$

由式 (4A.3) 可反解压强：

$$
p
=(\gamma-1)
\left(
E-\frac{\rho v^2}{2}-\frac{B^2}{2}
\right).
\tag{4A.4}
$$

因此 problem generator 写入 primitive 变量后，Athena 将其转换为守恒变量。
正压验收直接检查式 (4A.4) 的结果，而不是只检查输入常数。

理想部分的总能量通量为

$$
\boldsymbol{F}_E
=\left(E+p+\frac{B^2}{2}\right)\boldsymbol v
-(\boldsymbol v\cdot\boldsymbol B)\boldsymbol B .
\tag{4A.5}
$$

Ohmic 与黏性模块在此基础上加入耗散通量。正式构建取
\(\gamma=5/3\)、\(\eta=\nu=0.002\)；若 `athena -c` 没有显示
resistivity 和 viscosity 为 `ON`，该二进制不得用于正式结果。

### 4A.2 感应方程与 constrained transport

欧姆定律写为

$$
\boldsymbol E
=-\boldsymbol v\times\boldsymbol B
+\eta\boldsymbol J,
\qquad
\boldsymbol J=\boldsymbol\nabla\times\boldsymbol B.
\tag{4A.6}
$$

代入 Faraday 定律
\(\partial_\tau\boldsymbol B=-\boldsymbol\nabla\times\boldsymbol E\)，得到

$$
\frac{\partial\boldsymbol B}{\partial\tau}
=\boldsymbol\nabla\times
\left(
\boldsymbol v\times\boldsymbol B-\eta\boldsymbol J
\right),
\qquad
\boldsymbol\nabla\cdot\boldsymbol B=0 .
\tag{4A.7}
$$

二维情况下取离散矢势 \(A_z\) 放在网格角点。面心磁场由相邻角点差分：

$$
B_{x,i+\frac12,j}
=
\frac{
A_{z,i+\frac12,j+\frac12}
-A_{z,i+\frac12,j-\frac12}
}{\Delta y},
\tag{4A.8}
$$

$$
B_{y,i,j+\frac12}
=-
\frac{
A_{z,i+\frac12,j+\frac12}
-A_{z,i-\frac12,j+\frac12}
}{\Delta x}.
\tag{4A.9}
$$

将式 (4A.8)–(4A.9) 代入单元中心离散散度，

$$
(\boldsymbol\nabla_h\cdot\boldsymbol B)_{i,j}
=
\frac{B_{x,i+\frac12,j}-B_{x,i-\frac12,j}}{\Delta x}
+
\frac{B_{y,i,j+\frac12}-B_{y,i,j-\frac12}}{\Delta y},
\tag{4A.10}
$$

每个 \(A_z\) 项以相反符号成对抵消，故初态离散散度只剩浮点舍入误差。
Athena 的 constrained transport 随后用边中心电场更新面心磁场，使这一拓扑
关系在时间推进中保持。

### 4A.3 周期双 Harris 平衡的推导

目标平衡磁场为

$$
B_x(y)=B_0\left[
\tanh\frac{y+L_y/4}{a}
-\tanh\frac{y-L_y/4}{a}-1
\right].
\tag{4A.11}
$$

对一维静态平衡，式 (4A.2) 的 \(y\) 分量化为

$$
\frac{\mathrm d}{\mathrm dy}
\left(p+\frac{B_x^2}{2}\right)=0.
\tag{4A.12}
$$

积分得到总压常数

$$
p+\frac{B_x^2}{2}
=p_{\rm bg}+\frac{B_0^2}{2},
\tag{4A.13}
$$

从而

$$
p(y)
=p_{\rm bg}
+\frac{B_0^2-B_x^2(y)}{2}.
\tag{4A.14}
$$

这解释了为什么压力不能设为全域常数：若 \(B_x\) 在片层内减小，气体压强必须
相应升高，才能抵消磁压梯度。正式 problem generator 先在角点构造与式
(4A.11) 一致的周期 \(A_z\)，再叠加幅度 0.04、宽度 0.45 的确定性撕裂扰动，
最后由式 (4A.8)–(4A.9) 写入面心场。

### 4A.4 CTU、HLLD 与三阶重构的角色

一个 Athena 时间步可概括为：

1. 从单元中心 primitive 变量重构左右界面状态；
2. 用三阶空间重构降低平滑区截断误差；
3. 在每个界面用 HLLD 近似 Riemann 解计算 MHD 数值通量；
4. CTU 预测横向通量耦合，得到时间中心化界面通量；
5. 用守恒通量更新 \(\rho,\rho\boldsymbol v,E\)；
6. 用边中心电场执行 constrained transport 更新面心磁场；
7. 加入显式 Ohmic 和 isotropic viscosity 耗散。

HLLD 是正式基线；Roe 只作为数值敏感性 smoke 对照。MPI、SMR 和 STS
在首个科学基线中关闭，以先隔离物理与离散正确性。

### 4A.5 正式重联率与 X 点电场

由二维面内磁场求磁通函数时，使用

$$
B_x=-\frac{\partial\psi}{\partial y},
\qquad
B_y=\frac{\partial\psi}{\partial x},
\tag{4A.15}
$$

因此

$$
\nabla^2\psi
=\frac{\partial B_y}{\partial x}
-\frac{\partial B_x}{\partial y}
=J_z .
\tag{4A.16}
$$

BIN 适配器在真实网格上反演 \(\psi\)，识别 O/X 点，并计算

$$
\Delta\psi_{OX}(\tau)=\psi_O(\tau)-\psi_X(\tau),
\tag{4A.17}
$$

$$
R(\tau)
=\left|
\frac{\mathrm d\Delta\psi_{OX}}{\mathrm d\tau}
\right|.
\tag{4A.18}
$$

再由式 (4A.6) 的 \(z\) 分量得到

$$
E_z(X)
=-(v_xB_y-v_yB_x)_X+\eta J_z(X).
\tag{4A.19}
$$

式 (4A.18) 是正式活动曲线，式 (4A.19) 是独立交叉检查。旧的
\(\eta\max|j|\) 只保留在 RMHD 历史方法中，不再作为 Athena 正式重联率。

---

## 4. Python RMHD 回归模型：方程的逐步推导

### 4.1 从不可压缩电阻 MHD 出发

忽略重力和热力学演化，归一化不可压缩电阻 MHD 可写为

$$
\nabla\cdot\boldsymbol{v}=0,
\qquad
\nabla\cdot\boldsymbol{B}=0 ,
\tag{4.1}
$$

$$
\frac{\partial\boldsymbol{B}}{\partial\tau}
=\nabla\times(\boldsymbol{v}\times\boldsymbol{B})
+\eta\nabla^2\boldsymbol{B},
\tag{4.2}
$$

$$
\frac{\partial\boldsymbol{v}}{\partial\tau}
+(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}
=-\nabla P
+\boldsymbol{F}_{\mathrm{L}}
+\nu\nabla^2\boldsymbol{v}.
\tag{4.3}
$$

式中，\(P\) 为包含流体压强和可并入梯度项的总压强，\(\boldsymbol{F}_{\mathrm{L}}\) 为归一化 Lorentz 力。二维模型假定\(\partial/\partial z=0\)，并只演化 \(x\)-\(y\) 平面内的磁场和速度。

### 4.2 用势函数自动满足无散条件

定义

$$
\boldsymbol{B}
=\hat{\boldsymbol{z}}\times\nabla\psi
=\left(-\frac{\partial\psi}{\partial y},
\frac{\partial\psi}{\partial x},0\right),
\tag{4.4}
$$

$$
\boldsymbol{v}
=\hat{\boldsymbol{z}}\times\nabla\phi
=\left(-\frac{\partial\phi}{\partial y},
\frac{\partial\phi}{\partial x},0\right).
\tag{4.5}
$$

对磁场求散度：

$$
\begin{aligned}
\nabla\cdot\boldsymbol{B}
&=\frac{\partial}{\partial x}
\left(-\frac{\partial\psi}{\partial y}\right)
+\frac{\partial}{\partial y}
\left(\frac{\partial\psi}{\partial x}\right) \\
&=-\frac{\partial^2\psi}{\partial x\partial y}
+\frac{\partial^2\psi}{\partial y\partial x}
=0 .
\end{aligned}
\tag{4.6}
$$

同理，

$$
\nabla\cdot\boldsymbol{v}=0 .
\tag{4.7}
$$

只要 \(\psi\) 和 \(\phi\) 足够光滑，混合偏导可交换，式 (4.6) 和式 (4.7)
恒成立。这就是势函数表示能在离散误差范围内保持无散约束的原因。

### 4.3 涡度和源码电流变量

速度的垂直旋度为

$$
\begin{aligned}
\omega
&=(\nabla\times\boldsymbol{v})_z \\
&=\frac{\partial v_y}{\partial x}
-\frac{\partial v_x}{\partial y} \\
&=\frac{\partial^2\phi}{\partial x^2}
+\frac{\partial^2\phi}{\partial y^2}
=\nabla^2\phi .
\end{aligned}
\tag{4.8}
$$

源码采用

$$
j=-\nabla^2\psi .
\tag{4.9}
$$

注意：按式 (4.4) 直接计算物理旋度，有\((\nabla\times\boldsymbol{B})_z=\nabla^2\psi\)。因此式 (4.9) 是源码变量的显式符号约定。

### 4.4 Poisson 括号与平流项

定义二维 Poisson 括号

$$
[a,b]
=\frac{\partial a}{\partial x}\frac{\partial b}{\partial y}
-\frac{\partial a}{\partial y}\frac{\partial b}{\partial x}.
\tag{4.10}
$$

由式 (4.5) 可得

$$
\begin{aligned}
\boldsymbol{v}\cdot\nabla b
&=v_x\frac{\partial b}{\partial x}
+v_y\frac{\partial b}{\partial y} \\
&=-\frac{\partial\phi}{\partial y}\frac{\partial b}{\partial x}
+\frac{\partial\phi}{\partial x}\frac{\partial b}{\partial y} \\
&=[\phi,b].
\end{aligned}
\tag{4.11}
$$

因此 \([\phi,b]\) 表示标量 \(b\) 被不可压缩流场平流。

### 4.5 磁通方程

在二维势函数表示下，将感应方程投影到磁通函数，可写为

$$
\frac{\partial\psi}{\partial\tau}
+\boldsymbol{v}\cdot\nabla\psi
=\eta\nabla^2\psi .
\tag{4.12}
$$

使用式 (4.11)：

$$
\boxed{
\frac{\partial\psi}{\partial\tau}
+[\phi,\psi]
=\eta\nabla^2\psi
}
\tag{4.13}
$$

或写成数值右端项

$$
\frac{\partial\psi}{\partial\tau}
=-[\phi,\psi]+\eta\nabla^2\psi .
\tag{4.14}
$$

第一项表示磁通随流体平流，第二项表示电阻扩散。

### 4.6 涡度方程与符号审计

对动量方程取 \(z\) 向旋度，可消去纯梯度压强项：

$$
\nabla\times(-\nabla P)=\boldsymbol{0}.
\tag{4.15}
$$

惯性项变为涡度平流，黏性项变为涡度扩散。在
\(\boldsymbol{B}=(-\partial_y\psi,\partial_x\psi)\) 和
\(j=-\nabla^2\psi\) 的定义下，能量一致的主模型采用

$$
\boxed{
\frac{\partial\omega}{\partial\tau}
+[\phi,\omega]
=[j,\psi]+\nu\nabla^2\omega
}
\tag{4.16}
$$

即

$$
\frac{\partial\omega}{\partial\tau}
=-[\phi,\omega]+[j,\psi]+\nu\nabla^2\omega .
\tag{4.17}
$$

在周期边界和理想极限 \(\eta=\nu=0\) 下，非线性能量交换满足

$$
\frac{\mathrm{d}}{\mathrm{d}\tau}(E_B+E_K)=0 .
\tag{4.18}
$$

源码用瞬时能量交换残差验证式 (4.18)。旧
\([\psi,j]=-[j,\psi]\) 仅由
`--lorentz-convention legacy` 保留为诊断对照，不用于正式结果。

---

## 5. 周期双 Harris 电流片初值

### 5.1 为什么使用双电流片

Fourier 伪谱法天然假设周期边界。单 Harris 磁场在计算域两端通常不能无缝周期连接，因此源码使用两条方向相反的电流片，使磁场在周期域中闭合。

令

$$
y_{-}=-c,
\qquad
y_{+}=c,
\qquad
c=\chi L_y ,
\tag{5.1}
$$

式中，\(L_y\) 为 \(y\) 方向域长，\(\chi\) 为电流片中心位置比例，\(a\) 为半厚度。

### 5.2 平衡磁通

源码中的平衡磁通为

$$
\psi_0(y)
=-a\ln\cosh\left(\frac{y-y_-}{a}\right)
+a\ln\cosh\left(\frac{y-y_+}{a}\right)
+y .
\tag{5.2}
$$

使用

$$
\frac{\mathrm{d}}{\mathrm{d}y}
\left[
a\ln\cosh\left(\frac{y-y_s}{a}\right)
\right]
=\tanh\left(\frac{y-y_s}{a}\right),
\tag{5.3}
$$

可得

$$
\frac{\mathrm{d}\psi_0}{\mathrm{d}y}
=-\tanh\left(\frac{y-y_-}{a}\right)
+\tanh\left(\frac{y-y_+}{a}\right)+1 .
\tag{5.4}
$$

由于 \(\psi_0\) 与 \(x\) 无关，

$$
B_{y0}=\frac{\partial\psi_0}{\partial x}=0 ,
\tag{5.5}
$$

而

$$
\boxed{
B_{x0}
=-\frac{\mathrm{d}\psi_0}{\mathrm{d}y}
=\tanh\left(\frac{y-y_-}{a}\right)
-\tanh\left(\frac{y-y_+}{a}\right)-1
}.
\tag{5.6}
$$

两项双曲正切分别在 \(y_-\) 和 \(y_+\) 附近快速翻转，形成两条窄电流片。

### 5.3 平衡电流

利用

$$
\frac{\mathrm{d}}{\mathrm{d}y}
\tanh\left(\frac{y-y_s}{a}\right)
=\frac{1}{a}
\operatorname{sech}^{2}\left(\frac{y-y_s}{a}\right),
\tag{5.7}
$$

由式 (4.9) 得

$$
\boxed{
j_0(y)
=\frac{1}{a}\operatorname{sech}^{2}
\left(\frac{y-y_-}{a}\right)
-\frac{1}{a}\operatorname{sech}^{2}
\left(\frac{y-y_+}{a}\right)
}.
\tag{5.8}
$$

两条电流片的符号相反，峰值尺度随 \(1/a\) 增大。因此减小 \(a\) 会产生更尖锐的电流梯度，同时也提高空间分辨率和时间步长要求。

### 5.4 撕裂扰动

为触发确定性的岛链演化，源码加入

$$
\delta\psi(x,y)
=\epsilon\cos x
\left[
\exp\left(-\frac{(y-y_-)^2}{w^2}\right)
-\exp\left(-\frac{(y-y_+)^2}{w^2}\right)
\right].
\tag{5.9}
$$

初始条件为

$$
\psi(x,y,0)=\mathcal{D}_{2/3}\{\psi_0+\delta\psi\},
\qquad
\omega(x,y,0)=0 ,
\tag{5.10}
$$

式中，\(\epsilon\) 是扰动振幅，\(w\) 是局地包络宽度，\(\mathcal{D}_{2/3}\) 表示 \(2/3\) 谱滤波。初始扰动不使用随机数，所以 RMHD部分由配置完全确定；随机种子只影响射电背景和 spike 目录。

---

## 6. Fourier 伪谱离散与时间推进

### 6.1 周期网格

在

$$
x\in[-L_x/2,L_x/2),
\qquad
y\in[-L_y/2,L_y/2)
\tag{6.1}
$$

上取

$$
x_p=-\frac{L_x}{2}+p\Delta x,
\quad
\Delta x=\frac{L_x}{N_x},
\quad
p=0,\ldots,N_x-1 ,
\tag{6.2}
$$

$$
y_q=-\frac{L_y}{2}+q\Delta y,
\quad
\Delta y=\frac{L_y}{N_y},
\quad
q=0,\ldots,N_y-1 .
\tag{6.3}
$$

端点不重复，保证离散数据与 FFT 的周期假设一致。

### 6.2 Fourier 导数

对周期函数 \(g(x,y)\)，写成

$$
g(x,y)
=\sum_{m,n}\widehat{g}_{mn}
\exp\!\left[\mathrm{i}(k_{x,m}x+k_{y,n}y)\right].
\tag{6.4}
$$

由指数函数求导规则，

$$
\widehat{\frac{\partial g}{\partial x}}
=\mathrm{i}k_x\widehat{g},
\qquad
\widehat{\frac{\partial g}{\partial y}}
=\mathrm{i}k_y\widehat{g},
\tag{6.5}
$$

$$
\widehat{\nabla^2 g}
=-(k_x^2+k_y^2)\widehat{g}.
\tag{6.6}
$$

因此，数值步骤是：

1. 对实空间数组做二维 FFT；
2. 在谱空间乘 \(\mathrm{i}k_x\)、\(\mathrm{i}k_y\) 或
   \(-k^2\)；
3. 做逆 FFT；
4. 取理论上应为实数的实部。

### 6.3 Poisson 反演

由

$$
\omega=\nabla^2\phi
\tag{6.7}
$$

和式 (6.6)，非零波数满足

$$
\widehat{\omega}
=-k^2\widehat{\phi},
\qquad
\widehat{\phi}
=-\frac{\widehat{\omega}}{k^2},
\quad k^2>0 .
\tag{6.8}
$$

\(k=0\) 模式表示 \(\phi\) 的任意常数。常数不改变速度，因此源码将
\(\widehat{\phi}_{00}\) 设为零。

### 6.4 非线性括号和 \(2/3\) 去混叠

Poisson 括号包含导数乘积。两个最高波数模相乘会产生超出 Nyquist 极限的模，
并折回低波数产生 aliasing。令 \(m_x,m_y\) 为 FFT 整数模序号，源码严格保留

$$
|m_x|<\frac{N_x}{3},
\qquad
|m_y|<\frac{N_y}{3}
\tag{6.9}
$$

的谱系数，其余设为零。右端项和每个完整时间步后的状态都经过同一掩膜。

这里的 \(2/3\) 规则用于控制二次非线性混叠，不等于增加真实物理耗散，也不能替代空间收敛测试。

### 6.5 半离散右端项

记状态

$$
\boldsymbol{U}
=
\begin{pmatrix}
\psi\\
\omega
\end{pmatrix},
\qquad
\frac{\mathrm{d}\boldsymbol{U}}{\mathrm{d}\tau}
=\boldsymbol{F}(\boldsymbol{U}).
\tag{6.10}
$$

每次计算 \(\boldsymbol{F}\) 时：

1. 由 \(\nabla^2\phi=\omega\) 反演 \(\phi\)；
2. 由 \(j=-\nabla^2\psi\) 计算电流变量；
3. 计算 \([\phi,\psi]\)、\([\phi,\omega]\) 和 \([j,\psi]\)；
4. 加入电阻和黏性扩散；
5. 对两个右端项做 \(2/3\) 过滤。

对应源码的核心形式是：

```python
phi = grid.poisson_solve(omega)
current = -grid.laplacian(psi)

d_psi = (
    -grid.bracket(phi, psi)
    + config.resistivity * grid.laplacian(psi)
)
d_omega = (
    -grid.bracket(phi, omega)
    + grid.bracket(current, psi)
    + config.viscosity * grid.laplacian(omega)
)
return grid.filter(d_psi), grid.filter(d_omega)
```

### 6.6 四阶 Runge-Kutta

固定时间步长 \(\Delta\tau\) 下，经典 RK4 为

$$
\boldsymbol{K}_1
=\boldsymbol{F}(\boldsymbol{U}^{n}),
\tag{6.11}
$$

$$
\boldsymbol{K}_2
=\boldsymbol{F}
\left(
\boldsymbol{U}^{n}
+\frac{\Delta\tau}{2}\boldsymbol{K}_1
\right),
\tag{6.12}
$$

$$
\boldsymbol{K}_3
=\boldsymbol{F}
\left(
\boldsymbol{U}^{n}
+\frac{\Delta\tau}{2}\boldsymbol{K}_2
\right),
\tag{6.13}
$$

$$
\boldsymbol{K}_4
=\boldsymbol{F}
\left(
\boldsymbol{U}^{n}
+\Delta\tau\boldsymbol{K}_3
\right),
\tag{6.14}
$$

$$
\boxed{
\boldsymbol{U}^{n+1}
=\boldsymbol{U}^{n}
+\frac{\Delta\tau}{6}
\left(
\boldsymbol{K}_1
+2\boldsymbol{K}_2
+2\boldsymbol{K}_3
+\boldsymbol{K}_4
\right)
}.
\tag{6.15}
$$

当前代码没有自适应时间步，也不会随时间自动提高网格分辨率。提高精度必须通过显式的 \((N_x,N_y,\Delta\tau)\) 收敛研究完成，见第 10 节。

### 6.7 RMHD 回归诊断量

源码计算域平均磁能和动能：

$$
E_B
=\frac{1}{2}
\left\langle B_x^2+B_y^2\right\rangle ,
\tag{6.16}
$$

$$
E_K
=\frac{1}{2}
\left\langle v_x^2+v_y^2\right\rangle .
\tag{6.17}
$$

并记录

$$
j_{\max}=\max_{x,y}|j|,
\qquad
v_{\max}=\max_{x,y}\sqrt{v_x^2+v_y^2},
\tag{6.18}
$$

$$
R(\tau)=\eta j_{\max}(\tau).
\tag{6.19}
$$

式 (6.17) 是全域无密度权重的归一化动能，不是某个喷流区域的\(\frac12\int\rho v^2\,\mathrm{d}V\)。式 (6.18) 的 \(v_{\max}\) 也是全域最大速度，不能单独证明该像素属于重联喷流。

无散残差定义为

$$
\epsilon_{\nabla\cdot B}
=
\frac{
\left\langle(\nabla\cdot\boldsymbol{B})^2\right\rangle^{1/2}
}{
\left\langle|\boldsymbol{B}|^2\right\rangle^{1/2}
}.
\tag{6.20}
$$

它用于检查数值结构，不代表模型已经通过全部物理验证。

---

## 7. 以 jet 为核心的活动代理与条件耦合

### 7.1 正式重联活动

Athena 后端以式 (4A.18) 的局地 O/X 点磁通差变化率作为 \(R(\tau)\)，并以
式 (4A.19) 的 \(E_z(X)\) 交叉检查。RMHD 回归后端才使用式 (6.19)。随后对
所选后端的 \(R\) 做归一化：

$$
q_R(\tau)
=
\operatorname{clip}
\left[
\frac{R(\tau)-R(0)}
{\max R-\min R},
0,1
\right].
\tag{7.1}
$$

若 \(R\) 的极差接近零，则令 \(q_R=0\)。随后通过线性插值，将 MHD 时间
\(\tau\) 映射到射电时间 \(t\)。

全时段线性重采样的 \(q_R\) 继续调制 Type III 主脊；条件采样则使用第 7.5 节
定义的压缩 onset 映射。两套时间用途在元数据中分别标记，避免把代理映射误当
作量纲恢复。

### 7.2 为什么不能只使用 \(v_{\max}\)

全域最大速度可能出现在：

- 电流片附近的真实双向出流；
- 磁岛边界的旋转流；
- 数值域其他局地梯度；
- 少数异常网格点。

所以 jet 判据必须同时限定空间区域、速度方向和时间连续性。

### 7.3 jet 区域

RMHD 回归后端仍可用两条电流片带状掩膜。Athena 正式后端先从磁通函数定位每个
X 点，再在其左右建立出流窗口 \(\Omega_X^\pm\)。带状几何可写为

$$
\Omega_{\mathrm{sheet}}
=
\left\{
(x,y):
\min_{s\in\{-,+\}}|y-y_s|
\leq c_J a
\right\},
\tag{7.2}
$$

式中，\(c_J\) 是无量纲带宽系数。正式 Athena 诊断再与 X 点左右窗口相交，
排除远离重联区的磁岛旋转流。对于水平电流片，主要出流方向沿 \(x\)，因此用
\(v_x\) 的正、负分位数构造稳健代理。

### 7.4 稳健 jet 活动量

源码定义

$$
V_J(\tau)
=Q_p
\left(
|v_x(x,y,\tau)|
\;\middle|\;
(x,y)\in\Omega_{\mathrm{sheet}}
\right),
\tag{7.3}
$$

其中 \(Q_p\) 是第 \(p\) 分位数。主结果固定 \(p=0.95\)，而不是直接取单个
像素最大值。再定义

$$
q_J(\tau)
=
\operatorname{clip}
\left[
\frac{V_J(\tau)-V_{J,0}}
{V_{J,\max}-V_{J,0}+\varepsilon},
0,1
\right].
\tag{7.4}
$$

\(\varepsilon\) 是防止零除的小正数，\(V_{J,0}\) 可取初始值或静默阶段中位数。

为了确认“双向”而不是单向局地流，源码分别计算

$$
V_J^{+}=Q_p(v_x\mid v_x>0,\Omega_{\mathrm{sheet}}),
\qquad
V_J^{-}=Q_p(-v_x\mid v_x<0,\Omega_{\mathrm{sheet}}),
\tag{7.5}
$$

并要求

$$
\min(V_J^{+},V_J^{-})>V_{\mathrm{thr}} .
\tag{7.6}
$$

### 7.5 jet 起始时刻与射电时间对齐

定义 jet 起始时刻为

$$
\tau_J
=
\inf
\left\{
\tau:
q_J(\tau)\geq\theta_J
\ \text{且连续保持至少 }m\text{ 个快照}
\right\}.
\tag{7.7}
$$

主结果取 \(\theta_J=\theta_R=0.6\)，并要求连续 3 个快照。随后将
\([\tau_J,\tau_{\mathrm{end}}]\) 线性压缩到
\([0.08,0.75]\) s，再插值得到条件采样专用的 \(q_J(t)\) 和 \(q_R(t)\)。
这一步是代理时间标定，不是 RMHD 量纲恢复。

### 7.6 伴随 jet 的 spike 条件

实现先构造候选时间集合

$$
\mathcal{T}_{\mathrm{cand}}
=
\left\{
t\in\mathcal{W}_{\mathrm{onset}}:
q_J(t)\geq\theta_J,\
q_R(t)\geq\theta_R
\right\}.
\tag{7.8}
$$

在候选集合内按

$$
\mathcal{P}(t)
\propto
q_J(t)\,q_R(t)
\tag{7.9}
$$

抽取 spike 时刻。若 \(\mathcal{T}_{\mathrm{cand}}\) 为空，返回形状为
`(0, 5)` 的空 catalog，并在元数据中记录 `no_event`；阈值不会被静默放宽。

核心实现位于 `physics/radio.py`：

```python
def sample_jet_conditioned_times(
    radio_times,
    jet_activity,
    reconnection_activity,
    onset_mask,
    spike_count,
    rng,
    jet_threshold=0.6,
    reconnection_threshold=0.6,
):
    valid = (
        onset_mask
        & (jet_activity >= jet_threshold)
        & (reconnection_activity >= reconnection_threshold)
    )
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return np.empty(0, dtype=float)

    weights = jet_activity[candidates] * reconnection_activity[candidates]
    weights = weights / weights.sum()
    indices = rng.choice(
        candidates,
        size=min(spike_count, candidates.size),
        replace=False,
        p=weights,
    )
    return radio_times[indices]
```

实际函数在高密度连续时间网格上做加权求积，并采用不放回抽样，因此固定 seed
可复现且中心时刻不重复。

### 7.7 伴随性的量化指标

对 \(N_{\mathrm{sp}}\) 个 spike，定义同期率

$$
C_{\mathrm{jet}}
=\frac{1}{N_{\mathrm{sp}}}
\sum_{k=1}^{N_{\mathrm{sp}}}
\mathbf{1}
\left[
q_J(t_k)\geq\theta_J
\ \land\
q_R(t_k)\geq\theta_R
\right].
\tag{7.10}
$$

定义相对最近 jet 起始时刻的延迟

$$
\Delta t_{J,k}=t_k-t_{J,\mathrm{nearest}},
\tag{7.11}
$$

以及 topping 余量

$$
\Delta f_k=f_k-f_{\mathrm{III}}(t_k).
\tag{7.12}
$$

研究结论不应只展示一张动态图，而应同时报告\(C_{\mathrm{jet}}\)、\(\Delta t_{J,k}\)、\(\Delta f_k\) 的分布以及阈值敏感性。

---

## 8. 从电子束到 Type III 主脊

### 8.1 匀速电子束

令电子束速度为

$$
v_b=\beta c,
\qquad
0<\beta<1 ,
\tag{8.1}
$$

则 Lorentz 因子为

$$
\gamma
=\frac{1}{\sqrt{1-\beta^2}} .
\tag{8.2}
$$

若起始高度取 \(h_0=0\)，匀速运动给出

$$
h(t)=v_bt=\beta ct .
\tag{8.3}
$$

源码以 Mm 存储高度，因此

$$
h_{\mathrm{Mm}}(t)
=\frac{\beta ct}{10^6}.
\tag{8.4}
$$

这里 \(c\) 的单位为 m s\(^{-1}\)，\(t\) 的单位为 s。当前模型不积分
Lorentz 力，不考虑速度弥散和能量损失。

对应纯函数为：

```python
speed_m_s = speed_fraction_c * SPEED_OF_LIGHT_M_S
height_mm = speed_m_s * np.asarray(times_s, dtype=float) / 1.0e6
gamma = 1.0 / np.sqrt(1.0 - speed_fraction_c**2)
```

### 8.2 指数日冕密度

假设相对密度梯度为常数：

$$
\frac{1}{n_e}\frac{\mathrm{d}n_e}{\mathrm{d}h}
=-\frac{1}{H}.
\tag{8.5}
$$

分离变量：

$$
\frac{\mathrm{d}n_e}{n_e}
=-\frac{\mathrm{d}h}{H}.
\tag{8.6}
$$

从 \(h_0\) 积分到 \(h\)：

$$
\int_{n_0}^{n_e}\frac{\mathrm{d}n'}{n'}
=-\int_{h_0}^{h}\frac{\mathrm{d}h'}{H},
\tag{8.7}
$$

$$
\ln\frac{n_e}{n_0}
=-\frac{h-h_0}{H}.
\tag{8.8}
$$

因此

$$
\boxed{
n_e(h)
=n_0
\exp\left[-\frac{h-h_0}{H}\right]
}.
\tag{8.9}
$$

\(n_0\) 的单位为 cm\(^{-3}\)，\(h\) 和 \(H\) 必须使用相同长度单位。

### 8.3 电子等离子体频率

考虑电子相对静止离子背景发生小位移 \(\xi\)。电荷分离产生的恢复电场与位移成正比。式 (8.10)–式 (8.13) 暂用 SI 制，其中 \(n_e\) 的单位为 m\(^{-3}\)：

$$
E=\frac{n_e e}{\varepsilon_0}\xi .
\tag{8.10}
$$

电子运动方程为

$$
m_e\frac{\mathrm{d}^2\xi}{\mathrm{d}t^2}
=-eE
=-\frac{n_e e^2}{\varepsilon_0}\xi .
\tag{8.11}
$$

与简谐振子方程

$$
\frac{\mathrm{d}^2\xi}{\mathrm{d}t^2}
+\omega_{\mathrm{pe}}^2\xi=0
\tag{8.12}
$$

比较，得到

$$
\omega_{\mathrm{pe}}
=
\sqrt{\frac{n_e e^2}{m_e\varepsilon_0}},
\qquad
f_{\mathrm{pe}}
=\frac{\omega_{\mathrm{pe}}}{2\pi}.
\tag{8.13}
$$

当 \(n_e\) 用 cm\(^{-3}\) 表示时，常用数值形式为

$$
\boxed{
f_{\mathrm{pe}}[\mathrm{Hz}]
\simeq 8980
\sqrt{n_e[\mathrm{cm}^{-3}]}
}.
\tag{8.14}
$$

当前代码采用基频映射，没有额外乘以谐波数。

### 8.4 由起始频率确定基底密度

设电子束在 \(h_0=0\) 处对应主脊起始频率 \(f_0\)，则

$$
f_0\times10^6
=8980\sqrt{n_0}.
\tag{8.15}
$$

解得

$$
\boxed{
n_0
=
\left(
\frac{f_0\times10^6}{8980}
\right)^2
}.
\tag{8.16}
$$

式中 \(f_0\) 用 MHz 输入，所得 \(n_0\) 为 cm\(^{-3}\)。

### 8.5 Type III 主脊

将式 (8.9) 代入式 (8.14)：

$$
\begin{aligned}
f_{\mathrm{III}}(h)
&=8980\sqrt{
n_0\exp\left(-\frac{h}{H}\right)
} \\
&=8980\sqrt{n_0}
\exp\left(-\frac{h}{2H}\right).
\end{aligned}
\tag{8.17}
$$

再用式 (8.4) 和式 (8.15)：

$$
\boxed{
f_{\mathrm{III}}(t)
=f_0
\exp\left[
-\frac{\beta ct}{2H\times10^6}
\right]
}.
\tag{8.18}
$$

对时间求导：

$$
\boxed{
\frac{\mathrm{d}f_{\mathrm{III}}}{\mathrm{d}t}
=-\frac{\beta c}{2H\times10^6}
f_{\mathrm{III}}(t)<0
}.
\tag{8.19}
$$

因此指数密度与向外匀速电子束自然给出负频漂。式 (8.19) 还表明，在这一简化模型中，漂移率主要约束组合参数 \(\beta/H\)，单独反演 \(\beta\) 或 \(H\) 存在退化。

### 8.6 Type III-like 主脊强度

当前动态谱主脊写为

$$
I_{\mathrm{III}}(f,t)
=A_R(t)
\exp
\left[
-\frac{1}{2}
\left(
\frac{f-f_{\mathrm{III}}(t)}{w_R(t)}
\right)^2
\right],
\tag{8.20}
$$

其中

$$
A_R(t)=0.35+0.65q_R(t),
\qquad
w_R(t)=5+2q_R(t)\ \mathrm{MHz}.
\tag{8.21}
$$

式 (8.21) 是可视化代理，不是由辐射转移推导出的物理定律。

---

## 9. 严格 Spike-Topping 的构造

### 9.1 起始时间窗

定义

$$
t_{\mathrm{end}}
=
\min(t_{\mathrm{cap}},\alpha T),
\tag{9.1}
$$

$$
\boxed{
t_k\sim
\mathcal{U}
\left(
t_{\mathrm{start}},
t_{\mathrm{end}}
\right)
}.
\tag{9.2}
$$

当前默认值为

$$
t_{\mathrm{start}}=0.08\ \mathrm{s},
\qquad
\alpha=0.25,
\qquad
t_{\mathrm{cap}}=0.75\ \mathrm{s}.
\tag{9.3}
$$

配置要求

$$
0\leq t_{\mathrm{start}}
<
\min(t_{\mathrm{cap}},\alpha T),
\qquad
0<\alpha\leq1.
\tag{9.4}
$$

### 9.2 严格正频偏

在 \(t_k\) 处插值得到主脊频率

$$
f_{\mathrm{ridge},k}
=f_{\mathrm{III}}(t_k).
\tag{9.5}
$$

受观测频带上限 \(f_{\max}\) 限制，可用最大偏移为

$$
\Delta f_{k,\max}^{\mathrm{avail}}
=
\min
\left(
\Delta f_{\max},
f_{\max}-f_{\mathrm{ridge},k}
\right).
\tag{9.6}
$$

若

$$
\Delta f_{k,\max}^{\mathrm{avail}}
\leq\Delta f_{\min},
\tag{9.7}
$$

则频带没有足够空间保证严格正偏移，源码抛出 `ValueError`。否则

$$
\Delta f_k
\sim
\mathcal{U}
\left(
\Delta f_{\min},
\Delta f_{k,\max}^{\mathrm{avail}}
\right),
\tag{9.8}
$$

$$
\boxed{
f_k=f_{\mathrm{ridge},k}+\Delta f_k
>f_{\mathrm{ridge},k}
}.
\tag{9.9}
$$

当前默认偏移范围为 \(5\)–\(40\) MHz，并受式 (9.6) 进一步裁剪。

对应源码核心逻辑为：

```python
ridge_at_center = float(
    np.interp(center_time, times, ridge_frequency)
)
available_offset_max = min(
    config.spike_frequency_offset_max_mhz,
    config.max_frequency_mhz - ridge_at_center,
)
if available_offset_max <= config.spike_frequency_offset_min_mhz:
    raise ValueError(
        "Frequency band leaves no room for a strictly positive "
        "Spike-Topping offset at the selected onset time."
    )

frequency_offset = float(
    rng.uniform(
        config.spike_frequency_offset_min_mhz,
        available_offset_max,
    )
)
center_frequency = ridge_at_center + frequency_offset
```

### 9.3 二维 Gaussian spike

第 \(k\) 个 spike 定义为

$$
\boxed{
I_k(f,t)
=A_k
\exp
\left[
-\frac{(t-t_k)^2}{2\sigma_{t,k}^2}
-\frac{(f-f_k)^2}{2\sigma_{f,k}^2}
\right]
}.
\tag{9.10}
$$

式中：

- \(A_k\) 为归一化振幅；
- \(\sigma_{t,k}\) 为时间标准差，单位 s；
- \(\sigma_{f,k}\) 为频率标准差，单位 MHz；
- \((t_k,f_k)\) 为中心。

Gaussian 的半高全宽为

$$
\mathrm{FWHM}_t
=2\sqrt{2\ln2}\,\sigma_t,
\qquad
\mathrm{FWHM}_f
=2\sqrt{2\ln2}\,\sigma_f.
\tag{9.11}
$$

当前源码从预设范围抽取\(\sigma_t\)、\(\sigma_f\) 和 \(A\)。这些范围是现象学参数，正式观测比较时应由时间分辨率、频率分辨率和观测统计约束。

### 9.4 总动态谱与归一化

未归一化动态谱为

$$
\widetilde{I}(f,t)
=I_{\mathrm{bg}}(f,t)
+I_{\mathrm{III}}(f,t)
+\sum_{k=1}^{N_{\mathrm{sp}}}I_k(f,t).
\tag{9.12}
$$

最后做

$$
I(f,t)
=
\frac{
\widetilde{I}(f,t)-\min\widetilde{I}
}{
\max\widetilde{I}-\min\widetilde{I}
}.
\tag{9.13}
$$

所以 \(I\in[0,1]\) 是相对强度，不是 Jy、sfu 或亮温。

### 9.5 spike 目录与随机种子

`spike_catalog` 保持五列：

```text
[center_time_s,
 center_frequency_mhz,
 sigma_time_s,
 sigma_frequency_mhz,
 amplitude]
```

固定 `--seed` 可复现：

- 射电背景噪声；
- spike 中心时刻；
- spike 频率偏移；
- spike 宽度和振幅。

它不改变确定性的双 Harris 初值和 RMHD 时间推进。

---

## 10. 精度提升、参数研究和验收思路

### 10.1 精度不会随时间自动提高

当前求解器使用固定 \(N_x\)、\(N_y\) 和 \(\Delta\tau\)。模拟时间增加只会产生更多时间步，不会自动提高空间或时间精度。相反，若长时间演化形成更细尺度结构，固定网格可能变得更难解析。

### 10.2 建议的收敛研究

至少设置三组网格：

| 级别 | 建议网格 | 时间步处理 | 用途 |
|---|---:|---|---|
| 粗网格 | \(48\times48\) | 基准 \(\Delta\tau\) | 调试和趋势筛选 |
| 中网格 | \(96\times96\) | 基准或减半 | 当前 standard 基线 |
| 细网格 | \(192\times192\) 或更高 | 按稳定性缩小 | 收敛评估 |

对诊断量 \(Q\)，可计算相邻分辨率差

$$
\epsilon_Q^{(N,2N)}
=
\frac{|Q_{2N}-Q_N|}
{\max(|Q_{2N}|,\epsilon)}.
\tag{10.1}
$$

建议比较：

- \(j_{\max}(\tau)\) 峰值和峰值时刻；
- \(V_J(\tau)\) 与 jet 起始时刻；
- 总能量漂移；
- \(\epsilon_{\nabla\cdot B}\)；
- Type III 主脊的单调性；
- \(C_{\mathrm{jet}}\)、\(\Delta t_J\) 和 \(\Delta f\) 的统计。

只有关键结论对网格、时间步和输出步长稳定，才能称为数值收敛。

### 10.3 时间步研究

固定空间网格，依次使用

$$
\Delta\tau,\quad
\frac{\Delta\tau}{2},\quad
\frac{\Delta\tau}{4}.
\tag{10.2}
$$

RK4 在光滑解和稳定区间内具有四阶时间精度，但完整误差还受谱截断、去混叠和电流片解析度影响。不能仅凭 RK4 的形式阶数推断整个模拟已达到四阶收敛。

### 10.4 物理参数组

建议一次只改变一组机制参数：

- **电流片：** \(a\)、\(\chi\)、\(\epsilon\)、\(w\)；
- **耗散：** \(\eta\)、\(\nu\) 及磁 Prandtl 数 \(\nu/\eta\)；
- **jet 判据：** \(c_J\)、\(p\)、\(\theta_J\)、连续快照数 \(m\)；
- **电子束：** \(\beta\)、\(H\)、\(f_0\)；
- **topping：** 起始窗、\(\Delta f_{\min}\)、\(\Delta f_{\max}\)、
  \(\sigma_t\)、\(\sigma_f\)、\(N_{\mathrm{sp}}\)；
- **随机性：** 多个 seed 的重复实验。

### 10.5 必要对照实验

1. 无扰动或减小扰动：检查 jet 与 spike 条件是否消失或延迟；
2. 固定 \(q_R\)：区分主脊幅度调制与频漂几何；
3. 关闭 spike：获得纯 Type III 主脊基线；
4. spike 不使用 jet 条件：与式 (7.8) 的条件模型比较；
5. 打乱 \(q_J(t)\)：估计偶然同期率；
6. 改变 \(\theta_J,\theta_R\)：检查伴随性结论的阈值敏感性。

### 10.6 与观测比较的量

可比较：

- Type III 漂移率 \(\mathrm{d}f/\mathrm{d}t\)；
- 起始频率与终止频率；
- spike 持续时间和相对带宽；
- topping 余量 \(\Delta f_k\)；
- jet 起始与 spike 的时间延迟；
- spike 数量随 jet／重联活动的变化。

由于模型没有绝对辐射标定和传播效应，不应直接比较绝对强度或从当前代理唯一反演
电子束能谱。

---

## 11. 公式—源码映射与代表性代码

### 11.1 实际项目结构

```text
solarphysics/
├── Local/                      # 被忽略的构建、原始数据和本地归档
└── simulation/
    ├── configs/                # Athena/双后端和事件配置
    ├── fluxrope_demo/
    │   └── athena4.2/          # 唯一权威 Athena C 源码树
    │       ├── configure
    │       ├── src/prob/
    │       │   ├── fluxrope.c
    │       │   ├── spike_topping_jet.c
    │       │   └── spike_topping_solar_jet.c
    │       └── tst/2D-mhd/
    │           ├── athinput.spike_topping_jet
    │           └── athinput.spike_topping_solar_jet
    ├── amrvac/
    │   ├── amrvac/             # 固化的 GPLv3 vendor 源码快照
    │   ├── VENDOR_PROVENANCE.md
    │   └── spike_topping_solar_jet/
    ├── Mercury/
    │   ├── README_中文.md
    │   ├── run_mercury_bowshock.m
    │   └── bowshock_analysis/  # MATLAB 源码；MAT/ZIP/结果不进入 Git
    ├── server/gridview/        # 可审核的服务器脚本模板
    ├── windows/                # WSL2 环境与验证说明
    ├── environment.solar-simulation.yml
    ├── locks/
    │   ├── solar-simulation-osx-arm64.txt
    │   └── solar-simulation-linux-64.txt
    ├── README.md
    ├── readme.pdf
    ├── RESULTS.md
    ├── Spike_Topping_TypeIII_*teacher_report.pptx
    ├── Spike_Topping_TypeIII_complete_speaker_script_CN.md
    └── spike_typeIII_visual/
        ├── config.py
        ├── athena.py
        ├── athena_io.py
        ├── experiments.py
        ├── main.py
        ├── physics/
        │   ├── fields.py
        │   ├── jet.py
        │   ├── rmhd.py
        │   └── radio.py
        ├── visualization/
        │   ├── figures.py
        │   └── animations.py
        ├── validate_outputs.py
        ├── tests/
        │   ├── test_amrvac_backend.py
        │   ├── test_athena_backend.py
        │   └── test_simulation.py
        └── outputs/             # 本地历史结果，被 Git 忽略
```

这里的权威求解器是 **Athena 4.2 基础版加项目最小兼容补丁**：保留
flux-rope 示例，加入二维双 Harris 与 2.5D solar-jet 问题，并修正原版
`viscosity.c` 对 `Real3Vect` 分量名的编译不兼容。正式构建始终复制源码到
`Local/athena/`，不得在该权威源码树内保存对象文件、二进制、配置日志或原始
运行结果。

### 11.2 映射表

| 理论或功能 | 源码位置 | 主要对象 |
|---|---|---|
| Athena 双 Harris 初态 | `fluxrope_demo/athena4.2/src/prob/spike_topping_jet.c` | `problem`、`get_eta_user` |
| Athena 输入 | `fluxrope_demo/athena4.2/tst/2D-mhd/athinput.spike_topping_jet` | 网格、输出、物理参数 |
| doctor／build／run／ingest／benchmark | `spike_typeIII_visual/athena.py` | Athena 工作流 CLI |
| Athena BIN 读取与 MPI 拼接 | `spike_typeIII_visual/athena_io.py` | `read_athena_bin_series` |
| HDF5 桥接 | `spike_typeIII_visual/athena_io.py` | `write_bridge_hdf5`、`read_bridge_hdf5` |
| 后端无关场接口 | `physics/fields.py` | `MHDFieldSeries`、`FieldGrid` |
| quick／standard 参数 | `spike_typeIII_visual/config.py` | `MHDConfig`、`RadioConfig`、`profile_config` |
| 谱网格、导数和 Poisson 反演 | `physics/rmhd.py` | `SpectralGrid` |
| 双 Harris 初值 | `physics/rmhd.py` | `double_harris_flux` |
| RMHD 右端项 | `physics/rmhd.py` | `_rhs` |
| RK4 推进 | `physics/rmhd.py` | `_rk4_step` |
| RMHD 能量、电流、速度 | `physics/rmhd.py` | `_diagnostics` |
| X 点窗口、双向速度和 onset | `physics/jet.py` | `diagnose_jet` |
| 压缩时间映射 | `physics/jet.py` | `map_active_interval_to_radio_time` |
| 电子束运动学 | `physics/radio.py` | `electron_beam_kinematics` |
| 指数密度 | `physics/radio.py` | `exponential_coronal_density_cm3` |
| 等离子体频率 | `physics/radio.py` | `plasma_frequency_hz` |
| Type III 主脊 | `physics/radio.py` | `typeiii_ridge_frequency_mhz` |
| 二维 Gaussian spike | `physics/radio.py` | `gaussian_spike_pulse` |
| 动态谱合成 | `physics/radio.py` | `synthesize_radio_proxy` |
| jet 条件加权抽样 | `physics/radio.py` | `sample_jet_conditioned_times` |
| 空间／时间收敛与对照 | `experiments.py` | `run_suite` |
| 双后端 CLI、schema v3 和 SHA-256 | `main.py` | `main`、`run`、`_write_data`、`_write_manifest` |
| GIF／MP4 编码 | `visualization/animations.py` | `save_animations` |
| 产物结构与摘要校验 | `validate_outputs.py` | `validate` |

### 11.3 势函数到物理场

```python
def fields(self, psi, omega):
    phi = self.poisson_solve(omega)
    magnetic_x = -self.derivative_y(psi)
    magnetic_y = self.derivative_x(psi)
    velocity_x = -self.derivative_y(phi)
    velocity_y = self.derivative_x(phi)
    current = -self.laplacian(psi)
    return (
        magnetic_x,
        magnetic_y,
        velocity_x,
        velocity_y,
        current,
        phi,
    )
```

### 11.4 双 Harris 初值

```python
flux = (
    -width * np.log(np.cosh((y - y_lower) / width))
    + width * np.log(np.cosh((y - y_upper) / width))
    + y
)
envelope = (
    np.exp(-((y - y_lower) / perturbation_width) ** 2)
    - np.exp(-((y - y_upper) / perturbation_width) ** 2)
)
perturbation = amplitude * np.cos(grid.x_mesh) * envelope
psi = grid.filter(flux + perturbation)
```

### 11.5 射电纯函数

```python
def exponential_coronal_density_cm3(
    height_mm,
    base_density_cm3,
    scale_height_mm,
):
    return base_density_cm3 * np.exp(
        -np.asarray(height_mm, dtype=float) / scale_height_mm
    )


def typeiii_ridge_frequency_mhz(
    height_mm,
    base_density_cm3,
    scale_height_mm,
):
    density_cm3 = exponential_coronal_density_cm3(
        height_mm,
        base_density_cm3,
        scale_height_mm,
    )
    return plasma_frequency_hz(density_cm3) / 1.0e6
```

### 11.6 正式 Athena 配置与 RMHD 回归配置

Athena `standard` 固定为 \(512\times256\)、\(t_{\rm end}=2\)，并在输入中取
\(\eta=\nu=0.002\)、\(\gamma=5/3\)、\(\beta=1\)。档位为：

| profile | 网格 | 用途 |
|---|---:|---|
| smoke | \(128\times64\) | 构建与解析检查 |
| coarse | \(256\times128\) | 收敛粗网格 |
| standard | \(512\times256\) | 正式基线 |
| fine | \(1024\times512\) | 收敛细网格 |

Python RMHD 的 `standard` 回归配置为：

```python
MHDConfig(
    nx=96,
    ny=96,
    lx=4.0 * np.pi,
    ly=2.0 * np.pi,
    sheet_half_width=0.20,
    sheet_center_fraction=0.25,
    perturbation_amplitude=0.04,
    perturbation_width=0.45,
    resistivity=0.002,
    viscosity=0.002,
    dt=0.005,
    steps=400,
    snapshot_stride=10,
    lorentz_convention="physical",
)
```

jet 条件固定为：

```python
JetConfig(
    sheet_half_width_factor=2.0,
    velocity_quantile=0.95,
    jet_threshold=0.60,
    reconnection_threshold=0.60,
    consecutive_snapshots=3,
)
```

射电默认配置包括：

```python
RadioConfig(
    start_frequency_mhz=300.0,
    min_frequency_mhz=20.0,
    max_frequency_mhz=350.0,
    density_scale_height_mm=50.0,
    beam_speed_fraction_c=0.20,
    duration_s=4.5,
    time_samples=180,
    frequency_samples=256,
    spike_count=12,
    spike_onset_start_s=0.08,
    spike_onset_fraction=0.25,
    spike_onset_cap_s=0.75,
    spike_frequency_offset_min_mhz=5.0,
    spike_frequency_offset_max_mhz=40.0,
)
```

RMHD `quick` 只降低网格、步数和射电采样用于快速检查。Athena 分辨率由
`athena run --profile ...` 选择，不与 RMHD profile 混用。

---

## 12. 动画选择、输出与可复现性

### 12.1 CLI 接口

Athena 工作流在 Apple Silicon Mac 上已经验证。构建和原始输出自动进入被忽略的
`Local/athena/`：

```bash
# 以下命令从 solarphysics/simulation/ 执行

# 检查工具链并构建正式 serial HLLD 二进制
python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build --flux hlld

# BINARY 使用 build 返回的路径
python -m spike_typeIII_visual.athena run \
  --binary BINARY \
  --profile standard \
  --run-id physical_jet_seed20260726

# 将真实 BIN 网格和时间写入 schema-v3 HDF5
python -m spike_typeIII_visual.athena ingest \
  --run-dir ../Local/athena/runs/physical_jet_seed20260726 \
  --output ../Local/athena/bridge/physical_jet_seed20260726.h5

# 正式射电结果必须显式选择 Athena
python -m spike_typeIII_visual.main \
  --profile standard \
  --seed 20260726 \
  --mhd-backend athena \
  --athena-dataset ../Local/athena/bridge/physical_jet_seed20260726.h5 \
  --spike-coupling jet \
  --time-calibration proxy \
  --animation-format both \
  --output-dir spike_typeIII_visual/outputs/runs/athena_physical_jet_seed20260726
```

Alfvén 时间标定不提供隐藏默认尺度，必须同时显式给出：

```bash
python -m spike_typeIII_visual.main \
  --mhd-backend athena \
  --athena-dataset BRIDGE_H5 \
  --time-calibration alfven \
  --length-scale-mm L0_MM \
  --magnetic-field-gauss B0_G \
  --electron-density-cm3 NE0_CM3 \
  --animation-format none
```

为兼容旧脚本，不提供 `--mhd-backend` 时仍使用 RMHD。两种后端的动画选择均为：

```bash
python -m spike_typeIII_visual.main --animation-format none
python -m spike_typeIII_visual.main --animation-format gif
python -m spike_typeIII_visual.main --animation-format mp4
python -m spike_typeIII_visual.main --animation-format both
```

`--skip-animations` 是兼容旧命令的别名，等价于
`--animation-format none`。二者同时显式给出会报参数冲突。

RMHD 科学实验入口仍为：

```bash
python -m spike_typeIII_visual.experiments
```

### 12.2 四类动画

| 物理内容 | 文件主名 | GIF | MP4 |
|---|---|---|---|
| 电流与磁通等值线 | `tearing` | `tearing.gif` | `tearing.mp4` |
| 双向速度结构 | `jet` | `jet.gif` | `jet.mp4` |
| 电子束高度 | `electron_beam` | `electron_beam.gif` | `electron_beam.mp4` |
| Type III 动态谱 | `typeIII` | `typeIII.gif` | `typeIII.mp4` |

选择 `both` 时，每类动画只渲染一次 RGB 帧，再分别编码为 GIF 和 MP4，避免重复绘图。MP4 使用 H.264、`yuv420p`、\(960\times540\) 和 10 fps。

### 12.3 输出清单

每次运行的权威范围由`data/run_metadata.json` 中的 `exports.animation_formats` 和`SHA256SUMS.txt` 决定。复用输出目录时，程序不会自动删除旧格式动画，因此不能仅凭
目录中“存在某文件”判断它属于本次运行。

结构预期为：

```text
outputs/runs/athena_physical_jet_seed20260726/
├── figures/                    # 13 张 1600×900 PNG
├── animations/                 # 按 none/gif/mp4/both 选择
├── data/
│   ├── diagnostics.csv
│   ├── mhd_bridge.h5
│   ├── mhd_snapshots.npz
│   └── run_metadata.json
└── SHA256SUMS.txt
```

Athena 正式 `both` 清单覆盖 25 个产物。具体范围始终以
`exports.animation_formats` 和清单为准。`SHA256SUMS.txt` 不列出自身；
`validation_report.json` 只有显式
调用校验器并指定报告路径时才生成，也不属于主程序摘要清单。

### 12.4 输出校验

可运行：

```bash
python -m spike_typeIII_visual.validate_outputs \
  spike_typeIII_visual/outputs/runs/athena_physical_jet_seed20260726
```

若需要报告：

```bash
python -m spike_typeIII_visual.validate_outputs \
  spike_typeIII_visual/outputs/runs/athena_physical_jet_seed20260726 \
  --report \
  spike_typeIII_visual/outputs/runs/athena_physical_jet_seed20260726/validation_report.json
```

校验器检查：

- 13 张 PNG 的尺寸和非空白程度；
- GIF／MP4 的可读取性、尺寸和最低帧数；
- schema v2/v3 元数据和 NPZ／HDF5 必要字段；
- Athena 密度、压力、速度、磁场、电流、能量、活动数组和真实网格；
- Type III 主脊严格下降；
- spike catalog 五列、起始窗、严格正频偏和 jet／重联阈值；
- 元数据中的能量漂移和无散残差阈值；
- SHA-256 清单使用 UTF-8、LF 和 POSIX 相对路径；
- 清单覆盖范围、文件存在性和摘要一致性。

校验报告只保存通用目录名，不写入用户目录、用户名或主机名。

---

## 13. 性能模型与电脑资源利用

### 13.1 Athena 正式后端的当前性能事实

- 正式求解器为 Athena C CPU 代码；RTX 4060 不会自动参与 MHD 求解。
- reference 构建使用可移植 `-O3`；performance 构建额外使用本机指令优化，
  只有科学数组等价后才可采用。
- Mac smoke 的 1、2、4、8 ranks 各重复 3 次，中位 wall time 分别为
  1.1487、1.1646、1.1551、1.1958 s。
- 没有任何 MPI 配置比 1 rank 快 20%，所以 Mac 正式基线使用 serial。
- MPI 固定 `OMP_NUM_THREADS=1` 和 `OPENBLAS_NUM_THREADS=1`，避免线程过度订阅。
- BIN 是科学数据源，VTK 仅作视觉交叉检查；动画从 HDF5 桥接生成，避免重复
  保存大量 VTK。
- Athena C 没有分阶段 I/O 计时接口，因此 I/O 占比记录为 `null` 并说明原因，
  不用总 wall time 猜测。

WSL2 计划实测 1、2、4、6、8、12 ranks。只有相对 1 rank 加速超过 20%且
serial/MPI 核心诊断相对差异不超过 \(10^{-6}\)、事件分类一致时，才采用 MPI。
参数扫描必须满足

$$
N_{\rm concurrent}\times N_{\rm ranks/job}
\leq N_{\rm physical\ cores}.
\tag{13.1}
$$

### 13.2 AthenaK／RTX 4060 路线

Athena C 不宣称 CUDA 加速。RTX 4060 当前仅可用于外部 NVENC 媒体编码实验；
正式 GPU 求解路线是迁移同一 problem 到 AthenaK：

1. 先在 WSL2/Linux 文件系统完成 Athena C CPU/MPI 基线；
2. AthenaK 使用 Kokkos CUDA 与 Ada 8.9 架构配置；
3. 在 \(256\times128\) 运行同一初态和物理参数；
4. 比较质量、能量、无散、onset、\(R\)、\(E_z\) 和事件分类；
5. 核心诊断相对差异低于 5%后，才生成 GPU 正式结果。

### 13.R1 Python RMHD 回归后端的性能事实

RMHD 回归求解器具有以下特征：

- NumPy 数组和 `numpy.fft`；
- 单进程、顺序 RK4；
- 每个 RK4 时间步调用四次右端项；
- 静态图和动画由 Matplotlib 在 CPU 上渲染；
- 保存快照时将主要场数组转换为 `float32`，求解阶段保持 `float64`；
- 没有 CUDA、CuPy、JAX、PyTorch 或多 GPU 代码。

因此 RMHD 在 Apple Silicon Mac 和 Windows 工作站运行的是同一 CPU 源码；
Windows 机器中的 NVIDIA GPU 不会被自动调用。

### 13.R2 RMHD 计算量和内存尺度

二维 FFT 的典型复杂度为

$$
\mathcal{O}
\left[
N_xN_y\log(N_xN_y)
\right].
\tag{13.2}
$$

每个时间步有四次 RK4 右端计算，而每次右端又包含多次 FFT，因此总计算量近似随

$$
\mathcal{O}
\left[
N_{\mathrm{step}}
N_xN_y\log(N_xN_y)
\right]
\tag{13.3}
$$

增长。单个 `float64` 场的裸存储为

$$
M_{\mathrm{field}}
=8N_xN_y\ \mathrm{bytes}.
\tag{13.4}
$$

实际峰值内存还包括复数谱数组、RK4 中间状态、导数临时数组、快照和动画 RGB帧，不能只用式 (13.3) 估计。

### 13.R3 RMHD 实用优化

不改物理代码时，优先级建议为：

1. 参数调试使用 `quick` 和 `--animation-format none`；
2. 只对选中的科学案例生成动画；
3. 批量扫描时，每个案例使用独立 seed 和独立输出目录；
4. 数值计算可进程级并行，动画编码应串行或低并发；
5. 保持求解 `float64`，不要为追求速度直接降到 `float32`；
6. 提升网格前先做单任务峰值内存和耗时测量；
7. 不同时启动多个 `both` 动画任务，避免 RGB 帧占用大量内存。

### 13.R4 RMHD 进程级参数扫描

当前仓库没有批量扫描入口。若后续实现，可按 CPU、内存和任务数动态限流：

$$
W
=
\min
\left[
N_{\mathrm{case}},
\max(1,N_{\mathrm{CPU}}-N_{\mathrm{reserve}}),
\max\left(
1,
\left\lfloor
\frac{\alpha_M M_{\mathrm{available}}}
{M_{\mathrm{peak/job}}}
\right\rfloor
\right)
\right].
\tag{13.5}
$$

式中，\(W\) 是 worker 数，\(\alpha_M\) 可取约 \(0.7\) 作为内存安全比例，
\(M_{\mathrm{peak/job}}\) 必须通过单任务实测获得。

未来调度骨架可以写成：

```python
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor


def choose_workers(case_count, available_gib, peak_job_gib):
    by_cpu = max(1, (os.cpu_count() or 2) - 2)
    by_memory = max(1, int(0.7 * available_gib / peak_job_gib))
    return min(case_count, by_cpu, by_memory)


def main():
    cases = build_independent_cases()
    workers = choose_workers(
        case_count=len(cases),
        available_gib=MEASURED_AVAILABLE_GIB,
        peak_job_gib=MEASURED_PEAK_JOB_GIB,
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        list(executor.map(run_one_case, cases))


if __name__ == "__main__":
    main()
```

`spawn` 和 `if __name__ == "__main__"` 对 Windows 尤其重要。每个任务必须写入
唯一目录，不能并发覆盖同一个 `SHA256SUMS.txt`。

若底层数值库自身创建多线程，多进程扫描时还应限制每个进程的内部线程数，避免过度订阅。具体线程数应以本机基准测试为准，不在公开文档中写死。

### 13.R5 RMHD 多线程 FFT

当前 `numpy.fft` 调用没有 worker 参数。若未来改为 `scipy.fft`，可在保持数学接口不变的前提下评估 worker：

```python
from scipy import fft

with fft.set_workers(worker_count):
    transformed = fft.fft2(field)
```

这是后续性能实验，不是当前实现。必须比较数组误差、能量漂移、无散残差和完整收敛结果，不能只比较耗时。

### 13.R6 RMHD 的旧 CuPy 设想

对于远高于 \(96\times96\) 的网格或大量参数案例，NVIDIA GPU 才可能抵消内核启动和主机—显存传输开销。未来 CuPy 迁移需要：

1. 抽象 `numpy`／`cupy` 数组后端；
2. 将 FFT、Poisson 反演、括号和 RK4 中间数组常驻显存；
3. 只在保存快照或绘图时执行 `cupy.asnumpy`；
4. 初期保持 `float64`；
5. 用相同物理配置比较 CPU／GPU 统计量，而不是要求不同随机后端逐位一致；
6. 将射电随机代理保留在 CPU，可维持现有 seed 的 NumPy 可复现性；
7. 不把 Matplotlib 动画渲染误认为 GPU 加速部分。

当前源码未完成这些改动，所以只安装 CuPy 不会使 RTX GPU 自动参与计算。
本项目正式 GPU 路线已经改为 AthenaK；本小节只保留为 RMHD 设计历史，不再是
当前推荐方案。

---

## 14. `solar_simulation` 最小环境与双平台配置

`solar_simulation` 是当前 Athena 工作的最小权威环境，不需要安装完整
`solarphysics_env_latest`。环境文件不含本机 `prefix`：

```yaml
name: solar_simulation
channels:
  - conda-forge
dependencies:
  - python=3.14
  - numpy
  - scipy
  - matplotlib
  - pillow
  - pyvista
  - vtk
  - h5py
  - imageio
  - imageio-ffmpeg
  - ffmpeg
  - pytest
  - ruff
  - openmpi
```

创建或更新：

```bash
conda env create -f environment.solar-simulation.yml
# 已存在时：
conda env update -n solar_simulation \
  -f environment.solar-simulation.yml \
  --prune
```

动画模块使用惰性导入，所以 `--animation-format none` 不会要求媒体编码器在
MHD 求解前加载。项目不引入 CuPy、Numba、Dask 或 mpi4py。

### 14.A Apple Silicon Mac

1. 用 `locks/solar-simulation-osx-arm64.txt` 记录已经解析的精确包构建；
2. Athena reference 使用系统 Clang，serial HLLD 为正式基线；
3. Conda Open MPI 的 wrapper 在 Mac 上通过 `OMPI_CC=/usr/bin/clang`
   指向系统编译器；
4. 当前 smoke 基准不支持默认启用 MPI，正式运行保留 1 rank；
5. 运行前可将 `MPLCONFIGDIR` 和字体缓存指向项目私有临时目录，避免在用户目录
   写缓存，也避免把个人路径写入日志。

### 14.B Windows／WSL2

正式 Windows 路线使用 WSL2 和 Linux 文件系统，不以原生 Windows Athena
二进制为首选。公开硬件描述只保留能力等级：32 GB 内存、i7-13700H 级 CPU、
RTX 4060 级 GPU，不记录品牌、机器名、账户名、序列号或 GPU UUID。

建议步骤：

```bash
# 在 WSL2 shell 中，仓库应位于 Linux 文件系统
conda create -n solar_simulation -c conda-forge \
  python=3.14 numpy scipy matplotlib pillow pyvista vtk h5py \
  imageio imageio-ffmpeg ffmpeg pytest ruff openmpi

conda activate solar_simulation
python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build --mpi --flux hlld
```

`locks/solar-simulation-linux-64.txt` 是解析后的 Linux 锁定输入，但只有在目标
WSL2 电脑完成创建、测试和 MPI 基准后，才能称为该电脑的验证环境。MPI 基准
采用 1、2、4、6、8、12 ranks，各重复 3 次，并固定
`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。

RTX 4060 初期不参与 Athena C 求解；可选 NVENC 编码必须与 CPU 编码做媒体
兼容性检查。AthenaK 迁移标准见第 13.2 节。

### 14.C 环境自检

```bash
python -c "import numpy, scipy, matplotlib, h5py, pyvista, vtk; print('science imports ok')"
python -c "import imageio, imageio_ffmpeg, PIL; print('media imports ok')"
mpirun --version
ffmpeg -version
python -m spike_typeIII_visual.athena doctor
```

环境与锁定文件的 SHA-256 记录在 [`RESULTS.md`](RESULTS.md)。外发日志前必须
删除环境前缀、用户目录、主机名和设备信息。

### 14.R1 Python RMHD 的旧轻量依赖说明

以下内容仅保留给不运行 Athena 的 RMHD 快速回归：

| 用途 | 库 | 是否必需 |
|---|---|---|
| 数组、FFT、随机数 | `numpy` | 必需 |
| 静态图和动画帧 | `matplotlib` | 必需 |
| GIF／MP4 写出和读取 | `imageio` | 必需 |
| PNG／GIF 处理与校验 | `pillow` | 必需 |
| MP4 所需 FFmpeg 后端 | `imageio-ffmpeg` | 仅 MP4／both |
| 单元测试 | `pytest` | 仅开发 |
| 静态检查 | `ruff` | 仅开发 |

下列库不属于当前运行环境，只在完成相应源码改造后用于性能实验：

| 未来实验 | 库 | 当前安装后的效果 |
|---|---|---|
| 多 worker FFT | `scipy` | 不会改变现有 `numpy.fft` 调用 |
| NVIDIA 数组后端 | `cupy` | 不会使当前 NumPy 源码自动使用 GPU |

源码语法要求 Python 3.10 或更高版本，建议新环境使用 Python 3.12。两台电脑使用
同一份源码，不建立 `macos/` 或 `windows/` 副本；平台差异只放在环境和启动命令
中。

当前验证环境已安装 `imageio-ffmpeg`。轻量环境用户应在自己的实际环境名中
安装该包；源码提示中的环境名只是工作区默认示例，不表示必须安装完整科研环境。

### 14.R2 RMHD-only Apple Silicon 环境（可选）

推荐用 Miniforge 创建独立 CPU 环境：

```bash
conda create -n spike_typeiii_mac \
  -c conda-forge \
  python=3.12 \
  numpy \
  matplotlib \
  imageio \
  pillow

conda activate spike_typeiii_mac
```

按用途添加可选依赖：

```bash
# 仅 MP4 或 both
conda install -n spike_typeiii_mac \
  -c conda-forge \
  imageio-ffmpeg

# 仅开发、测试和静态检查
conda install -n spike_typeiii_mac \
  -c conda-forge \
  pytest \
  ruff

# 仅未来 scipy.fft worker 对照
conda install -n spike_typeiii_mac \
  -c conda-forge \
  scipy
```

Mac 使用建议：

- 当前 RMHD 走 NumPy/CPU，适合先完成科学正确性和收敛研究；
- 单次 standard 任务保持默认线程策略；
- 批量扫描才考虑进程级并行，并为系统保留至少 1–2 个逻辑核心；
- 扫描阶段关闭动画，最终只为少量案例编码；
- 不在 README、日志或截图中记录用户名、设备名、序列号和绝对路径。

如果不希望激活环境，可使用：

```bash
conda run -n spike_typeiii_mac \
  python -m spike_typeIII_visual.main \
  --profile quick \
  --animation-format none \
  --output-dir PROJECT_OUTPUT
```

同类命令已在当前 Apple Silicon Mac 上执行；Windows 配置仍为未实测建议。

### 14.R3 RMHD-only 原生 Windows 环境（非正式 Athena 路线）

公开文档只按能力等级描述该设备：

> 64 位 Windows、32 GB 级内存、Intel i7-13700H 级 CPU、
> RTX 4060 级 NVIDIA GPU。

不记录电脑品牌、具体系列、机器名、账户名、序列号、GPU UUID 或本地目录。

#### 14.R3.1 可直接使用的 CPU 环境

在 Miniforge PowerShell 中：

```powershell
conda create -n spike_typeiii_win `
  -c conda-forge `
  python=3.12 `
  numpy `
  matplotlib `
  imageio `
  pillow

conda activate spike_typeiii_win
```

按用途安装：

```powershell
# 仅 MP4 或 both
conda install -n spike_typeiii_win `
  -c conda-forge `
  imageio-ffmpeg

# 仅开发、测试和静态检查
conda install -n spike_typeiii_win `
  -c conda-forge `
  pytest `
  ruff

# 仅未来 scipy.fft worker 对照
conda install -n spike_typeiii_win `
  -c conda-forge `
  scipy
```

当前 \(96\times96\) 基线规模较小，先使用 CPU 版最稳妥。按数组尺度估计，
32 GB 级内存对基线应有余量，但可用分辨率和安全并发数必须由单任务峰值内存实测
确定。本文不承诺 Windows 运行时间或相对 Mac 的加速比，因为本轮没有执行基准
测试。

#### 14.R3.2 历史 CuPy 设想

只有完成第 13.6 节的数组后端改造后，才创建独立 GPU 环境。不要把 GPU 依赖
混入稳定 CPU 环境。

Conda 路线：

```powershell
conda create -n spike_typeiii_gpu `
  -c conda-forge `
  python=3.12 `
  numpy `
  scipy `
  matplotlib `
  imageio `
  pillow `
  imageio-ffmpeg `
  cupy
```

或者，在一个不含其他 CuPy 包的独立环境中，根据本机 NVIDIA 驱动选择官方
CUDA 12.x wheel：

```powershell
python -m pip install "cupy-cuda12x[ctk]"
```

两条路线二选一。不要在同一环境中同时安装 `cupy`、`cupy-cuda11x` 和
`cupy-cuda12x`，以免包冲突。驱动兼容性应以安装时的 CuPy 和 NVIDIA 官方文档
为准。

`nvidia-smi` 可在本机用于确认驱动和显存状态，但不要把完整输出复制到公开
README、问题单或截图中，因为其中可能包含设备标识和进程信息。

#### 14.R3.3 RMHD 性能建议

- 当前代码先用 CPU；不要因为存在 RTX GPU 就假定已经加速；
- 参数扫描使用 `spawn` 进程和独立输出目录；
- 为交互系统保留 CPU 和内存余量；
- 多进程扫描时限制底层线程，避免混合大小核和内部线程造成过度订阅；
- GPU 版本先在中、高分辨率上与 CPU 做科学回归，再决定是否作为默认后端；
- 动画渲染仍主要在 CPU，多个 MP4 编码任务应限流。

### 14.R4 RMHD-only 环境自检

以下命令只检查导入，不运行模拟：

```bash
python -c "import numpy, matplotlib, imageio, PIL; print('core imports ok')"
```

MP4 环境可额外检查：

```bash
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

第二条命令会输出本地路径，公开截图前应遮蔽用户名和用户目录。

### 14.R5 环境记录补充

本地归档可记录：

```bash
conda env export --from-history > environment-history.yml
```

`--from-history` 只记录显式请求的软件包，通常比完整 `conda env export` 更少包含
平台构建细节。外发前仍需人工检查文件内容，确认没有本地 prefix、私有 channel
或用户名。

---

## 15. 局限与下一步研究

### 15.1 当前局限

1. Athena full-MHD 已通过当前离散、收敛和对照验收，但仍缺解析线性撕裂
   增长率基准；
2. MHD 与射电时间默认使用线性 `proxy` 映射；未提供尺度时不能解释为真实秒；
3. jet 条件采样已实现，但正事件只说明条件代理成立，不能证明微观因果；
4. 当前 jet 判据是二维 X 点两侧 \(v_x\) 分位数，不是三维喷流分割；
5. 电子束没有能谱、速度弥散和输运；
6. 射电模型没有波增长、转换效率和传播；
7. Mac serial/MPI 小案例已实测，但 WSL2、RTX 4060 与 AthenaK 尚未实测；
8. 历史输出保留原位；本轮结果写入独立版本化目录。

---

## 16. 参考资料

### 16.1 理论与主题

1. 陈耀：《等离子体物理学基础》，科学出版社，2019，
   ISBN 978-7-03-061388-2。
2. Harris, E. G. (1962), *On a plasma sheath separating regions of oppositely
   directed magnetic field*.
3. Furth, H. P., Killeen, J., & Rosenbluth, M. N. (1963),
   *Finite-resistivity instabilities of a sheet pinch*.
4. Forbes, T. G., & Isenberg, P. A. (1991),
   *A catastrophe mechanism for coronal mass ejections*.
5. Reid, H. A. S., & Ratcliffe, H. (2014),
   *A review of solar type III radio bursts*.
6. Wyper, P. F., DeVore, C. R., & Antiochos, S. K. (2016)，与磁重联喷流及
   breakout-jet 模型相关的研究。

### 16.2 软件与环境
**推荐安装**
- miniforge 安装: <https://conda-forge.org/download/>

**按需安装**
- CuPy 安装说明：<https://docs.cupy.dev/en/stable/install.html>
- SciPy FFT worker 上下文：
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.set_workers.html>
- SciPy FFT 接口：
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html>
- NVIDIA CUDA on Windows 安装指南：
  <https://docs.nvidia.com/cuda/pdf/CUDA_Installation_Guide_Windows.pdf>

---

## 17. 本轮实施过程记录

以下 1–10 是前一轮 RMHD 方法审计；11–20 是本轮 Athena 正式后端实施。

1. 复核 `README.md`、现有 RMHD、射电代理、动画和校验器，确认旧源码的
   Lorentz 括号次序、截止模边界和 jet 条件耦合与方法要求不一致。
2. 在 `MHDConfig` 中加入 physical／legacy 明示约定；physical 使用
   \([j,\psi]\)，legacy 仅用于诊断对照。
3. 将去混叠掩膜改为整数模严格条件
   `|m_x| < N_x/3`、`|m_y| < N_y/3`，并加入截止边界测试。
4. 新增 `JetConfig` 和 `physics/jet.py`：构造双片层掩膜，计算正、负和双向
   \(v_x\) 分位速度，归一化 \(q_J,q_R\)，并以连续 3 个快照确认 onset。
5. 将 onset 后的 RMHD 区间压缩到射电起始窗，只用于事件条件采样；Type III
   主脊的全时段活动调制保持独立。
6. 在高密度连续时间网格上按 \(q_Jq_R\) 不放回抽样；无候选时生成合法空
   catalog，不放宽阈值。
7. 将元数据升级为 schema v2，写入活动数组、映射角色、事件状态、topping
   余量和导出格式；移除持久化绝对路径、用户名和主机名。
8. 新增谱算子、Poisson 反演、能量交换、jet 映射、零事件、动画和清单测试；
   新增空间／时间收敛、符号、无扰动、uniform、打乱活动、阈值与多 seed 对照。
9. 在独立版本化目录生成主结果和实验汇总，保留历史 `outputs/` 根目录不变；
   详细数值和通过／失败项集中记录于 [`RESULTS.md`](RESULTS.md)。
10. 生成 RMHD 阶段教师汇报 PPT，并对当时的 `no_event` 与模型边界作说明。
11. 审计现有 Athena 二进制，确认旧构建未启用 resistivity、viscosity 和 MPI，
    旧阶跃电流片不能作为正式双 Harris 结果。
12. 新增独立 `spike_topping_jet.c` problem generator；用离散矢势写入面心
    双 Harris 磁场和扰动，并按总压平衡设置压强。
13. 修复 Athena C 黏性模块中阻止 viscosity 构建的分量字段错误；完成
    serial/MPI、reference/performance、HLLD/Roe 四类构建检查。
14. 新增 `athena doctor|build|run|ingest|benchmark`，将编译树、BIN/VTK/HST、
    日志和基准全部隔离在 `Local/athena/`。
15. 新增 Athena BIN 解析、MPI 子域拼接和 `MHDFieldSeries`；HDF5 场数组使用
    `(1,ny,nx)` chunk、LZF、float32，时间与标量保持 float64。
16. 将正式重联率改为 O/X 点磁通差变化率，并用 X 点 \(E_z\) 交叉验证；
    jet 诊断限定到 X 点两侧出流窗口。
17. 元数据升级为 schema v3，同时兼容读取 v2；新增 Athena／RMHD 双后端和
    `proxy`／显式 Alfvén 两类时间标定。
18. 在 Mac 执行 smoke、\(256\times128\)、\(512\times256\)、
    \(1024\times512\)、无扰动、MPI、native 和 Roe 对照；详细实测值写入
    [`RESULTS.md`](RESULTS.md)。
19. 以固定 seed 生成 7 个严格 topping 事件、13 张图、4 个 GIF 和 4 个 MP4；
    validator、SHA-256 与同配置无动画重跑均通过。
20. 更新本 README、`RESULTS.md`、PDF 和 12 页教师汇报 PPT；PPT 仅使用
    Athena 新结果，并明确 WSL2／RTX 4060／AthenaK 尚未实测。

---

## 18. 下一阶段：面向 2025-01-24 事件的 2.5D 日冕 Jet

### 18.1 版本、状态与研究边界

本节是在既有 v3 方法上增加的 v4 设计和实现记录，不覆盖第 1–17 节的
双 Harris 基线，也不把低成本软件检查写成科学结果。本文统一采用三种状态：

| 状态 | 含义 |
|---|---|
| **计划中** | 已给出方程、接口或验收条件，但源码尚未完成或尚未进入验证 |
| **已实现** | 源码和测试入口已存在，不能据此声称物理结果成立 |
| **已验证** | 已按预先给出的物理与数值阈值运行并留下可复核记录 |

截至本轮：

- v3 二维周期双 Harris full-MHD 是**已验证基线**，数值见
  [`RESULTS.md`](RESULTS.md)；
- v4 三分量 BIN 读取、schema v4 HDF5、v3 向后兼容、惰性逐快照读取、
  显式 SI 量纲化和去隐私 `EventBundle` 是**已实现并通过单元测试**；
- `spike_topping_solar_jet` 的绝热问题生成器是**已实现**，已通过 Athena
  double／CTU／HLLD／三阶／电阻／黏性构建和极短初始化检查；
- 120 s smoke 已运行但因静态大气压强触底、质量／能量预算超限而
  **未通过**；600 s static/coarse/standard/fine、restart、收敛、开放边界
  反射率、热传导、辐射、AIA 定量图和事件拟合均**尚未验证**；
- 既有 `outputs/`、`RESULTS.md` 数值和教师 PPT 仍只代表 v3，不代表 v4。

本阶段称为**事件约束前向模拟**：观测给出时间、频率、形态和速度的允许范围，
模型在该范围内预测 jet 与射电代理；当前不做参数后验反演，也不把候选
spike 当作拟合真值。

### 18.2 为什么二维电流片 outflow 不等同于日冕 jet

v3 的双 Harris 模型在周期盒内产生 X 点两侧的局地双向
reconnection exhaust。真实日冕 jet 还要求：

1. 重联区上方存在可将物质输运到高日冕的开放或长尺度磁通；
2. 色球—过渡区—日冕具有强密度和温度分层；
3. 底边光球运动持续注入磁自由能，而不是直接写入一股向上的人工速度；
4. 顶边允许质量、磁通和能量离开，侧边反射受到控制；
5. 诊断对象从“X 点附近的 \(v_x\)”变成“沿开放场向上传播的整体结构”。

因此，v4 同时保留两个量：

- **局地重联出流**：用于定位 X 点、重联时刻和短尺度能量转换；
- **全局日冕 jet**：用于测量顶部高度、轴向速度、宽度、质量流率和能量通量。

两者时间顺序应为“重联增强 \(\rightarrow\) 局地 exhaust
\(\rightarrow\) 全局 jet 上升”，但不预设三者必须完全同步。

### 18.3 事件约束及不可混淆的速度

当前审查后的核心射电窗口为

$$
t_{\rm evt}\in
[04{:}48{:}30,\ 04{:}49{:}00]\ {\rm UT},
\qquad
f\in[140.003,\ 464.500]\ {\rm MHz},
\tag{18.1}
$$

动态谱中值采样约为

$$
\Delta t_{\rm radio}=0.401374\ {\rm s}.
\tag{18.2}
$$

基于密度模型推得的
\(v_b\simeq0.044\!-\!0.187c\) 是**电子束速度**，它通过频率漂移约束高能电子
传播；MHD jet 体速度是流体速度 \(\boldsymbol v\) 的轴向分量，通常远低于
电子束速度。两者不得共用同一符号、同一先验或同一验收区间。

当前 `drift_003`、`drift_004`、`drift_007` 和 `drift_010` 保持
`candidate`，不作为严格 topping 的拟合真值。公开配置
[`configs/event_20250124_sanitized.json`](configs/event_20250124_sanitized.json)
只包含逻辑数据 ID 和审查值，不包含原始文件路径、用户名、邮箱、主机名或
下载任务信息。

### 18.4 从三维电阻 MHD 到 2.5D

#### 18.4.1 2.5D 假设

从三维空间取

$$
\frac{\partial}{\partial z}=0,
\qquad
\boldsymbol v=(v_x,v_y,v_z),
\qquad
\boldsymbol B=(B_x,B_y,B_z).
\tag{18.3}
$$

“二维”只描述所有标量在 \(x,y\) 平面变化；\(v_z\) 和 \(B_z\) 仍被完整演化。
这正是 Athena 中 \(N_z=1\) 且保留 `M3/B3` 的 2.5D 模式。

#### 18.4.2 质量与动量方程

质量守恒为

$$
\frac{\partial\rho}{\partial t}
+\boldsymbol\nabla\boldsymbol\cdot(\rho\boldsymbol v)=0.
\tag{18.4}
$$

动量守恒写成

$$
\frac{\partial(\rho\boldsymbol v)}{\partial t}
+\boldsymbol\nabla\boldsymbol\cdot
\left[
\rho\boldsymbol v\boldsymbol v
+\left(p+\frac{B^2}{2\mu_0}\right)\boldsymbol I
-\frac{\boldsymbol B\boldsymbol B}{\mu_0}
-\boldsymbol\Pi
\right]
=\rho\boldsymbol g+\boldsymbol S_{\rm sponge}.
\tag{18.5}
$$

推导步骤是：将单流体动量方程中的 Lorentz 力
\(\boldsymbol J\times\boldsymbol B\) 用

$$
\boldsymbol J\times\boldsymbol B
=-\boldsymbol\nabla\frac{B^2}{2\mu_0}
+\boldsymbol\nabla\boldsymbol\cdot
\frac{\boldsymbol B\boldsymbol B}{\mu_0}
\tag{18.6}
$$

改写成守恒通量，再把黏性应力 \(\boldsymbol\Pi\)、太阳重力
\(\boldsymbol g=-g_\odot(y)\hat{\boldsymbol y}\) 和海绵层源项加入右端。
绝热基线已经包含重力；显式海绵源尚处于计划状态，当前侧边先使用零梯度外流。

#### 18.4.3 感应方程和无散条件

广义 Ohm 定律在当前电阻 MHD 层级简化为

$$
\boldsymbol E=-\boldsymbol v\times\boldsymbol B
+\eta_{\rm SI}\boldsymbol J,
\tag{18.7}
$$

代入 Faraday 定律得到

$$
\frac{\partial\boldsymbol B}{\partial t}
=\boldsymbol\nabla\times
\left(\boldsymbol v\times\boldsymbol B
-\eta\boldsymbol\nabla\times\boldsymbol B\right),
\qquad
\boldsymbol\nabla\boldsymbol\cdot\boldsymbol B=0.
\tag{18.8}
$$

Athena 用角点矢势构造初态面心磁场，再由 constrained transport 更新，因此
正式无散验收必须使用面心场或 HST 的 CT 诊断；由 cell-centered BIN 做有限差分
只能作为可视化检查，不能用于 \(10^{-12}\) 阈值。

#### 18.4.4 三分量电流、涡量与电场

由式 (18.3) 直接展开旋度：

$$
\boldsymbol J
=\frac{1}{\mu_0}\boldsymbol\nabla\times\boldsymbol B
=\frac{1}{\mu_0}
\left(
\frac{\partial B_z}{\partial y},
-\frac{\partial B_z}{\partial x},
\frac{\partial B_y}{\partial x}
-\frac{\partial B_x}{\partial y}
\right),
\tag{18.9}
$$

$$
\boldsymbol\omega
=\boldsymbol\nabla\times\boldsymbol v
=
\left(
\frac{\partial v_z}{\partial y},
-\frac{\partial v_z}{\partial x},
\frac{\partial v_y}{\partial x}
-\frac{\partial v_x}{\partial y}
\right).
\tag{18.10}
$$

特别地，

$$
E_z=-(v_xB_y-v_yB_x)+\eta_{\rm SI}J_z,
\tag{18.11}
$$

而平行电场为

$$
E_\parallel
=\frac{\boldsymbol E\boldsymbol\cdot\boldsymbol B}
{|\boldsymbol B|}
=\eta_{\rm SI}
\frac{\boldsymbol J\boldsymbol\cdot\boldsymbol B}{|\boldsymbol B|}.
\tag{18.12}
$$

理想项不贡献 \(\boldsymbol E\cdot\boldsymbol B\)，所以式 (18.12) 明确指出当前
电阻模型中平行耗散的来源。schema v4 持久化
`velocity_z`、`magnetic_z`、`current_x/y/z` 和 `omega_x/y/z`；
读取 v3 时，新分量被显式置零，而不是伪造历史 2.5D 数据。

#### 18.4.5 三分量总能量与非绝热项

总能量密度必须包含全部三分量：

$$
E=
\frac{p}{\gamma-1}
+\frac{\rho}{2}(v_x^2+v_y^2+v_z^2)
+\frac{1}{2\mu_0}(B_x^2+B_y^2+B_z^2).
\tag{18.13}
$$

能量方程写为

$$
\begin{aligned}
\frac{\partial E}{\partial t}
&+\boldsymbol\nabla\boldsymbol\cdot
\left[
\left(E+p+\frac{B^2}{2\mu_0}\right)\boldsymbol v
-\frac{\boldsymbol B(\boldsymbol v\boldsymbol\cdot\boldsymbol B)}{\mu_0}
\right] \\
&=\rho\boldsymbol g\boldsymbol\cdot\boldsymbol v
+\boldsymbol\nabla\boldsymbol\cdot
\left(\kappa_\parallel\hat{\boldsymbol b}
\hat{\boldsymbol b}\boldsymbol\cdot\boldsymbol\nabla T\right)
-n_en_{\rm H}\Lambda(T)+H_{\rm bg}+Q_\eta+Q_\nu .
\end{aligned}
\tag{18.14}
$$

式中 \(\hat{\boldsymbol b}=\boldsymbol B/|\boldsymbol B|\)，
\(\kappa_\parallel\) 为 Spitzer 型场向热传导系数，\(\Lambda(T)\) 为光学薄
辐射损失函数，\(H_{\rm bg}\) 为维持初始大气的背景加热，\(Q_\eta,Q_\nu\)
分别为 Ohmic 和黏性耗散。实施顺序固定为：

1. 绝热 + 重力；
2. 加入场向热传导；
3. 加入辐射和与初态平衡的背景加热；
4. 完整非绝热联合运行。

当前只完成第 1 步。Athena C 二维传导原实现若以
\(B_x^2+B_y^2\) 归一化方向，2.5D 时必须先改为
\(B_x^2+B_y^2+B_z^2\)，并通过 \(B_z=0\) 回归后才能启用。
热传导会改变 2.5D jet 的温度和形态，不能只作为绘图修饰；相关方法对照见
[2.5D jet 与热传导研究](https://arxiv.org/abs/2005.13647)。

### 18.5 显式 SI 归一化

v4 不允许隐藏物理尺度。取

$$
L_0=10\ {\rm Mm},\qquad
B_0=10\ {\rm G},\qquad
n_{e0}=10^9\ {\rm cm^{-3}},
\tag{18.15}
$$

并令

$$
\rho_0=\mu_e m_p n_{e0},\qquad
v_{\rm A0}=\frac{B_0}{\sqrt{\mu_0\rho_0}},\qquad
t_0=\frac{L_0}{v_{\rm A0}},\qquad
p_0=\frac{B_0^2}{\mu_0}.
\tag{18.16}
$$

式中 \(\mu_e=1.2\) 是每个电子对应的平均质子质量系数。源码
`physics/normalization.py` 从 Mm、G、cm\(^{-3}\) 转为 SI，并把输入尺度和
派生的 \(\rho_0,v_{\rm A0},t_0,p_0\) 写入私有运行 manifest／v4 bridge
单位元数据。若缺少任一尺度，`event` 或 `alfven` 结果不得解释为真实物理量。

### 18.6 色球—过渡区—日冕与离散静水平衡

温度用平滑过渡表示：

$$
T(y)=T_{\rm ch}
+\frac{T_{\rm cor}-T_{\rm ch}}{2}
\left[
1+\tanh\left(\frac{y-y_{\rm tr}}{w_{\rm tr}}\right)
\right],
\tag{18.17}
$$

基线取
\(T_{\rm ch}=2\times10^4\ {\rm K}\)、
\(T_{\rm cor}=1.5\times10^6\ {\rm K}\)。
静水平衡由

$$
\frac{\mathrm dp}{\mathrm dy}=-\rho g_\odot(y),
\qquad
p=\rho\mathcal R T
\tag{18.18}
$$

给出。积分形式为

$$
p(y)=p(0)\exp\left[
-\int_0^y\frac{g_\odot(y')}{\mathcal R T(y')}\,\mathrm dy'
\right],
\qquad
\rho(y)=\frac{p(y)}{\mathcal R T(y)}.
\tag{18.19}
$$

在网格上，推荐用界面中点量构造

$$
p_{j+1}
=p_j\exp\left[
-\frac{g_{j+1/2}\Delta y}
{\mathcal R T_{j+1/2}}
\right],
\qquad
\rho_j=\frac{p_j}{\mathcal R T_j}.
\tag{18.20}
$$

这样压力梯度和重力源使用同一离散信息，可显著减少无驱动大气的伪速度。
当前绝热 C 基线实现了式 (18.19) 的中点积分；在 600 s static 验收前仍标为
“已实现、未验证”。验收要求无驱动最大 Mach 数小于 \(10^{-3}\)。

### 18.7 开放背景场、埋藏偶极、磁零点和导引场

在无量纲坐标中取面外矢势

$$
A_z(x,y)
=-B_{\rm open}x
+m\frac{x}{x^2+(y+d)^2},
\qquad
m=B_{\rm open}(y_{\rm null}+d)^2 .
\tag{18.21}
$$

由

$$
B_x=\frac{\partial A_z}{\partial y},
\qquad
B_y=-\frac{\partial A_z}{\partial x}
\tag{18.22}
$$

得到开放背景场叠加埋藏二维偶极；在 \(x=0,y=y_{\rm null}\) 处
\(B_y=0\)，形成目标磁零点。再加入

$$
B_z=\chi_g B_0,\qquad \chi_g=0.5
\tag{18.23}
$$

作为导引场。源码在角点采样 \(A_z\)，以离散旋度写入面心
\(B_x,B_y\)，所以初态 CT 无散。现有双 Harris 问题也增加
`guide_field_ratio`；\(\chi_g=0\) 必须回归 v3，目标相对差异
小于 \(10^{-10}\)。

### 18.8 底边驱动、顶边 diode 与侧边海绵

驱动只施加在 line-tied 底边，不在日冕内部写入 jet：

$$
v_x(x,0,t)
=-0.02v_{\rm A0}
\tanh\left(\frac{x}{w_d}\right)
\exp\left[-\left(\frac{x}{w_d}\right)^2\right]D(t),
\tag{18.24}
$$

$$
v_z(x,0,t)
=0.01v_{\rm A0}
\tanh\left(\frac{x}{w_d}\right)
\exp\left[-\left(\frac{x}{w_d}\right)^2\right]D(t),
\qquad
v_y(x,0,t)=0.
\tag{18.25}
$$

时间包络 \(D(t)\) 用余弦平滑连接：

$$
D(t)=
\begin{cases}
0, & 0\le t<120\ {\rm s},\\
\frac12[1-\cos\pi(t-120)/60], & 120\le t<180\ {\rm s},\\
1, & 180\le t<300\ {\rm s},\\
\frac12[1+\cos\pi(t-300)/60], & 300\le t<360\ {\rm s},\\
0, & 360\le t\le600\ {\rm s}.
\end{cases}
\tag{18.26}
$$

顶边采用 diode 条件：所有量零梯度外推，但将指向计算域内部的法向动量截为零，

$$
(\rho v_y)_{\rm ghost}
=\max[(\rho v_y)_{\rm last},0].
\tag{18.27}
$$

侧边目标海绵为

$$
\frac{\partial(U-U_0)}{\partial t}
=-\sigma(x)(U-U_0),
\qquad
\sigma(x)=\sigma_{\max}S\!\left(
\frac{|x|-x_s}{x_{\max}-x_s}
\right),
\tag{18.28}
$$

其中 \(S\) 是从 0 到 1 的光滑函数。式 (18.28) 尚未实现；当前侧边使用零梯度
外流，因此开放边界反射率 \(<5\%\) 的验收尚未完成。

### 18.9 背景加异常电阻

正式候选使用

$$
\eta(J)=\eta_{\rm bg}
+(\eta_{\max}-\eta_{\rm bg})
\left\{
1-\exp\left[
-\left(\frac{\max(|J_z|-J_c,0)}{\Delta J}\right)^2
\right]
\right\}.
\tag{18.29}
$$

当 \(|J_z|\le J_c\) 时只保留背景电阻；超过阈值后平滑趋近
\(\eta_{\max}\)，其峰值按有效 Lundquist 数

$$
S_{\rm eff}=\frac{L_{\rm cs}v_{\rm A}}{\eta_{\max}}=5000
\tag{18.30}
$$

换算。`get_eta_user` 已实现式 (18.29) 的无量纲形式；均匀电阻仍必须作为
对照，不能只报告异常电阻案例。

### 18.10 全局 jet 诊断

在每个时刻先由重联 X 点和开放场方向定义 jet 轴
\(\hat{\boldsymbol s}\)，轴向速度为

$$
v_\parallel=\boldsymbol v\boldsymbol\cdot\hat{\boldsymbol s}.
\tag{18.31}
$$

对满足 \(v_\parallel>v_{\rm cut}\)、密度或温度相对背景显著增强且与开放磁通
连通的区域 \(\Omega_J(t)\)，定义：

$$
h_J(t)=\max_{\Omega_J}y,
\qquad
v_J(t)=P_{95}\{v_\parallel:\Omega_J\},
\tag{18.32}
$$

$$
w_J(t)=
\frac{A[\Omega_J(t)]}
{h_J(t)-y_{\rm base}},
\tag{18.33}
$$

$$
\dot M_J(t)
=D_{\rm LOS}
\int_{\Gamma_J}\rho v_\parallel\,\mathrm d\ell,
\tag{18.34}
$$

$$
F_{\rm kin}
=D_{\rm LOS}\int_{\Gamma_J}
\frac12\rho v^2v_\parallel\,\mathrm d\ell,
\qquad
F_{\rm enth}
=D_{\rm LOS}\int_{\Gamma_J}
\frac{\gamma}{\gamma-1}pv_\parallel\,\mathrm d\ell .
\tag{18.35}
$$

式中 \(P_{95}\) 为第 95 百分位，\(\Gamma_J\) 为垂直 jet 轴的截面，
\(D_{\rm LOS}\) 为 2.5D 假定的视线深度。阈值必须由无驱动案例和参数扫描固定，
不能根据希望得到的事件数事后降低。

### 18.11 MHD 密度到高空 Newkirk 延拓

MHD 域到 \(y=200\ {\rm Mm}\)，射电电子束需要延拓到约
\(0.5R_\odot\)。令 \(h_1=160\ {\rm Mm}\)、\(h_2=200\ {\rm Mm}\)，使用
三次 smoothstep

$$
s(\xi)=3\xi^2-2\xi^3,\qquad
\xi=\frac{h-h_1}{h_2-h_1},
\tag{18.36}
$$

平滑连接

$$
n_e(h)
=\{1-s(\xi)\}n_{e,\rm MHD}(h)
+s(\xi)n_{e,\rm 2N}(h),
\tag{18.37}
$$

其中 \(n_{e,\rm 2N}\) 是 2× Newkirk 密度。这样在连接区两端函数及一阶导数
连续。射电频率基线取二次谐波

$$
f_{\rm radio}=2f_{\rm pe}
=\frac{2}{2\pi}
\sqrt{\frac{n_ee^2}{m_e\epsilon_0}},
\tag{18.38}
$$

基频 \(f_{\rm pe}\) 作为敏感性对照。无 jet–重联联合候选时仍输出
`no_event`，不得降低阈值。

### 18.12 合成 AIA 强度

对于光学薄通道 \(\lambda\)，像素强度近似为

$$
I_\lambda(x,y,t)
=\int n_e^2(x,y,z,t)R_\lambda[T(x,y,z,t)]\,\mathrm dz.
\tag{18.39}
$$

2.5D 中假设视线深度为 \(D_{\rm LOS}\)，故

$$
I_\lambda
\approx n_e^2D_{\rm LOS}R_\lambda(T).
\tag{18.40}
$$

生成可与观测比较的图像还需：

1. 使用与观测日期和校准版本一致的 \(R_\lambda(T)\)；
2. 对式 (18.40) 卷积 AIA 通道 PSF；
3. 重采样到 AIA 像素尺度；
4. 按观测 cadence 积分或抽样；
5. 将 \(D_{\rm LOS}\)、曝光时间和响应不确定性列入元数据。

不能用密度图直接命名为 “AIA 171 Å 图”。AIA 定量响应方法参考
[AIA 温度响应标定](https://link.springer.com/article/10.1007/s11207-013-0452-z)。
304 Å 涉及更复杂的形成条件，本阶段只作定性形态对照，不纳入光学薄 DEM
定量结论。

### 18.13 EventBundle、schema v4 与流式数据

事件接口为：

```bash
python -m spike_typeIII_visual.events build \
  --event-config configs/event_20250124_sanitized.json \
  --output Local/events/event_20250124.json
```

输出只保存：

- UTC 核心窗、频率范围和 cadence；
- 已审查 ROI／WCS 状态；
- confirmed／candidate／excluded 漂移段及误差；
- AIA jet 轴、高度、宽度和时距速度的审查值；
- 射电源中心、协方差和逻辑数据 ID；
- 对规范化 JSON 内容计算的 SHA-256。

构建器拒绝邮箱、Windows 盘符路径以及个人 home 绝对路径。原始 FITS、动态谱、
下载脚本和观测目录不进入模拟交付物。

选择 `--time-calibration event` 后，主程序从 EventBundle 的 UTC 端点计算
射电时长，以 bundle cadence 生成时间采样，并采用 bundle 频率上下限；不会
从原始数据目录或隐藏默认值读取这些事件尺度。

schema v4 的全场 HDF5 使用 `(1,ny,nx)` chunk、LZF 和 float32 场数组；
时间、单位和标量诊断保持 float64。`MHDFieldDataset` 以 context manager
逐快照读取，动画也应逐帧编码。NPZ 从 v4 起只保存射电数组和小型诊断，不再
复制完整 MHD 历史。v3 bridge 仍可读取，缺失的 2.5D 分量明确补零。

### 18.14 运行接口与档位

环境和配置检查：

```bash
conda activate solar_simulation
python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build \
  --problem spike_topping_solar_jet
```

正式运行接口：

```bash
python -m spike_typeIII_visual.athena run \
  --binary Local/athena/build/solar_jet_reference_serial/bin/athena \
  --case-config configs/athena_2p5d_solar_jet.yaml \
  --profile jet-standard \
  --run-id athena_2p5d_solar_jet_seed20260726
```

射电事件标定接口：

```bash
python -m spike_typeIII_visual.main \
  --mhd-backend athena \
  --athena-dataset Local/athena/runs/CASE/bridge_v4.h5 \
  --time-calibration event \
  --event-bundle Local/events/event_20250124.json
```

运行档位为：

| 档位 | 网格 | 物理时长 | 当前状态 |
|---|---:|---:|---|
| `jet-smoke` | \(128\times256\) | 120 s | 已运行；静态稳定性未通过 |
| `jet-static` | \(256\times512\) | 600 s | 已配置，尚未运行 |
| `jet-coarse` | \(256\times512\) | 600 s | 已配置，尚未运行 |
| `jet-standard` | \(512\times1024\) | 600 s | 已配置，尚未运行 |
| `jet-fine` | \(1024\times2048\) | 600 s | 已配置，仅最佳案例使用 |

常规输出为 HST 0.25 s、BIN 5 s、restart 60 s。确定 onset 后，从 onset
前 30 s 的 restart 重跑到 onset 后 60 s，BIN cadence 改为 0.5 s。
原始 BIN／RST／日志只留在忽略的 `Local/athena/`；公开结果只包含审查后的
bridge、图、表、报告和摘要。

### 18.15 参数扫描与双平台性能

以 seed `20260726` 生成 16 点 Sobol 扫描，参数包括
\(B_0,n_e,T_{\rm cor},B_z/B_0,y_{\rm null}\)、驱动强度和
\(S_{\rm eff}\)。流程固定为：

1. 16 点 coarse；
2. 按预先给出的事件距离和数值质量评分选前三个 standard；
3. 仅最佳且已收敛案例进入 fine；
4. 每个案例都保留无驱动或均匀电阻对照。

Mac 与 WSL2 均固定 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
Mac 先测试 1/2/4/8 ranks，i7-13700H 测试 1/2/4/6/8/12 ranks；
只有中位 wall time 提升超过 20%、标量诊断相对差异不超过 \(10^{-6}\) 且
事件分类一致时才采用 MPI。RTX 4060 在 Athena C 阶段不参与 MHD 求解；
AthenaK 迁移仍是独立的后续验收阶段。

### 18.16 v4 验收门槛

在以下条件全部满足前，README 不把 v4 改为“已验证”，`RESULTS.md` 和教师
PPT 也不引用 v4 科学结论：

1. \(B_z=0\) 回归 v3，相对差异 \(<10^{-10}\)；
2. 面心 CT 归一化 \(\nabla\cdot\boldsymbol B<10^{-12}\)；
3. static 600 s 最大 Mach 数 \(<10^{-3}\)，且不产生 jet；
4. 开放边界反射率 \(<5\%\)；
5. 完整能量预算残差绝对值 \(<2\%\)；
6. restart 与连续运行核心标量差异 \(<10^{-8}\)；
7. standard→fine 的核心 jet／重联指标差异 \(<10\%\)；
8. 流式后处理峰值内存 \(<8\) GB；
9. 所有 spike 满足事件窗、\(q_J,q_R\ge0.6\) 和 \(\Delta f>0\)；
10. schema v4、SHA-256、环境锁定和隐私扫描全部通过。

### 18.17 v4 源码映射

| 方法 | 源码或配置 |
|---|---|
| 2.5D 分层、开放场、底边驱动 | `fluxrope_demo/athena4.2/src/prob/spike_topping_solar_jet.c` |
| v4 Athena 输入 | `fluxrope_demo/athena4.2/tst/2D-mhd/athinput.spike_topping_solar_jet` |
| 真实尺度与物理阶段 | `configs/athena_2p5d_solar_jet.yaml` |
| 三分量 BIN、schema v4、惰性读取 | `spike_typeIII_visual/athena_io.py` |
| 三分量统一场容器 | `spike_typeIII_visual/physics/fields.py` |
| SI 归一化 | `spike_typeIII_visual/physics/normalization.py` |
| AIA 响应卷积前向模型 | `spike_typeIII_visual/physics/synthetic_aia.py` |
| 去隐私事件束与 CLI | `spike_typeIII_visual/events.py` |
| 固定 seed Sobol 案例表 | `spike_typeIII_visual/solar_jet_sweep.py` |
| v2/v3/v4 输出校验 | `spike_typeIII_visual/validate_outputs.py` |

### 18.18 本轮增量过程日志

21. 复核 v3 bridge，确认 BIN 已含 \(v_z,B_z\)，但旧代码在解析后丢弃；
    将三分量速度、磁场、电流和涡量贯通到 `MHDFieldSeries` 与 schema v4。
22. 新增 `MHDFieldDataset` 逐快照读取；v3 bridge 读取时显式零填充新分量，
    保持历史兼容。
23. 为 v3 双 Harris 增加 `guide_field_ratio`，并把 \(B_z\) 纳入总压和总能量。
24. 新增 `spike_topping_solar_jet` 绝热问题：分层、重力、开放场与埋藏偶极、
    导引场、底边汇聚／剪切、顶边 diode 和电流触发电阻。
25. 新增显式 `PhysicalNormalization`、无本地定位信息的 `EventBundle`、
    `event` 时间标定和公开的审查配置模板。
26. 扩展 Athena CLI，增加 solar-jet build、`jet-*` 档位和
    `--case-config`；所有原始产物仍隔离在 `Local/athena/`。
27. Python 全套低成本回归通过；solar-jet Athena 构建通过，并完成
    \(32\times64\)、极短 \(t=0.001\) 的私有初始化检查。该检查只验证软件
    链路，不能替代 120 s smoke 或 600 s 科学验收。
28. 未运行 600 s、Sobol 扫描、standard/fine、非绝热或 AIA 定量流程；
    未更新历史结果和教师 PPT，未提交、未推送。
29. 完成 \(128\times256\)、120 s smoke。边界修正后初态 CT 无散达到
    \(2.09\times10^{-16}\)，但质量变化约 \(-4.88\%\)、能量变化约
    \(-3.86\%\)，压强触及数值下限；因此停止后续高成本运行，详细失败记录
    见 [`RESULTS.md`](RESULTS.md) 第 11 节。

---

## 19. Athena 4.2 + MPI-AMRVAC 双后端增强

本节是在前述 v3/v4 方法之上的增量，不删除 RMHD 推导，也不改变历史 Athena
科学结果。统一配置为
`configs/dual_backend_2p5d_solar_jet.yaml`。本轮只进行构建、解析基准和
\(t_{\rm end}=0.2\) 的无驱动开发 smoke；没有运行 600 s、参数扫描或事件
拟合，因此不能由本节产生新的太阳物理结论。

### 19.1 求解器分工与状态

| 层级 | 数值方法 | 本课题中的职责 | 当前状态 |
|---|---|---|---|
| Athena 4.2 项目补丁 | 固定网格、HLLD、CTU、三阶重构、CT | v3 历史科学基线；v5 参考网格 | **v3 已验证；v5 开发 smoke** |
| MPI-AMRVAC 本地部署快照 | HLLD、AMR、Powell 型无散控制 | AMR、2.5D、非绝热模块的独立交叉检查 | **开发 smoke；物理门槛未通过** |
| Python RMHD | FFT、RK4、严格 \(2/3\) 掩膜 | 快速回归和机制对照 | 已保留 |
| 射电代理 | jet 与重联联合条件 | 严格 topping、`no_event` 逻辑 | 已实现 |

本地 AMRVAC 目录是一个旧版部署快照，不代表最新上游，也不被 Python 构建
流程原位修改。构建器把 vendor 内容复制到
`Local/amrvac/build/<content-hash>/`，排除嵌套 Git、对象、模块、库、二进制
和缓存；公开案例位于 `amrvac/spike_topping_solar_jet/`。原有
`amrvac/solar_reconnection/` 只作 legacy 格式样本，不作为事件模型或科学
比较结果。

### 19.2 统一 2.5D 状态与三分量能量

两个后端都令

$$
\frac{\partial}{\partial z}=0,\qquad
\boldsymbol v=(v_x,v_y,v_z),\qquad
\boldsymbol B=(B_x,B_y,B_z).
\tag{19.1}
$$

空间是二维的，但动量和磁场仍有三个分量。总能量密度为

$$
E=\frac{p}{\gamma-1}
+\frac{m_x^2+m_y^2+m_z^2}{2\rho}
+\frac{B_x^2+B_y^2+B_z^2}{2}.
\tag{19.2}
$$

由式 (19.2) 反解压强：

$$
p=(\gamma-1)
\left[
E-\frac{m_x^2+m_y^2+m_z^2}{2\rho}
-\frac{B_x^2+B_y^2+B_z^2}{2}
\right].
\tag{19.3}
$$

式中 \(m_i=\rho v_i\)。这一步不能把 `ndim=2` 当成“只有两个分量”；
否则会遗漏 \(m_z\) 与 \(B_z\)，系统性高估热压。新的
`amrvac_io.py` 强制 `ndim=2, ndir=3`，并用三分量恢复式 (19.3)。

AMRVAC 开启 `B0field` 时，原生 DAT 中的 \(\boldsymbol b\) 是扰动磁场，
静态背景 \(\boldsymbol B_0\) 不进入保存的扰动能量。因此其正确关系是

$$
E_{\rm dat}
=\frac{p}{\gamma-1}
+\frac{|\boldsymbol m|^2}{2\rho}
+\frac{|\boldsymbol b|^2}{2},
\qquad
\boldsymbol B_{\rm total}=\boldsymbol b+\boldsymbol B_0.
\tag{19.4}
$$

不能在式 (19.4) 中减去 \(|\boldsymbol B_{\rm total}|^2/2\)。读取器要求
隐私安全的 `bridge_sidecar.json` 来重建 \(\boldsymbol B_0\)；缺失 sidecar
或背景场类型未知时拒绝 ingest。对 legacy DAT/VTU 的开发交叉检查表明，
按式 (19.4) 恢复的压强和总磁场与 float32 VTU 的极值在其精度内一致。

### 19.3 静水平衡闭式推导

温度仍采用双平台共享的平滑跃迁：

$$
T(y)=a+\frac{b-a}{2}
\left[1+\tanh\left(\frac{y-y_{\rm tr}}{w}\right)\right],
\tag{19.5}
$$

其中 \(a=T_{\rm ch}\)、\(b=T_{\rm cor}\)。静水平衡

$$
\frac{{\rm d}p}{{\rm d}y}=-\rho g,
\qquad
\rho=\frac{p}{T}
\tag{19.6}
$$

给出

$$
\frac{{\rm d}\ln p}{{\rm d}y}=-\frac{g}{T(y)},\qquad
p(y)=p_0\exp\left[-g\int_0^y\frac{{\rm d}y'}{T(y')}\right].
\tag{19.7}
$$

令

$$
u=\frac{2(y-y_{\rm tr})}{w},\qquad
c=\frac{a-b}{ab},
\tag{19.8}
$$

可取原函数

$$
F(y)=\frac{w}{2}
\begin{cases}
\displaystyle
\frac{u}{b}
+c\left[\ln b+\ln\left(1+\frac{a}{b}e^{-u}\right)\right],
&u\ge0,\\[6pt]
\displaystyle
\frac{u}{a}
+c\left[\ln a+\ln\left(1+\frac{b}{a}e^u\right)\right],
&u<0.
\end{cases}
\tag{19.9}
$$

于是

$$
p(y)=p_0\exp\{-g[F(y)-F(0)]\},\qquad
\rho(y)=\frac{p(y)}{T(y)}.
\tag{19.10}
$$

分段形式避免在高、低温极限计算巨大指数；与高精度数值积分在测试点上的
绝对及相对误差均小于 \(10^{-12}\)。Athena 的内部单元和 ghost 热力学状态
现在共用式 (19.5)、(19.9)、(19.10)，不再逐单元执行中点积分。

### 19.4 背景磁场、边界和热传导

共享矢势为

$$
A_z(x,y)=-B_{\rm open}x
+M\frac{x}{x^2+(y+d)^2},
\qquad
M=B_{\rm open}(y_{\rm null}+d)^2.
\tag{19.11}
$$

由 \(\boldsymbol B_\perp=\boldsymbol\nabla A_z\times\hat{\boldsymbol z}\)
得到

$$
B_x=-\frac{2Mx(y+d)}
{\left[x^2+(y+d)^2\right]^2},
\tag{19.12}
$$

$$
B_y=B_{\rm open}
-M\frac{(y+d)^2-x^2}
{\left[x^2+(y+d)^2\right]^2},
\qquad
B_z=0.5B_{\rm open}.
\tag{19.13}
$$

Athena 内部面心磁场继续由矢势差分生成；底部 ghost 面也从同一矢势延拓，
不再复制首个内部磁场。`jet-static` 会强制
`problem/drive_enabled=0`，即使命令行另给驱动值也不能重新打开。

对场向热流

$$
\boldsymbol q_\parallel
=-\kappa_\parallel
\frac{\boldsymbol B(\boldsymbol B\boldsymbol\cdot\nabla T)}
{B_x^2+B_y^2+B_z^2},
\tag{19.14}
$$

2.5D 中虽有 \(\partial_zT=0\)，分母仍必须包含 \(B_z^2\)。Athena 4.2
项目补丁已修正两个面向通量分支。当前只通过公式—源码与制造表达式单元检查；
尚未用该模块生成 solar-jet 科学结果。

### 19.5 AMR 到统一分析网格的守恒投影

DAT-v5 读取器解析 header、tree、leaf block 和 64 位偏移。若目标层级为
\(\ell_\ast\)，层级 \(\ell\) 的每个叶单元在每个方向复制

$$
r=2^{\ell_\ast-\ell}
\tag{19.15}
$$

次。原单元体积为 \(\Delta V_\ell\)，细单元体积为
\(\Delta V_\ast=\Delta V_\ell/r^{n_{\rm dim}}\)。对任意守恒量 \(U\)，

$$
\sum_{q=1}^{r^{n_{\rm dim}}}U\,\Delta V_\ast
=U\,r^{n_{\rm dim}}
\frac{\Delta V_\ell}{r^{n_{\rm dim}}}
=U\Delta V_\ell.
\tag{19.16}
$$

因此该分片常数投影严格保持质量、动量和总能量积分。读取器还要求 coverage
恰为 1；叶块重叠、缺口、错误版本、非 Cartesian 2.5D 或字段缺失都会报错。
HDF5 场布局统一为 `[time,y,x]`、float32、LZF 和单快照 chunk；时间与诊断
保持 float64。

### 19.6 局部 exhaust、全局 jet 与射电条件

局地重联 exhaust 与日冕全局 jet 不再混称。局地活动取 X 点窗口内沿片层
方向的正、负速度分位数：

$$
v_{\rm bi}(t)=
\min\left[
P_{95}(v_{\rm out}>0),
P_{95}(-v_{\rm out}<0)
\right].
\tag{19.17}
$$

全局 jet 活动取 X 点上方开放通道内向上速度的第 95 百分位：

$$
v_{\rm global}(t)
=P_{95}\{v_{\rm vertical}>0:\Omega_{\rm open}\}.
\tag{19.18}
$$

`MHDGeometry` 提供 `sheet_normal`、`local_outflow_direction` 和
`global_jet_direction`，诊断不再固定为“法向 \(y\)、出流 \(x\)”。
开放场案例中的联合活动使用

$$
q_J(t)=\min[q_{\rm local}(t),q_{\rm global}(t)],
\qquad
q_{\rm joint}(t)=q_J(t)q_R(t).
\tag{19.19}
$$

只有 \(q_J,q_R\) 同时越过预注册阈值，才允许在 Type III 起始窗抽样 spike；
无候选时返回形状 `(0,5)` 的 catalog 和 `no_event`，不降低阈值。

### 19.7 schema v5、接口与验证规则

schema v5 保留 v3/v4 所有场，并新增或规范：

- solver、源码内容哈希、原生格式与版本；
- AMR 层级、analysis grid、守恒投影方法；
- 总磁场／扰动磁场存储约定和能量约定；
- 归一化单位以及三种几何方向；
- 本次运行实际网格和时长，而不是 profile 默认值。

每个新 bridge 还包含逐快照 `development_diagnostics` 组：

| 字段 | 定义与用途 |
|---|---|
| `minimum_density`, `minimum_pressure` | positivity 与下限接近程度 |
| `density_floor_count`, `pressure_floor_count` | 达到配置 floor 的分析网格单元数 |
| `maximum_mach` | \(\max |\boldsymbol v|/\sqrt{\gamma p/\rho}\) |
| `boundary_mass_flux_outward` | 矩形边界上的外向 \(\oint\rho\boldsymbol v\cdot\mathrm d\boldsymbol S\) |
| `boundary_energy_flux_outward` | 理想 MHD 总能流 \(\oint[(E+p+B^2/2)\boldsymbol v-(\boldsymbol v\cdot\boldsymbol B)\boldsymbol B]\cdot\mathrm d\boldsymbol S\) |
| `local_resistivity_minimum/maximum` | 根据本次电流触发模型重算的 \(\eta\) 范围 |
| `global_jet_speed_p95`, `global_jet_activity` | 开放通道向上速度 95% 分位和相对初态归一化活动 |

边界通量采用 analysis grid 的线积分并约定外向为正。它们用于开发诊断；在
源项和求解器原生面通量尚未合并前，不能把它们单独解释为闭合能量预算。

主入口为：

```bash
python -m spike_typeIII_visual.main \
  --mhd-backend amrvac \
  --mhd-dataset Local/amrvac/runs/CASE/bridge.h5 \
  --animation-format none
```

Athena 同样使用 `--mhd-dataset`；旧 `--athena-dataset` 仍是兼容别名。两个
参数同时出现时立即报冲突，别名不能用于 AMRVAC。

Athena 工作流：

```bash
python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build \
  --problem spike_topping_solar_jet --jobs 4
python -m spike_typeIII_visual.athena run \
  --binary Local/athena/build/solar_jet_reference_serial/bin/athena \
  --profile jet-static --run-id CASE
python -m spike_typeIII_visual.athena ingest \
  --run-dir Local/athena/runs/CASE --output Local/athena/runs/CASE/bridge.h5
```

构建以源码指纹缓存；`--rebuild` 强制重建。持久化 stdout/stderr 前会清理
用户名、主机、IP 和绝对 home 路径；run manifest 记录 override 后的真实
网格、时长和源码哈希。

AMRVAC 工作流：

```bash
python -m spike_typeIII_visual.amrvac doctor
python -m spike_typeIII_visual.amrvac build --jobs 4
python -m spike_typeIII_visual.amrvac run \
  --binary Local/amrvac/build/HASH/case/amrvac \
  --run-id CASE
python -m spike_typeIII_visual.amrvac ingest \
  --run-dir Local/amrvac/runs/CASE \
  --output Local/amrvac/runs/CASE/bridge.h5
```

validator 对 Athena 使用 CT 无散标准；对 AMRVAC 检查 positivity、sidecar
provenance、dat-v5 和归一化 `divB`。未建立开放边界能量通量预算前，不把
Athena 封闭域能量漂移阈值机械套到 AMRVAC。

### 19.8 Mac 与 WSL2 最小环境

Python 公共环境保持：

```yaml
name: solar_simulation
channels: [conda-forge]
dependencies:
  - python=3.14
  - numpy
  - scipy
  - matplotlib
  - pillow
  - h5py
  - pyvista
  - vtk
  - imageio
  - imageio-ffmpeg
  - ffmpeg
  - pytest
  - ruff
  - openmpi
```

Apple Silicon Mac 使用 Miniforge、Apple Clang、GNU Fortran 和 conda-forge
OpenMPI。WSL2 使用同一个无 `prefix` 环境文件，并在 Linux 文件系统内编译；
建议安装 `build-essential gfortran perl make`。两平台均固定
`OMP_NUM_THREADS=1` 与 `OPENBLAS_NUM_THREADS=1`，总 MPI ranks 不超过分配给
该任务的物理核心。RTX 4060 本阶段只可用于视频编码；Athena C 与当前
AMRVAC smoke 不宣称 GPU 求解加速。

### 19.9 本轮开发验证结果

本轮实测而非科学结论：

| 检查 | Athena 4.2 | MPI-AMRVAC | 判定 |
|---|---:|---:|---|
| 开发网格 | \(128\times256\) | \(64\times128\)，AMR level 2 | 按计划运行 |
| \(t_{\rm end}\) | 0.2 | 0.2 | 按计划运行 |
| 最小 \(\rho\) | 0.4262 | 0.4262 | positivity 通过 |
| 最小 \(p\) | 0.4261 | 0.4262 | positivity 通过 |
| 最大 Mach 数 | 0.1074 | 0.1204 | **未通过 \(10^{-3}\)** |
| 归一化 `divB` 最大值 | \(2.10\times10^{-15}\) | \(2.16\times10^{-2}\) | Athena 通过；AMRVAC 未通过 |
| 场体积平均能量变化 | \(-5.07\times10^{-5}\) | \(-2.02\times10^{-4}\) | 仅开发记录 |
| restart 相对差异最大量级 | \(2.24\times10^{-5}\) | 未执行 | **未通过 \(10^{-8}\)** |

共同初态在统一网格上的 \(\rho,p,B_z\) 高度一致；\(B_x,B_y\) 的最大相对
差异约 2.04% 和 0.79%，来源包括 Athena 面心 CT 离散与 AMRVAC
cell-centered `B0field` 采样。它们可用于定位离散不一致，不能作为演化一致性
证据。

因此本轮结论严格限定为：

1. 两个后端均能构建、短时运行并产出可读原生文件；
2. DAT-v5、三分量压强、总磁场和 schema-v5 软件链路通过开发测试；
3. Athena 的 CT 无散检查通过；
4. 无驱动静态平衡、AMRVAC Powell `divB` 和 Athena restart 等价性未通过；
5. 在修复这些失败前不运行 600 s、不做事件拟合、不更新任何科学结论。

### 19.10 公式—源码—配置—字段映射

| 物理或数值对象 | 实现 | 配置 | schema-v5 字段或属性 |
|---|---|---|---|
| 闭式静水平衡，式 (19.9)–(19.10) | `spike_topping_solar_jet.c`、AMRVAC `mod_usr.t` | `temperature_*`、`gravity_code` | `rho`、`pressure` |
| 矢势开放场，式 (19.11)–(19.13) | 两个 problem/case | `b_open`、`null_height` | `magnetic_x/y/z` |
| 异常电阻 | Athena `get_eta_user` | `CASE=2`、`eta_*` | `provenance_json.resistivity_model` |
| 2.5D 导热投影，式 (19.14) | Athena `conduction.c` | 后续 conduction build | 源码哈希 |
| DAT-v5 与式 (19.4) | `amrvac_io.py` | `bridge_sidecar.json` | `native_format_version=5` |
| AMR 守恒投影，式 (19.15)–(19.16) | `project_dat_to_uniform` | `refine_max_level` | `projection_method` |
| 几何驱动 jet，式 (19.17)–(19.19) | `physics/jet.py` | `diagnostic_geometry` | geometry JSON、活动数组 |
| 公共后端入口 | `main.py` | `--mhd-backend` | `mhd_source` |
| 后端专用校验 | `validate_outputs.py` | metadata backend | validation report |

<div style="break-before: page;"></div>

### 19.11 本轮过程日志

30. 将 Athena 静水平衡从逐单元数值积分改为稳定闭式，并增加 \(10^{-12}\)
    高精度积分对照测试。
31. 修正 2.5D 各向异性热传导分母的 \(B_z^2\)，显式设置 `CASE=2`。
32. AddressSanitizer 发现 `get_eta_user` 在最外 ghost 上越界；改为边界夹紧的
    cell-centered 电流模板后，异常电阻 smoke 不再崩溃。
33. Athena 构建增加源码指纹缓存、`--jobs`、`--rebuild`、真实 override
    manifest 和日志隐私清理；`jet-static` 强制关闭驱动。
34. 新增独立 AMRVAC 案例、只读 vendor 复制构建、doctor/build/run/ingest/
    benchmark CLI 和必需 sidecar。
35. 新增 DAT-v5 header/tree/block 解析、三分量热压恢复、`B0field` 重建、
    AMR 守恒投影以及 DAT/VTU 交叉测试。
36. schema 升级到 v5，同时兼容读取 v3/v4；公共 CLI 新增 `amrvac` 和
    `--mhd-dataset`。
37. jet 诊断改为几何驱动，并区分局地双向 exhaust 与全局向上 jet。
38. Ruff 与 38 项 pytest 全部通过；两个后端开发 smoke 均执行成功。
39. 按预注册门槛记录静态 Mach、AMRVAC `divB` 和 restart 失败，不调参制造
    正结果；没有运行 600 s，没有改写历史正式输出。

---

## 20. 服务器 RMHD 科学生产工作流

状态：**服务器 CPU 与 WSL2 CUDA 正式 RMHD 对照已完成并通过；后续服务器
任务仍由用户按指导书独立操作。**

Windows CUDA 科学生产模块已以增量方式并入现有 RMHD 源码，Athena 4.2、
AMRVAC 和既有 schema 接口保持不变。硬件中性 profile
`rmhd-medium-event`、`rmhd-fine-event`、`rmhd-fine-control` 与相应历史
`cuda-*` profile 的数值参数相同；执行设备由 `--rmhd-engine`、
`--device` 和 `--precision` 独立选择。长任务支持配置哈希约束的原子
checkpoint 与 `--resume`；HDF5 重渲染采用逐快照读取，避免同时载入 event
和 control 的完整场数组。

GridView 字段、CPU 分区基准、环境建立、严格执行顺序、checkpoint 恢复、
科学门槛、隐私检查和三档下载的唯一操作说明见
[`docs/SERVER_OPERATION_GUIDE_CN.md`](docs/SERVER_OPERATION_GUIDE_CN.md)。
本地只能用 `python -m spike_typeIII_visual.server render-scripts` 生成可审核
脚本；该入口不包含 SSH、`sbatch` 或 GridView API。

正式 CPU 和 CUDA 均使用 seed `20260726`、\(1024\times512\) 网格、
float64、相同时间步、耗散和事件门槛。两者均得到 12 个严格 topping 事件，
control 均返回 `no_event`；主要科学标量最大绝对差异不超过
\(1.34\times10^{-15}\)。CUDA 的 RMHD 求解加速为 \(5.20\times\)，包含
4K 渲染的总流程加速为 \(2.03\times\)。详细配置、数值门槛、CPU 分区基准、
媒体规格和结论边界见 [`RESULTS.md`](RESULTS.md) 第 14 节。

本项目完整教师汇报见
[`Spike_Topping_TypeIII_complete_teacher_report.pptx`](Spike_Topping_TypeIII_complete_teacher_report.pptx)，
配套逐页讲稿见
[`Spike_Topping_TypeIII_complete_speaker_script_CN.md`](Spike_Topping_TypeIII_complete_speaker_script_CN.md)。
另提供简约的山东大学青岛校区实景版
[`Spike_Topping_TypeIII_SDU_Qingdao_teacher_report.pptx`](Spike_Topping_TypeIII_SDU_Qingdao_teacher_report.pptx)；
其科学数据、结论与完整教师汇报一致，校园照片来自山东大学（青岛）官网公开图志，
只用于封面、附录转场和结尾页。
太阳物理深色主题版见
[`Spike_Topping_TypeIII_solar_physics_teacher_report.pptx`](Spike_Topping_TypeIII_solar_physics_teacher_report.pptx)；
该版本同样保持 23 页科学内容不变，采用日冕橙红、EUV 青色与深空蓝黑配色，
并在封面、附录转场和结尾使用 NASA/SDO 的真实 AIA 193 Å 与 131 Å 观测图像。
融合山东大学青岛身份与太阳物理主题的版本见
[`Spike_Topping_TypeIII_SDU_Qingdao_solar_physics_teacher_report.pptx`](Spike_Topping_TypeIII_SDU_Qingdao_solar_physics_teacher_report.pptx)；
该版本以山东大学红标识学校身份，以日冕橙和 EUV 青标识学科主题，并在三张关键
页面同时编排山东大学青岛校区实景与 NASA/SDO 真实观测图像。
旧版教师 PPT 和历史结果继续保留，不被本次汇报覆盖。

CUDA 加速仅适用于 Torch RMHD；不得据此声称 Athena 4.2、AMRVAC 或 AthenaK
已实现 GPU 正式求解。新服务器任务仍应使用独立结果目录和配置哈希，禁止在
任务运行中替换源码。

---

## 21. Git 版本控制与本地数据边界

`simulation/` 采用“可复现源码 + 精选交付物”的版本控制策略。Git 跟踪
Python、Athena 4.2、MPI-AMRVAC vendor 快照、项目配置、测试、平台脚本、
本文档、结果审计、讲稿和最终教师汇报 PPT；不使用 Git LFS。

以下内容仅保留在本机，不进入 Git：

- `spike_typeIII_visual/outputs/` 中的 HDF5、NPZ、动画和运行产物；
- Athena、AMRVAC 的二进制、对象文件、DAT/VTU/VTK、运行日志和源码压缩包；
- Mercury 的 MAT/ZIP、人工复核状态和生成图表；
- Windows/WSL2 传输副本、PPT 渲染工作区、检查日志和各种缓存；
- 带机器路径的环境文件、checkpoint 及 `Local/` 运行树。

AMRVAC 以普通目录保存，不含嵌套 `.git`。上游地址、固定提交、许可证和
本地差异记录在
[`amrvac/VENDOR_PROVENANCE.md`](amrvac/VENDOR_PROVENANCE.md)。
历史运行结果仍可在本机用于复核，但 `RESULTS.md` 和对应 SHA-256 报告才是
公开结论的文字依据。提交前必须检查暂存区范围、单文件大小、隐私定位符、
LF 文本格式和被忽略产物，且不得把服务器账户、主机信息或个人绝对路径写入
持久化文件。
