# 磁重联驱动 Jet 的多层数值模拟

云南天文台暑期学校结课汇报讲稿
山东大学（青岛）

---

# 第一部分　逐页讲稿

## 第 1 页　磁重联驱动 Jet 的多层数值模拟

**建议时间：40 秒**

**核心结论**

本项目在可控双电流片中重现了 tearing、磁重联和局部双向 Jet，并形成可量化、可复现的科学基线。

**可直接朗读**

各位老师、同学好。我的汇报主题是磁重联驱动 jet 的多层数值模拟。
研究目的，是在一个可控、可重复的模型中确认 tearing 如何改变磁拓扑，
磁重联如何与局部双向出流共同演化，并给出稳定的 Jet 起始判据。
本次正式结果包括高分辨率演化、Jet 结构诊断、control 对照和跨平台复现，
汇报将依次介绍目的与背景、重要理论、使用方法、已实现内容以及科学结论和未来建议。

**图表指向**

- 左侧标题给出本次工作的主问题。
- 右侧校园照片只承担山东大学青岛校区版式识别。
- 页底给出本次汇报的五部分结构。

**过渡句**

首先说明为什么磁重联和 tearing 是研究 Jet 的重要物理起点。

**必须强调**

- 当前正式结果是二维局部重联出流基线。
- 2.5D 日冕 jet 仍属于下一阶段。

**避免误读**

- 不把二维双向 outflow 直接称为完整日冕 jet。

---

## 第 2 页　为什么研究磁重联驱动 Jet

**建议时间：50 秒**

**核心结论**

磁重联能够释放磁能并产生高速 outflow，tearing 则提供形成 X/O 点和局部 Jet 的自然路径。

**可直接朗读**

日冕 Jet 是磁化等离子体的快速定向流动，其动力学核心之一是磁能如何转化为动能。
磁重联改变磁场连接关系并释放磁张力，能够在 X 点两侧产生方向相反的 outflow。
tearing 不稳定性使电流片自然分裂为磁岛，并形成 X 点和 O 点拓扑，
因此它为研究“电流片如何形成 Jet”提供了清晰机制链。
本次工作从最小可控模型出发，把这一机制转化为可以直接计算和验证的问题。

**图表指向**

- 左侧关键帧：局部双向 Jet 随时间形成。
- 右侧说明：磁重联、tearing、X/O 点和双向 \(v_x\) 之间的关系。

**过渡句**

下面用 RMHD 方程说明怎样以两个标量描述这一平面磁场和流动。

**必须强调**

- Jet 的研究重点是磁拓扑变化、重联活动和双向 outflow 的共同演化。

**避免误读**

- 当前正式模型用于局部重联机制；真实日冕环境的扩展放在讲稿后部说明。

---

## 第 3 页　RMHD 用两个标量描述平面磁场和流动

**建议时间：55 秒**

**核心结论**

RMHD 通过磁通函数 \(\psi\) 和流函数 \(\phi\) 保留磁重联与平面 outflow 的关键动力学。

**可直接朗读**

平面磁场写成 \(\boldsymbol B_\perp=\nabla\psi\times\hat{\boldsymbol z}\)，
平面速度写成 \(\boldsymbol v_\perp=\nabla\phi\times\hat{\boldsymbol z}\)。
电流和涡量分别为 \(j=-\nabla^2\psi\) 与 \(\omega=\nabla^2\phi\)。
磁通方程描述磁力线随流体平流并受电阻扩散，
涡量方程中的 \([j,\psi]\) 表示洛伦兹力驱动。
这里使用能量一致的符号，使理想项只在磁能和动能之间交换能量。

**图表指向**

- 左侧图：双 Harris 平衡磁场与两条电流片。
- 右侧公式：场变量定义、磁通方程和涡量方程。

**过渡句**

有了控制方程之后，下一页说明 tearing 怎样依次形成重联和 Jet。

**必须强调**

- \([j,\psi]\) 的顺序与能量一致性直接相关。
- \(\eta\) 和 \(\nu\) 分别控制磁通扩散与涡量耗散。

**避免误读**

- RMHD 是不可压缩二维近似；完整方程和适用范围见讲稿后部。

---

## 第 4 页　Tearing—重联—Jet 的物理机制与起始判据

**建议时间：50 秒**

**核心结论**

双 Harris 平衡经过 tearing、拓扑重排和磁张力释放，最终形成可量化的持续双向 Jet。

**可直接朗读**

机制链从双 Harris 平衡开始。确定性扰动激发 tearing，电流片随后形成磁岛和
X/O 点；磁通重排使磁张力释放，并在 X 点两侧产生方向相反的 outflow。
为了把这一过程量化，我们在片层和 X 点附近设置诊断窗口，
分别计算正向和负向 \(v_x\) 的 95% 分位。
当 Jet 活动 \(q_J\) 和重联活动 \(q_R\) 同时超过 0.6，
并连续保持三个快照时，记录 Jet onset。

**图表指向**

- 上方五列：平衡、扰动、撕裂、重联和 Jet。
- 下方左侧：Jet onset 的双活动连续门槛。
- 下方右侧：片层局域、方向相反和时间连续三个科学判据。

**过渡句**

下面说明这些方程和判据在程序中怎样形成完整计算链。

**必须强调**

- Jet onset 必须同时满足空间、方向和时间连续性。

**避免误读**

- 单个高速像素或单帧结构不构成 Jet onset。

---

## 第 5 页　使用方法：模型、数值求解与 Jet 诊断

**建议时间：55 秒**

**核心结论**

本次采用双 Harris 初态、FFT 伪谱、RK4 时间推进和 Jet／重联联合诊断方法。

**可直接朗读**

方法从双 Harris 平衡和确定性 tearing 扰动开始。
空间方向使用 FFT 谱导数与 Poisson 反演，非线性 Poisson 括号应用严格
\(2/3\) 去混叠，时间方向用经典 RK4 推进。
正式配置采用 \(1024\times512\) 网格、6400 个时间步、401 个快照和固定 seed。
每个快照自动提取 X/O 点，并在 X 点两侧计算正、负 \(v_x\) 的分位速度，
再用 \(q_J\)、\(q_R\) 连续三帧的联合门槛记录 Jet onset。

**图表指向**

- 上方五个环节：初态、空间、非线性、时间和诊断。
- 下方左侧：正式配置的网格、步数、快照和 seed。
- 下方右侧：Jet 与重联活动的联合识别方法。

**过渡句**

下一页直接展示这套方法得到的 tearing 与磁拓扑演化。

**必须强调**

- 方法页只说明如何计算和识别，不提前给出结果结论。

**避免误读**

- Athena 与 AMRVAC 的状态、环境和数据桥细节集中在讲稿后部。

---

## 第 6 页　已实现内容：Tearing 演化与 X/O 点拓扑形成

**建议时间：60 秒**

**核心结论**

正式计算重现了扰动增长、电流片重排、磁岛形成和 X/O 点拓扑演化。

**可直接朗读**

左侧关键帧显示，初始的双电流片在确定性扰动作用下逐步发生 tearing。
扰动首先改变局部电流分布，随后磁岛增长，X 点和 O 点拓扑变得清晰。
右侧诊断显示磁能、动能和拓扑量随时间协同演化。
这个结果说明，方程本身能够从受控初态产生磁拓扑重排，
并通过磁张力释放为后续 X 点两侧的双向 outflow 提供驱动力。

**图表指向**

- 左侧：从初态到成熟磁岛的 tearing 关键帧。
- 右侧：磁能、动能和磁拓扑随时间的诊断。
- 页底：从电流片重排到双向 outflow 驱动力的物理联系。

**过渡句**

在 X/O 点拓扑形成之后，下一页展示得到的局部持续双向 Jet。

**必须强调**

- tearing 的结果是磁拓扑发生变化，而不只是图像亮度改变。
- X 点和 O 点分别对应重联区域与磁岛中心。

**避免误读**

- 不把单一关键帧解释成完整时间演化；判断依据同时包含诊断曲线。

---

## 第 7 页　已实现内容：局部持续双向 Jet 的形成与诊断

**建议时间：65 秒**

**核心结论**

正式高分辨率计算在 \(\tau=1.18\) 出现持续 jet onset，并形成片层局域的反向 outflow。

**可直接朗读**

左上展示 X 点两侧反向 outflow 的关键帧；中上左图给出片层局域的正向、
负向和双向分位速度，中上右图给出 Jet 与重联活动。
两类活动越过门槛后持续存在，因此在无量纲时间一点一八记录 jet onset。
下方结构图显示速度主要集中在两条片层和 X 点两侧，黄色窗口就是正式诊断区域。
最终最大速度约为 \(0.0693v_A\)，最终磁通差 \(\psi_O-\psi_X\) 约为 0.138。
这些量支持局部双向重联出流已经稳定形成。

**图表指向**

- 左上：局部双向 Jet 关键帧。
- 中上：双向速度与 \(q_J,q_R\)。
- 下方：片层局域的 jet 结构和诊断窗口。
- 右侧：onset、速度和磁通差。

**过渡句**

物理结构清楚以后，还需要确认它不是数值误差或硬件差异造成的。

**必须强调**

- 所有速度均为无量纲量。
- 结论限于局部 reconnection exhaust。

**避免误读**

- 不把 \(0.0693v_A\) 直接换算成真实日冕速度。

---

## 第 8 页　已实现内容：数值可信度与跨平台复现

**建议时间：60 秒**

**核心结论**

无散、能量预算、网格/时间步对照和跨平台复现共同支持 Jet 结果的可信度。

**可直接朗读**

正式计算使用 \(1024\times512\) 网格、6400 个时间步和 401 个快照。
归一化磁场散度 RMS 为 \(3.74\times10^{-14}\)，能量预算最大残差约为
\(3.14\times10^{-8}\)。同一 float64 配置下，服务器 AMD7742 CPU 用时
976.85 秒，RTX 4060 CUDA 用时 187.95 秒，模拟部分加速约 5.20 倍。
两端关键标量差异不超过 \(1.34\times10^{-15}\)，jet onset 和活动分类一致。
网格与时间步对照中的核心指标保持一致趋势，因此结果同时具备守恒性、
分辨率稳定性和硬件独立性。

**图表指向**

- 左侧：数值验收指标。
- 右侧：CPU/CUDA 时间与一致性。
- 页底：独立硬件得到一致科学诊断的意义。

**过渡句**

有了这些可信度证据，下一页集中给出本次模拟得到的科学认识。

**必须强调**

- 性能差异与科学分类是两个问题。
- E74809 只用于 CPU 分区短基准。

**避免误读**

- Athena 和 AMRVAC 的性能与开发验证状态见讲稿后部。

---

## 第 9 页　模拟揭示的 Jet 形成与演化特征

**建议时间：55 秒**

**核心结论**

tearing、磁通重联和持续双向 outflow 在拓扑、时间和空间上形成一致证据链。

**可直接朗读**

首先，磁岛增长和 X/O 点形成表明 tearing 已经重排电流片拓扑。
其次，磁通差达到 0.138，同时 \(q_R\) 增强，说明磁通重联伴随拓扑演化。
Jet onset 出现在 \(\tau=1.18\)，并由 \(q_J\) 与 \(q_R\) 连续三帧共同确定，
表明重联活动与双向出流协同增强。
最终速度场在两条片层和 X 点两侧保持相反方向、局域分布和连续演化。
control 中没有出现对应 Jet onset，进一步支持扰动—重联—Jet 的物理联系。

**图表指向**

- 表格依次给出拓扑演化、重联活动、Jet 起始、Jet 结构和 control 对照。
- 第三列是由数值证据得到的科学认识。

**过渡句**

最后一页总结科学结论，并给出三类后续研究建议。

**必须强调**

- control 对照是机制判断的重要证据。

**避免误读**

- 模型适用范围与未覆盖物理统一放在讲稿后部。

---

## 第 10 页　科学结论与未来工作建议

**建议时间：60 秒**

**核心结论**

本次模拟得到清晰的 tearing—重联—Jet 科学链，并形成三个直接的后续研究方向。

**可直接朗读**

科学结论可以归纳为三点。第一，tearing 形成 X/O 点拓扑并驱动局部双向 Jet。
第二，Jet onset 为 \(\tau=1.18\)，最终最大速度为 \(0.0693v_A\)，
磁通差达到 0.138，结构保持片层局域和连续。
第三，\(q_J\) 与 \(q_R\) 同步增强，control 对照进一步支持
扰动—重联—Jet 的物理联系，整体形成时间和空间上一致的证据链。

未来工作建议集中在三个方向：将模型扩展到重力分层与开放磁场的 2.5D 环境；
加入导引场、热传导、辐射和背景加热；使用 Athena 4.2 与 MPI-AMRVAC
进行独立交叉验证，并开展磁场、密度、驱动和耗散参数扫描。

**图表指向**

- 左侧：本次得到的科学结论。
- 右侧：面向日冕 Jet 的未来工作建议。
- 页底：本次汇报的最终结论。

**收束句**

本次工作的价值，是把 jet 的触发、诊断、验证和计算流程做成可靠基线，
并从模拟中提取出可比较的 Jet 科学量。谢谢大家。

**必须强调**

- 科学结论与未来建议分栏呈现。
- 模型限制、未覆盖物理和开发状态只在讲稿后部回答。

---

# 第二部分　完整项目注释与答辩备查

## 1. 研究问题和模型边界

本项目研究磁重联如何产生局部双向 outflow，以及怎样把这一基线扩展成沿开放磁场
传播的全局日冕 jet。研究对象分为两层：

1. **局部 reconnection exhaust**：X 点附近由磁张力驱动的反向流。
2. **全局 coronal jet**：具有有限高度、宽度、质量流率和能量通量，并向开放日冕传播。

二维不可压缩 RMHD 适合研究第一层。第二层需要可压缩 2.5D full-MHD、
开放边界、重力分层和热物理。

---

## 2. 二维 RMHD 的定义

取 \(\partial/\partial z=0\)，用磁通函数和流函数表示平面磁场与速度：

\[
\boldsymbol B_\perp=\nabla\psi\times\hat{\boldsymbol z},
\qquad
\boldsymbol v_\perp=\nabla\phi\times\hat{\boldsymbol z}.
\]

对应分量为

\[
B_x=\frac{\partial\psi}{\partial y},
\quad
B_y=-\frac{\partial\psi}{\partial x},
\qquad
v_x=\frac{\partial\phi}{\partial y},
\quad
v_y=-\frac{\partial\phi}{\partial x}.
\]

这种表示自动给出

\[
\nabla\cdot\boldsymbol B_\perp=0,
\qquad
\nabla\cdot\boldsymbol v_\perp=0.
\]

项目采用

\[
j=-\nabla^2\psi,
\qquad
\omega=\nabla^2\phi
\]

作为电流和涡量变量。不同教材可能采用相反的涡量符号，因此比较方程时必须同时
检查 \(\boldsymbol v_\perp\) 和 Poisson 括号定义。

二维 Poisson 括号为

\[
[a,b]
=
\frac{\partial a}{\partial x}\frac{\partial b}{\partial y}
-
\frac{\partial a}{\partial y}\frac{\partial b}{\partial x}.
\]

---

## 3. 感应方程

无量纲电阻感应方程为

\[
\frac{\partial\boldsymbol B}{\partial\tau}
=
\nabla\times(\boldsymbol v\times\boldsymbol B)
+\eta\nabla^2\boldsymbol B.
\]

代入磁通函数表示，并取 \(z\) 分量势函数，可得

\[
\boxed{
\frac{\partial\psi}{\partial\tau}
+[\phi,\psi]
=
\eta\nabla^2\psi
}.
\]

左侧第二项表示磁通被不可压缩流平流；右侧表示电阻扩散。

---

## 4. 涡量方程与洛伦兹项

从不可压缩动量方程

\[
\frac{\partial\boldsymbol v}{\partial\tau}
+\boldsymbol v\cdot\nabla\boldsymbol v
=
-\nabla P
+\boldsymbol j\times\boldsymbol B
+\nu\nabla^2\boldsymbol v
\]

取旋度，压力梯度被消去。按本项目符号约定得到

\[
\boxed{
\frac{\partial\omega}{\partial\tau}
+[\phi,\omega]
=
[j,\psi]
+\nu\nabla^2\omega
}.
\]

\([j,\psi]\) 是项目正式结果使用的能量一致洛伦兹项。旧的相反符号形式只保留为
诊断对照，不进入正式 jet 结果。

---

## 5. 能量预算

二维 RMHD 总能量定义为

\[
E
=
\frac12\int_\Omega
\left(
|\nabla\psi|^2+|\nabla\phi|^2
\right)\,\mathrm dA.
\]

分别对应平面磁能和动能。周期边界下，对控制方程分部积分可得

\[
\frac{\mathrm dE}{\mathrm d\tau}
=
-\eta\int_\Omega j^2\,\mathrm dA
-\nu\int_\Omega\omega^2\,\mathrm dA.
\]

理想情况下 \(\eta=\nu=0\)，非线性括号只在磁能和动能之间交换能量，不改变总量。
正式计算除记录总能量变化外，还直接积分耗散项，检查能量预算残差。

---

## 6. 周期双 Harris 初态

计算域为

\[
L_x=4\pi,\qquad L_y=2\pi,
\]

两条片层中心位于

\[
y_1=-L_y/4,\qquad y_2=+L_y/4.
\]

典型平衡场写为

\[
B_x(y)
=
B_0\left[
\tanh\frac{y-y_1}{a}
-
\tanh\frac{y-y_2}{a}
-1
\right],
\]

其中 \(a\) 是片层半宽。磁通函数由

\[
\frac{\partial\psi_0}{\partial y}=B_x(y)
\]

积分得到，并在周期域内统一处理常数和平均模。

确定性 tearing 扰动采用局域高斯包络与周期 \(x\) 模相乘：

\[
\delta\psi
=
A\cos(k_xx)
\left[
\exp\!\left(-\frac{(y-y_1)^2}{w^2}\right)
-
\exp\!\left(-\frac{(y-y_2)^2}{w^2}\right)
\right].
\]

正式基线取 \(A=0.04\)、\(a=0.20\)、\(w=0.45\)。

---

## 7. FFT 伪谱离散

对周期变量

\[
f(x,y)=\sum_{k_x,k_y}\hat f_{k_x,k_y}
\exp[i(k_xx+k_yy)]
\]

有

\[
\widehat{\partial_x f}=ik_x\hat f,
\qquad
\widehat{\partial_y f}=ik_y\hat f,
\qquad
\widehat{\nabla^2 f}=-(k_x^2+k_y^2)\hat f.
\]

Poisson 反演为

\[
\hat\phi_{\boldsymbol k}
=
-\frac{\hat\omega_{\boldsymbol k}}{k_x^2+k_y^2},
\qquad \boldsymbol k\ne0,
\]

零模设为零，用于固定流函数规范。

---

## 8. 严格 \(2/3\) 去混叠

非线性括号在实空间相乘会卷积谱模。为避免高波数折回低波数，项目使用严格掩膜：

\[
|m_x|<N_x/3,
\qquad
|m_y|<N_y/3.
\]

截止边界模不保留。掩膜应用于非线性右端项，并在每个完整 RK4 时间步后再次应用。
去混叠只控制离散卷积误差，不能代替真实电阻或黏性。

---

## 9. RK4 时间推进

将状态记为

\[
\boldsymbol U=(\omega,\psi),
\qquad
\frac{\mathrm d\boldsymbol U}{\mathrm d\tau}
=
\mathcal F(\boldsymbol U).
\]

经典四阶 Runge–Kutta 为

\[
\begin{aligned}
k_1&=\mathcal F(U^n),\\
k_2&=\mathcal F(U^n+\tfrac12\Delta\tau k_1),\\
k_3&=\mathcal F(U^n+\tfrac12\Delta\tau k_2),\\
k_4&=\mathcal F(U^n+\Delta\tau k_3),\\
U^{n+1}&=U^n+\frac{\Delta\tau}{6}
(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

正式 fine 配置使用 \(\Delta\tau=0.00125\)、6400 步，终止时间为 \(\tau=8\)。

---

## 10. Jet 几何窗口

Jet 诊断不使用全域最大速度，而只使用片层与 X 点附近的几何窗口。
片层法向距离满足

\[
d_\perp\le c_Ja,
\qquad c_J=2.
\]

沿 outflow 方向还要求位于 X 点半窗口内。周期域中距离使用最短周期距离，
避免 X 点靠近边界时发生诊断断裂。

---

## 11. 双向速度判据

在诊断窗口内分别计算

\[
u_+(t)=Q_{0.95}\{v_x:v_x>0\},
\]

\[
u_-(t)=Q_{0.95}\{-v_x:v_x<0\}.
\]

双向 jet 速度定义为

\[
u_J(t)=\min[u_+(t),u_-(t)].
\]

取较小值意味着只有正负两侧都出现稳健高速流时，\(u_J\) 才会升高。
这样可以抑制单侧噪声和孤立极值。

---

## 12. Jet 活动与 onset

活动归一化为

\[
q_J(t)
=
\operatorname{clip}
\left[
\frac{u_J(t)-u_J(0)}
{\max u_J-\min u_J},
0,1
\right].
\]

重联活动使用拓扑磁通差

\[
\Delta\psi(t)=\psi_O(t)-\psi_X(t)
\]

的变化率：

\[
R(t)
=
\left|
\frac{\mathrm d\Delta\psi}{\mathrm dt}
\right|,
\qquad
q_R(t)=\operatorname{normalize}[R(t)].
\]

Jet onset 必须满足

\[
q_J\ge0.6,
\qquad
q_R\ge0.6
\]

并连续保持三个快照。门槛不因没有事件而自动降低。

---

## 13. 正式 RMHD 参数

| 参数 | 正式值 |
|---|---:|
| 网格 | \(1024\times512\) |
| 计算域 | \(4\pi\times2\pi\) |
| 片层半宽 | 0.20 |
| 扰动幅度 | 0.04 |
| 扰动宽度 | 0.45 |
| 电阻 \(\eta\) | 0.002 |
| 黏性 \(\nu\) | 0.002 |
| 时间步 | 0.00125 |
| 步数 | 6400 |
| 快照间隔 | 16 步 |
| 快照数量 | 401 |
| Jet 分位数 | 0.95 |
| Jet 门槛 | 0.60 |
| 重联门槛 | 0.60 |
| 连续确认 | 3 个快照 |

---

## 14. 正式结果

正式 CPU 结果记录：

- jet onset：\(\tau=1.18\)；
- 最终最大速度：\(0.0692967v_A\)；
- 最终磁通差：\(\psi_O-\psi_X=0.137996\)；
- 归一化磁场散度 RMS：\(3.7366\times10^{-14}\)；
- 能量预算最大绝对残差：\(3.1365\times10^{-8}\)。

这些值来自正式交付包中的运行元数据和 HDF5 场数据。

---

## 15. CPU 与 CUDA

同一 float64 配置下：

| 平台 | 模拟时间 |
|---|---:|
| AMD7742 CPU | 976.85 s |
| RTX 4060 CUDA | 187.95 s |

CUDA 模拟阶段加速约

\[
\frac{976.85}{187.95}\approx5.20.
\]

关键标量差异不超过 \(1.34\times10^{-15}\)，jet onset 与活动分类一致。
这说明硬件主要改变运行时间，没有改变本次结果分类。

Python RMHD 不应简单使用 `mpirun -np 64`，因为那会启动 64 个相互独立、
重复占用内存的 Python 任务，而不是自动分解一个 FFT 网格。

---

## 16. 数据和可复现性

正式运行采用：

- 固定随机 seed；
- 配置哈希；
- 原子 checkpoint；
- HDF5 逐快照写入；
- JSON 元数据；
- CSV 标量诊断；
- SHA-256 清单；
- 独立 validator。

HDF5 大场数据按快照流式读取，避免一次性把全部三维数组载入内存。
checkpoint 恢复前必须验证网格、耗散、步长和配置哈希一致。

---

## 17. Athena 4.2 的定位

Athena 4.2 用于固定网格 full-MHD：

- double precision；
- CTU 时间积分；
- HLLD Riemann 求解器；
- constrained transport 保持磁场无散；
- 显式电阻和黏性；
- 三分量速度、磁场、密度、压力和总能量。

Athena 的正式构建必须确认 resistivity、viscosity 和问题生成器实际启用。
当前二维双 Harris full-MHD 基线已经验证；真实量纲 2.5D solar-jet 仍在开发。

---

## 18. MPI-AMRVAC 的定位

MPI-AMRVAC 用于：

- 自适应网格 AMR；
- 2.5D 三分量演化；
- 重力分层；
- 场向热传导；
- 光学薄辐射和背景加热；
- 独立求解器交叉验证。

当前本地 AMRVAC 是部署快照，不能宣称代表最新上游版本。
开发 smoke 只说明接口和数据读取可用，不能当作长时科学结果。

---

## 19. 公共 HDF5 数据桥

求解器无关的字段序列统一暴露：

\[
t,\ x,\ y,\ \rho,\ p,
(v_x,v_y,v_z),\ (B_x,B_y,B_z),\ J_z.
\]

元数据同时记录：

- solver 与源码哈希；
- 原生数据格式；
- 网格和 AMR 层级；
- 单位归一化；
- sheet normal；
- outflow 方向；
- 全局 jet 方向；
- 投影或重采样方法。

这样 RMHD、Athena 和 AMRVAC 可以共享 jet 诊断，但仍保留各自的无散和守恒标准。

---

## 20. 2.5D Full-MHD 方程

在 \(\partial/\partial z=0\) 下保留三分量：

\[
\boldsymbol v=(v_x,v_y,v_z),
\qquad
\boldsymbol B=(B_x,B_y,B_z).
\]

质量方程：

\[
\frac{\partial\rho}{\partial t}
+\nabla\cdot(\rho\boldsymbol v)=0.
\]

动量方程：

\[
\frac{\partial(\rho\boldsymbol v)}{\partial t}
+\nabla\cdot
\left[
\rho\boldsymbol v\boldsymbol v
+\left(p+\frac{B^2}{2\mu_0}\right)\boldsymbol I
-\frac{\boldsymbol B\boldsymbol B}{\mu_0}
\right]
=
\rho\boldsymbol g+\nabla\cdot\boldsymbol\Pi.
\]

感应方程：

\[
\frac{\partial\boldsymbol B}{\partial t}
=
\nabla\times
\left(
\boldsymbol v\times\boldsymbol B
-\eta\boldsymbol J
\right),
\qquad
\nabla\cdot\boldsymbol B=0.
\]

总能量密度：

\[
e
=
\frac{p}{\gamma-1}
+\frac12\rho v^2
+\frac{B^2}{2\mu_0}.
\]

能量方程还加入重力功、场向热传导、辐射损失和背景加热。

---

## 21. 重力分层和静水平衡

太阳重力取

\[
g(y)
=
-\frac{GM_\odot}{(R_\odot+y)^2}.
\]

静水平衡满足

\[
\frac{\mathrm dp}{\mathrm dy}=\rho g(y).
\]

结合

\[
p=\frac{\rho k_BT}{\mu m_p}
\]

得到

\[
\frac{\mathrm d\ln p}{\mathrm dy}
=
\frac{\mu m_pg(y)}{k_BT(y)}.
\]

给定色球—过渡区—日冕温度剖面后，可稳定积分得到压力和密度。
无驱动模型必须长期保持低 Mach 数，否则说明初态或边界不平衡。

---

## 22. 开放磁场和底边驱动

下一阶段背景磁场由开放场与埋藏偶极叠加，并通过矢势构造以保持无散。
底边采用 line-tied 条件，只施加缓慢汇聚和面外剪切：

\[
v_{\rm conv}\sim0.02v_A,
\qquad
v_{\rm shear}\sim0.01v_A.
\]

驱动按时间平滑开启和关闭，不能直接设置向上的喷流速度。
顶部和侧边使用 diode／开放边界，并配置海绵层抑制反射。

---

## 23. 热物理

场向热传导热流为

\[
\boldsymbol q_\parallel
=
-\kappa_\parallel
(\hat{\boldsymbol b}\cdot\nabla T)\hat{\boldsymbol b},
\qquad
\hat{\boldsymbol b}=\frac{\boldsymbol B}{|\boldsymbol B|}.
\]

在 2.5D 中，投影分母必须包含

\[
B_x^2+B_y^2+B_z^2.
\]

辐射项近似为

\[
Q_{\rm rad}=-n_en_H\Lambda(T),
\]

背景加热 \(Q_{\rm heat}\) 用于维持初始大气。实现顺序为：

1. 绝热基线；
2. 场向热传导；
3. 辐射和背景加热；
4. 完整非绝热模型。

---

## 24. 全局 Jet 诊断

开放场模型中需要新增：

- jet 前沿高度 \(h_J(t)\)；
- 轴向速度 \(v_J(t)\)；
- 横向宽度 \(w_J(t)\)；
- 质量流率
  \[
  \dot M=\int_A\rho v_\parallel\,\mathrm dA;
  \]
- 动能通量
  \[
  F_K=\int_A\frac12\rho v^2v_\parallel\,\mathrm dA;
  \]
- 焓通量
  \[
  F_H=\int_A\frac{\gamma}{\gamma-1}pv_\parallel\,\mathrm dA;
  \]
- Poynting 通量。

局部 \(q_J\) 与全局 plume 活动必须同时报告，不能用一个标量替代全部结构。

---

## 25. 下一阶段验收

建议验收顺序：

1. \(B_z=0\) 回归二维基线；
2. 初态 \(\nabla\cdot\boldsymbol B\) 达到离散精度；
3. 无驱动长时运行不得产生持续 jet；
4. 开放边界反射率低于设定阈值；
5. restart 与连续运行一致；
6. standard 到 fine 的核心 jet 指标差异低于 10%；
7. Athena 与 AMRVAC 的事件分类一致；
8. 热传导制造解通过后才启用正式非绝热结果。

任一条件失败，都应保留失败记录，不通过降低 jet 门槛制造正结果。

---

## 26. 代码—公式映射

| 内容 | 主要模块 |
|---|---|
| RMHD 网格与时间推进 | `physics/rmhd.py` |
| 场变量恢复 | `physics/fields.py` |
| Jet 几何与 onset | `physics/jet.py` |
| 正式生产运行 | `production.py`、`main.py` |
| Athena 接口 | `athena.py`、`athena_io.py` |
| AMRVAC 接口 | `amrvac.py`、`amrvac_io.py` |
| HDF5 流式数据 | `field_series.py`、相关 IO 模块 |
| 静态图 | `visualization/figures.py` |
| 动画 | `visualization/animations.py` |
| 输出校验 | `validate_outputs.py`、`server_validation.py` |

---

## 27. 可能提问与回答

### 问题 1：为什么先用 RMHD？

RMHD 计算成本较低，适合验证 tearing、重联和双向 outflow 的核心链条，
也便于进行分辨率、时间步和硬件一致性检查。

### 问题 2：为什么不直接做三维？

三维成本更高，而且在二维基线尚未稳定前，很难区分物理效应和数值问题。
研究顺序应先通过二维与 2.5D 验收，再考虑三维 fan–spine、kink 和扭转传播。

### 问题 3：二维 outflow 能叫 jet 吗？

可以称为局部 reconnection jet 或 exhaust，但不能直接称为完整日冕 jet。
汇报中必须明确限定。

### 问题 4：Jet 是否被人工设置？

没有。初态只设置磁场和小幅 tearing 扰动，速度由控制方程演化产生。

### 问题 5：为什么使用 95% 分位而不是最大速度？

最大值容易被单个网格噪声控制。95% 分位对局部高速区更稳健，同时保留主要 outflow。

### 问题 6：为什么取正负速度的较小值？

这样只有两侧都存在相反方向的流时才得到较高 jet 指标，可排除单侧噪声。

### 问题 7：为什么要求连续三个快照？

连续条件排除瞬时越阈值，确保 onset 对输出噪声不敏感。

### 问题 8：能量总量为什么会变化？

模型包含显式电阻和黏性，总能量应按耗散率下降。关键检查是能量预算残差，
而不是要求耗散模型中的总能量完全不变。

### 问题 9：CUDA 是否改变物理结果？

本次 float64 同配置比较中，关键标量和 jet 分类一致。CUDA 主要缩短运行时间。

### 问题 10：为什么不能直接用 64 个 MPI 进程跑 Python RMHD？

当前 Python 求解器没有空间域 MPI 分解。直接启动多个进程只会重复运行和重复占用内存。

### 问题 11：Athena 和 AMRVAC 为什么都需要？

Athena 提供固定网格 CT/HLLD 基线，AMRVAC 提供 AMR 和非绝热扩展。
两个独立后端能减少单一代码实现造成的系统误差。

### 问题 12：下一步最关键的验收是什么？

无驱动重力分层大气必须稳定，开放边界必须低反射，standard 到 fine 的 jet 指标必须收敛。

### 问题 13：什么时候可以称为全局日冕 jet？

当模型在开放磁场中形成具有可测高度、宽度、质量流率和能量通量的上升 plume，
并通过边界、守恒和分辨率验收后，才适合称为全局日冕 jet。

---

## 28. 最终可引用结论

> 在二维双 Harris RMHD 基线中，tearing 重排电流片并形成片层局域、
> 持续、双向的 reconnection outflow。该结果通过无散、能量预算和
> CPU/CUDA 一致性检查，但仍不等同于具有重力分层、开放磁场和热物理的
> 完整日冕 jet。下一阶段应使用 Athena 4.2 与 MPI-AMRVAC 对 2.5D
> 全局 jet 进行独立交叉验证。
