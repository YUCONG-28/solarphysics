# MATLAB 水星弓激波识别与拟合

本目录在 Git 中只保存可复用的 MATLAB 源码、测试和数据质量说明。2013 年
12 月的原始 MAT 文件、传输 ZIP、自动生成图表和人工复核结果均为本地研究
数据，由 `.gitignore` 排除。

## 系统要求

- MATLAB R2025b（macOS、Windows 或 Linux）
- 不需要 Optimization Toolbox 等附加工具箱

## 本地数据布局

将 31 个只读 MAT 文件放在源码旁的 `201312_01s/`，或者保留
`201312_01s.zip` 让启动器在首次运行时解压。两个路径都不会进入 Git。

```text
Mercury/
├── README_中文.md
├── run_mercury_bowshock.m
├── bowshock_analysis/          # Git 跟踪的 MATLAB 源码与说明
├── 201312_01s/                 # 本地原始数据，不进入 Git
├── 201312_01s.zip              # 可选本地传输包，不进入 Git
└── results/                    # 本地输出，不进入 Git
```

## 启动

在 MATLAB 中打开 `run_mercury_bowshock.m` 并点击 **Run**，或执行：

```matlab
cd('Mercury 所在目录')
run_mercury_bowshock
```

程序会检查 31 个日文件，为 `bowshock_analysis/` 临时增加 MATLAB 路径，
随后打开人工复核应用。所有新结果写入 `results/`，不会修改原始 MAT 文件。

## 操作与科学边界

- 日期范围为 2013-12-01 至 2013-12-31。
- 紫色虚线为自动预选，绿色实线为已确认，红色点线为已拒绝。
- `|B| > 1000 nT` 的周期性仪器异常及其前后保护窗不进入候选评分。
- 2013-12-27 ICME 作为科学事件保留，不按仪器异常删除。
- 最终拟合只使用人工确认点；自动结果标记为 `ProvisionalAuto`。

详细算法、异常区间和质量审查分别见
`bowshock_analysis/README.md` 与
`bowshock_analysis/DATA_QUALITY_REVIEW.md`。

若本地数据和人工状态文件齐全，可在 `bowshock_analysis/` 中运行：

```matlab
test_mercury_bowshock
```
