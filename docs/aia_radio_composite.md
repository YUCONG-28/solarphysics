# AIA Radio Composite 使用说明

`aia-radio-composite` 是本地 Streamlit 科研前端，用于生成同步组合图：

1. 一个或多个无间距 AIA EUV 面板，以及质量控制后的 Radio Gaussian
   contour 和中心；
2. HPLN/HPLT arcsec ROI 多频 Radio intensity 与频谱选带流量双轴图；可将
   多频曲线合并显示，或为每个频段生成一张独立双轴图；
3. 与 Radio 时间同步的 DART 或 CSO（槎山）dynamic spectrum。

前端只负责参数校验、既有科学 API 编排、交互显示和导出。AIA/Radio FITS
读取、Gaussian fitting、ROI mask、DART reader 和 CSO reader 均由
`solar_toolkit` 提供。

## 安装

要求 Python 3.10 以上、Miniforge，以及仓库支持的
`solarphysics_env_latest` 环境。在仓库根目录执行：

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda env update -n solarphysics_env_latest -f .\Apps\environment.miniforge.yml
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[quality-ml]"
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Apps
```

macOS：

```bash
/Users/<user>/miniforge3/bin/conda env update -n solarphysics_env_latest -f Apps/environment.miniforge.yml
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Python[quality-ml]"
/Users/<user>/miniforge3/bin/conda run -n solarphysics_env_latest python -m pip install -e "./Apps"
```

## 运行

Windows：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend aia-radio-composite
```

可以预填目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend aia-radio-composite `
  --aia-dir "D:\data\aia" `
  --radio-dir "D:\data\radio" `
  --spectrum-path "D:\data\dart" `
  --output-dir "D:\results"
```

macOS：

```bash
./Apps/run.sh frontend aia-radio-composite \
  --aia-dir /data/aia \
  --radio-dir /data/radio \
  --spectrum-path /data/dart \
  --output-dir /data/results
```

用 `--help` 查看全部 launcher 参数；用 `--dry-run --no-browser` 检查命令而
不启动服务器。

## 输入数据格式

### AIA

- 输入为包含 AIA FITS 的目录。
- 支持 94、131、171、193、211、304、335 和 1600 Å。
- 前端调用 `scan_aia_folder`、`find_nearest_aia` 和
  `read_aia_background`，按选择的 UTC 找最近帧。

### Radio

- 输入为 Radio FITS 根目录，可以包含频率和偏振子目录。
- 顶部支持 149、164、190、205、223 和 238 MHz。
- 偏振支持 `RR`、`LL` 和 `RR+LL`。
- `RR+LL` 的配对、Radio FITS 读取、坐标处理、Gaussian fitting 和质量
  判定均使用现有 `solar_toolkit` API。

### DART

选择包含一组标准四文件的目录：

- `SpecDataIdB.fits`
- `SpecDataVP.fits`
- `SpecFrequency.fits`
- `SpecTime.fits`

文件名可以有前缀，但必须以上述标准名称结尾。显示数据为源文件中的
Stokes I dB；不会再次取对数。

### CSO（槎山）

可以选择一个 FITS 文件或包含 FITS 的目录。共享 reader 使用：

- 主 HDU 的二维或三维 spectrum；
- `DATE-OBS` / `DATE_OBS`；
- 表扩展中的 `frequency` 和 `time`；
- `POLARIZA`；
- `BUNIT` 或 `QUANTITY`。

UTC 轴按 `DATE-OBS + time 秒` 建立。三维双偏振文件由现有 CSO reader
拆分，前端再选择与控件匹配的 `RR` 或 `LL`。

## 网页操作流程

1. 在 Sidebar 使用各路径字段下方的 **Browse** 按钮选择 AIA、Radio、
   Spectrum 和输出路径。DART 选择目录；CSO 可以选择单个 FITS 或目录。
2. 多选 AIA 波段（按选择顺序、每行最多 3 个面板）、参考 UTC、Radio
   频率、偏振、ROI 模式及 spectrum 类型。
   **Gaussian display** 可分别开关拟合中心和等值线，并用拟合峰值百分比
   （例如 `95%`）设置等值线范围。未选择等值线时不会额外绘制 FWHM
   椭圆或其他拟合轮廓。
   自定义 HPC 范围超出 AIA 观测区域时，扩展画布可选择黑色或白色；多 AIA
   网格只在底行显示 HPLN 坐标、左列显示 HPLT 坐标。
3. 点击 **Build top panel**。顶部显示 AIA 背景和通过既有质量控制的 Radio
   Gaussian overlay；其下方另行显示与参考 UTC 匹配的原始射电 FITS
   强度图。
4. 在原始射电源图（不是 AIA 背景或合成 PNG）上选择 `box` 或 `lasso`，
   点击 **Confirm ROI**。此选择器复用现有 ROI lightcurve 前端，保存内容
   仅为 HPLN/HPLT arcsec，不保存 pixel coordinates。**Radio source
   intensity** 中的低/高百分位控制显示强度（例如 `90%–99%`），不会改变
   ROI 科学数据。
5. 在 Panel 2 点击 **Load CSO / DART spectrum**。系统以每个 ROI 成像
   频率为中心自动匹配频带，完整带宽由用户设置。
6. 在 **UTC display windows** 分别设置流量和频谱的 UTC 起止时间，例如
   `04:48:30–04:49:00`。选择 ROI lightcurve 频率；默认是 149、164、190、
   223、238 MHz，205 MHz 可手动加入。点击 **Extract dual-axis flux**。
7. 双轴图左轴显示各成像频率的 `raw_sum`、`raw_mean` 或 `raw_peak`；右轴
   显示频带内原始频率通道的有限值均值。使用 **Flux plot layout** 可将
   所有频段合并到一张图，或选择 **One chart per frequency**，为每个频段
   分别生成一张只包含对应成像 ROI 流量和频谱流量的双轴图。两类数据保留
   各自 UTC 采样，不插值或重采样。
8. 在 **Spectrum display** 中可限制频谱图显示频段和强度范围；这些控件
   仅改变显示，不裁剪原始频谱通道，也不改变频带流量计算。流量图和频谱图
   使用两个请求窗口的 UTC 并集作为同一横轴范围，并在
   **Reference UTC** 位置绘制相同的垂直虚线。
9. 点击 **Generate synchronized composite** 生成静态组合图。分频显示
   模式会在组合图中保留多个独立流量行。
10. 使用四个下载按钮，或填写输出目录后点击
   **Save PNG / JSON / ROI CSV / spectrum CSV**。
11. 在 **Video export** 设置 FPS，点击 **Generate MP4 video**。第一个
    所选射电频率的真实观测 UTC 构成视频帧序列；每帧重新匹配全部射电频率
    （0.1 秒容差）和全部 AIA 波段（12 秒容差），重绘顶部网格，并让下方
    所有流量图及频谱图中的虚线同步移动。分频模式在每个视频帧中保持独立
    流量行。不完整时刻会跳过，少于 2 个完整帧时明确失败。FPS 只控制播放
    速度，不插值、不重复帧。

路径、主题、波段、偏振、ROI 模式、lightcurve 频率、HPC 显示范围、指标和
最后确认的 arcsec ROI、频谱频带和对应数据源会写入忽略版本控制的
`Local/state`。刷新页面或下次启动时自动恢复；原生文件对话框也会从最近
使用的目录开始。Sidebar 的 **Reset UI State** 可清除这些记录。
旧版在 AIA/Radio 合成 PNG 上确认的 ROI 不会被迁移为射电源 ROI，首次使用
新版时需要在原始射电图上重新确认。

## 输出

每次保存生成同一 stem 的四个文件；若名称已存在，会追加 `_002`、`_003`
等后缀，不覆盖旧结果。

| 文件 | 内容 |
| --- | --- |
| `.png` | AIA/Radio Gaussian、一个或多个双轴流量行、dynamic spectrum 组合图 |
| `.json` | 请求摘要、拟合、频带、通道数、单位、UTC 范围与 SHA-256 |
| `.csv` | ROI lightcurve 标准列与既有提取器的 provenance/quality 列 |
| `_spectrum_flux.csv` | CSO/DART 选带流量、请求/实际频带、单位和通道数 |
| `.mp4` | 动态多 AIA/多射电顶部、一个或多个双轴流量行和动态频谱同步视频 |
| `_video.json` | FPS、真实帧 UTC、逐源匹配时间差、跳帧明细和视频 SHA-256 |

标准 ROI 列为：

```text
time, frequency, raw_sum, raw_mean, raw_peak, quality_flag
```

CSV 还保留 `obs_time`、`freq_mhz`、偏振、文件路径、coverage 和
`quality_detail` 等源字段。DART 直接复用原始通道 narrowband 提取；CSO
对原始通道逐时刻求有限值均值。组合图的所有流量行和频谱 panel 共用两类
数据的 UTC 并集范围，顶部匹配帧时间以竖线标在全部时间轴上。

## 科学实现边界

- Gaussian：`solar_toolkit.radio.gaussian`
- AIA：`solar_toolkit.aia`
- Radio ROI：`solar_toolkit.radio.roi_lightcurve`
- DART：`solar_toolkit.radio.dart_spectrogram`
- CSO：`solar_toolkit.radio.cso`
