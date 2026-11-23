# AIA Radio Composite 使用说明

## 用途与读者

用途：说明这个本地 Streamlit 适配界面组合了哪些已有 API，以及哪些产物是
科学结果、哪些只是显示。

读者：用私有数据运行这个前端并判断合成图能否作为科学依据的用户，以及审阅
文档修改是否守住边界的评审者。

`aia-radio-composite` 是保留的本地 Streamlit 适配界面，用于组合已有 API：
AIA 波段图与射电 Gaussian overlay、射电源 ROI 多频曲线，以及同一 UTC 轴的
DART 或 CSO dynamic spectrum。科学读取、拟合和提取由 `solar_toolkit` 提供；
[包内说明](../Apps/solar_apps/frontends/radio/aia_radio_composite/README.md)
列出实现边界。

本文不支持的单一断言：合成图或显示控件本身构成一次科学测量。
[假设与适用范围](#假设与适用范围) 列出边界，其中 AIA 合成 PNG 仅供预览。

结构见[写作规范](writing-standard.md)，术语见[术语表](terminology.md)。

## 适用范围

覆盖：路径授权与启动参数、四类输入（AIA、Radio、DART、CSO）、九步操作与
显示契约、静态与视频产物。

不覆盖：

- 科学读取、拟合与提取：由 `solar_toolkit` 提供，
  [包内说明](../Apps/solar_apps/frontends/radio/aia_radio_composite/README.md)
  列出实现边界。
- 路径授权之外的访问：未配置时路径访问不会自动放开，见[验证](#验证)。
- 对结果本身的科学判定：本文只描述软件行为。

## 定义与记号

| 记号或字段 | 单位、坐标系 | 含义 |
| --- | --- | --- |
| allowed roots | 目录列表 | 允许文件访问的根目录；Windows 用 `;` 分隔，macOS 用 `:` |
| 图像时间窗口 | UTC，左闭右开 | `[开始 UTC, 结束 UTC)`，先筛选原始图像，再计数和匹配 |
| 参考 UTC | UTC | 浏览与同步标记位置；不改变图像窗口 |
| ROI | HPLN/HPLT arcsec | 在原始射电图上确认的源区域 |
| `raw_sum`、`raw_mean`、`raw_peak` | 原始通道值 | 左轴显示的 ROI 提取量 |
| 带内有限值均值 | 选带流量单位 | 右轴显示的频带均值 |
| 显示归一化 | 拟合峰值百分比 | **Gaussian display** 的等值线级别 |
| FPS | 帧每秒 | 只控制播放速度 |
| stem | 文件名主干 | 静态产物的共同主干；重名追加 `_002`、`_003` |
| `N_rows`、`S_K` | 计数 | 输出行数 `N_rows`，有效帧数最多的一条流 `S_K` |

### 输入格式

| 输入 | 格式与处理 |
| --- | --- |
| AIA | FITS 目录；支持 94、131、171、193、211、304、335、1600 Å，通过 `scan_aia_folder` 索引观测 UTC 和精确 HDU；序列按固定计划调用 `read_aia_background`，单时刻调用仍保留 `find_nearest_aia` 接口。 |
| Radio | FITS 根目录，可含频率和偏振子目录；支持 149、164、190、205、223、238 MHz 与 `RR`、`LL`、`RR+LL`。配对、坐标、Gaussian fitting 和质量判定使用现有 API。 |
| DART | 同组 `SpecDataIdB.fits`、`SpecDataVP.fits`、`SpecFrequency.fits`、`SpecTime.fits`；文件名可带前缀。显示源 Stokes I dB，不再次取对数。 |
| CSO | FITS 文件或目录；读取二维/三维 spectrum、`DATE-OBS`/`DATE_OBS`、表扩展 `frequency`/`time`、`POLARIZA` 和 `BUNIT`/`QUANTITY`。UTC 为起始日期加秒偏移，双偏振按控件选择。 |

## 方法

### 时间匹配与合成

根主轴与输出时间轴不同；相邻层分别就近不保证根帧与输出帧直接最近，逐层
时差可能累积。界面和元数据分别保留父子时差、相对输出时差及超限标记。
RR/LL 各自先筛选窗口，再严格一对一合成；显示只能重复完整合成对。
详细算法与版本迁移见[时间匹配说明](time_matching.md)。

输出行数由有效帧数最多的一条流决定：

```math
N_{\mathrm{rows}} = \left| S_K \right| \tag{1}
```

符号见[定义与记号](#定义与记号)；配对、容差与质量判定的定义在
[时间匹配说明](time_matching.md)。

### 操作与显示契约

1. 在 Sidebar 选择 AIA、Radio、Spectrum 和输出目录。DART 选择目录；CSO
   可以选择单个 FITS。路径通过当前 allowed roots 校验。
2. 选择 AIA 波段、参考 UTC、Radio 频率、偏振和 spectrum 类型，并设置独立的
   图像时间窗口 `[开始 UTC, 结束 UTC)`。窗口先筛选原始图像，再计数和匹配；
   参考 UTC 用于浏览，不改变窗口。AIA 波段按
   选择顺序排列，每行最多三个面板。**Gaussian display** 分别控制中心和
   等值线；等值线按拟合峰值百分比绘制，未启用时不额外画 FWHM 轮廓。
3. **Build top panel** 匹配背景和通过质量控制的射电拟合，并显示原始射电图。
   在原始射电图上使用 box/lasso，再 **Confirm ROI**。ROI 保存为 HPLN/HPLT
   arcsec；AIA 合成 PNG 仅供预览。射电显示百分位不改变 ROI 科学数据。
4. **Load CSO / DART spectrum** 为所选成像频率匹配频带，完整带宽由用户指定。
5. 在 **UTC display windows** 填入数据覆盖的流量和频谱提取区间。时间格式的
   合成示例为 `2000-01-01` 的 `00:00:00–00:00:10`；实际分析使用自己的 UTC。
   默认 ROI 频率为 149、164、190、223、238 MHz，205 MHz 可加入。
   **Extract dual-axis flux** 提取原始通道数据。
6. 左轴显示 ROI `raw_sum`、`raw_mean` 或 `raw_peak`；右轴显示频带内有限值
   均值。**Flux plot layout** 可合并所有频段，或每个频段单独成行。两类数据
   保留各自采样，不插值或重采样。
7. **Spectrum display** 的频段和强度限制只改变显示，不裁剪原始通道，也不
   改变选带流量。流量和频谱用两个请求窗口的 UTC 并集作为共同横轴，在
   **Reference UTC** 位置显示同步标记。流量、频谱显示范围不会扩大图像匹配窗口。
8. **Generate synchronized composite** 生成静态组合图。下载按钮或
   **Save PNG / JSON / ROI CSV / spectrum CSV** 保存对应产品。
9. **Generate MP4 video** 与预览共用固定的分级匹配计划。各独立 AIA 波长和
   Radio 频率先按窗口筛选，再按有效帧数递增排序：最少帧一路为根主轴，最多
   帧的最后一路决定输出行数和真实 UTC。每个相邻层在全覆盖和不倒退约束下
   最小化总绝对时差，最终引用沿父子映射继承，不在绘制时重新查找近邻。
   各非空路区间内有效帧全部使用，允许重复当前帧；空路显示无数据，全部为空
   时无法导出。超出容差仍保留并标注，不因缺谱删除图像帧。FPS 只控制播放
   速度，各路时间保持不变或增加，不进行科学数据插值。

私有 UI 状态包含路径、主题、波段、偏振、显示控制及确认的 arcsec ROI。
**Reset UI State** 清除该前端的当前字段。旧版在 AIA/Radio 合成 PNG 上选择的
ROI 不会自动转换成射电源 ROI，需要在原始射电图上重新确认。

## 假设与适用范围

下列判读边界对应具体步骤：步骤正文保留它在操作中的位置，本节集中说明一次
运行能证明什么、不能证明什么。

- 路径授权只允许文件访问，科学文件、UTC、ROI 和分析参数仍由用户提供。
- 输出和状态应保留在 `Local/` 或仓库外，不放入公开源码目录。
- 图像窗口先筛选原始图像，再计数和匹配；参考 UTC 用于浏览，不改变窗口。
- 等值线按拟合峰值百分比绘制，未启用时不额外画 FWHM 轮廓。
- 射电显示百分位不改变 ROI 科学数据；AIA 合成 PNG 仅供预览。
- 两类数据保留各自采样，不插值或重采样；流量、频谱显示范围不会扩大图像
  匹配窗口。
- 超出容差仍保留并标注，不因缺谱删除图像帧；空路显示无数据，全部为空时
  无法导出。
- FPS 只控制播放速度，各路时间保持不变或增加，不进行科学数据插值。
- 旧版在 AIA/Radio 合成 PNG 上选择的 ROI 不会自动转换成射电源 ROI，需要在
  原始射电图上重新确认。

## 失效模式与不确定度

| 触发 | 检测 | 后果 | 保留记录 |
| --- | --- | --- | --- |
| 未配置 allowed roots，或目标目录不在其中 | allowed roots 校验不通过 | 路径访问不会自动放开 | 启动命令中传入的 roots 列表 |
| 外部输出目录未加入 allowed roots | `--output-dir` 指向的目录不在 allowed roots 内 | 需要把对应目录加入 allowed roots | `--output-dir` 的取值 |
| 某一路在窗口内完全为空 | 该路显示无数据 | 全部为空时无法导出 | 空路面板 |
| 时差超出容差 | 超限标记 | 图像帧仍保留并标注 | 父子时差、相对输出时差与质量标记 |
| 相邻层分别就近 | 逐层时差可能累积 | 根帧与输出帧不保证直接最近 | 界面与元数据中的两类时差 |
| 旧版 PNG 上选择的 ROI | 不会自动转换 | 射电源 ROI 仍需重新确认 | 私有 UI 状态中的 arcsec ROI |
| 频段或强度限制被修改 | 只影响显示 | 不裁剪原始通道，也不改变选带流量 | 原始通道数据与选带流量 |

## 验证

### 准备与运行

先按 [Apps 安装说明](../Apps/README.md#install) 安装两个源码分区，并执行
`admin init`。使用 Miniforge `solarphysics_env_latest` 和 Python 3.14。
在私有 `Local/configs/paths.local.yaml` 中配置 `apps.allowed_roots`，或通过
`--allowed-roots` 显式传入数据目录。初始化模板的列表为空；未配置时路径访问
不会自动放开。

所有命令从仓库根目录执行。下面的路径是占位符，应替换为已存在的私有目录。

Windows：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend aia-radio-composite `
  --allowed-roots "<absolute-data-root>" `
  --aia-dir "<absolute-data-root>\aia" `
  --radio-dir "<absolute-data-root>\radio" `
  --spectrum-path "<absolute-data-root>\dart"
```

macOS：

```bash
./Apps/run.sh frontend aia-radio-composite \
  --allowed-roots /absolute/path/to/data \
  --aia-dir /absolute/path/to/data/aia \
  --radio-dir /absolute/path/to/data/radio \
  --spectrum-path /absolute/path/to/data/dart
```

默认输出在私有运行时的 `outputs/aia_radio_composite` 下，通常为
`Local/outputs/aia_radio_composite`。外部输出可用 `--output-dir` 指定，并把对应
目录加入 allowed roots。多目录列表在 Windows 用 `;` 分隔，在 macOS 用 `:`。
输出和状态应保留在 `Local/` 或仓库外，不放入公开源码目录。

用 `--help` 查看 launcher 参数；`--dry-run --no-browser` 打印解析后的启动命令，
不启动 Streamlit 服务器。

判断一次运行是否按上述契约完成，以下列产物为准；文件名、列名与后缀规则见下表。

### 输出格式

静态保存使用同一 stem 的 PNG、JSON 和两个 CSV。已有名称追加 `_002`、
`_003` 等后缀，保留此前文件。视频保存 MP4 及对应 JSON。

| 文件 | 内容 |
| --- | --- |
| `.png` | AIA/radio 图、双轴流量行和 spectrum |
| `.json` | 请求参数、拟合、频带、通道数、单位、UTC 范围及 SHA-256 |
| `.csv` | ROI 提取值及 provenance/quality 列 |
| `_spectrum_flux.csv` | CSO/DART 选带流量、请求/实际频带、单位和通道数 |
| `.mp4` | 按固定计划的输出 UTC 升序播放的组合视频 |
| `_video.json` | FPS、计划及输入指纹、根/输出身份、逐帧精确引用、时差和质量标记、视频 SHA-256 |

ROI 标准列为：

```text
time, frequency, raw_sum, raw_mean, raw_peak, quality_flag
```

CSV 还保留源字段，如 `obs_time`、`freq_mhz`、偏振、文件路径、coverage 和
`quality_detail`。DART 复用原始通道 narrowband 提取；CSO 对原始通道逐时刻求
有限值均值。所有生成文件是用户的私有分析产物，不随源码提交。

## 代码锚点

| 步骤 | 模块或入口 |
| --- | --- |
| 启动与路径授权 | `Apps/run.sh frontend aia-radio-composite`、`Apps/run.ps1`、`--allowed-roots` |
| AIA 索引与背景 | `scan_aia_folder`、`read_aia_background`、`find_nearest_aia` |
| 射电配对、坐标、拟合与质量判定 | `solar_toolkit` 现有 API |
| 流量与频谱提取 | `solar_toolkit` 现有 API，见[包内说明](../Apps/solar_apps/frontends/radio/aia_radio_composite/README.md) |
| 组合产物 | **Generate synchronized composite**、**Generate MP4 video** |
