# AIA Radio Composite 使用说明

`aia-radio-composite` 是保留的本地 Streamlit 适配界面，用于组合已有 API：
AIA 波段图与射电 Gaussian overlay、射电源 ROI 多频曲线，以及同一 UTC 轴的
DART 或 CSO dynamic spectrum。科学读取、拟合和提取由 `solar_toolkit` 提供；
[包内说明](../Apps/solar_apps/frontends/radio/aia_radio_composite/README.md)
列出实现边界。

## 准备与运行

先按 [Apps 安装说明](../Apps/README.md#install) 安装两个源码分区，并执行
`admin init`。使用 Miniforge `solarphysics_env_latest` 和 Python 3.14。
在私有 `Local/configs/paths.local.yaml` 中配置 `apps.allowed_roots`，或通过
`--allowed-roots` 显式传入数据目录。初始化模板的列表为空；未配置时路径访问
不会自动放开。路径授权只允许文件访问，科学文件、UTC、ROI 和分析参数仍由
用户提供。

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

## 输入格式

| 输入 | 格式与处理 |
| --- | --- |
| AIA | FITS 目录；支持 94、131、171、193、211、304、335、1600 Å，通过 `scan_aia_folder` 索引观测 UTC 和精确 HDU；序列按固定计划调用 `read_aia_background`，单时刻调用仍保留 `find_nearest_aia` 接口。 |
| Radio | FITS 根目录，可含频率和偏振子目录；支持 149、164、190、205、223、238 MHz 与 `RR`、`LL`、`RR+LL`。配对、坐标、Gaussian fitting 和质量判定使用现有 API。 |
| DART | 同组 `SpecDataIdB.fits`、`SpecDataVP.fits`、`SpecFrequency.fits`、`SpecTime.fits`；文件名可带前缀。显示源 Stokes I dB，不再次取对数。 |
| CSO | FITS 文件或目录；读取二维/三维 spectrum、`DATE-OBS`/`DATE_OBS`、表扩展 `frequency`/`time`、`POLARIZA` 和 `BUNIT`/`QUANTITY`。UTC 为起始日期加秒偏移，双偏振按控件选择。 |

## 操作与显示契约

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

根主轴与输出时间轴不同；相邻层分别就近不保证根帧与输出帧直接最近，逐层
时差可能累积。界面和元数据分别保留父子时差、相对输出时差及超限标记。
RR/LL 各自先筛选窗口，再严格一对一合成；显示只能重复完整合成对。
详细算法与版本迁移见[时间匹配说明](time_matching.md)。

私有 UI 状态包含路径、主题、波段、偏振、显示控制及确认的 arcsec ROI。
**Reset UI State** 清除该前端的当前字段。旧版在 AIA/Radio 合成 PNG 上选择的
ROI 不会自动转换成射电源 ROI，需要在原始射电图上重新确认。

## 输出格式

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
