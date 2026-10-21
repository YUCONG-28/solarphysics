# Solar Physics Toolkit

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

可按需组合的太阳物理基础代码库。主要内容是 `solar_toolkit` 中的数据读取、
时间匹配、图像处理、几何计算、射电分析和绘图模块。使用者根据自己的输入、
参数和科学问题选择模块，组合分析流程。

少量示例与应用用于展示模块的组合方式。真实任务的观测数据、事件配置、
处理条件和结果验证由使用者自行准备。

| 能力 | 范围 |
| --- | --- |
| 数据与时间 | 显式 FITS/表格读写、UTC/TAI 处理和按时间排列的序列匹配 |
| 太阳图像 | AIA/HMI 归一化、掩码、地图几何及 CME 差分 |
| 射电分析 | 频谱读取、源拟合、重投影、ROI 统计和轨迹 |
| X 射线时间序列 | SXR 读取、平滑、导数及预计算亮温的显示 |
| 模型 | Gaussian 模型；[全球 PFSS](Python/solar_toolkit/modeling/pfss/README.md) 需要另装后端和有效的全球径向边界 |
| 质量诊断 | 规则证据；学习诊断需要显式标签及独立的 [选装依赖](Python/README.md#install) |

[接口索引](Python/docs/FUNCTION_MAP.md) 说明输入、单位和失败行为。
定标方法与物理模型假设由使用者根据自己的数据选择。

## 从基础模块开始

科学库支持 Python 3.10 及以上；本仓库开发使用 Miniforge 的
`solarphysics_env_latest` 环境及 Python 3.14。首次克隆后，将 `<miniforge-root>`
替换为实际 Miniforge 安装目录，从仓库根目录执行。

macOS/Linux：

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" create -n solarphysics_env_latest -c conda-forge python=3.14 pip
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pip install -e ./Python
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
```

Windows PowerShell：

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda create -n solarphysics_env_latest -c conda-forge python=3.14 pip
& $Conda run -n solarphysics_env_latest python -m pip install -e .\Python
& $Conda run -n solarphysics_env_latest python Python/examples/public_api/time_matching_example.py
```

若同名环境已存在，先检查其中的 Python 版本，跳过 `create` 命令。
示例不需要观测数据，会输出：

```text
aia.lev1_euv_12s.2000-01-01T000012Z.171.image_lev1.fits
```

不需要安装应用层即可使用科学库；下面的最近时间匹配返回所选输入的索引：

```python
from datetime import datetime, timedelta, timezone

from solar_toolkit.radio.reprojection import nearest_time_index

target = datetime(2000, 1, 1, tzinfo=timezone.utc)
samples = [target - timedelta(seconds=1), target + timedelta(seconds=2)]
print(nearest_time_index(target, samples))  # 0
```

| 需求 | 入口 |
| --- | --- |
| 查找模块、安装和验证科学库 | [Python 使用说明](Python/README.md) |
| 查找公共接口及兼容入口 | [接口索引](Python/docs/FUNCTION_MAP.md) |
| 从小型确定性示例开始 | [公共 API 示例](Python/examples/public_api/README.md) |
| 组合频谱读取、时间匹配与显示 | [射电示例](Python/examples/radio/README.md) |
| 读取用户提供的 EUVI 图像 | [STEREO 示例](Python/examples/stereo/README.md) |
| 组合 SXR 数据读取与曲线绘制 | [SXR 示例](Python/examples/sxr/README.md) |

示例生成的数据和输出保存在仓库的 `Local/` 或仓库外的用户指定目录。示例中的合成输入只用于
说明接口和检查软件行为，不构成真实观测证据。

## 应用展示

`Apps` 提供科学模块的界面适配和工作流组合。应用层需要 Python 3.14，
并以可编辑方式安装两个源码分区；安装与运行方法见 [应用说明](Apps/README.md)。桌面展示入口：

```bash
./Apps/run.sh frontend app-v1
```

Windows 使用 `Apps\run.ps1 frontend app-v1`。应用展示需要使用者明确提供输入、
配置和允许访问的路径；安装本身不包含真实任务参数或数据。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `Python/` | 可独立安装的科学基础库、测试和小型示例 |
| `Apps/` | 应用展示、界面适配、工作流与测试 |
| `tools/` | 通用代码维护和数据集合清单工具 |
| `environment/` | 按平台记录的依赖锁及重放工具说明 |
| `docs/` | 技术文档索引、时间匹配和射电比较说明 |
| `Local/` | 忽略的私有配置、数据、状态和输出 |

公开仓库只维护代码和必要技术说明。研究笔记、文献选择、科研进度、
真实事件参数、观测数据与分析成果由使用者在 `Local/` 或仓库外管理。

依赖方向为 `solar_apps -> solar_toolkit`。基础库导入时不启动界面、下载数据
或执行分析。参见 [架构](ARCHITECTURE.md)、[贡献指南](CONTRIBUTING.zh-CN.md)
及 [文档索引](docs/README.md)。

## 贡献与维护

修改前阅读 [贡献指南](CONTRIBUTING.zh-CN.md)。问题报告和 PR 使用最小合成输入，
说明预期行为、实际行为和适用环境。安装与测试依据现有项目配置，平台环境锁的
适用范围见 [环境说明](environment/README.md)。

日常分支、提交和分阶段发布的维护说明见 [开发流程](WORKFLOW_README.md)。
已有使用者可查阅 [迁移说明](docs/migration.md) 和 [变更记录](Python/CHANGELOG.md)。

## 许可与引用

自有的根目录文件、`tools/`、`docs/` 和 `environment/` 适用根目录
[MIT 许可](LICENSE)。科学库保留 [Python/LICENSE](Python/LICENSE)。
Apps 包含 MIT 源码、GPL-3.0-only 的 App 1.0 源码及 MPL-2.0 媒体资源；
再分发前请阅读 [组件许可范围](LICENSING.md)。

作者 Y. Li（山东大学）的引用元数据见根目录 [CITATION.cff](CITATION.cff)，
科学库另有 [分区引用信息](Python/CITATION.cff)。引用时注明实际使用的版本或
提交；这些文件不表示项目已获得 DOI。
