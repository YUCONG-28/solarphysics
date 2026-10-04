# Solar Physics Toolkit

[中文](README.md) | [English](README.en.md)

[![CI](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/YUCONG-28/solarphysics/actions/workflows/ci.yml)

可按需组合的太阳物理基础代码库。主要内容是 `solar_toolkit` 中的数据读取、
时间匹配、图像处理、几何计算、射电分析和绘图模块。使用者根据自己的输入、
参数和科学问题选择模块，组合分析流程。

少量示例与应用用于展示模块的组合方式。真实任务的观测数据、事件配置、
处理条件和结果验证由使用者自行准备。

## 从基础模块开始

科学库支持 Python 3.10 及以上；本仓库开发使用 Miniforge 的
`solarphysics_env_latest` 环境。从仓库根目录安装：

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pip install -e ./Python
```

Windows 使用 `<miniforge-root>\Scripts\conda.exe` 执行相同命令。
不需要安装应用层即可使用科学库：

```python
from datetime import datetime, timedelta, timezone

from solar_toolkit.radio.reprojection import nearest_time_index

target = datetime(2000, 1, 1, tzinfo=timezone.utc)
samples = [target - timedelta(seconds=1), target + timedelta(seconds=2)]
index = nearest_time_index(target, samples)
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
安装与运行方法见 [应用说明](Apps/README.md)。桌面展示入口：

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
| `docs/` | 接口、架构与开发说明 |
| `Local/` | 忽略的私有配置、数据、状态和输出 |

公开仓库只维护代码和必要技术说明。研究笔记、文献选择、科研进度、
真实事件参数、观测数据与分析成果由使用者在 `Local/` 或仓库外管理。

依赖方向为 `solar_apps -> solar_toolkit`。基础库导入时不启动界面、下载数据
或执行分析。参见 [架构](ARCHITECTURE.md)、[贡献指南](CONTRIBUTING.md)
及 [文档索引](docs/README.md)。

## 贡献与维护

修改前阅读 [贡献指南](CONTRIBUTING.md)。问题报告和 PR 使用最小合成输入，
说明预期行为、实际行为和适用环境。安装与测试依据现有项目配置，平台环境锁的
适用范围见 [环境说明](environment/README.md)。

日常分支、提交和独立发布工具的维护说明见 [开发流程](WORKFLOW_README.md)。

## 许可与引用

科学库遵循 [MIT 许可](Python/LICENSE)。应用代码及其资源的许可分别见
[Apps/LICENSE](Apps/LICENSE)、[桌面应用许可](Apps/solar_apps/frontends/app_v1/LICENSE.md)
和 [媒体资源声明](Apps/solar_apps/ui/media/NOTICE.txt)。各组件保留自身许可，
仓库整体不适用单一的 MIT 声明。
科学库引用信息见 [CITATION.cff](Python/CITATION.cff)。
