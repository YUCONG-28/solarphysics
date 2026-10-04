# 贡献指南

[中文](CONTRIBUTING.md) | [English](CONTRIBUTING.en.md) | [首页](README.md)

本仓库维护可组合的科学基础模块、少量合成示例和应用适配。
科学计算放在 `Python/solar_toolkit`，界面和任务组织放在 `Apps/solar_apps`，
通用维护工具放在 `tools`。遵守所在目录的 `AGENTS.md`，保留公开接口和组件许可。

## 准备环境

以下命令均从仓库根目录执行。使用 Miniforge 的 `solarphysics_env_latest`；
`solarphysics_env` 只用于明确要求的兼容检查。基础库开发安装：

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pip install -e './Python[dev]'
```

Windows PowerShell 对应命令：

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
& $Conda run -n solarphysics_env_latest python -m pip install -e ".\Python[dev]"
```

选装依赖与科学接口要求见 [库开发说明](Python/CONTRIBUTING.md)。
应用开发使用 Python 3.14，按照 [Apps 开发说明](Apps/docs/development.md)
安装应用依赖。平台锁及重放方式见 [环境说明](environment/README.md)。

## 提交可复现的修改

1. 从最新远端 `main` 创建分支；保留已有未提交内容，只暂存本次相关文件。
2. 为问题准备最小合成输入，明确形状、单位、UTC 时间约定和预期行为。
3. 运行相关测试；修改共享行为时扩大验证范围。说明已运行的检查及其限制。
4. 检查实际暂存内容，再提交并创建 PR。合并、删除分支和正式发布分别需要明确授权。

从根目录执行基础检查；临时目录应为本次运行独用：

```bash
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m pytest Python/tests --basetemp Local/tmp/pytest-library-check
<miniforge-root>/bin/conda run -n solarphysics_env_latest python -m ruff check --config Python/pyproject.toml Python/solar_toolkit Python/tests
<miniforge-root>/bin/conda run -n solarphysics_env_latest python tools/public_source_policy.py --staged
```

相关应用测试、产物构建和独立安装检查分别见
[Apps 开发说明](Apps/docs/development.md) 与 [库开发说明](Python/CONTRIBUTING.md)。
分支、快捷保存和维护工具见 [开发流程](WORKFLOW_README.md)。

## 公开内容边界

提交、问题报告和 PR 只包含通用代码、合成复现材料和必要技术说明。
真实事件配置、观测数据、研究进展、研究结果、个人路径、凭据及原始运行日志
保留在 `Local/` 或仓库外。运行输出也只能写入这两类位置，不能通过符号链接
写回公开源码目录。

公开配置类型可保留；具体科研参数由调用者明确提供。引用现有实现时保留
作者归属和许可证。文档应描述当前代码行为，不把历史检查记录当作当前验证。
