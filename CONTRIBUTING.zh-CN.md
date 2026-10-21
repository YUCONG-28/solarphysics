# 贡献指南

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md) | [首页](README.zh-CN.md)

本仓库维护可组合的科学基础模块、少量合成示例和应用适配。
科学计算放在 `Python/solar_toolkit`，界面和任务组织放在 `Apps/solar_apps`，
通用维护工具放在 `tools`。保留公开接口和 [组件许可](LICENSING.md)。

## 准备环境

以下命令均从仓库根目录执行。使用 Miniforge 的 `solarphysics_env_latest`；
`solarphysics_env` 只用于明确选择的兼容检查。首次使用先按
[首页说明创建环境](README.zh-CN.md#从基础模块开始)，再安装基础库开发依赖：

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pip install -e './Python[dev]'
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
4. 检查实际暂存内容，再提交并创建 PR。维护者在审阅完成且必要 CI 检查通过后合并；
   版本准备和正式发布属于两个独立维护阶段。

从根目录执行基础检查；临时目录应为本次运行独用：

```bash
pytest_tmp="Local/tmp/pytest-library-$(date +%s)-$$"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pytest Python/tests --basetemp "$pytest_tmp"
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m ruff check --config Python/pyproject.toml Python/solar_toolkit Python/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python tools/public_source_policy.py --staged
```

相关应用测试、产物构建和独立安装检查分别见
[Apps 开发说明](Apps/docs/development.md) 与 [库开发说明](Python/CONTRIBUTING.md)。
分支、快捷保存和维护工具见 [开发流程](WORKFLOW_README.md)。

可选的仓库级 pre-commit 检查也通过 Miniforge 执行。首次运行会下载配置的
hook 环境；格式化 hook 可能修改文件，暂存前需审阅差异：

```bash
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pre_commit run --all-files
```

Windows 使用 `& $Conda run -n solarphysics_env_latest`，并为本次测试在
`Local/tmp/` 下指定独用的临时路径。pre-commit 检查用于补充相关测试。

## 公开内容边界

提交、问题报告和 PR 只包含通用代码、合成复现材料和必要技术说明。
真实事件配置、观测数据、研究进展、研究结果、个人路径、凭据及原始运行日志
保留在 `Local/` 或仓库外。运行输出也只能写入这两类位置，不能通过符号链接
写回公开源码目录。

公开配置类型可保留；具体科研参数由调用者明确提供。引用现有实现时保留
作者归属和许可证。文档应描述当前代码行为，不把历史检查记录当作当前验证。
