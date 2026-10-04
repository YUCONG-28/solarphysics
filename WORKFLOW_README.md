# 仓库维护流程

日常贡献从[贡献指南](CONTRIBUTING.md)开始。本页记录分支、选择性提交、快捷
命令与独立发布工具。所有命令从仓库根目录执行；示例参数应替换为本次任务的值。

## 1. 工作区边界

科学库在 `Python/`，应用适配在 `Apps/`，维护工具在 `tools/`。
公开内容只包括通用代码、测试、合成示例和必要说明。机器配置、观测数据、
真实事件参数、研究进度、结果、个人路径、凭据和原始运行日志留在 `Local/`
或仓库外。运行输出不得通过符号链接写入公开源码目录。

修改前阅读适用的 `AGENTS.md`。已有未提交内容必须保留；忽略规则不代替暂存
检查。代码、安装与测试说明分别见[架构](ARCHITECTURE.md)、
[库开发说明](Python/CONTRIBUTING.md)和[应用开发说明](Apps/docs/development.md)。

## 2. 开始工作

先检查工作区。只有在当前目录干净、且切换不会影响已有工作时，才运行以下
步骤；否则在独立工作目录继续。`--ff-only` 在历史分叉时停止。

```bash
git status --short
git switch main
git pull --ff-only origin main
git switch -c docs/example-change
```

分支名称应描述本次修改。不要把示例分支名复用于不相关的任务。

## 3. 检查、暂存和提交

先运行与改动相关的测试；共享行为变化时扩大验证范围。开发使用 Miniforge
`solarphysics_env_latest`，兼容环境只用于明确要求的比较。测试和构建输出
使用本次运行独用的私有目录。

```bash
git diff
git add -- <path-one> <path-two>
git diff --cached
"<miniforge-root>/bin/conda" run -n solarphysics_env_latest python tools/public_source_policy.py --staged
git commit -m "docs: clarify module composition"
```

Windows 使用 `<miniforge-root>\Scripts\conda.exe` 执行相同的 `run -n` 参数。
公共内容检查读取实际暂存的 Git 对象，支持删除操作与特殊文件名。它检查结构
和内容边界；人工审阅与凭据检查仍需覆盖完整暂存差异。提交说明描述通用行为
和验证结果，不附带研究记录或原始验证日志。

## 4. 推送与 PR

检查后推送当前功能分支，创建面向 `main` 的 PR：

```bash
git push -u origin HEAD
```

PR 应说明具体问题、修改后的行为、兼容性影响、已运行的检查和未验证的部分。
等待当前提交的 CI 通过，并处理审查意见。合并需要明确授权，不使用强制推送
改写已公开的提交。

分支落后时，获取最新远端引用并将 `origin/main` 合入当前功能分支：

```bash
git fetch origin
git merge origin/main
```

逐项解决冲突后重新检查与测试；不要机械选择整边内容。

## 5. 合并后的清理与撤销

确认 PR 已合并后再同步 `main`：

```bash
git switch main
git pull --ff-only origin main
git branch --merged main
```

只有获得明确授权且确认目标分支已合并，才使用 `git branch -d <branch>`。
删除远端分支也需要授权，再执行 `git push origin --delete <branch>`。

取消暂存并保留文件内容：

```bash
git restore --staged -- <path>
```

不要通过硬重置、批量清理或强制删除来处理未确认的本地工作。若凭据已经进入
提交，先停止推送、轮换凭据并单独处理历史；删除当前文件不能清除历史记录。

## 6. 快捷维护命令

这些命令使用已安装的应用维护入口。macOS 示例中的 `./Apps/run.sh` 在
Windows 对应 `Apps\run.ps1`。

| 命令 | 当前行为 |
| --- | --- |
| `tools quick check` | 对科学库运行 `pip check`、编译检查与 Ruff；不代替测试套件。 |
| `tools quick save -m "<message>" -- <paths>` | 暂存指定路径、检查实际暂存内容、显示统计并提交整个暂存区。 |
| `tools quick push` | 推送当前功能分支，并在 `gh` 可用时创建面向 `main` 的 PR；拒绝在 `main` 执行。 |
| `tools quick update -m "<message>" -- <paths>` | 依次执行 `check`、`save`、`push`，任一步失败即停止。 |

`save` 本身不限制分支，也不会等待交互式审阅；使用前按第 2 节创建功能分支，
检查整个暂存区，避免已有无关内容一并提交。公开内容检查失败时，命令会取消
本次指定路径的暂存。`update` 的分支拒绝发生在最后的 `push` 阶段，因此也
必须先确认分支。这些工具不会执行 `git add .` 或强制推送。

## 7. 独立发布工具参考

源代码更新与发布是分开的维护操作。只有明确决定发布时才使用 `tools release`；
普通文档或源码 PR 不需要修改版本、打标签或创建 Release。

```bash
./Apps/run.sh tools release check
./Apps/run.sh tools release run --bump patch --note "<release-note>"
```

`check` 要求当前为干净的 `main`，执行 `git fetch origin` 并核对远端同步与
环境锁状态。`run` 默认为预览，`--bump` 接受 `patch`、`minor`、`major`。
显式添加 `--execute` 才执行版本和发布修改：同步两份 `_version.py`，更新
`Python/CHANGELOG.md` 的 `## Unreleased` 段，提交、创建标签、推送，并在
`gh` 可用时创建 GitHub Release。正式执行前必须已获得发布授权，且相关测试
已在功能分支通过后合并。保留此工具说明不代表任何产品已经发布。
