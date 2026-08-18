# 项目保存、Git 与 CLI 标准操作手册

本文档用于规范 `solarphysics` 工作区的日常保存、分支开发、测试、推送和合并流程。默认集成分支为 `main`，推荐通过 Pull Request（PR）合并，不直接向 `main` 提交开发改动。

> 本文中的 Git 命令是操作说明。创建或更新本文档本身不代表已经执行暂存、提交、合并或推送。

## 1. 工作区边界

仓库中的公开内容分为：

- `Python/`：可复用的科学计算代码和测试；
- `Apps/`：应用、CLI、前端和应用测试；
- `Paper/`：文献证据和出版元数据；
- `tools/literature/`：文献目录的检索、验证和发布工具；
- `Local/`：私有运行时状态，只保存在本机。

以下内容不得提交或推送：

- `Local/`、`Local-migration-backup/`；
- `2023/`、`2024/`、`2025/`、`2026/`、`overview/` 中的观测数据；
- `Apps/outputs/`、`Apps/logs/`、`Apps/tmp/`、构建产物和测试缓存；
- `Apps/configs/paths.local.yaml` 等机器专用配置；
- 私人绝对路径、访问令牌、密码、密钥、Cookie 和其他凭据；
- 未经审核的原始观测、批量生成图像、个人资料和运行历史。

提交前即使文件已被 `.gitignore` 忽略，也应主动检查暂存区，不能只依赖忽略规则。

## 2. 开始工作

先切换到仓库根目录，再同步 `main`。`--ff-only` 会在远端历史与本地历史分叉时停止，避免意外生成合并提交。

```bash
cd <solarphysics-repository>
git switch main
git pull --ff-only origin main
git switch -c feature/<简短名称>
```

分支名应简短并说明目的，例如：

```bash
git switch -c feature/improve-radio-cli
git switch -c docs/update-workflow
git switch -c fix/aia-time-matching
```

开始修改前阅读适用范围内的 `AGENTS.md`。修改 `Python/`、`Apps/` 或 `Paper/` 时，还应阅读对应子目录中的说明。

## 3. 保存和提交

编辑器中的“保存”只把内容写入磁盘；Git 提交才是可追踪的项目快照。推荐按以下顺序操作。

### 3.1 检查改动

```bash
git status --short
git diff
```

确认列表中没有私有数据、生成文件或无关改动。不要使用 `git add .` 代替检查。

### 3.2 选择性暂存

只暂存本次任务需要的明确路径：

```bash
git add -- <path-one> <path-two>
git diff --cached
```

再次检查暂存差异，尤其关注：

- 是否包含密钥、令牌、私人路径或观测数据；
- 是否混入格式化噪声或无关文件；
- 是否有调试输出、临时配置或生成产物；
- 文档、测试与实现是否保持一致。

### 3.3 创建提交

```bash
git commit -m "<类型>: <简明说明>"
```

常用类型包括 `feat`、`fix`、`docs`、`test` 和 `refactor`。一个提交应表达一个完整目的；测试未通过或差异尚未审阅时不要提交。

## 4. 测试和 CLI 验证

默认使用 Miniforge 环境 `solarphysics_env_latest`。只有明确进行兼容性比较时才使用旧环境 `solarphysics_env`；不要使用系统 Python、裸 `pip`、裸 `pytest` 或已移除的 `solarphysics_backup`。

### 4.1 macOS

设置 Miniforge 的实际位置：

```bash
MINIFORGE_CONDA="<miniforge-root>/bin/conda"
```

检查公共 Python 包：

```bash
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pip check
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m compileall -q Python/solar_toolkit Python/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m ruff check Python/solar_toolkit Python/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pytest Python/tests
```

对 Apps 改动先运行 CLI 帮助冒烟测试：

```bash
./Apps/run.sh frontend app-v1 --help
```

也可将 `app-v1` 替换为需要验证的前端 ID，例如 `workbench`、`image-viewer`、`image-composer`、`bad-frame-review`、`source-map`、`dart-spectrogram`、`roi-lightcurve`、`radio-composite`、`source-trajectory` 或 `aia-radio-composite`。

Apps 共享平台、UI、CLI 或工作流发生变化时，运行：

```bash
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m compileall -q Apps/solar_apps Apps/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m ruff check Apps/solar_apps Apps/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m black --check Apps/solar_apps Apps/tests
"$MINIFORGE_CONDA" run -n solarphysics_env_latest python -m pytest Apps/tests --basetemp Local/tmp/pytest-apps-<任务标识>
```

每次测试使用新的 `<任务标识>`，避免多个测试进程共享临时目录。

### 4.2 Windows PowerShell

设置 Miniforge 的实际位置：

```powershell
$Conda = "<miniforge-root>\Scripts\conda.exe"
```

检查公共 Python 包：

```powershell
& $Conda run -n solarphysics_env_latest python -m pip check
& $Conda run -n solarphysics_env_latest python -m compileall -q Python\solar_toolkit Python\tests
& $Conda run -n solarphysics_env_latest python -m ruff check Python\solar_toolkit Python\tests
& $Conda run -n solarphysics_env_latest python -m pytest Python\tests
```

运行 CLI 帮助冒烟测试：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apps\run.ps1 frontend app-v1 --help
```

Apps 共享平台、UI、CLI 或工作流发生变化时，运行：

```powershell
& $Conda run -n solarphysics_env_latest python -m compileall -q Apps\solar_apps Apps\tests
& $Conda run -n solarphysics_env_latest python -m ruff check Apps\solar_apps Apps\tests
& $Conda run -n solarphysics_env_latest python -m black --check Apps\solar_apps Apps\tests
& $Conda run -n solarphysics_env_latest python -m pytest Apps\tests --basetemp Local\tmp\pytest-apps-<任务标识>
```

前端改动还应检查 Light、Dark 和 Auto 模式。测试结束后关闭 Qt 窗口，并用 `Ctrl+C` 停止仍在运行的浏览器或 Streamlit 服务。

## 5. 推送和 Pull Request

测试通过并确认暂存内容后，将功能分支推送到远端：

```bash
git push -u origin feature/<简短名称>
```

然后在 GitHub 创建面向 `main` 的 PR：

1. 标题说明改动目的；
2. 描述改动范围、验证命令和已知限制；
3. 确认没有上传私有数据、凭据或机器专用路径；
4. 等待 CI 全部通过；
5. 处理审查意见后再合并。

不要直接向 `main` 强制推送，也不要用 `--force` 或 `--force-with-lease` 绕过正常协作流程。

## 6. 分支落后和合并冲突

如果 PR 提示功能分支落后，采用普通合并更新分支，避免改写已推送历史：

```bash
git fetch origin
git switch feature/<简短名称>
git merge origin/main
```

发生冲突时：

1. 使用 `git status` 查看冲突文件；
2. 逐个理解并编辑冲突，删除冲突标记；
3. 运行与改动相关的测试；
4. 只暂存已解决的文件；
5. 使用 `git diff --cached` 复查；
6. 完成合并提交并重新推送功能分支。

不确定如何解决冲突时应停止并寻求审查，不要机械选择 “ours” 或 “theirs”。

## 7. 合并后的同步和清理

确认 PR 已在 GitHub 合并后，同步本地 `main`：

```bash
git switch main
git pull --ff-only origin main
git branch --merged main
```

只有在列表中确认目标分支已合并，并获得明确确认后，才安全删除本地分支：

```bash
git branch -d feature/<简短名称>
```

远程分支也必须单独确认后才能删除：

```bash
git push origin --delete feature/<简短名称>
```

不要用 `git branch -D` 强制删除尚未确认合并的分支。

## 8. 安全撤销

取消暂存但保留磁盘上的改动：

```bash
git restore --staged -- <path>
```

查看某个文件尚未暂存的差异：

```bash
git diff -- <path>
```

只有在确认文件中的本地改动可以永久丢弃后，才执行：

```bash
git restore -- <path>
```

如果误暂存私有文件，应先使用 `git restore --staged` 取消暂存，再确认它受 `.gitignore` 保护。若凭据已经提交或推送，应立即停止后续操作、轮换凭据并寻求历史清理支持；仅删除工作区文件不能消除 Git 历史中的秘密。

日常操作中不要使用：

- `git reset --hard`；
- `git clean -fd`；
- `git push --force`；
- 未检查内容的 `git add .`；
- 直接覆盖或删除 `main`。

## 9. 常见失败处理

### 测试失败

- 保留完整错误输出，先运行最小相关测试定位问题；
- 判断失败是否来自本次改动；
- 修复后重新运行相关测试，共享行为变化时再运行完整测试；
- 不通过删除测试、更新快照或隐藏异常来制造“通过”结果。

### `git pull --ff-only` 失败

- 这表示本地与远端历史可能已分叉；
- 不要立即重置或强制推送；
- 使用 `git status` 和 `git log --oneline --decorate --graph --all` 查看历史，再决定保留哪部分提交。

### 误暂存私有文件

```bash
git restore --staged -- <private-path>
git status --short
```

确认文件不再位于暂存区，并检查 `.gitignore` 是否覆盖对应类别。

### CLI 启动失败

- 确认使用 `Apps/run.sh` 或 `Apps/run.ps1`；
- 确认环境名为 `solarphysics_env_latest`；
- 确认 Miniforge 路径存在；
- 先运行 `frontend <id> --help`，再启动完整界面；
- 不自动回退到系统 Python 或其他环境。

## 10. 快捷更新命令（`tools quick`）

日常的「检查 → 选择性提交 → 推送 → 建 PR」可以通过启动器内的
`tools quick` 命令组完成。它们严格遵循上文第 2~5 节的分支与 PR 约定，
不会执行 `git add .`、不会 force-push、也不会在 `main` 上直接提交。

### 10.1 快速检查

运行公共 Python 包的 `pip check`、`compileall` 和 `ruff`：

```bash
./Apps/run.sh tools quick check
```

### 10.2 选择性提交

只暂存明确给出的路径，然后复查暂存差异并提交。若暂存内容命中私有
数据或生成产物的后缀/目录（`Local`、`outputs`、`logs`、观测与媒体后缀
等），命令会拒绝提交并自动取消暂存：

```bash
./Apps/run.sh tools quick save -m "feat: <说明>" -- <path-one> <path-two>
```

### 10.3 推送并创建 PR

在功能分支上执行 `git push -u`，并在 `gh` 可用时创建面向 `main` 的 PR：

```bash
./Apps/run.sh tools quick push
```

`main` 上运行 `quick push` 会被拒绝，先按第 2 节创建功能分支。

### 10.4 一键更新

`check + save + push` 的串联：

```bash
./Apps/run.sh tools quick update -m "feat: <说明>" -- <path-one> <path-two>
```

## 11. 发布命令（`tools release`）

`tools release` 负责版本升级、变更日志改写、打 tag、推送并创建 GitHub
Release。**默认是 dry-run**，只有显式传入 `--execute` 才会写盘和推送；
它不会 force-push、不会改写历史，也只在 `main` 干净且与远端同步时执行。

### 11.1 只做前置检查（不写任何文件）

```bash
./Apps/run.sh tools release check
```

检查内容包括：位于 `main`、工作区干净、无未推送提交、与 `origin/main`
同步，并报告当前版本与环境锁门禁状态。

### 11.2 预览一次发布

```bash
./Apps/run.sh tools release run --bump patch --note "<发布说明>"
```

`--bump` 取 `patch`、`minor` 或 `major`。命令会打印将要改动的文件与将要
执行的 Git 命令，但不会修改任何内容。

### 11.3 真正执行

```bash
./Apps/run.sh tools release run --bump patch --note "<发布说明>" --execute
```

执行步骤：更新两份 `_version.py`（`Python/solar_toolkit/_version.py` 与
`Apps/solar_apps/_version.py`，二者必须一致）、把 `## Unreleased` 改写为
新版本段、`git add` 版本与变更日志、提交 `chore: release v<版本>`、打
annotated tag、推送 `main` 与 tag，最后在 `gh` 可用时创建 GitHub Release。

> 发布前应先在功能分支通过全部相关测试，再合并到 `main`。版本号只允许
> 通过 `tools release` 修改，禁止手改 `_version.py`。

## 12. 提交或创建 PR 前检查清单

- [ ] 当前位于预期的功能分支，而不是直接在 `main` 开发；
- [ ] `git status --short` 只显示本次任务相关文件；
- [ ] `git diff` 和 `git diff --cached` 已人工审阅；
- [ ] 没有私有数据、凭据、个人路径、日志、输出或缓存；
- [ ] 使用 `solarphysics_env_latest` 运行了与改动相关的测试；
- [ ] CLI 改动已通过对应 `frontend <id> --help` 冒烟测试；
- [ ] Apps 测试使用了唯一的 `Local/tmp` 临时目录；
- [ ] 文档、测试和实际行为一致；
- [ ] PR 面向 `main`，且 CI 通过后才合并；
- [ ] 分支删除安排在确认合并之后。
