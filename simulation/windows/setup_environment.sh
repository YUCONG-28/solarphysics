#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "${script_dir}/.." && pwd)"
environment_file="${bundle_root}/simulation/environment.solar-simulation.yml"

if command -v mamba >/dev/null 2>&1; then
  environment_tool="mamba"
elif command -v conda >/dev/null 2>&1; then
  environment_tool="conda"
else
  echo "未找到 conda/mamba。请先在 WSL2 安装 Linux x86-64 Miniforge。" >&2
  exit 1
fi

if "${environment_tool}" env list | awk '{print $1}' | grep -qx solar_simulation; then
  echo "solar_simulation 已存在；按 YAML 更新。"
  "${environment_tool}" env update -n solar_simulation -f "${environment_file}" --prune
else
  "${environment_tool}" env create -f "${environment_file}"
fi

echo "环境已准备。下一步执行：conda activate solar_simulation"
