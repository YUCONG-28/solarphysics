#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "${script_dir}/.." && pwd)"
simulation_root="${bundle_root}/simulation"
jobs="${JOBS:-8}"
ranks="${RANKS:-1}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${simulation_root}"

python -m spike_typeIII_visual.athena doctor
python -m spike_typeIII_visual.athena build \
  --problem spike_topping_solar_jet \
  --jobs "${jobs}"

athena_binary="${bundle_root}/Local/athena/build/solar_jet_reference_serial/bin/athena"
python -m spike_typeIII_visual.athena run \
  --binary "${athena_binary}" \
  --profile jet-smoke \
  --run-id windows_athena_dev_static \
  --ranks "${ranks}" \
  --overwrite \
  --override time/tlim=0.2 \
  --override problem/drive_enabled=0 \
  --override output2/dt=0.1
python -m spike_typeIII_visual.athena ingest \
  --run-dir "${bundle_root}/Local/athena/runs/windows_athena_dev_static" \
  --output "${bundle_root}/Local/athena/runs/windows_athena_dev_static/bridge.h5"

python -m spike_typeIII_visual.amrvac doctor
python -m spike_typeIII_visual.amrvac build --jobs "${jobs}"
amrvac_binary="$(find "${bundle_root}/Local/amrvac/build" \
  -type f -path '*/case/amrvac' -perm -u+x | sort | head -n 1)"
if [[ -z "${amrvac_binary}" ]]; then
  echo "未找到 AMRVAC 构建产物。" >&2
  exit 1
fi
python -m spike_typeIII_visual.amrvac run \
  --binary "${amrvac_binary}" \
  --run-id windows_amrvac_dev_static \
  --ranks "${ranks}" \
  --overwrite
python -m spike_typeIII_visual.amrvac ingest \
  --run-dir "${bundle_root}/Local/amrvac/runs/windows_amrvac_dev_static" \
  --output "${bundle_root}/Local/amrvac/runs/windows_amrvac_dev_static/bridge.h5"

echo
echo "开发 smoke 已完成。结果位于 ${bundle_root}/Local/。"
echo "注意：运行成功不代表静态 Mach、AMRVAC divB 或 restart 已通过物理门槛。"
