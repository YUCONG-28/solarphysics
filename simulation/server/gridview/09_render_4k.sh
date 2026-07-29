#!/usr/bin/env bash
set -euo pipefail
source ./server_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"
event="${RESULT_ROOT}/rmhd_fine_event_seed${SEED}"
control="${RESULT_ROOT}/rmhd_fine_control_seed${SEED}"
python -m spike_typeIII_visual.main --profile rmhd-fine-event \
  --stage render --mhd-dataset "${event}/data/rmhd_fields.h5" \
  --control-run-dir "${control}" --render-profile scientific-4k \
  --animation-format both --output-dir "${event}"
