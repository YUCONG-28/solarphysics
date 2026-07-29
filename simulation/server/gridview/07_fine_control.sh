#!/usr/bin/env bash
set -euo pipefail
source ./server_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS=1
python -m spike_typeIII_visual.main --profile rmhd-fine-control \
  --rmhd-engine torch --device cpu --precision float64 --stage simulate \
  --animation-format none --checkpoint-every 400 \
  --output-dir "${RESULT_ROOT}/rmhd_fine_control_seed${SEED}"
