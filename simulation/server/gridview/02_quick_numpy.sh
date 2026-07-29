#!/usr/bin/env bash
set -euo pipefail
source ./server_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m spike_typeIII_visual.main --profile quick --rmhd-engine numpy \
  --device cpu --precision float64 --animation-format none \
  --output-dir "${RESULT_ROOT}/quick_numpy_seed${SEED}"
