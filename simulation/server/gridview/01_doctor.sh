#!/usr/bin/env bash
set -euo pipefail
source ./server_env.sh
module load mpi/openmpi/gnu/4.0.3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"
python -m spike_typeIII_visual.server doctor
python -m spike_typeIII_visual.main --help >/dev/null
df -h "${RESULT_ROOT}"
