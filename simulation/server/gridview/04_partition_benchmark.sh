#!/usr/bin/env bash
set -euo pipefail
source ./server_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS=1
mkdir -p "${RESULT_ROOT}/benchmarks"
benchmark_label="${BENCH_LABEL:-${SLURM_JOB_PARTITION:-unknown}}"
for repeat in 1 2 3; do
  /usr/bin/time -p python -m spike_typeIII_visual.main \
    --profile quick --rmhd-engine torch --device cpu --precision float64 \
    --stage simulate --animation-format none \
    --output-dir "${RESULT_ROOT}/benchmarks/${benchmark_label}_r${repeat}"
done
