#!/usr/bin/env bash
set -euo pipefail

source ./server_env.sh
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"

event="${RESULT_ROOT}/rmhd_fine_event_seed${SEED}"
delivery="${RESULT_ROOT}/deliveries_seed${SEED}"
mkdir -p "${delivery}"

python -m spike_typeIII_visual.server validate --run-root "${event}"

for tier in report-lite presentation research-complete; do
  python -m spike_typeIII_visual.server package \
    --run-root "${event}" \
    --tier "${tier}" \
    --output "${delivery}/${tier}_seed${SEED}.tar.gz"
done

(
  cd "${delivery}"
  sha256sum *.tar.gz > DELIVERY_SHA256SUMS.txt
  sha256sum -c DELIVERY_SHA256SUMS.txt
)

if tar -tzf "${delivery}/report-lite_seed${SEED}.tar.gz" \
  | grep -Eq '\.(h5|mkv|mp4|gif|log)$|checkpoint'; then
  echo "report-lite contains a forbidden large/private artifact." >&2
  exit 1
fi

if tar -tzf "${delivery}/presentation_seed${SEED}.tar.gz" \
  | grep -Eq '\.(h5|mkv|log)$|checkpoint'; then
  echo "presentation contains a forbidden research/private artifact." >&2
  exit 1
fi

echo "CPU dual-partition validation and tiered packaging passed."
