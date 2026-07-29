#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate torch-cuda
cd /mnt/d/solarphysics/simulation

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/spike_typeiii_mpl

python -m spike_typeIII_visual.production \
  --seed 20260726 \
  --target scientific-4k \
  --resume
