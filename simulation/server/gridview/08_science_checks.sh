#!/usr/bin/env bash
set -euo pipefail

source ./server_env.sh
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}/simulation"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS=1

medium="${RESULT_ROOT}/rmhd_medium_event_seed${SEED}"
event="${RESULT_ROOT}/rmhd_fine_event_seed${SEED}"
control="${RESULT_ROOT}/rmhd_fine_control_seed${SEED}"
science="${RESULT_ROOT}/science_checks_seed${SEED}"
mkdir -p "${science}"

for path in \
  "${medium}/data/rmhd_fields.h5" \
  "${event}/data/rmhd_fields.h5" \
  "${control}/data/rmhd_fields.h5"; do
  test -s "${path}"
done

python -m spike_typeIII_visual.energy_check \
  --dissipative "${event}/data/rmhd_fields.h5" \
  --engine torch --device cpu \
  --output "${science}/energy_gates.json"

python -m spike_typeIII_visual.radio_check \
  --hdf5 "${event}/data/rmhd_fields.h5" \
  --seed "${SEED}" \
  --output "${science}/radio_gates.json"

python -m spike_typeIII_visual.convergence \
  --coarse "${medium}/data/rmhd_fields.h5" \
  --fine "${event}/data/rmhd_fields.h5" \
  --output "${science}/convergence_medium_fine.json"

python -m spike_typeIII_visual.timestep_check \
  --reference "${event}/data/rmhd_fields.h5" \
  --engine torch --device cpu \
  --output "${science}/timestep_halving_1024x512.json"

python - "${medium}" "${event}" "${control}" "${science}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

medium, event, control, science = map(Path, sys.argv[1:])


def metadata(run):
    return json.loads((run / "data/run_metadata.json").read_text())


event_diag = metadata(event)["diagnostics"]
control_diag = metadata(control)["diagnostics"]
event_control = {
    "schema": "rmhd-event-control-gates-v1",
    "event_status": event_diag["event_status"],
    "event_count": event_diag["event_count"],
    "event_jet_coincidence_fraction": event_diag["jet_coincidence_fraction"],
    "event_minimum_topping_margin_mhz": event_diag[
        "minimum_topping_margin_mhz"
    ],
    "control_status": control_diag["event_status"],
    "control_count": control_diag["event_count"],
}
event_control["passed"] = (
    event_control["event_status"] == "events"
    and int(event_control["event_count"]) == 12
    and float(event_control["event_jet_coincidence_fraction"]) == 1.0
    and float(event_control["event_minimum_topping_margin_mhz"]) > 0.0
    and event_control["control_status"] == "no_event"
    and int(event_control["control_count"]) == 0
)
(science / "event_control_gates.json").write_text(
    json.dumps(event_control, indent=2) + "\n"
)

reports = {}
for path in sorted(science.glob("*_gates.json")) + [
    science / "convergence_medium_fine.json",
    science / "timestep_halving_1024x512.json",
]:
    payload = json.loads(path.read_text())
    if "passed" in payload:
        passed = bool(payload["passed"])
    elif "core_diagnostics_below_5_percent" in payload:
        passed = bool(payload["core_diagnostics_below_5_percent"])
    else:
        passed = bool(payload["core_diagnostics_below_1_percent"])
    reports[path.name] = {"passed": passed}

summary = {
    "schema": "rmhd-cpu-science-suite-v1",
    "reports": reports,
    "passed": all(item["passed"] for item in reports.values()),
}
(science / "science_suite_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

lines = []
for path in sorted(science.glob("*.json")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.name}")
(science / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["passed"] else 1)
PY

(cd "${science}" && sha256sum -c SHA256SUMS.txt)
