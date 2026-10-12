#!/bin/bash
# One array task of scripts/submit_slurm.sh: (set, task partition) -> scripts/gen_set.sh.
# Resubmitting the same array resumes: finished units carry _DONE, finished tasks a
# _state/<task>.done marker, and seeds that cannot produce data are cached in _dropped.json.
set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts"
[[ -f "${SCRIPT_DIR}/gen_set.sh" ]] || SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

a=${SLURM_ARRAY_TASK_ID}
export SET_ID=$(( a / NUM_TASK_PARTITIONS ))
part=$(( a % NUM_TASK_PARTITIONS ))
read -r -a ALL <<< "${TASKS}"
SUB=(); for (( t = part; t < ${#ALL[@]}; t += NUM_TASK_PARTITIONS )); do SUB+=("${ALL[$t]}"); done
export TASKS="${SUB[*]}"

# the GPUs SLURM gave this job, renumbered from 0 inside the job
NG=$(nvidia-smi -L | wc -l)
export GPUS=$(seq -s, 0 $(( NG - 1 )))

echo "host=$(hostname) set=${SET_ID} part=${part} gpus=${GPUS} tasks=${TASKS}"
nvidia-smi -L
bash "${SCRIPT_DIR}/gen_set.sh"
