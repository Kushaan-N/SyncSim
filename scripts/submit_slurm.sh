#!/usr/bin/env bash
# ============================================================================
# Submit SyncSim generation as a 2-D SLURM array on the AICR cluster:
#   array index a -> set  = a / NUM_TASK_PARTITIONS   (seed shard -> set_<set>/)
#                    part = a % NUM_TASK_PARTITIONS   (round-robin slice of the task list)
# Each job takes GPUS_PER_JOB GPUs and runs its tasks MAX_PARALLEL at a time via
# scripts/gen_set.sh, which stays the single source of generation config (every env var
# exported here is forwarded with --export=ALL).
#
#   NUM_SETS=4 ./scripts/submit_slurm.sh
#   DRY_RUN=1 NUM_SETS=2 TASKS="handover_block lift_pot" ./scripts/submit_slurm.sh
#
# Partitions (docs.aicr.ai/running-jobs/gpu-jobs): rtx-batch (RTX PRO 6000, 24 h) is the
# right one -- SAPIEN's ray-tracing shader needs RT cores, which B200s do not have.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NUM_SETS:=4}"
: "${TASKS_PER_PARTITION:=8}"
: "${GPUS_PER_JOB:=1}"
: "${MAX_PARALLEL:=4}"
: "${PARTITION:=rtx-batch}"
: "${ACCOUNT:=chuangg_umass}"
: "${TIME_LIMIT:=24:00:00}"
: "${CPUS_PER_JOB:=16}"
: "${MEM_GB:=128}"
: "${MAX_CONCURRENT:=0}"
: "${DRY_RUN:=0}"
: "${ROBOTWIN_ROOT:=/home/yuncongyang_umass/kn/RoboTwin}"
: "${OUTPUT_BASE:=/scratch/yuncongyang_umass/syncsim/data}"
: "${TASKS:=ALL}"

if [[ "${TASKS}" == "ALL" ]]; then
  TASKS=$(cd "${ROBOTWIN_ROOT}/envs" && ls *.py | sed 's/\.py$//' \
          | grep -v -E '^(_|__init__|model_utils)' | tr '\n' ' ')
fi
read -r -a TASK_ARR <<< "${TASKS}"
NT=${#TASK_ARR[@]}
NUM_TASK_PARTITIONS=$(( (NT + TASKS_PER_PARTITION - 1) / TASKS_PER_PARTITION ))
ARRAY_SIZE=$(( NUM_SETS * NUM_TASK_PARTITIONS ))
LOG_DIR="${OUTPUT_BASE}/_slurm"; mkdir -p "${LOG_DIR}"

echo "tasks=${NT} partitions=${NUM_TASK_PARTITIONS} sets=${NUM_SETS} -> ${ARRAY_SIZE} jobs on ${PARTITION}"
RANGE="0-$(( ARRAY_SIZE - 1 ))"; [[ "${MAX_CONCURRENT}" -gt 0 ]] && RANGE="${RANGE}%${MAX_CONCURRENT}"

export TASKS NUM_TASK_PARTITIONS MAX_PARALLEL ROBOTWIN_ROOT OUTPUT_BASE
CMD=(sbatch --parsable --array="${RANGE}" --partition="${PARTITION}" --account="${ACCOUNT}"
     --gpus="${GPUS_PER_JOB}" --cpus-per-task="${CPUS_PER_JOB}" --mem="${MEM_GB}G"
     --time="${TIME_LIMIT}" --job-name=syncsim-gen --export=ALL
     --output="${LOG_DIR}/%A_%a.out" "${SCRIPT_DIR}/gen.sbatch")
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "${CMD[*]}"
  for (( a = 0; a < ARRAY_SIZE; a++ )); do
    s=$(( a / NUM_TASK_PARTITIONS )); p=$(( a % NUM_TASK_PARTITIONS ))
    sub=(); for (( t = p; t < NT; t += NUM_TASK_PARTITIONS )); do sub+=("${TASK_ARR[$t]}"); done
    echo "  job ${a}: set ${s}: ${sub[*]}"
  done
  exit 0
fi
"${CMD[@]}"
