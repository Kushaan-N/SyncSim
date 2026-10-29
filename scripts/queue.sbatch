#!/bin/bash
# One node of a large queue run (scripts/queue_worker.py): WORKERS_PER_GPU generator workers
# on every GPU SLURM gave this job, all pulling from QUEUE until it is empty.
#
#   QUEUE=/scratch/.../rt2_queue WORKERS_PER_GPU=3 \
#     sbatch -p rtx-batch -A chuangg_umass --gres gpu:8 -c 128 --mem 900G -t 1-00:00:00 \
#            --array 0-3 --export ALL scripts/queue.sbatch
#
# Safe to preempt / cancel / resubmit at any time: claims go stale after 15 min and the
# generator resumes from unit _DONE markers.
set -euo pipefail
: "${QUEUE:?set QUEUE}"
: "${WORKERS_PER_GPU:=3}"
: "${PYTHON_BIN:=/work/umass/chuangg_umass/yuncongyang_umass/syncsim/envs/rt/bin/python}"
: "${ROBOTWIN_ROOT:=/home/yuncongyang_umass/kn/RoboTwin}"
export PYTHON_BIN ROBOTWIN_ROOT
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[[ -f "${REPO}/scripts/queue_worker.py" ]] || REPO="$(cd "$(dirname "$0")/.." && pwd)"

NG=$(nvidia-smi -L | wc -l)
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
THREADS=$(( NCPU / (NG * WORKERS_PER_GPU) )); (( THREADS < 1 )) && THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpus=${NG} workers/gpu=${WORKERS_PER_GPU} threads=${THREADS}"

# fail fast on a broken environment (see gen_set.sh)
(cd "${ROBOTWIN_ROOT}" && CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -c \
  "import envs._base_task; from curobo.wrap.reacher.motion_gen import MotionGen") \
  || { echo "PREFLIGHT FAILED"; exit 1; }

for (( g = 0; g < NG; g++ )); do
  for (( w = 0; w < WORKERS_PER_GPU; w++ )); do
    "${PYTHON_BIN}" -u "${REPO}/scripts/queue_worker.py" work --queue "${QUEUE}" --gpu "${g}" \
      --threads "${THREADS}" --worker_id "$(hostname)-${SLURM_JOB_ID:-x}-g${g}w${w}" \
      ${WORKER_FLAGS:-} &
    sleep 5   # stagger start-up (RoboTwin import + curobo warm-up)
  done
done
wait
echo "[queue.sbatch] all workers exited"
