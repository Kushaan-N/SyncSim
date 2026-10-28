#!/usr/bin/env bash
# Add contacts.npz / contacts.json to every rollout of an existing data root, as a SLURM array
# of N shards (scripts/annotate_contacts.py; unrendered replay, existing files untouched).
#
#   DATA_ROOT=<data> N_SHARDS=48 scripts/submit_contacts.sh [extra annotate_contacts.py args]
#
# Rollouts that already have contacts.json are skipped, so a re-submit only does what is left.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${N_SHARDS:=48}"
: "${PARTITION:=rtx-batch,rtx-devel}"
: "${ACCOUNT:=chuangg_umass}"
: "${TIME:=4:00:00}"            # rtx-devel caps jobs at 4 h
: "${PYTHON_BIN:=/work/umass/chuangg_umass/yuncongyang_umass/syncsim/envs/rt/bin/python}"
: "${ROBOTWIN_ROOT:=/home/yuncongyang_umass/kn/RoboTwin}"
LOG_DIR="${DATA_ROOT}/_contacts_logs"
mkdir -p "${LOG_DIR}"
EXTRA="$*"
sbatch -p "${PARTITION}" -A "${ACCOUNT}" --gres=gpu:1 -c 4 --mem=32G -t "${TIME}" \
  --array=0-$((N_SHARDS - 1)) -J syncsim-contacts -o "${LOG_DIR}/shard_%a_%A.out" \
  --wrap "cd ${REPO} && ROBOTWIN_ROOT=${ROBOTWIN_ROOT} ${PYTHON_BIN} scripts/annotate_contacts.py \
          --data_root ${DATA_ROOT} --shard \${SLURM_ARRAY_TASK_ID}/${N_SHARDS} ${EXTRA}"
