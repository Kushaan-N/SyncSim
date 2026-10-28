#!/usr/bin/env bash
# ============================================================================
# Regenerate, in place, the counterfactuals listed in an exclude file written by
#   python scripts/verify_set.py --set_dir <set> --exclude_out bad.tsv
# Only counterfactual_* lines are used (unit-level problems, e.g. a failed wrist mount,
# cannot be repaired by re-rolling a counterfactual -- leave those in the exclude list).
# Each (set, task) group runs `generate.py --redo_list` with the embodiment / task config
# recorded in that task's base_init.json, so it reuses the original scene, cameras and
# expert. Old dirs are backed up under <data root>/_redo_backup/.
#
#   LIST=bad.tsv bash scripts/redo_cf.sh                      # needs a GPU (run under sbatch)
#   LIST=bad.tsv DRY_RUN=1 bash scripts/redo_cf.sh            # print the groups and commands
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${SCRIPT_DIR}")"
: "${LIST:?set LIST to a verify_set.py --exclude_out file}"
: "${PYTHON:=/work/umass/chuangg_umass/yuncongyang_umass/syncsim/envs/rt/bin/python}"
: "${MAX_PARALLEL:=4}"
: "${DRY_RUN:=0}"

GROUPS_TSV=$(mktemp)
"${PYTHON}" -I - "${LIST}" > "${GROUPS_TSV}" <<'EOF'
import json, os, sys
from collections import defaultdict
groups = defaultdict(list)
for line in open(sys.argv[1]):
    path = line.split("\t")[0].strip()
    if not os.path.basename(path).startswith("counterfactual_"):
        continue
    unit = os.path.dirname(path)
    task_dir = os.path.dirname(os.path.dirname(unit))
    groups[task_dir].append((path, unit))
for task_dir, items in sorted(groups.items()):
    cfg = json.load(open(os.path.join(items[0][1], "base_init.json")))["config"]
    print("\t".join([os.path.dirname(task_dir), cfg["task"], cfg["task_config"],
                     cfg["embodiment"] or "-", str(len(items))]))   # "-": bash read collapses empty fields
EOF

running=0
while IFS=$'\t' read -r set_dir task task_config embodiment n; do
  cmd=("${PYTHON}" -m syncsim.generate --task "${task}" --task_config "${task_config}"
       --output_dir "${set_dir}" --redo_list "${LIST}")
  [[ "${embodiment}" != "-" ]] && cmd+=(--embodiment "${embodiment}")
  echo "[redo] ${set_dir##*/}/${task}: ${n} counterfactual(s)"
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "  ${cmd[*]}"; continue
  fi
  mkdir -p "${set_dir}/_logs"
  (cd "${REPO}" && "${cmd[@]}" > "${set_dir}/_logs/${task}.redo.log" 2>&1) &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then wait -n; running=$((running - 1)); fi
done < "${GROUPS_TSV}"
wait
rm -f "${GROUPS_TSV}"
