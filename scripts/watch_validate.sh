#!/usr/bin/env bash
# ============================================================================
# Validate a generation run INCREMENTALLY: every time a task finishes in any set
# (<set>/_state/<task>.done appears), run scripts/verify_set.py on just that task and
# write <set>/_validate/<task>.json (+ .txt, + montages). A one-line verdict per task goes
# to <data_root>/_validate.log, so a regression shows up hours before the run ends.
#
#   DATA_ROOT=/path/to/data nohup bash scripts/watch_validate.sh &
#   ONCE=1 DATA_ROOT=/path/to/data bash scripts/watch_validate.sh     # one sweep, then exit
#
# It only reads the data, so it is safe to run next to the generation jobs. It exits by
# itself once no job named syncsim-* is queued and every finished task has been validated.
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${PYTHON:=/scratch/yuncongyang_umass/syncsim/envs/rt/bin/python}"
: "${INTERVAL:=300}"
: "${ONCE:=0}"
: "${MONTAGES:=1}"
LOG="${DATA_ROOT}/_validate.log"

sweep() {
  local did=0
  for set_dir in "${DATA_ROOT}"/set_*; do
    [[ -d "${set_dir}/_state" ]] || continue
    mkdir -p "${set_dir}/_validate"
    for done_f in "${set_dir}"/_state/*.done; do
      [[ -e "${done_f}" ]] || continue
      task=$(basename "${done_f}" .done)
      out="${set_dir}/_validate/${task}.json"
      [[ -f "${out}" && "${out}" -nt "${done_f}" ]] && continue
      margs=()
      [[ "${MONTAGES}" == 1 ]] && margs=(--montage_dir "${set_dir}/_validate/montage")
      # -I: never import modules from the data directory or the cwd
      (cd "${set_dir}" && "${PYTHON}" -I "${SCRIPT_DIR}/verify_set.py" --set_dir "${set_dir}" \
          --tasks "${task}" --json_out "${out}.tmp" "${margs[@]}" \
          > "${set_dir}/_validate/${task}.txt" 2>&1)
      rc=$?
      if [[ -f "${out}.tmp" ]]; then
        mv "${out}.tmp" "${out}"
        verdict=$("${PYTHON}" -I -c "import json,sys; r=json.load(open(sys.argv[1])); \
print(f\"{r['units']} units, {len(r['errors'])} error(s), {len(r['warnings'])} warning(s)\")" "${out}")
      else
        verdict="validator crashed (rc=${rc}), see ${set_dir}/_validate/${task}.txt"
      fi
      tag="OK  "; [[ ${rc} -ne 0 ]] && tag="FAIL"
      echo "$(date '+%F %T') ${tag} $(basename "${set_dir}")/${task}: ${verdict}" | tee -a "${LOG}"
      did=1
    done
  done
  return ${did}
}

while true; do
  sweep
  [[ "${ONCE}" == 1 ]] && break
  if [[ -z "$(squeue -h -u "${USER}" -o '%j' 2>/dev/null | grep '^syncsim-')" ]]; then
    sweep    # catch tasks that finished since the last sweep
    echo "$(date '+%F %T') no syncsim jobs left; watcher exiting" | tee -a "${LOG}"
    break
  fi
  sleep "${INTERVAL}"
done
