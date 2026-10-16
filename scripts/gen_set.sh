#!/usr/bin/env bash
# ============================================================================
# SyncSim set driver: generate one SET (a seed shard) for a list of RoboTwin tasks.
#
# THE SINGLE PLACE TO CHANGE GENERATION DEFAULTS. Every knob of syncsim/generate.py
# that a run normally touches is an env var below; anything not listed falls back to
# the Config dataclass default in syncsim/generate.py.
#
# Tasks run as separate processes on the GPU(s) given by GPUS, MAX_PARALLEL at a time
# (SAPIEN rendering + curobo planning share one GPU fine; each process ~3-5 GB GPU mem).
#
#   SET_ID=0 TASKS="handover_block lift_pot" ./scripts/gen_set.sh
#   SET_ID=0 PRINT_CONFIG=1 ./scripts/gen_set.sh
#   SET_ID=0 DRY_RUN=1 ./scripts/gen_set.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- run control ------------------------------------------------------------
: "${SET_ID:?set SET_ID (seed shard index)}"
: "${OUTPUT_BASE:=/scratch/yuncongyang_umass/syncsim/data}"
: "${PYTHON_BIN:=/scratch/yuncongyang_umass/syncsim/envs/rt/bin/python}"
: "${ROBOTWIN_ROOT:=/home/yuncongyang_umass/kn/RoboTwin}"
: "${TASKS:=ALL}"                    # ALL = every task module in RoboTwin/envs
: "${GPUS:=0}"                       # comma list; tasks round-robin over them
: "${MAX_PARALLEL:=4}"
: "${DRY_RUN:=0}"
: "${PRINT_CONFIG:=0}"
: "${OVERWRITE:=0}"

# ---- seeds: set s covers seeds [SEED_BASE + s*SEEDS_PER_SET, +SEEDS_PER_SET) --------
: "${SEED_BASE:=0}"
: "${SEEDS_PER_SET:=40}"
: "${MAX_EPISODES:=0}"               # cap kept episodes per task per set (0 = all seeds)

# ---- GENERATION PARAMETERS ------------------------------------------------------
# scene / RoboTwin
: "${TASK_CONFIG:=demo_randomized}"  # RoboTwin domain randomisation preset
: "${EMBODIMENT:=}"                  # "" = preset's (aloha-agilex); e.g. "piper,franka-panda,0.6"
: "${RANDOM_BACKGROUND:=}"           # "" keeps the preset
: "${CLUTTERED_TABLE:=}"
: "${RANDOM_LIGHT:=}"
# rendering
: "${IMAGE_SIZE:=512}"
: "${FRAME_STRIDE:=12}"              # 250 Hz physics / 12 = 20.8 fps (robocasa: 20 Hz)
: "${SHADER:=rt}"
: "${RT_SPP:=32}"
: "${RECORD_WRIST:=true}"
# 3rd-person rig
: "${NUM_CAMERA_UNITS:=1}"
: "${NUM_VIEWS:=2}"
: "${CAMERA_SEED_SALT:=$SET_ID}"     # every set samples fresh viewpoints
: "${AZIMUTH_RANGE_DEG:=110}"
: "${ELEV_MIN_DEG:=15}"
: "${ELEV_MAX_DEG:=65}"
: "${RADIUS_MIN:=0.75}"
: "${RADIUS_MAX:=1.35}"
: "${MIN_SEPARATION_DEG:=45}"
: "${NUM_CANDIDATES:=96}"
: "${CHECK_FRAMES:=4}"
: "${GATE_ARMS:=both}"               # both | active | any
# wrist mounts
: "${WRIST_POS_JITTER:=0.01}"
: "${WRIST_ROT_JITTER_DEG:=5}"
# calibration
: "${RECORD_CALIBRATION:=true}"
: "${CALIB_TRANS_M:=0.05,0.08}"
: "${CALIB_ROT_DEG:=12,20}"
: "${CALIB_TRANS_SPEED_MPS:=0.08}"
: "${CALIB_ROT_SPEED_DPS:=20}"          # 12-20 deg in ~12-21 frames, like the translations
# counterfactuals
: "${NUM_CF:=4}"
: "${CF_MODE_PROBS:=noisy_replay:0.35,scripted_diverge:0.3,freeze_arm:0.2,gripper_flip:0.15}"
: "${CF_ARM_PROBS:=left:0.35,right:0.35,both:0.3}"
: "${CF_MIN_TCP_DEVIATION_M:=0.05}"
: "${NOISE_SIGMA_RAD:=0.12}"

SEED_START=$(( SEED_BASE + SET_ID * SEEDS_PER_SET ))
SET_DIR="${OUTPUT_BASE}/set_${SET_ID}"
LOG_DIR="${SET_DIR}/_logs"; STATE_DIR="${SET_DIR}/_state"

if [[ "${TASKS}" == "ALL" ]]; then
  TASKS=$(cd "${ROBOTWIN_ROOT}/envs" && ls *.py | sed 's/\.py$//' \
          | grep -v -E '^(_|__init__|model_utils)' | tr '\n' ' ')
fi
read -r -a TASK_ARR <<< "${TASKS}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

# Builds ARGS (an array, so empty values survive as "" instead of vanishing). Optional
# string knobs are only forwarded when set, so the generator's own default applies otherwise.
build_args() {
  ARGS=(--task_config "${TASK_CONFIG}"
    --output_dir "${SET_DIR}" --seed_start "${SEED_START}" --num_seeds "${SEEDS_PER_SET}"
    --max_episodes "${MAX_EPISODES}" --num_camera_units "${NUM_CAMERA_UNITS}"
    --overwrite "$([[ ${OVERWRITE} == 1 ]] && echo true || echo false)"
    --camera_seed_salt "${CAMERA_SEED_SALT}"
    --image_size "${IMAGE_SIZE}" --frame_stride "${FRAME_STRIDE}" --shader "${SHADER}"
    --rt_spp "${RT_SPP}" --record_wrist "${RECORD_WRIST}"
    --num_views "${NUM_VIEWS}" --azimuth_range_deg "${AZIMUTH_RANGE_DEG}"
    --elev_min_deg "${ELEV_MIN_DEG}" --elev_max_deg "${ELEV_MAX_DEG}"
    --radius_min "${RADIUS_MIN}" --radius_max "${RADIUS_MAX}"
    --min_separation_deg "${MIN_SEPARATION_DEG}" --num_candidates "${NUM_CANDIDATES}"
    --check_frames "${CHECK_FRAMES}" --gate_arms "${GATE_ARMS}"
    --wrist_pos_jitter "${WRIST_POS_JITTER}" --wrist_rot_jitter_deg "${WRIST_ROT_JITTER_DEG}"
    --record_calibration "${RECORD_CALIBRATION}" --calib_trans_m "${CALIB_TRANS_M}"
    --calib_rot_deg "${CALIB_ROT_DEG}" --calib_trans_speed_mps "${CALIB_TRANS_SPEED_MPS}"
    --calib_rot_speed_dps "${CALIB_ROT_SPEED_DPS}"
    --num_counterfactuals "${NUM_CF}" --cf_mode_probs "${CF_MODE_PROBS}"
    --cf_arm_probs "${CF_ARM_PROBS}" --cf_min_tcp_deviation_m "${CF_MIN_TCP_DEVIATION_M}"
    --noise_sigma_rad "${NOISE_SIGMA_RAD}")
  [[ -n "${EMBODIMENT}" ]] && ARGS+=(--embodiment "${EMBODIMENT}")
  [[ -n "${RANDOM_BACKGROUND}" ]] && ARGS+=(--random_background "${RANDOM_BACKGROUND}")
  [[ -n "${CLUTTERED_TABLE}" ]] && ARGS+=(--cluttered_table "${CLUTTERED_TABLE}")
  [[ -n "${RANDOM_LIGHT}" ]] && ARGS+=(--random_light "${RANDOM_LIGHT}")
  return 0
}
build_args

if [[ "${PRINT_CONFIG}" == 1 ]]; then
  echo "SET_DIR=${SET_DIR} seeds=[${SEED_START}, $((SEED_START + SEEDS_PER_SET)))"
  echo "TASKS(${#TASK_ARR[@]})=${TASKS}"
  echo "GPUS=${GPUS} MAX_PARALLEL=${MAX_PARALLEL}"
  printf '%s\n' "${ARGS[@]}" | paste - - | column -t -s $'\t'
  exit 0
fi

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
export ROBOTWIN_ROOT PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

# Fail fast on a broken environment instead of failing every task on import: a file purge
# once deleted part of the conda env mid-run, and every task started afterwards died in
# seconds. Imports RoboTwin's base task (open3d, toppra, mplib, sapien) and curobo's planner
# (needs a GPU, which the generator needs anyway).
if [[ "${DRY_RUN}" != 1 && "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  if ! (cd "${ROBOTWIN_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "${PYTHON_BIN}" -c \
        "import envs._base_task; from curobo.wrap.reacher.motion_gen import MotionGen" \
        > "${LOG_DIR}/_preflight.log" 2>&1); then
    echo "[gen_set] PREFLIGHT FAILED: the Python environment cannot import the generator stack"
    tail -5 "${LOG_DIR}/_preflight.log"
    exit 1
  fi
fi

run_task() {
  local task="$1" gpu="$2"
  if [[ -f "${STATE_DIR}/${task}.done" && "${OVERWRITE}" != 1 ]]; then
    echo "[skip] ${task} (done marker)"; return 0
  fi
  local cmd=("${PYTHON_BIN}" -m syncsim.generate --task "${task}" "${ARGS[@]}")
  if [[ "${DRY_RUN}" == 1 ]]; then echo "CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"; return 0; fi
  echo "[run ] ${task} on GPU ${gpu} -> ${LOG_DIR}/${task}.log"
  if (cd "${REPO}" && CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >> "${LOG_DIR}/${task}.log" 2>&1); then
    touch "${STATE_DIR}/${task}.done"; echo "[done] ${task}"
  else
    echo "[FAIL] ${task} (see ${LOG_DIR}/${task}.log)"
  fi
}

i=0
for task in "${TASK_ARR[@]}"; do
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  run_task "${task}" "${gpu}" &
  i=$(( i + 1 ))
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
done
wait
echo "[gen_set] set ${SET_ID} finished"
