# SyncSim

## About

SyncSim generates training data for **action-conditioned world models of bimanual (two-arm)
robots**. It runs on the [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
simulator and turns each scripted demonstration into a small, multi-view episode bundle:

* the **expert** demonstration,
* a **visual calibration** clip that shows how each arm's actions move it on screen, and
* **counterfactual** rollouts: the same scene with different actions,

each filmed by several randomized third-person cameras and a camera on each wrist. The data
follows the visual-calibration recipe of SyncWorld (a world model that reads a short
calibration clip to learn how actions map to pixels in a new setup), extended from one arm to
two.

## What it generates

* **Randomized cameras.** Each episode gets N simultaneous third-person cameras sampled around
  both grippers, at varied azimuth, elevation and distance, and at least 45° apart. A view is
  kept only if both hands stay visible at several points of the episode, checked with
  segmentation so occlusion counts. Each wrist camera gets a slightly jittered mount, kept
  only if its own gripper stays in frame and its ray-traced view is clean.
* **Bimanual visual calibration.** Every end-effector degree of freedom of each arm (x, y, z,
  roll, pitch, yaw) is swept out and back in a straight line while everything else holds
  still: 12 legs (2 arms × 6 DoF), each out and back. The exact frame span of every leg is
  recorded.
* **Counterfactual episodes**, each checked to actually differ from the expert and to stay
  physically plausible (no end effector faster than 1 m/s):
  * `grasp_miss`: one of the expert's grasps approached a few cm off (sideways or above),
    so the fingers close beside the object or catch it off-centre;
  * `noisy_replay`: small, smooth end-effector offsets (capped at 6 cm) on the left, right,
    or both arms -- an imprecise arm, not a wandering one;
  * `scripted_diverge`: the expert up to a breakpoint, then scripted end-effector motions;
  * `freeze_arm`: one arm stops, at a point where it still had somewhere to go, while the
    other continues, which breaks the two arms' coordination;
  * `gripper_flip`: one gripper opens or closes when it shouldn't.
* **Contact ground truth**, per frame: which robot part (each gripper's fingers and body, each
  arm, the base) touches which object, the table or the other arm, with force and contact
  point; plus contact windows, grasp / hold windows, collisions the rollout caused, and how
  far each touched object moved. Counterfactual edit points are drawn near the expert's
  grasps and contacts (`cf_focus_prob`), so most of them change what happens to an object.
* **Any two-arm embodiment RoboTwin supports**, including mixed pairs (e.g. a Piper left arm
  and a Franka right arm).

Every rollout is a replay of a recorded per-physics-step control stream from the same seeded
scene, so expert and counterfactuals are directly comparable, and any rollout can be
re-rendered from new viewpoints. Design rationale: [`DESIGN.md`](DESIGN.md).

## Requirements

* RoboTwin 2.0 with its assets, set up per its instructions; point `ROBOTWIN_ROOT` at it.
* Its Python environment, plus `imageio-ffmpeg`.
* An NVIDIA GPU with ray-tracing support for the default `rt` shader (`--shader default`
  uses the rasterizer instead).

## Usage

One task, a few seeds:

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
python -m syncsim.generate --task handover_block --seed_start 0 --num_seeds 5 \
    --output_dir data/set_0
python -m syncsim.generate --help          # every option
```

Many tasks for one seed shard ("set"). `scripts/gen_set.sh` holds all generation defaults as
environment variables:

```bash
SET_ID=0 TASKS="handover_block lift_pot" ./scripts/gen_set.sh
SET_ID=0 PRINT_CONFIG=1 ./scripts/gen_set.sh                         # show the resolved config
SET_ID=0 EMBODIMENT="piper,franka-panda,0.6" ./scripts/gen_set.sh    # two different arms
```

The full task suite on a SLURM cluster, as an array of (set × task group) jobs:

```bash
DRY_RUN=1 NUM_SETS=2 PARTITION=<partition> ACCOUNT=<account> ./scripts/submit_slurm.sh
NUM_SETS=2 PARTITION=<partition> ACCOUNT=<account> ./scripts/submit_slurm.sh
```

Runs resume where they stopped: finished units are marked `_DONE`, and seeds that were
dropped (expert failure, unstable scene, camera gate, or an error) are remembered in
`<task>/_dropped.json`.

Check a generated set without the simulator (file structure, frame counts, counterfactual
divergence per mode, task success rates; `--montage_dir` saves frame grids of every view):

```bash
python scripts/verify_set.py --set_dir data/set_0 --montage_dir montage/
```

To repair a set, list every bad rollout (`--dark_scan` also decodes every wrist video to
catch views that go dark mid-episode), then regenerate the listed counterfactuals in place
on a GPU node (the originals are backed up under `<data>/_redo_backup/`). Validate again,
and move whatever still fails (e.g. a unit whose wrist mount failed, which a redo cannot fix)
out of the dataset tree into `<data>/_quarantine/`; `--restore` moves it back.

```bash
python scripts/verify_set.py --set_dir data/set_0 --dark_scan --exclude_out bad.tsv
LIST=bad.tsv DRY_RUN=1 bash scripts/redo_cf.sh      # show what would be regenerated
LIST=bad.tsv bash scripts/redo_cf.sh
python scripts/verify_set.py --set_dir data/set_0 --dark_scan --exclude_out still_bad.tsv
python scripts/quarantine.py --list still_bad.tsv
```

While a run is in progress, validate each task as soon as it finishes (verdicts go to
`<data>/_validate.log`), and summarise a whole data root (yield per task, camera spread,
calibration coverage, counterfactual divergence and success rates):

```bash
DATA_ROOT=data nohup bash scripts/watch_validate.sh &
python scripts/dataset_stats.py --root data --json_out stats.json
```

Add contact labels to a data root generated without them (an unrendered replay of every
rollout's saved control stream; checked against the stored object poses, videos untouched),
then build the interaction index and the distribution numbers per counterfactual mode:

```bash
DATA_ROOT=data N_SHARDS=48 bash scripts/submit_contacts.sh
python scripts/contact_stats.py --data_root data --by_task
python scripts/contact_stats.py --data_root data_v2 --compare data/contact_index.jsonl
```

`contact_index.jsonl` (one row per rollout) and `contact_events.jsonl` (one row per contact
window) are the tables to sample training data from, e.g. every rollout with an arm-object
collision, or every grasp that ended with the object dropped.

## Output

```
set_<id>/<task>/episode_<seed>/camera_poses_<m>/
    views.json            camera rig: per-view pose, azimuth / elevation / distance, visibility
                          check results; wrist mounts
    base_init.json        seed, embodiment, scene randomization, action layout, full config
    expert_stream.npz     the expert's control stream (replayable)
    calibration/          + move_range.json: per leg arm, DoF, sign, amount, frame spans,
                            measured motion
    expert/               + metadata.json
    counterfactual_<k>/   + metadata.json: mode, arms, breakpoint, divergence from the expert
                            (overall and per arm), whether the task still succeeded

each rollout directory:
    agentview_rgb/video.mp4, agentview_rgb_1/video.mp4    third-person views
    wrist_left_rgb/video.mp4, wrist_right_rgb/video.mp4   wrist views
    pose.pkl       per arm: gripper pose [T,7] (xyz + quat xyzw), 4x4 matrix, gripper opening
    actions.npy    [T,16]: per arm gripper-centre xyz, quat xyzw, opening (left 0:8, right 8:16)
    joints.npy     measured joint positions of both arms
    object_poses.npz   world pose of every movable object per frame
    contacts.npz   per frame x part pair: touching share of the frame, impulse, contact point
    contacts.json  contact windows (frame x to y, force, object motion during / after),
                   held windows per arm, and a summary (interaction share, event counts)
```

Defaults: 512×512 video at ~20.8 fps (physics at 250 Hz, a frame every 12 steps), two
third-person views, four counterfactuals per episode.

## Repository layout

| path | contents |
|---|---|
| `syncsim/generate.py` | per-task generator and its configuration |
| `syncsim/rt_env.py` | RoboTwin bridge; control-stream recording and replay |
| `syncsim/cameras.py` | camera sampling, visibility check, wrist-mount jitter |
| `syncsim/calibration.py` | bimanual calibration sweep |
| `syncsim/counterfactual.py` | counterfactual modes and divergence measurement |
| `syncsim/motion.py` | straight-line end-effector motion planning |
| `syncsim/recorder.py`, `syncsim/rollout.py` | rendering, file output, stepping |
| `scripts/` | batch generation, SLURM submission, offline verification and statistics |
