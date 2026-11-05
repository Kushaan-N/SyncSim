# SyncSim

SyncSim is a data generator for **action-conditioned world models of two-armed robots**: models
that, given what a robot sees and the actions it is about to take, predict what it will see
next. It runs on the [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) simulator
and turns each scripted task demonstration into an episode that shows not only the task done
right, but also many physically plausible ways it can go differently.

The data follows the visual-calibration recipe of SyncWorld (a world model that watches a short
calibration clip to learn how actions map to pixels in a new camera setup), extended from one
arm to two.

## The dataset

The full run over the RoboTwin 2.0 release is published on the Hugging Face Hub as
[`ohthatjeff/syncsim-rt2`](https://huggingface.co/datasets/ohthatjeff/syncsim-rt2):

* **96,402 episodes**, ~2 TB
* **50 tasks**: handing objects between arms, stacking blocks and bowls, placing objects in
  baskets or on stands, hanging a mug, opening a laptop or microwave, pressing a stapler or
  bell, ...
* **4 robots**: aloha-agilex, arx-x5, franka and ur5
* **Two scene styles**: clean tables, and domain-randomized ones (textures, lighting, clutter)
* **Per episode**: one expert demonstration, one calibration clip and four counterfactuals,
  each filmed by four cameras, with contact and segmentation labels

The dataset card explains the file layout and how to download only the parts you need.

## What an episode contains

### The expert

The task performed successfully by RoboTwin's scripted planner. For the published dataset these
are the expert trajectories of the RoboTwin 2.0 release, replayed in the same seeded scenes.

### A calibration clip

Each arm's end effector moves out and back along each of its six degrees of freedom (x, y, z,
roll, pitch, yaw) while everything else holds still: 12 short motions in total, with the exact
frames of each one recorded. A model can watch this clip to learn how actions show up on screen
from these particular cameras, before it has to predict anything.

### Counterfactuals

Rollouts in the same scene where the actions are deliberately changed part-way through, so the
outcome differs from the expert. There are five kinds:

| Kind | What changes |
|---|---|
| **grasp miss** | one grasp approaches a few centimetres off, so the fingers close beside the object or catch it off-centre |
| **noisy replay** | small, smooth offsets on one or both arms: an imprecise arm, not a wandering one |
| **scripted diverge** | the expert up to a point, then different scripted motions |
| **freeze arm** | one arm stops while it still had work to do, and the other carries on |
| **gripper flip** | a gripper opens or closes when it shouldn't |

A counterfactual is kept only if it really differs from the expert, stays physically plausible
(no end effector faster than 1 m/s), and remains a believable variation of the task: no arm
wandering off, no flung objects, no smashed distractors. Most perturbations are placed near the
expert's grasps and contacts, so they change what happens to an object rather than just moving
an arm through empty space. Some counterfactuals still complete the task, many don't, and every
one records whether it succeeded.

### Cameras

Every rollout is filmed by four cameras at 512×512 and ~21 fps:

* **two third-person cameras**, placed at random around the robot for each episode (varied
  angle, height and distance, at least 45° apart), and kept only if both hands stay visible
  throughout;
* **one camera on each wrist**, with a slightly jittered mount.

The expert, calibration and counterfactuals of an episode share the same cameras, so they can
be compared frame by frame.

### Labels

Besides video, every rollout comes with:

* **actions and robot state**: both arms' end-effector poses, gripper openings and joint
  positions, frame by frame;
* **object poses** for every movable object, frame by frame;
* **contacts**: which robot part (fingers, gripper, arm, base) touches which object, the table
  or the other arm, with force; plus when each object was grasped, held and released;
* **segmentation masks**: a lossless video next to each camera's video, whose pixel values name
  what is visible (each arm's parts, every object, the table), using the same names as the
  contacts.

Together these make it possible to pick out specific events, e.g. every grasp that slipped, or
every collision between an arm and an object.

## How it works

Every rollout (expert, calibration, counterfactual) is a replay of a recorded, physics-step
by physics-step control stream from the same seeded starting scene. The simulator is
deterministic, so the expert and its counterfactuals differ only where their actions differ,
and any rollout can be re-rendered later from new viewpoints. Counterfactuals are first tried
out without rendering, and only the ones that pass the checks above are rendered.

The reasoning behind these choices is in [`DESIGN.md`](DESIGN.md).

## Running it

Requirements: RoboTwin 2.0 with its assets (point `ROBOTWIN_ROOT` at it), its Python
environment plus `imageio-ffmpeg`, and an NVIDIA GPU with ray tracing (or `--shader default`
for the rasterizer).

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
python -m syncsim.generate --task handover_block --seed_start 0 --num_seeds 5 --output_dir data/set_0
python -m syncsim.generate --help     # every option
```

Larger runs go through the scripts in `scripts/`: SLURM submission, a shared work queue for
replaying the whole RoboTwin release, offline validation, regeneration of bad counterfactuals,
and adding contact or mask labels to existing data. Each script documents its usage at the top.

| Path | Contents |
|---|---|
| `syncsim/generate.py` | per-task generator and its configuration |
| `syncsim/rt_env.py` | RoboTwin bridge; control-stream recording and replay |
| `syncsim/cameras.py` | camera sampling, visibility check, wrist-mount jitter |
| `syncsim/calibration.py` | bimanual calibration sweep |
| `syncsim/counterfactual.py` | counterfactual kinds and divergence measurement |
| `syncsim/contacts.py`, `syncsim/masks.py` | contact and segmentation labels |
| `syncsim/motion.py` | straight-line end-effector motion planning |
| `syncsim/recorder.py`, `syncsim/rollout.py` | rendering, file output, stepping |
| `scripts/` | batch generation, SLURM, validation, repair and statistics |
