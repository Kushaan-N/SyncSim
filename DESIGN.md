# SyncSim design notes — porting the robocasa generator to bimanual RoboTwin

What each feature is for, how it maps from the single-arm robocasa generator
(`robocasa-dev/datagen/`, see its `DESIGN.md`) to two arms on RoboTwin 2.0, and what to watch
out for. `README.md` has usage; numbers quoted here are measured unless marked otherwise.

---

## 0. Why RoboTwin 2.0 (not RoboDojo)

| | RoboTwin 2.0 | RoboDojo |
|---|---|---|
| released data **generation** code | yes — scripted experts for 50 bimanual tasks, seed search + replay | no — the repo is the evaluation client; the trajectory-synthesis pipeline is not released |
| simulator | SAPIEN 3 (Vulkan; rasteriser or ray tracer) | Isaac Sim 5.1 / Isaac Lab 2.3 |
| embodiments | aloha-agilex (dual-arm), and any pair of franka / piper / ARX-X5 / ur5-wsg as `[left, right, gap]` | ARX X5 in sim |
| license | MIT | MIT |

Counterfactual and calibration generation need to **re-execute modified controls from a
seeded initial scene**, which needs the generation code, not just an evaluator. RoboTwin also
lets one generator cover several bimanual embodiments (cross-embodiment is exactly what
SyncWorld's calibration is for). RoboDojo's sim tasks are a good *evaluation* target later.

---

## 1. The one structural change: everything is a control-stream replay

Robocasa rendered the expert by **replaying recorded MuJoCo states** and every other rollout
by **stepping actions**, and that asymmetry leaked everywhere (a 1–2 cm replay floor, one
control step of pose offset, a separate unrendered "action replay" just to measure
divergence).

RoboTwin's expert is a scripted planner whose motions all bottom out in three calls —
`Robot.set_arm_joints`, `Robot.set_gripper`, `scene.step`. `rt_env.StreamRecorder` hooks
them during the one planning run and flattens the targets in effect at every physics step
into a **ControlStream** (`left_q/dq`, `right_q/dq`, `left_g`, `right_g`, gripper rate
limits). Then:

* the **expert** video is the stream replayed from `setup_demo(seed)`;
* **counterfactuals** are edited streams (or a stream prefix + live-planned suffix);
* **calibration** is a live-planned stream.

All go through `Rollout.run`, so the expert is the divergence reference itself and there is
no state/action floor. Two guards: the expert stream is re-validated with `check_success()`
after its rendered replay (unit dropped otherwise), and RoboTwin's own joint-limit legality
check (`planned_joints_legal`) is kept.

**Determinism.** The rendered pass of a counterfactual is the replay of the stream its
*unrendered probe* executed — identical commands from an identical seeded scene. Live planning
never happens in a rendered pass.

---

## 2. Third-person cameras

Robocasa: orbit around the EE, gate on gripper segmentation pixels at N check frames, ≥45°
baseline angle between simultaneous views, 150-attempt resampling with a state restore per
candidate.

SyncSim (`cameras.py`):

* **Aim target** = centroid of *both* TCPs over the check frames (robocasa: one EE).
* **Sampling**: azimuth ±`AZIMUTH_RANGE_DEG` around the embodiment's own head-camera
  direction, explicit elevation band, true 3D radius (robocasa learned the hard way that a
  horizontal radius silently inflates distance), small aim/roll jitter.
* **Geometric pre-rejection** (free): inside the back wall (RoboTwin's wall face is at
  y = 0.4), under the table top over its footprint, under the floor.
* **Gate in one replay**: every candidate is rendered by a small *rasterised* segmentation
  camera at each check frame of a single unrendered expert replay, and candidates are
  eliminated as they fail. Cost = (#candidates × #check frames) tiny renders + one physics
  replay; no state save/restore API needed.
* **Which hands must be visible** (`GATE_ARMS`): `both` (default — a bimanual world model must
  see both hands to learn which action dims move which arm), `active` (arms that move in
  the expert), `any`.
* **Multi-view** selection is greedy in sampling order with the baseline-angle constraint.

No base-motion filter and no camera-drift filter: RoboTwin robots are fixed-base and the
cameras are world-static, so both robocasa failure modes cannot occur.

## 3. Wrist cameras

Two wrist cameras (one per arm) on RoboTwin's `left_camera` / `right_camera` links, each with a
jittered mount (±1 cm, ±5°) chosen by the robocasa **two-sided** gate on the fraction of the
wrist image covered by that arm's own gripper; falls back to the stock mount (recorded as
`stock_mount: true`).

**The gate must see what the renderer sees.** The gate's segmentation camera is rasterised,
which culls back faces: a lens jittered ~5 mm *back* into the wrist housing still sees the
scene, while the ray tracer used for the videos renders the inside of the housing -- a dark or
grey wall for the whole episode (≈10 % of pilot episodes). So (1) the position jitter is never
backward along the optical axis, and (2) each wrist candidate is also ray-traced at the gate's
first check frame (the defect is fixed to the link, so it shows there; later frames are
object close-ups where the renderers legitimately differ) and must be non-flat and correlate with the rasterised view (16×16 block
correlation ≥ 0.5; stock mounts score ≈ 0.9, lens-in-housing mounts ≤ 0.3). The scores are
recorded in `views.json` (`gate_rt_check`, `gate_rt_ok`) and checked by `verify_set.py`.

---

## 4. Calibration (bimanual)

Per arm, each of its 6 EE DoFs (x, y, z, roll, pitch, yaw — world-frame translations,
rotations about world axes through the gripper centre) goes **out → hold → back → hold** while
the other arm and both grippers hold still: **12 legs** (2 arms × 6 DoF), each filmed out
and back, i.e. 24 single-DoF sweeps. Arm order random; within an arm translations (shuffled)
then rotations (shuffled), each with a random sign. (Robocasa shuffles the translations but
uses a fixed Pitch, Yaw, Roll order for rotations.)

Robocasa commanded per-step OSC deltas, so magnitude = speed. RoboTwin is joint-position
controlled, so each leg is **planned as a straight line** from the current state — mplib's
screw planner, else a chain of short curobo hops that cannot change IK branch
(`motion.plan_to`) — and **re-timed to a constant speed** (`motion.retime`); magnitude
(distance) and speed are independent knobs. (The v1 dataset was in fact planned entirely by
the hop chain: commit 65f78c4 deleted `_screw_planner` and `plan_to`'s bare `except` swallowed
the NameError; restored, `plan_to.last` now records which planner ran, and code errors are
re-raised instead of read as "unplannable".) Translations are 5–8 cm at 8 cm/s, rotations
12–20° at 20°/s, so both last ~12–21 frames (at the earlier 40°/s a rotation leg lasted
only ~10, too few once a clip is temporally subsampled).
Unreachable legs retry at 60 % and 35 % of the magnitude, else are recorded as skipped.

**Probe, then film.** The sweep is first run unrendered; a leg whose measured motion misses
the commanded single DoF (typically y−/z− toward the table or the robot's own body) is
retried with the opposite sign or a smaller magnitude. Only the ACCEPTED legs are then
replayed and filmed (`calibration.render_clean`), so the clip never contains a failed,
partial or off-axis motion. Before this, 90 % of clips contained 1–4 rejected attempts.

**Rotations pivot at the gripper centre.** RoboTwin's end-effector point sits 0.12 m behind
the gripper centre; rotating about it moves the gripper centre 3–4 cm per 20°, which in the
recorded (gripper-centre) actions makes a roll leg look like a translation too. Legs and
scripted counterfactual motions therefore rotate about the gripper centre (filmed drift over
the 10,590 rotation legs of the full dataset: < 2 mm in 99 %, 6.4 mm at most).

`move_range.json` records the exact out/back **frame spans** of every leg, so downstream code
can cut out each leg directly instead of rediscovering it from pose deltas.

---

## 5. Counterfactuals

| mode | what | robocasa analogue |
|---|---|---|
| `noisy_replay` | expert stream + smooth sparse **end-effector** offsets (`noise_tcp_sigma_m` per point, accumulated offset capped at `noise_max_offset_m`) on left / right / both arms; offsets **persist** after their ramp (`noise_persist`). `noise_space=joint` is the v1 joint-space version | `noisy_replay` (OSC deltas integrate, so robocasa offsets persisted too) |
| `grasp_miss` | one of the arm's grasps (a gripper-close command) approached with the TCP shifted 2.5–6 cm sideways or upward: ramped in over the last 15 frames before the close, held 15 frames after it, then ramped out over 20 so the empty-handed arm rejoins the expert's path | new — the commonest real manipulation failure, made by a small offset |
| `scripted_diverge` | expert prefix → 1–2 scripted EE motions planned live for the chosen arm(s); unchosen arm + grippers **hold** | `scripted_diverge` with the opt-in `hold` gripper mode |
| `freeze_arm` | from a breakpoint one arm holds while the other keeps executing the expert | new — bimanual only: breaks coordination (handover never happens) with natural motion |
| `gripper_flip` | one arm's gripper command inverted from a sampled step to the end | `--cf_gripper_flip_prob` |

Robocasa lessons applied:

* **Make sure a perturbation acts.** Robocasa's legacy perturbation hit only the sign-read
  gripper dim and was a no-op. Here every counterfactual is probed unrendered, measured
  against the expert (max TCP deviation and max gripper rotation per arm, gripper-command
  mismatch), and re-rolled with ×1.6 magnitude unless it moves ≥ 5 cm, rotates ≥ 10°, or
  its gripper differs on ≥ 5 % of frames. Which arms actually diverged is recorded
  (`divergence.diverging_arms`); for `scripted_diverge` the unchosen arm holds, so it can
  diverge too even though `arms` names only the scripted one.
* **Make sure it is physically plausible.** A short scripted segment that executes a long
  joint-space path (RoboTwin's planner may swing through another IK branch) moved the arm
  at up to 12 m/s in the first full run. Scripted plans are now refused if any joint
  travels more than 2 rad, and timed so no joint exceeds 1.5 rad/s; every attempt of every
  mode is rejected if a TCP exceeds 1 m/s (or 1.5× the expert's own peak), and an
  implausible attempt is never kept, not even as the most divergent fallback.
  `scripted_diverge` also requires a plan for every arm it chose — previously a failed plan
  left a "counterfactual" whose only change was the other arm holding still.
* **Make sure it acts, for every mode.** `freeze_arm` draws its breakpoint only from frames
  after which the frozen arm's TCP still travels ≥ `freeze_min_remaining_m` (8 cm; else freezing
  is a no-op: late breakpoints, or the idle arm of a one-armed task); with no such frame for
  either arm the counterfactual becomes `noisy_replay`. If every attempt of any mode stays
  below the divergence gate, a planner-free `noisy_replay` fallback runs and the more
  divergent rollout is kept.
* **Reserve a suffix.** Breakpoints keep ≥ `CF_MIN_SUFFIX_FRAMES` after them (robocasa's
  `scripted_diverge` had a 7 % near-expert tail from late breakpoints).
* **Mix the task into seeds** (robocasa's `cf_task_seed_salt` fix): all RNG streams include
  the task name, seed, camera unit and a per-set salt.

**Why v1's noisy replay was replaced.** Its offsets were raw joint angles (σ 0.12 rad per
point) that persisted and added up: a random walk. A shoulder offset swings the whole arm, so
the median v1 noisy counterfactual drifted 16 cm and 11 % drifted over 30 cm (up to 50 cm) --
the arm wandered off with no purpose. Now each point is an end-effector offset at fixed
gripper orientation, turned into a joint offset by IK at that frame (`taskspace.py`), and the
total is capped. One IK solve per point, not one per frame: re-solving along the trajectory
flipped IK branches near wrist singularities (0.5–0.9 rad jumps, 1–7 m/s TCP spikes). A
constant joint offset can still move the TCP further at another pose, so its effect is
checked by forward kinematics at every frame and scaled down to the cap (`scale` in the
metadata). In a test on stack_blocks_two the new noisy rollouts stay within 4–7 cm of the
expert at the expert's speed, and `grasp_miss` lands 65–95 % of the requested offset at the
grasp (least for upward shifts). The first `grasp_miss` kept its offset to the end; on tasks
that lift high afterwards (lift_pot) the same bound then cut it to ~30 % at the grasp itself,
hence the hold-and-release window. `magnitude_m` is the offset after that scaling,
`requested_m` the draw.

**Interaction focus (`cf_focus_prob`, default 0.7).** Uniformly drawn edit points mostly
fall in free-space motion. With this probability a mode's edit point (breakpoint, flip step,
noise points) is drawn within 20 frames before to 8 after an expert interaction: a gripper
close / open, or a robot-object contact onset when contacts are recorded. This is the knob for
how interaction-dense the counterfactual set is; `scripts/contact_stats.py` measures the
result.

**The world counts as divergence.** A counterfactual whose objects end ≥ 3 cm
(`cf_min_object_deviation_m`) from where the expert's do has diverged even if the arm is
within the TCP gate -- a 3 cm grasp miss that leaves the block on the table.

Each counterfactual's `metadata.json` records mode, arms, breakpoint, the edit parameters,
the measured divergence (overall and per arm), the attempt log, and whether the task still
**succeeded** — a free label for outcome prediction.

---

## 5b. Contacts

`contacts.py` records state, not events (robocasa's `physical_state.py` philosophy): every
physics step, SAPIEN's contact list reduced to PART pairs -- `left/right_finger` (gripper
finger links), `left/right_gripper` (rest of the gripper), `left/right_arm`, `robot_base`,
`obj:<name>` (free-moving), `fix:<name>` (named static / kinematic task objects: the bell, the
stapler, a target pad, a dustbin), `art:<name>` (articulated objects), `env:<name>` (table,
wall, ground only). A fixed task object counts as an object for every interaction number:
before `fix:` existed, pressing the bell read as robot-environment contact.
Contacts inside one arm's chain are dropped (joint-internal). Steps are binned into video
frames exactly like poses, so `contacts.npz` row k is frame k. Repeated actor names (both
blocks of stack_blocks_two are "box") become `box#0`, `box#1`.

Events are derived, cheap to recompute (`annotate_contacts.py --rederive`, CPU only):
contact windows (bridging 1-frame gaps), held windows (the arm's fingers on an object while
that gripper is commanded CLOSED and the object is off every static surface -- scenery or a
fixed object -- or lifted > 1 cm; without the closed test an open finger resting on a stacked
block read as held in 6 % of sampled v1 windows), and per
rollout the share of frames with robot-object contact / an object held, event counts per
kind, and the collisions the rollout caused (arm or base on an object, robot on the table, arm
on arm, object-object pairs not touching at set-up). `contact_stats.py` turns them into the
sampling index and the per-mode distribution numbers to track across generator iterations.

Data generated before contacts existed is annotated by `annotate_contacts.py`: an unrendered
replay of each rollout's saved stream from its seed. Physics is deterministic, so the replay
is the filmed rollout -- checked per rollout against `object_poses.npz` and the TCPs in
`pose.pkl`, and reported in `contacts.json["replay_check"]`. SAPIEN poses are float32, so an
exact replay shows exactly 0: all 10,589 v1 rollouts did (84 of them, in scenes with no
free-moving object, are checked through the TCPs only). Offline annotation of the 366 pilot
rollouts that recorded contacts live reproduced them bit for bit (touch, impulse and contact
point), so offline and live contacts are interchangeable.

Note that re-generating a seed does NOT reproduce its expert: RoboTwin's GPU motion planner
is not deterministic, so two generation runs of the same seed share the scene but not the
expert motion. Comparisons between generator versions are statistical (same tasks and
seeds), not paired rollout by rollout.

## 5c. Object masks

`masks.py` writes, beside every RGB video, a label video whose pixels are part ids with the
contacts' part names (0 = nothing drawn: outside the room). They come from SAPIEN's rasterised
segmentation (the entity each pixel shows) through cameras with the recorded pose, field of
view and size; wrist cameras are re-posed from their link every frame as the recorder does.
Encoded as lossless FFV1 gray8 (bit-exact round trip).

Checked against the filmed RGB on a handover episode: of the frames' strongly red pixels,
98-100 % are labelled as the red box in both wrist views and camera B, and under 0.4 % as the
table (camera A also sees a red stapler, which is labelled as the stapler). The first version clipped at the default 5 cm near plane: the rasteriser
drops geometry nearer than that, the ray tracer that films the RGB does not, so the bottom of
a block held in front of a wrist camera was missing and the table showed through it in the
mask. Mask cameras now use a 2 mm near plane.

## 6. Output and conventions

* Layout `set_<id>/<task>/episode_<seed>/camera_poses_<m>/...` mirrors robocasa, with
  `wrist_left_rgb/` + `wrist_right_rgb/` instead of `wrist_rgb/`. Episodes are named by
  **seed** (RoboTwin skips failing seeds; a seed is the stable identity).
* `actions.npy` `[T,16]`: per arm TCP xyz + quat **xyzw** + commanded gripper opening — two
  copies of the single-arm 8-dim per-frame layout.
* `pose.pkl` keys extend robocasa's (`left_/right_gripper_pose|matrix|open`).
* Frame rate: physics 250 Hz, `FRAME_STRIDE=12` → 20.8 fps (robocasa 20 Hz).
* Every control stream is saved (`expert_stream.npz`, `*/stream.npz`), so any rollout can be
  re-rendered from a new viewpoint without re-planning.

## 7. Cluster-specific gotchas

* Blackwell GPUs need torch built for CUDA 12.8 and curobo compiled for sm_100/sm_120.
* RoboTwin's ray-tracing denoiser `oidn` fails on Blackwell ("OIDN Error: invalid handle");
  SyncSim switches to `optix` after every scene setup.
* The login node has no AVX (open3d → "Illegal instruction") and kills large `cicc` compiles:
  import/run/compile on compute nodes.
* SAPIEN's `rt` shader needs RT cores: use `rtx-batch`, not the B200 partitions.

## 8. Not ported (yet)

* Rig-paired episodes (`PAIR_SIZE`) — needs a per-pair rig validated across seeds.
* Per-video texture re-sampling (`--vary_textures`) — RoboTwin draws textures from the seed;
  across seeds the variety is already there.
* Per-instance geometry and depth from robocasa's physical-event ground truth -- contacts,
  object masks, object poses and articulated-object joints are recorded.
* Robocasa's shared-breakpoint `scripted_diverge` subcases (two futures from one prefix),
  its multi-DoF scripted motions, and its gripper flip on top of `noisy_replay` — here the
  flip is its own mode and scripted motions are single-DoF per arm.
* Camera diversity knobs: the front-bias cone about the gripper approach axis, local position
  jitter, and full orientation jitter (only roll is jittered, so the target stays centred).
  The elevation band is 15–65° rather than robocasa's −5–45° (robocasa found low views
  both more diverse and more often accepted; lowering the band here is untested).

## 9. Repairing a generated set

`verify_set.py --exclude_out bad.tsv` lists every rollout or unit with an error (including
implausible TCP speeds, missing scripted plans and, with `--dark_scan`, wrist videos that go
dark mid-episode). `scripts/redo_cf.sh` regenerates the listed counterfactuals in place with
the current gates (`generate.py --redo_list`), reusing each unit's saved expert stream, rig
and expert poses; scene and rendering settings are restored from the unit's
`base_init.json`, only the counterfactual settings are new, and the old dirs are backed up
under `<data root>/_redo_backup/`. Unit-level problems (e.g. a failed wrist mount) cannot
be repaired this way; `scripts/quarantine.py` moves whatever still fails after a redo out of
the dataset tree (`<data root>/_quarantine/`, reversible), so loaders that walk the tree
never see it.

## 10. Scaling to the RoboTwin 2.0 release

* **Experts are the release's own.** RoboTwin collects in two stages: plan per seed and save
  the joint trajectories (`_traj_data/episode<i>.pkl`), then replay them with planning off to
  record. `rt_env.run_release_expert` is that second stage with SyncSim's stream recorder, so
  our expert is the release's demonstration (no GPU-nondeterministic re-planning), and every
  release episode keeps its identity (`release.json`: zip, episode index, seed).
* **Single-arm pairs need the release's arm spacing**, which the release does not record. It
  was measured by matching each robot's initial end-effector poses against the release HDF5s:
  arx-x5 0.6 m, franka 0.8 m, ur5 0.8 m (all match to 0.1 mm; ur5 at 0.6 m even makes some
  seeds unstable). They live in `rt_env.RELEASE_EMBODIMENTS`.
* **Scene check.** The replay's scene info (object ids, clutter, textures) is compared with the
  release's `scene_info.json`. Clean scenes match; in randomized scenes a few clutter objects
  can differ (asset-version drift since the release), which `scene_matches_release` flags.
  The expert must still succeed in our scene, as for every SyncSim expert.
* **Fetching.** Only the small members of each release zip are needed, each read with one
  HTTP range request (huggingface_hub's filesystem spent ~80 s per open here, and block-wise
  reads cost one request per 128 KB). Run it from a compute node: dozens of parallel
  streams saturated the login node.
* **Throughput.** Rendering (ray tracing) is the bottleneck: one RTX PRO 6000 is ~96% busy
  with 4 workers. `scripts/queue_worker.py` spreads ~10-episode items over all workers in
  roughly FIFO order (clean before randomized), interleaving tasks and robots so a partial
  run is balanced. Claims are `mkdir`s with a heartbeat; a claim idle for 15 min is taken
  over, so preempted jobs lose at most the episode in flight.

