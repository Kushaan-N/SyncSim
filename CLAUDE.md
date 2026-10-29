# CLAUDE.md — SyncSim

SyncSim generates **SyncWorld-style world-model training data on the RoboTwin 2.0 bimanual
simulator**: per episode, a bimanual visual-calibration sweep, the scripted expert, and
counterfactual rollouts, each filmed by N randomized third-person cameras plus two jittered
wrist cameras. It is the two-arm port of the single-arm RoboCasa generator in
`robocasa-dev/datagen/` (read its `DESIGN.md` for the reasoning behind most choices).

## Repo rules (from the user — follow exactly)

- **Commit one file per commit.** Never stage more than one path in a commit.
- **No Claude co-author / attribution trailer** in commit messages or PR bodies.
- Keep `README.md` (usage) and `DESIGN.md` (why) current when behaviour changes.

## Layout

```
syncsim/rt_env.py         RoboTwin bridge: task args, setup, ControlStream record/replay, expert
syncsim/cameras.py        3rd-person rig sampling + one-replay segmentation gate, wrist jitter
syncsim/motion.py         straight-line EE displacements (mplib screw, else short curobo hops), re-timed
syncsim/calibration.py    bimanual 6-DoF-per-arm sweep with exact per-leg frame spans
syncsim/counterfactual.py noisy_replay / scripted_diverge / freeze_arm / gripper_flip / grasp_miss,
                          interaction focus, divergence (TCP, gripper, objects) + speed
syncsim/taskspace.py      TCP offset -> joint offset (pinocchio FK in the ROOT frame -> world), bounded
syncsim/contacts.py       per-frame part-pair contacts (ContactTracker) + derived windows / summary
syncsim/masks.py          per-frame part-id mask videos for every camera (MaskRig), lossless FFV1
syncsim/recorder.py       cameras -> mp4s, pose.pkl / actions.npy / joints.npy / object_poses.npz
syncsim/rollout.py        steps a stream (or live segments) and captures every frame_stride steps
syncsim/generate.py       per-task driver + Config (every knob) -> `python -m syncsim.generate`;
                          `--redo_list` regenerates listed counterfactuals in place
scripts/gen_set.sh        SINGLE SOURCE OF RUN DEFAULTS (env vars) for one seed shard ("set")
scripts/submit_slurm.sh   2-D SLURM array (set x task partition) -> scripts/gen.sbatch
scripts/gen.sbatch        one array task: runs gen_set.sh for its (set, task partition)
scripts/verify_set.py     audit a generated set without re-simulating; --exclude_out, --dark_scan
scripts/watch_validate.sh validate each task as soon as it finishes (-> <data>/_validate.log)
scripts/dataset_stats.py  yield / camera / calibration / counterfactual stats for a data root
scripts/redo_cf.sh        regenerate the counterfactuals in a verify_set exclude list
scripts/quarantine.py     move what still fails out of the dataset tree (reversible)
scripts/annotate_contacts.py  add contacts to existing rollouts by unrendered stream replay (+ drift check)
scripts/submit_contacts.sh    SLURM array of annotate_contacts.py shards over a data root
scripts/contact_stats.py      contact_index.jsonl / contact_events.jsonl + per-mode interaction numbers
scripts/annotate_masks.py     add masks to existing rollouts by replay + segmentation render
scripts/submit_masks.sh       SLURM array of annotate_masks.py shards over a data root
scripts/fetch_robotwin_release.py  seeds / scene info / planner trajectories of the RoboTwin 2.0 release
scripts/queue_worker.py       shared work queue (build / work / status) for release-scale runs
scripts/queue.sbatch          one node of a queue run: WORKERS_PER_GPU workers per GPU
```

## Environment (AICR cluster)

- Conda env: `/work/umass/chuangg_umass/yuncongyang_umass/syncsim/envs/rt` (python 3.10; a
  `conda create --clone` of the old /scratch env, moved 2026-10-25: /scratch's 30-day purge
  goes by file mtime and conda keeps package build mtimes, so it kept deleting env files --
  libstdc++, libexpat, ... -- and `import sapien` broke). Never put the env back on /scratch.
  RoboTwin checkout:
  `/home/yuncongyang_umass/kn/RoboTwin` (`ROBOTWIN_ROOT`), assets downloaded there.
- GPUs are Blackwell (RTX PRO 6000 sm_120, B200 sm_100): torch is **2.7.1+cu128**, not
  RoboTwin's pinned 2.4.1; curobo v0.7.8 is compiled with `TORCH_CUDA_ARCH_LIST="10.0;12.0"`
  against conda's CUDA 12.8 (`/scratch/yuncongyang_umass/syncsim/install_curobo.sh`).
- **The login node is a QEMU VM without AVX**: `import open3d` (hence RoboTwin) dies with
  "Illegal instruction" there. Run anything that imports RoboTwin on a compute node.
- Use partition **`rtx-batch`** (account `chuangg_umass`): SAPIEN's `rt` shader needs RT cores,
  which B200s lack. `rtx-devel` is a single node and usually full.
- RoboTwin paths are relative (`./assets`), so code `chdir`s into `ROBOTWIN_ROOT`
  (`rt_env.enter_robotwin`). Run the generator as a module from the repo root.

## Invariants worth preserving

- **Every rollout is a ControlStream replay from the seeded initial scene.** Expert,
  calibration and counterfactuals go through `Rollout.run`; the expert is NOT a state replay,
  so it is the divergence reference itself (no robocasa-style 1-2 cm replay floor).
- A counterfactual is kept only if it diverges AND is physically plausible (peak TCP speed);
  never relax the plausibility gate to make a mode "succeed" — fix the planning instead.
- Counterfactuals are decided on an **unrendered probe** and then the *executed* stream is
  replayed with rendering — identical commands, deterministic physics, so the rendered
  rollout is the probed one. Keep live planning out of the rendered pass.
- Camera gate = ONE unrendered expert replay with every candidate rendered (small rasterized
  segmentation camera) at the check frames. Don't reintroduce per-candidate re-simulation.
- Shader is a global SAPIEN setting read at camera creation: always create cameras through
  `cameras.add_camera(..., shader=...)`, which restores RoboTwin's `rt` afterwards.
- Quaternions on disk are **xyzw** (robocasa convention); RoboTwin/SAPIEN are wxyz internally.
- Contacts are physics, not rendering: record them through `Rollout(..., contacts=tracker)`
  with the tracker built on the freshly reset task BEFORE the Rollout (it captures frame 0).
- Mask cameras need a near plane of a few mm (`masks.MASK_NEAR_M`): the rasteriser clips
  what the ray tracer does not, and wrist cameras see held objects from a few cm.
- Task-space edits solve IK once per edit point and bound the trajectory-wide TCP effect;
  don't go back to per-frame IK along a trajectory (IK-branch flips -> m/s spikes).
- Release experts (`--release_dir`) replay the release's saved trajectories; single-arm pairs
  need the release's arm spacing (`rt_env.RELEASE_EMBODIMENTS`, measured: don't guess).
- `actions.npy` is `[T, 16]` = per arm (TCP xyz, quat xyzw, gripper opening); layout is in
  `base_init.json["action_layout"]`. Change it only together with that layout.
