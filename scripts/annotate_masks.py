#!/usr/bin/env python
"""Add object masks (<view>/mask.mkv + masks.json, see syncsim/masks.py) to rollouts generated
without them: rebuild the unit's camera rig from views.json, replay each rollout's saved
control stream from its seeded scene, and capture a rasterised segmentation frame for every
camera at exactly the frames the videos were filmed at. RGB videos and every other file are
left untouched.

Checked, not assumed: the replayed TCPs are compared with pose.pkl (an exact replay shows 0;
masks.json "replay_check"), and the mask frame count must equal the video frame count.

    python scripts/annotate_masks.py --data_root <data> [--task T] [--shard i/n]

Needs a GPU node (rendering), like the generator.
"""

import argparse
import dataclasses
import glob
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def rollouts_of(unit, kinds):
    out = []
    if "expert" in kinds and os.path.isdir(os.path.join(unit, "expert")):
        out.append((os.path.join(unit, "expert"), os.path.join(unit, "expert_stream.npz")))
    if "calibration" in kinds:
        d = os.path.join(unit, "calibration")
        if os.path.exists(os.path.join(d, "stream.npz")):
            out.append((d, os.path.join(d, "stream.npz")))
    if "counterfactual" in kinds:
        for d in sorted(glob.glob(os.path.join(unit, "counterfactual_*"))):
            if os.path.exists(os.path.join(d, "stream.npz")):
                out.append((d, os.path.join(d, "stream.npz")))
    return out


def rig_of(unit, cfg):
    with open(os.path.join(unit, "views.json")) as f:
        v = json.load(f)
    return {"views": [{"mat": x["cam2world_sapien"]} for x in v["views"]],
            "wrist": {a: {"rel_mat": w["rel_mat"]} for a, w in v.get("wrist", {}).items()},
            # the filmed field of view, as recorded (falls back to the run's config)
            "fovy_deg": v.get("fovy_deg", cfg.fovy_deg),
            "wrist_fovy_deg": v.get("wrist_fovy_deg", cfg.wrist_fovy_deg),
            "image_size": v.get("image_size", cfg.image_size)}


def replay_masks(gen, seed, stream, stride, rig, size, record_wrist):
    from syncsim.masks import MaskRig
    from syncsim.rt_env import apply_step
    gen._reset(seed)
    task = gen.task
    mr = MaskRig(task, rig, size, size, record_wrist=record_wrist)

    def tcps():
        r = task.robot
        return np.stack([np.asarray(r.get_left_tcp_pose())[:3], np.asarray(r.get_right_tcp_pose())[:3]])

    mr.capture()
    tcp = [tcps()]
    for i in range(stream.n):
        apply_step(task.robot, stream, i)
        task.scene.step()
        if (i + 1) % stride == 0:
            mr.capture()
            tcp.append(tcps())
    return mr, np.asarray(tcp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes units i, i+n, ...")
    ap.add_argument("--kinds", default="expert,calibration,counterfactual")
    ap.add_argument("--units_file", default="", help="only the unit dirs listed (one per line)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many units (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_drift_m", type=float, default=0.005)
    a = ap.parse_args()

    from syncsim.generate import Config, Generator
    from syncsim.rt_env import ControlStream
    kinds = set(a.kinds.split(","))
    si, sn = map(int, a.shard.split("/"))
    pat = os.path.join(a.data_root, "set_*", a.task or "*", "episode_*", "camera_poses_*", "_DONE")
    units = sorted(os.path.dirname(p) for p in glob.glob(pat))
    if a.units_file:
        want = {os.path.abspath(l.strip()) for l in open(a.units_file) if l.strip()}
        units = [u for u in units if os.path.abspath(u) in want]
    units.sort(key=lambda u: (u.split(os.sep)[-3], u))      # one Generator serves a task's units
    units = units[si::sn]
    if a.limit:
        units = units[:a.limit]
    try:
        commit = subprocess.check_output(["git", "-C", os.path.dirname(_HERE), "rev-parse",
                                          "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    fields = {f.name for f in dataclasses.fields(Config)}
    gens = {}
    n_done = n_skip = n_fail = 0
    bad = []
    t0 = time.time()
    print(f"[masks] shard {a.shard}: {len(units)} unit(s)", flush=True)
    for u in units:
        with open(os.path.join(u, "base_init.json")) as f:
            bi = json.load(f)
        orig = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in bi["config"].items() if k in fields}
        orig.update(shader="default", output_dir=os.path.dirname(os.path.dirname(os.path.dirname(u))),
                    redo_list="", overwrite=False)
        key = (orig["task"], str(orig.get("embodiment")), orig.get("task_config"),
               orig.get("random_background"), orig.get("cluttered_table"), orig.get("random_light"),
               orig.get("random_table_height"))
        if key not in gens:
            for g in gens.values():
                g.rt.close_task(g.task, clear_cache=True)
            gens = {key: Generator(Config(**orig))}
        gen = gens[key]
        cfg = gen.cfg
        seed = int(bi["seed"])
        stride = int(orig.get("frame_stride", 12))
        rig = rig_of(u, cfg)
        for rdir, spath in rollouts_of(u, kinds):
            if os.path.exists(os.path.join(rdir, "masks.json")) and not a.overwrite:
                n_skip += 1
                continue
            try:
                stream = ControlStream.load(spath)
                mr, tcp = replay_masks(gen, seed, stream, stride, rig, rig["image_size"],
                                       bool(rig["wrist"]) and cfg.record_wrist)
                P = pickle.load(open(os.path.join(rdir, "pose.pkl"), "rb"))
                st = np.stack([P["left_gripper_pose"][:, :3], P["right_gripper_pose"][:, :3]], axis=1)
                n = min(len(st), len(tcp))
                drift = float(np.linalg.norm(st[:n] - tcp[:n], axis=2).max())
                chk = {"frames_stored": int(len(st)), "frames_replayed": int(len(tcp)),
                       "max_tcp_drift_m": drift}
                meta = mr.save(rdir, float(P.get("fps", 1.0 / (gen.task.scene.get_timestep() * stride))),
                               extra={"source": "offline_replay", "syncsim_commit": commit,
                                      "replay_check": chk})
                n_done += 1
                if drift > a.max_drift_m or len(st) != len(tcp):
                    bad.append((rdir, drift, len(st), len(tcp)))
                vis = sorted({p for v in meta["views"].values() for p in v["parts_visible"]})
                print(f"  {os.path.relpath(rdir, a.data_root)}: {len(tcp)} fr, {len(meta['views'])} views, "
                      f"{len(vis)} parts visible, tcp drift {drift:.1e} m", flush=True)
            except Exception:
                traceback.print_exc()
                n_fail += 1
                print(f"  FAILED {rdir}", flush=True)
            finally:
                gen.rt.close_task(gen.task)
    print(f"[masks] done {n_done}, skipped {n_skip}, failed {n_fail}, "
          f"mismatched {len(bad)}  ({time.time() - t0:.0f}s)")
    for r, d, s, p in bad:
        print(f"  MISMATCH drift {d:.4f} m frames stored {s} replayed {p}  {r}")


if __name__ == "__main__":
    main()
