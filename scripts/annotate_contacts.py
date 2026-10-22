#!/usr/bin/env python
"""Add contact ground truth (contacts.npz + contacts.json, see syncsim/contacts.py) to rollouts
that were generated without it, by replaying each rollout's saved control stream UNRENDERED
from its seeded scene. Videos and every other file are left untouched.

Every rollout is a deterministic stream replay from the seeded initial scene, so the replay
is the filmed rollout; this is checked, not assumed: the replayed object poses are compared
with the rollout's object_poses.npz and the result is written to contacts.json
("replay_check"). A rollout whose replay drifts more than --max_drift_m is reported.

    python scripts/annotate_contacts.py --data_root <data> [--task T] [--shard i/n]

Needs a GPU node (RoboTwin creates its scene cameras on setup), like the generator.
"""

import argparse
import dataclasses
import glob
import json
import os
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


def replay_contacts(gen, seed, stream, stride):
    """Unrendered replay of `stream` from the seeded scene -> (tracker, object pose track)."""
    from syncsim.contacts import ContactTracker
    from syncsim.rt_env import apply_step
    gen._reset(seed)
    task = gen.task
    tr = ContactTracker(task, stride)

    def obj_poses():
        return np.array([np.concatenate([e.get_pose().p, e.get_pose().q]) for e in tr.objects],
                        dtype=np.float64).reshape(-1, 7)

    track = [obj_poses()]
    for i in range(stream.n):
        apply_step(task.robot, stream, i)
        task.scene.step()
        tr.after_step()
        if (i + 1) % stride == 0:
            tr.end_frame()
            track.append(obj_poses())
    return tr, np.asarray(track), [e.get_name() for e in tr.objects]


def replay_check(rollout_dir, track, names, n_frames_contacts):
    p = os.path.join(rollout_dir, "object_poses.npz")
    if not os.path.exists(p):
        return {"checked": False, "reason": "no object_poses.npz"}
    z = np.load(p)
    from syncsim.contacts import unique_names
    stored, snames = z["poses"], unique_names(z["names"])
    names = unique_names(names)
    idx = [names.index(n) for n in snames if n in names]
    sidx = [i for i, n in enumerate(snames) if n in names]
    T = min(len(stored), len(track))
    out = {"checked": True, "frames_stored": int(len(stored)), "frames_replayed": int(len(track)),
           "frames_contacts": int(n_frames_contacts), "objects_matched": len(idx),
           "objects_stored": len(snames)}
    if idx and T:
        d = np.linalg.norm(stored[:T][:, sidx, :3] - track[:T][:, idx, :3], axis=2)
        out["max_drift_m"] = float(d.max())
        out["final_max_drift_m"] = float(d[-1].max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--sets", default="", help="comma list of set dirs to include (default all)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes units i, i+n, ...")
    ap.add_argument("--kinds", default="expert,calibration,counterfactual")
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
    if a.sets:
        keep = set(a.sets.split(","))
        units = [u for u in units if os.path.relpath(u, a.data_root).split(os.sep)[0] in keep]
    # group by task so one Generator (one RoboTwin task object) serves many units
    units.sort(key=lambda u: (u.split(os.sep)[-3], u))
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
    drifted = []
    t0 = time.time()
    print(f"[contacts] shard {a.shard}: {len(units)} unit(s)", flush=True)
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
        seed = int(bi["seed"])
        stride = int(orig.get("frame_stride", 12))
        for rdir, spath in rollouts_of(u, kinds):
            if os.path.exists(os.path.join(rdir, "contacts.json")) and not a.overwrite:
                n_skip += 1
                continue
            try:
                stream = ControlStream.load(spath)
                tr, track, names = replay_contacts(gen, seed, stream, stride)
                chk = replay_check(rdir, track, names, len(tr.frames))
                z = np.load(os.path.join(rdir, "object_poses.npz"))
                ev = tr.save(rdir, object_poses=z["poses"], object_names=z["names"],
                             extra={"source": "offline_replay", "syncsim_commit": commit,
                                    "replay_check": chk})
                n_done += 1
                if chk.get("max_drift_m", 0) > a.max_drift_m:
                    drifted.append((rdir, chk["max_drift_m"]))
                s = ev["summary"]
                print(f"  {os.path.relpath(rdir, a.data_root)}: {s['frames']} fr  "
                      f"robot-obj {s['robot_object_contact_frac']:.2f}  held {s['held_frac']:.2f}  "
                      f"drift {chk.get('max_drift_m', float('nan')):.1e} m", flush=True)
            except Exception:
                traceback.print_exc()
                n_fail += 1
                print(f"  FAILED {rdir}", flush=True)
            finally:
                gen.rt.close_task(gen.task)
    print(f"[contacts] done {n_done}, skipped {n_skip}, failed {n_fail}, "
          f"drift > {a.max_drift_m} m: {len(drifted)}  ({time.time() - t0:.0f}s)")
    for r, d in drifted:
        print(f"  DRIFT {d:.4f} m  {r}")


if __name__ == "__main__":
    main()
