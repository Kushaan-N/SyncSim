#!/usr/bin/env python
"""Add contact ground truth (contacts.npz + contacts.json, see syncsim/contacts.py) to rollouts
that were generated without it, by replaying each rollout's saved control stream UNRENDERED
from its seeded scene. Videos and every other file are left untouched.

Every rollout is a deterministic stream replay from the seeded initial scene, so the replay
is the filmed rollout; this is checked, not assumed: the replayed object positions and both
TCPs are compared with the rollout's object_poses.npz and pose.pkl, and the result is written
to contacts.json ("replay_check"). A rollout whose replay drifts more than --max_drift_m is
reported. (SAPIEN poses are float32, so an exact replay shows a drift of exactly 0.)

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

    def tcps():
        r = task.robot
        return np.stack([np.asarray(r.get_left_tcp_pose())[:3], np.asarray(r.get_right_tcp_pose())[:3]])

    track, tcp = [obj_poses()], [tcps()]
    for i in range(stream.n):
        apply_step(task.robot, stream, i)
        task.scene.step()
        tr.after_step()
        if (i + 1) % stride == 0:
            tr.end_frame()
            track.append(obj_poses())
            tcp.append(tcps())
    return tr, np.asarray(track), [e.get_name() for e in tr.objects], np.asarray(tcp)


def replay_check(rollout_dir, track, names, n_frames_contacts, tcp=None):
    """Replayed vs stored: object positions (object_poses.npz) and both TCPs (pose.pkl). The
    TCP check covers rollouts whose scene has no free-moving object (press_stapler,
    turn_switch, ...), where the object check has nothing to compare."""
    p = os.path.join(rollout_dir, "object_poses.npz")
    if not os.path.exists(p):
        return {"checked": False, "reason": "no object_poses.npz"}
    tcp_out = {}
    pp = os.path.join(rollout_dir, "pose.pkl")
    if tcp is not None and os.path.exists(pp):
        import pickle
        P = pickle.load(open(pp, "rb"))
        st = np.stack([P["left_gripper_pose"][:, :3], P["right_gripper_pose"][:, :3]], axis=1)
        Tn = min(len(st), len(tcp))
        tcp_out = {"tcp_frames_stored": int(len(st)),
                   "max_tcp_drift_m": float(np.linalg.norm(st[:Tn] - tcp[:Tn], axis=2).max())}
    z = np.load(p)
    from syncsim.contacts import unique_names
    stored, snames = z["poses"], unique_names(z["names"])
    names = unique_names(names)
    idx = [names.index(n) for n in snames if n in names]
    sidx = [i for i, n in enumerate(snames) if n in names]
    T = min(len(stored), len(track))
    out = {"checked": True, "frames_stored": int(len(stored)), "frames_replayed": int(len(track)),
           "frames_contacts": int(n_frames_contacts), "objects_matched": len(idx),
           "objects_stored": len(snames), **tcp_out}
    if idx and T:
        d = np.linalg.norm(stored[:T][:, sidx, :3] - track[:T][:, idx, :3], axis=2)
        out["max_drift_m"] = float(d.max())
        out["final_max_drift_m"] = float(d[-1].max())
    return out


def gripper_open_of(rollout_dir):
    import pickle
    pp = os.path.join(rollout_dir, "pose.pkl")
    if not os.path.exists(pp):
        return None
    P = pickle.load(open(pp, "rb"))
    return {a: np.asarray(P[f"{a}_gripper_open"]) for a in ("left", "right")}


def rederive(units, kinds, suffix):
    """Recompute <name>.json from the saved <name>.npz with the CURRENT event code (no
    simulation, no GPU): picks up label / event fixes on already-annotated data. The replay
    check, source and commit of the original annotation are kept; the dense arrays are not
    rewritten, so v1 part names stay as recorded and are relabelled on read."""
    from syncsim.contacts import events
    n = 0
    for u in units:
        for rdir, _ in rollouts_of(u, kinds):
            name = "contacts" + suffix
            npz, js = os.path.join(rdir, f"{name}.npz"), os.path.join(rdir, f"{name}.json")
            if not os.path.exists(npz):
                continue
            A = dict(np.load(npz))
            z = np.load(os.path.join(rdir, "object_poses.npz"))
            old = json.load(open(js)) if os.path.exists(js) else {}
            ev = events(A, z["poses"], z["names"], gripper_open=gripper_open_of(rdir))
            for k in ("source", "syncsim_commit", "replay_check"):
                if k in old:
                    ev[k] = old[k]
            fixed = old.get("fixed_objects", old.get("unlabelled_bodies"))
            if fixed:
                ev["fixed_objects"] = fixed
            ev["rederived"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp = js + ".tmp"
            with open(tmp, "w") as f:
                json.dump(ev, f, indent=1)
            os.replace(tmp, js)
            n += 1
    print(f"[contacts] rederived {n} rollout(s)")


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
    ap.add_argument("--units_file", default="", help="only the unit dirs listed (one per line)")
    ap.add_argument("--rederive", action="store_true",
                    help="no replay: recompute contacts.json from contacts.npz (CPU only)")
    ap.add_argument("--suffix", default="",
                    help="write contacts<suffix>.npz/json instead (e.g. _offline, to compare "
                         "with live-recorded contacts without overwriting them)")
    a = ap.parse_args()

    kinds = set(a.kinds.split(","))
    si, sn = map(int, a.shard.split("/"))
    pat = os.path.join(a.data_root, "set_*", a.task or "*", "episode_*", "camera_poses_*", "_DONE")
    units = sorted(os.path.dirname(p) for p in glob.glob(pat))
    if a.units_file:
        want = {os.path.abspath(l.strip()) for l in open(a.units_file) if l.strip()}
        units = [u for u in units if os.path.abspath(u) in want]
    if a.sets:
        keep = set(a.sets.split(","))
        units = [u for u in units if os.path.relpath(u, a.data_root).split(os.sep)[0] in keep]
    # group by task so one Generator (one RoboTwin task object) serves many units
    units.sort(key=lambda u: (u.split(os.sep)[-3], u))
    units = units[si::sn]
    if a.limit:
        units = units[:a.limit]
    if a.rederive:
        return rederive(units, kinds, a.suffix)
    from syncsim.generate import Config, Generator
    from syncsim.rt_env import ControlStream
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
            name = "contacts" + a.suffix
            if os.path.exists(os.path.join(rdir, f"{name}.json")) and not a.overwrite:
                n_skip += 1
                continue
            try:
                stream = ControlStream.load(spath)
                tr, track, names, tcp = replay_contacts(gen, seed, stream, stride)
                chk = replay_check(rdir, track, names, len(tr.frames), tcp)
                z = np.load(os.path.join(rdir, "object_poses.npz"))
                ev = tr.save(rdir, object_poses=z["poses"], object_names=z["names"],
                             extra={"source": "offline_replay", "syncsim_commit": commit,
                                    "replay_check": chk}, name=name,
                             gripper_open=gripper_open_of(rdir))
                n_done += 1
                drift = max(chk.get("max_drift_m", 0), chk.get("max_tcp_drift_m", 0))
                if drift > a.max_drift_m:
                    drifted.append((rdir, drift))
                s = ev["summary"]
                print(f"  {os.path.relpath(rdir, a.data_root)}: {s['frames']} fr  "
                      f"robot-obj {s['robot_object_contact_frac']:.2f}  held {s['held_frac']:.2f}  "
                      f"obj drift {chk.get('max_drift_m', float('nan')):.1e} m  "
                      f"tcp drift {chk.get('max_tcp_drift_m', float('nan')):.1e} m", flush=True)
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
