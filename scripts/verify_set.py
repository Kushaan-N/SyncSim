#!/usr/bin/env python
"""Audit a generated SyncSim set from the files on disk (no simulator needed).

    python scripts/verify_set.py --set_dir /scratch/.../data/set_0 [--montage_dir out/]

Checks every camera unit for: all rollout dirs and their files, every video's frame count ==
len(pose.pkl), action shape [T, 16], calibration legs reached / skipped, and summarises
counterfactual divergence per mode (how many fell below the gate) and task success rates.
With --montage_dir it also writes, per unit, a PNG grid of [first | middle | last] frames of
every view of the expert -- the quickest way to eyeball camera framing.
Exit status is 1 when any unit is structurally broken.
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np


def count_frames(path):
    import imageio.v2 as iio
    r = iio.get_reader(path)
    try:
        return r.count_frames()
    finally:
        r.close()


def read_frames(path, idx):
    import imageio.v2 as iio
    r = iio.get_reader(path)
    try:
        n = r.count_frames()
        return [r.get_data(min(max(i if i >= 0 else n + i, 0), n - 1)) for i in idx]
    finally:
        r.close()


def check_rollout(d, problems):
    pose_p = os.path.join(d, "pose.pkl")
    if not os.path.exists(pose_p):
        problems.append(f"{d}: missing pose.pkl")
        return None
    with open(pose_p, "rb") as f:
        pose = pickle.load(f)
    T = len(pose["left_gripper_pose"])
    acts = np.load(os.path.join(d, "actions.npy"))
    if acts.shape != (T, 16):
        problems.append(f"{d}: actions {acts.shape} != ({T}, 16)")
    vids = sorted(glob.glob(os.path.join(d, "*_rgb*", "video.mp4")))
    if not vids:
        problems.append(f"{d}: no videos")
    for v in vids:
        n = count_frames(v)
        if n != T:
            problems.append(f"{d}: {os.path.relpath(v, d)} has {n} frames, pose.pkl {T}")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_dir", required=True)
    ap.add_argument("--montage_dir", default="")
    ap.add_argument("--max_units", type=int, default=0)
    a = ap.parse_args()

    units = sorted(glob.glob(os.path.join(a.set_dir, "*", "episode_*", "camera_poses_*")))
    units = [u for u in units if os.path.exists(os.path.join(u, "_DONE"))]
    if a.max_units:
        units = units[:a.max_units]
    problems = []
    by_mode = defaultdict(list)
    per_task = defaultdict(lambda: {"units": 0, "expert_frames": [], "cf_success": [],
                                    "calib_skipped": 0, "calib_legs": 0})
    for u in units:
        task = u.split(os.sep)[-3]
        st = per_task[task]
        st["units"] += 1
        T = check_rollout(os.path.join(u, "expert"), problems)
        if T:
            st["expert_frames"].append(T)
        cal = os.path.join(u, "calibration")
        if os.path.isdir(cal):
            check_rollout(cal, problems)
            mr = json.load(open(os.path.join(cal, "move_range.json")))
            st["calib_legs"] += len(mr["legs"])
            st["calib_skipped"] += len(mr["skipped"])
        for cf in sorted(glob.glob(os.path.join(u, "counterfactual_*"))):
            check_rollout(cf, problems)
            md = json.load(open(os.path.join(cf, "metadata.json")))
            div = md["divergence"]
            by_mode[md["counterfactual_mode"]].append(
                (div.get("max_tcp_m", 0.0), div.get("gripper_mismatch_frac", 0.0),
                 bool(div.get("accepted")), bool(md.get("task_success")), md.get("arms")))
            st["cf_success"].append(bool(md.get("task_success")))
        if a.montage_dir:
            montage(u, a.montage_dir)

    print(f"units checked: {len(units)}")
    print(f"{'task':32s} {'units':>5s} {'exp T med':>9s} {'calib skip':>10s} {'cf succ':>7s}")
    for t, st in sorted(per_task.items()):
        med = int(np.median(st["expert_frames"])) if st["expert_frames"] else 0
        cs = np.mean(st["cf_success"]) if st["cf_success"] else float("nan")
        print(f"{t:32s} {st['units']:5d} {med:9d} {st['calib_skipped']:4d}/{st['calib_legs']:<5d} {cs:7.2f}")
    print(f"\n{'cf mode':18s} {'n':>4s} {'med max_tcp':>11s} {'p10':>6s} {'below gate':>10s} {'task succ':>9s}")
    for mode, rows in sorted(by_mode.items()):
        d = np.array([r[0] for r in rows])
        print(f"{mode:18s} {len(rows):4d} {np.median(d):11.3f} {np.percentile(d, 10):6.3f} "
              f"{sum(not r[2] for r in rows):10d} {np.mean([r[3] for r in rows]):9.2f}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:50]:
            print("  " + p)
        sys.exit(1)
    print("\nno structural problems")


def montage(unit, out_dir):
    import imageio.v2 as iio
    exp = os.path.join(unit, "expert")
    rows = []
    for v in sorted(glob.glob(os.path.join(exp, "*_rgb*", "video.mp4"))):
        rows.append(np.concatenate(read_frames(v, [0, count_frames(v) // 2, -1]), axis=1))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    os.makedirs(out_dir, exist_ok=True)
    name = "__".join(unit.split(os.sep)[-3:]) + ".jpg"
    iio.imwrite(os.path.join(out_dir, name), np.concatenate(rows, axis=0)[::2, ::2])


if __name__ == "__main__":
    main()
