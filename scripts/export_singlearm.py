#!/usr/bin/env python
"""Export a SyncSim set as SINGLE-ARM episodes that SyncWorld's stock loader
(`gripperhead_fdm_dataset.py`) reads unchanged -- for smoke-testing the training pipeline
before it is made bimanual. Videos are symlinked, nothing large is copied.

    python scripts/export_singlearm.py --set_dir data/set_0 --out_root export/set_0 \
        [--tasks a,b] [--arms left,right] [--max_episodes N]

Per (task, arm, episode, camera unit, third-person view v) it writes

    <out_root>/<task>__<arm>/episode_<seed>/camera_poses_<m>_v<v>/
        expert/ calibration/ counterfactual_<k>/
            agentview_rgb/video.mp4 -> view v          (the loader's main camera)
            wrist_rgb/video.mp4     -> that arm's wrist camera
            pose.pkl                gripper_matrix / gripper_open / gripper_pose of that arm
            metadata.json           -> original
        calibration/move_range.pkl  movement_order of that arm's legs ("X+", "Pitch-", ...), in
                                    SyncWorld's axis convention (see `loader_token`)

LIMITS (why this is a smoke test, not training data): the action is one arm's while both
arms move in the video, and counterfactuals that edited only the OTHER arm would look like
the expert in action space -- those are skipped.
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

VIEW_DIR = lambda v: "agentview_rgb" if v == 0 else f"agentview_rgb_{v}"
# SyncWorld's calib_segments axis order: x, y, z, then ZYX euler angles yaw, pitch, roll
LOADER_AXES = ("X", "Y", "Z", "Yaw", "Pitch", "Roll")


def _euler_zyx_deg(R):
    """(yaw, pitch, roll) in degrees, ZYX -- the same convention as SyncWorld's
    calib_segments._rotmat_to_euler_zyx."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if np.cos(pitch) < 1e-8:
        return np.degrees([np.arctan2(-R[0, 1], R[1, 1]) if R[2, 0] < 0
                           else np.arctan2(R[0, 1], R[1, 1]), pitch, 0.0])
    return np.degrees([np.arctan2(R[1, 0], R[0, 0]), pitch, np.arctan2(R[2, 1], R[2, 2])])


def loader_token(mats, leg):
    """movement_order token of one calibration leg as SyncWorld's loader sees it.

    SyncSim legs rotate about WORLD axes, but the loader names rotations by the ZYX euler
    angle of the gripper matrix that changes most -- which world axis that is depends on the
    gripper's resting orientation (on aloha-agilex a world-x "roll" leg moves euler pitch,
    with the opposite sign). So the token is measured from the leg's own filmed poses."""
    a, b = leg["out_start_frame"], min(leg["out_end_frame"], len(mats) - 1)
    if leg["dof"] in ("x", "y", "z"):
        d = mats[b][:3, 3] - mats[a][:3, 3]
        k = int(np.argmax(np.abs(d)))
    else:
        d = (_euler_zyx_deg(mats[b][:3, :3]) - _euler_zyx_deg(mats[a][:3, :3]) + 180) % 360 - 180
        k = 3 + int(np.argmax(np.abs(d)))
        d = np.concatenate([np.zeros(3), d])
    return LOADER_AXES[k] + ("+" if d[k] >= 0 else "-")


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def export_rollout(src, dst, arm, view):
    vsrc = os.path.join(src, VIEW_DIR(view), "video.mp4")
    if not os.path.isfile(vsrc):
        return False
    with open(os.path.join(src, "pose.pkl"), "rb") as f:
        P = pickle.load(f)
    _link(vsrc, os.path.join(dst, "agentview_rgb", "video.mp4"))
    wsrc = os.path.join(src, f"wrist_{arm}_rgb", "video.mp4")
    if os.path.isfile(wsrc):
        _link(wsrc, os.path.join(dst, "wrist_rgb", "video.mp4"))
    if os.path.isfile(os.path.join(src, "metadata.json")):
        _link(os.path.join(src, "metadata.json"), os.path.join(dst, "metadata.json"))
    out = {
        "gripper_matrix": np.asarray(P[f"{arm}_gripper_matrix"], dtype=np.float32),
        "gripper_open": np.asarray(P[f"{arm}_gripper_open"], dtype=np.float32),
        "gripper_pose": np.asarray(P[f"{arm}_gripper_pose"], dtype=np.float32),   # xyz + xyzw
        "fps": P.get("fps"), "save_stride": P.get("save_stride"),
        "syncsim_arm": arm, "syncsim_view": view,
    }
    with open(os.path.join(dst, "pose.pkl"), "wb") as f:
        pickle.dump(out, f)
    return True


def keep_cf(md, arm):
    """Skip counterfactuals whose edit is invisible in this arm's action stream."""
    return md.get("arms") in (arm, "both")


def export_unit(unit, out_unit_base, arm, n_views):
    made = 0
    for v in range(n_views):
        dst_unit = f"{out_unit_base}_v{v}"
        if not export_rollout(os.path.join(unit, "expert"), os.path.join(dst_unit, "expert"), arm, v):
            continue
        cal = os.path.join(unit, "calibration")
        if os.path.isdir(cal) and export_rollout(cal, os.path.join(dst_unit, "calibration"), arm, v):
            mr = json.load(open(os.path.join(cal, "move_range.json")))
            with open(os.path.join(cal, "pose.pkl"), "rb") as f:
                mats = np.asarray(pickle.load(f)[f"{arm}_gripper_matrix"], dtype=np.float64)
            order = [loader_token(mats, l) for l in mr["legs"] if l["arm"] == arm]
            with open(os.path.join(dst_unit, "calibration", "move_range.pkl"), "wb") as f:
                pickle.dump({"movement_order": order, "syncsim_arm": arm,
                             "legs": [l for l in mr["legs"] if l["arm"] == arm]}, f)
        for cf in sorted(glob.glob(os.path.join(unit, "counterfactual_*"))):
            md = json.load(open(os.path.join(cf, "metadata.json")))
            if keep_cf(md, arm):
                export_rollout(cf, os.path.join(dst_unit, os.path.basename(cf)), arm, v)
        made += 1
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--max_episodes", type=int, default=0, help="per task (0 = all)")
    a = ap.parse_args()
    keep = set(a.tasks.split(",")) if a.tasks else None
    n_units = 0
    for task_dir in sorted(glob.glob(os.path.join(a.set_dir, "*"))):
        task = os.path.basename(task_dir)
        if task.startswith("_") or not os.path.isdir(task_dir) or (keep and task not in keep):
            continue
        eps = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
        eps = [e for e in eps if glob.glob(os.path.join(e, "camera_poses_*", "_DONE"))]
        if a.max_episodes:
            eps = eps[:a.max_episodes]
        for ep in eps:
            for unit in sorted(glob.glob(os.path.join(ep, "camera_poses_*"))):
                if not os.path.exists(os.path.join(unit, "_DONE")):
                    continue
                n_views = json.load(open(os.path.join(unit, "views.json")))["num_views"]
                for arm in a.arms.split(","):
                    base = os.path.join(a.out_root, f"{task}__{arm}", os.path.basename(ep),
                                        os.path.basename(unit))
                    n_units += export_unit(unit, base, arm, n_views)
    print(f"exported {n_units} single-arm unit(s) to {a.out_root}")


if __name__ == "__main__":
    main()
