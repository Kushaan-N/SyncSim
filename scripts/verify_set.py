#!/usr/bin/env python
"""Validate a generated SyncSim set from the files on disk (no simulator needed).

    python scripts/verify_set.py --set_dir data/set_0 [--montage_dir out/] [--max_units N]

Every check is an ASSERTION about what the data must be, so a regression shows up as an
ERROR instead of having to be spotted by eye. Per camera unit:

structure   every rollout has pose.pkl / actions.npy / joints.npy / object_poses.npz /
            metadata, every video has exactly len(pose.pkl) frames, actions == pose.pkl
videos      sampled frames are not blank (mean / std bounds)
rig         simultaneous views are >= min_separation_deg apart; wrist mounts recorded and
            their ray-traced view matched the rasterised one in the gate
calibration each filmed leg moves ONLY the commanded DoF of ONLY the commanded arm, by
            roughly the commanded amount; translation legs do not rotate, rotation legs do
            not translate the gripper centre; the arm returns to where the leg started;
            NOTHING moves between legs (no stray / rejected motions in the clip); grippers
            never change; frames outside legs are still
expert      the task succeeds
counterfact divergence recomputed from pose.pkl equals metadata (render == probe);
            accepted by the divergence gate; mode semantics hold:
              scripted_diverge / freeze_arm: identical to the expert before the breakpoint
              freeze_arm: the frozen arm's command is constant after the breakpoint
                          (being shoved by the other arm is reported, not failed)
              gripper_flip: identical before the flip, flipped gripper differs after it
              noisy_replay: identical to the expert before the first perturbation ramp
objects     object_poses.npz covers every frame; articulated objects have joint tracks

Exit status 1 when any ERROR is found. WARN lines are reported but do not fail.
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

# ---- tolerances (metres / degrees) -------------------------------------------------------
PREFIX_TOL_M = 1e-4          # "identical to the expert" -- replay is deterministic
DIV_TOL_M = 1e-4             # recomputed vs recorded divergence
LEG_MIN_FRAC = 0.6           # leg must realise >= 60 % of its commanded amount
LEG_OFFAXIS_M = 0.012        # off-axis drift allowed on a translation leg
LEG_SPURIOUS_ROT_DEG = 3.0   # rotation allowed on a translation leg
ROT_LEG_TRANS_M = 0.012      # gripper-centre translation allowed on a rotation leg
OTHER_ARM_M = 0.004          # the arm that is NOT calibrating may not move more than this
OTHER_ARM_DEG = 1.5
RETURN_M = 0.012             # a leg must end within this of where it started
STILL_STEP_M = 0.0025        # max per-frame TCP motion outside legs ("nothing moves")
STILL_STEP_DEG = 0.6
SETTLE_FRAMES = 3            # frames after a leg / breakpoint allowed for the arm to settle
FREEZE_DRIFT_M = 0.012       # frozen arm motion allowed after the breakpoint (+settle)


class Report:

    def __init__(self):
        self.errors, self.warns = [], []
        self.counts = Counter()

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}")
        self.counts["ERROR:" + msg.split(":")[0][:60]] += 1

    def warn(self, where, msg):
        self.warns.append(f"{where}: {msg}")
        self.counts["WARN:" + msg.split(":")[0][:60]] += 1


# ---- io helpers ----------------------------------------------------------------------------
def load_pose(d):
    with open(os.path.join(d, "pose.pkl"), "rb") as f:
        return pickle.load(f)


def tcp(pose):
    return np.stack([pose["left_gripper_pose"][:, :3], pose["right_gripper_pose"][:, :3]], 1)


def quat_xyzw(pose, arm):
    return np.asarray(pose[f"{arm}_gripper_pose"][:, 3:7])


def ang_deg(qa, qb):
    d = np.abs(np.sum(np.asarray(qa) * np.asarray(qb), axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, -1, 1)))


def video_info(path, sample=(0, 0.5, -1)):
    import imageio.v2 as iio
    r = iio.get_reader(path)
    try:
        n = r.count_frames()
        stats = []
        for s in sample:
            i = int(s * (n - 1)) if isinstance(s, float) else (n + s if s < 0 else s)
            fr = r.get_data(max(0, min(n - 1, i))).astype(np.float32)
            stats.append((float(fr.mean()), float(fr.std())))
        return n, stats
    finally:
        r.close()


# ---- checks ------------------------------------------------------------------------------
def check_rollout(d, rep, check_videos=True):
    req = ["pose.pkl", "actions.npy", "joints.npy", "object_poses.npz", "metadata.json"]
    if os.path.basename(d) == "calibration":
        req = [r for r in req if r != "metadata.json"] + ["move_range.json"]
    for r in req:
        if not os.path.exists(os.path.join(d, r)):
            rep.err(d, f"missing file: {r}")
            return None
    pose = load_pose(d)
    T = len(pose["left_gripper_pose"])
    acts = np.load(os.path.join(d, "actions.npy"))
    if acts.shape != (T, 16):
        rep.err(d, f"actions shape: {acts.shape} != ({T}, 16)")
    else:
        if np.abs(acts[:, 0:3] - pose["left_gripper_pose"][:, :3]).max() > 1e-4 or \
           np.abs(acts[:, 8:11] - pose["right_gripper_pose"][:, :3]).max() > 1e-4:
            rep.err(d, "actions/pose mismatch: actions.npy positions != pose.pkl")
        if not np.isfinite(acts).all():
            rep.err(d, "non-finite actions")
    jo = np.load(os.path.join(d, "joints.npy"))
    if len(jo) != T:
        rep.err(d, f"joints length: {len(jo)} != {T}")
    ob = np.load(os.path.join(d, "object_poses.npz"))
    if ob["poses"].shape[0] != T:
        rep.err(d, f"object poses length: {ob['poses'].shape[0]} != {T}")
    if "art_names" not in ob.files:
        rep.err(d, "object_poses.npz: no articulated-object record (art_names)")
    else:
        for i in range(len(ob["art_names"])):
            if ob[f"art{i}_qpos"].shape[0] != T:
                rep.err(d, f"articulation track length: art{i} {ob[f'art{i}_qpos'].shape[0]} != {T}")
    vids = sorted(glob.glob(os.path.join(d, "*_rgb*", "video.mp4")))
    if len(vids) < 3:
        rep.err(d, f"videos: only {len(vids)} video(s)")
    if check_videos:
        for v in vids:
            n, stats = video_info(v)
            if n != T:
                rep.err(d, f"video frame count: {os.path.relpath(v, d)} {n} != pose {T}")
            for m, s in stats:
                if not (8 < m < 247) or s < 4:
                    rep.err(d, f"blank-looking frame: {os.path.relpath(v, d)} mean {m:.0f} std {s:.1f}")
                    break
    return pose


def check_rig(u, rep):
    v = json.load(open(os.path.join(u, "views.json")))
    pos = [np.asarray(x["position"]) for x in v["views"]]
    tgt = np.asarray(v["aim_target"])
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            a, b = tgt - pos[i], tgt - pos[j]
            ang = np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
            if ang < v["min_separation_deg"] - 1e-3:
                rep.err(u, f"views too close: views {i},{j} {ang:.1f} deg")
    for arm in ("left", "right"):
        if arm not in v.get("wrist", {}):
            rep.err(u, f"wrist mount missing: {arm}")
            continue
        # the ray-traced wrist view must have matched the rasterised one in the gate (a lens
        # inside the wrist housing renders a dark / grey wall only the ray tracer shows)
        ok = v["wrist"][arm].get("gate_rt_ok")
        if ok is None:
            rep.warn(u, f"wrist mount {arm}: no ray-traced gate record (generated before the check)")
        elif not ok:
            rep.err(u, f"wrist mount {arm}: ray-traced view failed the gate {v['wrist'][arm].get('gate_rt_check')}")
    return v


def _leg_axis(dof):
    return {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}[dof]


def check_calibration(cal, pose, rep):
    mr = json.load(open(os.path.join(cal, "move_range.json")))
    T = len(pose["left_gripper_pose"])
    P = tcp(pose)
    Q = {a: quat_xyzw(pose, a) for a in ("left", "right")}
    for arm in ("left", "right"):
        g = np.asarray(pose[f"{arm}_gripper_open"])
        if np.ptp(g) > 1e-3:
            rep.err(cal, f"gripper changes during calibration: {arm} range {np.ptp(g):.3f}")
    if mr.get("skipped"):
        rep.warn(cal, f"calibration legs skipped: {mr['skipped']}")
    busy = np.zeros(T, dtype=bool)
    for leg in mr["legs"]:
        if "skipped" in leg:
            continue
        name = f"{leg['arm']}:{leg['dof']}{leg['sign']}"
        if "out_start_frame" not in leg:
            rep.err(cal, f"leg without frame span: {name}")
            continue
        a, b = leg["out_start_frame"], min(leg["out_end_frame"], T - 1)
        c, e = leg["back_start_frame"], min(leg["back_end_frame"], T - 1)
        busy[max(0, a - 1):min(T, e + SETTLE_FRAMES + 1)] = True
        arm, other = leg["arm"], ("right" if leg["arm"] == "left" else "left")
        ai, oi = (0, 1) if arm == "left" else (1, 0)
        amt = float(leg["realised"])
        dpos = P[b, ai] - P[a, ai]
        drot = float(ang_deg(Q[arm][a], Q[arm][b]))
        if leg["dof"] in ("x", "y", "z"):
            k = _leg_axis(leg["dof"])
            along = dpos[k]
            if np.sign(along) != np.sign(amt) or abs(along) < LEG_MIN_FRAC * abs(amt):
                rep.err(cal, f"calibration leg short/wrong way: {name} moved {along:+.3f} m of {amt:+.3f}")
            off = np.linalg.norm(np.delete(dpos, k))
            if off > LEG_OFFAXIS_M:
                rep.err(cal, f"calibration leg off-axis: {name} {off:.3f} m")
            if drot > LEG_SPURIOUS_ROT_DEG:
                rep.err(cal, f"calibration translation leg rotates: {name} {drot:.1f} deg")
        else:
            want = np.degrees(abs(amt))
            if drot < LEG_MIN_FRAC * want:
                rep.err(cal, f"calibration rotation leg short: {name} {drot:.1f} of {want:.1f} deg")
            if np.linalg.norm(dpos) > ROT_LEG_TRANS_M:
                rep.err(cal, f"calibration rotation leg translates TCP: {name} {np.linalg.norm(dpos):.3f} m")
        od = np.linalg.norm(P[a:e + 1, oi] - P[a, oi], axis=1).max()
        orr = float(ang_deg(Q[other][a:e + 1], Q[other][a]).max())
        if od > OTHER_ARM_M or orr > OTHER_ARM_DEG:
            rep.err(cal, f"other arm moves during leg: {name} {od * 1000:.1f} mm {orr:.1f} deg")
        ret = np.linalg.norm(P[min(e + SETTLE_FRAMES, T - 1), ai] - P[a, ai])
        if ret > RETURN_M:
            rep.err(cal, f"calibration leg does not return: {name} {ret:.3f} m")
    # nothing may move outside the legs (catches stray / rejected-attempt motion)
    step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(axis=1)
    rstep = np.maximum(ang_deg(Q["left"][1:], Q["left"][:-1]), ang_deg(Q["right"][1:], Q["right"][:-1]))
    idle = ~busy[1:]
    bad = idle & ((step > STILL_STEP_M) | (rstep > STILL_STEP_DEG))
    if bad.any():
        f = int(np.nonzero(bad)[0][0]) + 1
        rep.err(cal, f"motion outside calibration legs: {int(bad.sum())} frame(s), first at {f} "
                     f"({step[f - 1] * 1000:.1f} mm, {rstep[f - 1]:.2f} deg)")
    return mr


def check_counterfactual(cf, pose, exp_pose, rep):
    md = json.load(open(os.path.join(cf, "metadata.json")))
    mode = md["counterfactual_mode"]
    E, C = tcp(exp_pose), tcp(pose)
    n = min(len(E), len(C))
    d = np.linalg.norm(E[:n] - C[:n], axis=2)
    if abs(d.max() - md["divergence"]["max_tcp_m"]) > DIV_TOL_M:
        rep.err(cf, f"divergence mismatch (render != probe): {d.max():.4f} vs {md['divergence']['max_tcp_m']:.4f}")
    if not md["divergence"].get("accepted", False):
        rep.warn(cf, f"counterfactual below divergence gate: {mode}")
    bp = md.get("breakpoint_frame")
    if mode in ("scripted_diverge", "freeze_arm") and bp is not None:
        pre = d[:max(0, bp)]
        if len(pre) and pre.max() > PREFIX_TOL_M:
            rep.err(cf, f"prefix differs from expert: {mode} {pre.max():.4f} m before breakpoint {bp}")
    if mode == "freeze_arm":
        arm = md["arms"]
        # the COMMAND must be frozen exactly from the breakpoint on ...
        st = np.load(os.path.join(cf, "stream.npz"))
        bstep = md["breakpoint_step"]
        q = st[f"{arm}_q"][bstep:]
        if len(q) and np.abs(q - q[0]).max() > 1e-9:
            rep.err(cf, f"frozen arm command changes: {arm} {np.abs(q - q[0]).max():.2e} rad")
        # ... while the arm itself may be shoved by contact (the other, still-working arm
        # running into it is a real consequence of freezing one arm): warn, do not fail
        ai = 0 if arm == "left" else 1
        s = min(n - 1, bp + SETTLE_FRAMES)
        drift = np.linalg.norm(C[s:, ai] - C[s, ai], axis=1).max() if s < len(C) else 0.0
        if drift > FREEZE_DRIFT_M:
            gap = np.linalg.norm(C[s:, 0] - C[s:, 1], axis=1).min()
            rep.warn(cf, f"frozen arm displaced (contact): {arm} {drift:.3f} m, closest arm gap {gap:.3f} m")
        if not md.get("freeze_arm_moves_after_breakpoint", True):
            rep.err(cf, "freeze_arm froze an arm that does not move")
    if mode == "gripper_flip":
        fr = md["flip"]["flip_frame"]
        arm = md["flip"]["arm"]
        pre = d[:max(0, fr)]
        if len(pre) and pre.max() > PREFIX_TOL_M:
            rep.err(cf, f"prefix differs before gripper flip: {pre.max():.4f} m")
        ge, gc = np.asarray(exp_pose[f"{arm}_gripper_open"]), np.asarray(pose[f"{arm}_gripper_open"])
        m = min(len(ge), len(gc))
        if fr < m and np.all(np.abs(np.round(ge[fr:m]) - np.round(gc[fr:m])) < 0.5):
            rep.err(cf, f"gripper flip has no effect on {arm} gripper command")
    if mode == "noisy_replay":
        steps = [s for a in md["noise"]["points"].values() for s in a["steps"]]
        ramp = md["noise"]["interval_frames"]
        if steps:
            first = int(min(steps) // exp_pose["save_stride"]) - ramp - 1
            pre = d[:max(0, first)]
            if len(pre) and pre.max() > PREFIX_TOL_M:
                rep.err(cf, f"noisy replay differs before its first perturbation: {pre.max():.4f} m")
    return md


# ---- main ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_dir", required=True)
    ap.add_argument("--montage_dir", default="")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--no_videos", action="store_true", help="skip decoding videos (fast)")
    ap.add_argument("--show", type=int, default=40, help="max error lines to print")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    ap.add_argument("--json_out", default="", help="also write the findings here as JSON")
    a = ap.parse_args()

    units = sorted(glob.glob(os.path.join(a.set_dir, "*", "episode_*", "camera_poses_*")))
    units = [u for u in units if os.path.exists(os.path.join(u, "_DONE"))]
    if a.tasks:
        keep = set(a.tasks.split(","))
        units = [u for u in units if u.split(os.sep)[-3] in keep]
    if a.max_units:
        units = units[:a.max_units]
    rep = Report()
    by_mode = defaultdict(list)
    per_task = defaultdict(lambda: {"units": 0, "expert_T": [], "skipped": 0, "legs": 0})
    for u in units:
        task = u.split(os.sep)[-3]
        st = per_task[task]
        st["units"] += 1
        try:
            check_unit(u, st, by_mode, rep, a)
        except Exception as e:      # a truncated / corrupt file must be a finding, not a crash
            rep.err(u, f"unreadable unit: {type(e).__name__}: {e}"[:300])

    _print_summary(units, per_task, by_mode, rep, a)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"units": len(units), "errors": rep.errors, "warnings": rep.warns,
                       "counts": dict(rep.counts)}, f, indent=1)
    sys.exit(1 if rep.errors else 0)


def check_unit(u, st, by_mode, rep, a):
    check_rig(u, rep)
    exp = check_rollout(os.path.join(u, "expert"), rep, not a.no_videos)
    if exp is None:
        return
    st["expert_T"].append(len(exp["left_gripper_pose"]))
    em = json.load(open(os.path.join(u, "expert", "metadata.json")))
    if not em.get("success"):
        rep.err(u, "expert not successful")
    cal = os.path.join(u, "calibration")
    if os.path.isdir(cal):
        cp = check_rollout(cal, rep, not a.no_videos)
        if cp is not None:
            mr = check_calibration(cal, cp, rep)
            st["legs"] += len(mr["legs"])
            st["skipped"] += len(mr.get("skipped", []))
    for cf in sorted(glob.glob(os.path.join(u, "counterfactual_*"))):
        p = check_rollout(cf, rep, not a.no_videos)
        if p is None:
            continue
        md = check_counterfactual(cf, p, exp, rep)
        div = md["divergence"]
        by_mode[md["counterfactual_mode"]].append(
            (div.get("max_tcp_m", 0.0), bool(div.get("accepted")), bool(md.get("task_success"))))
    if a.montage_dir:
        montage(u, a.montage_dir)


def _print_summary(units, per_task, by_mode, rep, a):
    print(f"units checked: {len(units)}")
    print(f"{'task':28s} {'units':>5s} {'expert T':>8s} {'calib skipped':>13s}")
    for t, st in sorted(per_task.items()):
        med = int(np.median(st["expert_T"])) if st["expert_T"] else 0
        print(f"{t:28s} {st['units']:5d} {med:8d} {st['skipped']:6d}/{st['legs']:<6d}")
    print(f"\n{'cf mode':18s} {'n':>5s} {'med max_tcp':>11s} {'below gate':>10s} {'task succ':>9s}")
    for mode, rows in sorted(by_mode.items()):
        dd = np.array([r[0] for r in rows])
        print(f"{mode:18s} {len(rows):5d} {np.median(dd):11.3f} {sum(not r[1] for r in rows):10d} "
              f"{np.mean([r[2] for r in rows]):9.2f}")
    print("\nfinding counts:")
    for k, v in rep.counts.most_common():
        print(f"  {v:6d}  {k}")
    for e in rep.errors[:a.show]:
        print("ERROR " + e)
    print(f"\n{len(rep.errors)} error(s), {len(rep.warns)} warning(s)")


def montage(unit, out_dir):
    import imageio.v2 as iio
    rows = []
    for v in sorted(glob.glob(os.path.join(unit, "expert", "*_rgb*", "video.mp4"))):
        r = iio.get_reader(v)
        n = r.count_frames()
        rows.append(np.concatenate([r.get_data(i) for i in (0, n // 2, n - 1)], axis=1))
        r.close()
    if not rows:
        return
    w = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in rows]
    os.makedirs(out_dir, exist_ok=True)
    name = "__".join(unit.split(os.sep)[-4:-1]) + ".jpg"
    iio.imwrite(os.path.join(out_dir, name), np.concatenate(rows, axis=0)[::2, ::2])


if __name__ == "__main__":
    main()
