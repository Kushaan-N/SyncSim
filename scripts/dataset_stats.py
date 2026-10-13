#!/usr/bin/env python
"""Dataset-level statistics for a SyncSim data root (or one set), from the JSON files only
(no videos, no simulator) -- the skew a per-episode validator cannot see.

    python scripts/dataset_stats.py --root data [--json_out stats.json]

Reports, per set and overall:
  yield        seeds tried / episodes kept per task, drop reasons grouped by kind
  experts      length (frames) per task
  cameras      azimuth / elevation / radius / baseline spread of the third-person views,
               gate pass rate; wrist mounts that fell back to the stock mount, ray-trace scores
  calibration  legs per arm x DoF actually filmed, skipped legs, sign flips
  cf           per mode: count, divergence, below-gate, fallbacks, task-success rate --
               and tasks whose counterfactuals almost always still succeed (weak labels)
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _q(xs, qs=(0, 25, 50, 75, 100)):
    if not len(xs):
        return "-"
    v = np.percentile(np.asarray(xs, float), qs)
    return " / ".join(f"{x:.2f}" for x in v)


def drop_kind(reason):
    r = reason.split(":")[0]
    if r.startswith("joint_limit_excess"):
        return "joint_limit_excess"
    if r == "exception":
        m = re.match(r"exception: (\w+)", reason)
        return f"exception:{m.group(1) if m else '?'}"
    return r


def collect(set_dir):
    S = {"tasks": defaultdict(lambda: {"kept": 0, "dropped": Counter(), "expert_T": [],
                                        "cf_success": [], "units": 0}),
         "views": defaultdict(list), "gate_pass": [], "wrist_stock": Counter(),
         "wrist_rt_corr": [], "calib_legs": Counter(), "calib_skipped": Counter(),
         "calib_flipped": 0, "calib_total": 0,
         "cf": defaultdict(lambda: {"n": 0, "max_tcp": [], "below": 0, "fallback": 0,
                                    "success": 0, "arms": Counter()})}
    for task_dir in sorted(glob.glob(os.path.join(set_dir, "*"))):
        task = os.path.basename(task_dir)
        if task.startswith("_") or not os.path.isdir(task_dir):
            continue
        T = S["tasks"][task]
        for reason in (_load(os.path.join(task_dir, "_dropped.json")) or {}).values():
            T["dropped"][drop_kind(reason)] += 1
        for ep in sorted(glob.glob(os.path.join(task_dir, "episode_*"))):
            units = [u for u in sorted(glob.glob(os.path.join(ep, "camera_poses_*")))
                     if os.path.exists(os.path.join(u, "_DONE"))]
            if not units:
                continue
            T["kept"] += 1
            for u in units:
                T["units"] += 1
                _unit(u, T, S)
    return S


def _unit(u, T, S):
    em = _load(os.path.join(u, "expert", "metadata.json")) or {}
    if "frames" in em:
        T["expert_T"].append(em["frames"])
    v = _load(os.path.join(u, "views.json")) or {}
    if v.get("candidates"):
        S["gate_pass"].append(v.get("candidates_passed", 0) / v["candidates"])
    tgt = np.asarray(v.get("aim_target", [0, 0, 0]), float)
    pos = []
    for view in v.get("views", []):
        for k in ("azimuth_deg", "elevation_deg", "radius_m", "roll_deg"):
            if k in view:
                S["views"][k].append(view[k])
        pos.append(np.asarray(view["position"], float))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            a, b = pos[i] - tgt, pos[j] - tgt
            S["views"]["baseline_deg"].append(float(np.degrees(np.arccos(np.clip(
                a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))))
    for arm, w in v.get("wrist", {}).items():
        S["wrist_stock"][(arm, bool(w.get("stock_mount")))] += 1
        for sd_c in w.get("gate_rt_check") or []:
            S["wrist_rt_corr"].append(sd_c[1])
    mr = _load(os.path.join(u, "calibration", "move_range.json"))
    if mr:
        for leg in mr.get("legs", []):
            S["calib_total"] += 1
            S["calib_legs"][(leg["arm"], leg["dof"])] += 1
            S["calib_flipped"] += int(bool(leg.get("sign_flipped")))
        for sk in mr.get("skipped", []):
            key = (sk.get("arm"), sk.get("dof")) if isinstance(sk, dict) else str(sk)
            S["calib_skipped"][key] += 1
    for cf in sorted(glob.glob(os.path.join(u, "counterfactual_*"))):
        md = _load(os.path.join(cf, "metadata.json"))
        if not md:
            continue
        mode = md["counterfactual_mode"]
        C = S["cf"][mode]
        div = md.get("divergence", {})
        C["n"] += 1
        C["max_tcp"].append(div.get("max_tcp_m", 0.0))
        C["below"] += int(not div.get("accepted", True))
        log = div.get("attempt_log") or []
        C["fallback"] += int(bool(log) and log[0].get("mode") != mode)
        C["success"] += int(bool(md.get("task_success")))
        C["arms"][md.get("arms")] += 1
        T["cf_success"].append(bool(md.get("task_success")))


def report(name, S, weak_cf_success=0.6):
    out = [f"\n=================== {name} ==================="]
    tasks = S["tasks"]
    kept = sum(t["kept"] for t in tasks.values())
    drops = Counter()
    for t in tasks.values():
        drops.update(t["dropped"])
    tried = kept + sum(drops.values())
    out.append(f"episodes kept {kept} / seeds finished {tried} "
               f"({100 * kept / max(1, tried):.0f}%), tasks {len(tasks)}")
    out.append("drop reasons: " + ", ".join(f"{k} {v}" for k, v in drops.most_common()))
    out.append(f"\n{'task':28s} {'kept':>4s} {'drop':>4s} {'yield':>5s} {'exp T med':>9s} "
               f"{'cf succ':>7s}  top drop")
    for task, t in sorted(tasks.items(), key=lambda kv: kv[1]["kept"]):
        nd = sum(t["dropped"].values())
        top = t["dropped"].most_common(1)
        cs = np.mean(t["cf_success"]) if t["cf_success"] else float("nan")
        out.append(f"{task:28s} {t['kept']:4d} {nd:4d} {100 * t['kept'] / max(1, t['kept'] + nd):4.0f}% "
                   f"{int(np.median(t['expert_T'])) if t['expert_T'] else 0:9d} {cs:7.2f}  "
                   f"{top[0][0] + ' x' + str(top[0][1]) if top else ''}")
    out.append("\ncameras (min / q25 / median / q75 / max):")
    for k in ("azimuth_deg", "elevation_deg", "radius_m", "roll_deg", "baseline_deg"):
        out.append(f"  {k:14s} {_q(S['views'][k])}")
    out.append(f"  gate pass frac {_q(S['gate_pass'])}")
    ws = S["wrist_stock"]
    for arm in ("left", "right"):
        n = ws[(arm, True)] + ws[(arm, False)]
        out.append(f"  wrist {arm:5s} stock-mount fallback {ws[(arm, True)]}/{n}")
    out.append(f"  wrist rt-vs-raster corr {_q(S['wrist_rt_corr'])}")
    out.append(f"\ncalibration: {S['calib_total']} legs filmed, {S['calib_flipped']} sign-flipped, "
               f"skipped: {dict(S['calib_skipped']) or 'none'}")
    per = defaultdict(dict)
    for (arm, dof), n in S["calib_legs"].items():
        per[arm][dof] = n
    for arm in sorted(per):
        out.append(f"  {arm:5s} " + "  ".join(f"{d}:{per[arm].get(d, 0)}"
                                             for d in ("x", "y", "z", "roll", "pitch", "yaw")))
    out.append(f"\n{'cf mode':18s} {'n':>5s} {'max_tcp m (q25/med/q75)':>24s} {'below':>5s} "
               f"{'fallbk':>6s} {'succ':>5s}  arms")
    for mode, C in sorted(S["cf"].items()):
        q = np.percentile(C["max_tcp"], [25, 50, 75]) if C["max_tcp"] else [0, 0, 0]
        out.append(f"{mode:18s} {C['n']:5d} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f}    {C['below']:5d} "
                   f"{C['fallback']:6d} {C['success'] / max(1, C['n']):5.2f}  {dict(C['arms'])}")
    weak = [(task, np.mean(t["cf_success"])) for task, t in tasks.items()
            if len(t["cf_success"]) >= 8 and np.mean(t["cf_success"]) >= weak_cf_success]
    if weak:
        out.append(f"\ntasks whose counterfactuals still succeed >= {weak_cf_success:.0%} "
                   "(little failure signal): " + ", ".join(f"{t} {s:.2f}" for t, s in sorted(weak)))
    return "\n".join(out)


def _jsonable(o):
    if isinstance(o, dict):         # Counter / defaultdict with (arm, dof) tuple keys
        return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root (holding set_*) or one set dir")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()
    sets = sorted(glob.glob(os.path.join(a.root, "set_*"))) or [a.root]
    allS = {}
    for s in sets:
        S = collect(s)
        allS[os.path.basename(s)] = S
        print(report(os.path.basename(s), S))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({k: _jsonable(v) for k, v in allS.items()}, f, indent=1)


if __name__ == "__main__":
    main()
