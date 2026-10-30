#!/usr/bin/env python
"""Interaction index + distribution numbers for a data root that has contacts.json files
(written live by the generator, or added by scripts/annotate_contacts.py).

Writes two files next to the data (or to --out_dir):

  contact_index.jsonl   one row per rollout: where it is, what kind (expert / calibration /
                        counterfactual mode + arms), task success, and the contact summary
                        (interaction share, held share, event counts, objects touched). This is
                        the table to SAMPLE from: filter rows, or weight them by a column.
  contact_events.jsonl  one row per contact window (frame x to frame y): rollout, part pair,
                        kind, force, how far the object moved during and after. Use it to find
                        e.g. every arm-object bump, or every grasp that ended in a drop.

and prints the distribution numbers to track across generator iterations: per rollout kind /
counterfactual mode (and optionally per task), the share of frames with robot-object contact,
with an object held, mean interaction seconds, event rates, and NOVEL contact: part pairs that
touch in a rollout but never in its expert (an arm on the table, a block knocked into the
clutter), as a share of frames and of rollouts. A generator change that is meant to add interactions should
move these up for the modes it touches; --compare <old contact_index.jsonl> prints the delta.

    python scripts/contact_stats.py --data_root <data> [--by_task] [--compare old.jsonl]
    python scripts/contact_stats.py --data_root <release run> --set_glob '*' --done_only
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = (
    ("robot_object_contact_frac", "robot-obj frames", "{:.1%}"),
    ("held_frac", "held frames", "{:.1%}"),
    ("interaction_s", "interact s", "{:.2f}"),
    ("n_gripper_object_events", "grip-obj ev", "{:.2f}"),
    ("n_arm_object_events", "arm-obj ev", "{:.2f}"),
    ("n_robot_env_events", "robot-env ev", "{:.2f}"),
    ("n_caused_object_object_events", "obj-obj ev", "{:.2f}"),
    ("n_objects_touched", "objs touched", "{:.2f}"),
    ("novel_contact_frac", "novel frames", "{:.1%}"),
    ("has_novel_contact", "any novel", "{:.1%}"),
)
# "novel" = a part pair touching in this rollout that never touched in the same unit's expert
# (the arm hitting the table where the expert never did, a block knocked into the clutter, ...).
# Contacts the expert also makes (gripper on the target pad, the table under a grasp) are not
# novel, so the expert itself is 0 by definition.


def rollout_kind(rdir):
    b = os.path.basename(rdir)
    if b == "expert":
        return "expert", None, None
    if b == "calibration":
        return "calibration", None, None
    md = json.load(open(os.path.join(rdir, "metadata.json")))
    return "counterfactual", md.get("counterfactual_mode"), md.get("arms")


def row_of(rdir, root):
    c = json.load(open(os.path.join(rdir, "contacts.json")))
    s = c["summary"]
    kind, mode, arms = rollout_kind(rdir)
    rel = os.path.relpath(rdir, root).split(os.sep)
    md = json.load(open(os.path.join(rdir, "metadata.json"))) if os.path.exists(
        os.path.join(rdir, "metadata.json")) else {}
    succ = md.get("task_success", md.get("success"))
    row = {"path": os.path.relpath(rdir, root), "set": rel[0], "task": rel[1],
           "episode": rel[2], "unit": rel[3], "rollout": rel[4], "kind": kind,
           "mode": mode or kind, "arms": arms, "task_success": succ,
           "replay_drift_m": c.get("replay_check", {}).get("max_drift_m"),
           "source": c.get("source", "live")}
    row.update({k: v for k, v in s.items() if k not in ("objects_touched",
                                                        "objects_moved_gt_2cm")})
    row["objects_touched"] = s["objects_touched"]
    row["objects_moved"] = s["objects_moved_gt_2cm"]
    events = []
    for w in c["windows"]:
        if w["started"] == "initial":
            continue                    # resting contacts of the set-up scene
        events.append(dict(w, path=row["path"], task=row["task"], mode=row["mode"]))
    for arm, ws in c.get("held", {}).items():
        for w in ws:
            events.append(dict(w, a=f"{arm}_finger", b=w["object"], kind="held",
                               path=row["path"], task=row["task"], mode=row["mode"]))
    row["_pairs"] = sorted({(w["a"], w["b"]) for w in c["windows"] if w["started"] == "rollout"})
    return row, events


def add_novelty(rows, root):
    """novel_contact_frac / has_novel_contact per row, against the unit's expert."""
    import numpy as np_
    expert = {os.path.dirname(r["path"]): set(map(tuple, r["_pairs"]))
              for r in rows if r["kind"] == "expert"}
    for r in rows:
        unit = os.path.dirname(r["path"])
        E = expert.get(unit)
        if E is None or r["kind"] == "expert":
            r["novel_pairs"], r["novel_contact_frac"], r["has_novel_contact"] = [], 0.0, False
            continue
        novel = [p for p in map(tuple, r["_pairs"]) if p not in E]
        frac = 0.0
        if novel:
            z = np_.load(os.path.join(root, r["path"], "contacts.npz"))
            keys = list(zip(map(str, z["pair_a"]), map(str, z["pair_b"])))
            cols = [keys.index(p) for p in novel if p in keys]
            frac = float((z["touch"][:, cols] > 0).any(axis=1).mean()) if cols else 0.0
        r["novel_pairs"] = [f"{a}|{b}" for a, b in novel]
        r["novel_contact_frac"], r["has_novel_contact"] = frac, bool(novel)
    for r in rows:
        r.pop("_pairs", None)


def table(rows, group_keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out[g] = {"n": len(rs)}
        for key, _, _ in METRICS:
            out[g][key] = float(np.mean([float(r.get(key, 0.0)) for r in rs]))
        fails = [r for r in rs if r["task_success"] is False]
        out[g]["fail_rate"] = len(fails) / len(rs) if rs else 0.0
    return out


def print_table(t, group_keys, old=None):
    head = " / ".join(group_keys)
    cols = ["n"] + [lab for _, lab, _ in METRICS] + ["fail"]
    print(f"{head:34s}" + "".join(f"{c:>15s}" for c in cols))
    for g, v in t.items():
        name = " / ".join(str(x) for x in g)
        cells = [f"{v['n']:d}"]
        for key, _, fmt in METRICS:
            cell = fmt.format(v[key])
            if old is not None and g in old:
                d = v[key] - old[g][key]
                cell += f" ({'+' if d >= 0 else ''}{fmt.format(d)})"
            cells.append(cell)
        cells.append(f"{v['fail_rate']:.0%}")
        print(f"{name[:34]:34s}" + "".join(f"{c:>15s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--by_task", action="store_true")
    ap.add_argument("--compare", default="", help="an older contact_index.jsonl")
    ap.add_argument("--set_glob", default="set_*",
                    help="set dirs under the root ('*' for a release run: <emb>_<variant>/)")
    ap.add_argument("--done_only", action="store_true",
                    help="only units with _DONE (a run still writing)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = os.path.abspath(a.data_root)
    out_dir = a.out_dir or root
    rows, events = [], []
    paths = sorted(glob.glob(os.path.join(root, a.set_glob, "*", "episode_*", "camera_poses_*",
                                          "*", "contacts.json")))
    if a.done_only:
        paths = [p for p in paths
                 if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), "_DONE"))]

    def load(p):
        try:
            return row_of(os.path.dirname(p), root)
        except Exception as ex:          # a half-written file from a running job
            print(f"skip {p}: {ex}")
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(load, paths):
            if res is not None:
                rows.append(res[0])
                events.extend(res[1])
    if not rows:
        raise SystemExit(f"no contacts.json under {root}")
    add_novelty(rows, root)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "contact_index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "contact_events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    drift = [r["replay_drift_m"] for r in rows if r["replay_drift_m"] is not None]
    print(f"{len(rows)} rollouts, {len(events)} contact windows -> {out_dir}")
    if drift:
        print(f"offline replay drift vs stored object poses: median {np.median(drift):.1e} m, "
              f"max {max(drift):.1e} m, > 5 mm in {sum(d > 0.005 for d in drift)} rollouts")
    old = None
    if a.compare:
        old_rows = [json.loads(l) for l in open(a.compare)]
    keys = ["mode"]
    print()
    old = table(old_rows, keys) if a.compare else None
    print_table(table(rows, keys), keys, old)
    if a.by_task:
        keys = ["task", "mode"]
        print()
        old = table(old_rows, keys) if a.compare else None
        print_table(table(rows, keys), keys, old)


if __name__ == "__main__":
    main()
