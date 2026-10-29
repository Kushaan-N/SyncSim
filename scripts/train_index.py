#!/usr/bin/env python
"""Episode list of FINISHED rollouts, for training on a dataset that is still being generated.

A world-model trainer that walks the data roots itself would also pick up rollouts whose
camera unit is still being written. This lists only the expert and counterfactual leaves of
units with a `_DONE` marker (written last by the generator), as a JSON list of
{"path", "episode"} -- the episode-discovery cache format the trainer reads -- and writes it
atomically, so it can be refreshed while a run is going and picked up at the next restart.

    python scripts/train_index.py --roots data_rt2/aloha-agilex_clean_50,... --out train_index.json
    python scripts/train_index.py --roots_glob 'data_rt2/*' --out train_index.json
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


def unit_leaves(unit):
    if not os.path.exists(os.path.join(unit, "_DONE")):
        return []
    out = []
    for name in sorted(os.listdir(unit)):
        if name == "expert" or name.startswith("counterfactual_"):
            leaf = os.path.join(unit, name)
            if os.path.isfile(os.path.join(leaf, "pose.pkl")):
                out.append({"path": leaf, "episode": name})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", default="", help="comma list of data roots (each <root>/<task>/episode_*/camera_poses_*)")
    p.add_argument("--roots_glob", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    roots = [r for r in a.roots.split(",") if r] + sorted(glob.glob(a.roots_glob) if a.roots_glob else [])
    roots = [r for r in roots if os.path.isdir(r)]
    units = []
    for r in roots:
        units += glob.glob(os.path.join(r, "*", "episode_*", "camera_poses_*"))
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        leaves = [l for ls in ex.map(unit_leaves, units) for l in ls]
    leaves.sort(key=lambda e: e["path"])
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(leaves, f)
    os.replace(tmp, a.out)
    n_units = len({os.path.dirname(l["path"]) for l in leaves})
    n_exp = sum(l["episode"] == "expert" for l in leaves)
    print(f"[train_index] {len(roots)} root(s), {n_units}/{len(units)} finished units, "
          f"{len(leaves)} leaves ({n_exp} expert) in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
