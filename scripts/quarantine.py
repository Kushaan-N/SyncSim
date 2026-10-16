#!/usr/bin/env python
"""Move the rollouts / units listed in a verify_set.py exclude file out of the dataset tree, so
loaders that walk the data (SyncWorld's gripperhead loader has no exclude option) never see
them -- reversibly, nothing is deleted.

    python scripts/quarantine.py --list bad.tsv [--dry_run]      # move listed dirs out
    python scripts/quarantine.py --restore --root data            # move everything back

A listed dir <root>/set_k/<task>/episode_s/camera_poses_m[/<rollout>] goes to
<root>/_quarantine/set_k/... with the reason appended to <root>/_quarantine/manifest.tsv.
Use it for what `redo_cf.sh` cannot repair (unit-level problems such as a failed wrist mount)
or for counterfactuals that still fail after a redo. Quarantining a unit's expert or
calibration would leave an unusable unit, so those lines quarantine the whole unit.
"""

import argparse
import os
import shutil
import time


def data_root_of(path):
    """(data root, path relative to it): the root is the parent of the set_* ancestor."""
    parts = os.path.abspath(path).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("set_"):
            return os.sep.join(parts[:i]) or os.sep, os.sep.join(parts[i:])
    raise ValueError(f"{path}: not inside a set_* directory")


def target_of(path):
    """The dir to move: a counterfactual on its own, anything else as its whole unit."""
    p = os.path.abspath(path)
    if os.path.basename(p).startswith("camera_poses_"):
        return p
    if os.path.basename(os.path.dirname(p)).startswith("camera_poses_"):
        return p if os.path.basename(p).startswith("counterfactual_") else os.path.dirname(p)
    raise ValueError(f"{path}: not a camera unit or a rollout inside one")


def quarantine(list_path, dry_run):
    seen = set()
    for line in open(list_path):
        if not line.strip() or line.startswith("#"):
            continue
        path, _, reason = line.rstrip("\n").partition("\t")
        src = target_of(path.strip())
        if src in seen or any(src.startswith(s + os.sep) for s in seen):
            continue
        seen.add(src)
        if not os.path.exists(src):
            print(f"[skip] {src}: not there (already quarantined?)")
            continue
        root, rel = data_root_of(src)
        dst = os.path.join(root, "_quarantine", rel)
        print(f"[{'dry' if dry_run else 'move'}] {rel}  ({reason.strip()[:80]})")
        if dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        with open(os.path.join(root, "_quarantine", "manifest.tsv"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{rel}\t{reason.strip()}\n")


def restore(root):
    q = os.path.join(os.path.abspath(root), "_quarantine")
    for dirpath, dirnames, _ in os.walk(q):
        base = os.path.basename(dirpath)
        if base.startswith("camera_poses_") or base.startswith("counterfactual_"):
            rel = os.path.relpath(dirpath, q)
            dst = os.path.join(os.path.abspath(root), rel)
            if os.path.exists(dst) and base.startswith("camera_poses_"):
                continue        # the unit is in the dataset: only some of its rollouts were moved
            if os.path.exists(dst):
                print(f"[skip] {rel}: exists in the dataset")
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(dirpath, dst)
                print(f"[back] {rel}")
            dirnames[:] = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="", help="verify_set.py --exclude_out file")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--root", default="", help="data root (with --restore)")
    a = ap.parse_args()
    if a.restore:
        restore(a.root)
    else:
        quarantine(a.list, a.dry_run)


if __name__ == "__main__":
    main()
