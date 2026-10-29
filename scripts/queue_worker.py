#!/usr/bin/env python
"""Shared work queue for large SyncSim runs (the RoboTwin 2.0 release: ~110k experts).

A queue is a directory on a shared filesystem:

    <queue>/items.jsonl     one work item per line: {"id", "task", "release_dir", "output_dir",
                            "start", "count"}  (built by `build`)
    <queue>/claims/<id>/    a worker owns the item; owner.json + heartbeat mtime
    <queue>/done/<id>       item finished (the generator exited 0)
    <queue>/fail/<id>       failure count + last log tail

Workers on any number of nodes / SLURM jobs pull items until none are left. Claims are atomic
`mkdir`s; a claim whose heartbeat is older than --stale_s (job killed, node preempted) is
taken over, and the generator resumes from the unit-level _DONE markers, so preemption costs
at most the episode in flight.

    python scripts/queue_worker.py build --queue Q --release_root R --output_root O [--chunk 10]
    python scripts/queue_worker.py work  --queue Q --gpu 0 [--worker_id host-0]
    python scripts/queue_worker.py status --queue Q
    python scripts/queue_worker.py requeue --queue Q --reason release_replay_not_successful

`work` runs `python -m syncsim.generate` per item with the run defaults of scripts/gen_set.sh
(EMIT_ARGS=1), then the item's task / release dir / episode range / output dir.
"""

import argparse
import glob
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYTHON = os.environ.get("PYTHON_BIN", "/work/umass/chuangg_umass/yuncongyang_umass/syncsim/envs/rt/bin/python")


def _safe(item_id):
    return item_id.replace("/", "__")


# ------------------------------------------------------------------------------------------
def build(a):
    """One item per `chunk` release episodes of every complete <task>/<emb>_<variant> dir."""
    items = []
    for d in sorted(glob.glob(os.path.join(a.release_root, "*", "*_*"))):
        if not os.path.exists(os.path.join(d, "_COMPLETE")):
            continue
        task, ev = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        if a.tasks and task not in a.tasks.split(","):
            continue
        if a.variants and ev.split("_", 1)[1] not in a.variants.split(","):
            continue
        with open(os.path.join(d, "seed.txt")) as f:
            n = len(f.read().split())
        n = min(n, a.max_episodes) if a.max_episodes else n
        for s in range(0, n, a.chunk):
            items.append({"id": f"{ev}/{task}/{s:04d}", "task": task, "release_dir": d,
                          "output_dir": os.path.join(a.output_root, ev),
                          "start": s, "count": min(a.chunk, n - s)})
    # interleave tasks / embodiments so early results cover everything and no single task's
    # GPU-heavy scenes pile up on one node
    random.Random(0).shuffle(items)
    os.makedirs(a.queue, exist_ok=True)
    path = os.path.join(a.queue, "items.jsonl")
    old = set()
    if os.path.exists(path):
        with open(path) as f:
            old = {json.loads(l)["id"] for l in f if l.strip()}
    new = [it for it in items if it["id"] not in old]
    with open(path, "a") as f:
        for it in new:
            f.write(json.dumps(it) + "\n")
    for sub in ("claims", "done", "fail"):
        os.makedirs(os.path.join(a.queue, sub), exist_ok=True)
    print(f"[queue] {len(new)} new item(s) ({len(old) + len(new)} total, "
          f"{sum(i['count'] for i in new)} episodes added)")


def _load_items(q):
    with open(os.path.join(q, "items.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------
class Claim:
    def __init__(self, q, item_id, owner, stale_s):
        self.dir = os.path.join(q, "claims", _safe(item_id))
        self.owner, self.stale_s, self._stop = owner, stale_s, threading.Event()

    def acquire(self):
        try:
            os.mkdir(self.dir)
        except FileExistsError:
            hb = os.path.join(self.dir, "heartbeat")
            try:
                age = time.time() - os.path.getmtime(hb if os.path.exists(hb) else self.dir)
            except FileNotFoundError:
                return False
            if age < self.stale_s:
                return False
            # stale: take it over atomically (only one taker wins the rename)
            grave = f"{self.dir}.stale.{os.getpid()}.{time.time():.0f}"
            try:
                os.rename(self.dir, grave)
            except OSError:
                return False
            shutil.rmtree(grave, ignore_errors=True)
            try:
                os.mkdir(self.dir)
            except FileExistsError:
                return False
        with open(os.path.join(self.dir, "owner.json"), "w") as f:
            json.dump({"owner": self.owner, "since": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "slurm_job": os.environ.get("SLURM_JOB_ID")}, f)
        self._beat()
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _beat(self):
        with open(os.path.join(self.dir, "heartbeat"), "w") as f:
            f.write(str(time.time()))

    def _loop(self):
        while not self._stop.wait(60):
            try:
                self._beat()
            except OSError:
                pass

    def release(self):
        self._stop.set()
        shutil.rmtree(self.dir, ignore_errors=True)


def _base_args():
    env = dict(os.environ, SET_ID="0", EMIT_ARGS="1")
    out = subprocess.run(["bash", os.path.join(HERE, "gen_set.sh")], env=env, check=True,
                         capture_output=True).stdout
    return [x.decode() for x in out.split(b"\0") if x]


def work(a):
    q = a.queue
    owner = a.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    rng = random.Random(owner)
    n_done = 0
    while True:
        items = _load_items(q)
        done = set(os.listdir(os.path.join(q, "done")))
        fails = {}
        for p in glob.glob(os.path.join(q, "fail", "*")):
            try:
                fails[os.path.basename(p)] = json.load(open(p))["count"]
            except Exception:
                pass
        todo = [it for it in items if _safe(it["id"]) not in done
                and fails.get(_safe(it["id"]), 0) < a.max_failures]
        if not todo:
            print(f"[worker {owner}] queue empty after {n_done} item(s)", flush=True)
            return
        # items run roughly in file order (earlier `build`s first: e.g. clean before
        # randomized), with a random pick among the next few so workers do not all race
        # for the same item
        head = min(len(todo), 64)
        order = rng.sample(range(head), head) + list(range(head, len(todo)))
        it = None
        for j in order:
            cand = todo[j]
            c = Claim(q, cand["id"], owner, a.stale_s)
            if c.acquire():
                it = cand
                break
        if it is None:
            if a.wait_when_busy:
                time.sleep(120)
                continue
            print(f"[worker {owner}] nothing claimable", flush=True)
            return
        sid = _safe(it["id"])
        log_dir = os.path.join(it["output_dir"], "_logs")
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, f"{it['task']}.{it['start']:04d}.log")
        # re-read the run defaults per item, so a default changed mid-run (gen_set.sh) applies
        # to every item started afterwards
        cmd = [PYTHON, "-m", "syncsim.generate", *_base_args(), "--task", it["task"],
               "--release_dir", it["release_dir"], "--output_dir", it["output_dir"],
               "--seed_start", str(it["start"]), "--num_seeds", str(it["count"]),
               "--max_episodes", "0"] + a.extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED="1",
                   PYTHONWARNINGS="ignore", OMP_NUM_THREADS=str(a.threads),
                   MKL_NUM_THREADS=str(a.threads))
        t0 = time.time()
        print(f"[worker {owner}] {it['id']} -> {log}", flush=True)
        with open(log, "a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {owner} gpu {a.gpu}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc == 0:
            with open(os.path.join(q, "done", sid), "w") as f:
                json.dump({"owner": owner, "seconds": round(dt), "episodes": it["count"],
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
            n_done += 1
        else:
            fp = os.path.join(q, "fail", sid)
            cnt = fails.get(sid, 0) + 1
            with open(log) as f:
                tail = f.read()[-2000:]
            with open(fp, "w") as f:
                json.dump({"count": cnt, "rc": rc, "owner": owner, "tail": tail}, f)
        c.release()
        print(f"[worker {owner}] {it['id']} rc={rc} in {dt:.0f}s", flush=True)


def requeue(a):
    """Retry dropped episodes whose drop reason starts with --reason: forget the drops and
    un-finish their items (the generator skips the episodes that are already complete)."""
    n_ep = n_items = 0
    for it in _load_items(a.queue):
        sid = _safe(it["id"])
        if not os.path.exists(os.path.join(a.queue, "done", sid)):
            continue
        if os.path.isdir(os.path.join(a.queue, "claims", sid)):
            continue
        dpath = os.path.join(it["output_dir"], it["task"], "_dropped.json")
        if not os.path.exists(dpath):
            continue
        with open(os.path.join(it["release_dir"], "seed.txt")) as f:
            seeds = {str(x) for x in f.read().split()[it["start"]:it["start"] + it["count"]]}
        with open(dpath) as f:
            drops = json.load(f)
        hit = [k for k, v in drops.items() if k in seeds and v.startswith(a.reason)]
        if not hit:
            continue
        n_ep += len(hit)
        n_items += 1
        if a.dry_run:
            continue
        for k in hit:
            drops.pop(k)
        with open(dpath + ".tmp", "w") as f:
            json.dump(drops, f, indent=1)
        os.replace(dpath + ".tmp", dpath)
        os.remove(os.path.join(a.queue, "done", sid))
        fp = os.path.join(a.queue, "fail", sid)
        if os.path.exists(fp):
            os.remove(fp)
    print(f"[requeue] {n_ep} episode(s) in {n_items} item(s){' (dry run)' if a.dry_run else ''}")


def status(a):
    items = _load_items(a.queue)
    done = {}
    for p in glob.glob(os.path.join(a.queue, "done", "*")):
        try:
            done[os.path.basename(p)] = json.load(open(p))
        except Exception:
            pass
    claims = os.listdir(os.path.join(a.queue, "claims"))
    fails = glob.glob(os.path.join(a.queue, "fail", "*"))
    n_ep = sum(i["count"] for i in items)
    n_ep_done = sum(d["episodes"] for d in done.values())
    secs = [d["seconds"] / max(1, d["episodes"]) for d in done.values()]
    print(f"items {len(done)}/{len(items)} done, {len(claims)} running, {len(fails)} with failures")
    print(f"episodes (tried) {n_ep_done}/{n_ep}")
    if secs:
        secs.sort()
        print(f"worker-seconds per episode: median {secs[len(secs) // 2]:.0f}, "
              f"p90 {secs[int(len(secs) * 0.9)]:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--queue", required=True)
    b.add_argument("--release_root", required=True)
    b.add_argument("--output_root", required=True)
    b.add_argument("--chunk", type=int, default=10)
    b.add_argument("--tasks", default="")
    b.add_argument("--variants", default="")
    b.add_argument("--max_episodes", type=int, default=0, help="per release dir (0 = all)")
    w = sub.add_parser("work")
    w.add_argument("--queue", required=True)
    w.add_argument("--gpu", default="0")
    w.add_argument("--worker_id", default="")
    w.add_argument("--threads", type=int, default=4)
    w.add_argument("--stale_s", type=float, default=900)
    w.add_argument("--max_failures", type=int, default=2)
    w.add_argument("--wait_when_busy", action="store_true")
    w.add_argument("extra", nargs=argparse.REMAINDER, help="extra generator flags after --")
    s = sub.add_parser("status")
    s.add_argument("--queue", required=True)
    r = sub.add_parser("requeue")
    r.add_argument("--queue", required=True)
    r.add_argument("--reason", required=True, help="drop-reason prefix, e.g. release_replay_not_successful")
    r.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    if getattr(a, "extra", None) and a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    {"build": build, "work": work, "status": status, "requeue": requeue}[a.cmd](a)


if __name__ == "__main__":
    main()
