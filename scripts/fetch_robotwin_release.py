#!/usr/bin/env python
"""Fetch the expert-replay parts of the RoboTwin 2.0 release (huggingface TianxingChen/RoboTwin2.0).

Each release zip (dataset/<task>/<embodiment>_<clean_50|randomized_500>.zip) holds the seeds,
scene info, planner trajectories (_traj_data/episode<i>.pkl), language instructions and the
rendered HDF5/videos. SyncSim replays the planner trajectories and renders its own cameras, so
only the small members are needed: they are read straight out of the remote zips with HTTP
range requests (~40 MB of a ~5 GB zip), never downloading the HDF5s.

    python scripts/fetch_robotwin_release.py --out /scratch/.../rt2_release [--tasks a,b] [--workers 8]

Writes <out>/<task>/<embodiment>_<variant>/{seed.txt,scene_info.json,_traj_data/,instructions/}
and a _COMPLETE marker per zip; reruns skip completed zips.
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0"
KEEP = ("seed.txt", "scene_info.json")
KEEP_DIRS = ("/_traj_data/", "/instructions/")


def wanted(name):
    return name.endswith(KEEP) or any(d in name for d in KEEP_DIRS)


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file over HTTP range requests, in cached blocks.
    (huggingface_hub's HfFileSystem spends ~80 s per open resolving the repo here.)"""

    def __init__(self, url, block=1 << 17):
        import requests
        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos, self.block, self.cache = r.url, int(r.headers["Content-Length"]), 0, block, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1
            self.cache[i] = self._get(lo, hi)
        return self.cache[i]

    def _get(self, lo, hi):
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                r.raise_for_status()
                return r.content
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if self.block == 0:
            d = self._get(self.pos, self.pos + n - 1) if n else b""
            self.pos += len(d)
            return d
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def read_member(f, info):
    """One member in ONE ranged request (local header + data), instead of block-by-block reads
    through zipfile: the wanted members are small and scattered between multi-MB HDF5s."""
    import struct
    import zlib
    off = info.header_offset
    f.seek(off)
    raw = f.read(30 + len(info.orig_filename.encode()) + 1024 + info.compress_size)
    n_name, n_extra = struct.unpack("<HH", raw[26:30])
    data = raw[30 + n_name + n_extra:30 + n_name + n_extra + info.compress_size]
    if len(data) < info.compress_size:          # unusually long local extra field
        f.seek(off + 30 + n_name + n_extra)
        data = f.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(out) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return out


def fetch_zip(remote, dest):
    if os.path.exists(os.path.join(dest, "_COMPLETE")):
        return remote, 0, "skip"
    with RangeFile(f"{BASE}/resolve/main/{remote}") as f:
        z = zipfile.ZipFile(f)
        infos = [i for i in z.infolist() if wanted(i.filename) and not i.filename.endswith("/")]
        f.block = 0                              # member reads below go straight to the server
        n_bytes = 0
        for info in infos:
            n = info.filename
            # strip the zip's top-level dir (<embodiment>_<variant>/...)
            rel = n.split("/", 1)[1] if "/" in n else n
            path = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = read_member(f, info)
            with open(path + ".tmp", "wb") as g:
                g.write(data)
            os.replace(path + ".tmp", path)
            n_bytes += len(data)
        names = infos
    with open(os.path.join(dest, "_COMPLETE"), "w") as g:
        g.write(f"{len(names)} files {n_bytes} bytes {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return remote, n_bytes, f"{len(names)} files"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="", help="comma list (default: all)")
    p.add_argument("--variants", default="clean_50,randomized_500")
    p.add_argument("--embodiments", default="aloha-agilex,arx-x5,franka,ur5")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    import requests
    tasks = a.tasks.split(",") if a.tasks else sorted(
        os.path.basename(x["path"]) for x in requests.get(
            BASE.replace("/datasets/", "/api/datasets/") + "/tree/main/dataset", timeout=60).json()
        if x["type"] == "directory")
    jobs = []
    for t in tasks:
        for e in a.embodiments.split(","):
            for v in a.variants.split(","):
                jobs.append((f"dataset/{t}/{e}_{v}.zip", os.path.join(a.out, t, f"{e}_{v}")))
    print(f"[fetch] {len(jobs)} zips from {len(tasks)} tasks", flush=True)
    failed = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_zip, r, d): r for r, d in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r, n, msg = fu.result()
                print(f"[{i}/{len(jobs)}] {r.split('/', 1)[1]}: {msg} {n / 1e6:.1f} MB", flush=True)
            except Exception as e:
                failed.append(futs[fu])
                print(f"[{i}/{len(jobs)}] FAIL {futs[fu]}: {type(e).__name__}: {e}", flush=True)
    print(f"[fetch] done, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
