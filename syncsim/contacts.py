"""Per-frame contact ground truth for a rollout, and the interaction events mined from it.

WHAT IS RECORDED (state, not events)
------------------------------------
Every physics step, SAPIEN's contact list is reduced to PART pairs. A part is

    left_finger / right_finger     a gripper finger link (child of a gripper joint)
    left_gripper / right_gripper   the rest of the gripper (palm, camera mount, ...)
    left_arm / right_arm           every other link on that arm's chain
    robot_base                     robot links on neither arm chain (torso, mobile base)
    obj:<name>                     a free-moving scene actor (task objects + clutter)
    fix:<name>                     a named static / kinematic task object (the bell, the stapler,
                                   a target pad, a dustbin): an object the robot interacts with,
                                   that just does not move freely
    art:<name>                     a link of a non-robot articulation (laptop lid, drawer, ...)
    env:<name>                     static scenery: table, wall, ground only

Pairs inside one robot part group (finger-finger of the same gripper, links of one arm) are
dropped: they are joint-internal, not interactions. Contacts are binned into video frames
exactly like the recorder bins poses: frame k covers physics steps ((k-1)*stride, k*stride],
frame 0 is the scene as set up. Per frame and pair we keep

    touch    fraction of the frame's physics steps with the pair touching
    impulse  summed normal impulse over those steps (N s); mean force = impulse / frame time
    pos      impulse-weighted mean contact point (world, m)

Contacts are a function of the simulation, not of what is filmed, so they can be recorded
on an unrendered replay of a rollout's saved control stream (scripts/annotate_contacts.py)
and are byte-identical to what a live recording would give.

WHAT IS DERIVED (events, cheap to recompute)
--------------------------------------------
`events()` turns the dense arrays + the object pose track into windows (frame x to frame y)
and per-rollout summary numbers: grasp / hold windows, gripper and arm contacts with each
object, robot-table and arm-arm collisions, object-object contacts that the rollout CAUSED
(pairs not touching in frame 0), how far each touched object moved during and after each
contact, and the interaction share of the rollout. These are what the dataset index
(scripts/contact_stats.py) samples on.
"""

import json
import os

import numpy as np
import sapien

ROBOT_PARTS = ("left_finger", "left_gripper", "left_arm", "right_finger", "right_gripper",
               "right_arm", "robot_base")
STATIC_NAMES = ("table", "wall", "ground")      # scenery; any other unlabelled actor is fix:
TOUCH_SEPARATION_M = 5e-4     # a contact point this close (or with impulse) counts as touching
ROBOT_OBJECT_KINDS = ("gripper-obj", "arm-obj", "body-obj", "art-gripper", "arm-art", "art-body")


def unique_names(names):
    """Actor names made unique in order: RoboTwin names both blocks of stack_blocks_two
    "box", so a repeated name gets "#k" (k-th occurrence, from 0) on EVERY occurrence."""
    names = [str(n) for n in names]
    dup = {n for n in names if names.count(n) > 1}
    seen = {}
    out = []
    for n in names:
        if n in dup:
            out.append(f"{n}#{seen.get(n, 0)}")
            seen[n] = seen.get(n, 0) + 1
        else:
            out.append(n)
    return out


def _eid(component_or_entity):
    e = getattr(component_or_entity, "entity", component_or_entity)
    return int(e.per_scene_id)


def _subtree(art, root_link):
    """Links of articulation `art` at or below `root_link`."""
    kids = {}
    for j in art.get_joints():
        p, c = j.get_parent_link(), j.get_child_link()
        if p is not None and c is not None:
            kids.setdefault(_eid(p), []).append(c)
    out, stack = [], [root_link]
    while stack:
        link = stack.pop()
        out.append(link)
        stack.extend(kids.get(_eid(link), []))
    return out


def robot_part_map(robot):
    """entity id -> robot part name, for every link of the robot's articulation(s)."""
    parts = {}
    arts = {_eid(robot.left_entity.get_root()): robot.left_entity,
            _eid(robot.right_entity.get_root()): robot.right_entity}
    for art in arts.values():
        for link in art.get_links():
            parts[_eid(link)] = "robot_base"
    for arm in ("left", "right"):
        art = getattr(robot, f"{arm}_entity")
        joints = getattr(robot, f"{arm}_arm_joints")
        for link in _subtree(art, joints[0].get_child_link()):
            parts[_eid(link)] = f"{arm}_arm"
        grip = getattr(robot, f"{arm}_gripper")             # [(joint, mult, offset), ...]
        palm = grip[0][0].get_parent_link()
        for link in _subtree(art, palm):
            parts[_eid(link)] = f"{arm}_gripper"
        for name in getattr(robot, f"{arm}_fix_gripper_name", []) or []:
            link = art.find_link_by_name(name)
            if link is not None:
                parts[_eid(link)] = f"{arm}_gripper"
        for j, _, _ in grip:
            parts[_eid(j.get_child_link())] = f"{arm}_finger"
    return parts


def _group(part):
    """Parts whose mutual contacts are joint-internal share a group."""
    if part.startswith(("left_", "right_")):
        return part.split("_")[0]
    return part


def relabel(part):
    """v1 labels -> current: contacts recorded before fix: existed called every unlabelled
    actor env:<name>; only table / wall / ground are scenery."""
    if part.startswith("env:") and part[4:] not in STATIC_NAMES:
        return "fix:" + part[4:]
    return part


def pair_kind(a, b):
    """Coarse category of a part pair, used for summaries and sampling."""
    def cls(p):
        if p.endswith(("_finger", "_gripper")):
            return "gripper"
        if p.endswith("_arm"):
            return "arm"
        if p == "robot_base":
            return "body"
        c = p.split(":")[0]                     # obj / fix / art / env
        return "obj" if c == "fix" else c       # a fixed task object is still an object
    ca, cb = sorted((cls(a), cls(b)))
    if {ca, cb} <= {"gripper", "arm", "body"}:
        return "robot-robot"
    return f"{ca}-{cb}"


class ContactTracker:
    """Accumulates part-pair contacts every physics step; `end_frame()` closes a frame."""

    def __init__(self, task, frame_stride):
        self.scene = task.scene
        self.stride = int(frame_stride)
        self.dt = float(task.scene.get_timestep())
        self.label = robot_part_map(task.robot)
        from .recorder import dynamic_actors, object_articulations
        self.objects = dynamic_actors(self.scene)
        for e, n in zip(self.objects, unique_names(e.get_name() for e in self.objects)):
            self.label[_eid(e)] = f"obj:{n}"
        robot_ids = set(self.label)
        for a in object_articulations(self.scene, task.robot):
            for link in a.get_links():
                if _eid(link) not in robot_ids:
                    self.label[_eid(link)] = f"art:{a.get_name()}"
        self.pairs = {}                         # (a, b) -> column index
        self.frames = []                        # per frame: {col: (touch frac, impulse, pos*imp)}
        self.cur = {}
        self.unknown = set()
        self._steps_in_frame = 0
        self.after_step()                       # frame 0 = the contacts of the set-up scene
        self.end_frame()

    def _name(self, body):
        eid = _eid(body)
        lab = self.label.get(eid)
        if lab is None:
            name = body.entity.get_name() or f"id{eid}"
            if name in STATIC_NAMES:
                lab = f"env:{name}"
            else:
                lab = f"fix:{name}"
                self.unknown.add(name)
            self.label[eid] = lab
        return lab

    def _read(self):
        """Touching part pairs of the most recent physics step: {col: (impulse, pos*impulse)}."""
        out = {}
        for c in self.scene.get_contacts():
            a, b = self._name(c.bodies[0]), self._name(c.bodies[1])
            if a == b or (a in ROBOT_PARTS and b in ROBOT_PARTS and _group(a) == _group(b)):
                continue
            imp, pw, touching = 0.0, np.zeros(3), False
            for pt in c.points:
                m = float(np.linalg.norm(pt.impulse))
                touching |= m > 0 or pt.separation < TOUCH_SEPARATION_M
                imp += m
                pw += m * np.asarray(pt.position, dtype=np.float64)
            if not touching:
                continue
            col = self.pairs.setdefault((a, b) if a <= b else (b, a), len(self.pairs))
            i0, p0 = out.get(col, (0.0, 0.0))
            out[col] = (i0 + imp, p0 + pw)
        return out

    def after_step(self):
        """Call after every scene.step()."""
        for col, (imp, pw) in self._read().items():
            acc = self.cur.setdefault(col, [0, 0.0, np.zeros(3)])
            acc[0] += 1
            acc[1] += imp
            acc[2] += pw
        self._steps_in_frame += 1

    def end_frame(self):
        steps = max(1, self._steps_in_frame)
        self.frames.append({col: (v[0] / steps, v[1], v[2]) for col, v in self.cur.items()})
        self.cur = {}
        self._steps_in_frame = 0

    # ------------------------------------------------------------------------------------
    def arrays(self):
        T, P = len(self.frames), len(self.pairs)
        touch = np.zeros((T, P), np.float32)
        impulse = np.zeros((T, P), np.float32)
        pos = np.zeros((T, P, 3), np.float32)
        for t, fr in enumerate(self.frames):
            for col, (tf, imp, pw) in fr.items():
                touch[t, col] = tf
                impulse[t, col] = imp
                if imp > 0:
                    pos[t, col] = pw / imp
        keys = sorted(self.pairs, key=self.pairs.get)
        return {"pair_a": np.array([k[0] for k in keys]), "pair_b": np.array([k[1] for k in keys]),
                "pair_kind": np.array([pair_kind(*k) for k in keys]),
                "touch": touch, "impulse": impulse, "pos": pos,
                "frame_time_s": np.float32(self.dt * self.stride)}

    def save(self, out_dir, object_poses=None, object_names=None, extra=None, name="contacts",
             gripper_open=None):
        """Write <name>.npz (dense) + <name>.json (events + summary). `gripper_open`
        {arm: [T] commanded opening, 1 = open} lets held windows require a closed gripper."""
        A = self.arrays()
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **A)
        ev = events(A, object_poses, object_names, gripper_open=gripper_open)
        if extra:
            ev.update(extra)
        if self.unknown:
            ev["fixed_objects"] = sorted(self.unknown)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(ev, f, indent=1)
        return ev


# ----------------------------------------------------------------------------------------
# events: windows + summary, from the dense arrays alone (plus object poses if given)
# ----------------------------------------------------------------------------------------
def windows(active, max_gap=1, min_len=1):
    """[start, end] (inclusive) runs of True in a 1-D bool array, bridging gaps <= max_gap."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    out, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e - 1 <= max_gap:
            e = i
        else:
            out.append([int(s), int(e)])
            s = e = i
    out.append([int(s), int(e)])
    return [w for w in out if w[1] - w[0] + 1 >= min_len]


def _disp(track, a, b):
    if track is None:
        return None
    b = min(b, len(track) - 1)
    return float(np.linalg.norm(track[b, :3] - track[a, :3]))


def events(A, object_poses=None, object_names=None, after_frames=10, lift_m=0.01,
           gripper_open=None):
    """Contact windows and per-rollout interaction summary. `object_poses` [T, O, 7] (xyz +
    quat) and `object_names` [O] (actor names, as in object_poses.npz, in recorder order) add
    object motion; repeated names are disambiguated exactly as the tracker's labels are.
    `gripper_open` {arm: [T] commanded opening in [0, 1]}: held needs that gripper closed.
    v1 part labels (env:<task object>) are mapped to fix: on the fly (`relabel`)."""
    pa = [relabel(str(x)) for x in A["pair_a"]]
    pb = [relabel(str(x)) for x in A["pair_b"]]
    kind = [pair_kind(a, b) for a, b in zip(pa, pb)]
    touch, imp = np.asarray(A["touch"]), np.asarray(A["impulse"])
    T = touch.shape[0]
    ft = float(A["frame_time_s"])
    on = touch > 0
    obj_track = {}
    if object_poses is not None and object_names is not None:
        P = np.asarray(object_poses)
        for i, n in enumerate(unique_names(object_names)):
            obj_track[f"obj:{n}"] = P[:T, i]

    def obj_of(a, b):
        for p in (a, b):
            if p.startswith(("obj:", "fix:", "art:")):
                return p
        return None

    wins = []
    for j in range(len(pa)):
        for s, e in windows(on[:, j]):
            scene = ("obj:", "fix:", "art:", "env:")
            if s == 0 and pa[j].startswith(scene) and pb[j].startswith(scene):
                started = "initial"             # resting contact present at set-up
            else:
                started = "rollout"
            w = {"a": pa[j], "b": pb[j], "kind": kind[j], "start": s, "end": e,
                 "frames": e - s + 1, "started": started,
                 "peak_force_n": float(imp[s:e + 1, j].max() / ft),
                 "mean_force_n": float(imp[s:e + 1, j].mean() / ft)}
            o = obj_of(pa[j], pb[j])
            if o is not None:
                w["object"] = o
            if o in obj_track:
                w["obj_disp_during_m"] = _disp(obj_track[o], s, e)
                w["obj_disp_after_m"] = _disp(obj_track[o], e, e + after_frames)
            wins.append(w)

    # ---- per arm: held / grasp windows ------------------------------------------------
    held = {}
    for arm in ("left", "right"):
        fing = [j for j in range(len(pa)) if f"{arm}_finger" in (pa[j], pb[j])
                and obj_of(pa[j], pb[j])]
        by_obj = {}
        for j in fing:
            by_obj.setdefault(obj_of(pa[j], pb[j]), []).append(j)
        closed = None
        if gripper_open is not None and arm in gripper_open:
            g = np.asarray(gripper_open[arm], dtype=np.float64)[:T]
            closed = np.zeros(T, bool)
            closed[:len(g)] = g < 0.5
        for o, cols in by_obj.items():
            if o.startswith("fix:"):
                continue                        # a fixed object is pressed / touched, not held
            # held = the arm's fingers touch the object, its gripper is commanded closed, and
            # the object is off every static surface (scenery or a fixed object) or above where
            # it started -- a lifted / carried object. Without the closed test an open finger
            # resting on a stacked block read as "held" (43 of 729 sampled v1 windows).
            pinch = on[:, cols].any(axis=1)
            if closed is not None:
                pinch &= closed
            off = np.ones(T, bool)
            for j in range(len(pa)):
                other = pb[j] if pa[j] == o else pa[j] if pb[j] == o else None
                if other is not None and other.startswith(("env:", "fix:")):
                    off &= ~on[:, j]
            lifted = np.zeros(T, bool)
            if o in obj_track:
                z = obj_track[o][:, 2]
                lifted = z > z[0] + lift_m
            h = pinch & (off | lifted)
            for s0, e0 in windows(h, max_gap=2, min_len=3):
                # a bridged gap may not span an OPEN gripper (a flip that opens it for 2 frames)
                ok = closed[s0:e0 + 1] if closed is not None else np.ones(e0 - s0 + 1, bool)
                for s, e in windows(ok, max_gap=0, min_len=3):
                    held.setdefault(arm, []).append({"object": o, "start": s0 + s, "end": s0 + e,
                                                     "frames": e - s + 1})

    # ---- summary ----------------------------------------------------------------------
    def frac(mask):
        return float(mask.mean()) if T else 0.0

    def cols(pred):
        return [j for j in range(len(pa)) if pred(pa[j], pb[j], kind[j])]

    robot_obj = cols(lambda a, b, k: k in ROBOT_OBJECT_KINDS)
    grip_obj = cols(lambda a, b, k: k in ("gripper-obj", "art-gripper"))
    arm_obj = cols(lambda a, b, k: k in ("arm-obj", "arm-art", "body-obj", "art-body"))
    robot_env = cols(lambda a, b, k: k in ("env-gripper", "arm-env", "body-env"))
    arm_arm = cols(lambda a, b, k: k == "robot-robot")
    obj_obj_new = []
    for j in cols(lambda a, b, k: k in ("obj-obj", "art-obj")):
        if not on[0, j] and on[:, j].any():
            obj_obj_new.append(j)

    def any_on(cs):
        return on[:, cs].any(axis=1) if cs else np.zeros(T, bool)

    def n_win(cs):
        return sum(len(windows(on[:, j])) for j in cs)

    held_frames = np.zeros(T, bool)
    for arm, ws in held.items():
        for w in ws:
            held_frames[w["start"]:w["end"] + 1] = True
    touched = sorted({w["object"] for w in wins if "object" in w and w["kind"] in ROBOT_OBJECT_KINDS})
    moved = {o: float(np.linalg.norm(obj_track[o][-1, :3] - obj_track[o][0, :3]))
             for o in obj_track}
    summary = {
        "frames": int(T), "frame_time_s": ft,
        "robot_object_contact_frac": frac(any_on(robot_obj)),
        "gripper_object_contact_frac": frac(any_on(grip_obj)),
        "held_frac": frac(held_frames),
        "arm_object_contact_frac": frac(any_on(arm_obj)),
        "robot_env_contact_frac": frac(any_on(robot_env)),
        "arm_arm_contact_frac": frac(any_on(arm_arm)),
        "n_gripper_object_events": n_win(grip_obj),
        "n_arm_object_events": n_win(arm_obj),
        "n_robot_env_events": n_win(robot_env),
        "n_arm_arm_events": n_win(arm_arm),
        "n_caused_object_object_events": n_win(obj_obj_new),
        "n_held_windows": sum(len(v) for v in held.values()),
        "interaction_s": float(any_on(robot_obj).sum() * ft),
        "objects_touched": touched,
        "n_objects_touched": len(touched),
        "objects_moved_gt_2cm": sorted(o for o, d in moved.items() if d > 0.02),
    }
    return {"format": 1, "summary": summary, "held": held, "windows": wins}
