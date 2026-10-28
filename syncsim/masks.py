"""Per-frame object masks (instance segmentation) for every camera of a rollout.

Each filmed view gets a label video beside its RGB one, frame for frame:

    agentview_rgb/mask.mkv, agentview_rgb_1/mask.mkv, wrist_left_rgb/mask.mkv, ...
    masks.json      label table: pixel value -> part name, plus the views written

A pixel's value is the PART it shows, with the same part names as contacts.py, so a mask and a
contact window talk about the same things:

    0                    background: nothing drawn there (outside the room)
    1..7                 left_finger, left_gripper, left_arm, right_finger, right_gripper,
                         right_arm, robot_base
    8..                  obj:<name>, fix:<name>, art:<name>, env:<name> (table, wall, ground),
                         numbered in a stable order per rollout (see masks.json)

Masks come from SAPIEN's rasterised "Segmentation" picture (channel 1 = the entity a pixel
shows; robot links are entities, so each link maps to its part) taken by cameras with the
SAME pose, field of view and size as the RGB cameras -- wrist cameras re-posed from their link
every frame exactly as the recorder does. Geometry is identical, so a mask aligns with the
ray-traced RGB frame up to anti-aliasing on silhouettes.

Encoding: lossless FFV1 8-bit gray in Matroska (bit-exact round trip; masks compress to a few
hundred KB per view). Read with `read_mask_video`.

Masks are a function of the simulation + camera rig, not of what was filmed, so like contacts
they can be added to an existing rollout by an unrendered-physics replay of its saved stream
(scripts/annotate_masks.py) or captured live (`Rollout(..., masks=MaskRig(...))`).
"""

import json
import os

import numpy as np
import sapien

from .cameras import add_camera, sapien_pose_to_mat
from .contacts import ROBOT_PARTS, STATIC_NAMES, _eid, robot_part_map, unique_names

MASK_FILE = "mask.mkv"
# near clipping plane of the mask cameras. The rasteriser drops geometry nearer than `near`,
# and a wrist camera sits a few cm from a held object: at the default 5 cm the bottom of a
# block in the gripper was cut away and the table showed through it in the mask, while the
# ray-traced RGB (which has no near cut) showed the whole block.
MASK_NEAR_M = 0.002


def part_labels(task):
    """entity id -> part name for every labelled entity of the scene (contacts.py's names)."""
    from .recorder import dynamic_actors, object_articulations
    scene = task.scene
    lab = dict(robot_part_map(task.robot))
    objs = dynamic_actors(scene)
    for e, n in zip(objs, unique_names(e.get_name() for e in objs)):
        lab[_eid(e)] = f"obj:{n}"
    for a in object_articulations(scene, task.robot):
        for link in a.get_links():
            lab.setdefault(_eid(link), f"art:{a.get_name()}")
    for e in scene.get_entities():
        eid = _eid(e)
        if eid in lab:
            continue
        name = e.get_name()
        if not name or name.startswith(("syncsim_", "camera", "head_camera", "front_camera")):
            continue
        has_body = any(isinstance(c, (sapien.physx.PhysxRigidBaseComponent,
                                      sapien.render.RenderBodyComponent))
                       for c in e.get_components())
        if has_body:
            lab[eid] = f"env:{name}" if name in STATIC_NAMES else f"fix:{name}"
    return lab


def label_ids(labels):
    """part name -> pixel value: robot parts first (fixed values 1..7), then the scene's parts
    in sorted order. Returns (name -> value, value -> name)."""
    names = list(ROBOT_PARTS) + sorted({v for v in labels.values() if v not in ROBOT_PARTS})
    if len(names) > 255:
        raise ValueError(f"{len(names)} parts do not fit an 8-bit mask")
    to_id = {n: i + 1 for i, n in enumerate(names)}
    return to_id, {i: n for n, i in to_id.items()}


class MaskRig:
    """Rasterised segmentation cameras mirroring a recorder rig; `capture()` once per frame."""

    def __init__(self, task, rig, H, W, record_wrist=True):
        from .recorder import WRIST_DIRS, agentview_dir
        self.task, self.scene, self.robot = task, task.scene, task.robot
        self.cams = []                                   # (dir name, camera, wrist arm | None, rel)
        for v, view in enumerate(rig["views"]):
            cam = add_camera(self.scene, f"syncsim_mask_{v}", W, H, rig["fovy_deg"], shader="default",
                             near=MASK_NEAR_M)
            cam.entity.set_pose(sapien.Pose(np.asarray(view["mat"])))
            self.cams.append((agentview_dir(v), cam, None, None))
        if record_wrist:
            for a in ("left", "right"):
                cam = add_camera(self.scene, f"syncsim_mask_wrist_{a}", W, H, rig["wrist_fovy_deg"],
                                 shader="default", near=MASK_NEAR_M)
                self.cams.append((WRIST_DIRS[a], cam, a, np.asarray(rig["wrist"][a]["rel_mat"])))
        self.labels = part_labels(task)
        self.to_id, self.names = label_ids(self.labels)
        top = max(self.labels) + 1 if self.labels else 1
        self.lut = np.zeros(max(top, 1), dtype=np.uint8)
        for eid, name in self.labels.items():
            self.lut[eid] = self.to_id[name]
        self.frames = {d: [] for d, *_ in self.cams}

    def capture(self):
        for d, cam, arm, rel in self.cams:
            if arm is not None:
                link = getattr(self.robot, f"{arm}_camera")
                cam.entity.set_pose(sapien.Pose(sapien_pose_to_mat(link.get_pose()) @ rel))
        self.scene.update_render()
        for d, cam, _, _ in self.cams:
            cam.take_picture()
            ids = cam.get_picture("Segmentation")[..., 1].astype(np.int64)
            ids = np.where((ids >= 0) & (ids < len(self.lut)), ids, 0)
            self.frames[d].append(self.lut[ids])

    def save(self, out_dir, fps, extra=None):
        views = {}
        for d, fr in self.frames.items():
            if not fr:
                continue
            path = os.path.join(out_dir, d, MASK_FILE)
            write_mask_video(fr, path, fps)
            present = np.unique(np.stack(fr))
            views[d] = {"file": f"{d}/{MASK_FILE}", "frames": len(fr),
                        "parts_visible": [self.names[int(i)] for i in present if int(i) in self.names]}
        meta = {"format": 1, "encoding": "ffv1 gray8 (lossless), pixel value = part id, 0 = none",
                "labels": {str(i): n for i, n in self.names.items()}, "views": views}
        if extra:
            meta.update(extra)
        tmp = os.path.join(out_dir, "masks.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(out_dir, "masks.json"))
        return meta


def write_mask_video(frames, path, fps):
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = iio.get_writer(path, fps=fps, codec="ffv1", pixelformat="gray", macro_block_size=1,
                       quality=None, ffmpeg_params=["-level", "3"])
    try:
        for f in frames:
            w.append_data(np.asarray(f, dtype=np.uint8))
    finally:
        w.close()


def read_mask_video(path):
    """[T, H, W] uint8 part ids (exactly as written)."""
    import imageio.v2 as iio
    r = iio.get_reader(path)
    try:
        out = np.stack([np.asarray(f) for f in r])
    finally:
        r.close()
    return out[..., 0] if out.ndim == 4 else out
