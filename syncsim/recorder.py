"""Per-rollout frame capture: N simultaneous 3rd-person cameras + both wrist cameras, plus the
bimanual pose / action streams written next to the videos.

On-disk layout of one rollout dir (mirrors robocasa-dev's, extended to two arms):

    agentview_rgb/video.mp4        view 0      (legacy single-view name)
    agentview_rgb_<v>/video.mp4    view v >= 1
    wrist_left_rgb/video.mp4       left wrist camera (jittered mount)
    wrist_right_rgb/video.mp4      right wrist camera
    pose.pkl                       per frame, both arms (see `save`)
    actions.npy                    [T, 16] per-frame action, see ACTION_LAYOUT
    joints.npy                     [T, dl+1+dr+1] measured arm joints + gripper (RoboTwin "qpos")
    object_poses.npz               per frame world pose of every free-moving scene actor, and
                                   joint positions + root pose of every articulated object
    metadata.json                  written by the caller

Quaternions are stored xyzw (scipy / robosuite order, as in robocasa's pose.pkl) even though
RoboTwin / SAPIEN use wxyz internally.
"""

import json
import os
import pickle

import numpy as np
import sapien
import transforms3d as t3d

from .cameras import add_camera, pose7_to_mat, sapien_pose_to_mat

ACTION_LAYOUT = {
    "left_pos": [0, 3], "left_quat_xyzw": [3, 7], "left_gripper": [7, 8],
    "right_pos": [8, 11], "right_quat_xyzw": [11, 15], "right_gripper": [15, 16],
    "frame": "world", "pose": "TCP (RoboTwin get_*_tcp_pose: gripper centre)",
    "gripper": "commanded opening in [0, 1], 1 = open",
}
WRIST_DIRS = {"left": "wrist_left_rgb", "right": "wrist_right_rgb"}


def agentview_dir(v):
    return "agentview_rgb" if v == 0 else f"agentview_rgb_{v}"


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def write_video(frames, path, fps, crf=18):
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
                       pixelformat="yuv420p", ffmpeg_params=["-crf", str(crf), "-preset", "fast"])
    try:
        for f in frames:
            w.append_data(np.asarray(f, dtype=np.uint8))
    finally:
        w.close()


def dynamic_actors(scene):
    """Non-static, named scene actors (task objects + clutter) in a stable order."""
    out = []
    for e in scene.get_all_actors():
        name = e.get_name()
        if not name or name in ("table", "wall", "ground"):
            continue
        comp = e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if comp is None:
            continue
        out.append(e)
    return out


def object_articulations(scene, robot):
    """Non-robot articulations with at least one movable joint, in a stable order."""
    def root_id(a):
        return int(a.get_root().entity.per_scene_id)

    robot_roots = {root_id(robot.left_entity), root_id(robot.right_entity)}
    out = []
    for a in scene.get_all_articulations():
        if root_id(a) in robot_roots or not a.get_active_joints():
            continue
        out.append(a)
    return out


class RigRecorder:
    """Adds the rig's cameras to a freshly set-up task scene and captures frames."""

    def __init__(self, task, rig, H, W, shader="rt", record_wrist=True):
        self.task = task
        self.scene = task.scene
        self.robot = task.robot
        self.agent = []
        for v, view in enumerate(rig["views"]):
            cam = add_camera(self.scene, f"syncsim_agent_{v}", W, H, rig["fovy_deg"], shader=shader)
            cam.entity.set_pose(sapien.Pose(np.asarray(view["mat"])))
            self.agent.append(cam)
        self.wrist = {}
        if record_wrist:
            for a in ("left", "right"):
                self.wrist[a] = (add_camera(self.scene, f"syncsim_wrist_{a}", W, H,
                                            rig["wrist_fovy_deg"], shader=shader),
                                 np.asarray(rig["wrist"][a]["rel_mat"]))
        self.frames = {agentview_dir(v): [] for v in range(len(self.agent))}
        for a in self.wrist:
            self.frames[WRIST_DIRS[a]] = []
        self.actors = dynamic_actors(self.scene)
        self.arts = object_articulations(self.scene, self.robot)
        self.log = {k: [] for k in ("left_pose", "right_pose", "left_mat", "right_mat",
                                    "left_g", "right_g", "left_g_real", "right_g_real",
                                    "joints", "step", "objects", "art_qpos", "art_root")}

    # ------------------------------------------------------------------------------------
    def poses_now(self):
        r = self.robot
        lp = np.asarray(r.get_left_tcp_pose(), dtype=np.float64)      # xyz + wxyz
        rp = np.asarray(r.get_right_tcp_pose(), dtype=np.float64)
        return lp, rp

    def capture(self, step_idx, render=True):
        r = self.robot
        if render:
            for a, (cam, rel) in self.wrist.items():
                link = getattr(r, f"{a}_camera")
                cam.entity.set_pose(sapien.Pose(sapien_pose_to_mat(link.get_pose()) @ rel))
            self.scene.update_render()
            for v, cam in enumerate(self.agent):
                cam.take_picture()
                self.frames[agentview_dir(v)].append(_rgb(cam))
            for a, (cam, _) in self.wrist.items():
                cam.take_picture()
                self.frames[WRIST_DIRS[a]].append(_rgb(cam))
        lp, rp = self.poses_now()
        L = self.log
        L["left_pose"].append(lp); L["right_pose"].append(rp)
        L["left_mat"].append(pose7_to_mat(lp)); L["right_mat"].append(pose7_to_mat(rp))
        L["left_g"].append(float(r.left_gripper_val)); L["right_g"].append(float(r.right_gripper_val))
        g_real = r.get_normal_real_gripper_val()
        L["left_g_real"].append(float(g_real[0])); L["right_g_real"].append(float(g_real[1]))
        L["joints"].append(np.asarray(r.get_left_arm_real_jointState()
                                      + r.get_right_arm_real_jointState(), dtype=np.float64))
        L["step"].append(int(step_idx))
        L["objects"].append(np.array([np.concatenate([e.get_pose().p, e.get_pose().q])
                                      for e in self.actors], dtype=np.float64).reshape(-1, 7))
        L["art_qpos"].append([np.asarray(a.get_qpos(), dtype=np.float64) for a in self.arts])
        L["art_root"].append([np.concatenate([a.get_root_pose().p, a.get_root_pose().q])
                              for a in self.arts])

    # ------------------------------------------------------------------------------------
    def tcp_track(self):
        """[T, 2, 3] TCP positions of (left, right)."""
        return np.stack([np.asarray(self.log["left_pose"])[:, :3],
                         np.asarray(self.log["right_pose"])[:, :3]], axis=1)

    def gripper_track(self):
        return np.stack([self.log["left_g"], self.log["right_g"]], axis=1)

    def actions(self):
        L = self.log
        lp, rp = np.asarray(L["left_pose"]), np.asarray(L["right_pose"])
        return np.concatenate([lp[:, :3], wxyz_to_xyzw(lp[:, 3:7]), np.asarray(L["left_g"])[:, None],
                               rp[:, :3], wxyz_to_xyzw(rp[:, 3:7]), np.asarray(L["right_g"])[:, None]],
                              axis=1).astype(np.float32)

    def save(self, out_dir, fps, frame_stride, crf=18, videos=True):
        os.makedirs(out_dir, exist_ok=True)
        if videos:
            for d, fr in self.frames.items():
                if fr:
                    write_video(fr, os.path.join(out_dir, d, "video.mp4"), fps, crf=crf)
        L = self.log
        lp, rp = np.asarray(L["left_pose"]), np.asarray(L["right_pose"])
        pose = {
            "left_gripper_pose": np.concatenate([lp[:, :3], wxyz_to_xyzw(lp[:, 3:7])], axis=1),
            "right_gripper_pose": np.concatenate([rp[:, :3], wxyz_to_xyzw(rp[:, 3:7])], axis=1),
            "left_gripper_matrix": np.asarray(L["left_mat"]),
            "right_gripper_matrix": np.asarray(L["right_mat"]),
            "left_gripper_open": np.asarray(L["left_g"]),
            "right_gripper_open": np.asarray(L["right_g"]),
            "left_gripper_open_measured": np.asarray(L["left_g_real"]),
            "right_gripper_open_measured": np.asarray(L["right_g_real"]),
            "physics_step": np.asarray(L["step"], dtype=np.int64),
            "save_stride": int(frame_stride),
            "fps": float(fps),
            "quat_order": "xyzw",
        }
        with open(os.path.join(out_dir, "pose.pkl"), "wb") as f:
            pickle.dump(pose, f)
        np.save(os.path.join(out_dir, "actions.npy"), self.actions())
        np.save(os.path.join(out_dir, "joints.npy"), np.asarray(L["joints"], dtype=np.float32))
        arts = {}
        for i, a in enumerate(self.arts):
            # articulated objects (laptop lid, microwave door, cabinet drawer, ...): their state
            # is joint positions, not a free-body pose. One array per articulation (dof varies).
            arts[f"art{i}_qpos"] = np.asarray([f[i] for f in L["art_qpos"]], dtype=np.float32)
            arts[f"art{i}_root_pose"] = np.asarray([f[i] for f in L["art_root"]], dtype=np.float32)
            arts[f"art{i}_joint_names"] = np.array([j.get_name() for j in a.get_active_joints()])
        np.savez_compressed(os.path.join(out_dir, "object_poses.npz"),
                            poses=np.asarray(L["objects"], dtype=np.float32),
                            names=np.array([e.get_name() for e in self.actors]),
                            art_names=np.array([a.get_name() for a in self.arts]),
                            quat_order="wxyz", **arts)
        return len(L["step"])


def _rgb(cam):
    rgba = cam.get_picture("Color")
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
