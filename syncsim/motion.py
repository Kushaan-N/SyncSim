"""Live scripted end-effector motions (calibration legs, scripted counterfactual suffixes).

A motion is "move arm A's end effector by a translation (world frame) or a rotation (about a
world axis, through the gripper centre)". It is planned from the CURRENT joint state as a
SCREW motion (mplib `plan_screw`: the end effector moves on a straight line / rotates about a
fixed axis), then RE-TIMED so its Cartesian speed is a controlled constant -- for calibration
"how fast does a unit of motion look" is the quantity being shown.

Why not the curobo planner RoboTwin's expert uses: it plans collision-free JOINT-space paths
and may change IK branch, so a 7 cm translation goal can swing the whole arm up and around
(observed on aloha-agilex). That shows the wrong thing for a single-DoF calibration leg.
curobo stays available as `method="curobo"` (used as a fallback for scripted counterfactual
suffixes, where a natural-looking detour is fine).

The result is a ControlStream segment that commands the moving arm along the re-timed path and
HOLDS everything else (other arm, both grippers) at its current target.
"""

import math

import numpy as np
import transforms3d as t3d

from .rt_env import STREAM_KEYS, ControlStream, current_targets

DOF_AXES = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}
TRANS_DOFS = ("x", "y", "z")
ROT_DOFS = ("roll", "pitch", "yaw")


def ee_pose(robot, arm):
    return np.asarray(getattr(robot, f"get_{arm}_ee_pose")(), dtype=np.float64)   # xyz + wxyz


# RoboTwin's `get_*_ee_pose` point sits 0.12 m behind the gripper centre (TCP) along the
# pose's own x axis, for every embodiment: Robot._trans_endpose offsets both from the same
# link by `gripper_bias - 0.12` and `gripper_bias` respectively.
EE_TO_TCP = 0.12


def tcp_point(pose7):
    """Gripper-centre position of an ee pose (xyz + wxyz)."""
    p = np.asarray(pose7, dtype=np.float64)
    return p[:3] + t3d.quaternions.quat2mat(p[3:7]) @ np.array([EE_TO_TCP, 0.0, 0.0])


def displaced_pose(pose7, dof, amount):
    """ee pose7 (xyz + wxyz) moved by `amount`: m along a world axis for x/y/z, rad about a
    world axis THROUGH THE GRIPPER CENTRE for roll/pitch/yaw. Rotating about the ee point
    instead would swing the gripper centre 3-4 cm per 20 deg, so a "roll" leg would also be a
    translation in the recorded (TCP) actions."""
    out = np.array(pose7, dtype=np.float64)
    if dof in TRANS_DOFS:
        out[DOF_AXES[dof]] += amount
    else:
        axis = np.zeros(3)
        axis[DOF_AXES[dof]] = 1.0
        tcp = tcp_point(out)
        R = t3d.axangles.axangle2mat(axis, amount) @ t3d.quaternions.quat2mat(out[3:7])
        out[3:7] = t3d.quaternions.mat2quat(R)
        out[:3] = tcp - R @ np.array([EE_TO_TCP, 0.0, 0.0])
    return out


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def plan_cartesian_chain(robot, arm, target7, step_m=0.006, step_rad=0.03, max_jump=0.25):
    """Straight-line EE path as a CHAIN of short curobo plans between closely spaced
    waypoints, each starting from the previous plan's end (`last_qpos`). Short hops cannot
    swing to another IK branch, so the chain follows the line. Used when the screw planner
    fails: from singular start poses (piper's all-zero home) and on embodiments whose URDF
    mplib's IK does not reproduce (piper: IK fails even at the arm's current pose)."""
    start7 = ee_pose(robot, arm)
    target7 = np.asarray(target7, float)
    dist = float(np.linalg.norm(target7[:3] - start7[:3]))
    ang = 2 * math.acos(min(1.0, abs(float(np.dot(start7[3:7], target7[3:7])))))
    n = max(1, int(math.ceil(max(dist / step_m, ang / step_rad))))
    ent = getattr(robot, f"{arm}_entity")
    act = ent.get_active_joints()
    idx = [act.index(j) for j in getattr(robot, f"{arm}_arm_joints")]
    q = np.asarray(ent.get_qpos(), dtype=np.float64)
    path = [q[idx].copy()]
    plan = getattr(robot, f"{arm}_plan_path")
    for i in range(1, n + 1):
        t = i / n
        w = np.concatenate([start7[:3] + t * (target7[:3] - start7[:3]),
                            _slerp(start7[3:7], target7[3:7], t)])
        # float32 like Entity.get_qpos(): curobo builds its start state with torch.tensor(...)
        # and a float64 qpos makes the planner raise (returned as {"error": ...})
        res = plan(list(map(float, w)), last_qpos=q.astype(np.float32))
        if not isinstance(res, dict) or res.get("status") != "Success":
            return None
        pos = np.asarray(res["position"], dtype=np.float64)
        if np.abs(pos[-1] - q[idx]).max() > max_jump:      # branch flip, not a short hop
            return None
        path.extend(pos[1:])
        q[idx] = pos[-1]
    return np.asarray(path)


def plan_to(robot, arm, target7, method="screw"):
    """Joint path [n, dof] moving `arm`'s end effector (RoboTwin ee-pose convention, xyz +
    wxyz) in a straight line to target7, or None.

    method: screw (screw plan, else a chain of short curobo hops -- both straight lines) | curobo (RoboTwin's
    collision-aware joint-space planner) | screw_then_curobo (straight line if possible)."""
    if method in ("screw", "screw_then_curobo"):
        try:
            pose = robot._trans_from_gripper_to_endlink(list(map(float, target7)), arm_tag=arm)
            qpos = getattr(robot, f"{arm}_entity").get_qpos()
            res = _screw_planner(robot, arm).plan_screw(qpos, pose)
            if isinstance(res, dict) and res.get("status") == "Success" and len(res["position"]):
                return np.asarray(res["position"], dtype=np.float64)
        except Exception:
            pass
        try:
            path = plan_cartesian_chain(robot, arm, target7)
            if path is not None:
                return path
        except Exception:
            pass
        if method == "screw":
            return None
    try:
        res = getattr(robot, f"{arm}_plan_path")(list(map(float, target7)))
    except Exception:
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"], dtype=np.float64)


def retime(path, n_steps, dt):
    """Resample a joint path [n, d] to n_steps samples, constant speed along the index;
    velocities by finite difference."""
    path = np.asarray(path, dtype=np.float64)
    n_steps = max(2, int(n_steps))
    if len(path) == 1:
        path = np.repeat(path, 2, axis=0)
    # arc-length parametrise in joint space so the resampling is uniform
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        q = np.repeat(path[-1:], n_steps, axis=0)
        return q, np.zeros_like(q)
    u = np.linspace(0, s[-1], n_steps)
    q = np.stack([np.interp(u, s, path[:, j]) for j in range(path.shape[1])], axis=1)
    dq = np.gradient(q, dt, axis=0)
    dq[-1] = 0.0
    return q, dq


def arm_segment(robot, arm, q, dq, base=None):
    """ControlStream that drives `arm` along (q, dq) and holds everything else."""
    base = base or current_targets(robot)
    n = len(q)
    seg = {}
    for k in STREAM_KEYS:
        v = np.asarray(base[k])
        seg[k] = np.repeat(v[None], n, axis=0) if v.ndim else np.full(n, float(v))
    seg[f"{arm}_q"] = np.asarray(q)
    seg[f"{arm}_dq"] = np.asarray(dq)
    return ControlStream(seg)


def hold(robot, n):
    """n steps holding every current target (both arms, both grippers)."""
    b = current_targets(robot)
    q = np.repeat(b["left_q"][None], int(n), axis=0)
    return arm_segment(robot, "left", q, np.zeros_like(q), base=b)
