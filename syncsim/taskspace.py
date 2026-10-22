"""Task-space edits of a recorded joint-space control stream.

The expert stream commands JOINT targets. An edit that should read as "the gripper ended up
3 cm to the left" (an imprecise approach, a missed grasp) is specified as a TCP offset AT ONE
MOMENT (the grasp, a noise point) and mapped to a joint offset there: the joint target that
puts the TCP `delta` away with the gripper ORIENTATION held, found by a few damped
least-squares steps on a numerical Jacobian of the arm's own forward
kinematics (SAPIEN's pinocchio model of the articulation, so nothing in the scene moves; its
poses are in the articulation ROOT frame -- aloha-agilex's root sits at y = -0.65 m, turned 90
deg -- and are mapped to world here). That joint offset is then ramped in smoothly and kept.

Why one solve, not one per frame: re-solving along the trajectory (tried first) flips IK
branches near wrist singularities -- 0.5-0.9 rad jumps between frames, TCP spikes of
1-7 m/s. A constant joint offset is smooth by construction; away from the solve frame its TCP
effect varies with the arm's pose, so `bound_ramps` checks it by forward kinematics at every
frame and scales the offset down if it would move the TCP beyond the edit's cap anywhere.
What the gates then measure beyond that is physics: contact with the scene.
Joint offsets are then interpolated between frames, so the edit is as smooth as the offset
track. Unlike additive joint noise, the TCP displacement is bounded by construction: a shoulder
offset cannot swing the hand 40 cm away.
"""

import numpy as np
import transforms3d as t3d


def _eid(x):
    return int(getattr(x, "entity", x).per_scene_id)


class ArmKinematics:
    """Forward kinematics of one arm's TCP from its joint targets, independent of the scene."""

    def __init__(self, robot, arm):
        art = getattr(robot, f"{arm}_entity")
        self.model = art.create_pinocchio_model()
        links = art.get_links()
        ee_link = getattr(robot, f"{arm}_ee").get_child_link()
        self.li = [_eid(l) for l in links].index(_eid(ee_link))
        act = art.get_active_joints()
        act_ids = [_eid(j.get_child_link()) for j in act]
        self.idx = [act_ids.index(_eid(j.get_child_link()))
                    for j in getattr(robot, f"{arm}_arm_joints")]
        self.q0 = np.asarray(art.get_qpos(), dtype=np.float64)
        root = art.get_root_pose()
        self.R_root = t3d.quaternions.quat2mat(root.q)
        self.p_root = np.asarray(root.p, dtype=np.float64)
        lim = np.asarray(art.get_qlimits(), dtype=np.float64)[self.idx]
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        # the TCP as a fixed point of the ee link (RoboTwin offsets it from the ee joint frame,
        # which is rigid with the link), measured on the live articulation once
        tcp = np.asarray(getattr(robot, f"get_{arm}_tcp_pose")(), dtype=np.float64)
        lp = ee_link.get_entity_pose() if hasattr(ee_link, "get_entity_pose") else ee_link.entity.pose
        R = t3d.quaternions.quat2mat(lp.q)
        self.p_local = R.T @ (tcp[:3] - np.asarray(lp.p))
        q_now = self.q0[self.idx]
        # the model's base frame is the articulation root; check it reproduces the live TCP
        self.base_err = float(np.linalg.norm(self.fk(q_now)[0] - tcp[:3]))

    def fk(self, q_arm):
        q = self.q0.copy()
        q[self.idx] = q_arm
        self.model.compute_forward_kinematics(q)
        P = self.model.get_link_pose(self.li)                 # root frame
        R = self.R_root @ t3d.quaternions.quat2mat(P.q)
        return self.p_root + self.R_root @ np.asarray(P.p) + R @ self.p_local, R

    def jacobian(self, q_arm, eps=1e-5):
        p0, R0 = self.fk(q_arm)
        J = np.zeros((6, len(q_arm)))
        for j in range(len(q_arm)):
            q = np.array(q_arm, dtype=np.float64)
            q[j] += eps
            p, R = self.fk(q)
            J[:3, j] = (p - p0) / eps
            ax, ang = t3d.axangles.mat2axangle(R @ R0.T)
            J[3:, j] = np.asarray(ax) * ang / eps
        return J

    def offset(self, q_arm, delta, q_init=None, iters=6, damping=1e-3, rot_weight=0.3, tol=2e-4,
               max_step=0.15):
        """Joint target whose TCP is `delta` (m, world) from q_arm's, same orientation.
        q_init: starting guess (default q_arm); steps are capped at `max_step` rad."""
        q_arm = np.asarray(q_arm, dtype=np.float64)
        p_ref, R_ref = self.fk(q_arm)
        target = p_ref + np.asarray(delta, dtype=np.float64)
        q = (q_arm if q_init is None else np.asarray(q_init, dtype=np.float64)).copy()
        for _ in range(iters):
            p, R = self.fk(q)
            e_p = target - p
            ax, ang = t3d.axangles.mat2axangle(R_ref @ R.T)
            e = np.concatenate([e_p, rot_weight * np.asarray(ax) * ang])
            if np.linalg.norm(e_p) < tol and ang < 1e-3:
                break
            J = self.jacobian(q)
            J[3:] *= rot_weight
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            m = np.abs(dq).max()
            if m > max_step:
                dq *= max_step / m
            q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.linalg.norm(self.fk(q)[0] - target))


def solve_offset(kin, q_arm, delta, max_residual_m=0.01, shrink=(1.0, 0.7, 0.5)):
    """Joint offset that moves `q_arm`'s TCP by `delta` (world, m) at fixed orientation,
    shrinking the request until the solve is accurate. Returns (dq, achieved delta, residual)
    or raises ValueError (no accurate solve: a near-singular pose)."""
    best = None
    for f in shrink:
        d = np.asarray(delta, dtype=np.float64) * f
        q, r = kin.offset(q_arm, d)
        if best is None or r < best[2]:
            best = (q - np.asarray(q_arm, dtype=np.float64), d, r)
        if r <= max_residual_m:
            return q - np.asarray(q_arm, dtype=np.float64), d, r
    raise ValueError(f"no accurate IK offset (best residual {best[2]:.3f} m)")


def tcp_effect(kin, q, eps, frame_stride):
    """Largest TCP displacement (m) that joint offsets eps [n, d] cause along joint targets
    q [n, d], by forward kinematics at every video frame (no scene, so no contacts)."""
    out = 0.0
    for i in range(0, len(q), frame_stride):
        if np.abs(eps[i]).max() < 1e-9:
            continue
        out = max(out, float(np.linalg.norm(kin.fk(q[i] + eps[i])[0] - kin.fk(q[i])[0])))
    return out


def bound_ramps(kin, q, ramps, cap_m, frame_stride):
    """Scale all joint offsets together so their TCP effect stays <= cap_m along the whole
    trajectory: a joint offset solved at one pose can move the TCP further at another (an
    outstretched arm). Returns (ramps, effect before, scale)."""
    eps = np.zeros_like(q)
    for dq, w in ramps:
        eps += np.asarray(w)[:, None] * np.asarray(dq)[None]
    eff = tcp_effect(kin, q, eps, frame_stride)
    scale = min(1.0, cap_m / eff) if eff > 0 else 1.0
    return [(np.asarray(dq) * scale, w) for dq, w in ramps], eff, scale


def apply_joint_ramps(stream, arm, ramps, dt):
    """Add sum_i w_i(t) * dq_i to `arm`'s joint targets; `ramps` is a list of (dq [d],
    weight [n steps]). Velocity targets get the matching derivative."""
    s = stream.copy()
    q = np.asarray(s[f"{arm}_q"], dtype=np.float64)
    eps = np.zeros_like(q)
    for dq, w in ramps:
        eps += np.asarray(w, dtype=np.float64)[:, None] * np.asarray(dq, dtype=np.float64)[None]
    s[f"{arm}_q"] = q + eps
    s[f"{arm}_dq"] = np.asarray(s[f"{arm}_dq"]) + np.gradient(eps, dt, axis=0)
    return s


def smooth_ramp(n, a, b):
    """[n] weights: 0 before frame a, smoothstep up to 1 at frame b, 1 after."""
    t = np.clip((np.arange(n) - a) / max(1.0, float(b - a)), 0.0, 1.0)
    return t * t * (3 - 2 * t)
