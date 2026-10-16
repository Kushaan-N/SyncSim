"""Bimanual counterfactual episodes.

Every counterfactual starts from the expert's seeded initial scene and differs from the expert
in its CONTROLS only. Modes (one per counterfactual index, drawn with `cf_mode_probs`):

  noisy_replay     the expert control stream with sparse, smooth JOINT-SPACE perturbations
                   on one arm, the other, or both (`cf_arm_probs`). By default the offsets
                   PERSIST after their ramp (`noise_persist`), the joint-space analogue of
                   robocasa's integrating OSC deltas -- a pure bump would let the arm snap
                   back to the expert path. (Gripper flips are their own mode, below.)
  scripted_diverge expert prefix to a breakpoint, then 1-2 scripted end-effector motions
                   planned LIVE from the breakpoint state for the chosen arm(s); the other
                   arm (if not chosen) and both grippers HOLD their breakpoint command
                   (robocasa's opt-in `hold` mode -- its legacy default reopened the gripper,
                   confounding "arm went elsewhere" with "arm dropped the object").
                   Every chosen arm must get a plan in every segment, and a plan whose joint
                   path is a long detour (an IK-branch swing) is refused; segments are timed
                   so neither the end effector nor any joint exceeds its speed limit.
  freeze_arm       BIMANUAL-ONLY. From the breakpoint one arm holds its command while the
                   other keeps executing the expert: coordination breaks (no handover, the
                   second hand never arrives) without either arm doing anything unnatural.
  gripper_flip     the expert stream with one arm's gripper command inverted from a sampled
                   step to the end: a premature release / an unwanted grasp.

Robocasa lessons kept:
  * a perturbation that does not visibly change anything is useless -> every rollout is
    measured against the expert (max TCP deviation and max gripper rotation, per arm) and
    re-rolled with escalated magnitude below `cf_min_tcp_deviation_m` /
    `cf_min_rot_deviation_deg`, unless its gripper commands differ from the expert's on
    >= `cf_min_gripper_mismatch_frac` of frames (a release IS a counterfactual);
  * a rollout must also be physically plausible: no end effector may move faster than
    `cf_max_tcp_speed_mps` (or 1.5x the expert's own peak, if that is higher) -- a short
    segment that executes a long joint-space path swings the arm at metres per second;
  * breakpoints reserve a minimum suffix, so a late breakpoint cannot yield an expert-with-a-
    3-step-tail;
  * the gripper is perturbed by an explicit flip, never by additive noise (RoboTwin grippers
    are position-driven, so noise WOULD act -- but mid-grasp noise mostly jiggles the fingers;
    a flip is the event we want to show).
Unlike robocasa, expert and counterfactuals are both stream replays, so there is no
state-replay floor and the expert itself is the divergence reference.
"""

import math

import numpy as np

from .motion import (ROT_DOFS, TRANS_DOFS, arm_segment, displaced_pose, ee_pose, hold, plan_to,
                     retime)
from .rt_env import ControlStream, current_targets

MODES = ("noisy_replay", "scripted_diverge", "freeze_arm", "gripper_flip")


def pick(rng, probs: dict):
    keys = list(probs)
    p = np.asarray([probs[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def arms_of(sel):
    return ["left", "right"] if sel == "both" else [sel]


def sample_breakpoint(rng, n_steps, frame_stride, min_prefix_frames, min_suffix_frames):
    lo = min_prefix_frames * frame_stride
    hi = n_steps - min_suffix_frames * frame_stride
    if hi <= lo:
        lo, hi = 0, max(1, n_steps // 2)
    # breakpoints sit on frame boundaries so "frame index of the divergence" is exact
    p = int(rng.integers(lo, hi))
    return (p // frame_stride) * frame_stride


# ----------------------------------------------------------------------------------------
# stream edits
# ----------------------------------------------------------------------------------------
def _smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def noisy_replay(stream, rng, cfg, arms, escalation, frame_stride, dt):
    s = stream.copy()
    n = s.n
    dt_frames = cfg.noise_interval_frames * frame_stride       # ramp half-width in steps
    info = {"arms": arms, "points": {}, "sigma_rad": cfg.noise_sigma_rad * escalation,
            "persist": bool(cfg.noise_persist), "interval_frames": cfg.noise_interval_frames}
    for arm in arms:
        q = s[f"{arm}_q"]
        d = q.shape[1]
        lo, hi = dt_frames, max(dt_frames + 1, n - dt_frames)
        k = max(1, int(round(cfg.noise_points_per_100_frames * (n / frame_stride) / 100.0)))
        pts = sorted(rng.choice(np.arange(lo, hi), size=min(k, hi - lo), replace=False).tolist())
        eps = np.zeros_like(q)
        t = np.arange(n, dtype=np.float64)
        deltas = []
        for p in pts:
            delta = rng.normal(0.0, cfg.noise_sigma_rad * escalation, size=d)
            if cfg.noise_wrist_scale != 1.0 and d >= 3:
                delta[-3:] *= cfg.noise_wrist_scale       # wrist joints swing the TCP less
            if cfg.noise_persist:
                w = _smooth_step((t - (p - dt_frames)) / (2.0 * dt_frames))
            else:
                w = np.clip(1.0 - np.abs(t - p) / dt_frames, 0.0, None)
            eps += w[:, None] * delta[None]
            deltas.append(delta.tolist())
        s[f"{arm}_q"] = q + eps
        s[f"{arm}_dq"] = s[f"{arm}_dq"] + np.gradient(eps, axis=0) / dt
        info["points"][arm] = {"steps": pts, "deltas": deltas}
    return s, info


def gripper_flip(stream, rng, cfg, arm, frame_stride):
    s = stream.copy()
    n = s.n
    lo = int(cfg.flip_lo_frac * n); hi = max(lo + 1, int(cfg.flip_hi_frac * n))
    q = int(rng.integers(lo, hi))
    g = s[f"{arm}_g"]
    g[q:] = 1.0 - np.round(g[q:])          # open <-> closed
    s[f"{arm}_geps"][q:] = np.maximum(s[f"{arm}_geps"][q:], 0.1)
    return s, {"arm": arm, "flip_step": q, "flip_frame": q // frame_stride}


def freeze_arm(stream, p, arm):
    s = stream.copy()
    s[f"{arm}_q"][p:] = s[f"{arm}_q"][p]
    s[f"{arm}_dq"][p:] = 0.0
    s[f"{arm}_g"][p:] = s[f"{arm}_g"][p]
    return s, {"arm": arm, "freeze_step": int(p)}


# ----------------------------------------------------------------------------------------
# live scripted suffix
# ----------------------------------------------------------------------------------------
def scripted_suffix(rollout, rng, cfg, arms, escalation):
    """Plan + execute 1-2 scripted EE segments for `arms` from the current state. Both arms
    move simultaneously when both are chosen (each planned, retimed to a common length)."""
    task = rollout.task
    robot = task.robot
    dt = task.scene.get_timestep()
    n_seg = 2 if rng.random() < cfg.diverge_two_segment_prob else 1
    log = []
    for si in range(n_seg):
        base = current_targets(robot)
        plans = {}
        for arm in arms:
            rot = rng.random() >= cfg.diverge_trans_prob
            dof = rng.choice(ROT_DOFS if rot else TRANS_DOFS)
            if rot:
                mag = np.deg2rad(rng.uniform(*cfg.diverge_rot_deg)) * escalation
                speed = np.deg2rad(cfg.diverge_rot_speed_dps)
            else:
                mag = rng.uniform(*cfg.diverge_trans_m) * escalation
                speed = cfg.diverge_trans_speed_mps
            sign = 1.0 if rng.random() < 0.5 else -1.0
            start7 = ee_pose(robot, arm)
            refused = []
            for shrink in (1.0, 0.6, 0.35):
                target = displaced_pose(start7, dof, sign * mag * shrink)
                # straight-line (screw / short-hop chain) first, then RoboTwin's joint-space
                # planner -- whose path may swing through another IK branch, hence the check
                for method in ("screw", "curobo"):
                    path = plan_to(robot, arm, target, method=method)
                    if path is None:
                        continue
                    travel = joint_travel(path)
                    if travel > cfg.diverge_max_joint_travel_rad:
                        refused.append(f"{method}@{shrink}: joint travel {travel:.2f} rad")
                        continue
                    n_cart = mag * shrink / speed / dt
                    n_joint = joint_arc_length(path) / cfg.diverge_max_joint_speed_rps / dt
                    n = max(10, int(math.ceil(max(n_cart, n_joint))))
                    plans[arm] = (path, n)
                    log.append({"segment": si, "arm": arm, "dof": str(dof),
                                "amount": float(sign * mag * shrink), "planner": method,
                                "joint_travel_rad": float(travel), "steps": int(n)})
                    break
                if arm in plans:
                    break
            if arm not in plans:
                # an arm the mode chose that cannot move would leave a counterfactual whose
                # only change is the OTHER arm holding still -- not what the label says
                raise RuntimeError(f"scripted_diverge: no admissible plan for {arm} {dof} "
                                   f"(segment {si}; {'; '.join(refused) or 'planning failed'})")
        n = max(v[1] for v in plans.values())
        seg = None
        for arm, (path, _) in plans.items():
            q, dq = retime(path, n, dt)
            part = arm_segment(robot, arm, q, dq, base=base)
            if seg is None:
                seg = part
            else:
                seg[f"{arm}_q"], seg[f"{arm}_dq"] = part[f"{arm}_q"], part[f"{arm}_dq"]
        rollout.run(seg)
        if si + 1 < n_seg:
            rollout.run(hold(robot, cfg.diverge_pause_steps))
    if cfg.diverge_tail_steps > 0:
        rollout.run(hold(robot, cfg.diverge_tail_steps))
    return log


# ----------------------------------------------------------------------------------------
# divergence measurement
# ----------------------------------------------------------------------------------------
def joint_travel(path):
    """Largest total travel of any single joint along a joint path [n, d] (rad)."""
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.abs(np.diff(p, axis=0)).sum(axis=0).max())


def joint_arc_length(path):
    """Joint-space arc length of a path [n, d]; `retime` spreads it uniformly over time, so
    arc length / duration bounds every joint's speed."""
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def quat_angle_deg(qa, qb):
    """Rotation angle between unit quaternions (any consistent component order), [...]."""
    d = np.abs(np.sum(np.asarray(qa) * np.asarray(qb), axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0)))


def tcp_speed(tcp, fps):
    """Per-arm peak TCP speed (m/s) and largest single-frame TCP jump (m) of a [T, 2, 3]
    track sampled at `fps`."""
    t = np.asarray(tcp, dtype=np.float64)
    if len(t) < 2:
        return {a: {"peak_mps": 0.0, "max_jump_m": 0.0} for a in ("left", "right")}
    step = np.linalg.norm(np.diff(t, axis=0), axis=2)                     # [T-1, 2]
    return {a: {"peak_mps": float(step[:, j].max() * fps), "max_jump_m": float(step[:, j].max())}
            for j, a in enumerate(("left", "right"))}


def divergence(expert_tcp, cf_tcp, expert_g, cf_g, expert_quat=None, cf_quat=None):
    """expert_tcp/cf_tcp [T, 2, 3]; expert_g/cf_g [T, 2] gripper commands; optional
    expert_quat/cf_quat [T, 2, 4] TCP orientations (same component order in both)."""
    n = min(len(expert_tcp), len(cf_tcp))
    if n == 0:
        return {"n_compared": 0}
    d = np.linalg.norm(np.asarray(expert_tcp[:n]) - np.asarray(cf_tcp[:n]), axis=2)   # [n, 2]
    gm = np.abs(np.round(np.asarray(expert_g[:n])) - np.round(np.asarray(cf_g[:n]))) > 0.5
    out = {"n_compared": int(n), "len_expert": int(len(expert_tcp)), "len_cf": int(len(cf_tcp)),
           "max_tcp_m": float(d.max()), "mean_tcp_m": float(d.mean()),
           "final_tcp_m": float(d[-1].max()),
           "gripper_mismatch_frac": float(gm.any(axis=1).mean())}
    r = None
    if expert_quat is not None and cf_quat is not None:
        r = quat_angle_deg(np.asarray(expert_quat[:n]), np.asarray(cf_quat[:n]))      # [n, 2]
        out["max_rot_deg"] = float(r.max())
    for j, arm in enumerate(("left", "right")):
        out[arm] = {"max_tcp_m": float(d[:, j].max()), "mean_tcp_m": float(d[:, j].mean()),
                    "gripper_mismatch_frac": float(gm[:, j].mean())}
        if r is not None:
            out[arm]["max_rot_deg"] = float(r[:, j].max())
    return out
