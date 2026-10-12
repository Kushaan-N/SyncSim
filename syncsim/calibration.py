"""Bimanual visual-calibration sweep (SyncWorld §3.1, robocasa-dev DESIGN.md §1, two arms).

Per arm, every one of its 6 end-effector DoFs is swept OUT -> HOLD -> BACK -> HOLD while the
other arm and both grippers hold still, so each leg shows exactly one control dimension of one
arm moving. Order: arms in random order; within an arm the three translations (shuffled) then
the three rotations (shuffled), each with a random sign -- the ordering/sign randomisation is
the paper's and robocasa's. The magnitude is drawn per leg; the SPEED is fixed per DoF group,
so distance and speed are independent levers, as in robocasa (there via OSC magnitude x steps,
here via planned-then-retimed paths).

Grippers stay open (RoboTwin opens both during scene init) and are never actuated, so the
clip never contains a gripper event.

Unlike robocasa, `move_range.json` records the exact FRAME span of every leg, so the SyncWorld
segment extractor does not have to rediscover the "strongest signed run" from pose deltas.
"""

import numpy as np

from .motion import (ROT_DOFS, TRANS_DOFS, arm_segment, displaced_pose, ee_pose, hold,
                     plan_to, retime)


def _measured(a7, b7, dof):
    """What the leg actually did: translation vector (m) and rotation angle (deg) a -> b."""
    import transforms3d as t3d
    dp = np.asarray(b7[:3]) - np.asarray(a7[:3])
    Ra, Rb = t3d.quaternions.quat2mat(a7[3:7]), t3d.quaternions.quat2mat(b7[3:7])
    ang = np.degrees(np.arccos(np.clip((np.trace(Rb @ Ra.T) - 1) / 2, -1, 1)))
    return {"dpos_m": dp.tolist(), "dist_m": float(np.linalg.norm(dp)), "rot_deg": float(ang)}


def _tracks(meas, dof, amount, cfg):
    """Did the leg move ~only the commanded DoF by ~the commanded amount?"""
    tol = cfg.calib_track_tol
    if dof in TRANS_DOFS:
        axis = "xyz".index(dof)
        d = np.asarray(meas["dpos_m"])
        along = d[axis]
        off = np.linalg.norm(np.delete(d, axis))
        if abs(along - amount) > tol * abs(amount):
            return False, f"moved {along:+.3f} m along {dof}, commanded {amount:+.3f}"
        if off > 0.01 + tol * abs(amount):
            return False, f"off-axis translation {off:.3f} m"
        if meas["rot_deg"] > cfg.calib_max_spurious_rot_deg:
            return False, f"spurious rotation {meas['rot_deg']:.1f} deg"
    else:
        want = np.degrees(abs(amount))
        if abs(meas["rot_deg"] - want) > tol * want:
            return False, f"rotated {meas['rot_deg']:.1f} deg, commanded {want:.1f}"
        if meas["dist_m"] > cfg.calib_max_spurious_trans_m:
            return False, f"spurious translation {meas['dist_m']:.3f} m"
    return True, ""


def record_calibration(rollout, rng, cfg):
    task = rollout.task
    robot = task.robot
    dt = task.scene.get_timestep()
    arms = ["left", "right"]
    if cfg.calib_shuffle_arms:
        rng.shuffle(arms)
    legs = []
    if cfg.calib_warmup_steps > 0:
        # let the grippers finish opening / the arms settle; captured, but marked
        rollout.mark("warmup_start")
        rollout.run(hold(robot, cfg.calib_warmup_steps))
    for arm in arms:
        trans = list(TRANS_DOFS); rot = list(ROT_DOFS)
        rng.shuffle(trans); rng.shuffle(rot)
        for dof in trans + rot:
            is_rot = dof in ROT_DOFS
            sign = 1.0 if rng.random() < 0.5 else -1.0
            if is_rot:
                lo, hi = cfg.calib_rot_deg
                mag = np.deg2rad(rng.uniform(lo, hi))
                speed = np.deg2rad(cfg.calib_rot_speed_dps)
            else:
                lo, hi = cfg.calib_trans_m
                mag = rng.uniform(lo, hi)
                speed = cfg.calib_trans_speed_mps
            start7 = ee_pose(robot, arm)
            leg = {"arm": arm, "dof": dof, "sign": "+" if sign > 0 else "-",
                   "commanded": float(sign * mag), "units": "rad" if is_rot else "m"}
            done = False
            # full magnitude with either sign before shrinking (e.g. z- from a home pose that
            # already sits near the table becomes z+ rather than a 2 cm z-)
            tries = [(sg, f) for f in (1.0, 0.6, 0.35) for sg in (sign, -sign)]
            attempts = []
            for sg, shrink in tries:
                amount = sg * mag * shrink
                path = plan_to(robot, arm, displaced_pose(start7, dof, amount))
                if path is None:
                    continue
                n = max(10, int(np.ceil(abs(amount) / speed / dt)))
                q, dq = retime(path, n, dt)
                leg["sign"] = "+" if sg > 0 else "-"
                leg["out_start_frame"] = rollout.frame
                rollout.mark(f"{arm}_{dof}{leg['sign']}_out")
                rollout.run(arm_segment(robot, arm, q, dq))
                leg["out_end_frame"] = rollout.frame
                out7 = ee_pose(robot, arm)
                rollout.run(hold(robot, cfg.calib_pause_steps))
                # back = the same straight line in reverse, so the leg ends where it began
                leg["back_start_frame"] = rollout.frame
                rollout.mark(f"{arm}_{dof}{leg['sign']}_back")
                rollout.run(arm_segment(robot, arm, q[::-1].copy(), -dq[::-1].copy()))
                leg["back_end_frame"] = rollout.frame
                rollout.run(hold(robot, cfg.calib_pause_steps))
                end7 = ee_pose(robot, arm)
                meas = _measured(start7, out7, dof)
                ok, why = _tracks(meas, dof, amount, cfg)
                attempts.append({"amount": float(amount), "tracking_ok": ok, "why": why,
                                 "out_frames": [leg["out_start_frame"], leg["out_end_frame"]],
                                 "measured": meas})
                if not ok:
                    # the arm did something other than the commanded single-DoF motion
                    # (joint limit / contact): it is back at start7 -- try the next option.
                    # The rejected attempt stays in the video; its frames are listed in
                    # `attempts` and excluded from the leg's out/back spans.
                    continue
                leg.update(realised=float(amount), shrink=float(shrink), steps=int(n),
                           sign_flipped=bool(sg != sign), measured=meas,
                           return_error_m=float(np.linalg.norm(end7[:3] - start7[:3])))
                done = True
                break
            leg["attempts"] = attempts
            if not done:
                leg["skipped"] = "unreachable"
            legs.append(leg)
    return {
        "legs": legs,
        "arm_order": arms,
        "movement_order": [f"{l['arm']}:{l['dof']}{l['sign']}" for l in legs if "skipped" not in l],
        "skipped": [f"{l['arm']}:{l['dof']}" for l in legs if "skipped" in l],
        "trans_range_m": list(cfg.calib_trans_m), "rot_range_deg": list(cfg.calib_rot_deg),
        "trans_speed_mps": cfg.calib_trans_speed_mps, "rot_speed_dps": cfg.calib_rot_speed_dps,
        "pause_steps": cfg.calib_pause_steps, "warmup_steps": cfg.calib_warmup_steps,
        "marks": rollout.marks,
    }
