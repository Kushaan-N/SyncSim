#!/usr/bin/env python
"""SyncSim generator: SyncWorld-style training data from RoboTwin 2.0 bimanual tasks.

For one task and a range of seeds, writes

    <output_dir>/<task>/episode_<seed>/camera_poses_<m>/
        views.json                 the camera rig (N 3rd-person views + 2 wrist mounts), gate log
        base_init.json             seed, embodiment, task config, scene info, provenance
        expert_stream.npz          the expert's dense control stream (replayable)
        calibration/               bimanual 6-DoF-per-arm sweep   (+ move_range.json)
        expert/                    the scripted expert
        counterfactual_<k:03d>/    one counterfactual per index (mode in metadata.json)
        _DONE                      written last; resume skips units that have it

See syncsim/{rt_env,cameras,calibration,counterfactual,recorder}.py for the design of each part
and DESIGN.md for the robocasa -> RoboTwin mapping.

Run with cwd anywhere; ROBOTWIN_ROOT (default /home/yuncongyang_umass/kn/RoboTwin) must hold an
installed RoboTwin with assets.
"""

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


@dataclass
class Config:
    # ---- what to generate -------------------------------------------------------------
    task: str = "handover_block"
    task_config: str = "demo_randomized"     # RoboTwin env_cfg/task_config/<name>.yml
    embodiment: str = ""                     # "" = task config's; "aloha-agilex" or "piper,franka-panda,0.6"
    output_dir: str = "/scratch/yuncongyang_umass/syncsim/data/set_0"
    seed_start: int = 0
    num_seeds: int = 50                      # seeds tried (failed seeds are skipped, not retried)
    max_episodes: int = 0                    # stop after this many kept episodes (0 = no cap)
    num_camera_units: int = 1                # independent camera rigs per episode
    overwrite: bool = False
    camera_seed_salt: int = 0                # different salt -> different viewpoints for same seeds
    # ---- rendering --------------------------------------------------------------------
    image_size: int = 512
    frame_stride: int = 12                   # physics steps per video frame (250 Hz / 12 = 20.8 fps)
    shader: str = "rt"                       # rt (RoboTwin's ray tracer) | default (rasteriser)
    rt_spp: int = 32                         # ray-tracing samples per pixel (RoboTwin uses 32)
    rt_denoiser: str = "optix"               # RoboTwin's "oidn" fails on Blackwell GPUs ("invalid handle")
    video_crf: int = 18
    record_wrist: bool = True
    # ---- RoboTwin domain randomisation overrides (None = keep the task config's) -------
    random_background: str = ""              # "", "true", "false"
    cluttered_table: str = ""
    random_light: str = ""
    random_table_height: float = -1.0        # <0 = keep
    # ---- 3rd-person camera rig ----------------------------------------------------------
    num_views: int = 2                       # simultaneous 3rd-person cameras
    fovy_deg: float = 45.0
    azimuth_range_deg: float = 110.0         # +- around the head camera's direction from the target
    elev_min_deg: float = 15.0
    elev_max_deg: float = 65.0
    radius_min: float = 0.75                 # true 3D distance to the aim target (m)
    radius_max: float = 1.35
    aim_jitter_m: float = 0.04
    roll_jitter_deg: float = 4.0
    min_separation_deg: float = 45.0         # min baseline angle between simultaneous views
    num_candidates: int = 96
    check_frames: int = 4                    # gripper visibility checked at this many frames
    gate_arms: str = "both"                  # both | active | any
    min_gripper_px_frac: float = 0.0015      # of the gate image (~400 px at 512^2)
    gate_size: int = 160
    # ---- wrist cameras --------------------------------------------------------------------
    wrist_fovy_deg: float = 70.0
    wrist_candidates: int = 12
    wrist_pos_jitter: float = 0.01
    wrist_rot_jitter_deg: float = 5.0
    wrist_min_frac: float = 0.03
    wrist_max_frac: float = 0.35
    # ---- calibration ----------------------------------------------------------------------
    record_calibration: bool = True
    calib_trans_m: tuple = (0.05, 0.08)
    calib_rot_deg: tuple = (12.0, 20.0)
    calib_trans_speed_mps: float = 0.08
    calib_rot_speed_dps: float = 40.0
    calib_pause_steps: int = 60              # physics steps (0.24 s) held between legs
    calib_warmup_steps: int = 0
    calib_shuffle_arms: bool = True
    calib_track_tol: float = 0.25            # leg rejected if its commanded DoF misses by >25 %
    calib_max_spurious_rot_deg: float = 5.0  # ... or a translation leg rotates more than this
    calib_max_spurious_trans_m: float = 0.03 # ... or a rotation leg translates more than this
    calib_track_tol: float = 0.25            # leg rejected if its commanded DoF misses by >25 %
    calib_max_spurious_rot_deg: float = 5.0  # ... or a translation leg rotates more than this
    calib_max_spurious_trans_m: float = 0.03 # ... or a rotation leg translates more than this
    # ---- counterfactuals --------------------------------------------------------------------
    num_counterfactuals: int = 4
    cf_mode_probs: str = "noisy_replay:0.35,scripted_diverge:0.3,freeze_arm:0.2,gripper_flip:0.15"
    cf_arm_probs: str = "left:0.35,right:0.35,both:0.3"
    cf_min_tcp_deviation_m: float = 0.05
    cf_min_gripper_mismatch_frac: float = 0.05
    cf_max_attempts: int = 3
    cf_escalation: float = 1.6
    cf_min_prefix_frames: int = 4
    cf_min_suffix_frames: int = 24
    freeze_min_motion_rad: float = 0.05      # freeze_arm only freezes an arm that still moves this much
    noise_sigma_rad: float = 0.12
    noise_wrist_scale: float = 0.6
    noise_interval_frames: int = 10          # ramp half-width
    noise_points_per_100_frames: float = 2.0
    noise_persist: bool = True
    flip_lo_frac: float = 0.25
    flip_hi_frac: float = 0.85
    diverge_two_segment_prob: float = 0.5
    diverge_trans_prob: float = 0.6
    diverge_trans_m: tuple = (0.06, 0.14)
    diverge_rot_deg: tuple = (15.0, 35.0)
    diverge_trans_speed_mps: float = 0.12
    diverge_rot_speed_dps: float = 50.0
    diverge_pause_steps: int = 40
    diverge_tail_steps: int = 120


def _parse_probs(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        out[k.strip()] = float(v)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        d = f.default
        flag = f"--{f.name}"
        if isinstance(d, bool):
            p.add_argument(flag, type=lambda x: str(x).lower() in ("1", "true", "yes", "on"), default=d)
        elif isinstance(d, tuple):
            p.add_argument(flag, type=lambda x: tuple(float(v) for v in x.split(",")), default=d)
        else:
            p.add_argument(flag, type=type(d), default=d)
    return p


# ----------------------------------------------------------------------------------------
def _domain_overrides(cfg):
    dr = {}
    for k in ("random_background", "cluttered_table", "random_light"):
        v = getattr(cfg, k)
        if v != "":
            dr[k] = v.lower() in ("1", "true", "yes", "on")
    if cfg.random_table_height >= 0:
        dr["random_table_height"] = cfg.random_table_height
    dr["random_head_camera_dis"] = 0          # the head camera is not used; ours are sampled
    return {"domain_randomization": dr}


def _embodiment(cfg):
    if not cfg.embodiment:
        return None
    parts = [x.strip() for x in cfg.embodiment.split(",")]
    if len(parts) == 3:
        return [parts[0], parts[1], float(parts[2])]
    return parts


def _unit_rng(cfg, seed, m, tag):
    s = (seed * 1_000_003 + m * 97 + cfg.camera_seed_salt * 2_654_435_761
         + zlib.crc32(f"{cfg.task}:{tag}".encode())) & 0x7FFFFFFF
    return np.random.default_rng(s)


def _rig_json(rig, cfg):
    return {
        "num_views": len(rig["views"]), "fovy_deg": cfg.fovy_deg,
        "wrist_fovy_deg": cfg.wrist_fovy_deg, "image_size": cfg.image_size,
        "aim_target": rig["target"], "check_frames": rig["check_frames"],
        "gate_arms": rig["gate_arms"], "need_arms": rig["need_arms"],
        "arms_move_in_expert": rig["arms_move"], "min_gripper_px_at_gate": rig["min_gripper_px"],
        "gate_size": rig["gate_size"], "candidates": rig["candidates"],
        "candidates_passed": rig["alive"],
        "min_separation_deg": cfg.min_separation_deg,
        "views": [{
            "index": v_i, "dir": "agentview_rgb" if v_i == 0 else f"agentview_rgb_{v_i}",
            "cam2world_sapien": v["mat"], "position": v["pos"], "aim": v["aim"],
            "azimuth_deg": v["azimuth_deg"], "elevation_deg": v["elevation_deg"],
            "radius_m": v["radius_m"], "roll_deg": v["roll_deg"],
            "gate_gripper_px": v["gripper_px"],
        } for v_i, v in enumerate(rig["views"])],
        "wrist": {a: {"dir": f"wrist_{a}_rgb", "link": f"{a}_camera",
                      "rel_mat": w["rel_mat"], "stock_mount": w["stock"],
                      "gate_gripper_frac": w["grip_frac"]} for a, w in rig["wrist"].items()},
        "camera_convention": "SAPIEN cam2world: x forward, y left, z up",
    }


def _rig_for_recorder(rig, cfg):
    return {"views": rig["views"], "wrist": rig["wrist"], "fovy_deg": cfg.fovy_deg,
            "wrist_fovy_deg": cfg.wrist_fovy_deg}


# ----------------------------------------------------------------------------------------
class Generator:

    def __init__(self, cfg):
        from . import rt_env
        self.cfg = cfg
        self.root = rt_env.enter_robotwin()
        self.rt = rt_env
        self.args = rt_env.build_task_args(cfg.task, cfg.task_config, _embodiment(cfg),
                                           overrides=_domain_overrides(cfg))
        self.task = rt_env.make_task(cfg.task)
        self.task_dir = os.path.join(cfg.output_dir, cfg.task)
        os.makedirs(self.task_dir, exist_ok=True)
        self.dropped_path = os.path.join(self.task_dir, "_dropped.json")
        self.dropped = {}
        if os.path.exists(self.dropped_path) and not cfg.overwrite:
            with open(self.dropped_path) as f:
                self.dropped = {int(k): v for k, v in json.load(f).items()}

    # ------------------------------------------------------------------------------------
    def _drop(self, seed, reason):
        self.dropped[int(seed)] = reason
        tmp = self.dropped_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({str(k): v for k, v in sorted(self.dropped.items())}, f, indent=1)
        os.replace(tmp, self.dropped_path)

    def _reset(self, seed):
        import sapien
        self.rt.close_task(self.task)
        self.rt.setup_task(self.task, self.args, seed, need_plan=False)
        if self.cfg.shader == "rt":
            # RoboTwin's setup_scene sets these globally on every setup; override after it
            sapien.render.set_ray_tracing_samples_per_pixel(int(self.cfg.rt_spp))
            sapien.render.set_ray_tracing_denoiser(self.cfg.rt_denoiser)

    def _recorder(self, rig):
        from .recorder import RigRecorder
        return RigRecorder(self.task, _rig_for_recorder(rig, self.cfg), self.cfg.image_size,
                           self.cfg.image_size, shader=self.cfg.shader,
                           record_wrist=self.cfg.record_wrist)

    @property
    def fps(self):
        return 1.0 / (self.task.scene.get_timestep() * self.cfg.frame_stride)

    # ------------------------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        kept = 0
        t_start = time.time()
        for seed in range(cfg.seed_start, cfg.seed_start + cfg.num_seeds):
            if cfg.max_episodes and kept >= cfg.max_episodes:
                break
            ep_dir = os.path.join(self.task_dir, f"episode_{seed}")
            if not cfg.overwrite and self._episode_complete(ep_dir):
                kept += 1
                print(f"[skip] {cfg.task} seed {seed}: complete")
                continue
            if seed in self.dropped and not cfg.overwrite:
                print(f"[skip] {cfg.task} seed {seed}: dropped earlier ({self.dropped[seed]})")
                continue
            t0 = time.time()
            try:
                ok = self.episode(seed, ep_dir)
            except Exception as e:
                traceback.print_exc()
                ok = False
                self._drop(seed, f"exception: {type(e).__name__}: {e}"[:300])
                shutil.rmtree(ep_dir, ignore_errors=True)
            finally:
                self.rt.close_task(self.task, clear_cache=True)
            kept += int(bool(ok))
            print(f"[{'done' if ok else 'drop'}] {cfg.task} seed {seed} in {time.time() - t0:.0f}s "
                  f"(kept {kept}, {time.time() - t_start:.0f}s total)", flush=True)
        print(f"[syncsim] {cfg.task}: kept {kept} episode(s) from seeds "
              f"[{cfg.seed_start}, {cfg.seed_start + cfg.num_seeds})")

    def _episode_complete(self, ep_dir):
        return all(os.path.exists(os.path.join(ep_dir, f"camera_poses_{m}", "_DONE"))
                   for m in range(self.cfg.num_camera_units))

    # ------------------------------------------------------------------------------------
    def episode(self, seed, ep_dir):
        cfg, rt = self.cfg, self.rt
        from envs.utils.create_actor import UnStableError
        try:
            ok, reason, stream, paths, info = rt.run_expert(self.task, self.args, seed)
        except UnStableError as e:
            self._drop(seed, f"unstable_scene: {e}"[:300])
            return False
        if not ok:
            self._drop(seed, reason)
            return False
        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
        kept_units, reasons = 0, []
        for m in range(cfg.num_camera_units):
            unit = os.path.join(ep_dir, f"camera_poses_{m}")
            ok, why = self.unit(seed, m, unit, stream, info)
            kept_units += int(ok)
            if not ok:
                reasons.append(why)
        if kept_units == 0:
            shutil.rmtree(ep_dir, ignore_errors=True)
            self._drop(seed, reasons[0] if reasons else "no_unit_kept")
            return False
        return True

    def unit(self, seed, m, unit, stream, info):
        cfg = self.cfg
        from .cameras import gate_and_select
        from .calibration import record_calibration
        from .recorder import write_json
        from .rollout import Rollout

        rng = _unit_rng(cfg, seed, m, "rig")
        reset = lambda: self._reset(seed)
        rig = gate_and_select(self.task, reset, stream, cfg, rng, self.args, cfg.num_views,
                              cfg.frame_stride)
        if not rig["ok"]:
            print(f"    [gate] seed {seed} unit {m}: {rig['reason']}")
            return False, f"camera_gate: {rig['reason']}"
        os.makedirs(unit, exist_ok=True)
        write_json(os.path.join(unit, "views.json"), _rig_json(rig, cfg))
        stream.save(os.path.join(unit, "expert_stream.npz"))

        # ---- expert -------------------------------------------------------------------
        reset()
        rec = self._recorder(rig)
        ro = Rollout(self.task, rec, cfg.frame_stride, poll_success=True)
        ro.run(stream)
        succ = ro.success()
        expert_success = succ["final"]
        if not expert_success:
            print(f"    [expert] seed {seed}: stream replay no longer succeeds -- unit dropped")
            shutil.rmtree(unit, ignore_errors=True)
            return False, "expert_replay_not_successful"
        exp_dir = os.path.join(unit, "expert")
        n_frames = rec.save(exp_dir, self.fps, cfg.frame_stride, crf=cfg.video_crf)
        expert_tcp, expert_g = rec.tcp_track(), rec.gripper_track()
        write_json(os.path.join(exp_dir, "metadata.json"), {
            "episode_type": "expert", "seed": seed, "task": cfg.task, "success": expert_success,
            "first_success_frame": succ["first_frame"],
            "frames": n_frames, "physics_steps": int(stream.n), "frame_stride": cfg.frame_stride,
            "fps": self.fps, "action_source": "robotwin_scripted_expert/stream_replay",
            "task_info": info})
        self._base_init(unit, seed, m, rig)

        # ---- calibration ----------------------------------------------------------------
        if cfg.record_calibration:
            # probe (unrendered, may retry legs) -> film only the accepted legs
            from .calibration import render_clean
            reset()
            probe = Rollout(self.task, _TrackOnly(self.task), cfg.frame_stride, render=False)
            mr = record_calibration(probe, _unit_rng(cfg, seed, m, "calibration"), cfg)
            reset()
            rec = self._recorder(rig)
            ro = Rollout(self.task, rec, cfg.frame_stride)
            render_clean(ro, mr)
            cal_dir = os.path.join(unit, "calibration")
            mr["frames"] = rec.save(cal_dir, self.fps, cfg.frame_stride, crf=cfg.video_crf)
            mr["filmed"] = _filmed_leg_motion(rec, mr)
            ro.stream().save(os.path.join(cal_dir, "stream.npz"))
            write_json(os.path.join(cal_dir, "move_range.json"), mr)

        # ---- counterfactuals --------------------------------------------------------------
        for k in range(cfg.num_counterfactuals):
            self.counterfactual(seed, m, k, unit, rig, stream, expert_tcp, expert_g)

        with open(os.path.join(unit, "_DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return True, "ok"

    def _base_init(self, unit, seed, m, rig):
        from .recorder import ACTION_LAYOUT, write_json
        t = self.task
        a = self.args
        write_json(os.path.join(unit, "base_init.json"), {
            "seed": seed, "camera_unit": m, "task": self.cfg.task,
            "task_config": self.cfg.task_config, "embodiment": a["embodiment"],
            "embodiment_name": a["embodiment_name"], "dual_arm_embodied": a["dual_arm_embodied"],
            "domain_randomization": a["domain_randomization"],
            "scene_info": {"texture_info": getattr(t, "info", {}).get("texture_info"),
                           "cluttered_table_info": getattr(t, "info", {}).get("cluttered_table_info"),
                           "table_z_bias": float(getattr(t, "table_z_bias", 0.0))},
            "physics_timestep": float(t.scene.get_timestep()),
            "frame_stride": self.cfg.frame_stride, "fps": self.fps,
            "action_layout": ACTION_LAYOUT,
            "left_arm_dof": len(t.robot.left_arm_joints), "right_arm_dof": len(t.robot.right_arm_joints),
            "config": dataclasses.asdict(self.cfg),
            "robotwin_root": self.root,
        })

    # ------------------------------------------------------------------------------------
    def _cf_attempt(self, seed, mode, arm_sel, esc, arng, stream, expert_tcp, expert_g, attempt):
        """One unrendered probe of a counterfactual -> (executed stream, meta, divergence, ok)."""
        cfg = self.cfg
        from . import counterfactual as CF
        from .rollout import Rollout
        self._reset(seed)
        dt = self.task.scene.get_timestep()
        ro = Rollout(self.task, _TrackOnly(self.task), cfg.frame_stride, render=False,
                     poll_success=True)
        meta = {"counterfactual_mode": mode, "arms": arm_sel, "attempt": attempt,
                "escalation": esc}
        if mode == "noisy_replay":
            s, info = CF.noisy_replay(stream, arng, cfg, CF.arms_of(arm_sel), esc,
                                      cfg.frame_stride, dt)
            ro.run(s)
            meta["noise"] = info
        elif mode == "gripper_flip":
            s, info = CF.gripper_flip(stream, arng, cfg, arm_sel, cfg.frame_stride)
            ro.run(s)
            meta["flip"] = info
        else:
            p = CF.sample_breakpoint(arng, stream.n, cfg.frame_stride,
                                     cfg.cf_min_prefix_frames, cfg.cf_min_suffix_frames)
            meta["breakpoint_step"] = int(p)
            meta["breakpoint_frame"] = int(p // cfg.frame_stride)
            if mode == "freeze_arm":
                # freezing an arm that never moves again is a no-op (click_bell: the bell is
                # on one side, the other arm stays home) -- freeze one that still moves
                moving = [a for a in ("left", "right")
                          if np.ptp(stream[f"{a}_q"][p:], axis=0).max() > cfg.freeze_min_motion_rad]
                if arm_sel not in moving and moving:
                    arm_sel = moving[0] if len(moving) == 1 else str(arng.choice(moving))
                    meta["arms"] = arm_sel
                meta["freeze_arm_moves_after_breakpoint"] = arm_sel in moving
                s, info = CF.freeze_arm(stream, p, arm_sel)
                ro.run(s)
                meta["freeze"] = info
            else:
                ro.run(stream.slice(0, p))
                meta["scripted"] = CF.scripted_suffix(ro, arng, cfg, CF.arms_of(arm_sel), esc)
        executed = ro.stream()
        div = CF.divergence(expert_tcp, ro.rec.tcp_track(), expert_g, ro.rec.gripper_track())
        ok = (div.get("max_tcp_m", 0) >= cfg.cf_min_tcp_deviation_m
              or div.get("gripper_mismatch_frac", 0) >= cfg.cf_min_gripper_mismatch_frac)
        return executed, meta, div, ok

    def counterfactual(self, seed, m, k, unit, rig, stream, expert_tcp, expert_g):
        cfg = self.cfg
        from . import counterfactual as CF
        from .recorder import write_json
        from .rollout import Rollout

        rng = _unit_rng(cfg, seed, m, f"cf{k}")
        mode = CF.pick(rng, _parse_probs(cfg.cf_mode_probs))
        arm_sel = CF.pick(rng, _parse_probs(cfg.cf_arm_probs))
        if mode in ("freeze_arm", "gripper_flip") and arm_sel == "both":
            arm_sel = "left" if rng.random() < 0.5 else "right"
        attempts = []
        final = None
        n_att = max(1, cfg.cf_max_attempts)
        for attempt in range(n_att + 1):
            if attempt == n_att:
                if final is not None:
                    break
                # every attempt raised (e.g. a planner error in a scripted suffix): fall back to
                # a perturbation, which needs no planning, rather than losing the whole episode
                mode, arm_sel = "noisy_replay", ("both" if arm_sel == "both" else arm_sel)
            esc = cfg.cf_escalation ** min(attempt, n_att - 1)
            arng = np.random.default_rng(rng.integers(1 << 31) + attempt)
            try:
                res = self._cf_attempt(seed, mode, arm_sel, esc, arng, stream, expert_tcp,
                                       expert_g, attempt)
            except Exception as e:
                traceback.print_exc()
                attempts.append({"attempt": attempt, "mode": mode,
                                 "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            executed, meta, div, ok = res
            arm_sel = meta["arms"]
            attempts.append({"attempt": attempt, "mode": mode, "escalation": esc,
                             "accepted": bool(ok), "max_tcp_m": div.get("max_tcp_m"),
                             "gripper_mismatch_frac": div.get("gripper_mismatch_frac")})
            final = res
            if ok:
                break
        if final is None:
            raise RuntimeError(f"counterfactual {k}: every attempt failed: {attempts}")
        executed, meta, div, ok = final
        # ---- render the accepted (or last) executed stream exactly ------------------------
        self._reset(seed)
        rec = self._recorder(rig)
        ro = Rollout(self.task, rec, cfg.frame_stride, poll_success=True)
        ro.run(executed)
        succ = ro.success()
        success = succ["final"]
        cf_dir = os.path.join(unit, f"counterfactual_{k:03d}")
        n_frames = rec.save(cf_dir, self.fps, cfg.frame_stride, crf=cfg.video_crf)
        executed.save(os.path.join(cf_dir, "stream.npz"))
        meta.update({"episode_type": "counterfactual", "k": k, "seed": seed,
                     "task_success": success, "task_success_ever": succ["ever"],
                     "first_success_frame": succ["first_frame"], "frames": n_frames,
                     "physics_steps": int(executed.n), "fps": self.fps,
                     "divergence": dict(div, threshold_m=cfg.cf_min_tcp_deviation_m,
                                        accepted=bool(ok), attempt_log=attempts)})
        write_json(os.path.join(cf_dir, "metadata.json"), meta)
        print(f"    [cf] seed {seed} unit {m} cf{k}: {mode}/{arm_sel} max_tcp "
              f"{div.get('max_tcp_m', 0):.3f} m gmis {div.get('gripper_mismatch_frac', 0):.2f} "
              f"success={success} attempts={len(attempts)}{'' if ok else ' (BELOW GATE)'}")


def _filmed_leg_motion(rec, mr):
    """Per accepted leg, the TCP motion actually filmed between out_start and out_end
    (position change in m, rotation in deg) -- a check that the clean replay shows each leg."""
    import transforms3d as t3d
    out = []
    for leg in mr["legs"]:
        if "out_start_frame" not in leg:
            continue
        P = np.asarray(rec.log[f"{leg['arm']}_pose"])
        a, b = P[leg["out_start_frame"]], P[min(leg["out_end_frame"], len(P) - 1)]
        Ra, Rb = t3d.quaternions.quat2mat(a[3:7]), t3d.quaternions.quat2mat(b[3:7])
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rb @ Ra.T) - 1) / 2, -1, 1))))
        out.append({"leg": f"{leg['arm']}:{leg['dof']}{leg['sign']}",
                    "dpos_m": (b[:3] - a[:3]).round(4).tolist(), "rot_deg": round(ang, 2)})
    return out


class _TrackOnly:
    """Recorder stand-in for unrendered probe rollouts: logs TCPs + grippers only."""

    def __init__(self, task):
        self.robot = task.robot
        self.tcp, self.g = [], []

    def capture(self, step_idx, render=False):
        r = self.robot
        self.tcp.append([r.get_left_tcp_pose()[:3], r.get_right_tcp_pose()[:3]])
        self.g.append([r.left_gripper_val, r.right_gripper_val])

    def tcp_track(self):
        return np.asarray(self.tcp, dtype=np.float64)

    def gripper_track(self):
        return np.asarray(self.g, dtype=np.float64)


def main(argv=None):
    cfg = Config(**vars(build_parser().parse_args(argv)))
    Generator(cfg).run()


if __name__ == "__main__":
    main()
