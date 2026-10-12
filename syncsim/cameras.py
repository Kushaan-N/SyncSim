"""Third-person camera randomization + wrist-mount jitter for the bimanual RoboTwin scene.

Port of the robocasa generator's camera design (robocasa-dev/datagen/DESIGN.md §2-§3, §5)
to a two-arm tabletop:

* A candidate 3rd-person camera is sampled on a sphere around an AIM TARGET (the centroid of
  BOTH grippers' TCPs over the gate's check frames, so a view that frames the whole bimanual
  motion is preferred) -- azimuth around the embodiment's own head-camera direction,
  an explicit ELEVATION band, a true 3D radius -- and re-aimed at the target with a small
  aim/roll jitter.
* Candidates that are physically impossible are rejected before any rendering: inside the
  back wall, under the table top, under the floor.
* The GATE is occlusion-aware: a small segmentation camera counts each gripper's link pixels
  at N linearly spaced CHECK FRAMES of the expert (all of them must pass). Which grippers must
  be visible is `--gate_arms`: `both` (default; a bimanual world model has to see both hands
  to learn which action dims move which arm), `active` (only arms that move in the expert),
  or `any`.
* Instead of resampling + re-simulating per candidate (robocasa restores mujoco states), ALL
  candidates are evaluated during ONE unrendered replay of the expert stream: at each check
  frame every still-alive candidate is rendered once, so the cost is
  (#candidates x #check frames) small segmentation renders and one physics replay.
* N simultaneous views are then picked greedily in sampling order with a minimum BASELINE
  ANGLE between their rays to the target (robocasa found the baseline angle, not azimuth,
  tracks co-visibility).
* Each wrist camera mount is jittered (+-pos, +-rot) with a TWO-SIDED gate on the fraction of
  the wrist frame its own gripper covers (too little = tipped out of frame, too much = lens
  inside the hand).
"""

import math

import numpy as np
import sapien
import transforms3d as t3d

from .rt_env import replay_stream


# ----------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------
def look_at_mat(cam_pos, target, roll_deg=0.0):
    """SAPIEN camera pose (x forward, y left, z up) looking from cam_pos at target."""
    fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, fwd)
    if np.linalg.norm(left) < 1e-6:          # looking straight down/up
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    if roll_deg:
        r = math.radians(roll_deg)
        left, up = (math.cos(r) * left + math.sin(r) * up, -math.sin(r) * left + math.cos(r) * up)
    m = np.eye(4)
    m[:3, :3] = np.stack([fwd, left, up], axis=1)
    m[:3, 3] = cam_pos
    return m


def mat_to_pose7(m):
    """4x4 -> [x, y, z, qw, qx, qy, qz] (SAPIEN / transforms3d quaternion order)."""
    return np.concatenate([m[:3, 3], t3d.quaternions.mat2quat(m[:3, :3])])


def pose7_to_mat(p):
    m = np.eye(4)
    m[:3, :3] = t3d.quaternions.quat2mat(p[3:7])
    m[:3, 3] = p[:3]
    return m


def sapien_pose_to_mat(pose):
    return pose.to_transformation_matrix()


def baseline_angle_deg(cam_a, cam_b, target):
    a = np.asarray(target) - np.asarray(cam_a)
    b = np.asarray(target) - np.asarray(cam_b)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ----------------------------------------------------------------------------------------
# robot introspection
# ----------------------------------------------------------------------------------------
def _descendant_links(link):
    out, stack = [], [link]
    while stack:
        l = stack.pop()
        out.append(l)
        stack.extend(list(getattr(l, "children", []) or []))
    return out


def gripper_link_ids(robot):
    """{'left': set(per_scene_id), 'right': set(...)} of each gripper's links -- the
    end-effector link and everything below it (palm, fingers, gripper-mounted parts)."""
    out = {}
    for arm in ("left", "right"):
        ee_joint = getattr(robot, f"{arm}_ee")
        root = ee_joint.child_link
        links = _descendant_links(root)
        ent = getattr(robot, f"{arm}_entity")
        fix_names = set(getattr(robot, f"{arm}_fix_gripper_name", []) or [])
        for l in ent.get_links():
            if l.get_name() in fix_names:
                links.append(l)
        for g in getattr(robot, f"{arm}_gripper"):
            links.extend(_descendant_links(g[0].child_link))
        out[arm] = {int(l.entity.per_scene_id) for l in links}
    return out


def tcp_positions(robot):
    l = np.asarray(robot.get_left_tcp_pose()[:3])
    r = np.asarray(robot.get_right_tcp_pose()[:3])
    return l, r


def head_camera_info(args):
    for cam in args["left_embodiment_config"].get("static_camera_list", []):
        if cam.get("name") == "head_camera":
            return cam
    return None


# ----------------------------------------------------------------------------------------
# candidate sampling
# ----------------------------------------------------------------------------------------
TABLE_TOP_Z = 0.74            # RoboTwin create_table_and_wall(table_height=0.74) (+ z bias <= 0)
TABLE_HALF = (0.6, 0.35)      # 1.2 x 0.7 table centred on table_xy_bias
WALL_FRONT_Y = 0.4            # wall box centre y=1, half y=0.6


def physically_ok(p, table_xy=(0.0, 0.0), margin=0.05):
    x, y, z = p
    if y > WALL_FRONT_Y - margin:
        return False
    if z < 0.3:
        return False
    on_table = (abs(x - table_xy[0]) < TABLE_HALF[0] + margin
                and abs(y - table_xy[1]) < TABLE_HALF[1] + margin)
    if on_table and z < TABLE_TOP_Z + 0.08:
        return False
    return True


def sample_candidates(rng, target, ref_dir_xy, n, cfg, table_xy=(0.0, 0.0)):
    """n candidate camera 4x4 poses (+ their sampling params) around `target`.
    ref_dir_xy: horizontal unit vector target->default head camera (azimuth 0)."""
    phi0 = math.atan2(ref_dir_xy[1], ref_dir_xy[0])
    out, tries = [], 0
    while len(out) < n and tries < n * 20:
        tries += 1
        dphi = math.radians(rng.uniform(-cfg.azimuth_range_deg, cfg.azimuth_range_deg))
        elev = math.radians(rng.uniform(cfg.elev_min_deg, cfg.elev_max_deg))
        r = rng.uniform(cfg.radius_min, cfg.radius_max)
        phi = phi0 + dphi
        p = np.asarray(target) + r * np.array([math.cos(elev) * math.cos(phi),
                                               math.cos(elev) * math.sin(phi),
                                               math.sin(elev)])
        if not physically_ok(p, table_xy):
            continue
        aim = np.asarray(target) + rng.uniform(-cfg.aim_jitter_m, cfg.aim_jitter_m, size=3)
        roll = rng.uniform(-cfg.roll_jitter_deg, cfg.roll_jitter_deg)
        out.append({"mat": look_at_mat(p, aim, roll), "pos": p, "aim": aim,
                    "azimuth_deg": math.degrees(dphi), "elevation_deg": math.degrees(elev),
                    "radius_m": r, "roll_deg": roll})
    return out


def sample_wrist_jitters(rng, n, pos_jitter, rot_jitter_deg):
    out = [np.eye(4)]                       # candidate 0 = the stock mount
    for _ in range(n - 1):
        m = np.eye(4)
        e = np.radians(rng.uniform(-rot_jitter_deg, rot_jitter_deg, size=3))
        m[:3, :3] = t3d.euler.euler2mat(*e)
        m[:3, 3] = rng.uniform(-pos_jitter, pos_jitter, size=3)
        # never back along the optical axis (camera x): that pushes the lens into the wrist
        # housing, which the rasterised gate sees through (back faces culled) but the ray
        # tracer renders as a dark/grey wall for the whole episode
        m[0, 3] = abs(m[0, 3])
        out.append(m)
    return out


# ----------------------------------------------------------------------------------------
# gate: ONE unrendered replay, all candidates rendered at the check frames
# ----------------------------------------------------------------------------------------
def check_frame_indices(n_frames, k):
    if n_frames <= 1 or k <= 1:
        return [0]
    return sorted({int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)})


def add_camera(scene, name, width, height, fovy_deg, shader="default", near=0.05, far=100):
    """scene.add_camera with an explicit shader. The shader dir is a global SAPIEN setting
    read at camera creation, so it is set, the camera made, and RoboTwin's 'rt' restored."""
    sapien.render.set_camera_shader_dir(shader)
    try:
        return scene.add_camera(name=name, width=int(width), height=int(height),
                                fovy=np.deg2rad(fovy_deg), near=near, far=far)
    finally:
        sapien.render.set_camera_shader_dir("rt")


class SegCam:
    """A small rasterized segmentation camera that can be moved anywhere."""

    def __init__(self, scene, size, fovy_deg, name="syncsim_gate_cam"):
        self.cam = add_camera(scene, name, size, size, fovy_deg, shader="default")
        self.size = size

    def seg(self, scene, mat):
        self.cam.entity.set_pose(sapien.Pose(mat))
        scene.update_render()
        self.cam.take_picture()
        return self.cam.get_picture("Segmentation")[..., 1].astype(np.int64)

    def gray(self):
        """Grey image of the last picture (call after `seg`)."""
        return np.clip(self.cam.get_picture("Color")[..., :3], 0, 1).mean(-1)


def _blocks(g, k=16):
    h = g.shape[0] // k * k
    return g[:h, :h].reshape(k, h // k, k, h // k).mean((1, 3))


def rt_view_ok(raster_gray, rt_gray, min_std, min_corr):
    """A wrist view must look the same to the ray tracer as to the rasteriser. Inside a mesh
    (lens pushed into the wrist housing) the rasteriser culls the back faces and sees out,
    while the ray tracer renders the inside: a flat dark/grey frame uncorrelated with the
    raster one. Returns (ok, rt_std, corr)."""
    a, b = _blocks(raster_gray).ravel(), _blocks(rt_gray).ravel()
    sd = float(rt_gray.std() * 255)
    if sd < min_std or a.std() < 1e-6 or b.std() < 1e-6:
        return False, sd, 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c >= min_corr, sd, c


def gate_and_select(task, reset_fn, stream, cfg, rng, args, n_views, frame_stride):
    """`reset_fn()` must put `task` back at the seeded initial scene (it is called twice:
    once for the aim-target pass, once for the gate pass).

    Returns dict(ok=True, views=[...], wrist={'left':..., 'right':...}, target, ...) or
    dict(ok=False, reason=...) when no rig passes."""
    reset_fn()
    robot, scene = task.robot, task.scene
    n_frames = stream.n // frame_stride + 1
    check = check_frame_indices(n_frames, cfg.check_frames)
    grip_ids = gripper_link_ids(robot)

    # ---- pass 1: TCP positions at the check frames (aim target) + which arms move ----
    tcps = {}

    def grab_tcp(fi, si):
        if fi in check:
            tcps[fi] = tcp_positions(robot)

    replay_stream(task, stream, frame_stride, on_frame=grab_tcp)
    left_pts = np.array([tcps[f][0] for f in check])
    right_pts = np.array([tcps[f][1] for f in check])
    target = np.concatenate([left_pts, right_pts]).mean(axis=0)
    target[2] = max(target[2], TABLE_TOP_Z + 0.05)
    moves = {"left": float(np.ptp(stream["left_q"], axis=0).max()) > 1e-3,
             "right": float(np.ptp(stream["right_q"], axis=0).max()) > 1e-3}
    if cfg.gate_arms == "both":
        need = ["left", "right"]
    elif cfg.gate_arms == "active":
        need = [a for a in ("left", "right") if moves[a]] or ["left", "right"]
    else:
        need = None   # any

    head = head_camera_info(args)
    if head is not None:
        ref = np.asarray(head["position"], float)[:2] - target[:2]
    else:
        ref = np.array([0.0, -1.0])
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    table_xy = tuple(getattr(task, "table_xy_bias", (0.0, 0.0)))
    cands = sample_candidates(rng, target, ref, cfg.num_candidates, cfg, table_xy)
    wrist_cands = {a: sample_wrist_jitters(rng, cfg.wrist_candidates, cfg.wrist_pos_jitter,
                                           cfg.wrist_rot_jitter_deg) for a in ("left", "right")}

    # ---- pass 2: replay again from the initial scene, render every live candidate ----
    reset_fn()
    segcam = SegCam(task.scene, cfg.gate_size, cfg.fovy_deg)
    wseg = SegCam(task.scene, cfg.gate_size, cfg.wrist_fovy_deg, name="syncsim_gate_wrist")
    wrt = add_camera(task.scene, "syncsim_gate_wrist_rt", cfg.gate_size, cfg.gate_size,
                     cfg.wrist_fovy_deg, shader="rt") if cfg.wrist_rt_check else None
    px_min = cfg.min_gripper_px_frac * cfg.gate_size * cfg.gate_size
    alive = list(range(len(cands)))
    px_log = {i: [] for i in range(len(cands))}
    wfrac = {a: {i: [] for i in range(len(wrist_cands[a]))} for a in ("left", "right")}
    wrt_ok = {a: {i: True for i in range(len(wrist_cands[a]))} for a in ("left", "right")}
    wrt_log = {a: {i: [] for i in range(len(wrist_cands[a]))} for a in ("left", "right")}

    def at_frame(fi, si):
        nonlocal alive
        if fi not in check:
            return
        sc = task.scene
        still = []
        for i in alive:
            s = segcam.seg(sc, cands[i]["mat"])
            counts = {a: int(np.isin(s, list(grip_ids[a])).sum()) for a in ("left", "right")}
            px_log[i].append(counts)
            if need is None:
                ok = max(counts.values()) >= px_min
            else:
                ok = all(counts[a] >= px_min for a in need)
            if ok:
                still.append(i)
        alive = still
        for a in ("left", "right"):
            link_mat = sapien_pose_to_mat(getattr(task.robot, f"{a}_camera").get_pose())
            for j, jm in enumerate(wrist_cands[a]):
                s = wseg.seg(sc, link_mat @ jm)
                wfrac[a][j].append(float(np.isin(s, list(grip_ids[a])).mean()))
                # lens-in-housing is fixed relative to the link, so it shows from the first
                # check frame on; later frames are close-ups of objects (reflective bowls,
                # transparent cups) where the two renderers legitimately disagree
                if wrt is not None and fi == check[0]:
                    wrt.entity.set_pose(sapien.Pose(link_mat @ jm))
                    sc.update_render()
                    wrt.take_picture()
                    rg = np.clip(wrt.get_picture("Color")[..., :3], 0, 1).mean(-1)
                    ok, sd, c = rt_view_ok(wseg.gray(), rg, cfg.wrist_rt_min_std,
                                           cfg.wrist_rt_min_corr)
                    wrt_log[a][j].append([round(sd, 1), round(c, 3)])
                    wrt_ok[a][j] = ok

    replay_stream(task, stream, frame_stride, on_frame=at_frame)

    # ---- greedy multi-view selection ----
    views = []
    for i in alive:
        c = cands[i]
        if all(baseline_angle_deg(c["pos"], v["pos"], target) >= cfg.min_separation_deg
               for v in views):
            views.append(dict(c, index=i, gripper_px=px_log[i]))
        if len(views) == n_views:
            break
    if len(views) < n_views:
        return {"ok": False, "reason": f"only {len(views)}/{n_views} views passed "
                                       f"({len(alive)} of {len(cands)} candidates survived)",
                "alive": len(alive), "candidates": len(cands)}

    wrist = {}
    for a in ("left", "right"):
        def passes(j):
            fr = wfrac[a][j]
            return (bool(fr) and min(fr) >= cfg.wrist_min_frac and max(fr) <= cfg.wrist_max_frac
                    and wrt_ok[a][j])
        # first jittered mount that passes; else the stock mount (flagged by `stock`)
        pick = next((j for j in range(1, len(wrist_cands[a])) if passes(j)), 0)
        wrist[a] = {"rel_mat": wrist_cands[a][pick], "candidate": pick,
                    "grip_frac": wfrac[a][pick], "stock": pick == 0,
                    "rt_check": wrt_log[a][pick], "rt_ok": wrt_ok[a][pick]}
    return {"ok": True, "views": views, "wrist": wrist, "target": target,
            "check_frames": check, "gate_arms": cfg.gate_arms, "need_arms": need,
            "arms_move": moves, "alive": len(alive), "candidates": len(cands),
            "min_gripper_px": px_min, "gate_size": cfg.gate_size}
