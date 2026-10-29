"""RoboTwin bridge: build a task, record the expert as a DENSE CONTROL STREAM, replay it.

Why a control stream
--------------------
RoboTwin's expert is a scripted planner (`play_once`): every `move()` plans a joint path
(curobo) and then executes it physics-step by physics-step through two primitives,
`Robot.set_arm_joints(pos, vel, arm)` and `Robot.set_gripper(val, arm, eps)`, followed by
`scene.step()`. We hook those three calls while the expert runs once and flatten what they
commanded into per-physics-step arrays:

    left_q   [N, dl]   left arm joint drive target in effect at step i
    left_dq  [N, dl]   left arm joint velocity target
    right_q  [N, dr]   right_dq [N, dr]
    left_g   [N]       left gripper command in [0, 1] (what set_gripper was asked for)
    right_g  [N]       right gripper command
    left_geps / right_geps [N]   the per-call gripper rate limit RoboTwin passes

Every rollout we render -- expert, calibration, every counterfactual -- is then a replay of
such a stream from the same seeded initial scene (`replay_stream`). That removes the
state-replay vs action-replay asymmetry the robocasa pipeline had to work around: the expert
video and its counterfactuals are produced by the same mechanism, so "how far did the
counterfactual diverge" can be measured against the expert itself.

The replay drives BOTH arms and BOTH grippers on every physics step with the target in effect
(RoboTwin only re-sends targets for the arm being moved; the drive keeps the last one). The
two are equivalent up to the gravity-compensation force, which RoboTwin refreshes on every
command call and we therefore refresh every step -- the expert is re-validated with
`check_success()` after its replay and dropped if the replay no longer succeeds.

All RoboTwin paths are relative ("./assets/..."), so the process must run with
cwd == ROBOTWIN_ROOT; `enter_robotwin()` does that and puts it on sys.path.
"""

import importlib
import os
import sys
from copy import deepcopy

import numpy as np
import yaml

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/home/yuncongyang_umass/kn/RoboTwin")


def enter_robotwin(root=None):
    root = os.path.abspath(root or ROBOTWIN_ROOT)
    if not os.path.isdir(os.path.join(root, "envs")):
        raise FileNotFoundError(f"ROBOTWIN_ROOT={root} does not look like a RoboTwin checkout")
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # two DIFFERENT single-arm robots run one curobo planner per arm in subprocesses
    # (Robot.set_planner); CUDA in a forked child fails, so spawn, as collect_data.py does
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    return root


# ----------------------------------------------------------------------------------------
# task construction
# ----------------------------------------------------------------------------------------
def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_task_args(task_name, task_config="demo_randomized", embodiment=None, overrides=None):
    """The `args` dict RoboTwin's collect_data.py builds, without its side effects.

    `embodiment` is RoboTwin's list form: ["aloha-agilex"] (one dual-arm robot) or
    ["piper", "franka-panda", 0.6] (two single-arm robots `interval` metres apart).
    """
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    args = _load_yaml(os.path.join(CONFIGS_PATH, f"{task_config}.yml"))
    args["task_name"] = task_name
    args["task_config"] = task_config
    if embodiment is not None:
        args["embodiment"] = list(embodiment)
    emb_types = _load_yaml(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"))

    def emb_file(name):
        return emb_types[name]["file_path"]

    emb = args["embodiment"]
    if len(emb) == 1:
        args["left_robot_file"] = args["right_robot_file"] = emb_file(emb[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(emb[0])
    elif len(emb) == 3:
        args["left_robot_file"] = emb_file(emb[0])
        args["right_robot_file"] = emb_file(emb[1])
        args["embodiment_dis"] = float(emb[2])
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{emb[0]}+{emb[1]}"
    else:
        raise ValueError("embodiment must have 1 or 3 entries")
    args["left_embodiment_config"] = _load_yaml(os.path.join(args["left_robot_file"], "config.yml"))
    args["right_embodiment_config"] = _load_yaml(os.path.join(args["right_robot_file"], "config.yml"))

    # we never use RoboTwin's own saving / viewer / point clouds
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = False
    args["data_type"] = dict(args.get("data_type") or {}, rgb=True, depth=False,
                             pointcloud=False, observer=False, third_view=False)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(args.get(k), dict):
            args[k] = dict(args[k], **v)
        else:
            args[k] = v
    return args


def make_task(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    return getattr(mod, task_name)()


def setup_task(task, args, seed, need_plan, joint_paths=None):
    """(Re)initialise `task` for `seed`. Raises RoboTwin's UnStableError on unstable scenes."""
    kw = dict(args)
    kw["need_plan"] = bool(need_plan)
    kw["left_joint_path"] = deepcopy(joint_paths["left"]) if joint_paths else []
    kw["right_joint_path"] = deepcopy(joint_paths["right"]) if joint_paths else []
    task.setup_demo(now_ep_num=0, seed=int(seed), **kw)
    return task


def close_task(task, clear_cache=False):
    try:
        task.close_env(clear_cache=clear_cache)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------
# control stream
# ----------------------------------------------------------------------------------------
STREAM_KEYS = ("left_q", "left_dq", "right_q", "right_dq", "left_g", "right_g",
               "left_geps", "right_geps")


class ControlStream(dict):
    """dict of per-physics-step arrays (see module docstring) + `n` steps."""

    @property
    def n(self):
        return int(len(self["left_g"]))

    def copy(self):
        return ControlStream({k: np.array(v, copy=True) for k, v in self.items()})

    def slice(self, a, b):
        return ControlStream({k: np.asarray(v)[a:b] for k, v in self.items()})

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if p is not None and p.n > 0]
        return ControlStream({k: np.concatenate([np.asarray(p[k]) for p in parts], axis=0)
                              for k in STREAM_KEYS})

    def hold(self, n_steps, at=-1):
        """n steps that keep the targets of step `at` (velocity targets zeroed)."""
        out = {}
        for k in STREAM_KEYS:
            v = np.asarray(self[k])[at]
            if k.endswith("_dq"):
                v = np.zeros_like(v)
            out[k] = np.repeat(np.asarray(v)[None], int(n_steps), axis=0)
        return ControlStream(out)

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.items()})

    @staticmethod
    def load(path):
        d = np.load(path)
        return ControlStream({k: d[k] for k in STREAM_KEYS})


def current_targets(robot):
    """Drive targets currently in effect, i.e. the state a stream continues from."""
    lq = np.array([j.get_drive_target()[0] for j in robot.left_arm_joints], dtype=np.float64)
    rq = np.array([j.get_drive_target()[0] for j in robot.right_arm_joints], dtype=np.float64)
    return {"left_q": lq, "left_dq": np.zeros_like(lq), "right_q": rq,
            "right_dq": np.zeros_like(rq), "left_g": float(robot.left_gripper_val),
            "right_g": float(robot.right_gripper_val), "left_geps": 0.1, "right_geps": 0.1}


class StreamRecorder:
    """Hooks Robot.set_arm_joints / Robot.set_gripper / scene.step on a live task and
    records the targets in effect at every physics step."""

    def __init__(self, task):
        self.task = task
        self.robot = task.robot
        self.scene = task.scene
        self.state = current_targets(self.robot)
        self.rows = {k: [] for k in STREAM_KEYS}
        self._orig = None

    def __enter__(self):
        robot, scene, st = self.robot, self.scene, self.state
        o_arm, o_grip, o_step = robot.set_arm_joints, robot.set_gripper, scene.step
        self._orig = (o_arm, o_grip, o_step)

        def set_arm_joints(target_position, target_velocity, arm_tag):
            st[f"{arm_tag}_q"] = np.asarray(target_position, dtype=np.float64).copy()
            st[f"{arm_tag}_dq"] = np.asarray(target_velocity, dtype=np.float64).copy()
            return o_arm(target_position, target_velocity, arm_tag)

        def set_gripper(gripper_val, arm_tag, gripper_eps=0.1):
            st[f"{arm_tag}_g"] = float(np.clip(gripper_val, 0, 1))
            st[f"{arm_tag}_geps"] = float(gripper_eps)
            return o_grip(gripper_val, arm_tag, gripper_eps)

        def step():
            for k in STREAM_KEYS:
                self.rows[k].append(np.copy(st[k]))
            return o_step()

        robot.set_arm_joints, robot.set_gripper, scene.step = set_arm_joints, set_gripper, step
        return self

    def __exit__(self, *exc):
        o_arm, o_grip, o_step = self._orig
        # instance attributes shadow the bound methods; deleting them restores the class's
        for obj, name in ((self.robot, "set_arm_joints"), (self.robot, "set_gripper"),
                          (self.scene, "step")):
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        if getattr(self.scene, "step", None) is not o_step and not hasattr(type(self.scene), "step"):
            self.scene.step = o_step
        return False

    def stream(self):
        return ControlStream({k: np.asarray(v) for k, v in self.rows.items()})


def apply_step(robot, s, i):
    """Command step i of stream s on both arms + both grippers (no scene.step)."""
    robot.set_arm_joints(s["left_q"][i], s["left_dq"][i], "left")
    robot.set_arm_joints(s["right_q"][i], s["right_dq"][i], "right")
    robot.set_gripper(float(s["left_g"][i]), "left", float(s["left_geps"][i]))
    robot.set_gripper(float(s["right_g"][i]), "right", float(s["right_geps"][i]))


def replay_stream(task, s, frame_stride, on_frame=None, start=0, stop=None, frame_offset=0):
    """Step stream s[start:stop] on a freshly set-up task; call on_frame(frame_idx, step_idx)
    BEFORE the first step and after every `frame_stride` physics steps (frame k shows the
    world after k*frame_stride steps). Returns the number of frames emitted."""
    stop = s.n if stop is None else min(stop, s.n)
    nf = 0
    if on_frame is not None and start == 0:
        on_frame(frame_offset, 0)
        nf += 1
    for i in range(start, stop):
        apply_step(task.robot, s, i)
        task.scene.step()
        if on_frame is not None and (i + 1) % frame_stride == 0:
            on_frame(frame_offset + (i + 1) // frame_stride, i + 1)
            nf += 1
    return nf


# ----------------------------------------------------------------------------------------
# expert: plan once while recording, keep the joint paths for exact re-planning-free reuse
# ----------------------------------------------------------------------------------------
def run_expert(task, args, seed):
    """Set up `seed`, run the scripted expert with recording on. Returns
    (ok, reason, stream, joint_paths, info)."""
    from envs.utils.create_actor import UnStableError  # noqa: F401  (raised by setup)

    setup_task(task, args, seed, need_plan=True)
    with StreamRecorder(task) as rec:
        info = task.play_once()
    stream = rec.stream()
    if not task.plan_success:
        return False, "plan_failed", stream, None, info
    if not task.check_success():
        return False, "expert_not_successful", stream, None, info
    ok, excess = task.planned_joints_legal()
    if not ok:
        return False, f"joint_limit_excess_{excess:.3f}", stream, None, info
    paths = {"left": deepcopy(task.left_joint_path), "right": deepcopy(task.right_joint_path)}
    return True, "ok", stream, paths, info


def run_release_expert(task, args, seed, joint_paths):
    """Replay a RoboTwin 2.0 release expert: set up its `seed` and run `play_once` on the saved
    planner trajectories (`_traj_data/episode<i>.pkl`) with planning off -- exactly RoboTwin's
    own collection stage 2. Returns (ok, reason, stream, joint_paths, info) like run_expert."""
    setup_task(task, args, seed, need_plan=False, joint_paths=joint_paths)
    with StreamRecorder(task) as rec:
        info = task.play_once()
    stream = rec.stream()
    if not task.plan_success:
        return False, "release_path_exhausted", stream, None, info
    if not task.check_success():
        return False, "release_replay_not_successful", stream, None, info
    return True, "ok", stream, joint_paths, info


# release embodiment name -> RoboTwin embodiment list (single-arm robots come as a pair)
RELEASE_EMBODIMENTS = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5", "ARX-X5", 0.6],          # spacing measured against the release
    "franka": ["franka-panda", "franka-panda", 0.8],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}
