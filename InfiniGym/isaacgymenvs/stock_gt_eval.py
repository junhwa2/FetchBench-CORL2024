"""Headless stock-GT baseline — FetchMeshCurobo (GT annotated grasps + curobo),
run headless-batch to isolate robot+curobo+GT-grasp capability.

The base `FetchMeshCurobo.solve()` path is instrumented for INTERACTIVE debugging
(`input("vis?")`, a forced `if True:` trimesh viz ending in `scene.show()`), which
blocks/crashes headless. Rather than edit the base file (no-edit convention), we
subclass it and, for the one offending method, neutralise just the two debug
touch-points around a `super()` call:

  * `input(...)`      -> returns "" (no EOFError on a dead stdin)
  * `grasp_vis_debug` -> no-op (kills the blocking `scene.show()` GUI window)

Everything else (grasp sampling, IK, motion gen, execution, success eval) is the
unmodified base behaviour. We re-register the subclass under the base task name so
`task=FetchMeshCurobo` reuses the existing config verbatim.

Usage
-----
  cd InfiniGym
  export ASSET_PATH=/path/to/asset_release
  export PYTHONPATH=$PWD
  python -u isaacgymenvs/stock_gt_eval.py task=FetchMeshCurobo \
      scene=benchmark_eval/RigidObjDesk_0 headless=True scene.num_tasks=25
"""

import builtins
from datetime import datetime

import hydra
from omegaconf import DictConfig

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np

from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import FetchMeshCurobo
from isaacgymenvs.tasks.fetch.utils.vlm_target.hydra_boot import build_vec_env


class FetchMeshCuroboHeadless(FetchMeshCurobo):
    """FetchMeshCurobo with the interactive debug tail of
    `sample_goal_obj_collision_free_grasp_pose` neutralised for headless runs."""

    def sample_goal_obj_collision_free_grasp_pose(self):
        orig_input = builtins.input
        orig_vis = self.grasp_vis_debug
        builtins.input = lambda *a, **k: ""       # auto-answer "vis?"/"continue?"
        self.grasp_vis_debug = lambda *a, **k: None  # drop the blocking scene.show()
        try:
            return super().sample_goal_obj_collision_free_grasp_pose()
        finally:
            builtins.input = orig_input
            self.grasp_vis_debug = orig_vis


# Re-register under the base name so task=FetchMeshCurobo builds this subclass.
try:
    from isaacgymenvs.tasks import isaacgym_task_map
    isaacgym_task_map["FetchMeshCurobo"] = FetchMeshCuroboHeadless
except Exception as _exc:  # pragma: no cover
    print("[stock_gt] task registration deferred: {}: {}".format(
        type(_exc).__name__, _exc))


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def main(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = "{}_{}_stockgt_{}".format(
        cfg.scene.name, cfg.task.name, time_str)
    vec_env = build_vec_env(cfg, experiment_name)

    n = int(cfg.scene.num_tasks)
    count = 0
    success = 0.0
    per_task = []
    try:
        for i in range(n):
            vec_env.reset_task(i)
            _rgb, log = vec_env.solve()
            res = vec_env.eval()
            arr = np.array(res["success"]).astype(np.float32)
            s = float(arr.sum())
            c = int(np.prod(arr.shape)) if arr.size else 1
            count += c
            success += s
            per_task.append((i, s, c))
            print("[stockgt] task {}: success={}/{}".format(i, s, c), flush=True)
    finally:
        rate = 100.0 * success / max(count, 1)
        print("[stockgt] SUCCESS RATE: {}/{} = {:.1f}%".format(
            int(success), count, rate), flush=True)
        print("[stockgt] per_task:", per_task, flush=True)
        vec_env.exit()


if __name__ == "__main__":
    main()
