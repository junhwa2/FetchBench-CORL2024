# Phase 2 of VLM+VORM: run the stock singulation planner on the VLM's target
# (from a targets.npz sidecar) with the VORM-driven KB, WITHOUT mutating the
# benchmark dataset. Mirrors isaacgymenvs/eval.py but runs
# FetchMeshCuroboGORunOverride, which reads solution.target_override.
#
# Usage:
#   cd InfiniGym
#   export ASSET_PATH=/path/to/asset_release_v1.3     # Task symlink set
#   python isaacgymenvs/vlm_phase2.py task=FetchMeshCuroboGORunOverride \
#       scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
#       task.solution.obs_folder=Obstruction_260513_60_111_sim/benchmark_eval \
#       task.solution.pred_folder=vorm_obstruction_260516 \
#       task.solution.target_override=$PWD/vlm_target_logs/<exp>/targets.npz
#
# Then join Phase 1 + Phase 2 for the instruction-conditioned metric:
#   python scripts/vlm_join_results.py \
#       --vlm-log InfiniGym/vlm_target_logs/<exp> \
#       --sim-run InfiniGym/runs/<this run>

import hydra
from omegaconf import DictConfig, OmegaConf

import os
from datetime import datetime

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import imageio.v3 as iio

from isaacgymenvs.tasks.fetch.utils.vlm_target.hydra_boot import build_vec_env
from isaacgymenvs.utils.hj_utils import npy_to_csv

# Importing the override task self-registers FetchMeshCuroboGORunOverride into
# isaacgym_task_map - no edit to isaacgymenvs/tasks/__init__.py required.
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo_go_run_override import (  # noqa: F401
    FetchMeshCuroboGORunOverride,
)


def log_videos(path, idx, videos, fps=24):
    if not videos:
        print("[log_videos] no frames for task {} - skipped".format(idx))
        return
    if not os.path.exists(path):
        os.makedirs(path)
    iio.imwrite("{}/log_{}.mp4".format(path, idx), np.stack(videos, axis=0), fps=fps)


def log_results(path, results):
    from collections import defaultdict
    count, success = 0, 0
    log = defaultdict(list)
    for i, res in enumerate(results):
        count += np.product(*np.array(res["success"]).shape)
        success += np.array(res["success"]).astype(np.float32).sum()
        for k, v in res.items():
            log[k].append(v)
    log = dict(log)
    print("Success Rate: ", success / count if count else 0.0)
    np.save("{}/result.npy".format(path), log)
    npy_to_csv("{}/result.npy".format(path), "{}/result.csv".format(path))


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def launch(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = "{}_{}_{}_{}".format(
        cfg.scene.name, cfg.task.name, cfg.task.prefix, time_str)
    experiment_dir = os.path.join("runs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    vec_env = build_vec_env(cfg, experiment_name)

    results = []
    for i in range(cfg.scene.num_tasks):
        vec_env.reset_task(i)
        rgb, log = vec_env.solve()
        res = vec_env.eval()
        res["extra"] = log
        results.append(res)
        log_videos("./videos/{}".format(experiment_name), i, rgb, fps=24)

    log_results("./runs/{}".format(experiment_name), results)

    vec_env.exit()
    exit()


if __name__ == "__main__":
    launch()
