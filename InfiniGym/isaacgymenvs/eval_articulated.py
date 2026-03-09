
import hydra
from omegaconf import DictConfig, OmegaConf

import logging
import os
from datetime import datetime
import sys
import time
import select
import json
import xml.etree.ElementTree as ET

import isaacgym
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import isaacgymenvs
import imageio.v3 as iio
import numpy as np


def _generated_scene_has_articulation(scene_name: str) -> bool:
    asset_path = os.environ.get("ASSET_PATH", "../../asset_release")
    cfg_path = os.path.join(asset_path, "Task", scene_name, "asset_config.json")
    if not os.path.exists(cfg_path):
        return False

    with open(cfg_path, "r") as f:
        asset_cfg = json.load(f)

    scene_cfg = asset_cfg.get("scene_config", {})
    urdf_path = os.path.join(scene_cfg.get("asset_root", ""), scene_cfg.get("urdf_file", "asset.urdf"))
    if not os.path.exists(urdf_path):
        return False

    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "fixed") != "fixed":
            return True
    return False


def _requires_cpu_fallback(scene_list) -> bool:
    return any(_generated_scene_has_articulation(name) for name in scene_list)


def log_videos(path, idx, videos, fps=10):
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    print(f'{path}/log_{idx}.mp4')
    iio.imwrite(f'{path}/log_{idx}.mp4', np.stack(videos, axis=0), fps=fps)


def log_results(path, results):
    count, success = 0, 0

    log = {
        'z_threshold': [],
        'x_threshold': [],
        'e_threshold': [],
        'success': [],
        'label': [],
        'task_repeat': [],
        'extra': []
    }

    for i, res in enumerate(results):
        count += np.product(*np.array(res['success']).shape)
        success += np.array(res['success']).astype(np.float32).sum()

        for k, v in res.items():
            log[k].append(v)

    print("Success Rate: ", success / count)

    np.save(f'{path}/result.npy', log)


def wait_for_enter_with_render(vec_env, prompt: str):
    print(prompt, flush=True)
    if getattr(vec_env, "headless", True) or getattr(vec_env, "viewer", None) is None:
        try:
            input()
        except EOFError:
            pass
        return

    try:
        while True:
            if hasattr(vec_env, "env_physics_step") and hasattr(vec_env, "post_phy_step"):
                vec_env.env_physics_step()
                vec_env.post_phy_step()
            vec_env.render()
            time.sleep(0.01)
            if sys.stdin.isatty():
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    break
            else:
                # fallback for non-tty
                break
    except KeyboardInterrupt:
        pass


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def launch_eval_hydra(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    experiment_name = f'{cfg.scene.name}_{cfg.task.name}_{cfg.task.prefix}_{time_str}'
    experiment_dir = os.path.join('runs', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    # Load Eval Env Set
    cfg.task.task.scene_config_path = cfg.scene.scene_list
    cfg.task.experiment_name = experiment_name

    # GPU PhysX can segfault when creating articulated scene actors during eval.
    # If any selected generated scene contains non-fixed joints, force CPU PhysX.
    if _requires_cpu_fallback(cfg.scene.scene_list):
        print("[Eval] Articulated scene detected; forcing CPU PhysX for stability.")
        cfg.pipeline = "cpu"
        cfg.sim_device = "cpu"
        cfg.rl_device = "cpu"
        cfg.task.sim.use_gpu_pipeline = False
        cfg.task.sim.physx.use_gpu = False

    # Create Vectorized Env
    vec_env = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg
    )

    if cfg.task.name.startswith('FetchPtdOptimus'):
        from isaacgymenvs.tasks.fetch.utils.optimus_utils import load_optimus_algo
        algo = load_optimus_algo(cfg["task"])
        vec_env.update_algo(algo)
    if cfg.task.name.startswith('FetchPtdImit'):
        from isaacgymenvs.tasks.fetch.utils.imit_utils import load_imitation_algo
        algo = load_imitation_algo(cfg["task"]["solution"]["ckpt_path"])
        vec_env.update_algo(algo)

    # Eval Env (view initial placement only; no physics stepping)
    results, logs = [], []
    for i in range(cfg.scene.num_tasks):
        print(">>>>>>>>>>> reset task")
        vec_env.reset_task(i)
        wait_for_enter_with_render(vec_env, f"Initial state for task {i}. Press Enter to continue...")
        # print(">>>>>>>>>>> eval")
        # res = vec_env.eval()

        # res['extra'] = log
        # results.append(res)
        # print(">>>>>>>>>>> log videos", len(rgb))
        # log_videos(f'./videos/{experiment_name}', i, rgb, fps=24)

    # Log Results
    # log_results(f'./runs/{experiment_name}', results)

    vec_env.exit()
    exit()


if __name__ == "__main__":
    launch_eval_hydra()


# python eval_copy.py task=FetchNaive headless=False scene.scene_list='[generated_20260304_143055]' scene.num_tasks=1