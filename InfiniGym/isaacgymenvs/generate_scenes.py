#!/usr/bin/env python3
"""
Scene Generation Script for FetchBench

This script creates new scene configurations with customizable object placement and clustering.
Generated scenes are saved to ${ASSET_PATH}/Task/ and can be used for eval/training.

Usage:
    python generate_scenes.py task=InfiniSceneGen headless=True
    
    # Adjust clustering (edit config/task/InfiniSceneGen.yaml):
    # - buffer_ratio: 0.05 (default 0.1) = more clustered
    # - dist_above_support: 0.005 = lower placement
    # - envSpacing: 2.0 = tighter overall space
    
    # Change objects:
    # - num_objects: number of rigid objects
    # - num_combos: number of combo objects
    # - scene_category: Desk, Shelf, Drawer, etc.
"""
import json
import pickle
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import argparse
import sys
import subprocess
import random
import gc
import copy
import numpy as np

import isaacgym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.tasks.fetch.infini_scene.infini_scenes import InfiniScene
from isaacgymenvs.tasks.fetch.utils.load_utils import InfiniSceneLoader


def check_saved_config_shapes(path):
    if not os.path.exists(path):
        print(f"[check_saved_task_config_shapes] Not found: {path}")
        return False
    # Check all required files exist
    asset_json = os.path.join(path, "asset_config.json")
    rearrange_npz = os.path.join(path, "rearrange_config.npz")
    task_npz = os.path.join(path, "task_config.npz")
    if not (os.path.exists(asset_json) and os.path.exists(rearrange_npz) and os.path.exists(task_npz)):
        print(f"[check_saved_task_config_shapes] Not all config files found in: {path}")
        return False
    loader = InfiniSceneLoader(path)
    loader.load_env_config()
    task_config = loader.load_task_config()
    loader.task_actor_states = task_config['task_init_state']
    loader.task_target_index = task_config['task_obj_index']
    loader.task_target_labels = task_config['task_obj_label']
    loader.task_camera_poses_ = task_config['task_camera_pose']
    loader.task_obj_indices_ = task_config['task_cand_obj_index']
    loader.task_obj_labels_ = task_config['task_cand_obj_label']
    return loader

def fix_obj(path):
    pkl_path = os.path.join(path, 'sampled_object_asset.pkl')
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        object_asset = pickle.load(f)
    return object_asset


def generate_scenes(target_envs, output_dir, cfg):
    set_np_formatting()
    
    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(-1, rank=global_rank)

    if cfg.get("pipeline", "gpu") != "cpu" or cfg.sim_device != "cpu":
        print("[SceneGen] Forcing CPU PhysX pipeline for stability with articulated scenes.")
    cfg.pipeline = "cpu"
    cfg.sim_device = "cpu"
    cfg.rl_device = "cpu"
    if "sim" in cfg.task:
        cfg.task.sim.use_gpu_pipeline = False
        if "physx" in cfg.task.sim:
            cfg.task.sim.physx.use_gpu = False
    
    os.makedirs(output_dir, exist_ok=True)
    
    cfg.task.sceneConfigPath = output_dir
    
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['task']['env']['numEnvs'] = int(target_envs)
    print_dict(cfg_dict)
    fixed_objects = fix_obj(output_dir)

    scene_gen = InfiniScene(
        cfg=cfg_dict['task'],
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=cfg.force_render,
        fixed_objects=fixed_objects
    )
    for _ in range(120):
        scene_gen.env_physics_step()
    scene_gen.post_phy_step()
    saved = scene_gen.save_env()
    scene_gen.log_camera_view_image(saved)

    loader = copy.deepcopy(scene_gen.loader)
    print("saved:", saved)

    del scene_gen
    gc.collect()
    return loader

def append_stable_loader(loader, loader_tmp):
    """Append list fields from `loader_tmp` into `loader`.

    This merges only Python-side config/task data (poses/labels/tasks).
    IsaacGym sim state is NOT merged.
    """
    if loader is False:
        return loader_tmp
    fields = [
        "robot_pose",
        "scene_pose",
        "object_poses",
        "camera_poses",
        "object_labels",
        "task_actor_states",
        "task_target_index",
        "task_target_labels",
        "task_camera_poses_",
        "task_obj_indices_",
        "task_obj_labels_",
    ]

    for name in fields:
        value = getattr(loader, name)
        if not isinstance(value, list):
            value = value.tolist() if hasattr(value, "tolist") else list(value)

        value_tmp = getattr(loader_tmp, name)
        if not isinstance(value_tmp, list):
            value_tmp = value_tmp.tolist() if hasattr(value_tmp, "tolist") else list(value_tmp)

        value.extend(value_tmp)
        setattr(loader, name, value)
    return loader
        # value.extend(value_tmp)


if __name__ == "__main__":
    # Single-process: Only run generate_scenes() if not enough scenes saved
    if "ASSET_PATH" not in os.environ:
        print("WARNING: ASSET_PATH not set. Using default '../asset_release'")
        print("Set it with: export ASSET_PATH=/path/to/FetchBench-CORL2024/asset_release")
        os.environ["ASSET_PATH"] = os.path.abspath("../asset_release")
    asset_path = os.environ.get("ASSET_PATH", "../../asset_release")

    # Use hydra to load config as in generate_scenes
    overrides = []
    for arg in sys.argv[1:]:
        if '=' in arg:
            overrides.append(arg)
    config_dir_abs = os.path.abspath("./config")
    with hydra.initialize_config_dir(config_dir=config_dir_abs):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    sceneIdx = cfg.task.env.sceneIdx

    cfg_dict = omegaconf_to_dict(cfg)
    num_envs = int(cfg_dict['task']['env']['numEnvs'])
    scene_category = cfg_dict['task']['env']['sceneCategory']
    scene_category_short = scene_category[:-len("SceneFactory")]
    
    output_dir = os.path.join(asset_path, "Task", "v1.2", scene_category_short, scene_category+"_"+str(sceneIdx[0]))
    print("num_envs:", num_envs)
    main_loader = check_saved_config_shapes(output_dir)


    if main_loader is not False:
        print(f"Saved count: {main_loader.task_actor_states.shape[0]}")
    else:
        print("Saved count: 0")
        

    if main_loader is not False:
        target_envs = num_envs - main_loader.task_actor_states.shape[0]
    else:
        target_envs = num_envs


    if (main_loader is False) or main_loader.task_actor_states.shape[0] < num_envs:
        loader = generate_scenes(target_envs, output_dir, cfg)
        main_loader = append_stable_loader(main_loader, loader)
        main_loader.save_env_config()
        print(f"\n✓ Scene generation complete. Total saved scenes: {(len(main_loader.scene_pose))}")

    else:
        print(f"[SKIP] Already have {main_loader.task_actor_states.shape[0]} scenes (numEnvs={num_envs}) in {output_dir}")
