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

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import argparse
import sys
import subprocess
import random

import isaacgym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def generate_scenes(cfg: DictConfig):
    # Set numpy formatting
    set_np_formatting()
    
    # Set seed
    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(-1, rank=global_rank)

    # Scene generation is typically low-throughput and Isaac Gym Preview can
    # segfault with articulated scene assets on GPU pipeline. Force CPU PhysX
    # for robustness during generation.
    if cfg.get("pipeline", "gpu") != "cpu" or cfg.sim_device != "cpu":
        print("[SceneGen] Forcing CPU PhysX pipeline for stability with articulated scenes.")
    cfg.pipeline = "cpu"
    cfg.sim_device = "cpu"
    cfg.rl_device = "cpu"
    if "sim" in cfg.task:
        cfg.task.sim.use_gpu_pipeline = False
        if "physx" in cfg.task.sim:
            cfg.task.sim.physx.use_gpu = False
    
    # Get output path
    asset_path = os.environ.get("ASSET_PATH", "../../asset_release")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_category = cfg.task.env.sceneCategory
    output_dir = os.path.join(asset_path, "Task", scene_category, f"generated_{timestamp}")
    
    # Override scene config path for InfiniScene BEFORE converting to dict
    cfg.task.sceneConfigPath = output_dir
    
    # Now convert to dict after setting sceneConfigPath
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    
    # Import after config is ready
    from isaacgymenvs.tasks.fetch.infini_scene.infini_scenes import InfiniScene
    
    # Choose one scene index at random from the list (use cfg.seed for determinism)
    sceneIdx = cfg_dict['task']['env']['sceneIdx']
    chosen = random.choice(sceneIdx)

    cfg_dict['task']['env']['sceneIdx'] = chosen

    # Create scene generator environment
    scene_gen = InfiniScene(
        cfg=cfg_dict['task'],
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=cfg.force_render,
    )

    # If arrangement generation failed during env creation, abort this attempt
    if getattr(scene_gen, "_arrangement_failed", False):
        print("✗ Arrangement failed during environment creation — skipping this attempt")
        # if hasattr(scene_gen, "destroy_env"):
        #     scene_gen.destroy_env()
        # Exit non-zero so caller (subprocess runner) knows this attempt failed
        sys.exit(1)
    
    
    # Run simulation to stabilize objects and keep the most stable snapshot
    stabilization_steps = cfg.task.get("stabilization_steps", 50)

    best_score = None
    best_step = -1
    saved = None

    # scene_gen.step(None)
    for step in range(stabilization_steps):
        for _ in range(120):
            scene_gen.env_physics_step()
            scene_gen.post_phy_step()

        # Stability score: lower mean speed is better
        obj_vel = scene_gen.states["obj_vel"]
        score = -obj_vel.norm(dim=-1).mean().item()

        if best_score is None or score > best_score:
            best_score = score
            best_step = step
            saved = scene_gen.save_env()
            scene_gen.log_camera_view_image(saved)

        print(f"  Step {step}/{stabilization_steps} (best={best_step})")
    
    # Fallback if nothing was saved during stabilization
    if saved is None:
        saved = scene_gen.save_env()

    # Count how many envs were saved in this run (saved is a list of bools per env)
    num_saved = int(sum(1 for s in saved if s)) if saved is not None else 0

    # If strict stability check failed, force-save latest state so generation does
    # not get stuck in endless retry loops for articulated scenes.
    if num_saved == 0:
        print("[SceneGen] Stability check failed; force-saving current snapshot.")
        saved = scene_gen.save_env(force_save=True)
        scene_gen.log_camera_view_image(saved)
        num_saved = int(sum(1 for s in saved if s)) if saved is not None else 0
    if num_saved > 0:
        print(f"\n{'='*80}")
        print(f"✓ Scene saved successfully to:")
        print(f"  {output_dir}")
        print(f"\nGenerated files:")
        print(f"  - asset_config.json")
        print(f"  - rearrange_config.npz")
        print(f"  - task_config.npz")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("✗ Scene generation failed - environment did not stabilize")
        print("Try adjusting:")
        print("  - stabilization_steps (increase)")
        print("  - buffer_ratio (increase for more spacing)")
        print("  - num_objects (decrease)")
        print(f"{'='*80}\n")
    
    # Cleanup
    # NOTE: destroy_env can block with some articulated-scene setups.
    # Keep cleanup optional; process exit will release resources in one-shot runs.
    if cfg.task.get("destroy_on_exit", False) and hasattr(scene_gen, "destroy_env"):
        scene_gen.destroy_env()
    # Exit with code 0 if at least one env was saved, else 1.
    sys.exit(0 if num_saved > 0 else 1)

if __name__ == "__main__":
    # Parse our custom args (remove them before Hydra runs)
    parser = argparse.ArgumentParser(description="Generate scenes multiple times")
    parser.add_argument("--num", "-n", type=int, default=1,
                        help="Number of times to generate the dataset")
    parser.add_argument("--child-attempt", action="store_true",
                        help="Internal flag: run exactly one generation attempt in this process")
    args, remaining_argv = parser.parse_known_args()

    # Check ASSET_PATH
    if "ASSET_PATH" not in os.environ:
        print("WARNING: ASSET_PATH not set. Using default '../asset_release'")
        print("Set it with: export ASSET_PATH=/path/to/FetchBench-CORL2024/asset_release")
        os.environ["ASSET_PATH"] = os.path.abspath("../asset_release")

    # Child mode: perform one hydra run in this process.
    if args.child_attempt:
        sys.argv[:] = [sys.argv[0]] + remaining_argv
        generate_scenes()
        sys.exit(0)

    # Parent mode: run each attempt in a fresh subprocess to avoid
    # Isaac Gym / PhysX singleton re-initialization crashes.
    success_count = 0
    attempts = 0
    script_path = os.path.abspath(__file__)

    while success_count < args.num:
        attempts += 1
        print(f"\n=== Attempt {attempts} (success {success_count}/{args.num}) ===\n")

        cmd = [sys.executable, script_path, "--num", "1", "--child-attempt", *remaining_argv]
        result = subprocess.run(cmd, env=os.environ.copy())
        ok = (result.returncode == 0)

        if ok:
            success_count += 1

    print(f"Completed {success_count}/{args.num} successful generations in {attempts} attempts.")
    sys.exit(0)
        


# python generate_scenes.py --num 2 headless=True
