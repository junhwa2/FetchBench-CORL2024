#!/usr/bin/env python3

import os
import hydra
from omegaconf import DictConfig
import sys
import random
import tempfile
import argparse
import csv
import subprocess
from typing import List, Dict, Optional, Any, Tuple

from isaacgymenvs.tasks.fetch.infini_scene.infini_scenes import InfiniScene
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import torch


_LOG_CSV_PATH: Optional[str] = None
_RUN_OBJECT: Optional[Dict[str, str]] = None


# Number of stability trials to record per object in batch mode.
_NUM_TRIALS: int = 5


def _csv_fieldnames() -> List[str]:
    # Per-object N-trial schema
    return [
        "object",  # "Category/ID"
        *[f"stable_{i}" for i in range(1, _NUM_TRIALS + 1)],
    ]


def _init_csv(path: str, append: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        if not append:
            writer.writeheader()
        else:
            if f.tell() == 0:
                writer.writeheader()


def _append_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        for row in rows:
            writer.writerow(row)

def _motion_stability_check(
    scene_gen,
    max_steps: int = 180,
    hold_steps: int = 170,
    vel_lin_eps: float = 0.015,
    vel_ang_eps: float = 0.15,
    pos_eps: float = 0.0005,
    log_every: int = 60,
):
    """Run physics and decide stabilization using obj vel + per-step pos change.

    Criteria per env (vectorized):
      - max linear speed across objects < vel_lin_eps
      - max angular speed across objects < vel_ang_eps
      - max per-step position delta across objects < pos_eps
    All three must hold for `hold_steps` consecutive steps.

        Returns:
            (stable_envs: torch.BoolTensor[num_envs], steps: int, window: (start, end) or (None, None))
    """
    device = scene_gen.device

    # Make sure tensors are fresh before we start measuring deltas.
    scene_gen.post_phy_step()
    prev_pos = scene_gen.states["obj_pos"].clone()

    stable_run = torch.zeros(scene_gen.num_envs, dtype=torch.int32, device=device)
    stable_envs = torch.zeros(scene_gen.num_envs, dtype=torch.bool, device=device)

    stable_window: Tuple[Optional[int], Optional[int]] = (None, None)

    for step in range(1, max_steps + 1):
        scene_gen.env_physics_step()
        scene_gen.post_phy_step()

        obj_pos = scene_gen.states["obj_pos"]  # (E, O, 3)
        obj_vel = scene_gen.states["obj_vel"]  # (E, O, 6)

        lin_speed = torch.linalg.vector_norm(obj_vel[..., :3], dim=-1)  # (E, O)
        ang_speed = torch.linalg.vector_norm(obj_vel[..., 3:], dim=-1)  # (E, O)
        pos_delta = torch.linalg.vector_norm(obj_pos - prev_pos, dim=-1)  # (E, O)

        # Reduce to env-level signals.
        env_lin = lin_speed.max(dim=-1).values
        env_ang = ang_speed.max(dim=-1).values
        env_pos = pos_delta.max(dim=-1).values

        env_ok = (env_lin < vel_lin_eps) & (env_ang < vel_ang_eps) & (env_pos < pos_eps)
        stable_run = torch.where(env_ok, stable_run + 1, torch.zeros_like(stable_run))
        stable_envs = stable_run >= hold_steps

        if log_every > 0 and (step % log_every == 0 or step == 1):
            # Print worst env stats for quick debugging.
            worst_lin = float(env_lin.max().item())
            worst_ang = float(env_ang.max().item())
            worst_pos = float(env_pos.max().item())
            num_stable = int(stable_envs.sum().item())
            print(
                f"[Stabilize] step={step}/{max_steps} stable={num_stable}/{scene_gen.num_envs} "
                f"worst_lin={worst_lin:.4f} worst_ang={worst_ang:.4f} worst_dpos={worst_pos:.6f}"
            )

        # Record the first window where env0 has held stability for hold_steps.
        # (We run numEnvs=1 for this script, but keep the logic explicit.)
        if stable_window[1] is None:
            run0 = int(stable_run[0].item())
            if run0 == hold_steps:
                stable_window = (step - hold_steps + 1, step)

        if bool(stable_envs.all().item()):
            return stable_envs, step, stable_window

        prev_pos = obj_pos.clone()
    return stable_envs, max_steps, stable_window


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_true_benchmark_objects(metadata_csv_path: str, eval_only: bool) -> List[dict]:
    """Load objects from benchmark metadata where use_benchmark=True (and use_eval if eval_only)."""
    objects: List[dict] = []
    with open(metadata_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _parse_bool(row.get("use_benchmark", "False")):
                continue
            if eval_only:
                if not _parse_bool(row.get("use_eval", "False")):
                    continue
            else:
                # keep both use_eval False and True? For parity with sample_random_objects(eval_only=False),
                # we take use_eval=False.
                if _parse_bool(row.get("use_eval", "False")):
                    continue

            cat = row.get("Category")
            oid = row.get("ID")
            if not cat or not oid:
                continue
            objects.append({"Category": cat, "ID": oid, "Path": row.get("Path", "")})
    return objects


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
    
    # Ensure ASSET_PATH exists (required by load_utils.py at import time).
    # Default: repo_root/asset_release
    if "ASSET_PATH" not in os.environ:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        os.environ["ASSET_PATH"] = os.path.join(repo_root, "asset_release")

    # We don't save scenes here; use a temp folder for sceneConfigPath.
    cfg.task.sceneConfigPath = tempfile.mkdtemp(prefix="fetchbench_stability_")
    
    # Now convert to dict after setting sceneConfigPath
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # Force these to 1 for stability testing during InfiniScene construction.
    # - TrimeshRearrangeScene(..., self.cfg['env']['numObjs'], ...)
    # - random_arrangement_JH(self.cfg['env']['numSceneObjs'], ...)
    cfg_dict.setdefault("task", {}).setdefault("env", {})
    cfg_dict["task"]["env"]["numObjs"] = 1
    cfg_dict["task"]["env"]["numSceneObjs"] = 1

    # If parent process passed an override object, keep it.
    # Expected: cfg.task.env.objectOverride = {'Category': ..., 'ID': ...}

    # Import after ASSET_PATH and cfg are ready
    
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

    # --- Stabilize with physics and evaluate stability (vel + pos) ---
    stab_cfg = cfg_dict.get("task", {})
    # Defaults must match _motion_stability_check() so that the behavior is
    # predictable even when users don't pass any Hydra overrides.
    max_steps = int(stab_cfg.get("stability_max_steps", 180))
    hold_steps = int(stab_cfg.get("stability_hold_steps", 160))
    vel_lin_eps = float(stab_cfg.get("stability_vel_lin_eps", 0.02))
    vel_ang_eps = float(stab_cfg.get("stability_vel_ang_eps", 0.2))
    pos_eps = float(stab_cfg.get("stability_pos_eps", 1e-3))
    log_every = int(stab_cfg.get("stability_log_every", 60))

    if hold_steps > max_steps:
        print(
            f"[Stabilize] warning: hold_steps({hold_steps}) > max_steps({max_steps}); "
            "stability can never become True unless you increase max_steps or reduce hold_steps."
        )

    print(
        "[Stabilize] params: "
        f"max_steps={max_steps} hold_steps={hold_steps} "
        f"vel_lin_eps={vel_lin_eps} vel_ang_eps={vel_ang_eps} pos_eps={pos_eps}"
    )

    stable_envs, used_steps, stable_window = _motion_stability_check(
        scene_gen,
        max_steps=max_steps,
        hold_steps=hold_steps,
        vel_lin_eps=vel_lin_eps,
        vel_ang_eps=vel_ang_eps,
        pos_eps=pos_eps,
        log_every=log_every,
    )

    stable_list = stable_envs.detach().cpu().numpy().tolist()

    print(f"[Stabilize] done: steps={used_steps} stable_envs={stable_list}")
    # input("start")
    # NOTE: CSV writing is handled by the parent process (batch mode).

    # Cleanup (optional)
    if cfg.task.get("destroy_on_exit", False) and hasattr(scene_gen, "destroy_env"):
        scene_gen.destroy_env()

    # Exit code: 0 if all envs stabilized, else 2.
    sys.exit(0 if bool(stable_envs.all().item()) else 2)


if __name__ == "__main__":
    # Parent mode: iterate all True benchmark objects and spawn one child process per object.
    # Child mode: run a single object specified by CLI args.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true", help="internal: run one object stability check")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--start", type=int, default=0, help="start index in filtered list")
    parser.add_argument("--log-csv", type=str, default="stability_logs.csv")
    parser.add_argument("--append", action="store_true", help="append to existing CSV instead of overwriting")
    args, remaining = parser.parse_known_args()

    if args.child:
        # Inject Hydra overrides for single object.
        if not args.category or not args.object_id:
            raise SystemExit("--child requires --category and --object-id")
        sys.argv = [sys.argv[0], *remaining,
                    f"+task.env.objectOverride.Category={args.category}",
                    f"+task.env.objectOverride.ID={args.object_id}"]
        generate_scenes()
        raise SystemExit(0)

    # Parent
    if "ASSET_PATH" not in os.environ:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        os.environ["ASSET_PATH"] = os.path.join(repo_root, "asset_release")

    metadata_csv = os.path.join(os.environ["ASSET_PATH"], "benchmark_objects", "metadata.csv")
    # Default cfg has task.use_eval=True for benchmark; follow that convention here.
    # If user wants the other split, pass `task.use_eval=False` in remaining args.
    eval_only = True
    for token in remaining:
        if token.strip() in {"task.use_eval=False", "+task.use_eval=False"}:
            eval_only = False
            break

    all_objs = _load_true_benchmark_objects(metadata_csv, eval_only=eval_only)
    if args.start > 0:
        all_objs = all_objs[args.start:]
    if args.limit and args.limit > 0:
        all_objs = all_objs[:args.limit]

    print(f"[Batch] objects={len(all_objs)} eval_only={eval_only} metadata={metadata_csv}")

    _init_csv(args.log_csv, append=args.append)

    script = os.path.abspath(__file__)
    passed: List[str] = []
    failed: List[str] = []
    for i, obj in enumerate(all_objs):
        cat, oid = obj["Category"], obj["ID"]
        tag = f"{cat}/{oid}"
        print(f"\n[Batch] ({i+1}/{len(all_objs)}) {tag}")

        outcomes: List[bool] = []
        for trial in range(_NUM_TRIALS):
            cmd = [
                sys.executable,
                script,
                "--child",
                "--category",
                cat,
                "--object-id",
                oid,
                *remaining,
            ]
            r = subprocess.run(cmd, env=os.environ.copy())
            ok = (r.returncode == 0)
            outcomes.append(ok)
            print(f"[Batch]   trial {trial + 1}/{_NUM_TRIALS}: {ok}")

        row: Dict[str, Any] = {"object": tag}
        row.update({f"stable_{k+1}": outcomes[k] for k in range(_NUM_TRIALS)})
        _append_csv_rows(args.log_csv, [row])

        if all(outcomes):
            passed.append(tag)
        else:
            failed.append(tag)

    print("\n" + "=" * 80)
    print(f"[Batch] done: passed={len(passed)} failed={len(failed)}")
    if failed:
        print("[Batch] failed examples:")
        for t in failed[:20]:
            print("  ", t)
    # Exit non-zero if any failures.
    raise SystemExit(0 if not failed else 2)

# python test_object_stability.py