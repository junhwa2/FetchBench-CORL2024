"""Shared hydra bootstrap for the VLM+VORM entry scripts.

The three entries — Phase 1 (`vlm_target_gen.py`), Phase 2 (`vlm_phase2.py`), and
the closed-loop server (`vlm_closed_loop/server_entry.py`) — used to duplicate the
identical env-construction block (numpy formatting, seed, scene wiring, and the
11-arg `isaacgymenvs.make` call). It lives here once.

Imports are lazy inside the function so this module never forces an import-order
dependency: the entry scripts must `import isaacgym` before torch, and they do so
at their top, well before calling `build_vec_env`.
"""

import os


def build_vec_env(cfg, experiment_name):
    """Apply the shared seed/scene/experiment wiring onto `cfg`, then build and
    return the vectorized env. `experiment_name` is entry-specific (each entry
    formats its own) and is written back to `cfg.task.experiment_name`."""
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import isaacgymenvs

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic,
                        rank=int(os.getenv("RANK", "0")))

    cfg.task.task.scene_config_path = cfg.scene.scene_list
    cfg.task.experiment_name = experiment_name

    return isaacgymenvs.make(
        cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device,
        cfg.rl_device, cfg.graphics_device_id, cfg.headless, cfg.multi_gpu,
        cfg.capture_video, cfg.force_render, cfg,
    )
