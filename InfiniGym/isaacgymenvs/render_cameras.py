"""Capture every camera's RGB for a few tasks, to show observation coverage.

Builds the same env the closed-loop server uses, resets to selected tasks, and
saves each camera view (the narrow observation cam(s) + the wide overview 'vis'
cam) so the perception-coverage gap is visible: a GT target absent from the
observation cam's point cloud is still present in the overview.
"""
import os
import hydra
from omegaconf import DictConfig

import isaacgym  # noqa: F401  (before torch)
import numpy as np
import imageio.v3 as iio

# import registers FetchMeshCuroboGORunVLMServer into isaacgym_task_map
from isaacgymenvs.tasks.fetch.vlm_closed_loop.server_task import (  # noqa: F401
    FetchMeshCuroboGORunVLMServer,
)
from isaacgymenvs.tasks.fetch.utils.vlm_target.hydra_boot import build_vec_env

OUT = "/tmp/claude-1000/-home-jo-HJ-FetchBench-CORL2024/b7c4031a-a89a-4f8b-bbd9-1db4910d5c02/scratchpad/cam_capture"
TASKS = [1, 3, 4, 8]   # 1,3 = has-grasp ; 4,8 = no_target_grasp


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def main(cfg: DictConfig):
    env = build_vec_env(cfg, "camcap")
    os.makedirs(OUT, exist_ok=True)
    for i in TASKS:
        env.reset_task(i)
        for _ in range(30):
            env.env_physics_step()
            env.post_phy_step()
        rgb, _ = env.get_camera_image(rgb=True, seg=False)
        cams = rgb[0]  # list of per-camera images for env 0
        for c, img in enumerate(cams):
            arr = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8)
            iio.imwrite(os.path.join(OUT, "task%d_cam%d.png" % (i, c)), arr)
        print("[camcap] task %d: saved %d cameras" % (i, len(cams)), flush=True)
    print("[camcap] output dir:", OUT, flush=True)
    env.exit()


if __name__ == "__main__":
    main()
