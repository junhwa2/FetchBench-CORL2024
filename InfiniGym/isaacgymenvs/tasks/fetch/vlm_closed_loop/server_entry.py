"""FetchBench sim server for the VLM+VORM closed loop (see VLM_SIM_RPC_SPEC.md).

Builds one Isaac Gym env for a scene, then serves requests from the vorm_pipeline
master over a file-RPC instead of running eval.py's fixed solve loop. The master
(`loop_runner_mj.py`) issues reset/observe/decide_goal/execute; this process owns
rendering, the sim-side Gemini-ER VLM, and grasp execution.

Usage
-----
  cd InfiniGym
  export ASSET_PATH=/path/to/asset_release_v1.3
  export GEMINI_API_KEY=...                       # for backend=gemini
  export VLM_RPC_DIR=/tmp/vlm_rpc/DeskShelf_0     # shared session dir

  python isaacgymenvs/tasks/fetch/vlm_closed_loop/server_entry.py \
      task=FetchMeshCuroboGORunVLMServer \
      scene=benchmark_eval/RigidObjDesk_0 headless=True

The master creates VLM_RPC_DIR (fresh) and connects to the same path. One server
process == one scene (Isaac Gym binds scene assets at construction); the master's
outer shell launches one server per scene. Within a scene the server resets
across tasks in-process, exactly like ThinkGrasp.
"""

import os
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import imageio.v3 as iio

from isaacgymenvs.tasks.fetch.utils.vlm_target.hydra_boot import build_vec_env

# Importing the task module self-registers FetchMeshCuroboGORunVLMServer into
# isaacgym_task_map - no edit to isaacgymenvs/tasks/__init__.py required.
from isaacgymenvs.tasks.fetch.vlm_closed_loop.server_task import (  # noqa: F401,E501
    FetchMeshCuroboGORunVLMServer,
)
from isaacgymenvs.tasks.fetch.vlm_closed_loop.sim_rpc import RpcServer


class _ServerApp(object):
    """Bridges RPC methods to the env and serialises observation payloads."""

    def __init__(self, env, root):
        self.env = env
        self._server = RpcServer(root, self.handle)
        self.io = self._server.io

    # -- payload serialisation ------------------------------------------- #
    def _write_obs(self, obs, rid):
        rgb_name = "obs_{}_rgb.png".format(rid)
        pc_name = "obs_{}_pc.ply".format(rid)
        qpos_name = "obs_{}_qpos.npy".format(rid)

        iio.imwrite(self.io.payload_path(rgb_name), np.asarray(obs["rgb"]))
        self.env.save_pc(obs["pc"], self.io.payload_path(pc_name))
        np.save(self.io.payload_path(qpos_name), np.asarray(obs["qpos"]))

        return {
            "step": int(obs["step"]),
            "rgb": self.io.rel(self.io.payload_path(rgb_name)),
            "pc": self.io.rel(self.io.payload_path(pc_name)),
            "qpos": self.io.rel(self.io.payload_path(qpos_name)),
            "cam": obs.get("cam", {}),
        }

    # -- RPC dispatch ---------------------------------------------------- #
    def handle(self, method, req):
        rid = req.get("id", 0)
        if method == "reset":
            body = self.env.server_reset(int(req["task_idx"]))
            body["obs"] = self._write_obs(self.env.server_observe(), rid)
            return body
        if method == "observe":
            return {"obs": self._write_obs(self.env.server_observe(), rid)}
        if method == "decide_goal":
            return self.env.server_decide_goal(req.get("instruction"))
        if method == "execute":
            res = self.env.server_execute(
                int(req["goal_seg_id"]),
                req.get("grasp_pose") or req.get("grasp_poses") or [],
                place_pose=req.get("place_pose"),
                release=req.get("release"),
            )
            res["obs"] = self._write_obs(self.env.server_observe(), rid)
            return res
        if method == "close":
            return {}
        raise ValueError("unknown method: {}".format(method))

    def serve(self, root):
        print("[vlm_sim_server] serving on {}".format(os.path.abspath(root)))
        self._server.serve_forever()


# config_path is relative to THIS file: vlm_closed_loop -> fetch -> tasks ->
# isaacgymenvs -> config (the same config dir the other entries use from
# isaacgymenvs/ as "./config").
@hydra.main(version_base="1.1", config_name="config",
            config_path="../../../config")
def launch_server_hydra(cfg: DictConfig):
    root = os.environ.get("VLM_RPC_DIR")
    if not root:
        root = OmegaConf.select(cfg, "rpc_dir")
    if not root:
        raise SystemExit(
            "set VLM_RPC_DIR=<shared session dir> (or +rpc_dir=<dir>) so the "
            "master and this server agree on the file-RPC directory.")

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = "{}_{}_server_{}".format(
        cfg.scene.name, cfg.task.name, time_str)

    vec_env = build_vec_env(cfg, experiment_name)

    app = _ServerApp(vec_env, root)
    try:
        app.serve(root)
    finally:
        vec_env.exit()


if __name__ == "__main__":
    launch_server_hydra()
