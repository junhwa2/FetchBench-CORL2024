"""Entry point for the MULTI-VIEW closed-loop server task.  NEW FILE ONLY.

Identical launch behaviour to server_entry.py, but additionally imports
server_task_mv so FetchMeshCuroboGORunVLMServerMV self-registers into
isaacgym_task_map before the env is built.  server_entry.py stays untouched.

Import order matters: server_entry imports `isaacgym` before torch and defines
`launch_server_hydra` (whose hydra config_path is anchored to server_entry.py —
same directory, so it resolves unchanged).  We import it FIRST, which runs that
isaacgym-before-torch bootstrap, then import server_task_mv (torch is now safe).

Run exactly like server_entry.py but with the MV task:

  cd InfiniGym
  export ASSET_PATH=/path/to/asset_release_v1.3
  export VLM_RPC_DIR=/tmp/vlm_rpc/DeskShelf_0
  python isaacgymenvs/tasks/fetch/vlm_closed_loop/server_entry_mv.py \
      task=FetchMeshCuroboGORunVLMServerMV \
      scene=benchmark_eval/RigidObjDesk_0 headless=True
"""
from isaacgymenvs.tasks.fetch.vlm_closed_loop.server_entry import (  # noqa: F401
    launch_server_hydra,
)

# Self-registers FetchMeshCuroboGORunVLMServerMV (isaacgym already imported by
# server_entry above, so the torch-touching imports here are safe).
from isaacgymenvs.tasks.fetch.vlm_closed_loop.server_task_mv import (  # noqa: F401,E501
    FetchMeshCuroboGORunVLMServerMV,
)


if __name__ == "__main__":
    launch_server_hydra()
