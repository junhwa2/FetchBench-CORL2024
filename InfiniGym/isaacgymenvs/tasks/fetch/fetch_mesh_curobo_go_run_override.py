"""Phase-2 target override WITHOUT editing existing code.

The external review proposed reading the planner target from a `targets.npz`
sidecar (so the benchmark dataset under $ASSET_PATH stays read-only) by editing
`FetchMeshCuroboGORun.solve()`. This project forbids modifying existing files,
so the same behaviour is achieved by SUBCLASSING instead: pred-mode `solve()`
reads its target through `self.load_planner_target_from_config()`, which is an
overridable method. We override it to consult `solution.target_override` first.

Behaviour is identical to the stock task when `target_override` is unset.

Phase 2:
    python isaacgymenvs/vlm_phase2.py task=FetchMeshCuroboGORunOverride \
        scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
        task.solution.obs_folder=<...>_sim/benchmark_eval \
        task.solution.pred_folder=vorm_obstruction_260516 \
        task.solution.target_override=<abs>/vlm_target_logs/<exp>/targets.npz

Registered here on import, so `isaacgymenvs/tasks/__init__.py` stays untouched.
"""

import numpy as np

from .fetch_mesh_curobo_go_run import FetchMeshCuroboGORun


class FetchMeshCuroboGORunOverride(FetchMeshCuroboGORun):
    """Stock GORun, but the cached-target read can come from a targets.npz
    sidecar (`solution.target_override`) instead of task_config.npz."""

    def _load_target_override(self, override_path):
        cache = getattr(self, "_target_override_cache", None)
        if cache is None or cache.get("path") != override_path:
            data = np.load(override_path, allow_pickle=True)
            idx = np.asarray(data["task_idx"]).reshape(-1).astype(int)
            tgt = np.asarray(data["target_old_id"]).reshape(-1).astype(int)
            cache = {"path": override_path,
                     "map": {int(i): int(t) for i, t in zip(idx, tgt)}}
            self._target_override_cache = cache
            print("[override] loaded target_override {} ({} tasks)"
                  .format(override_path, len(cache["map"])))
        return cache["map"]

    def load_planner_target_from_config(self, env_idx=0):
        """Read the VLM target from the sidecar when configured; otherwise fall
        back to the stock task_config.npz read (unchanged behaviour).

        Returns (target_old_id, has_marker), matching the parent. `has_marker`
        is False when this task_idx is absent from the sidecar, which the
        caller treats as an error (regenerate Phase 1 for this scene)."""
        override_path = self.cfg["solution"].get("target_override", None)
        if not override_path:
            return super(FetchMeshCuroboGORunOverride,
                         self).load_planner_target_from_config(env_idx)
        mapping = self._load_target_override(override_path)
        task_idx = int(self.get_task_idx())
        if task_idx not in mapping:
            return -1, False
        return int(mapping[task_idx]), True


# --- self-register so tasks/__init__.py stays untouched ------------------- #
try:
    from isaacgymenvs.tasks import isaacgym_task_map

    isaacgym_task_map["FetchMeshCuroboGORunOverride"] = FetchMeshCuroboGORunOverride
except Exception as _exc:  # pragma: no cover
    print("[GORunOverride] task registration deferred: {}: {}"
          .format(type(_exc).__name__, _exc))
