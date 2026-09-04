# Phase 1 of VLM+VORM: pick each task's target with a VLM, score the grounding,
# and write a targets.npz sidecar that Phase 2 consumes.
#
# Usage:
#   cd InfiniGym
#   export ASSET_PATH=/path/to/asset_release_v1.3   # with Task symlink set
#
#   # (a) plumbing check - must give target_accuracy ~1.0
#   python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
#       scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
#       task.solution.vlm.backend=oracle task.solution.vlm.oracle_mode=gt
#
#   # (b) CHANCE-LEVEL BASELINE - run this before reporting any Gemini number
#   python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
#       scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
#       task.solution.vlm.backend=oracle task.solution.vlm.oracle_mode=random
#
#   # (c) real Gemini Robotics-ER
#   export GEMINI_API_KEY=...
#   python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
#       scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True
#
# Then Phase 2 - WITHOUT mutating the benchmark dataset (no-edit subclass):
#   python isaacgymenvs/vlm_phase2.py task=FetchMeshCuroboGORunOverride \
#       scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
#       task.solution.obs_folder=<...>_sim/benchmark_eval \
#       task.solution.pred_folder=vorm_obstruction_260516 \
#       task.solution.target_override=<abs path>/vlm_target_logs/<exp>/targets.npz
#
# Finally join the two so success is scored against the INSTRUCTION:
#   python scripts/vlm_join_results.py \
#       --vlm-log InfiniGym/vlm_target_logs/<exp> \
#       --sim-run InfiniGym/runs/<phase2 exp>

import hydra
from omegaconf import DictConfig, OmegaConf

import os
import csv
import json
from collections import Counter
from datetime import datetime

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np

from isaacgymenvs.tasks.fetch.utils.vlm_target.hydra_boot import build_vec_env

# Importing the task module self-registers FetchMeshCuroboGORunVLM into
# isaacgym_task_map - no edit to isaacgymenvs/tasks/__init__.py required.
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo_go_run_vlm import (  # noqa: F401
    FetchMeshCuroboGORunVLM,
)


def _wilson_ci(k, n, z=1.96):
    """95% Wilson interval. 60 tasks on one scene is a small sample; reporting
    a bare point estimate invites reading noise as signal (the old
    normalize_category True-vs-False comparison differed by a single task)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / float(n)
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def _summarize(logs):
    n = len(logs)
    if n == 0:
        return {"num_tasks": 0}

    def cnt(pred):
        return sum(1 for l in logs if pred(l))

    n_correct = cnt(lambda l: l.get("correct"))                 # STRICT
    n_correct_lenient = cnt(lambda l: l.get("correct_lenient"))  # legacy
    n_resolved = cnt(lambda l: l.get("vlm_chosen_old") is not None)
    n_fallback = cnt(lambda l: l.get("used_fallback"))
    n_error = cnt(lambda l: l.get("vlm_error"))
    n_snapped = cnt(lambda l: l.get("snapped"))
    n_point_on_gt = cnt(lambda l: l.get("point_on_gt"))
    n_point_on_cand = cnt(lambda l: l.get("point_on_candidate"))

    # Review fix #5 - condition on the instructed object being on screen.
    vis = [l for l in logs if l.get("target_visible")]
    n_vis = len(vis)
    n_vis_correct = sum(1 for l in vis if l.get("correct"))

    # Review fix #4 - chance level for a uniform pick over visible candidates.
    chance = float(np.mean([1.0 / max(int(l.get("n_candidates_visible") or 1), 1)
                            for l in logs]))

    lo, hi = _wilson_ci(n_correct, n)
    vlo, vhi = _wilson_ci(n_vis_correct, n_vis)

    return {
        "num_tasks": n,
        # ---- headline: strict grounding accuracy -------------------------
        # A fallback (API error / unresolvable answer) counts as WRONG.
        "target_accuracy": n_correct / float(n),
        "target_accuracy_ci95": [lo, hi],
        # ---- what the VLM literally pointed at, before any snapping -------
        "point_accuracy": n_point_on_gt / float(n),
        "point_on_candidate_rate": n_point_on_cand / float(n),
        "snap_rate": n_snapped / float(n),
        # ---- conditioned on visibility -----------------------------------
        "num_target_visible": n_vis,
        "target_visible_rate": n_vis / float(n),
        "accuracy_on_visible": (n_vis_correct / float(n_vis)) if n_vis else None,
        "accuracy_on_visible_ci95": [vlo, vhi] if n_vis else None,
        # ---- baseline to compare against ---------------------------------
        "chance_level": chance,
        "mean_candidates_visible": float(np.mean(
            [int(l.get("n_candidates_visible") or 0) for l in logs])),
        # ---- pipeline health ---------------------------------------------
        "resolve_rate": n_resolved / float(n),
        "fallback_rate": n_fallback / float(n),
        "api_error_rate": n_error / float(n),
        "resolve_method_hist": dict(Counter(l.get("resolve_method") for l in logs)),
        # ---- legacy number, kept only for comparison ---------------------
        "target_accuracy_lenient_DEPRECATED": n_correct_lenient / float(n),
    }


def _write_targets_npz(out_dir, logs, scene):
    """Sidecar consumed by Phase 2 via task.solution.target_override.

    Replaces the old in-place rewrite of $ASSET_PATH/.../task_config.npz, which
    was order-dependent (any later GT-mode run clobbered it), not parallel-safe,
    and destroyed the benchmark ground truth it was scored against.
    """
    rows = [l for l in logs if l.get("chosen_old") is not None]
    if not rows:
        print("[vlm_target_gen] no resolved targets - targets.npz not written")
        return None
    path = os.path.join(out_dir, "targets.npz")
    np.savez(
        path,
        scene=np.array(str(scene)),
        task_idx=np.array([int(l["task_idx"]) for l in rows], dtype=np.int64),
        target_old_id=np.array([int(l["chosen_old"]) for l in rows], dtype=np.int64),
        target_label=np.array([str(l.get("chosen_label", "")) for l in rows],
                              dtype=object),
        correct=np.array([bool(l.get("correct")) for l in rows]),
        used_fallback=np.array([bool(l.get("used_fallback")) for l in rows]),
        gt_target_old=np.array([int(l["gt_target_old"]) for l in rows], dtype=np.int64),
    )
    print("[vlm_target_gen] wrote {} ({} tasks)".format(path, len(rows)))
    return path


def _write_summary(out_dir, logs, scene, cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    summary = _summarize(logs)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "tasks": logs}, f, indent=2, default=str)

    cols = ["task_idx", "scene", "instruction", "gt_target_old",
            "vlm_chosen_old", "chosen_old", "chosen_label",
            "correct", "correct_lenient", "point_on_gt", "raw_old_id",
            "raw_seg_name", "snapped", "resolve_method", "used_fallback",
            "target_visible", "gt_visible_px", "n_candidates",
            "n_candidates_visible", "cam_idx", "vlm_error"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for l in logs:
            w.writerow(l)

    _write_targets_npz(out_dir, logs, scene)

    # Config snapshot - model / temperature / thinking_budget all change the
    # result, so the run is not interpretable without them.
    if cfg is not None:
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    print("[vlm_target_gen] ---- summary ----")
    for k, v in summary.items():
        print("    {:<38} {}".format(k, v))
    acc, chance = summary.get("target_accuracy"), summary.get("chance_level")
    if acc is not None and chance:
        print("    {:<38} {:.3f}x chance".format("relative to chance", acc / chance))
    print("[vlm_target_gen] wrote {}/summary.json, results.csv, targets.npz"
          .format(out_dir))


@hydra.main(version_base="1.1", config_name="config", config_path="./config")
def launch(cfg: DictConfig):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = "{}_{}_{}_{}".format(
        cfg.scene.name, cfg.task.name, cfg.task.prefix, time_str)

    out_dir = os.path.join("vlm_target_logs", experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    vec_env = build_vec_env(cfg, experiment_name)

    logs = []
    for i in range(cfg.scene.num_tasks):
        vec_env.reset_task(i)
        _, log = vec_env.solve()
        if log:
            logs.append(log)

    _write_summary(out_dir, logs, cfg.scene.scene_list[0], cfg=cfg)

    vec_env.exit()
    exit()


if __name__ == "__main__":
    launch()
