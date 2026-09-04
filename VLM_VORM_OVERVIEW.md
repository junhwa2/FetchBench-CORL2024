# VLM + VORM — New-Code Overview

This document is the map of **all new code** added for the VLM+VORM work,
spanning two repositories. It explains what each file is, how the pieces fit
together, how to run them, and the conventions that keep the upstream code
untouched.

> **No-edit rule.** No existing/upstream file is modified in either repo.
> - **FetchBench-CORL2024** (`/home/jo/HJ/FetchBench-CORL2024`): *new files only*.
>   New runtime tasks **self-register** into `isaacgym_task_map` on import, so
>   `isaacgymenvs/tasks/__init__.py` is never edited.
> - **vorm_pipeline** (`/home/jo/HJ/vorm_pipeline`): if an existing file would
>   need changing, it is **copied to `<name>_mj.py`** and only the `*_mj` copy is
>   edited. Originals are never touched.
>
> Verified: `git status` shows **0 modified tracked files** in FetchBench.

---

## 1. What this is

Two related capabilities, sharing one VLM core:

1. **VLM target grounding (Phase 1 / Phase 2).** A batch, offline measurement:
   a Vision-Language Model (Gemini Robotics-ER) is shown a rendered scene + a
   language instruction ("Pick up the telephone.") and must point at the named
   object. Phase 1 records + scores the grounding; Phase 2 runs the singulation
   planner on the VLM's chosen target using VORM collision predictions.

2. **VLM+VORM closed loop (ThinkGrasp-style).** An interactive loop where the
   **vorm_pipeline master drives the FetchBench sim** over a file-RPC:
   `observe → decide (sim-side VLM) → plan (GraspGen+IK+VORM) → execute → re-observe`.
   The VLM decides which object to grasp next — the target if visible, or an
   **occluder to remove** if the target is hidden. This reveal reasoning is the
   core contribution: with a fully-occluded target there is no point cloud, so
   only the VLM can name what to remove.

Both live behind the same `vlm_target` core (the Gemini client + the
seg→object resolver).

---

## 2. Architecture at a glance

### Phase 1 / Phase 2 (grounding, single process)

```
vlm_target_gen.py  (task=FetchMeshCuroboGORunVLM)
  └─ render RGB+seg → Gemini-ER points → resolve to object id → score
     → writes vlm_target_logs/<exp>/{summary.json, results.csv, targets.npz}

vlm_phase2.py      (task=FetchMeshCuroboGORunOverride)
  └─ reads targets.npz → runs stock singulation planner with VORM collision_pred

scripts/vlm_join_results.py
  └─ joins Phase-1 grounding + Phase-2 execution → instruction success rate
```

### Closed loop (two processes, two conda envs, file-RPC)

```
[vorm env, py3.10]  loop_runner_mj.py  (MASTER)                [FetchBench env, py3.8]
  reset ─────────────────────────────────────────────────────►  server_entry.py (SERVER)
  observe  ◄── rgb.png / pc.ply / qpos.npy ───────────────────   builds ONE Isaac Gym env,
  decide_goal(instruction) ──► sim-side Gemini-ER reveal VLM ─►   serves reset/observe/
     ◄── {goal_seg_id, action: grasp_target|remove_occluder} ──   decide_goal/execute
  plan: GraspGen → IK → VORM → RelocationPlannerMJ(root=goal)
  execute(goal, grasp_poses) ──► curobo reach+grasp+carry ────►  re-render → new obs
  (repeat until target_grasped)
                      file-RPC:  req/<id>.json + resp/<id>.json + payload/*
```

---

## 3. File inventory

### FetchBench-CORL2024

#### ① Closed loop — `InfiniGym/isaacgymenvs/tasks/fetch/vlm_closed_loop/`

A self-contained subpackage holding everything specific to the closed loop.

| File | LOC | Role |
|---|---:|---|
| `__init__.py` | 10 | Package doc / marker. |
| `sim_rpc.py` | 142 | **File-RPC transport, server side.** `RpcServer.serve_forever()` polls `req/<id>.ready`, dispatches to a handler, writes `resp/<id>.json` atomically (via `os.replace`) and touches `resp/<id>.ready` as the barrier. `_Layout` owns the `req/ resp/ payload/` path scheme. `json.dump(..., default=str)` so a stray array can never crash the serve loop. The matching **client half lives on the vorm master** (`sim_backend_mj.py`), kept separate so the two conda envs stay independent — this module is server-only. |
| `reveal_client.py` | 116 | **Reveal-reasoning VLM client.** `RevealERClient(GeminiERClient)` adds `decide(image, instruction)` with a prompt that makes the model choose ONE next action — `grasp_target` (point at the target) or `remove_occluder` (point at the object to move away first) — and return structured JSON `{visible_objects, target_found, action, point, reason}`. `target_found=false` demotes a `grasp_target` to `remove_occluder` so the loop never confidently grasps a mis-identified object. `build_reveal_client(cfg)` returns it for `backend=gemini & reveal=True`, else defers to the stock factory. |
| `server_task.py` | 446 | **`FetchMeshCuroboGORunVLMServer`.** Multiple-inherits `(FetchMeshCuroboGORunVLM, FetchPointCloudBase)` so the depth point-cloud engine (`cam_point_clouds`, `get_camera_data`) is built in the cooperative `__init__` alongside GORun's curobo stack. Borrows `gen_pc_from_camera`/`_get_seg_color`/`_filter_pc`/`save_pc` from `FetchMeshCuroboGO` as bound class attrs (so GO's heavy `__init__` never runs). Exposes the four RPC methods: `server_reset`, `server_observe` (renders rgb + segmented base-frame cloud + qpos), `server_decide_goal` (runs the reveal VLM), `server_execute` (injects the master's external grasps into `_obs_grasp`/`_obs_ids`/`_active_plan` and reuses the *unmodified* `_execute_plan_step`; returns a per-stage `diag`). Self-registers into `isaacgym_task_map`. |
| `server_entry.py` | 128 | **Hydra entry + serve loop.** Builds one env via the shared `build_vec_env`, wraps it in `_ServerApp` (serialises each observation to `rgb.png` / `pc.ply` / `qpos.npy` under `payload/` and dispatches RPC methods), then `RpcServer(...).serve_forever()`. `config_path="../../../config"` (relative to this deeper location). |

#### ② VLM core — `InfiniGym/isaacgymenvs/tasks/fetch/utils/vlm_target/`

Shared by both the grounding pipeline and the closed loop. Deliberately free of
Isaac Gym imports so it is unit-testable on its own.

| File | LOC | Role |
|---|---:|---|
| `__init__.py` | 42 | Public exports (clients + resolver helpers). |
| `gemini_er_client.py` | 460 | **Gemini Robotics-ER client via REST** (`requests`, not the SDK — the SDK needs py≥3.9). `GeminiERClient` with `point()`, `point_samples()` (self-consistency), `choose_index()` (set-of-mark). Retry/backoff, `finishReason`/`promptFeedback` surfacing. `OracleVLMClient` (`gt`/`random`) gives an API-free plumbing check + chance baseline. `build_client(cfg)` factory. Model default `gemini-robotics-er-2-preview`; key via `GEMINI_API_KEY`/`GOOGLE_API_KEY`. |
| `target_resolver.py` | 264 | **Point → object resolution.** `resolve_point` maps a VLM point (normalized `[y,x]` 0–1000) to a scene object via the seg mask (direct hit → window vote → nearest-centroid snap, with a `snapped` flag). `pick_camera` chooses the view, `visible_pixel_count` / `visible_object_old_ids` / `seg_value_name` are visibility utilities. Convention: `seg_pixel == old_id + 4`. |
| `hydra_boot.py` | 34 | **Shared hydra bootstrap** `build_vec_env(cfg, experiment_name)`: the identical seed / scene-wiring / 11-arg `isaacgymenvs.make` block the three entries used to duplicate. Lazy imports so it never forces an isaacgym import-order dependency. |

#### ③ Phase-1 / Phase-2 tasks + entries

| File | LOC | Role |
|---|---:|---|
| `tasks/fetch/fetch_mesh_curobo_go_run_vlm.py` | 653 | **Phase-1 task `FetchMeshCuroboGORunVLM`.** Overrides `solve()` only: settles the scene, renders RGB+seg, builds the instruction from the object's category (with a left/right qualifier when duplicates are visible), queries the VLM, resolves the point to an object, and records a strict-correct grounding log. Never touches the singulation planner. |
| `tasks/fetch/fetch_mesh_curobo_go_run_override.py` | 69 | **Phase-2 task `FetchMeshCuroboGORunOverride`.** Stock `FetchMeshCuroboGORun` plus an optional `solution.target_override` that reads the planner target from the Phase-1 `targets.npz` sidecar instead of mutating `task_config.npz`. |
| `isaacgymenvs/vlm_target_gen.py` | 216 | **Phase-1 entry.** Loops tasks, writes `summary.json` (target/point accuracy, chance level, Wilson CI), `results.csv`, and `targets.npz`. |
| `isaacgymenvs/vlm_phase2.py` | 92 | **Phase-2 entry.** Runs the override task and logs videos + `result.npy/.csv`. |

#### ④ Config — `InfiniGym/isaacgymenvs/config/task/`

| File | Role |
|---|---|
| `FetchMeshCuroboGORunVLM.yaml` | Phase-1 config: the full `solution.vlm.*` block (backend, model, `label_source=category`, `prompt_mode`, ambiguity scope, fallback, …). Inherits `FetchMeshCuroboGORun`. |
| `FetchMeshCuroboGORunOverride.yaml` | Phase-2 config: inherits `FetchMeshCuroboGORun`, adds `solution.target_override: null`. |
| `FetchMeshCuroboGORunVLMServer.yaml` | Closed-loop server: inherits `FetchMeshCuroboGORunVLM`, adds `reveal: True`, `server_cam_idx`, `env.cam.depth_min/max` (needed by the point-cloud engine), `pc_bound_option`/`pc_voxel_size`. |

#### ⑤ Docs / scripts

| File | Role |
|---|---|
| `VLM_SIM_RPC_SPEC.md` | Closed-loop **RPC protocol spec** (transport rules, the four methods, payload formats) + launch recipe + first-run verification checklist. |
| `VLM_VORM.md` | Phase-1/2 pipeline documentation. |
| `VLM_VORM_OVERVIEW.md` | **This file** — the whole-project file map. |
| `VLM_VORM_INTERNALS.md` | **Internals reference** below the surface: perception/point-cloud generation + filters, per-stage data flow & tensor formats, execute/`diag` success gates, grasp sources (GraspGen vs ACRONYM), env-setup traps, the 3-way benchmark. |
| `KNOWN_ISSUES.md` | Confirmed bugs/findings (exclude-id inversion, perception coverage, curobo `log_error`, root drift) + "all pre-fix results invalid". |
| `scripts/vlm_join_results.py` | Joins Phase-1 grounding + Phase-2 execution → instruction-level success rate + decomposition. |
| `scripts/check_object_id_alignment.py` | Diagnoses `old_id → category` alignment for a scene (guards against combo-actor index shifts). |

### vorm_pipeline (`vorm_pipeline/`, all `*_mj`)

| File | LOC | Role |
|---|---:|---|
| `search_mj.py` | 85 | `RelocationPlannerMJ(RelocationPlanner)` adds `search_for(infer_result, root_seg_id)` — builds the same KB but forces the plan **root = the VLM-named object** instead of ranking candidates. With a hidden target there are no grasps to rank it by, so the root can only come from the VLM. |
| `sim_backend_mj.py` | 190 | `SimBackend` abstract contract (`reset`/`observe`/`decide_goal`/`execute`/`close`) + `IsaacGymBackend`, a file-RPC **client** that mirrors `sim_rpc`'s protocol byte-for-byte and resolves the server's payload paths (loads `qpos.npy`, hands `pc.ply` to `perception.from_ply`). A future `RealRobotBackend` (ROS2) implements the same contract without touching the loop. |
| `loop_runner_mj.py` | 314 | `LoopRunnerMJ(Main)` — reuses `main1.py`'s stage construction (Perception/GraspGen/IKFilter/VORM), swaps in `RelocationPlannerMJ`, and drives the closed loop. Per step it widens the grasp set sent to the sim (all of the chosen object's grasps, KB-first, capped) so the sim's exact IK isn't starved. Writes per-task JSON + `summary.json` + `results.csv` (Wilson CI). |

---

## 4. How the pieces connect

### Shared identifiers & frames

- **Seg id** = the segmentation pixel value = `old_id + 4` (robot=1, table=2,
  scene=3, objects≥4). It is the single stable object handle shared across the
  RGB image, the seg image, the `.ply` `id` channel, and the `goal_seg_id` in
  the RPC.
- **Grasp poses** are 7-vecs `[x,y,z, qw,qx,qy,qz]` (wxyz), robot base frame —
  the same convention `obs_data['grasp']` uses, and both sim and vorm use
  `franka_r3.yml`, so the EEF frame matches.
- **Point cloud** (`pc.ply`) is written by the server's `save_pc` (open3d
  tensor with `colors` + `id`) and read by vorm's `perception.from_ply`.

### Closed-loop step (master ↔ server)

1. Master `reset(task_idx)` → server places the scene, returns instruction +
   ground-truth target (scoring only) + the first observation.
2. Master `observe()` → server renders `rgb.png`, the fused segmented base-frame
   `pc.ply`, and `qpos.npy`.
3. Master `decide_goal(instruction)` → the **sim-side** reveal VLM returns
   `goal_seg_id` + `action`.
4. Master runs GraspGen + IK + VORM on the observed cloud, then
   `RelocationPlannerMJ.search_for(root = goal_seg_id)`; it selects the first
   plan step's object and hands the sim that object's grasp candidates.
5. Master `execute(goal_seg_id, grasp_poses, release)` → server injects them and
   runs the unmodified `_execute_plan_step` (curobo reach → close → carry),
   re-renders, and returns success + `diag` + the new observation.
6. Repeat until `target_grasped` or the step budget is hit.

---

## 5. How to run

### Phase 1 / Phase 2 (single process, FetchBench env)

```bash
cd InfiniGym
export PYTHONPATH=$PWD:$PYTHONPATH
export ASSET_PATH=/path/to/asset_release
export GEMINI_API_KEY=...            # or backend=oracle for a no-API check

# Phase 1 — grounding
python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
    scene=benchmark_eval/RigidObjDesk_0 headless=True

# Phase 2 — plan on the VLM target with VORM predictions
python isaacgymenvs/vlm_phase2.py task=FetchMeshCuroboGORunOverride \
    scene=benchmark_eval/RigidObjDesk_0 headless=True \
    task.solution.pred_folder=<VORM> \
    task.solution.target_override=<abs>/vlm_target_logs/<exp>/targets.npz

# Join
python scripts/vlm_join_results.py --vlm-log InfiniGym/vlm_target_logs/<exp> \
    --sim-run InfiniGym/runs/<phase2 exp>
```

### Closed loop (two terminals, two conda envs)

> One server == one scene (Isaac Gym binds scene assets at construction). Restart
> the server per scene, and pair it **1:1** with a master run (the server tracks a
> monotonic request id).

**Terminal A — FetchBench sim server** (isaacgym env, py3.8):
```bash
cd InfiniGym
export PYTHONPATH=$PWD:$PYTHONPATH          # so `import isaacgymenvs` resolves
export ASSET_PATH=/path/to/asset_release
export GEMINI_API_KEY=...
export VLM_RPC_DIR=/tmp/vlm_rpc/desk0
python isaacgymenvs/tasks/fetch/vlm_closed_loop/server_entry.py \
    task=FetchMeshCuroboGORunVLMServer scene=benchmark_eval/RigidObjDesk_0 headless=True
# ready when the log prints "serving on ..." (~3–5 min init)
```

**Terminal B — vorm master** (vorm env, py3.10), same RPC dir:
```bash
cd /home/jo/HJ/vorm_pipeline
conda deactivate; conda deactivate; conda activate vorm    # ensure py3.10 on PATH
export LD_LIBRARY_PATH=/home/jo/anaconda3/envs/vorm/lib:$LD_LIBRARY_PATH   # optree/CXXABI
python -m vorm_pipeline.loop_runner_mj \
    --rpc-dir /tmp/vlm_rpc/desk0 --config config/pipeline_local.yaml \
    --num-tasks 5 --max-steps 4 --result-dir outputs/loop_desk
```

Results land in `--result-dir`: one JSON per task (instruction, per-step VLM
action/point/reason, execution `diag`, success) + `summary.json` + `results.csv`.

---

## 6. Conventions that keep upstream untouched

| Technique | Where | Why |
|---|---|---|
| **Self-registration** | every new task module does `isaacgym_task_map["Name"] = Class` on import | avoids editing `isaacgymenvs/tasks/__init__.py` |
| **Subclassing** | `...VLM(GORun)`, `...Override(GORun)`, `...VLMServer(GORunVLM, FetchPointCloudBase)` | reuse execution/render/planner logic without copying it |
| **Method binding** | server borrows `gen_pc_from_camera` etc. from `FetchMeshCuroboGO` as class attrs | get the methods without running GO's heavy `__init__` |
| **`*_mj` copies** | vorm side (`search_mj`, `sim_backend_mj`, `loop_runner_mj`) | change behavior without editing vorm originals |
| **Two-process file-RPC** | server (FetchBench) ↔ master (vorm) | keep the two incompatible conda envs (py3.8 isaacgym vs py3.10 GraspGen/VORM) independent |

---

## 7. Status (verified on RTX 4090)

- **Grounding pipeline**: works; oracle-gt ≈ 1.0, oracle-random ≈ chance, real
  Gemini grounding measurable. Improved reveal prompt raised Desk grounding
  3/5 → 4/5 (fixed "desktop"→stapler confident-misgrounding).
- **Closed loop**: runs fully end-to-end (VLM → GraspGen → IK → VORM → planner →
  sim grasp), reveal reasoning fires for real. **First success on Desk: task
  "Pick up the chaise" solved** (grasp + carry + hold verified). Multi-step
  re-observe loop confirmed.
- **Current bottleneck (updated 2026-08-31, see `KNOWN_ISSUES.md`)**: the earlier
  "grasp quality" reading was distorted by a **config bug** — `grasp.exclude_ids`/
  `search.scene_ids` were empty, so the empty-fallback excluded the *highest* seg
  id (usually the target) as "background" while grasping scene seg 3 as a phantom
  object. Every pre-fix result is invalid. After the fix (`exclude_ids: [1,2,3]`),
  three controls on Desk (25 tasks) localise the real bottleneck:
  | harness | grasps | perception | success |
  |---|---|---|---|
  | stock-GT (`FetchMeshCurobo`) | ACRONYM (full mesh) | none | **92%** |
  | GraspGen-only (`--graspgen-only`) | GraspGen | observed cloud | 12% |
  | closed loop (VLM+VORM) | GraspGen | observed cloud | 20% |
  VLM/VORM are **not** the bottleneck (removing them ≈ same). The dominant issue is
  **perception coverage** — the 2 observation cameras miss most objects, so the GT
  target has 0 points in the cloud in ~14/25 tasks and GraspGen never runs on it;
  secondary is **grasp feasibility** (GraspGen grasps that pass the master IK filter
  but fail the sim's exact curobo — the master-IK ≠ sim-IK gap). stock-GT hits 92%
  precisely because it uses the GT mesh+pose and does not depend on the cloud.
  Levers, in order: camera coverage (poses/count), then master-IK↔sim-IK alignment.
- **Benchmark note**: the current benchmark has ~0% fully-occluded targets by
  design; the reveal reasoning is most meaningful after regenerating scenes with
  genuinely hidden targets.

See also: `VLM_SIM_RPC_SPEC.md` (protocol), `VLM_VORM.md` (grounding pipeline),
`VLM_VORM_INTERNALS.md` (perception / data-flow / diag / env internals), and
`KNOWN_ISSUES.md` (confirmed bugs & findings).
