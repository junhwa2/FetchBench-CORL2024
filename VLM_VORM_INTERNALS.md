# VLM + VORM — Internals reference

The `OVERVIEW`/`SPEC`/`VLM_VORM` docs describe **what exists and how to run it**. This one
documents the **black boxes below that surface** — perception, the per-stage data flow, the
success/diag semantics, grasp sources, and the environment traps — so the pipeline can be
understood and debugged without re-reading the inherited FetchBench/vorm code. Every claim is
tagged with a `file:line` you can jump to; if code and doc disagree, trust the code and fix
this file.

Scope: the **closed loop** (`vlm_closed_loop` server + `vorm_pipeline` master). Cross-refs:
[VLM_VORM_OVERVIEW.md](VLM_VORM_OVERVIEW.md), [VLM_SIM_RPC_SPEC.md](VLM_SIM_RPC_SPEC.md),
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

---

## 1. Perception — how the point cloud is built (and why it misses objects)

This is the stage with the least prior documentation and the **dominant bottleneck**
(see KNOWN_ISSUES: ~14/25 GT targets get 0 points).

### 1.1 Cameras

Created in [fetch_base.py:685-697](InfiniGym/isaacgymenvs/tasks/fetch/fetch_base.py#L685):

- **N observation cameras** = `cfg.env.cam.num_cam` (here **2**), poses set per task from
  `loader.camera_poses` / the task_config npz `task_camera_pose (num_tasks, 2, 6)` (pos xyz +
  look-at xyz), re-applied at [fetch_base.py:923-927](InfiniGym/isaacgymenvs/tasks/fetch/fetch_base.py#L923).
- **+1 "vis" camera** at a fixed third-person pose `[-2,0,2.5] → [0,0,0.5]`
  ([fetch_base.py:694](InfiniGym/isaacgymenvs/tasks/fetch/fetch_base.py#L694)). **Diagnostic
  only** — a wide overview; the point cloud does NOT use it.
- Properties ([FetchBase.yaml](InfiniGym/isaacgymenvs/config/task/FetchBase.yaml#L75)):
  `width 640, height 480, hov 70°`. Depth clip `min 0.15, max 2.5` m.

So the cloud comes from **2 observation cameras** whose poses are baked into the benchmark task
file. Their coverage of the table is what determines which objects are observed.

### 1.2 `gen_pc_from_camera` — the segmented cloud + its 3 filters

[fetch_mesh_curobo_go.py:425-470](InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_go.py#L425).
`get_camera_data(segmented_ptd=True)` returns `{seg_id: points(N,3)}` (base frame), then it
keeps a segment only if it passes **all three**:

1. **not robot(1) / table(2)** — structure, never grasped.
2. **in `task_cand_obj_index`** — objects (seg≥4) whose `old_id` is not a task candidate are
   dropped. Candidate set: from the npz if present, else **auto-derived** by
   `get_obj_tasks` = every object except `combo_org*`/`*on_floor`, replicated to all tasks
   ([fetch_base.py:729-731](InfiniGym/isaacgymenvs/tasks/fetch/fetch_base.py#L729)). In
   `asset_release` the npz has **no** candidate key, so the derived set is used (≈ all objects,
   so this filter rarely drops a real target).
3. **non-empty** — `len(pts)==0` is skipped ([:452](InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_go.py#L452)).
   **This is where camera coverage bites:** an object outside the 2 cameras' frustum / occluded /
   cut at the frame edge has 0 points → absent from the cloud entirely (not even in `obs_ids`).

Output `.ply`: `{xyz, rgb, id}` with `id = seg_id`. Written by `save_pc`, read by vorm's
`perception.from_ply`.

### 1.3 Consequence

A GT target absent from the cloud (filter 3) never reaches GraspGen → `no_target_grasp`.
stock-GT (§4) is immune because it uses the GT mesh+pose, not the cloud. **Fix lever: camera
poses/count (`task_camera_pose`, `env.cam.num_cam`), not the filters.**

> Two *different* exclusion layers, don't confuse them:
> - **Server pc filter** (above) — drops robot/table/non-candidate/empty from the `.ply`.
> - **vorm grasp `exclude_ids`** (§2, grasp_gen) — drops robot/table/scene from *grasp
>   generation*. This one had the inverted-default bug (KNOWN_ISSUES), now `[1,2,3]`.

---

## 2. Data flow & formats (master side, per step)

Frames: object-centroid frame → robot base frame. Grasps are **7-vec `[x,y,z, qw,qx,qy,qz]`**
(position + **wxyz** quaternion). The whole chain lives in
[loop_runner_mj._plan_step](../vorm_pipeline/vorm_pipeline/loop_runner_mj.py).

| step | call | output (shape / keys) | frame |
|---|---|---|---|
| observe | `perception.from_ply(pc)` | `seg_pc {xyz(M,3), rgb, id(M,) = seg_id}` | base |
| grasp gen | `grasp.gen_obj_grasp_poses(seg_pc)` | `self.obj_grasp_poses {seg_id: (n,7)}` | **obj** frame |
| stack | `grasp.annotated_grasp_pose()` | `gp {grasp_poses(1,K,7) wxyz, grasp_targets(1,K)=seg_id}` | base |
| IK | `ikf.solve_ik(gp)` → `ikf.filter(...)` | `ik {grasp_poses(Pose 1,K), grasp_success(1,K) bool, grasp_pose_ik(1,K,dof), grasp_targets(1,K)=seg_id}` | base |
| VORM | `infer.run(ik, seg_pc, perc)` | `infer {collision(G,K) bool, target(G,)=new_id, grasp(G,7), obs_ids(K,)=seg_id}` | base |
| plan | `planner.search_for(infer, root_seg)` | `{plan:[(new_id, grasp_idx_list), ...], kb{new_id:…}, reason}` | — |
| send | `_plan_step` picks `plan[0]`, `grasp7 = infer['grasp'][sel]` | list of 7-vec | base |
| exec | `backend.execute(step_seg, grasp7.tolist(), release)` | RPC to sim | base |

Key facts to not trip on:
- **`gen_obj_grasp_poses`** skips a segment if `<8` points, if `seg_id ∈ exclude_ids`, or if
  GraspGen returns 0; raises `RuntimeError("no graspable objects found")` if none survive
  ([grasp_gen.py:155-210](../vorm_pipeline/vorm_pipeline/grasp_gen.py#L155)). The master catches
  that and aborts the *task*, not the run.
- **`infer.run` scores only IK survivors** — `G = #(grasp_success)`, so `infer['grasp']` are the
  IK-feasible grasp poses; VORM's collision head labels them, it does **not** change the poses
  ([inference.py:138-207](../vorm_pipeline/vorm_pipeline/inference.py#L138)).
- **Three id spaces:**
  - `seg_id` = `old_id + 4` (robot=1, table=2, scene=3, objects≥4).
  - `old_id` = object index in the scene (the candidate-index space).
  - `new_id` = index into `obs_ids` (0..K-1); `obs_ids[new_id] = seg_id`. `infer['target']` and
    `kb` keys are **new_id**; `plan_step` maps back with `obs_ids[obj_new]`.
- **GraspGen-only control** (`--graspgen-only`, [loop_runner_mj.py](../vorm_pipeline/vorm_pipeline/loop_runner_mj.py))
  extracts the target's grasps straight from `ik` (`grasp_success & grasp_targets==target_seg`),
  bypassing VORM/planner — the reference for "how to get a target's grasps without the KB".

---

## 3. Execute & success — what `diag` means

Injection: `server_execute` sets `_obs_grasp`/`_obs_ids`/`_active_plan` and runs the **unmodified**
`_execute_plan_step` ([server_task.py:422-485](InfiniGym/isaacgymenvs/tasks/fetch/vlm_closed_loop/server_task.py#L422)).
A step must pass **four gates in order**
([fetch_mesh_curobo_go_run.py:1102-1167](InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_go_run.py#L1102)):

| gate | diag field | fail reason |
|---|---|---|
| ① IK + motion to a grasp | `grasp_plan_success` | `no_ik` / `grasp_motion_fail` |
| ② motion to carry to free space | `fetch_plan_success` | `fetch_motion_fail` |
| ③ object actually moved | `dbg_disp > 0.05 m` | `no_movement(disp=…)` |
| ④ object stayed in gripper | `dbg_dist_drift < 0.10 m` | `slipped(drift=…)` |

- `dbg_disp` = object travel from grasp-close to post-carry (rules out no-op).
- `dbg_dist_drift` = `|dist(obj,eef)_carry − dist(obj,eef)_close|` (grows if it slips/drops).
  Tolerance = `solution.execute_dist_drift_tol` (0.10).
- `execute_success = moved ∧ held`. Then `target_grasped = ok ∧ goal_is_target ∧ ¬release`
  ([server_task.py:471](InfiniGym/isaacgymenvs/tasks/fetch/vlm_closed_loop/server_task.py#L471)) →
  the task's success condition.
- A hard crash inside execute is caught as `diag.exec_exception` (loop survives). Note the curobo
  `log_error` bare-`raise` bug can mask the real message here (KNOWN_ISSUES).

Master abort reasons (per-task JSON `abort`): `vlm_no_goal` (VLM returned no point),
`root_ungraspable`/`bc_failed`/`root_seg_not_in_obs_ids` (planner dead-ends, no sim attempt),
`exec_failed` (sim attempted, gate ①–④ failed), `no_graspable_objects` (empty grasp gen).

---

## 4. Grasp sources — GraspGen vs GT (ACRONYM)

| | closed loop | stock-GT (`FetchMeshCurobo`) |
|---|---|---|
| source | **GraspGen** (learned), on the observed point cloud, runtime | **ACRONYM** poses per object mesh |
| storage | none (predicted) | `asset_release/benchmark_objects/<Cat>/<hash>/grasp_poses.h5` |
| filtering | none | Isaac Gym force-closure label |

GT path ([load_utils.py:130-150](InfiniGym/isaacgymenvs/tasks/fetch/utils/load_utils.py#L130),
[fetch_base.py:424-445](InfiniGym/isaacgymenvs/tasks/fetch/fetch_base.py#L424)): loads ACRONYM
`T` (4×4) + `acronym_label` (FleX) + `isaac_label_default/cvx`. `load_asset_grasp_poses` keeps
grasps by the label(s) selected in `solution.grasp_label` — the default `FetchMeshCurobo.yaml`
uses `gripper_type: cvx, use_isaac_force_label: True` → ACRONYM poses that pass Isaac
force-closure. This is why stock-GT reaches 92%: curated, physics-validated grasps on the full
known mesh, no perception dependency.

---

## 5. Environment setup — the traps (all hit this session)

Both processes are separate conda envs; the interactive shell has state a background/non-login
shell lacks. Checklist:

**Server (FetchBench, py3.8):**
- `conda activate FetchBench` (or `/home/jo/anaconda3/envs/FetchBench/bin/python`).
- `export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release` (in `.bashrc` line 176 —
  **not** read by non-interactive shells; inject explicitly).
- `export PYTHONPATH=$PWD` from `InfiniGym`.
- `export VLM_RPC_DIR=/tmp/vlm_rpc/desk0` — **required**; server SystemExits without it.
- `export GEMINI_API_KEY=...` (in `.bashrc` line 178, same non-interactive caveat) — needed for
  the server-side VLM; without it every `decide_goal` → `vlm_no_goal`.
- `headless=True`.

**Master (vorm, py3.10):**
- Use `/home/jo/anaconda3/envs/vorm/bin/python` **by absolute path** — the shell's `python`
  resolves to FetchBench's even when CONDA says base; `conda activate vorm`/`conda run` may not
  switch. (Verified: env python is 3.10.20.)
- `unset PYTHONPATH` (drop the ROS leak); `export LD_LIBRARY_PATH=/home/jo/anaconda3/envs/vorm/lib:$LD_LIBRARY_PATH`.
- `conda activate.d` scripts reference unset vars → keep `set +u` if scripting.

**Both / general:**
- `PYTHONUNBUFFERED=1` (or `python -u`) when redirecting to a log, or Python prints (incl.
  "serving on") stay buffered and never appear.
- **`rm -rf $VLM_RPC_DIR` before starting the server.** Stale `req/*` files make the server
  replay old requests and desync the id counter (the client wipes on start; the server does not).
- One server == one scene == one master run (monotonic id). Restart both + wipe the dir to re-run.

A working end-to-end launcher (server → wait for "serving on" → master → cleanup) is in this
session's scratchpad `rerun_objdiag.sh`; the control harnesses are `stock_gt_eval.py` (GT) and
`loop_runner_mj --graspgen-only` (GraspGen, no VLM/VORM).

---

## 6. The 3-way benchmark (why the numbers are what they are)

Same 25 Desk tasks, after the `exclude_ids` fix (pre-fix results are void — KNOWN_ISSUES):

| harness | target | grasps | perception | success |
|---|---|---|---|---|
| stock-GT | GT (task_obj_index) | ACRONYM+force-label | none (GT mesh/pose) | **23/25 = 92%** |
| GraspGen-only | GT | GraspGen | observed cloud | 3/25 = 12% |
| closed loop | VLM | GraspGen | observed cloud | 5/25 = 20% |

Reading: VLM/VORM are not the bottleneck (12% ≈ 20%). GraspGen-only 12% = **14 target-not-in-cloud**
(perception, §1.3) + **8 observed-but-sim-rejected** (feasibility, master-IK ≠ sim-IK) + 3 success.
Ceiling is 92%; the whole gap is perception coverage then grasp feasibility.
