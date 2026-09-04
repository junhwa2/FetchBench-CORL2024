# VLM+VORM ↔ Sim : File-RPC Interface Spec (v0)

Closed-loop (ThinkGrasp-style) integration between **vorm_pipeline** (master) and the
**FetchBench** Isaac Gym sim (server), over a **file-based local RPC**.

- **Master** = `vorm_pipeline/loop_runner_mj.py` (its own **py3.10** conda env, `vorm`).
  Orchestrates the observe → decide → plan → execute loop. Runs GraspGen + IK + VORM +
  RelocationPlanner on the observed point cloud.
- **Server** = `InfiniGym/isaacgymenvs/tasks/fetch/vlm_closed_loop/server_entry.py` (FetchBench py3.8 isaacgym env).
  Builds the env **once**, serves requests until `close`. Owns rendering, the sim-side
  **VLM** (`FetchMeshCuroboGORunVLM` → Gemini-ER), and grasp execution (curobo, from GORun).

Two processes, two conda envs, one shared **session directory**. No network, no new deps.
The VLM lives on the server side (it needs the rendered RGB + seg); VORM/planner live on
the master side (they need the observed point cloud). This matches:
"(b) vorm_pipeline이 sim을 구동" + "VLM은 sim 쪽" + sim-first.

---

## 1. Transport — single-in-flight request/response over files

One shared directory (`$VLM_RPC_DIR`, passed to **both** processes via CLI/env). Requests
are strictly serial: the master issues one request and **blocks** until its response, so a
monotonic integer `id` (starting at 0) fully identifies each exchange.

```
$VLM_RPC_DIR/
  req/
    <id>.json          # command (written by master, appears atomically via rename)
    <id>.ready         # zero-byte flag, created LAST by master  -> "request complete"
  resp/
    <id>.json          # result   (written by server, appears atomically via rename)
    <id>.ready         # zero-byte flag, created LAST by server  -> "response+payload complete"
  payload/
    obs_<id>_rgb.png   # large binaries referenced by resp JSON (relative paths)
    obs_<id>_pc.ply
    obs_<id>_qpos.npy
    ...
```

### Atomicity & synchronization (the only two rules that matter)

1. **Atomic JSON publish.** Write `X.json.tmp`, `os.replace()` → `X.json`. Same filesystem,
   so the reader never sees a half-written JSON.
2. **`.ready` is the barrier.** The producer writes *all* payload files, then the `.json`,
   then finally touches `.ready`. The consumer waits **only** on `.ready`; once it exists,
   every referenced file is guaranteed complete. Payload paths in JSON are **relative to
   `$VLM_RPC_DIR`**.

### Master loop (client)

```
id = 0
def call(method, **kw):
    global id
    write_atomic(f"req/{id}.json", {"v":0, "id":id, "method":method, **kw})
    touch(f"req/{id}.ready")
    wait_for(f"resp/{id}.ready", timeout=RESP_TIMEOUT)   # poll every POLL_SEC
    resp = read_json(f"resp/{id}.json")
    id += 1
    if not resp["ok"]: raise RpcError(resp["error"], resp.get("traceback"))
    return resp
```

### Server loop (serve)

```
next_id = 0
while True:
    wait_for(f"req/{next_id}.ready", timeout=None)       # poll every POLL_SEC
    req = read_json(f"req/{next_id}.json")
    try:
        result = dispatch(req)                            # may write payload/*
        resp = {"v":0, "id":next_id, "ok":True, **result}
    except Exception as e:
        resp = {"v":0, "id":next_id, "ok":False,
                "error":str(e), "traceback":format_exc()}
    write_atomic(f"resp/{next_id}.json", resp)
    touch(f"resp/{next_id}.ready")
    if req["method"] == "close": break
    next_id += 1
```

### Constants (shared)
- `POLL_SEC = 0.02` (file poll interval — irrelevant vs. per-step seconds of sim work)
- `RESP_TIMEOUT = 600` s default (curobo + grasp execution can be slow); `null` = wait forever.
- `v = 0` protocol version on every message; bump on any breaking change.

### Lifecycle / cleanup
- Master **creates** a fresh `$VLM_RPC_DIR` (empty `req/ resp/ payload/`) before launching
  the server, and removes it on clean exit.
- Server writing `resp/<id>` for an `id` whose `req` it didn't see ⇒ hard error (desync).
- On master timeout ⇒ assume server dead: read `resp/<id>.json` if present for a partial
  error, else raise; do not silently continue.

---

## 2. RPC methods (4 + close)

Poses are in the **robot base frame**, meters, z-up — the same frame FetchBench writes the
point cloud in and that GraspGen/curobo consume. Grasp poses are **7-vecs**
`[x,y,z, qw,qx,qy,qz]` (position + wxyz quaternion), the format `infer_result["grasp"]` and
the sim's IK use; the reserved `place_pose` is a 4×4 row-major flattened length-16 matrix.
`seg_id` is the **segmentation pixel value**
(`old_id + 4`; robot=1, table=2, scene=3, objects≥4) — the single stable object handle
shared across RGB, seg image, and `.ply` `id` channel.

### `reset` — start a task
```
req : {method:"reset", task_idx:int}
resp: {ok, instruction:str, target_seg_id:int, num_tasks:int, obs:<Obs>}
```
Scene is fixed at server launch (one FetchBench hydra invocation = one scene, matching how
GORun already runs). `target_seg_id` is ground truth — for scoring only; **never** given to
the VLM. `instruction` is the benchmark's "Pick up the {label}." (built by GORunVLM).

### `observe` — render current state
```
req : {method:"observe"}
resp: {ok, obs:<Obs>}
```
`<Obs>` object:
```
{
  "step": int,
  "rgb":  "payload/obs_<id>_rgb.png",       # HxWx3 uint8
  "pc":   "payload/obs_<id>_pc.ply",        # camera-observed seg cloud: xyz,rgb,id
  "qpos": "payload/obs_<id>_qpos.npy",      # float32 [7] arm joints
  "cam":  {"K":[9], "T_base_cam":[16], "width":W, "height":H}
}
```
The `.ply` is the **task-view partial** cloud (per user: "카메라 관측 점군, 씬 부분관측"),
directly loadable by vorm's existing `perception.from_ply` (xyz/rgb/id).

### `decide_goal` — sim-side VLM picks the next object to grasp
```
req : {method:"decide_goal", instruction:str}
resp: {ok, goal_seg_id:int, is_target:bool, action:"grasp_target"|"remove_occluder",
       vlm_point:[y,x], reason:str, target_visible:bool, snapped:bool}
```
Runs `FetchMeshCuroboGORunVLM`'s Gemini-ER call on the **current** rendered RGB.
- If the target is visible & the VLM points at it ⇒ `goal_seg_id = target`, `is_target=True`.
- If the target is fully occluded ⇒ the VLM reasons which **occluder** to remove and points
  at it ⇒ `goal_seg_id = occluder`, `is_target=False`. **This reveal reasoning is the whole
  contribution** — VORM cannot do it (no point cloud for the hidden target).
- `vlm_point` = raw normalized [y,x]∈[0,1000] (ER convention); `snapped` = point had to be
  snapped to nearest seg centroid.

### `execute` — plan + execute a grasp on the master's chosen grasp candidates
```
req : {method:"execute", goal_seg_id:int,
       grasp_poses:[[x,y,z,qw,qx,qy,qz], ...],
       release:bool | null, place_pose:[16] | null}
resp: {ok, success:bool, grasped_seg_id:int, target_grasped:bool,
       done:bool, step_failure:str | null, diag:{...}, obs:<Obs>}
```
Master sends a **list of candidate grasps** (7-vec, robot base, wxyz) for `goal_seg_id`,
computed from **its own** GraspGen+IK+VORM+RelocationPlanner on the observed cloud; the sim
IK-filters them and executes the **first feasible** one (curobo reach → close → carry), then
re-renders. `release` defaults to drop-aside for an occluder and keep-holding for the target;
`place_pose` is **reserved** (v0 uses the sim's free-space drop). `target_grasped` = the
benchmark's success condition on the GT target; `done=True` when solved. `step_failure`/`diag`
give the per-stage failure (`no_ik` / `grasp_motion_fail` / `fetch_motion_fail` / slip).

### `close` — shut down
```
req : {method:"close"}
resp: {ok}
```
Server tears down the env and exits after replying.

---

## 3. Closed loop (master, ThinkGrasp-style)

```
reset(task_idx) -> obs, instruction
repeat until done or step budget:
    d = decide_goal(instruction)                     # sim-side VLM on current RGB
    grasp, place = plan_mj(obs.pc, d.goal_seg_id)    # GraspGen+IK+VORM+RelocationPlanner_mj
    r = execute(d.goal_seg_id, grasp, place)         # sim executes, re-renders
    obs = r.obs
    if r.target_grasped: success; break
close()
```
`RelocationPlannerMJ` (in `search_mj.py`, a copy-then-edit of `search.py`) forces the plan
**root = the VLM-named goal** instead of re-deriving the target, so the VLM's reveal decision
drives the removal order. When the target is invisible there are no grasps for it, so the
loop root can only come from the VLM — exactly the gap this design fills.

---

## 4. Ownership / no-edit compliance

| Side | New files (only) |
|---|---|
| FetchBench | `tasks/fetch/vlm_closed_loop/` (sim_rpc + reveal_client + server_task + server_entry) — reuses existing `FetchMeshCuroboGORun` (execute/render) + `FetchMeshCuroboGORunVLM` (decide_goal). Self-registers; `tasks/__init__.py` untouched. |
| vorm_pipeline | `loop_runner_mj.py`, `sim_backend_mj.py` (file-RPC client + abstract `SimBackend` so a `RealRobotBackend`/ROS2 can slot in later), `search_mj.py` (`RelocationPlannerMJ`, root=goal). Copies of any file needing change get the `*_mj` suffix; originals never edited. |

Abstract `SimBackend` interface = the four methods above; `IsaacGymBackend` implements them
over file-RPC now, `RealRobotBackend` (ROS2) later — swap without touching the loop.

---

## 5. How to run (two terminals, two conda envs)

**Terminal A — FetchBench sim server** (isaacgym env):
```
cd InfiniGym
export PYTHONPATH=$PWD:$PYTHONPATH             # so `import isaacgymenvs` resolves
export ASSET_PATH=/path/to/asset_release
export GEMINI_API_KEY=...                        # backend=gemini
export VLM_RPC_DIR=/tmp/vlm_rpc/desk0            # shared session dir
python isaacgymenvs/tasks/fetch/vlm_closed_loop/server_entry.py \
    task=FetchMeshCuroboGORunVLMServer \
    scene=benchmark_eval/RigidObjDesk_0 headless=True
```

**Terminal B — vorm master** (RoboStack env), same RPC dir:
```
cd /home/jo/HJ/vorm_pipeline
python -m vorm_pipeline.loop_runner_mj --rpc-dir /tmp/vlm_rpc/desk0 \
    --config config/pipeline_local.yaml --max-steps 8
```
The master creates a fresh `--rpc-dir`; the server must point `VLM_RPC_DIR` at the same path.
One server == one scene; run the pair once per scene.

### First-run things to verify (untested cross-env assumptions)
1. **Grasp frame** — the loop sends `infer_result["grasp"]` 7-vecs `[x,y,z,qw,qx,qy,qz]`
   (robot base) straight to the sim's IK, assuming the same EEF-grasp convention as
   `obs_data['grasp']`. If IK/execution fails wholesale, the GraspGen grasp frame differs
   and needs a fixed offset before `execute`.
2. **`.ply` contract** — the server writes with `save_pc` (open3d tensor, `id` attribute);
   `perception.from_ply` must read `xyz/rgb/id`. Co-designed, but confirm on step 0.
3. **`is_target` proxy** — `server_decide_goal` marks is_target by category match to the
   instructed category (never the GT id). A dedicated VLM prompt outputting
   target-vs-occluder is the intended upgrade.
4. **Arm re-observation** — after an occluder removal the arm re-plans from its current
   pose; if it occludes the camera for the next `observe`, add a home reset in
   `server_execute`.
5. **Single-view occlusion** — `gen_pc_from_camera` fuses all sensor cameras while the VLM
   sees `server_cam_idx` only. For strictly single-view full occlusion, constrain the pc to
   one camera when regenerating the benchmark.

### Files (all new / `*_mj`, nothing existing edited)
| Repo | File |
|---|---|
| FetchBench | `tasks/fetch/vlm_closed_loop/sim_rpc.py` (transport, verified) |
| FetchBench | `tasks/fetch/vlm_closed_loop/reveal_client.py` (reveal-reasoning VLM client) |
| FetchBench | `tasks/fetch/vlm_closed_loop/server_task.py` (`FetchMeshCuroboGORunVLMServer`) |
| FetchBench | `tasks/fetch/vlm_closed_loop/server_entry.py` (hydra entry + serve loop) |
| FetchBench | `config/task/FetchMeshCuroboGORunVLMServer.yaml` |
| vorm | `search_mj.py` (`RelocationPlannerMJ.search_for`, root=goal) |
| vorm | `sim_backend_mj.py` (`SimBackend` + `IsaacGymBackend` file-RPC client) |
| vorm | `loop_runner_mj.py` (`LoopRunnerMJ`, the closed loop) |
```

