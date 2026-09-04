# Known issues — VLM+VORM closed loop

## [FIXED 2026-08-26] Structure/object seg exclusion was inverted → every run invalid

**Symptom.** On `benchmark_eval/RigidObjDesk_0` (DeskSceneFactory_48), 25-task closed-loop
runs gave ~1/25 success with ~68% of tasks aborting `root_ungraspable`.

**Root cause.** `vorm_pipeline/config/pipeline_local.yaml` had `grasp.exclude_ids: []` and
`search.scene_ids: []`. The empty-list fallback in `grasp_gen.py` / `search_mj.py` excludes
the **highest** seg id in the observed cloud, assuming it is the wall/background. But the RPC
seg convention (VLM_SIM_RPC_SPEC.md) is **robot=1, table=2, scene=3, objects≥4** — structure
is the **lowest** ids. The point cloud is table-cropped, so the only structure present is
`seg 3`, and the highest id is usually a real object (often the VLM's target).

Two symptoms, both confirmed from `[objdiag]` logs:

1. **Target dropped.** 17/18 `root_ungraspable` tasks had `goal_seg == max(seg)` → the target
   was excluded from grasp generation (`gg=0`, absent from KB) even though the VLM grounded it
   correctly.
2. **Phantom object.** `seg 3` (scene structure) was present with `gg>0, kb=graspable` in
   **26/26** planning steps → the scene was grasped as a fake object and injected into the KB,
   the VORM collision matrix, and the planner's `blocked_by` edges.

**Fix.** Set the documented structure ids explicitly:
`grasp.exclude_ids: [1, 2, 3]` and `search.scene_ids: [1, 2, 3]`. Config-only; no code change.

**⚠️ Every closed-loop result produced before this fix is invalid** — the collision
graph contained a phantom structure-object in *every* task, so even the occasional "success"
(e.g. task 1) was structurally wrong. Do **not** cite pre-fix numbers (the 1/25, the
root_ungraspable breakdowns, any objdiag counts) as pipeline performance.

**Not the cause** (ruled out during diagnosis): furniture being out-of-scope, IK
unreachability, GraspGen model failing (0 "returned 0 grasps" logs), VLM mis-grounding, and
VLM↔VORM ordering — all downstream of, or unrelated to, this exclusion bug.

## [OPEN] Bottleneck is our grasp selection, not the benchmark (stock-GT proof)

After the exclusion fix the closed loop scores **5/25 (20%)**, dominated by `exec_failed`
(sim curobo `no_ik` / `grasp_motion_fail` on the grasps we send). A stock-GT baseline —
`FetchMeshCurobo` (GT annotated grasps + the **same** curobo executor, no VLM/VORM) on the
same 25 tasks — scores **23/25 (92%)**. Per-task cross-reference:

- **18/25 FIXABLE**: we fail, stock-GT solves. (14 = grasp selection → our grasps pass the
  master IK filter but sim curobo rejects them = **master-IK ≠ sim-IK gap**; 3 = `vlm_no_goal`
  at a multi-step step-1; 1 = residual `root_ungraspable`.)
- **2/25 genuinely hard** (both fail): Desktop, Mirror (large / flat objects).
- **5/25 already solved.**

So the objects are graspable and the executor is capable — the ceiling is ~92% and the entire
gap is on our side. **Next action:** close the master-IK vs sim-IK gap (align the master IK
filter's curobo config/collision world with the sim executor, and/or widen the grasp set sent
to the sim), not scene/benchmark changes. Baseline harness: `InfiniGym/isaacgymenvs/stock_gt_eval.py`
(new file; headless subclass of FetchMeshCurobo, base untouched).

## [OPEN] Perception coverage: the observed point cloud misses most objects (3-way result)

Three controls on the same 25 tasks (all GT target where applicable):

| harness | target | grasps | perception | success |
|---|---|---|---|---|
| stock-GT (`FetchMeshCurobo`) | GT | ACRONYM (full mesh) | none (uses sim mesh+pose) | **23/25 = 92%** |
| GraspGen-only (`--graspgen-only`) | GT | GraspGen (point cloud) | observed cloud | **3/25 = 12%** |
| closed-loop | VLM | GraspGen | observed cloud | 5/25 = 20% |

GraspGen-only decomposition: **14/25 = GT target not present in the observed point cloud**
(no points → GraspGen never runs on it), **8/25 = target observed, GraspGen made ~100 grasps,
but sim curobo rejected them** (no_ik/grasp_motion_fail/slip), **3/25 success**.

Conclusions:
- **VLM/VORM are NOT the bottleneck** — removing them (GraspGen-only) gives 12% ≈ the 20%
  closed-loop (CIs overlap).
- **Dominant bottleneck = perception coverage.** The depth cloud captures only ~1-2 objects per
  scene (segs {4,5,6,13} in this scene); GT targets with higher seg ids (7,8,10,11,15,16,17) are
  never observed. stock-GT hits 92% precisely because it grasps the GT mesh/pose directly and does
  not depend on the observed cloud. Investigate camera count/viewpoint, frustum, depth clipping,
  crop/downsample in the pc pipeline.
- **Secondary = grasp feasibility.** Of the 11 tasks whose target IS observed, only 3 succeed;
  GraspGen's grasps mostly fail the sim's exact curobo (the master-IK ≠ sim-IK gap).

Control harnesses (new files / additive): `InfiniGym/isaacgymenvs/stock_gt_eval.py`,
`vorm_pipeline/loop_runner_mj.py --graspgen-only`.

## [OPEN] curobo `log_error` swallows the real message

`third_party/curobo/src/curobo/util/logger.py:45` does a bare `raise` after `logger.error`,
intended for in-except use, but curobo calls `log_error("msg")` outside any except in ~73
sites. When it fires with no active exception → `RuntimeError: No active exception to reraise`
and the real message is lost (only logged to the `curobo` logger). Intermittent; did not fire
in the post-fix run. If it recurs, either capture the curobo logger's stderr on the server
side, or (last resort, third-party edit) guard: `if sys.exc_info()[0] is None: raise
RuntimeError(txt)`.

## [OPEN] Root drift across closed-loop steps

First multi-step episode observed (task 23: step0 goal=5 succeeded → re-observe → step1
goal=6). The memoryless VLM can pick a different root after a successful removal, discarding
the planner's global ordering. Measure `goal_seg_id` change-rate across steps once multi-step
episodes become common (post-fix). See `VLM_VORM_INFORMED_TARGET_v*.md` §3.
