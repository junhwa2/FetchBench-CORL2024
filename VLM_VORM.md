# VLM + VORM pipeline

Adds language-instructed target selection (a VLM) on top of the existing VORM
obstruction predictions, **without modifying any existing code** (new files
only; the two runtime tasks self-register into `isaacgym_task_map` on import).
A Vision-Language Model (Gemini Robotics-ER) reads the scene image + a language
instruction and points at the target object; the resolved target is handed to a
no-edit subclass of `FetchMeshCuroboGORun`, which plans on it with the VORM-
driven KB.

## Two-phase design

```
Phase 1  (new task: FetchMeshCuroboGORunVLM, entry: vlm_target_gen.py)
  reset_task(i) - settle - render RGB+seg - build instruction "Pick up the {label}."
  - VLM points (or picks a numbered marker) - seg maps the answer to an obj
  - record it in vlm_target_logs/<exp>/targets.npz   (dataset NOT mutated)

Phase 2  (new subclass: FetchMeshCuroboGORunOverride, entry: vlm_phase2.py)
  pred_folder=<VORM preds>  +  target_override=<targets.npz>
  - plans on the VLM's target with the VORM-driven KB

Phase 3  (scripts/vlm_join_results.py)
  joins the two on task_idx - instruction-conditioned success
```

### Files (all additive — no existing file is edited)

- `InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_go_run_vlm.py` — `FetchMeshCuroboGORunVLM` (overrides `solve()` only)
- `InfiniGym/isaacgymenvs/tasks/fetch/fetch_mesh_curobo_go_run_override.py` — `FetchMeshCuroboGORunOverride` (Phase-2 sidecar reader; overrides `load_planner_target_from_config` only)
- `InfiniGym/isaacgymenvs/tasks/fetch/utils/vlm_target/`
  - `gemini_er_client.py` — Gemini REST client (retry/backoff, self-consistency, marker mode) + oracle client + factory
  - `target_resolver.py` — camera pick, point→obj_id, visibility helpers (pure numpy)
- `InfiniGym/isaacgymenvs/config/task/FetchMeshCuroboGORunVLM.yaml`, `FetchMeshCuroboGORunOverride.yaml`
- `InfiniGym/isaacgymenvs/vlm_target_gen.py` (Phase 1), `vlm_phase2.py` (Phase 2)
- `scripts/vlm_join_results.py` (Phase 3)

Both new tasks register themselves at import time, so `tasks/__init__.py` stays
untouched; the entry scripts import the task module before building the env.

> **No-edit note.** An external review proposed the sidecar/override by *editing*
> `FetchMeshCuroboGORun.solve()`. To honour the no-existing-code-edit rule, the
> same behaviour is obtained by subclassing instead: pred-mode `solve()` reads
> its target through `self.load_planner_target_from_config()`, an overridable
> method, so `FetchMeshCuroboGORunOverride` overrides just that. With
> `target_override` unset the subclass behaves byte-for-byte like the stock task.
> (The review's `persist_planner_target` guard is only needed for GT-mode
> sweeps, which this pipeline never runs, so it is not required.)

## Metrics — read this before quoting a number

`FetchMeshCuroboGORun.eval()` scores `success` against the object the *planner*
targeted, which in pred mode is whatever target was handed in. A perfectly
executed fetch of the **wrong** object therefore scores as success. The Success
Rate that the sim prints is not a language-grounding result.

Report these instead (all produced by the scripts):

| metric | source | meaning |
| --- | --- | --- |
| **`instruction_success_rate`** | join | `plan_success ∧ (chosen == instructed)`. **The headline number.** |
| `grounding_accuracy` | Phase 1 `target_accuracy` | right object chosen. **Strict**: an API error or unresolvable answer counts as wrong. |
| `point_accuracy` | Phase 1 | the raw pixel under the VLM's point belongs to the instructed object — grounding before any seg snapping. |
| `exec_success_given_correct_target` | join | manipulation quality, isolated from grounding. |
| `exec_success_given_wrong_target` | join | the inflation term. |
| `chance_level` | Phase 1 | mean `1 / n_visible_candidates`. Always compare against it. |
| `accuracy_on_visible` | Phase 1 | accuracy over tasks where the instructed object was actually on screen. |
| `raw_eval_success_rate_INFLATED` | join | what the sim prints, for contrast only. |

Deliberately superseded: the old `target_accuracy` counted a `fallback: gt`
task as correct, so a single HTTP 429 scored as a right answer; and
`resolve_rate` was structurally 1.0 because centroid snapping always succeeds.
Both are still emitted (`target_accuracy_lenient_DEPRECATED`, `snap_rate`) but
only for comparison with earlier runs.

## Instruction label system

`{label}` comes from `vlm.label_source`:

- **`category`** (default): the real object name from `asset_config.json`
  (`benchmark_objects/<Category>/…`), humanized (`DeskLamp`→`desk lamp`). When
  another object shares the category, a left/right qualifier is appended
  (`left`/`right`/`middle`/`leftmost`/`second-from-left`…), ordered by seg
  x-centroid in the chosen camera → `"Pick up the right bottle."`
- **`object_labels`**: the coarse benchmark label from `rearrange_config.npz` —
  not discriminative, kept only for comparison.

`ambiguity_scope` decides **which objects count as duplicates**:
- `visible` (default) — every object with ≥ `min_visible_px` px in the chosen
  view. The VLM sees non-candidates too, so an unqualified "the bottle" is
  genuinely ambiguous when a second bottle is in frame; scoring the model wrong
  for that measures the prompt, not the model.
- `candidate` — legacy behaviour (candidate subset only).

`normalize_category` makes duplicate detection case/synonym-insensitive
(`vase`/`Vase`, `plantcontainer`/`PottedPlant`). It only affects **whether** the
qualifier fires, never the displayed name.

## Query styles

- `prompt_mode: point` — free-form pointing, resolved through the seg mask.
- `prompt_mode: mark` — numbered markers are drawn on the visible candidates and
  the model returns one number. Removes seg-snapping ambiguity and the
  "pointed at an identical non-candidate" failure mode entirely. It is also an
  easier task, so report it as a **separate condition**, not a drop-in.
- `num_samples: N > 1` — self-consistency; N draws, majority vote on the
  resolved object (temperature is forced ≥ 0.4 after the first draw).

## Gemini backend (REST, Python-3.8 safe)

Calls the REST API with `requests` — **not** the `google-genai` SDK — because
this repo's pinned env is **Python 3.8** and google-genai requires ≥ 3.9. Do
**not** `pip install google-genai`. Endpoint:
`https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`,
auth via the `x-goog-api-key` header. Pointing output (`[y, x]` in 0–1000) is
parsed to fractional `(x, y)`. Transient failures (429/5xx/timeout) are retried
with exponential backoff; `promptFeedback.blockReason` and non-`STOP`
`finishReason` are surfaced as real errors, not empty answers.

## Why the dataset is no longer mutated

The old design wrote the chosen target back into
`$ASSET_PATH/Task/<scene>/task_config.npz`. That one slot has three writers — the
benchmark, GT-mode `FetchMeshCuroboGORun`, and the VLM pass — so a GT sweep
clobbers a VLM target, a late `.vlm_bak` captures a planner's choice rather than
the benchmark target, and runs cannot be parallelised. Phase 1 now writes
`vlm_target_logs/<exp>/targets.npz`; Phase 2 reads it via
`task.solution.target_override=<path>` through the `Override` subclass, and
`$ASSET_PATH` stays read-only. The legacy in-place write is still available
(`vlm.write_task_config=True`), in which case the gt reference is pinned to the
`.vlm_bak` snapshot so repeated runs don't drift.

## Running it

```bash
cd InfiniGym
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3   # Task symlink set

# --- Phase 1a: plumbing check (no API) - must give target_accuracy ~1.0 ---
python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
    scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
    task.solution.vlm.backend=oracle task.solution.vlm.oracle_mode=gt

# --- Phase 1b: CHANCE BASELINE - run before quoting any Gemini number ---
python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
    scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
    task.solution.vlm.backend=oracle task.solution.vlm.oracle_mode=random

# --- Phase 1c: real Gemini Robotics-ER ---
export GEMINI_API_KEY="<lab-key>"
python isaacgymenvs/vlm_target_gen.py task=FetchMeshCuroboGORunVLM \
    scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
    task.solution.vlm.save_debug_image=True

# --- Phase 2: plan on the VLM target + VORM preds (no dataset mutation) ---
#     obs_folder MUST be the "_sim" variant (it carries target/obs_ids).
python isaacgymenvs/vlm_phase2.py task=FetchMeshCuroboGORunOverride \
    scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True \
    task.solution.obs_folder=Obstruction_260513_60_111_sim/benchmark_eval \
    task.solution.pred_folder=vorm_obstruction_260516 \
    task.solution.target_override=$PWD/vlm_target_logs/<phase1_exp>/targets.npz

# --- Phase 3: the number that actually answers the research question ---
cd ..
python scripts/vlm_join_results.py \
    --vlm-log InfiniGym/vlm_target_logs/<phase1_exp> \
    --sim-run InfiniGym/runs/<phase2_exp>
```

Phase-1 diagnostics land in `InfiniGym/vlm_target_logs/<exp>/`: `summary.json`,
`results.csv`, `targets.npz`, `config.yaml`, per-task JSON, and RGB overlays
(red = VLM point, green = chosen, blue = ground truth) with
`save_debug_image=True`.

## Config (`task.solution.vlm.*`)

| key | default | meaning |
| --- | --- | --- |
| `backend` | `gemini` | `gemini` (REST API) or `oracle` (no API) |
| `model` | `gemini-robotics-er-2-preview` | Gemini model id |
| `api_key` | `null` | prefer the `GEMINI_API_KEY` env var |
| `temperature` | `0.0` | sampling temperature |
| `thinking_budget` | `0` | 0 = fastest; raise for harder scenes |
| `max_retries` / `retry_base_delay` | `3` / `2.0` | backoff on 429/5xx/timeout |
| `scene_description` | generic indoor | prompt preamble |
| `oracle_mode` | `gt` | `gt` (plumbing) \| `random` (chance baseline) |
| `prompt_mode` | `point` | `point` \| `mark` (numbered candidates) |
| `num_samples` | `1` | self-consistency draws |
| `label_source` | `category` | `category` \| `object_labels` |
| `normalize_category` | `True` | case/synonym-insensitive duplicate detection |
| `ambiguity_scope` | `visible` | `visible` (all on-screen objects) \| `candidate` (legacy) |
| `min_visible_px` | `30` | visibility threshold for the above |
| `instruction_template` | `Pick up the {label}.` | |
| `resolve_window` | `15` | px half-window for the seg window-vote |
| `allow_centroid_snap` | `True` | snap a missed point to the nearest candidate |
| `seg_offset` | `4` | `seg_pixel == old_id + seg_offset` |
| `fallback` | `gt` | `gt` \| `first_cand` \| `none` — always scored as **incorrect** |
| `settle_steps` | `null` | `null` → use `solution.init_steps` (60) |
| `use_vis_camera` | `False` | include the trailing fixed third-person camera |
| `write_task_config` | `False` | legacy in-place dataset overwrite |
| `save_debug_image` | `False` | RGB overlay per task |

Phase-2 side (`task.solution.*`): `target_override` (path to `targets.npz`,
`null` = stock `task_config.npz` read).

## Experiment matrix to fill in

One scene × 60 tasks gives a 95% CI of roughly ±0.13, so a 1-task difference is
noise. Run several scene categories, and fill this out:

| condition | target source | measures |
| --- | --- | --- |
| chance | oracle `random` | floor |
| oracle | oracle `gt` | plumbing (must be ~1.0) + execution ceiling |
| planner-chosen | stock GT mode | the original (search-based) system |
| VLM point | Gemini, `prompt_mode=point` | the proposed system |
| VLM mark | Gemini, `prompt_mode=mark` | grounding with the search space shown |

## Results measured so far (scene `Desk/DeskSceneFactory_46`, 60 tasks)

Validated under the **new strict scoring** (mean 11.1 visible candidates/task):

| run | metric | result |
| --- | --- | --- |
| Phase 1, oracle `gt` | target_accuracy / point_accuracy | **1.00 / 1.00** (plumbing OK; snap_rate 0, fallback 0) |
| Phase 1, oracle `random` | target_accuracy | **0.10** ≈ `chance_level` **0.102** (0.98× chance) |
| Phase 1, real Gemini | grounding_accuracy | **to re-measure** under strict scoring |
| Phase 2, VLM target + VORM | Success Rate (raw) | INFLATED — quote `instruction_success_rate` from the join instead |

The oracle runs pin the two reference points: `gt` = 1.0 confirms the pipeline,
`random` = chance is the floor every Gemini number must beat. Pre-review Gemini
runs read **0.53–0.55** grounding, but that was under the lenient scoring
(fallback counted correct); re-run Phase 1c to get the strict number. An earlier
"0.90" was a measurement artifact (gt overwritten between runs), now fixed by the
fixed-gt read and the read-only targets.npz channel.

## Known limitations / next directions

- **Camera choice uses ground-truth segmentation** (`pick_camera` maximises
  candidate pixel coverage and prefers views where the target is visible). A
  mild oracle leak; a fixed camera is cleaner for a perception claim.
- **Resolution.** Cameras render at 640×480 (`cfg.env.cam.width/height`); an
  object in a cluttered shelf can be 40–80 px. Phase 1 uses no point cloud, so
  raising it is safe and cheap: append
  `task.env.cam.width=1280 task.env.cam.height=960` to the Phase-1 command.
- Other levers: `thinking_budget > 0`, colour/material cues, `prompt_mode=mark`,
  `num_samples=5`, multiple camera views.
- **The planner no longer chooses the target.** Originally the target maximised
  feasibility (min_clear, tiebreak by grasp count/volume). Under instruction it
  is dictated, so Phase-2 success drops for reasons unrelated to grounding.
  Report the planner-chosen condition alongside and use
  `exec_success_given_correct_target` to separate the two effects.

## API key

Precedence: `task.solution.vlm.api_key=...` > `GEMINI_API_KEY` >
`GOOGLE_API_KEY`. Prefer the env var; do **not** commit the key in the yaml.
No SDK install is required.

```bash
export GEMINI_API_KEY="<lab-key>"          # e.g. in ~/.bashrc, one clean line
echo "${GEMINI_API_KEY:0:6}..."            # verify it is visible to the process
```

Write exactly `export GEMINI_API_KEY="..."` (not wrapped in `echo '...'`). The
Isaac Gym launch inherits the shell environment.
