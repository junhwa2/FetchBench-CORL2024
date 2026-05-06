# ChangeLog

## 2026-03-12
- Added new dataset: `asset_release_v1`(https://kist.gov-dooray.com/project/drive-files/4286545709009150728?fileType=file)
- Added new run config: `isaacgymenvs/config/scene/benchmark_eval_v1` (git)
- Deleted `numObjs` from `FetchBase.yaml`
    - `numObjs` is now inferred from the length of object_assets in `fetch_base.py`

### Run with asset_release_v1
```
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1
python isaacgymenvs/eval.py task=FetchBase scene=benchmark_eval_v1/RigidObjDoubleDoorCabinet_0
```

### Run with asset_release
```
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release
python isaacgymenvs/eval.py task=FetchBase scene=benchmark_eval/RigidObjDoubleDoorCabinet_0
```

## 2026-03-26
- Added new script: `generate_scenes.py`
- modified script: `config/task/FetchBase.yaml`

to run `generate_scenes.py`, modify `FetchBase.yaml`
```
env.sceneCategory: what you want to using scenes in asset_release/benchmark_scenes
env.sceneIdx: what you want to using scenes in asset_release/benchmark_scenes/~/
env.numObjs: what you want to using objects in asset_release/benchmark_objects (to see asset_release/benchmark_objects/metadata.csv, column True)
env.numSceneObjs: how many objects on(in) scenes
```
- modified script: `trimesh_scene.py`
    - add `random_arrangement_JH()`
- modified script: `infini_scenes.py`
    - modify `_create_envs()`
- modified script: `tasks/fetch/utils/load_utils.py`
    - modify `create_env_tasks()`

## 2026-04-03
- Added new script: `split_benchmark.py`
    - A script that divides the dataset into train and eval

## 2026-04-24
- Added new script: `find_task_configs.py`
    - A script to check whether the generated datasets were created correctly, including the number of tasks, total number of objects, etc.
- Added new script: `test_object_stability.py`
    - A script to identify how object stability varies depending on position
    
- modified script: `fecth_base`
    - add `fixed_objects` to the `create_sim()` arguments.
    - `fixed_objects` is used when the saved dataset does not contain the target number of tasks. It reloads the same object types as the saved dataset so that additional data can be generated consistently.
- modified script: `infini_scenes.py`
    - add `sample_random_asset_JH` to generate consistently based on the saved dataset(related `fixed_objects`).
    - modified `__init__()`, `load_object_asset()`, `_create_envs()` related `fixed_objects`
    - modified `check_env_status` to strictly check the stable state after applying the physics engine (z_pos)
- modified script: `load_utils.py`, `generate_scenes.py`
    - related `fixed_objects`