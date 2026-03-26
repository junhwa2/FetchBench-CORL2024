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
- Added new script: `isaacgymenvs/generate_scenes.py`
- modifed script: `isaacgymenvs/config/task/FetchBase.yaml`

to run `generate_scenes.py`, modify `FetchBase.yaml`
```
env.sceneCategory: what you want to using scenes in asset_release/benchmark_scenes
env.sceneIdx: what you want to using scenes in asset_release/benchmark_scenes/~/
env.numObjs: what you want to using objects in asset_release/benchmark_objects (to see asset_release/benchmark_objects/metadata.csv, column True)
env.numSceneObjs: how many objects on(in) scenes
```
- modifed script: `tasks/fetch/infini_scene/trimesh_scene.py`
    - add `random_arrangement_JH()`
- modifed script: `tasks/fetch/infini_scene/infini_scenes.py`
    - modify `_create_envs()`
- modifed script: `tasks/fetch/utils/load_utils.py`
    - modify `create_env_tasks()`