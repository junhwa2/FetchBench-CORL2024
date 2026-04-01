# ChangeLog
## 2026-04-01
- Added new dataset: `asset_release_v1_obs` (https://kist.gov-dooray.com/project/drive-files/4300558259664061648?fileType=file)
- Added new task `FetchMeshCuroboGO` for generating Grasp Obstruction dataset
- Updated `FetchPtd` for returning segmented point cloud
- Updated `FetchBase` for temporary setup for task_obj_cand_index, task_obj_cand_label (must be removed after v1.1)
- Added new third_party library pykin (https://github.com/jdj2261/pykin) for collision checking
- Added robot asset `franka_r3_cvx_pykin.urdf` for pykin

### Run with asset_release_v1
```
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1
python isaacgymenvs/eval.py task=FetchMeshCuroboGO scene=benchmark_eval_v1/RigidObjDoubleDoorCabinet_0
```

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