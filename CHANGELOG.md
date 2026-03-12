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