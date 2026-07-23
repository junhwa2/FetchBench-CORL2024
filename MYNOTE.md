## scene config 생성
python scripts/generate_scene_yaml.py --asset-root asset_release_v1.3/Task_260512_mini_small_60/benchmark_val --output-dir InfiniGym/isaacgymenvs/config/scene/benchmark_val_v1.3
python scripts/generate_scene_yaml.py --asset-root asset_release_v1.3/Task_260512_mini_small_60/benchmark_eval --output-dir InfiniGym/isaacgymenvs/config/scene/benchmark_eval_v1.3

## obstruction data 생성 방법
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3
ln -sfn Task_260512_CellShelf_60_123 $ASSET_PATH/Task
ls -la $ASSET_PATH/Task # 어디를 가리키고 있는지 확인 가능

cd InfiniGym/
python isaacgymenvs/obs_data_gen.py task=FetchMeshCuroboGO scene=benchmark_eval_v1.3/RigidObjCellShelfDesk_0 headless=True task.obs_data_gen.obs_path=Obstruction_mini_test_linear_2cam task.env.cam.num_cam=2 task.obs_data_gen.pc_new_id_method=valid

## obstruction data 확인 및 metadata 생성
cd scripts/
python validate_and_build_obs_metadata.py --root ../asset_release_v1.3/Obstruction_mini_test_linear/benchmark_eval --skip-preview

## baseline evaluation
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method fcl --pitch 0.025 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method fcl
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method grn
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method cbn --robot_to_model_t -0.7 0.0 -0.1 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method cbn
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method scn --robot_to_model_t -0.7 0.0 -0.1 // best
python eval.py --obstruction_root asset_release_v1.3/Obstruction_260513_60_111/benchmark_eval --method scn

## sim run
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3
ln -sfn Task_260513_60_111 $ASSET_PATH/Task
cd InfiniGym/

python isaacgymenvs/eval.py task=FetchMeshCuroboGORun scene=benchmark_eval_v1.3/RigidObjDesk_0

python isaacgymenvs/obs_data_gen.py task=FetchMeshCuroboGO scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True task.obs_data_gen.obs_path=Obstruction_test task.env.cam.num_cam=1 task.obs_data_gen.pc_new_id_method=valid
