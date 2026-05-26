#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FetchBench
export PYTHONPATH=/home/jo/HJ/FetchBench-CORL2024/InfiniGym:$PYTHONPATH
export ASSET_PATH=/home/jo/HJ/FetchBench-CORL2024/asset_release_v1.3
export DISABLE_PANDARALLEL=1

# Per-category counts: "<CATEGORY> <TRAIN> <VAL> <EVAL>"
SCENE_SET=(
    "CellShelfDesk          7 0 3" # 10 total
    "Desk                   4 1 1" # 6 total
    "DoubleDoorCabinet      4 1 1" # 6 total
    "Drawer                 4 1 1" # 6 total
    "EketShelf              4 0 1" # 5 total
    "LargeShelfDesk         6 1 2" # 9 total
    # "LayerShelf             7 1 2" # 10 total
    "RoundTable             5 1 1" # 7 total
    # "SingleDoorCabinetDesk  5 1 1" # 7 total
    # "TriangleShelfDesk      4 0 1" # 5 total
    # "DeskWall               6 1 2" # 9 total
    # "DrawerShelf            7 1 2" # 10 total
    # "LargeShelf             7 1 2" # 10 total
)

TASK_PATH=Task_260513_60_111
OBS_PATH=Obstruction_260513_60_111_sim


# Scene Config Generation
# python ../scripts/generate_scene_yaml.py --asset-root $ASSET_PATH/$TASK_PATH/benchmark_val --output-dir isaacgymenvs/config/scene/benchmark_val_v1.3
# python ../scripts/generate_scene_yaml.py --asset-root $ASSET_PATH/$TASK_PATH/benchmark_eval --output-dir isaacgymenvs/config/scene/benchmark_eval_v1.3
# python ../scripts/generate_scene_yaml.py --asset-root $ASSET_PATH/$TASK_PATH/benchmark_train --output-dir isaacgymenvs/config/scene/benchmark_train_v1.3


# Run Obstruction Data Generation
# ln -sfn Task_260513_60_111 $ASSET_PATH/Task
ln -sfn $TASK_PATH $ASSET_PATH/Task
[[ "$(readlink "$ASSET_PATH/Task" 2>/dev/null)" == "$TASK_PATH" ]] || ln -sfn "$TASK_PATH" "$ASSET_PATH/Task"

# LOG_DIR="logs_obsgen/$(date +%Y%m%d_%H%M%S)_${OBS_PATH}"
# mkdir -p "$LOG_DIR"
# echo "Logs: $LOG_DIR  (tail -f $LOG_DIR/<scene>.log to monitor)"

# run() {
#     local scene=$1
#     local log_name="${scene//\//_}.log"
#     echo "--- run $scene -> $LOG_DIR/$log_name ---"
#     { time python isaacgymenvs/obs_data_gen.py \
#         pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 \
#         task.sim.use_gpu_pipeline=True task.sim.physx.use_gpu=True \
#         task=FetchMeshCuroboGO headless=True \
#         task.obs_data_gen.obs_path=$OBS_PATH \
#         scene=$scene ; } > "$LOG_DIR/$log_name" 2>&1
# }

# echo "=== Start: $(date) ==="
# for entry in "${SCENE_SET[@]}"; do
#     read -r CAT N_TRAIN N_VAL N_EVAL <<< "$entry"
#     echo ">>> Category: $CAT (train=$N_TRAIN val=$N_VAL eval=$N_EVAL)"
#     for i in $(seq 0 $((N_EVAL - 1))); do
#         run benchmark_eval_v1.3/RigidObj${CAT}_$i
#     done
# done

# for entry in "${SCENE_SET[@]}"; do
#     read -r CAT N_TRAIN N_VAL N_EVAL <<< "$entry"
#     echo ">>> Category: $CAT (train=$N_TRAIN val=$N_VAL eval=$N_EVAL)"
#     for i in $(seq 0 $((N_TRAIN - 1))); do
#         run benchmark_train_v1.3/RigidObj${CAT}_$i
#     done
# done

# for entry in "${SCENE_SET[@]}"; do
#     read -r CAT N_TRAIN N_VAL N_EVAL <<< "$entry"
#     echo ">>> Category: $CAT (train=$N_TRAIN val=$N_VAL eval=$N_EVAL)"
#     for i in $(seq 0 $((N_VAL - 1))); do
#         run benchmark_val_v1.3/RigidObj${CAT}_$i
#     done
# done
# echo "=== End: $(date) ==="

# # Validate Obstruction Data and Generate Metadata
# python ../scripts/validate_and_build_obs_metadata.py --root $ASSET_PATH/$OBS_PATH/benchmark_val --skip-preview
# python ../scripts/validate_and_build_obs_metadata.py --root $ASSET_PATH/$OBS_PATH/benchmark_eval --skip-preview
# python ../scripts/validate_and_build_obs_metadata.py --root $ASSET_PATH/$OBS_PATH/benchmark_train --skip-preview

# Eval
# cd ..
# python eval.py --obstruction_root $ASSET_PATH/$OBS_PATH/benchmark_eval --method fcl --pitch 0.025
# python eval.py --obstruction_root $ASSET_PATH/$OBS_PATH/benchmark_eval --method grn

# Run Simulation with Obstruction Data
# python isaacgymenvs/eval.py task=FetchMeshCuroboGORun scene=benchmark_eval_v1.3/RigidObjDesk_0 headless=True
OBS_FOLDER=$OBS_PATH/benchmark_eval
# Each entry pairs "<pred_folder> <prefix>". PREFIX is forwarded to task.prefix
# (Hydra override) and also used as the log directory label, so it should be a
# short alphanumeric identifier. Use 'null' for pred_folder to fall back to GT.
PRED_FOLDERS_PREFIX=(
    # "null                               GT"
    "vorm_obstruction_260516            VORM"
    # "20260525_204327_fcl_urdf_0.025   FCL"
    # "20260525_205047_grn              GRN"
)

run_sim() {
    local scene=$1
    local log_name="${scene//\//_}.log"
    echo "--- run $scene -> $LOG_DIR/$log_name ---"
    local t0=$SECONDS
    { time python isaacgymenvs/eval.py \
        pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 \
        task.sim.use_gpu_pipeline=True task.sim.physx.use_gpu=True \
        task=FetchMeshCuroboGORun headless=True \
        task.prefix=$PREFIX \
        task.solution.obs_folder=$OBS_FOLDER \
        task.solution.pred_folder=$PRED_FOLDER \
        scene=$scene ; } > "$LOG_DIR/$log_name" 2>&1
    local dt=$(( SECONDS - t0 ))
    printf "    elapsed: %ds (%dm %02ds)\n" "$dt" "$((dt/60))" "$((dt%60))"
}

echo "=== Start: $(date) ==="
for pred_entry in "${PRED_FOLDERS_PREFIX[@]}"; do
    read -r PRED_FOLDER PREFIX <<< "$pred_entry"
    LOG_DIR="logs_sim/$(date +%Y%m%d_%H%M%S)_${OBS_PATH}_${PREFIX}"
    mkdir -p "$LOG_DIR"
    echo "=== pred_folder=$PRED_FOLDER  prefix=$PREFIX  ->  $LOG_DIR ==="
    for entry in "${SCENE_SET[@]}"; do
        read -r CAT N_TRAIN N_VAL N_EVAL <<< "$entry"
        echo ">>> Category: $CAT (train=$N_TRAIN val=$N_VAL eval=$N_EVAL)"
        for i in $(seq 0 $((N_EVAL - 1))); do
            run_sim benchmark_eval_v1.3/RigidObj${CAT}_$i
        done
    done
done
echo "=== End: $(date) ==="