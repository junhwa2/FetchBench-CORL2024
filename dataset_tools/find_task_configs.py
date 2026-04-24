import os
import numpy as np

NUM_ENVS = 50
NUM_OBJS = 203

# 환경 변수에서 ASSET_PATH를 가져오고, 없으면 기본 경로 사용
asset_path = os.environ.get("ASSET_PATH", "../asset_release")
base_dir = os.path.join(asset_path, "Task", "v1.2")

for main_dir in os.listdir(base_dir):
    main_path = os.path.join(base_dir, main_dir)
    if not os.path.isdir(main_path):
        continue
    for sub_dir in os.listdir(main_path):
        sub_path = os.path.join(main_path, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        npz_path = os.path.join(sub_path, "task_config.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                # print(len(data['task_init_state']))
                if len(data['task_init_state']) != NUM_ENVS:
                    print("NUM_ENVS <", NUM_ENVS)
                    print("\t", npz_path)
                    print("\t", len(data['task_init_state']))
                elif len(data['task_init_state'][0]) != NUM_OBJS:
                    print("NUM_OBJS <", NUM_OBJS)
                    print("\t", npz_path)
                    print("\t", len(data['task_init_state'][0]))
            except Exception as e:
                print(f"  Error reading {npz_path}: {e}")
        else:
            print(f"can't find {npz_path}")