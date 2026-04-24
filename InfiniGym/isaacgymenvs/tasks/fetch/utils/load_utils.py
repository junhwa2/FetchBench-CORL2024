import trimesh
import json
import numpy as np
import os
import h5py
import pandas as pd
import random


# Todo: Update this to be the asset path.
ASSET_PATH = os.environ["ASSET_PATH"]
SCENE_PATH = f"{ASSET_PATH}/Task"


def get_franka_panda_asset(type='franka_r3', mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': './assets',
            'urdf_file': f'urdf/franka_description/robots/{type}.urdf'
        }
    elif mode == 'benchmark':
        paths = {
            # 'asset_root': './assets',
            'asset_root': '../assets',
            'urdf_file': f'urdf/franka_description/robots/{type}.urdf'
        }
    else:
        raise NotImplementedError

    return paths

"""
Scene Asset
"""


def load_scene_asset(path):
    file_metadata = np.load(f'{path}/metadata.npy')

    meshes, files = [], []
    for f in file_metadata:
        mesh_file = f'{path}/{f}.obj'
        m = trimesh.load_mesh(mesh_file)
        if isinstance(m, trimesh.Trimesh):
            meshes.append(m)
        else:
            meshes.extend(m.geometry.values())

        files.append(mesh_file)

    for m in meshes:
        m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, 1, 0, 0], [0, 0, 0, 1]]))

    with open(f'{path}/support.json', 'r') as f:
        support = json.load(f)

    with open(f'{path}/collider.json',  'r') as f:
        collider = json.load(f)

    robot_cam_config = np.load(f'{path}/robot_cam_config.npy', allow_pickle=True).tolist()

    return {
        'meshes': meshes,
        'files': files,
        'support': support,
        'collider': collider,
        'robot_cam_config': robot_cam_config
    }


def get_scene_asset(type, idx, mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/scenes/{type}/assets/scene_{idx}',
            'urdf_file': 'asset.urdf'
        }
    elif mode == 'benchmark':
        paths = {
            'asset_root': f'{ASSET_PATH}/benchmark_scenes/{type}/assets/scene_{idx}',
            'urdf_file': 'asset.urdf'
        }
    else:
        raise NotImplementedError

    return {**paths, **load_scene_asset(paths['asset_root'])}


def load_env_scene(config):
    if 'benchmark_scenes' in config['asset_root']:
        mode = 'benchmark'
    else:
        mode = 'ws'
    type = config['asset_root'].split('/')[-3]
    idx = config['asset_root'].split('/')[-1].split('_')[-1]
    asset = get_scene_asset(type, idx, mode)
    asset['name'] = config['name']
    return asset


def sample_random_scene(scene_type, scene_idx, mode='ws'):
    asset = get_scene_asset(scene_type, f"{scene_idx:03}", mode)
    asset['name'] = 'support_000'
    return asset


"""
Object Assets
"""


def load_object_asset(obj_path):
    mesh = trimesh.load_mesh(f'{obj_path}/mesh.obj')
    # 현재 obj는 shapenet에서 직접 받은 obj로 정점 등이 기존 fetchbench와 달라 mtl을 사용하기위해서는 shapenet의 obj 사용 필수
    # shapenet의 obj는 list, tuple 형태로 로드되므로 하나의 trimesh.Trimesh로 합쳐줘야함
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    elif isinstance(mesh, (list, tuple)):
        mesh = trimesh.util.concatenate(mesh)
    assert isinstance(mesh, trimesh.Trimesh)
    stable_poses = np.load(f'{obj_path}/stable_poses.npy')

    if os.path.exists(f'{obj_path}/grasp_poses.h5'):
        grasp_poses, grasp_success = load_h5_grasps(f'{obj_path}/grasp_poses.h5')
    elif os.path.exists(f'{obj_path}/grasp_poses.npy'):
        grasp_poses = np.load(f'{obj_path}/grasp_poses.npy', allow_pickle=True).tolist()
        grasp_poses = np.array(grasp_poses)
        grasp_poses = grasp_poses @ np.array([[0, -1, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], dtype=np.float32)
        grasp_success = np.ones((len(grasp_poses), ), dtype=np.float32)
    else:
        print("No Grasp File Found!")
        raise NotImplementedError

    metadata = np.load(f'{obj_path}/metadata.npy', allow_pickle=True).tolist()

    # load isaac labels
    default_label = np.load(f'{obj_path}/isaac_label_default.npy', allow_pickle=True).tolist()
    cvx_label = np.load(f'{obj_path}/isaac_label_cvx_4part.npy', allow_pickle=True).tolist()

    return {
        'mesh': mesh,
        'file': f'{obj_path}/mesh.obj',
        'stable_poses': stable_poses,
        'metadata': metadata,
        'grasp_poses': {
            'T': grasp_poses,
            'acronym_label': grasp_success,
            'isaac_label_default': default_label,
            'isaac_label_cvx': cvx_label
        }
    }


# Todo: Update combo asset loading
def load_object_combo_asset(path, placement_type):
    file_metadata = np.load(f'{path}/metadata.npy', allow_pickle=True).tolist()

    meshes, files = [], []
    for f in file_metadata['meshes']:
        mesh_file = f'{path}/{f}.obj'
        m = trimesh.load_mesh(mesh_file)
        if isinstance(m, trimesh.Trimesh):
            meshes.append(m)
        else:
            meshes.extend(m.geometry.values())

        files.append(mesh_file)

    for m in meshes:
        m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, 1, 0, 0], [0, 0, 0, 1]]))

    mesh = trimesh.util.concatenate(meshes)
    bb = mesh.bounds

    if placement_type == 'support':
        translation = np.array([0, 0, -bb[0][2]])
        stable_poses = [trimesh.transformations.translation_matrix(translation)]
    elif placement_type == 'hanging':
        translation = np.array([-bb[0][0], 0, -bb[0][2]])
        stable_poses = [trimesh.transformations.translation_matrix(translation)]
    else:
        raise NotImplementedError

    metadata = np.load(f'{path}/metadata.npy', allow_pickle=True).tolist()

    return {
        'meshes': meshes,
        'files': files,
        'stable_poses': stable_poses,
        'metadata': metadata
    }


def load_h5_grasps(filename):
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)

    # convert to the standard panda_hand frame
    T = T @ np.array([[0, -1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

    return T, success


def get_object_asset(type, idx, mode='benchmark'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/objects/{type}/{idx}',
            'urdf_file': 'mesh.urdf',
        }
    elif mode == 'benchmark':
        paths = {
            'asset_root': f'{ASSET_PATH}/benchmark_objects/{type}/{idx}',
            'urdf_file': 'mesh.urdf',
        }
    else:
        raise NotImplementedError

    return {**paths, **load_object_asset(paths['asset_root'])}


# Todo: Update Combo Asset Load
def get_object_combo_asset(type, idx, mode='ws'):
    if mode == 'ws':
        paths = {
            'asset_root': f'{ASSET_PATH}/combos/{type}/{idx}',
            'urdf_file': ['organizer.urdf', 'object.urdf'],
            'combo_type': type
        }
    else:
        raise NotImplementedError

    if 'Hook' in type:
        paths['fixed_base'] = [True, False]
        paths['placement_type'] = 'hanging'
    else:
        paths['fixed_base'] = [False, False]
        paths['placement_type'] = 'support'

    return {**paths, **load_object_combo_asset(paths['asset_root'], placement_type=paths['placement_type'])}


def load_env_object(config):
    if 'benchmark_objects' in config['asset_root']:
        mode = 'benchmark'
    else:
        mode = 'ws'
    type = config['asset_root'].split('/')[-2]
    idx = config['asset_root'].split('/')[-1]
    asset = get_object_asset(type, idx, mode=mode)
    asset['name'] = config['name']

    return asset


# Todo: Update Combo Asset Load
def load_env_object_combo(config):
    type = config['asset_root'].split('/')[-2]
    idx = config['asset_root'].split('/')[-1]
    asset = get_object_combo_asset(type, idx)
    asset['name'] = config['name']

    return asset


def sample_random_objects(num_objs, eval_only=True,  mode='ws'):
    if mode == 'ws':
        metadata = pd.read_csv(f'{ASSET_PATH}/objects/metadata.csv')
        if eval_only:
            metadata = metadata.loc[metadata['Eval'] == True]
    elif mode == 'benchmark':
        metadata = pd.read_csv(f'{ASSET_PATH}/benchmark_objects/metadata.csv')
        metadata = metadata.loc[metadata['use_benchmark'] == True]
        if eval_only:
            metadata = metadata.loc[metadata['use_eval'] == True]
        else:
            metadata = metadata.loc[metadata['use_eval'] == False]
    else:
        raise NotImplementedError

    samples = metadata.sample(n=num_objs, replace=False)

    objects = []
    for i, (n, s) in enumerate(samples.iterrows()):
        o = get_object_asset(s['Category'], s['ID'], mode)
        o['name'] = f'obj_{i}'
        objects.append(o)

    return objects

def sample_random_objects_JH(object_asset, eval_only=True,  mode='benchmark'):
    if object_asset is not None:
        # object_asset에서 category/id 추출
        cat_id_set = set()
        for asset in object_asset:
            asset_root = asset.get('asset_root', None)
            if asset_root is not None:
                parts = asset_root.rstrip('/').split('/')
                if len(parts) >= 2:
                    category = parts[-2]
                    obj_id = parts[-1]
                    cat_id_set.add(f"{category}/{obj_id}")
        # metadata.csv 읽어서 path 컬럼과 비교
        metadata = pd.read_csv(f'{ASSET_PATH}/benchmark_objects/metadata.csv')
        if eval_only:
            metadata = metadata.loc[metadata['use_benchmark'] == True]
            metadata = metadata.loc[metadata['use_eval'] == True]
        # path 컬럼에서 category/id가 cat_id_set에 있는 row만 필터링
        samples = metadata[metadata['Path'].apply(lambda x: any(x.endswith(cid) for cid in cat_id_set))]
        objects = []
        for i, (n, s) in enumerate(samples.iterrows()):
            o = get_object_asset(s['Category'], s['ID'], mode)
            o['name'] = f'obj_{i}'
            objects.append(o)
        # print("objects")
        # print(objects)
        return objects

# Todo: Update Combo Asset Sample
def sample_random_combos(num_objs, combo_type, mode='ws'):
    # Todo: Random sample of objects
    combo_indices = np.random.choice(os.listdir(f'{ASSET_PATH}/combos/{combo_type}'), size=(num_objs,))

    combos = []
    for i, idx in enumerate(combo_indices):
        o = get_object_combo_asset(combo_type, idx, mode)
        o['name'] = f'obj_combo_{i}'
        combos.append(o)

    return combos


def get_env_config(config):
    return f'{SCENE_PATH}/{config}'


class InfiniSceneLoader(object):
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        self.scene_asset_config = {}
        self.object_asset_config = []
        self.combo_asset_config = []
        self.robot_asset_config = {}
        self.camera_config = {}

        self.scene_pose = []
        self.robot_pose = []
        self.camera_poses = []
        self.object_poses = []
        self.object_labels = []

        # 준화 task_config 추가
        self.task_actor_states = []
        self.task_target_index = []
        self.task_target_labels = []
        self.task_camera_poses_ = []
        self.task_obj_indices_ = []
        self.task_obj_labels_ = []

        self._num_compositions = 0

    def __len__(self):
        return len(self.scene_pose)

    def append_pose(self, pose, cat='scene'):
        if cat == 'scene':
            self.scene_pose = list(self.scene_pose)
            self.scene_pose.append(pose)
        elif cat == 'robot':
            self.robot_pose = list(self.robot_pose)
            self.robot_pose.append(pose)
        elif cat == 'camera':
            self.camera_poses = list(self.camera_poses)
            self.camera_poses.append(pose)
        elif cat == 'object':
            self.object_poses = list(self.object_poses)
            self.object_poses.append(pose)
        else:
            raise NotImplementedError

    def save_env_config(self):
        # vicinity check
        assert (len(self.scene_pose) == len(self.robot_pose) == len(self.camera_poses)
                == len(self.object_poses) == len(self.object_labels))

        self._num_compositions = len(self.scene_pose)
        # dump json
        with open(f'{self._path}/asset_config.json', 'w') as f:
            config = {
                'robot_config': self.robot_asset_config.copy(),
                'scene_config': self.scene_asset_config.copy(),
                'object_config': self.object_asset_config.copy(),
                'combo_config': self.combo_asset_config.copy(),
                'camera_config': self.camera_config.copy()
            }

            json.dump(config, f, indent=4)

        # dump ndarrays
        rearrange = {
            'robot_pose': self.robot_pose,
            'scene_pose': self.scene_pose,
            'object_poses': self.object_poses,
            'camera_poses': self.camera_poses,
            'object_labels': self.object_labels
        }
        np.savez(f'{self._path}/rearrange_config.npz', **rearrange)

        # if save_task_config:
        #     tasks = self.create_env_tasks()
        tasks = {
            'task_init_state': self.task_actor_states,
            'task_obj_index': self.task_target_index,
            'task_obj_label': self.task_target_labels,
            'task_camera_pose': self.task_camera_poses_,
            'task_cand_obj_index': self.task_obj_indices_, 
            'task_cand_obj_label': self.task_obj_labels_ 
        }
        np.savez(f'{self._path}/task_config.npz', **tasks)

    def append_task(self, tasks):
        self.task_actor_states.append(tasks['init_state'])
        self.task_target_index.append(tasks['obj_index'])
        self.task_target_labels.append(tasks['obj_label'])
        self.task_camera_poses_.append(tasks['camera_pose'])
        # print("*"*20)
        # print("append_task - task_cand_obj_index: ", tasks['cand_obj_index'])
        self.task_obj_indices_.append(tasks['cand_obj_index'])
        self.task_obj_labels_.append(tasks['cand_obj_label'])

    def create_env_tasks(self, env_idx):
        '''
        기존: 모든 env 만들어두고 env마다 task 하나씩 랜덤하게 뽑아서 할당
        변경: 안정화 상태인 env 하나만 만들어서 task 하나씩 할당
        '''
        init_root_states = self.get_scene_init_root_states()
        init_obj_labels = self.get_scene_init_obj_labels()
        init_camera_poses = self.get_camera_init_states()
        assert len(init_root_states) == len(init_obj_labels) == len(init_camera_poses)

        # for i in env_idx:
        # print("len init_root_states: ", len(init_root_states))

        # 안정화 상태에 대한 env 하나만 받기 때문에 k=0으로 고정 
            # obj_indices, task_labels = self.get_obj_tasks(init_obj_labels[k])
        obj_indices, task_labels = self.get_obj_tasks(init_obj_labels[env_idx])

        # target random하게 target obj 선택
        k = random.randrange(len(obj_indices))
        idx, label = obj_indices[k], task_labels[k]
        # print("len init_root_states: ", len(init_root_states))
        # print("use env_idx: ", env_idx)
        print(f"scene_{env_idx}: obj_indices: {obj_indices}")
        # print("create_env_tasks - self.obj_indices_: ", self.obj_indices_)
        return {
            # IMPORTANT: return only the current env's task.
            # If we return the accumulated lists here, `append_task()` will
            # re-extend previously appended items and cause duplication
            # (e.g., [first] then [first, second] -> [first, first, second]).
            'init_state': init_root_states[env_idx],
            'obj_index': idx,
            'obj_label': label,
            'camera_pose': init_camera_poses[env_idx],
            'cand_obj_index': obj_indices,
            'cand_obj_label': task_labels
        }

    def get_obj_tasks(self, obj_labels):
        task_obj_indices, task_labels = [], []
        for i, label in enumerate(obj_labels):
            if label.startswith('combo_org') or label.endswith('on_floor'):
                continue

            task_obj_indices.append(i)
            task_labels.append(label)

        return task_obj_indices, task_labels

    def load_env_config(self):
        with open(f'{self._path}/asset_config.json', 'r') as f:
            asset_config = json.load(f)

        self.scene_asset_config = asset_config['scene_config']
        self.robot_asset_config = asset_config['robot_config']
        self.object_asset_config = asset_config['object_config']
        self.combo_asset_config = asset_config['combo_config']
        self.camera_config = asset_config['camera_config']

        rearrange_config = np.load(f'{self._path}/rearrange_config.npz')

        self.scene_pose = rearrange_config['scene_pose']
        self.robot_pose = rearrange_config['robot_pose']
        self.object_poses = rearrange_config['object_poses']
        self.camera_poses = rearrange_config['camera_poses']
        self.object_labels = rearrange_config['object_labels']

        self._num_compositions = len(self.scene_pose)

    def load_task_config(self):
        task_config = np.load(f'{self._path}/task_config.npz', allow_pickle=True)
        return {
            'task_init_state': task_config['task_init_state'],
            'task_obj_index': task_config['task_obj_index'],
            'task_obj_label': task_config['task_obj_label'],
            'task_camera_pose': task_config['task_camera_pose'],
            'task_cand_obj_index': task_config['task_cand_obj_index'],
            'task_cand_obj_label': task_config['task_cand_obj_label']
        }

    def get_camera_init_states(self):
        return self.camera_poses[:]

    def get_scene_init_root_states(self):
        # default seq: [robot, table, scene, *objects]
        if isinstance(self.robot_pose, list):
            self.robot_pose = np.stack(self.robot_pose, axis=0)
        if isinstance(self.scene_pose, list):
            self.scene_pose = np.stack(self.scene_pose, axis=0)
        if isinstance(self.object_poses, list):
            self.object_poses = np.stack(self.object_poses, axis=0)

        root_state = \
            np.concatenate(
                [self.robot_pose.reshape(-1, 2, 13), self.scene_pose.reshape(-1, 1, 13), self.object_poses], axis=1)
        return root_state[:]

    def get_scene_init_obj_labels(self):
        return self.object_labels[:]
