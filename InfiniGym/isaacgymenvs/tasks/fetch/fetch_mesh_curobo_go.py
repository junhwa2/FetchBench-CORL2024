import numpy as np
import os
import torch
import trimesh.transformations as tr
import trimesh
import time
import copy
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

# cuRobo
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
    get_assets_path
    )
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sphere_fit import SphereFitType

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import (to_torch, get_axis_params, tensor_clamp,
                                                tf_vector, tf_combine, quat_mul, quat_conjugate,
                                                quat_to_angle_axis, tf_inverse, quat_apply,
                                                matrix_to_quaternion)

from isaacgymenvs.tasks.fetch.fetch_solution_base import FetchSolutionBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import FetchMeshCurobo
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.utils.load_utils import get_franka_panda_asset, ASSET_PATH

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene
from pykin.utils import plot_utils as p_utils

import open3d as o3d
import h5py
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import isaacgymenvs.tasks.fetch.utils.grn_utils as grn_utils

SPHERE_TYPE = {
    0: SphereFitType.SAMPLE_SURFACE,
    1: SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
}

PC_BOUND_TYPE = {
    'panda_cube': {'x': (-0.855, 0.855), 'y': (-0.855, 0.855), 'z': (-0.36, 1.19)},
    'panda_sphere': {'center': (0.0, 0.0, 0.333), 'radius': 0.855, 'z_min': -0.36},
    'panda_w_gripper_sphere': {'center': (0.0, 0.0, 0.333), 'radius': 0.855+0.205, 'z_min': -0.36},
}

class IsaacGymId:
    ROBOT = 1
    TABLE = 2
    SCENE = 3
    OBJECT_START = 4

class GraspType:
    TOP = 0
    FRONT = 1
    REAR = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5    
    directions = {
        TOP: np.array([0, 0, -1]),
        FRONT: np.array([-1, 0, 0]),
        REAR: np.array([1, 0, 0]),
        RIGHT: np.array([0, 1, 0]),
        LEFT: np.array([0, -1, 0]),
        BOTTOM: np.array([0, 0, 1]),
    }    
    colors = {
        TOP: [0, 0, 255],       # Blue
        FRONT: [255, 0, 0],     # Red
        REAR: [255, 0, 255],    # Magenta
        RIGHT: [255, 255, 0],   # Yellow
        LEFT: [0, 255, 0],      # Green
        BOTTOM: [0, 255, 255],  # Cyan
    }

def _as_o3d_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def image_to_video(obs_buf):
    video = []
    for s, images in enumerate(obs_buf):
        steps = []
        for e, imgs in enumerate(images):
            steps.append(np.concatenate(imgs, axis=0))
        video.append(np.concatenate(steps, axis=1))
    return video


class FetchMeshCuroboGO(FetchPointCloudBase, FetchSolutionBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.tensor_args = TensorDeviceType()
        
        # Setup cuRobo IK Solver
        # world_cuRobo_cfg_list = self._get_cuRobo_world_config()
        world_cuRobo_cfg_list = [WorldConfig()] * self.num_envs
        ik_config = IKSolverConfig.load_from_robot_config(
            self._get_cuRobo_robot_config(),
            world_cuRobo_cfg_list,
            rotation_threshold=self.cfg["solution"]["cuRobo"]["ik_rot_th"],
            position_threshold=self.cfg["solution"]["cuRobo"]["ik_pos_th"],
            num_seeds=self.cfg["solution"]["cuRobo"]["ik_num_seed"],
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=self.cfg["solution"]["cuRobo"]["collision_activation_dist"]
        )
        self.ik_solver = IKSolver(ik_config)
        
        # Setup pykin Collision Manager # [TODO] read from config
        robot_path = get_franka_panda_asset(type='franka_r3_cvx_pykin')
        self.pykin_robot = SingleArm(os.path.join(robot_path['asset_root'], 
                                                  robot_path['urdf_file']), 
                                    offset=Transform(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                                    has_gripper=True,
                                    gripper_name="panda_r3_gripper")
        self.pykin_robot.setup_link_name(base_name='panda_link0', eef_name='panda_link7')
        self.pykin_robot_collision = CollisionManager(is_robot=True)
        self.pykin_robot_collision.setup_robot_collision(self.pykin_robot, geom="collision")
        
        # Obstruction Dataset Generation Config
        self.max_grasp_pose = self.cfg["obs_data_gen"]["max_grasp_pose"]
        self.max_grasp_pose_per_object = self.cfg["obs_data_gen"]["max_grasp_pose_per_object"]
        self.forward_bias_mode = self.cfg["obs_data_gen"].get("forward_bias_mode", "linear")
        self.forward_bias_softmax_temperature = self.cfg["obs_data_gen"].get("forward_bias_softmax_temperature", 0.5)
        assert self.forward_bias_mode in ("linear", "softmax", "uniform"), \
            f"forward_bias_mode must be 'linear', 'softmax', or 'uniform', got {self.forward_bias_mode}"

        # Pre-collision FPS (1): pose metric, floor=1.0 (full pose floor as quality gate).
        # Post-collision per-group FPS (2) and clean-cap FPS (3): pose metric,
        # floor=filter_cap_floor (smaller than 1.0 so closer grasps are kept).
        self.filter_max_per_group = self.cfg["obs_data_gen"].get("filter_max_per_group", 10)
        self.filter_clean_max_ratio = self.cfg["obs_data_gen"].get("filter_clean_max_ratio", 0.5)
        self.filter_pose_pos_floor = self.cfg["obs_data_gen"].get("filter_pose_pos_floor", 0.02)
        self.filter_pose_rot_floor = self.cfg["obs_data_gen"].get("filter_pose_rot_floor", 0.2)
        self.filter_cap_floor = self.cfg["obs_data_gen"].get("filter_cap_floor", 0.3)

        # Global pose dedup after IK
        self.filter_global_after_ik = self.cfg["obs_data_gen"].get("filter_global_after_ik", True)
        self.filter_global_max_per_obj = self.cfg["obs_data_gen"].get("filter_global_max_per_obj", 15)
        self.filter_global_max_min = self.cfg["obs_data_gen"].get("filter_global_max_min", 30)
        self.filter_global_max_max = self.cfg["obs_data_gen"].get("filter_global_max_max", 200)
        self.pc_new_id_method = self.cfg["obs_data_gen"]["pc_new_id_method"]
        self.pc_bound_option = self.cfg["obs_data_gen"]["pc_bound_option"]
        self.pc_voxel_size = self.cfg["obs_data_gen"]["pc_voxel_size"]
        self.mesh_sample_points = self.cfg["obs_data_gen"]["mesh_sample_points"]
        self.mesh_sample_links = self.cfg["obs_data_gen"]["mesh_sample_links"]    
        self.debug_viz = self.cfg['obs_data_gen']['debug_viz']
        
        self.max_num_task_cand_obj = self.cfg["obs_data_gen"]["max_num_task_cand_obj"]  # (updated for each task) 
        self.new_id_map = None # [old_id (Pykin) -> new_id] (updated for each task)            
        self.ts = None # task snapshot (updated for each task)
        
        obs_path = self.cfg['obs_data_gen']['obs_path']
        obs_dir = f"{ASSET_PATH}/{obs_path}"
        if not os.path.exists(obs_dir):
            os.makedirs(obs_dir)
        with open(os.path.join(obs_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        obs_dir += f"/{self.cfg['task']['scene_config_path'][0]}"
        if not os.path.exists(obs_dir):
            os.makedirs(obs_dir)  
        self.obs_dir = obs_dir
        print("obs_dir:", self.obs_dir)
        
        grn_dir = obs_dir + '/grn'
        if not os.path.exists(grn_dir):
            os.makedirs(grn_dir)
                
        assert self.arm_control_type == 'joint'

    """
    Solver Utils (from FetchMeshCurobo)
    """
    def _get_pose_in_robot_frame(self):
        self._refresh()
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        sq, st = tf_combine(rq, rt, self._scene_base_state[..., 3:7].clone(), self._scene_base_state[..., :3].clone())
        dq, dt = tf_combine(rq, rt, self._table_base_state[..., 3:7].clone(), self._table_base_state[..., :3].clone())
        oq, ot = tf_combine(rq.unsqueeze(1).repeat(1, self.num_objs, 1),
                            rt.unsqueeze(1).repeat(1, self.num_objs, 1),
                            self.states["obj_quat"].clone(),
                            self.states["obj_pos"].clone())
        eq, et = tf_combine(rq, rt,  self.states["eef_quat"].clone(), self.states["eef_pos"].clone())

        pose = {
            'scene': {'quat': sq.to(self.tensor_args.device), 'pos': st.to(self.tensor_args.device)},
            'table': {'quat': dq.to(self.tensor_args.device), 'pos': dt.to(self.tensor_args.device)},
            'object': {'quat': oq.to(self.tensor_args.device), 'pos': ot.to(self.tensor_args.device)},
            'eef': {'quat': eq.to(self.tensor_args.device), 'pos': et.to(self.tensor_args.device)}
        }

        return pose

    def _get_cuRobo_robot_config(self):
        robot_config = load_yaml(join_path(get_robot_configs_path(), "franka_r3.yml"))["robot_cfg"]
        robot_cuRobo_cfg = RobotConfig.from_dict(robot_config)
        robot_cuRobo_cfg.cspace.velocity_scale *= self.cfg['solution']['cuRobo']['velocity_scale']
        robot_cuRobo_cfg.cspace.acceleration_scale *= self.cfg['solution']['cuRobo']['acceleration_scale']

        return robot_cuRobo_cfg

    """
    Pykin Utils
    """
    def _load_mesh(self, file):
        m = trimesh.load_mesh(file)
        if isinstance(m, trimesh.Trimesh):
            mesh = m
        elif isinstance(m, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(m.geometry.values()))
        elif isinstance(m, (list, tuple)):
            mesh = trimesh.util.concatenate(m)
        return mesh
        
    def _get_pykin_world_config(self, add_collider=False): #### id related
        pose = self.ts['pose']        
        
        # quaternion convention from xyzw to wxyz
        oq, sq, dq = pose['object']['quat'], pose['scene']['quat'], pose['table']['quat'] 
        oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
        sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
        dq = torch.concat([dq[..., -1:], dq[..., :-1]], dim=-1)

        # torch to numpy
        sq, st = sq.cpu().numpy(), pose['scene']['pos'].cpu().numpy()
        oq, ot = oq.cpu().numpy(), pose['object']['pos'].cpu().numpy()
        dq, dt = dq.cpu().numpy(), pose['table']['pos'].cpu().numpy()
               
        world_config_list = []
        world_info_list = []
        for i in range(self.num_envs):
            obs_cand_index = self.obs_ids[i]
            world_config = CollisionManager()
            world_info = {}
            # add object asset (in order of asset_config.yaml)
            for j, o in enumerate(self.object_asset[i]):
                if j in obs_cand_index:
                    name = f'obj_{j}'
                    mesh = self._load_mesh(o['file'])
                    h_mat = Transform(pos=ot[i][j], rot=oq[i][j]).h_mat
                    world_config.add_object(name, gtype='mesh', gparam=mesh, h_mat=h_mat)
                    world_info[name] = {'gparam': mesh, 'transform': h_mat, 'pykin_id': j, 
                                        'new_id': self.new_id_map[i].get(j, -1), 'category': 'object'}
            num_objs = len(self.object_asset[i])                
            if add_collider:
                # add scene collider asset (in order of collider.json in scene asset folder)
                for j, collider in enumerate(self.scene_asset[i]['collider']):
                    name = collider['obj_name']
                    gparam = collider['mesh']
                    world_to_scene = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                                [0, 1, 0, 0], [0, 0, 0, 1]])
                    world_to_collider = np.array(collider['T'])
                    base_to_scene = Transform(pos=st[i], rot=sq[i]).h_mat 
                    h_mat = base_to_scene @ tr.inverse_matrix(world_to_scene) @ world_to_collider
                    world_config.add_object(name, gtype='box', gparam=gparam, h_mat=h_mat)
                    world_info[name] = {'gparam': gparam, 'transform': h_mat, 'pykin_id': j + num_objs,
                                        'new_id': self.new_scene_id[i], 'category': 'scene'}
            else:
                # add scene .obj asset (in order of metadata.npy in scene asset folder)
                for j, file in enumerate(self.scene_asset[i]['files']):
                    name = file.split("/")[-1].split(".")[0]
                    mesh = self._load_mesh(file)
                    h_mat = Transform(pos=st[i], rot=sq[i]).h_mat
                    world_config.add_object(name, gtype='mesh', gparam=mesh, h_mat=h_mat)
                    world_info[name] = {'gparam': mesh, 'transform': h_mat, 'pykin_id': j  + num_objs, 
                                        'new_id': self.new_scene_id[i], 'category': 'scene'}

            world_config_list.append(world_config)
            world_info_list.append(world_info)
        
        return world_config_list, world_info_list

    def _update_pykin_robot_state(self, goal_qpos, open_gripper=True):
        """
        goal_qpos: in robot base frame, in shape (7,) with (q0, q1, q2, q3, q4, q5, q6)
        goal_eef: in robot base frame, in shape (7,) with (x, y, z, qw, qx, qy, qz)
        """
        
        if goal_qpos.shape[0] > 7:
            goal_qpos = goal_qpos[:7]
        goal_eef = self.pykin_robot.forward_kin(goal_qpos)[self.pykin_robot.eef_name] # transform matrix in robot base frames
        self.pykin_robot.set_transform(goal_qpos)
        if open_gripper:
            self.pykin_robot.open_gripper()
        for link, info in self.pykin_robot.info[self.pykin_robot_collision.geom].items():
            if link in self.pykin_robot_collision._objs:
                self.pykin_robot_collision.set_transform(name=link, h_mat=info[3])
        
        # convert goal_eef from Transform to Numpy array in shape (7,) with (x, y, z, qw, qx, qy, qz)
        goal_eef = np.concatenate([goal_eef.pos, goal_eef.rot]) # rot in wxyz
        return goal_eef

    """
    Point Cloud
    """    
    def _get_seg_color(self, seg_id): #### id related
        """
        seg_id: in IsaacGym ID convention
        """
        MAX_OBJECT_ID = 100
        
        fixed_seg_colors = {
            IsaacGymId.ROBOT: np.array([228, 26, 28], dtype=np.uint8),   # robot: red
            IsaacGymId.SCENE: np.array([255, 127, 0], dtype=np.uint8),   # scene: orange
        }
        if seg_id in fixed_seg_colors:
            return fixed_seg_colors[seg_id]

        # Use viridis colormap for objects
        cmap = plt.colormaps.get_cmap("viridis")
        object_id = min(max(seg_id, 0), MAX_OBJECT_ID - 1)        
        normalized_id = object_id / (MAX_OBJECT_ID - 1) if MAX_OBJECT_ID > 1 else 0.0
        color_rgba = cmap(normalized_id)
        color_rgb_uint8 = (np.array(color_rgba[:3]) * 255).astype(np.uint8)
        
        return color_rgb_uint8

    def _filter_pc(self, pc):
        bound_option = self.pc_bound_option
        voxel_size = self.pc_voxel_size
        
        xyz = pc["xyz"]
        rgb = pc["rgb"]
        id = pc["id"]
        
        if 'cube' in bound_option:
            x_min, x_max = PC_BOUND_TYPE[bound_option]['x']
            y_min, y_max = PC_BOUND_TYPE[bound_option]['y']
            z_min, z_max = PC_BOUND_TYPE[bound_option]['z']
            mask = (
                (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
                (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
                (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
            )
        elif 'sphere' in bound_option:
            center = np.array(PC_BOUND_TYPE[bound_option]['center'], dtype=np.float32)
            radius = PC_BOUND_TYPE[bound_option]['radius']
            z_min = PC_BOUND_TYPE[bound_option]['z_min']
            
            dist2 = np.sum((xyz - center) ** 2, axis=1)
            mask = (dist2 <= radius ** 2) & (xyz[:, 2] >= z_min)
        else:
            raise ValueError(f"Invalid bound_option: {bound_option}")
        
        xyz = xyz[mask]
        rgb = rgb[mask]
        id = id[mask]

        # Semantic-preserving downsampling: downsample each id independently,
        # then merge back so id/color consistency is not mixed across segments.
        id_flat = np.asarray(id, dtype=np.int32).reshape(-1)
        xyz_ds_list, rgb_ds_list, id_ds_list = [], [], []

        for seg_id in np.unique(id_flat):
            seg_mask = id_flat == int(seg_id)
            seg_xyz = np.asarray(xyz[seg_mask], dtype=np.float32)
            seg_rgb = np.asarray(rgb[seg_mask], dtype=np.uint8)
            if seg_xyz.shape[0] == 0:
                continue

            if voxel_size > 0:
                seg_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(seg_xyz, dtype=o3d.core.float32))
                seg_pcd.point["colors"] = o3d.core.Tensor(seg_rgb, dtype=o3d.core.uint8)
                seg_pcd_ds = seg_pcd.voxel_down_sample(voxel_size=voxel_size)
                seg_xyz_ds = np.asarray(seg_pcd_ds.point["positions"].cpu().numpy(), dtype=np.float32)
                seg_rgb_ds = np.asarray(seg_pcd_ds.point["colors"].cpu().numpy(), dtype=np.uint8)
            else:
                seg_xyz_ds = seg_xyz
                seg_rgb_ds = seg_rgb

            if seg_xyz_ds.shape[0] == 0:
                continue

            seg_id_ds = np.full((seg_xyz_ds.shape[0], 1), int(seg_id), dtype=np.int32)
            xyz_ds_list.append(seg_xyz_ds)
            rgb_ds_list.append(seg_rgb_ds)
            id_ds_list.append(seg_id_ds)

        if xyz_ds_list:
            xyz = np.concatenate(xyz_ds_list, axis=0)
            rgb = np.concatenate(rgb_ds_list, axis=0)
            id = np.concatenate(id_ds_list, axis=0)
        else:
            xyz = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)
            id = np.empty((0, 1), dtype=np.int32)
        
        assert xyz.shape[0] == rgb.shape[0] == id.shape[0], "Inconsistent point cloud shapes"
        assert xyz.shape[0] > 1, "Ingenstive point cloud: no points found in camera data"        

        return {
            "xyz": xyz,
            "rgb": rgb,
            "id": id,
        }

    def gen_pc_from_camera(self, env_idx=0): #### id related
        """
        - seg_pts: in old_id (IsaacGym) manner
        - pc: in old_id (IsaacGym) manner
        get point cloud from camera data, scene and non-task-candidate objects are excluded, filter by bound and downsample.
        """
        # Get segmented point cloud from camera data
        point_clouds = self.get_camera_data(
            tensor_ptd=True,
            ptd_in_robot_base=True,
            segmented_ptd=True,
            ptd_downscale=1
        )["camera_pointcloud_seg"]
        seg_pts = point_clouds[env_idx]["by_seg"] # {seg_id(int): points(np.ndarray, N x 3)}
        
        # Exclude robot/table/non-task-candidate objects and assign colors/ids based on old_id
        task_cand_obj_index = self.task_cand_obj_index[env_idx][self.get_task_idx()]
        xyz_list, rgb_list, id_list = [], [], []
        for seg_id, pts in sorted(seg_pts.items()):
            # skip for robot(1) and table(2)
            if seg_id == IsaacGymId.ROBOT or seg_id == IsaacGymId.TABLE:
                continue
            # skip for objects(4>) which are not task candidates
            if seg_id >= IsaacGymId.OBJECT_START:
                if (seg_id - IsaacGymId.OBJECT_START) not in task_cand_obj_index:
                    continue 
            # skip for empty pointcloud segments
            if len(pts) == 0:
                continue
                
            pts = np.asarray(pts, dtype=np.float32)
            color = self._get_seg_color(seg_id)

            xyz_list.append(pts)
            rgb_list.append(np.repeat(color[None, :], len(pts), axis=0))
            id_list.append(np.full((len(pts), 1), seg_id, dtype=np.int32))

        if xyz_list:
            xyz = np.concatenate(xyz_list, axis=0)
            rgb = np.concatenate(rgb_list, axis=0)
            id = np.concatenate(id_list, axis=0)
        else:
            xyz = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)
            id = np.empty((0, 1), dtype=np.int32)
            return {"xyz": xyz, "rgb": rgb, "id": id}
        
        # Filter point cloud by bound and downsample
        filtered_pc = self._filter_pc({"xyz": xyz, "rgb": rgb, "id": id})
        xyz, rgb, id = filtered_pc["xyz"], filtered_pc["rgb"], filtered_pc["id"]

        assert xyz.shape[0] == rgb.shape[0] == id.shape[0], "Inconsistent point cloud shapes"
        assert xyz.shape[0] > 1, "Ingenstive point cloud: no points found in camera data"        
        
        pc = {
            "xyz": xyz,
            "rgb": rgb,
            "id": id,
        }
        
        return pc

    def gen_pc_from_pykin_robot(self, geom="collision", sample_points=100000, seed=None, link_names=None):
        robot_meshes = []

        for link, info in self.pykin_robot.info[geom].items():
            gtype, mesh_data, h_mat = info[1], info[2], info[3]
            mesh_color = None if len(info) <= 4 else info[4]

            if link_names is not None and link not in link_names:
                continue
            if gtype != "mesh":
                continue

            meshes = mesh_data if isinstance(mesh_data, list) else [mesh_data]
            for idx, mesh in enumerate(meshes):
                if not isinstance(mesh, trimesh.Trimesh):
                    continue

                color = mesh_color
                if color is None:
                    if isinstance(mesh_data, list):
                        color = p_utils.get_mesh_color(self.pykin_robot, link, geom, idx)
                    else:
                        color = p_utils.get_mesh_color(self.pykin_robot, link, geom)

                m = mesh.copy()
                m.apply_transform(h_mat)
                if len(m.vertices) > 0:
                    robot_meshes.append(m)

        if not robot_meshes:
            raise ValueError("No robot meshes found for sampling.")

        merged_mesh = trimesh.util.concatenate(robot_meshes)

        if seed is not None:
            np.random.seed(seed)
        xyz, _ = trimesh.sample.sample_surface(merged_mesh, sample_points)
        xyz = np.asarray(xyz, dtype=np.float32)

        robot_seg_id = 1 # IsaacGym index convention
        robot_color = self._get_seg_color(robot_seg_id)
        rgb = np.repeat(robot_color[None, :], len(xyz), axis=0).astype(np.uint8)
        id = np.full((len(xyz), 1), robot_seg_id, dtype=np.int32)
        
        filtered_pc = self._filter_pc({"xyz": xyz, "rgb": rgb, "id": id})
        xyz, rgb, id = filtered_pc["xyz"], filtered_pc["rgb"], filtered_pc["id"]
        
        pc = {
            "xyz": xyz,
            "rgb": rgb,
            "id": id,
        }
        
        assert xyz.shape[0] == rgb.shape[0] == id.shape[0], "Inconsistent point cloud shapes"
        assert xyz.shape[0] > 1, "Ingenstive point cloud: no points found in camera data"        

        # if self.debug_viz:
        #     self.pointcloud_vis_debug([pc])

        return pc
    
    def save_pc(self, pc, filepath):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        xyz = np.asarray(pc["xyz"], dtype=np.float32)
        rgb = np.asarray(pc["rgb"], dtype=np.uint8)
        id = np.asarray(pc["id"], dtype=np.int32)
        
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz, dtype=o3d.core.float32))
        pcd.point["colors"] = o3d.core.Tensor(rgb, dtype=o3d.core.uint8)
        pcd.point["id"] = o3d.core.Tensor(id, dtype=o3d.core.int32)

        ok = o3d.t.io.write_point_cloud(filepath, pcd, write_ascii=False, compressed=False)
        if not ok:
            raise IOError("Failed to write point cloud")

    def load_pc(self, filepath):
        pcd = o3d.t.io.read_point_cloud(filepath)

        xyz = pcd.point["positions"].numpy().astype(np.float32)
        rgb = pcd.point["colors"].numpy().astype(np.uint8)
        id = pcd.point["id"].numpy().astype(np.int32)

        pc = {
            "xyz": xyz,
            "rgb": rgb,
            "id": id
        }
        return pc

    """
    Sample Grasp Pose
    """
    @staticmethod
    def _resample_linear_forward_bias(grasp_x, num_samples):
        """Per-grasp probability ∝ owner-object x (clamped non-negative)."""
        # multinomial is run on CPU because the CUDA kernel lacks a
        # deterministic implementation (torch_deterministic=True is set globally).
        weights = grasp_x.clamp(min=1e-6)
        probs = weights / weights.sum()
        return torch.multinomial(probs, num_samples, replacement=True)

    @staticmethod
    def _resample_softmax_forward_bias(grasp_x, num_samples, temperature):
        """Per-grasp probability ∝ exp(x / T). Smaller T -> sharper bias toward far obj."""
        probs = torch.softmax(grasp_x / temperature, dim=0)
        return torch.multinomial(probs, num_samples, replacement=True)

    @staticmethod
    def _resample_uniform(grasp_x, num_samples):
        """Uniform per-grasp probability — no bias. If each obj contributes the same
        number of grasps to the pool, this yields ~uniform per-object sampling too."""
        probs = torch.ones_like(grasp_x) / grasp_x.numel()
        return torch.multinomial(probs, num_samples, replacement=True)

    def sample_annotated_grasp_pose(self):
        """
        Build random grasp candidates from all task object candidates.

        Shapes:
            obj_grasp_poses: (num_envs, num_objs, num_grasp_pose, 7)
            task_obj_cand_index: (num_tasks, num_cand_obj)
            object pose: quat (num_envs, num_objs, 4), pos (num_envs, num_objs, 3)

        Returns:
            sample_grasps: (num_envs, max_grasp_pose, 7)
            sample_grasps_obj: (num_envs, max_grasp_pose, 1)
        """
        pose = self.ts['pose']
        oq, ot = pose['object']['quat'], pose['object']['pos']

        max_pose_seed = self.max_grasp_pose
        max_pose_per_obj = self.max_grasp_pose_per_object

        # Direction B: bias final resample toward objects farther in robot base +x.
        # Linear weighting: probability per grasp ∝ owner-obj x (clamped non-negative).

        # Task header
        total_tasks = self.task_actor_init_state.shape[1]
        print(f"\n========== [task {self.get_task_idx()+1}/{total_tasks}] grasp pose sampling ==========")

        # Pre-check: every candidate object must have at least MIN grasp poses.
        # Only print details for problematic objects; otherwise emit a single-line summary.
        MIN_GRASP_POSES_PER_OBJECT = 100
        insufficient = []
        all_counts = []
        for i in range(self.num_envs):
            for goal_idx in self.obs_ids[i]:
                poses = self.obj_grasp_poses[i][goal_idx]
                n_poses = 0 if poses is None else poses.shape[0]
                all_counts.append(n_poses)
                if n_poses < MIN_GRASP_POSES_PER_OBJECT:
                    insufficient.append((i, int(goal_idx), n_poses))

        if insufficient:
            print(f"[precheck] FAILED — insufficient grasp poses (<{MIN_GRASP_POSES_PER_OBJECT}):")
            for i, oid, n in insufficient:
                asset_root = self.object_asset[i][oid].get('asset_root', '?')
                tag = "PLACEHOLDER" if n == 1 else "INSUFFICIENT"
                print(f"  [env {i}] obj{oid} ({asset_root}): {n} poses  <-- {tag}")
            import sys
            sys.exit(1)
        else:
            print(f"[precheck] OK ({len(all_counts)} objs, "
                  f"min={min(all_counts)}, max={max(all_counts)}, "
                  f"mean={sum(all_counts)/len(all_counts):.0f})")

        sample_grasps = []
        sample_targets = []
        for i in range(self.num_envs):
            # cand_obj_index = self.task_cand_obj_index[i][self.get_task_idx()]
            cand_obj_index = self.obs_ids[i]

            obj_grasps_list = []
            obj_list = []
            for goal_idx in cand_obj_index:
                grasp_pose = self.obj_grasp_poses[i][goal_idx].to(self.tensor_args.device)
                grasp_quat, grasp_pos = grasp_pose[..., 3:7], grasp_pose[..., :3]

                oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(grasp_pose.shape[0], 1),
                              ot[i:i+1, goal_idx].repeat(grasp_pose.shape[0], 1))
                gq, gt = tf_combine(oq_i, ot_i, grasp_quat, grasp_pos)
                gq = torch.concat([gq[..., -1:], gq[..., :-1]], dim=-1) # convert xyzw to wxyz for cuRobo
                obj_grasps = torch.concat([gt, gq], dim=-1)
                pose_per_obj = min(obj_grasps.shape[0], max_pose_per_obj)
                obj_grasps = obj_grasps[torch.randint(obj_grasps.shape[0], size=(pose_per_obj,))]
                obj_grasps_list.append(obj_grasps)
                obj_list.append(torch.tensor([goal_idx] * pose_per_obj, dtype=torch.int32))

            all_obj_grasps = torch.cat(obj_grasps_list, dim=0)
            all_objs = torch.cat(obj_list, dim=0)

            # Compute per-cand-obj WORLD COM x (not root origin x), since some meshes
            # have origin far from the visual body (e.g., ShapeNet/infinigen assets).
            # world_com = root_pos + R(root_quat_xyzw) @ mesh_local_com
            cand_idx_list = [int(c) for c in cand_obj_index]
            cand_idx_t = torch.as_tensor(cand_idx_list, dtype=torch.long, device=ot.device)
            com_local = torch.stack([
                torch.as_tensor(self.obj_ref_point[i][k], dtype=ot.dtype, device=ot.device)
                for k in cand_idx_list
            ])    # (num_cand, 3)
            cand_com_world = ot[i, cand_idx_t] + quat_apply(oq[i, cand_idx_t], com_local)  # (num_cand, 3)

            # Debug: per-cand-obj x in robot base frame (sorted by x ascending)
            cand_x = cand_com_world[:, 0].detach().cpu().numpy()
            sorted_pairs = sorted(zip(cand_idx_list, cand_x), key=lambda p: p[1])
            print(f"[env {i}] cand x (robot base, +x=forward): "
                  f"min={cand_x.min():.3f} max={cand_x.max():.3f} mean={cand_x.mean():.3f} | "
                  + ", ".join(f"obj{int(idx)}={x:.3f}" for idx, x in sorted_pairs))

            # Sanity check: after COM correction, all visual centers should be in front of robot (+x).
            negative_objs = [(int(idx), float(x)) for idx, x in zip(cand_idx_list, cand_x) if x < 0]
            if negative_objs:
                print(f"[ERROR][env {i}] cand_x < 0 after COM correction for "
                      + ", ".join(f"obj{o}={x:.3f}" for o, x in negative_objs)
                      + " — object may be placed behind robot or COM metadata is incorrect.")

            # Weighted resample by grasp's owner-object world COM x.
            # Map pool obj idx -> cand list position -> precomputed COM x.
            obj_to_cand_k = {idx: k for k, idx in enumerate(cand_idx_list)}
            pool_cand_k = torch.tensor([obj_to_cand_k[int(o)] for o in all_objs.tolist()], dtype=torch.long)
            grasp_x = cand_com_world[pool_cand_k, 0].detach().cpu()    # (pool_size,)
            if self.forward_bias_mode == "linear":
                random_batch = self._resample_linear_forward_bias(grasp_x, max_pose_seed)
            elif self.forward_bias_mode == "softmax":
                random_batch = self._resample_softmax_forward_bias(
                    grasp_x, max_pose_seed, self.forward_bias_softmax_temperature)
            else:  # uniform
                random_batch = self._resample_uniform(grasp_x, max_pose_seed)

            # Per-object count of finally selected grasps (includes zero-picked candidates), sorted by count desc.
            picked_objs = all_objs[random_batch].numpy()
            uniq, counts = np.unique(picked_objs, return_counts=True)
            count_map = {int(o): int(c) for o, c in zip(uniq, counts)}
            sorted_objs = sorted((int(o) for o in cand_obj_index),
                                 key=lambda o: count_map.get(o, 0))
            parts = [f"obj{o}={count_map.get(o, 0)}" for o in sorted_objs]
            print(f"[env {i}] sampled {int(counts.sum())} grasps: " + ", ".join(parts))

            sample_grasps.append(all_obj_grasps[random_batch.to(all_obj_grasps.device)])
            sample_targets.append(all_objs[random_batch])

        sample_grasps = torch.stack(sample_grasps, dim=0)
        sample_targets = torch.stack(sample_targets, dim=0)

        # Debug Visualization
        if self.debug_viz:
            print("Debug Visualization of Sampled Grasp Poses")
            grasp_poses = []
            for i in range(sample_grasps.shape[1]):
                grasp_poses.append(Pose(sample_grasps[:, i, :3], sample_grasps[:, i, 3:7]))
            grasp_poses = Pose.vstack(grasp_poses, dim=1)
            for i in range(self.num_envs):
                grasp_success = torch.ones((sample_grasps.shape[1]), dtype=torch.bool)   
                self.grasp_vis_debug(grasp_poses[i], grasp_success, env_idx=i, show=True)

        res = {
            'grasp_poses': sample_grasps,
            'grasp_targets' : sample_targets,
        }
        
        return res

    def solve_ik(self, gp_result):
        """
        Solve batched IK for each grasp poses across environments.
        Args:
            annotated_grasp_pose: (num_envs, max_grasp_pose, 7) grasp pose candidates in robot frame

        Shapes:
            ik_result.success: (num_envs, 1)
            ik_result.solution: (num_envs, 1, dof)

        Returns:
            res:
                grasp_poses: (num_envs, max_grasp_pose, 7)
                grasp_success: (num_envs, max_grasp_pose)                
                grasp_pose_ik: (num_envs, max_grasp_pose, dof + 2)
        """
        # self.print_ik_collision_obstacle_states()
        sampled_grasp_pose = gp_result['grasp_poses']
        sampled_grasp_target = gp_result['grasp_targets']

        result_holder = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.tensor_args.device)
        ik_holder = (self.robot_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).to(self.tensor_args.device)).unsqueeze(1)
        grasp_poses, grasp_success, grasp_pose_ik = [], [], []

        for i in range(sampled_grasp_pose.shape[1]):
            grasp_candidate = sampled_grasp_pose[:, i]
            grasp_pose = Pose(grasp_candidate[..., :3], grasp_candidate[..., 3:7])
            ik_result = self.ik_solver.solve_batch_env(grasp_pose)
            torch.cuda.synchronize()

            # Use a default joint state for failed IK queries.
            ik = (ik_result.solution * ik_result.success.float().unsqueeze(-1) +
                  (1. - ik_result.success.float().unsqueeze(-1)) * ik_holder[..., :-2])

            grasp_poses.append(grasp_pose)
            grasp_success.append(result_holder & ik_result.success)
            grasp_pose_ik.append(torch.concat([ik, ik_holder[..., -2:]], dim=-1))

        grasp_poses = Pose.vstack(grasp_poses, dim=1)
        grasp_success = torch.cat(grasp_success, dim=1)
        grasp_pose_ik = torch.cat(grasp_pose_ik, dim=1)

        res = {
            "grasp_poses": grasp_poses,
            "grasp_success": grasp_success,
            "grasp_pose_ik": grasp_pose_ik,
            'grasp_targets': sampled_grasp_target,
        }
        
        # Debug Visualization
        # if self.debug_viz and self.viewer:
        if True:
            for i in range(self.num_envs):
                self.grasp_vis_debug(grasp_poses[i], grasp_success[i], env_idx=i, save=True)

        return res

    def solve_ik_batch(self, gp_result):
        """
        Same input/output contract as solve_ik, but solves all K grasp poses in a
        single batched cuRobo IK call instead of K sequential solve_batch_env calls.

        Assumes self.num_envs == 1 (single shared world config). The K candidate
        grasp poses are stacked into one batch and passed to IKSolver.solve_batch,
        eliminating the per-pose kernel-launch + cudaSynchronize overhead.
        """
        assert self.num_envs == 1, "solve_ik_batch assumes num_envs == 1"

        sampled_grasp_pose = gp_result['grasp_poses']        # (num_envs, K, 7)
        sampled_grasp_target = gp_result['grasp_targets']
        num_envs, K = sampled_grasp_pose.shape[0], sampled_grasp_pose.shape[1]

        flat = sampled_grasp_pose.reshape(num_envs * K, 7).to(self.tensor_args.device)
        grasp_pose_flat = Pose(flat[..., :3], flat[..., 3:7])

        ik_result = self.ik_solver.solve_batch(grasp_pose_flat)
        torch.cuda.synchronize()

        success_flat = ik_result.success.reshape(num_envs * K).bool()
        solution_flat = ik_result.solution.reshape(num_envs * K, -1)
        arm_dof = solution_flat.shape[-1]

        ik_holder_full = self.robot_default_dof_pos.to(solution_flat.device)
        arm_default = ik_holder_full[:arm_dof][None, :].expand(num_envs * K, -1)
        finger_default = ik_holder_full[arm_dof:][None, :].expand(num_envs * K, -1)

        ik_arm = torch.where(success_flat.unsqueeze(-1), solution_flat, arm_default)
        grasp_pose_ik_flat = torch.cat([ik_arm, finger_default], dim=-1)

        grasp_poses = Pose(
            grasp_pose_flat.position.view(num_envs, K, 3),
            grasp_pose_flat.quaternion.view(num_envs, K, 4),
            normalize_rotation=False,
        )
        grasp_success = success_flat.view(num_envs, K)
        grasp_pose_ik = grasp_pose_ik_flat.view(num_envs, K, -1)

        res = {
            "grasp_poses": grasp_poses,
            "grasp_success": grasp_success,
            "grasp_pose_ik": grasp_pose_ik,
            'grasp_targets': sampled_grasp_target,
        }

        # Debug Visualization (matches solve_ik)
        # if self.debug_viz and self.viewer:
        if True:
            for i in range(num_envs):
                self.grasp_vis_debug(grasp_poses[i], grasp_success[i], env_idx=i, save=True)

        return res

    """
    Collision Check
    """
    def _joint_distance(self, q1, q2, weights=np.ones(7, dtype=float)):
        """
        Compute the weighted L2 distance between two joint configurations.

        All joints are assumed to be revolute, so the angular difference is
        wrapped to the range [-pi, pi).

        Args:
            q1: Joint configuration of shape (7,)
            q2: Joint configuration of shape (7,)
            weights: Per-joint weights of shape (7,)

        Returns:
            Weighted joint distance
        """
        diff = q1 - q2
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(np.sqrt(np.sum(weights * diff * diff)))

    def _greedy_threshold_joint_sampling(
        self,
        q_list,
        candidate_indices,
        threshold
    ):
        """
        Perform threshold-based greedy sampling over a subset of joint configurations.

        Only configurations specified by candidate_indices are considered.
        A candidate is selected only if its distance to every previously selected
        configuration is greater than or equal to the threshold.

        Args:
            q_list: Full list of joint configurations with shape (N, 7)
            candidate_indices: Indices in q_list to consider as candidates
            threshold: Minimum allowed distance between selected configurations
            weights: Per-joint weights of shape (7,)

        Returns:
            Selected indices with respect to the original q_list
        """
        Q = np.asarray(q_list, dtype=float)

        if Q.ndim != 2 or Q.shape[1] != 7:
            raise ValueError(f"q_list must have shape (N, 7), but got {Q.shape}")

        candidate_indices = list(candidate_indices)
        if not candidate_indices:
            return []

        selected_indices = []

        for idx in candidate_indices:
            if not selected_indices:
                selected_indices.append(idx)
                continue

            if all(
                self._joint_distance(Q[idx], Q[selected_idx]) >= threshold
                for selected_idx in selected_indices
            ):
                selected_indices.append(idx)

        return selected_indices

    @staticmethod
    def _pose_pair_distance(pose_a, pose_b):
        """Pos (m) and rot (rad) distance between two 7-element poses [x,y,z,qw,qx,qy,qz]."""
        pos = float(np.linalg.norm(np.asarray(pose_a[:3]) - np.asarray(pose_b[:3])))
        dot = abs(float(np.dot(np.asarray(pose_a[3:7]), np.asarray(pose_b[3:7]))))
        dot = min(1.0, max(-1.0, dot))
        rot = 2.0 * float(np.arccos(dot))
        return pos, rot

    def _normalized_pair_dist(self, qa, pose_a, qb, pose_b, metric):
        """Distance for FPS, dispatched by metric.
          - 'pose'  : max(pos/pose_pos_floor, rot/pose_rot_floor); >= 1.0 means past floor
          - 'joint' : raw joint L2 (rad); used with floor=0 in callers (no quality cutoff)
        """
        if metric == "pose":
            pos, rot = self._pose_pair_distance(pose_a, pose_b)
            return max(pos / max(self.filter_pose_pos_floor, 1e-9),
                       rot / max(self.filter_pose_rot_floor, 1e-9))
        if metric == "joint":
            return self._joint_distance(qa, qb)
        raise ValueError(f"metric must be 'joint' or 'pose', got {metric}")

    def _farthest_point_sampling_with_floor(self, qpos, pose, candidate_ids, n_max, metric, floor=0.0):
        """Farthest-point sampling restricted to candidate_ids using the given metric.
        Stops when the next-best candidate's min-distance-to-selected < floor (quality floor).
        Set floor=0.0 to disable quality cutoff (just spread up to n_max).
        Returns selected indices (subset of candidate_ids, order = selection order).
        """
        candidate_ids = list(candidate_ids)
        if not candidate_ids:
            return []
        Q = np.asarray(qpos)
        P = np.asarray(pose)

        selected = [candidate_ids[0]]
        remaining = candidate_ids[1:]
        min_dist = {c: self._normalized_pair_dist(Q[c], P[c], Q[selected[0]], P[selected[0]], metric)
                    for c in remaining}

        while len(selected) < n_max and remaining:
            best_c = max(remaining, key=lambda c: min_dist[c])
            if min_dist[best_c] < floor:
                break
            selected.append(best_c)
            remaining.remove(best_c)
            for c in remaining:
                d = self._normalized_pair_dist(Q[c], P[c], Q[best_c], P[best_c], metric)
                if d < min_dist[c]:
                    min_dist[c] = d
        return selected

    def _global_fps_dedup_after_ik(self, ik_result):
        """Run a single global FPS pass on IK survivors (per env) using the pose
        metric (pose_pos_floor / pose_rot_floor define the quality floor). Reduces
        the K dimension of all ik_result fields; failed-IK candidates are dropped.
        Skipped when cfg.obs_data_gen.filter_global_after_ik is False.

        Assumes num_envs == 1 (downstream collision_check is the same).
        """
        if not self.filter_global_after_ik:
            return ik_result
        assert self.num_envs == 1, "_global_fps_dedup_after_ik assumes num_envs == 1"

        grasp_poses = ik_result["grasp_poses"]
        grasp_success = ik_result["grasp_success"]
        grasp_pose_ik = ik_result["grasp_pose_ik"]
        grasp_targets = ik_result["grasp_targets"]

        i = 0
        success_mask = grasp_success[i].detach().cpu().numpy().astype(bool)
        success_idx = np.where(success_mask)[0]
        if len(success_idx) == 0:
            print("[global FPS] no IK success — skipping dedup")
            return ik_result

        pos_succ = grasp_poses[i].position[success_idx].detach().cpu().numpy()
        quat_succ = grasp_poses[i].quaternion[success_idx].detach().cpu().numpy()
        pose_arr = np.concatenate([pos_succ, quat_succ], axis=-1)              # (n_succ, 7)
        qpos_arr = grasp_pose_ik[i][success_idx, :7].detach().cpu().numpy()    # (n_succ, 7)

        num_cand = len(self.obs_ids[i])
        target_n = int(max(self.filter_global_max_min,
                           min(self.filter_global_max_max,
                               num_cand * self.filter_global_max_per_obj)))

        # FPS on success-subset using pose metric (with pose floors as quality floor).
        selected_local = self._farthest_point_sampling_with_floor(
            qpos_arr, pose_arr, list(range(len(success_idx))),
            n_max=target_n, metric="pose", floor=1.0)

        n_before = int(len(success_idx))
        n_after = int(len(selected_local))
        print(f"[global FPS] IK success {n_before} -> after dedup {n_after} "
              f"(cap {target_n} = {num_cand} cand_obj x {self.filter_global_max_per_obj}, "
              f"metric=pose)")

        kept_orig = sorted(int(success_idx[k]) for k in selected_local)
        device = grasp_pose_ik.device
        kept_t = torch.as_tensor(kept_orig, dtype=torch.long, device=device)

        new_position = grasp_poses[i].position[kept_t].unsqueeze(0)
        new_quaternion = grasp_poses[i].quaternion[kept_t].unsqueeze(0)
        return {
            "grasp_poses": Pose(new_position, new_quaternion, normalize_rotation=False),
            "grasp_success": grasp_success[i][kept_t].unsqueeze(0),
            "grasp_pose_ik": grasp_pose_ik[i][kept_t].unsqueeze(0),
            "grasp_targets": grasp_targets[i][kept_t.cpu()].unsqueeze(0)
                if grasp_targets[i].device.type == "cpu"
                else grasp_targets[i][kept_t].unsqueeze(0),
        }

    def collision_check(self, ik_result): #### id related
        """
        Perform collision checking for IK results.
        world_info: in old_id (Pykin) manner
        """
        pykin_world_collision, pykin_world_collision_info = self._get_pykin_world_config()

        # map world object name to obs_id        
        obs_id_map = []
        for i in range(self.num_envs):
            name_to_new_id = {}
            world_info = pykin_world_collision_info[i]
            for k, v in world_info.items():
                if v['category'] == 'object':
                    name_to_new_id[k] = v['new_id']
                elif v['category'] == 'scene':
                    name_to_new_id[k] = v['new_id']
            obs_id_map.append(name_to_new_id)
        print("\n### obs_id_map:", obs_id_map)
            
        # collision check for each IK solution in each environment
        collision_check_time = []
        obs_qpos, obs_eef, obs_collision, obs_target, obs_grasp = [], [], [], [], []
        for i in range(self.num_envs):
            new_scene_id = self.new_scene_id[i]
            success_mask = ik_result["grasp_success"][i].detach().cpu().numpy()
            grasp_pose_ik = ik_result["grasp_pose_ik"][i].detach().cpu().numpy()
            grasp_target = ik_result["grasp_targets"][i].detach().cpu().numpy()
            grasp_pose = torch.cat([ik_result["grasp_poses"][i].position,
                                    ik_result["grasp_poses"][i].quaternion],
                                    dim=-1).detach().cpu().numpy() # in wxyz format
            qpose_per_env, eef_per_env, collision_per_env, target_per_env, grasp_per_env = [], [], [], [], []
            for j in range(len(success_mask)):
                if not success_mask[j]:
                    continue                
                
                # set robot to IK solution
                goal_qpos = grasp_pose_ik[j, :7]
                goal_eef = self._update_pykin_robot_state(goal_qpos) # in wxyz format
                # goal_obj = grasp_target[j]
                goal_obj = obs_id_map[i][f"obj_{grasp_target[j]}"] # old_id(pykin) -> new_id
                goal_grasp = grasp_pose[j] # in wxyz format

                # collision check
                start = time.time()
                result, name = self.pykin_robot_collision.in_collision_other(pykin_world_collision[i], 
                                                                            return_names=True)
                collision_check_time.append(time.time() - start)

                link_collision = {}
                if result:
                    for co1, co2 in name:
                        if co1 not in self.mesh_sample_links:
                            continue
                        # if obs_id_map[i][co2] not in self.new_obs_ids[i] and co2 != new_scene_id:
                        #     continue
                        link_collision.setdefault(co1, set()).add(obs_id_map[i][co2])
                total_collision = set().union(*link_collision.values()) if link_collision else set()
                
                # N = self.max_num_task_cand_obj + 1 # for scene
                collision_mask = np.zeros(new_scene_id + 1, dtype=np.bool8)
                collision_mask[list(total_collision)] = True
                # debug
                # print(">>>>>> in_collision_other:", result, total_collision)
                # if total_collision:            
                #     self.collision_vis_debug(total_collision)
                qpose_per_env.append(goal_qpos)
                eef_per_env.append(goal_eef)
                collision_per_env.append(collision_mask)
                target_per_env.append(goal_obj)
                grasp_per_env.append(goal_grasp)
            qpose_per_env = np.asarray(qpose_per_env) 
            eef_per_env = np.asarray(eef_per_env)
            collision_per_env = np.asarray(collision_per_env)
            target_per_env = np.asarray(target_per_env)
            grasp_per_env = np.asarray(grasp_per_env)

            # filter out results
            col_obs_dict = defaultdict(list)
            for j, col_mask in enumerate(collision_per_env):
                col_obs = np.where(col_mask)[0]
                col_obs_dict[tuple(col_obs)].append(j)
            
            # Per-(collision-pattern)-group filter:
            #   - non-clean groups: FPS(joint, no quality floor), cap=filter_max_per_group
            #   - clean group:      deferred; capped later via filter_clean_max_ratio
            collision_filtered_ids = []
            clean_ids = []
            for k, candidate_ids in col_obs_dict.items():
                if k == (new_scene_id,):
                    continue
                if len(k) == 0:
                    clean_ids = list(candidate_ids)
                    continue
                filtered_ids = self._farthest_point_sampling_with_floor(
                    qpose_per_env, grasp_per_env, candidate_ids,
                    n_max=self.filter_max_per_group, metric="pose",
                    floor=self.filter_cap_floor)
                collision_filtered_ids.extend(filtered_ids)

            # Apply clean-ratio cap so clean <= ratio * total_final_M.
            n_collision = len(collision_filtered_ids)
            ratio = self.filter_clean_max_ratio
            if not clean_ids:
                clean_kept = []
            elif ratio >= 1.0:
                clean_kept = clean_ids
            elif ratio <= 0.0:
                clean_kept = []
            else:
                # clean / (clean + collision) <= ratio  =>  clean <= ratio*collision/(1-ratio)
                clean_cap = int(ratio * n_collision / (1.0 - ratio))
                if len(clean_ids) > clean_cap:
                    clean_kept = self._farthest_point_sampling_with_floor(
                        qpose_per_env, grasp_per_env, clean_ids,
                        n_max=clean_cap, metric="pose",
                        floor=self.filter_cap_floor)
                else:
                    clean_kept = clean_ids
            n_clean_raw = len(clean_ids)
            n_clean = len(clean_kept)
            denom = max(n_collision + n_clean, 1)
            print(f"[clean cap] raw={n_clean_raw}, kept={n_clean} (collision={n_collision}, "
                  f"clean ratio after cap={100*n_clean/denom:.1f}%, target<={100*ratio:.0f}%)")

            total_filtered_ids = collision_filtered_ids + clean_kept
            
            # Save vis of filtered grasps. green = clean (no collision), red = has collision.
            if len(total_filtered_ids) > 0:
                filt_grasps = grasp_per_env[total_filtered_ids]   # (M, 7) wxyz
                filt_pose = Pose(
                    torch.from_numpy(filt_grasps[:, :3]).to(self.tensor_args.device),
                    torch.from_numpy(filt_grasps[:, 3:7]).to(self.tensor_args.device),
                )
                # Clean = no object/scene collision (collision_mask all False)
                filt_collision = collision_per_env[total_filtered_ids]            # (M, S) bool
                filt_clean = ~filt_collision.any(axis=-1)                          # (M,) bool
                filt_success = torch.as_tensor(filt_clean, dtype=torch.bool)
                self.grasp_vis_debug(filt_pose, filt_success, env_idx=i, save=True, subdir="collision")

            obs_qpos.append(qpose_per_env[total_filtered_ids])
            obs_eef.append(eef_per_env[total_filtered_ids])
            obs_collision.append(collision_per_env[total_filtered_ids])
            obs_target.append(target_per_env[total_filtered_ids])
            obs_grasp.append(grasp_per_env[total_filtered_ids])
        
        obs_qpos = np.asarray(obs_qpos, dtype=np.float32)
        obs_eef = np.asarray(obs_eef, dtype=np.float32) 
        obs_collision = np.asarray(obs_collision, dtype=np.bool8)
        obs_target = np.asarray(obs_target, dtype=np.int32)
        obs_grasp = np.asarray(obs_grasp, dtype=np.float32)
        print(f"Average collision check time: {np.mean(collision_check_time):.4f} seconds over {len(collision_check_time)} checks")

        # DIAG: pairwise grasp pose distances in final filtered dataset (per env)
        for env_i in range(obs_grasp.shape[0]):
            pos = obs_grasp[env_i, :, :3]   # (M, 3)
            quat = obs_grasp[env_i, :, 3:7] # (M, 4) wxyz
            M = len(pos)
            if M < 2:
                print(f"[DIAG][env {env_i}] M={M}, skipping pairwise (need >=2).")
                continue
            pos_d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            dot = np.clip(np.abs(quat @ quat.T), -1.0, 1.0)
            rot_d = 2.0 * np.arccos(dot)
            mask = ~np.eye(M, dtype=bool)
            pos_v = pos_d[mask]
            rot_v = rot_d[mask]
            n_pairs = mask.sum() // 2   # undirected
            close_pos_2cm = int(((pos_d < 0.02) & mask).sum() // 2)
            close_pos_5cm = int(((pos_d < 0.05) & mask).sum() // 2)
            close_both = int((((pos_d < 0.05) & (rot_d < 0.3)) & mask).sum() // 2)
            print(f"[DIAG][env {env_i}] final M={M}, pairs={n_pairs}"
                  f" | pos(m) min={pos_v.min():.3f} mean={pos_v.mean():.3f}"
                  f" | rot(rad) min={rot_v.min():.3f} mean={rot_v.mean():.3f}"
                  f" | <2cm:{close_pos_2cm} <5cm:{close_pos_5cm} <5cm&<0.3rad:{close_both}")
        
        res = {
            'obs_qpos': obs_qpos,
            'obs_eef': obs_eef,
            'obs_collision': obs_collision,
            'obs_target': obs_target,
            'obs_grasp': obs_grasp,
            'obs_id_map': obs_id_map,
        }
        
        return res
    
    """
    Your Solution
    """
    def set_task_snapshot(self):
        # Get pointcloud from IsaacGym
        pc_cams_gym = [self.gen_pc_from_camera(env_idx=i) for i in range(self.num_envs)]
        
        # Get task_cand_obj_index, valid_obj_index in old_id (Pykin) manner # TODO: rename Pykin old_id manner, should be asset_config manner?
        task_cand_obj_index = []
        valid_obj_index = [] # only keep task candidate objects that are visible in camera 
        for i in range(self.num_envs):
            valid_ids = set(pc_cams_gym[i]['id'].flatten())
            valid_ids = [x-4 for x in valid_ids if x >= IsaacGymId.OBJECT_START] # old_id (Pykin), only keep objects (exclude robot/table/scene)
            task_cand_obj_index.append(sorted(self.task_cand_obj_index[i][self.get_task_idx()])) # old_id (Pykin)
            valid_obj_index.append(valid_ids)            
        assert all(set(valid_obj_index[i]) <= set(task_cand_obj_index[i]) for i in range(self.num_envs)), \
                "Valid object index should be a subset of task candidate object index"
        
        # Update index mapping from old_id (Pykin) to new_id (obs_data)
        # new_id method can be either 'all' or 'valid``
        pc_new_id_method = self.pc_new_id_method
        new_id_map = [] # old_id (Pykin) to new_id (obs_data)
        new_scene_id = [] # new_id (obs_data) for scene
        new_obs_ids = [] # new_id (obs_data) for objects+scene
        obs_ids = [] # old_id (Pykin)
        for i in range(self.num_envs):
            if pc_new_id_method == 'all':
                new_id_map.append({old_id: new_id for new_id, old_id in enumerate(task_cand_obj_index[i])}) ### all object in scene
                new_scene_id.append(len(task_cand_obj_index[i]))
                new_obs_ids.append(list(range(len(task_cand_obj_index[i])+1))) ### all objects + scene
                obs_ids.append(task_cand_obj_index[i]) ### old_id (Pykin) for all objects in scene
            elif pc_new_id_method == 'valid':
                new_id_map.append({old_id: new_id for new_id, old_id in enumerate(valid_obj_index[i])}) ### only valid object in camera
                new_scene_id.append(len(valid_obj_index[i]))
                new_obs_ids.append(list(range(len(valid_obj_index[i])+1))) ### valid objects + scene
                obs_ids.append(valid_obj_index[i]) ### old_id (Pykin) for valid objects in camera
            else:
                raise ValueError(f"Invalid pc_new_id_method: {pc_new_id_method}")
        self.new_id_map = new_id_map
        self.new_scene_id = new_scene_id
        self.new_obs_ids = new_obs_ids ### remove?
        self.obs_ids = obs_ids
        if self.max_num_task_cand_obj is None:
            self.max_num_task_cand_obj = max(new_scene_id[i] for i in range(self.num_envs))
        
        # Update pc_cams id from old_id (IsaacGym) to new_id (obs_data)
        pc_cams = []
        for i in range(self.num_envs):
            pc_cams_id = pc_cams_gym[i]['id'].copy()
            pc_cams_id[pc_cams_gym[i]['id'] == IsaacGymId.SCENE] = new_scene_id[i] # old_id (IsaacGym) to new_id (obs_data)
            for old_id, new_id in new_id_map[i].items():
                pc_cams_id[pc_cams_gym[i]['id'] == (old_id+IsaacGymId.OBJECT_START)] = new_id # old_id (IsaacGym) to new_id (obs_data)
            pc_cams.append({'xyz': pc_cams_gym[i]['xyz'].copy(), 
                            'rgb': pc_cams_gym[i]['rgb'].copy(), 
                            'id': pc_cams_id})
        
        # Update task snapshot data
        pose = self._get_pose_in_robot_frame()
        self.ts = {'pose': pose, 'pc_cams': pc_cams}
        
        print("="*100)
        print("Object Candidate Index:", self.obs_ids)

        # Debug Visualization
        if self.debug_viz:            
            seg_pc_cams = defaultdict(list)
            for i in range(self.num_envs):
                pc_cams_xyz = pc_cams[i]['xyz'].copy()
                pc_cams_rgb = pc_cams[i]['rgb'].copy()
                pc_cams_id = pc_cams[i]['id'].copy()
                print(f"[DEBUG] new_obs_ids({len(self.new_obs_ids[i])}): {self.new_obs_ids[i]}")
                print(f"[DEBUG] pc_cams_id({len(set(pc_cams_id.flatten()))}): {set(pc_cams_id.flatten())}")
                for new_id in self.new_obs_ids[i]:
                    mask = (pc_cams_id.reshape(-1) == new_id)
                    seg_pc_cams[new_id].append({'xyz': pc_cams_xyz[mask],
                                                'rgb': pc_cams_rgb[mask],
                                                'id': pc_cams_id[mask]})
                    print(f"[DEBUG] Environment {i}, ID {new_id}: {len(seg_pc_cams[new_id][i]['xyz'])} points, rgb: {set(tuple(rgb) for rgb in seg_pc_cams[new_id][i]['rgb'])}")
                    self.pointcloud_vis_debug(seg_pc_cams[new_id])
               
    def make_obs_data(self, col_result):
        obs_dir = self.obs_dir
        # Remove stale PLYs from prior runs of the same task to keep PLY count
        # in sync with the freshly written h5 (collision rows == #robot PLYs).
        import glob
        for old in glob.glob(f"{obs_dir}/ply/pc_robot_t{self._task_idx}_*.ply"):
            os.remove(old)
        pc_cams, pc_robots = [], []
        for i in range(self.num_envs):
            pc_cam = self.ts['pc_cams'][i]
            pc_cam_path = f"ply/pc_cam_t{self._task_idx}.ply"
            self.save_pc(pc_cam, f"{obs_dir}/{pc_cam_path}")
            for j in range(len(col_result['obs_qpos'][i])):
                col_obs = np.where(col_result['obs_collision'][i][j])[0]
                col_qpos = col_result['obs_qpos'][i][j]
                self._update_pykin_robot_state(col_qpos)
                pc_robot = self.gen_pc_from_pykin_robot(sample_points=self.mesh_sample_points, 
                                                        link_names=self.mesh_sample_links)
                pc_robot_path = f"ply/pc_robot_t{self._task_idx}_{j}.ply"
                self.save_pc(pc_robot, f"{obs_dir}/{pc_robot_path}")
                pc_cams.append(pc_cam_path)
                pc_robots.append(pc_robot_path)
        
        pc_cams = np.asarray(pc_cams)
        pc_robots = np.asarray(pc_robots)
        eef = col_result['obs_eef'].reshape(-1, *col_result['obs_eef'].shape[2:])
        qpos = col_result['obs_qpos'].reshape(-1, *col_result['obs_qpos'].shape[2:])
        collision = col_result['obs_collision'].reshape(-1, *col_result['obs_collision'].shape[2:])
        grasp = col_result['obs_grasp'].reshape(-1, *col_result['obs_grasp'].shape[2:])
        targets = col_result['obs_target'].reshape(-1)
        obs_ids = np.asarray(self.obs_ids).reshape(-1)

        obs_data = {
            'eef': eef,
            'qpos': qpos,
            'collision': collision,
            'grasp': grasp,
            'pc_cam': pc_cams,
            'pc_robot': pc_robots,
            'target': targets,
            'obs_ids': obs_ids
        }

        print("Final dataset shapes: eef {}, qpos {}, collision {}, grasp {}, pc_cams {}, pc_robots {}, obs_ids {}".format(
            eef.shape, qpos.shape, collision.shape, grasp.shape, pc_cams.shape, pc_robots.shape, obs_ids.shape
        ))
        
        return obs_data

    def make_grn_data(self, col_result=None):        
        def _bounding_box(mesh, bb_type='axis_aligned'):
            if bb_type == 'axis_aligned':
                bb = mesh.bounding_box                
                extents, transform = bb.extents, bb.primitive.transform
            elif bb_type == 'oriented':
                bb = mesh.bounding_box_oriented
                extents, transform = bb.extents, bb.primitive.transform
            elif bb_type == 'yaw_only':
                vertices = np.asarray(mesh.vertices)
                transform_2d, extent_xy = trimesh.bounds.oriented_bounds_2D(vertices[:, :2])
                transform_2d_inv = np.linalg.inv(transform_2d)
                z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
                extents = np.array([
                    extent_xy[0],
                    extent_xy[1],
                    z_max - z_min,
                ])
                transform = np.eye(4)
                transform[:2, :2] = transform_2d_inv[:2, :2]
                transform[:2, 3] = transform_2d_inv[:2, 2]
                transform[2, 3] = (z_min + z_max) * 0.5
            else:
                raise ValueError(f"Invalid bounding box type: {bb_type}")
            return extents, transform
        
        def _get_grasp_type(object_pose_base, grasp_pose_base):
            """
            object_pose_base: (x, y, z, yaw)
                object pose in robot base frame.
                yaw is rotation around base Z axis, in radians.

            grasp_pose_base: (x, y, z, qw, qx, qy, qz)
                grasp/TCP pose in robot base frame.

            return:
                one of ["Top", "Bottom", "Front", "Rear", "Right", "Left"]

            Convention:
                grasp local +Z = approach direction from TCP/gripper toward object

                approach close to object local -Z -> Top
                approach close to object local +Z -> Bottom
                approach close to object local -X -> Front
                approach close to object local +X -> Rear
                approach close to object local +Y -> Right
                approach close to object local -Y -> Left
            """
            _, _, _, yaw = object_pose_base
            _, _, _, qw, qx, qy, qz = grasp_pose_base

            R_obj_base = tr.rotation_matrix(yaw, [0, 0, 1])[:3, :3]

            # trimesh.transformations expects quaternion as [qw, qx, qy, qz]
            R_grasp_base = tr.quaternion_matrix([qw, qx, qy, qz])[:3, :3]

            # grasp local +Z axis in robot base frame.
            # If your convention is local -Z = TCP/gripper -> object,
            # change this line to: approach_base = -R_grasp_base[:, 2]
            approach_base = R_grasp_base[:, 2]

            # express approach direction in object frame
            approach_obj = R_obj_base.T @ approach_base

            # if abs(z) >= abs(x) and abs(z) >= abs(y):
            #     return GraspType.TOP if z < 0 else GraspType.BOTTOM
            # if abs(x) >= abs(y):
            #     return GraspType.FRONT if x < 0 else GraspType.REAR
            # return GraspType.RIGHT if y > 0 else GraspType.LEFT
        
            # 6가지 방향 벡터 중 가장 가까운 방향 선택
            directions = [(GraspType.directions[gt], gt) 
                            for gt in [GraspType.TOP, GraspType.BOTTOM, GraspType.FRONT, 
                                        GraspType.REAR, GraspType.RIGHT, GraspType.LEFT]]
            return max(directions, key=lambda d: np.dot(approach_obj, d[0]))[1]
        
        # 1. make .pt file
        grn_data = []
        grn_id_map = [] # name to grn_id map
        obs_to_grn = []
        grn_to_obs = []
        _, pykin_world_collision_info = self._get_pykin_world_config(add_collider=True)
        for i in range(self.num_envs):
            scene = [self.get_task_idx()] ## should be modified if multiple env per task
            # set node_data (x, pos, mask, frame_ids, base_mask) from pkin world
            features, poses, masks, frame_ids, base_masks = [], [], [], [], []
            grn_id_map_env = {}
            obs_to_grn_env = defaultdict(list)
            grn_to_obs_env = []
            for grn_id, (k, v) in enumerate(pykin_world_collision_info[i].items()):
                grn_id_map_env[k] = grn_id # name -> grn_id
                obs_to_grn_env[v['new_id']].append(grn_id) # obs_id -> grn_id
                grn_to_obs_env.append(v['new_id']) # grn_id -> obs_id
                if v['category'] == 'object':
                    v['gparam'].apply_transform(v['transform'])
                    bb_extents, bb_transform = _bounding_box(v['gparam'], bb_type='yaw_only')
                    v['bb'] = trimesh.creation.box(extents=bb_extents, transform=bb_transform)
                    v['bb'].visual.face_colors = [255, 0, 0, 100]
                elif v['category'] == 'scene':
                    bb_extents = v['gparam']
                    bb_transform = v['transform']
                    v['bb'] = trimesh.creation.box(extents=bb_extents, transform=bb_transform)
                    v['bb'].visual.face_colors = [0, 255, 0, 100]
                    
                l, w, h = bb_extents
                x, y, z = bb_transform[0, 3], bb_transform[1, 3], bb_transform[2, 3]
                th = np.arctan2(bb_transform[1, 0], bb_transform[0, 0])
                features.append([l, w, h, x, y, z, th])
                poses.append([x, y, z, th])
                if v['category'] == 'object':
                    masks.append(True)
                    frame_ids.append(-1)
                    base_masks.append(False)
                elif v['category'] == 'scene':
                    masks.append(False)
                    frame_ids.append(-1)
                    base_masks.append(False)
                
            # set edge_data (edge_index, edge_attr, blocking_mask) from node_data
            edge_index, edge_attr, blocking_mask = [], [], []
            for trg in range(len(features)):
                if masks[trg] == False:
                    continue
                others = [idx for idx in range(len(features)) if idx != trg]
                # proximity edges
                for src in others:
                    if grn_utils.is_neighbor(features[trg][:3], features[src][:3], poses[trg], poses[src]):
                        edge_index.append([src, trg])
                        edge_attr.append([1.0, 0.0])
                        blocking_mask.append(True)
                # self-loop edges
                edge_index.append([trg, trg])
                edge_attr.append([0.0, 1.0])
                blocking_mask.append(False)
            # set empty label_data # label_data: IK_labels, F_labels, GO_labels
            IK_labels = torch.zeros((len(features), 5), dtype=torch.float32)
            F_labels = torch.zeros((len(features), 6), dtype=torch.float32)
            GO_labels = torch.zeros((len(edge_attr), 5), dtype=torch.float32)
            
            grn_data.append(Data(
                x=torch.tensor(features, dtype=torch.float32),
                pos=torch.tensor(poses, dtype=torch.float32),
                mask=torch.tensor(masks, dtype=torch.bool),
                frame_id=torch.tensor(frame_ids, dtype=torch.int64),
                base_mask=torch.tensor(base_masks, dtype=torch.bool),
                scene=torch.tensor(scene, dtype=torch.int64),
                edge_index=torch.tensor(edge_index, dtype=torch.int64).reshape(2, -1),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                blocking_mask=torch.tensor(blocking_mask, dtype=torch.bool),
                IK_labels=IK_labels, F_labels=F_labels, GO_labels=GO_labels))
            grn_id_map.append(grn_id_map_env)
            obs_to_grn.append(obs_to_grn_env)
            grn_to_obs.append(grn_to_obs_env)
        
        print("\n### grn_id_map:", grn_id_map)
        print("\n### obs_to_grn:", obs_to_grn)
        
        num_objs = grn_data[0]['x'].shape[0]
        num_edges = grn_data[0]['edge_attr'].shape[0]
        assert grn_data[0]['scene'].shape == (1,), f"Expected scene shape (1,), got {grn_data[0]['scene'].shape}"
        assert grn_data[0]['x'].shape == (num_objs, 7), f"Expected node features shape ({num_objs}, 7), got {grn_data[0]['x'].shape}"
        assert grn_data[0]['mask'].shape == (num_objs,), f"Expected node masks shape ({num_objs},), got {grn_data[0]['mask'].shape}"
        assert grn_data[0]['frame_id'].shape == (num_objs,), f"Expected node frame ids shape ({num_objs},), got {grn_data[0]['frame_id'].shape}"
        assert grn_data[0]['base_mask'].shape == (num_objs,), f"Expected node base masks shape ({num_objs},), got {grn_data[0]['base_mask'].shape}"
        assert grn_data[0]['edge_index'].shape == (2, num_edges), f"Expected edge index shape (2, {num_edges}), got {grn_data[0]['edge_index'].shape}"
        assert grn_data[0]['edge_attr'].shape == (num_edges, 2), f"Expected edge attributes shape ({num_edges}, 2), got {grn_data[0]['edge_attr'].shape}"
        assert grn_data[0]['blocking_mask'].shape == (num_edges,), f"Expected blocking mask shape ({num_edges},), got {grn_data[0]['blocking_mask'].shape}"
        assert grn_data[0]['IK_labels'].shape == (num_objs, 5), f"Expected IK labels shape ({num_objs}, 5), got {grn_data[0]['IK_labels'].shape}"
        assert grn_data[0]['F_labels'].shape == (num_objs, 6), f"Expected F labels shape ({num_objs}, 6), got {grn_data[0]['F_labels'].shape}"
        assert grn_data[0]['GO_labels'].shape == (num_edges, 5), f"Expected GO labels shape ({num_edges}, 5), got {grn_data[0]['GO_labels'].shape}"
        
        # 2. make .h5 file for grn (N grasp poses)
        # get grasp type from grasp pose and obj (from col_result?)
        oq, ot = self.ts['pose']['object']['quat'], self.ts['pose']['object']['pos'] # xyzw
        oq, ot = oq.cpu().numpy(), ot.cpu().numpy()
        grn_h5_data = []
        for i in range(self.num_envs):
            grn_gtypes, grn_targets, grn_collisions = [], [], []
            for j in range(col_result['obs_qpos'][i].shape[0]):
                target_obs_id = col_result['obs_target'][i][j] # obs_id
                target_grn_ids = obs_to_grn[i][target_obs_id] # obs_id to grn_id
                assert len(target_grn_ids) == 1, f"target with obs_id {target_obs_id} should be a single object with grn_id {target_grn_ids}"
                target_xyzth = grn_data[i]['pos'][target_grn_ids[0]].numpy() # (x, y, z, th)
                target_grasp = col_result['obs_grasp'][i][j] # (x, y, z, qw, qx, qy, qz)
                grasp_type = _get_grasp_type(target_xyzth, target_grasp)
                grn_gtypes.append(grasp_type)
                grn_targets.append(target_grn_ids[0]) # grn_id
                grn_collision = np.zeros(grn_data[i]['x'].shape[0], dtype=np.bool8)
                for col_obs_id in np.where(col_result['obs_collision'][i][j])[0]:
                    col_grn_ids = obs_to_grn[i][col_obs_id]
                    for col_grn_id in col_grn_ids:
                        grn_collision[col_grn_id] = True
                grn_collisions.append(grn_collision)                 
            grn_h5_data.append({
                'grn_target': np.array(grn_targets, dtype=np.int32),       # (num_grasp_pose, 1)
                'grn_eef': col_result['obs_eef'][i],                       # (num_grasp_pose, 7)
                'grn_grasp': col_result['obs_grasp'][i],                   # (num_grasp_pose, 7)
                'grn_gtype': np.array(grn_gtypes, dtype=np.int32),         # (num_grasp_pose, 1)
                'grn_collision': np.array(grn_collisions, dtype=np.bool8), # (num_grasp_pose, num_grn_id)
                'grn_id_map': grn_id_map[i],                               # (num_grn_id,)
                'grn_obs_map': np.array(grn_to_obs, dtype=np.int32)        # (num_grn_id,)
            })
        
        if self.debug_viz:
            self.grn_vis_debug(grn_data, grn_h5_data, pykin_world=pykin_world_collision_info) # visualize all grasp poses
            for target_id in set(grn_h5_data[0]['grn_target']):
                self.grn_vis_debug(grn_data, grn_h5_data, target_id=target_id)
        
        # 3. save grn_data and grn_h5_data
        grn_dir = f"{self.obs_dir}/grn"
        torch.save((grn_data[0], None), f"{grn_dir}/grouped_data_t{self.get_task_idx()}.pt")
        with h5py.File(f"{grn_dir}/grouped_data_t{self.get_task_idx()}.h5", "w") as f:
            f.create_dataset("grn_target", data=grn_h5_data[0]['grn_target'], dtype=np.int32)
            f.create_dataset("grn_eef", data=grn_h5_data[0]['grn_eef'], dtype=np.float32)
            f.create_dataset("grn_grasp", data=grn_h5_data[0]['grn_grasp'], dtype=np.float32)
            f.create_dataset("grn_gtype", data=grn_h5_data[0]['grn_gtype'], dtype=np.int32)
            f.create_dataset("grn_collision", data=grn_h5_data[0]['grn_collision'], dtype=np.bool8)
            # f.create_dataset("grn_id_map", data=np.array(list(grn_h5_data[0]['grn_id_map'].values()), dtype=np.int32), dtype=np.int32)
        
        return grn_data, grn_h5_data

    def solve(self):
        log = {}

        # self.set_target_color()
        self._solution_video = []
        self._video_frame = 0

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()

        rgb, seg = self.get_camera_image(rgb=True, seg=False)
        self.log_video(rgb)
        
        self.set_task_snapshot()
                
        # Sample Good Grasp Pose
        gp_result = self.sample_annotated_grasp_pose()

        # Solve IK
        # ik_result = self.solve_ik(gp_result)
        ik_result = self.solve_ik_batch(gp_result)
        print("Success IK solutions:", int(ik_result["grasp_success"].sum()))

        # Global pose dedup on IK survivors (before the expensive collision check)
        ik_result = self._global_fps_dedup_after_ik(ik_result)

        # Collision Checking
        col_result = self.collision_check(ik_result)
        # plot_collision_statistics(col_result['obs_collision'][0])

        # Make obs data
        obs_data = self.make_obs_data(col_result)
        grn_pt_data, grn_h5_data = self.make_grn_data(col_result)                
        
        # Save obs data 
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(f"{self.obs_dir}/obstruction_data_t{self.get_task_idx()}.h5", "w") as f:
            f.create_dataset("eef", data=np.asarray(obs_data['eef'], dtype=np.float32))
            f.create_dataset("qpos", data=np.asarray(obs_data['qpos'], dtype=np.float32))
            f.create_dataset("collision", data=np.asarray(obs_data['collision'], dtype=np.bool8))
            f.create_dataset("grasp", data=np.asarray(obs_data['grasp'], dtype=np.float32))
            f.create_dataset("pc_cam", data=np.asarray(obs_data['pc_cam'], dtype=str_dtype))
            f.create_dataset("pc_robot", data=np.asarray(obs_data['pc_robot'], dtype=str_dtype))
            f.create_dataset("target", data=np.asarray(obs_data['target'], dtype=np.int32))
            f.create_dataset("obs_ids", data=np.asarray(obs_data['obs_ids'], dtype=np.int32))

        # self.set_default_color()

        return image_to_video(self._solution_video), log
    
    """
    Debug Visualization
    """        
    def grasp_vis_debug(self, grasp_pose, grasp_success, env_idx=0, show=False, save=False, subdir="ik"):
        """
        Visualize grasp pose and environment in a separate window using trimesh.
        
        Args:
            grasp_pose: Pose tensor of shape (num_grasp_pose, 7) representing candidate grasp poses
            grasp_success: Boolean tensor of shape (num_grasp_pose,) indicating IK success for each grasp pose
            env_idx: Index of the environment to visualize
            show: Whether to display the visualization window
            save: Whether to save the visualization
        """        
        pose = self.ts['pose']
        
        scene = trimesh.Scene()
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        table_pose = pose['table']
        dq = torch.concat([table_pose['quat'][..., -1:], table_pose['quat'][..., :-1]], dim=-1)
        dq, dt = dq.cpu().numpy(), table_pose['pos'].cpu().numpy()

        table_translation = tr.translation_matrix(dt[env_idx])
        table_rotation = tr.quaternion_matrix(dq[env_idx])

        table = trimesh.creation.box(extents=self.table_asset[env_idx]['dim'], transform=table_translation @ table_rotation)
        scene.add_geometry(table)

        scene_pose = pose['scene']

        sq = torch.concat([scene_pose['quat'][..., -1:], scene_pose['quat'][..., :-1]], dim=-1)
        sq, st = sq.cpu().numpy(), scene_pose['pos'].cpu().numpy()

        # vis environment 0
        scene_translation = tr.translation_matrix(st[env_idx])
        scene_rotation = tr.quaternion_matrix(sq[env_idx])

        # vis scene
        for f in self.scene_asset[env_idx]['files']:
            mesh = trimesh.load(f)
            mesh = mesh.apply_transform(scene_translation @ scene_rotation)
            scene.add_geometry(mesh)

        object_poses = pose['object']
        oq = torch.concat([object_poses['quat'][..., -1:], object_poses['quat'][..., :-1]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_poses['pos'].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
            if i in self.task_cand_obj_index[env_idx][self.get_task_idx()]:
                trans = tr.translation_matrix(ot[env_idx][i])
                rot = tr.quaternion_matrix(oq[env_idx][i])
                mesh = o['mesh'].copy().apply_transform(trans @ rot)
                scene.add_geometry(mesh)

        # grasp pose
        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        for i in range(grasp_pose.position.shape[0]):
            trans = tr.translation_matrix(grasp_pose.position[i].cpu().numpy())
            rot = tr.quaternion_matrix(grasp_pose.quaternion[i].cpu().numpy())
            grasp = trans @ rot @ vis_rot
            if grasp_success[i]:
                command_marker = create_gripper_marker([0, 255, 0]).apply_transform(grasp)
            else:
                command_marker = create_gripper_marker([255, 0, 0]).apply_transform(grasp)
            scene.add_geometry(command_marker)
        
        cam_pos = np.array([-2.0, 0.0, 2.5], dtype=float)
        target = np.array([0.0, 0.0, 0.5], dtype=float)
        scene.camera_transform = make_camera_transform(cam_pos, target)

        print("\t[DEBUG] grasp_vis_debug: showing env {} with {} grasp poses (green: success, red: failure)".format(env_idx, grasp_pose.position.shape[0]))
        if show:
            scene.show()
        # [To-do] always return png and write to file outside of this function
        if save:
            filepath = f"{self.obs_dir}/{subdir}/test_e{env_idx}_t{self.get_task_idx()}.png"
            dirname = os.path.dirname(filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            png = scene.save_image(resolution=(1600, 1600))
            with open(filepath, "wb") as f:
                f.write(png)

    def collision_vis_debug(self, pykin_world_collision_info, objs_in_collision, env_idx=0):
        scene = trimesh.Scene()
        
        # vis axis
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)
        # vis robot
        scene = apply_robot_to_scene(trimesh_scene=scene, 
                                    robot=self.pykin_robot, 
                                    geom=self.pykin_robot_collision.geom)
        # vis world collision objects
        for c_name, c_info in pykin_world_collision_info[env_idx].items():
            gparam = c_info['gparam']
            transform = c_info['transform']
            # c_obj = _objs[c_name]["obj"]
            # transform = Transform(pos=c_obj.getTranslation(), 
            #                       rot=c_obj.getQuatRotation()).h_mat
            if isinstance(c_info['gparam'], trimesh.Trimesh):
                if c_name in objs_in_collision:
                    gparam = copy.deepcopy(gparam)
                    if isinstance(c_info['gparam'].visual, trimesh.visual.texture.TextureVisuals):
                        gparam.visual = gparam.visual.to_color()
                    gparam.visual.face_colors = [255, 0, 0, 150]
                scene.add_geometry(gparam, node_name=c_name, transform=transform)
            else:
                scene.add_geometry(trimesh.creation.box(extents=gparam,
                                                        transform=transform))  
        
        print("\t[DEBUG] collision_vis_debug: showing env {} with {} objects in collision (red)".format(env_idx, len(objs_in_collision)))
        scene.show()
  
    def grn_vis_debug(self, grn_data, grn_h5_data, **kwargs):
        scene = trimesh.Scene()
        
        # target_id = 0
        # target_dir = None 
        env_idx = kwargs.get('env_idx', 0)
        target_id = kwargs.get('target_id', None)
        target_dir = kwargs.get('target_dir', None)
        pykin_world = kwargs.get('pykin_world', None)
        
        # vis axis
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)
        
        if pykin_world is not None:
            # vis world collision objects
            for c_name, c_info in pykin_world[env_idx].items():
                gparam = c_info['gparam']
                transform = c_info['transform']
                if isinstance(c_info['gparam'], trimesh.Trimesh):
                    gparam = copy.deepcopy(gparam)
                    scene.add_geometry(gparam)
                else:
                    scene.add_geometry(trimesh.creation.box(extents=gparam,
                                                            transform=transform))
                if c_info['category'] == 'object':
                    c_info['bb'].visual.face_colors = [255, 0, 0, 100]
                    scene.add_geometry(c_info['bb'])
                elif c_info['category'] == 'scene':
                    c_info['bb'].visual.face_colors = [0, 255, 0, 100]
                    scene.add_geometry(c_info['bb'])
        
        else:
            # vis objects and scene with bounding boxes
            for i in range(grn_data[env_idx]['x'].shape[0]):
                l, w, h = grn_data[env_idx]['x'][i][:3].numpy()
                x, y, z, th = grn_data[env_idx]['x'][i][3:].numpy()
                transform = tr.compose_matrix(translate=[x, y, z], angles=[0, 0, th])
                bb = trimesh.creation.box(extents=[l, w, h], transform=transform)
                axis = trimesh.creation.axis(origin_size=0.008, axis_length=0.04, 
                                            axis_radius=0.004, transform=transform)
                
                if grn_data[env_idx]['mask'][i]: # object
                    if target_id is not None and i != target_id:
                        continue
                    bb.visual.face_colors = [255, 0, 0, 100]
                    scene.add_geometry(bb)
                    scene.add_geometry(axis)
                else: # scene
                    bb.visual.face_colors = [0, 255, 0, 100]
                    scene.add_geometry(bb)
            
            # vis grasps
            vis_rot = np.array([[0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

            for i in range(grn_h5_data[env_idx]['grn_grasp'].shape[0]):
                if target_id is not None and grn_h5_data[env_idx]['grn_target'][i] != target_id: 
                    continue
                if target_dir is not None and grn_h5_data[env_idx]['grn_gtype'][i] != target_dir:
                    continue
                grasp_pose = grn_h5_data[env_idx]['grn_grasp'][i]
                trans = tr.translation_matrix(grasp_pose[:3])
                rot = tr.quaternion_matrix(grasp_pose[3:7]) # in wxyz
                grasp_transform = trans @ rot @ vis_rot
                grasp_type = grn_h5_data[env_idx]['grn_gtype'][i]
                grasp_color = GraspType.colors.get(grasp_type, [128, 128, 128])
                grasp_marker = create_gripper_marker(grasp_color).apply_transform(grasp_transform)
                scene.add_geometry(grasp_marker)

        print("\t[DEBUG] grn_vis_debug: showing env {}".format(env_idx))
        
        # cam_pos = np.array([-2.0, 0.0, 2.5], dtype=float)
        cam_pos = np.array([-1.0, 0.0, 1.5], dtype=float)
        target = np.array([0.0, 0.0, 0.5], dtype=float)
        scene.camera_transform = make_camera_transform(cam_pos, target)
        scene.show()

    def pointcloud_vis_debug(self, pc_list):
        """
        Visualize multiple point clouds with Open3D.

        Parameters
        ----------
        pc_list : list of dict
            Each dict should contain:
                - "xyz": (N, 3) np.ndarray
                - "rgb": (N, 3) np.ndarray
                - "id": (N, 1) np.ndarray
        """
        geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])]

        for i, pc in enumerate(pc_list):
            xyz = np.asarray(pc["xyz"], dtype=np.float32)
            rgb = np.asarray(pc["rgb"], dtype=np.uint8)
            point_id = np.asarray(pc["id"], dtype=np.int32)

            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(f"pc_list[{i}]['xyz'] must have shape (N, 3), got {xyz.shape}")
            if rgb.ndim != 2 or rgb.shape != (len(xyz), 3):
                raise ValueError(f"pc_list[{i}]['rgb'] must have shape ({len(xyz)}, 3), got {rgb.shape}")
            if point_id.shape != (len(xyz), 1):
                raise ValueError(f"pc_list[{i}]['id'] must have shape ({len(xyz)}, 1), got {point_id.shape}")

            if len(xyz) == 0:
                continue

            geoms.append(_as_o3d_pcd(xyz, rgb))

        print(f"\t[DEBUG] pointcloud_vis_debug: showing {len(pc_list)} point clouds with Open3D")
        o3d.visualization.draw_geometries(geoms)    


"""
Util Functions
"""
def create_gripper_marker(color=[0, 0, 255], tube_radius=0.002, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of every cylinder (tube of the
            gripper outline). Defaults to 0.002 (matches the legacy hard-coded
            value); raise it to make the gripper easier to spot in renders.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def make_camera_transform(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
    """
    cam_pos: (3,) 카메라 위치
    target : (3,) 카메라가 바라볼 점
    up     : (3,) 월드 업 벡터
    return : (4,4) camera transform
    """
    # forward: 카메라가 target을 보는 방향
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # right
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # true up
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    # trimesh 카메라는 카메라 좌표계에서 ray가 z == -1 방향으로 나감
    # 그래서 camera frame의 -Z가 forward를 향하도록 맞춤
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -forward
    T[:3, 3] = cam_pos
    
    return T

def plot_collision_statistics(collision_results):
    import matplotlib.pyplot as plt
    from collections import Counter
    
    collision_counts_per_sample = [len(obj_ids) for obj_ids in collision_results]

    count_hist = Counter(collision_counts_per_sample)
    sorted_count_items = sorted(count_hist.items())
    x_counts = [item[0] for item in sorted_count_items]
    y_freqs = [item[1] for item in sorted_count_items]

    flat_obj_ids = [obj_id for obj_ids in collision_results for obj_id in obj_ids]
    obj_id_counter = Counter(flat_obj_ids)

    if flat_obj_ids:
        max_obj_id = max(flat_obj_ids)
        x_obj_ids = list(range(max_obj_id + 1))
        y_obj_counts = [obj_id_counter.get(obj_id, 0) for obj_id in x_obj_ids]
    else:
        x_obj_ids = [0]
        y_obj_counts = [0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: distribution of number of collision objects
    axes[0].bar(x_counts, y_freqs)
    axes[0].set_xlabel("Number of collision objects")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Distribution of collision object counts")
    if x_counts:
        axes[0].set_xticks(range(min(x_counts), max(x_counts) + 1))

    # Plot 2: collision frequency by object id
    bar_colors = ["C0"] * len(x_obj_ids)
    if bar_colors:
        bar_colors[-1] = "red"

    axes[1].bar(x_obj_ids, y_obj_counts, color=bar_colors)
    axes[1].set_xlabel("Collision object id")
    axes[1].set_ylabel("Collision frequency")
    axes[1].set_title("Collision frequency by object id")
    axes[1].set_xticks(x_obj_ids)

    plt.tight_layout()
    plt.show()