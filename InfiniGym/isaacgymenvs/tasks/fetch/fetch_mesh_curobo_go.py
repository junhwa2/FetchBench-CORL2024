import numpy as np
import os
import torch
import trimesh.transformations as tr
import trimesh
import time
import copy
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

SPHERE_TYPE = {
    0: SphereFitType.SAMPLE_SURFACE,
    1: SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
}

PC_BOUND_TYPE = {
    'panda_cube': {'x': (-0.855, 0.855), 'y': (-0.855, 0.855), 'z': (-0.36, 1.19)},
    'panda_sphere': {'center': (0.0, 0.0, 0.333), 'radius': 0.855, 'z_min': -0.36},
    'panda_w_gripper_sphere': {'center': (0.0, 0.0, 0.333), 'radius': 0.855+0.205, 'z_min': -0.36},
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
        
        # Setup pykin Collision Manager
        robot_path = get_franka_panda_asset(type='franka_r3_cvx_pykin')
        self.pykin_robot = SingleArm(os.path.join(robot_path['asset_root'], 
                                                  robot_path['urdf_file']), 
                                    offset=Transform(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                                    has_gripper=True,
                                    gripper_name="panda_r3_gripper")
        self.pykin_robot.setup_link_name(base_name='panda_link0', eef_name='panda_link7')        
        self.pykin_robot_collision = CollisionManager(is_robot=True)
        self.pykin_robot_collision.setup_robot_collision(self.pykin_robot, geom="collision")
        self.pykin_world_collision_info = []
        self.pykin_world_collision = self._get_pykin_world_config()
        
        assert self.arm_control_type == 'joint'
        
        # Obstruction Dataset Generation Config
        self.max_grasp_pose = self.cfg["obs_data_gen"]["max_grasp_pose"]
        self.max_grasp_pose_per_object = self.cfg["obs_data_gen"]["max_grasp_pose_per_object"]
        
        self.max_num_task_obj_cand = self.cfg["obs_data_gen"]["max_num_task_obj_cand"]
        
        self.pc_bound_option = self.cfg["obs_data_gen"]["pc_bound_option"]
        self.pc_voxel_size = self.cfg["obs_data_gen"]["pc_voxel_size"]
        self.mesh_sample_points = self.cfg["obs_data_gen"]["mesh_sample_points"]
        self.mesh_sample_links = self.cfg["obs_data_gen"]["mesh_sample_links"]


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
        
    def _get_pykin_world_config(self):
        pose = self._get_pose_in_robot_frame()
        oq, sq, dq = pose['object']['quat'], pose['scene']['quat'], pose['table']['quat'] 
        
        # quaternion convention from xyzw to wxyz
        oq = torch.concat([oq[..., -1:], oq[..., :-1]], dim=-1)
        sq = torch.concat([sq[..., -1:], sq[..., :-1]], dim=-1)
        dq = torch.concat([dq[..., -1:], dq[..., :-1]], dim=-1)

        # torch to numpy
        sq, st = sq.cpu().numpy(), pose['scene']['pos'].cpu().numpy()
        oq, ot = oq.cpu().numpy(), pose['object']['pos'].cpu().numpy()
        dq, dt = dq.cpu().numpy(), pose['table']['pos'].cpu().numpy()
        
        task_obj_cand_index = self.task_obj_cand_index[self._task_idx]
        print("Task Object Candidate Index:", task_obj_cand_index)
        
        world_config_list = []
        world_info_list = []
        for i in range(self.num_envs):
            world_config = CollisionManager()
            world_info = {}
            # add object asset (in order of asset_config.yaml)
            for j, o in enumerate(self.object_asset[i]):
                name = f'obj_{j}'
                mesh = self._load_mesh(o['file'])
                h_mat = Transform(pos=ot[i][j], rot=oq[i][j]).h_mat
                world_config.add_object(name, gtype='mesh', gparam=mesh, h_mat=h_mat)
                world_info[name] = {'gparam': mesh, 'transform': h_mat, 'id': j, 
                                    'category': 'object', 'is_task': j in task_obj_cand_index}
            num_objs = len(self.object_asset[i])
            # add scene asset (in order of metadata.npy in scene asset folder)
            for j, file in enumerate(self.scene_asset[i]['files']):
                name = file.split("/")[-1].split(".")[0]
                mesh = self._load_mesh(file)
                h_mat = Transform(pos=st[i], rot=sq[i]).h_mat
                world_config.add_object(name, gtype='mesh', gparam=mesh, h_mat=h_mat)
                world_info[name] = {'gparam': mesh, 'transform': h_mat, 'id': j + num_objs, 'category': 'scene'}
            num_scene = len(self.scene_asset[i]['files'])
            # add table asset
            name = 'table'
            dim = self.table_asset[i]['dim']
            h_mat = Transform(pos=dt[i], rot=dq[i]).h_mat
            world_config.add_object(name, gtype='box', gparam=dim, h_mat=h_mat)
            world_info[name] = {'gparam': dim, 'transform': h_mat, 'id': num_objs + num_scene, 'category': 'table'}
            world_config_list.append(world_config)
            world_info_list.append(world_info)
        
        # update pykin world collision info
        self.pykin_world_collision_info = world_info_list
        
        return world_config_list

    def _update_pykin_robot_state(self, goal_qpos, open_gripper=True):
        if goal_qpos.shape[0] > 7:
            goal_qpos = goal_qpos[:7]
        goal_eef = self.pykin_robot.forward_kin(goal_qpos)[self.pykin_robot.eef_name]                
        self.pykin_robot.set_transform(goal_qpos)
        if open_gripper:
            self.pykin_robot.open_gripper()
        for link, info in self.pykin_robot.info[self.pykin_robot_collision.geom].items():
            if link in self.pykin_robot_collision._objs:
                self.pykin_robot_collision.set_transform(name=link, h_mat=info[3])
        
        # convert goal_eef from Transform to Pose
        # goal_eef = Pose(
        #     torch.as_tensor(goal_eef.pos, dtype=torch.float32, device=self.tensor_args.device),
        #     torch.as_tensor(goal_eef.rot, dtype=torch.float32, device=self.tensor_args.device),
        # )
        # convert goal_eef from Transform to Numpy array in shape (7,) with (x, y, z, qx, qy, qz, qw)
        goal_eef = np.concatenate([goal_eef.pos, goal_eef.rot[[3, 0, 1, 2]]])
        return goal_eef

    def update_pykin_world_collider_pose(self):
        # # update pykin world collision info
        # self.pykin_world_collision_info = world_info_list
        pass

    """
    Point Cloud
    """    
    def _get_seg_color(self, seg_id):
        robot_id = -1
        scene_id = self.max_num_task_obj_cand
        
        fixed_seg_colors = {
            robot_id: np.array([255, 0, 255], dtype=np.uint8),   # robot: magenta
            scene_id: np.array([0, 255, 255], dtype=np.uint8),   # scene: cyan
        }
        if seg_id in fixed_seg_colors:
            return fixed_seg_colors[seg_id]

        # Use viridis colormap for objects
        num_obj_colors = max(int(self.max_num_task_obj_cand), 1)
        obj_idx = min(max(seg_id, 0), num_obj_colors - 1)
        
        cmap = plt.cm.get_cmap('viridis')
        normalized_idx = obj_idx / (num_obj_colors - 1) if num_obj_colors > 1 else 0.0
        color_rgba = cmap(normalized_idx)
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
        
        # downsample pc
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz, dtype=o3d.core.float32))
        pcd.point["colors"] = o3d.core.Tensor(rgb, dtype=o3d.core.uint8)
        pcd.point["id"] = o3d.core.Tensor(id, dtype=o3d.core.int32)

        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
        xyz = np.asarray(pcd_ds.point["positions"].cpu().numpy(), dtype=np.float32)
        rgb = np.asarray(pcd_ds.point["colors"].cpu().numpy(), dtype=np.uint8)
        id = np.asarray(pcd_ds.point["id"].cpu().numpy(), dtype=np.int32)        
        
        assert xyz.shape[0] == rgb.shape[0] == id.shape[0], "Inconsistent point cloud shapes"
        assert xyz.shape[0] > 1, "Ingenstive point cloud: no points found in camera data"        

        return {
            "xyz": xyz,
            "rgb": rgb,
            "id": id,
        }

    def gen_pc_from_camera(self, seg_ids=None, env_idx=0):
        point_clouds = self.get_camera_data(
            tensor_ptd=True,
            ptd_in_robot_base=True,
            segmented_ptd=True,
            ptd_downscale=1
        )["camera_pointcloud_seg"]
        seg_pts = point_clouds[env_idx]["by_seg"] # {seg_id(int): points(np.ndarray, N x 3)}
        
        if seg_ids is None:
            seg_ids = self.task_obj_cand_index[self._task_idx] 
            # seg_ids = [3] + [sid+4 for sid in seg_ids] # scene and all task object (object id start from 4)
            seg_ids = seg_ids + [self.max_num_task_obj_cand]
        print(f"Generating point cloud from camera for env {env_idx}, seg_ids: {seg_ids}")
        
        xyz_list, rgb_list, id_list = [], [], []
        for seg_id, pts in sorted(seg_pts.items()):
            # skip for robot(1) and table(2)
            if seg_id == 1 or seg_id == 2:
                continue
            
            # change seg_id only for scene(3) and objects(4>)
            if seg_id >= 4:
                seg_id -= 4 # To-do: num_objects != max_num_task_obj_cand
            elif seg_id == 3:
                seg_id = self.max_num_task_obj_cand
            
            if len(pts) == 0 or seg_id not in seg_ids:
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
        
        filtered_pc = self._filter_pc({"xyz": xyz, "rgb": rgb, "id": id})
        xyz, rgb, id = filtered_pc["xyz"], filtered_pc["rgb"], filtered_pc["id"]

        assert xyz.shape[0] == rgb.shape[0] == id.shape[0], "Inconsistent point cloud shapes"
        assert xyz.shape[0] > 1, "Ingenstive point cloud: no points found in camera data"        
        
        pc = {
            "xyz": xyz,
            "rgb": rgb,
            "id": id,
        }
        
        if self.debug_viz and self.viewer:
            self.pointcloud_vis_debug([pc])

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

        robot_seg_id = -1
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

        if self.debug_viz and self.viewer:
            self.pointcloud_vis_debug([pc])

        return pc
    
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

        o3d.visualization.draw_geometries(geoms)    

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
    def sample_all_obj_annotated_grasp_pose(self):
        """
        Build random grasp candidates from all task object candidates.

        Shapes:
            obj_grasp_poses: (num_envs, num_objs, num_grasp_pose, 7)
            task_obj_cand_index: (num_tasks, num_cand_obj)
            object pose: quat (num_envs, num_objs, 4), pos (num_envs, num_objs, 3)

        Returns:
            sample_grasps: (num_envs, max_grasp_pose, 7)
        """
        pose = self._get_pose_in_robot_frame()
        oq, ot = pose['object']['quat'], pose['object']['pos']

        max_pose_seed = self.max_grasp_pose
        max_pose_per_obj = self.max_grasp_pose_per_object
        task_obj_cand_index = self.task_obj_cand_index[self.get_task_idx()]

        sample_grasps = []
        for i in range(self.num_envs):
            obj_grasps_list = []
            for goal_idx in task_obj_cand_index:
                grasp_pose = self.obj_grasp_poses[i][goal_idx].to(self.tensor_args.device)
                grasp_quat, grasp_pos = grasp_pose[..., 3:7], grasp_pose[..., :3]

                oq_i, ot_i = (oq[i:i+1, goal_idx].repeat(grasp_pose.shape[0], 1),
                              ot[i:i+1, goal_idx].repeat(grasp_pose.shape[0], 1))
                gq, gt = tf_combine(oq_i, ot_i, grasp_quat, grasp_pos)
                gq = torch.concat([gq[..., -1:], gq[..., :-1]], dim=-1)
                obj_grasps = torch.concat([gt, gq], dim=-1)
                pose_per_obj = min(obj_grasps.shape[0], max_pose_per_obj)
                obj_grasps = obj_grasps[torch.randint(obj_grasps.shape[0], size=(pose_per_obj,))]
                # print(f"Object {goal_idx} grasp poses shape: {obj_grasps.shape}")
                obj_grasps_list.append(obj_grasps)

            all_obj_grasps = torch.cat(obj_grasps_list, dim=0)
            random_batch = torch.randint(all_obj_grasps.shape[0], size=(max_pose_seed,))
            # random_batch = torch.range(0, max_pose_seed, dtype=torch.int)
            # random_batch = torch.tensor([40]*10, dtype=torch.int)
            sample_grasps.append(all_obj_grasps[random_batch])

        sample_grasps = torch.stack(sample_grasps, dim=0)
        
        # Debug Visualization
        if self.debug_viz and self.viewer:
            grasp_poses = []
            for i in range(sample_grasps.shape[1]):
                grasp_poses.append(Pose(sample_grasps[:, i, :3], sample_grasps[:, i, 3:7]))
            grasp_poses = Pose.vstack(grasp_poses, dim=1)
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                grasp_success = torch.ones((sample_grasps.shape[1]), dtype=torch.bool)   
                self.grasp_vis_debug(pose, grasp_poses[i], grasp_success, env_idx=i)

        return sample_grasps

    def solve_ik(self, annotated_grasp_pose):
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

        result_holder = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.tensor_args.device)
        ik_holder = (self.robot_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).to(self.tensor_args.device)).unsqueeze(1)
        grasp_poses, grasp_success, grasp_pose_ik = [], [], []

        for i in range(annotated_grasp_pose.shape[1]):
            grasp_candidate = annotated_grasp_pose[:, i]
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
        }
        
        # Debug Visualization
        if self.debug_viz and self.viewer:
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                self.grasp_vis_debug(pose, grasp_poses[i], grasp_success[i], env_idx=i)
        
        return res

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
  
    def collision_check(self, ik_result):
        """
        Perform collision checking for IK results.
        """        
        # map world object name to obs_id        
        obs_index_map = []
        obs_info_list = []
        for i in range(self.num_envs):
            name_to_oid = {}
            obs_info = []
            world_info = self.pykin_world_collision_info[i]
            for k, v in sorted(world_info.items(), key=lambda item: item[1]['id']):
                if v['category'] == 'object':
                    if v['is_task']:
                        name_to_oid[k] = v['id']
                        obs_info.append({'name': k, 'category': v['category']})
                elif v['category'] == 'scene':
                    name_to_oid[k] = self.max_num_task_obj_cand
            obs_info.append({'name': 'scene', 'category': 'scene'})
            obs_index_map.append(name_to_oid)
            obs_info_list.append(obs_info)
        print("Observation Index Map (world name to obs_id):", obs_index_map[0])
            
        # collision check for each IK solution in each environment
        obs_qpos, obs_eef, obs_collision = [], [], []
        for i in range(self.num_envs):
            success_mask = ik_result["grasp_success"][i].detach().cpu().numpy()
            grasp_pose_ik = ik_result["grasp_pose_ik"][i].detach().cpu().numpy()
            qpose_per_env, eef_per_env, collision_per_env = [], [], []
            for j in range(len(success_mask)):
                if not success_mask[j]:
                    continue                
                
                # set robot to IK solution
                goal_qpos = grasp_pose_ik[j, :7]
                goal_eef = self._update_pykin_robot_state(goal_qpos)

                # collision check
                result, name = self.pykin_robot_collision.in_collision_other(self.pykin_world_collision[i], 
                                                                            return_names=True)
                link_collision = {}
                if result:
                    for co1, co2 in name:
                        # link_collision.setdefault(co1, set()).add(co2)
                        link_collision.setdefault(co1, set()).add(obs_index_map[i][co2])                        
                total_collision = set().union(*link_collision.values()) if link_collision else set()
                # total_collision = np.asarray(list(total_collision), dtype=np.int32)
                
                N = self.max_num_task_obj_cand + 1 # for scene
                collision_mask = np.zeros(N, dtype=np.bool8)
                collision_mask[list(total_collision)] = True
                # debug
                # print(">>>>>> in_collision_other:", result, total_collision)
                # if total_collision:            
                #     self.collision_vis_debug(total_collision)
                qpose_per_env.append(goal_qpos)
                eef_per_env.append(goal_eef)
                collision_per_env.append(collision_mask)
            
            qpose_per_env = np.asarray(qpose_per_env) 
            eef_per_env = np.asarray(eef_per_env)
            collision_per_env = np.asarray(collision_per_env)
                        
            # filter out results
            # col_obs_dict = defaultdict(list)
            # for j, col_obs in enumerate(collision_per_env):
            #     col_obs_dict[tuple(col_obs)].append(j)
            col_obs_dict = defaultdict(list)
            for j, col_mask in enumerate(collision_per_env):
                col_obs = np.where(col_mask)[0]
                col_obs_dict[tuple(col_obs)].append(j)
            
            scene_id = self.max_num_task_obj_cand
            total_filtered_ids = []
            for k, candidate_ids in col_obs_dict.items():
                if k == (scene_id,):
                    continue
                filtered_ids = self._greedy_threshold_joint_sampling(qpose_per_env, candidate_ids, threshold=0.3) ## To-do: pose diff로 변경
                # print(f"Collision obs: {k}, candidate ids: {len(candidate_ids)}, filtered ids: {len(filtered_ids)}")
                total_filtered_ids.extend(filtered_ids)
            
            if self.debug_viz and self.viewer:
            # if True:
                pose = self._get_pose_in_robot_frame()
                obs_filtered = torch.zeros(qpose_per_env.shape[0], dtype=torch.bool)
                obs_filtered[total_filtered_ids] = True
                self.grasp_vis_debug(pose, eef_per_env[0], obs_filtered, env_idx=i) ####### To-do: Not Working
            
            obs_qpos.append(qpose_per_env[total_filtered_ids])
            obs_eef.append(eef_per_env[total_filtered_ids])
            obs_collision.append(collision_per_env[total_filtered_ids])
        
        obs_qpos = np.asarray(obs_qpos, dtype=np.float32)
        obs_eef = np.asarray(obs_eef, dtype=np.float32) 
        # obs_collision = np.asarray(obs_collision, dtype=object)
        obs_collision = np.asarray(obs_collision, dtype=np.bool8)
        
        res = {
            'obs_qpos': obs_qpos,
            'obs_eef': obs_eef,
            'obs_collision': obs_collision
        }
        
        # save results (1): obs_info
        log_obs_info = {
            'obs_id': [list(range(len(obs_info))) for obs_info in obs_info_list],
            'obs_name': [[obs['name'] for obs in obs_info] for obs_info in obs_info_list],
            'obs_category': [[obs['category'] for obs in obs_info] for obs_info in obs_info_list],
        }
        log_collision_info = {}
        
        return res

    """
    Your Solution
    """
    def make_obs_data(self, col_result, obs_dir):
        pc_cams, pc_robots = [], []
        for i in range(self.num_envs):
            print(f"Environment {i}:")
            pc_cam = self.gen_pc_from_camera(env_idx=i)
            pc_cam_path = f"ply/pc_cam_t{self._task_idx}.ply"        
            self.save_pc(pc_cam, f"{obs_dir}/{pc_cam_path}")
            for j in range(len(col_result['obs_qpos'][i])):
                col_obs = np.where(col_result['obs_collision'][i][j])[0]
                col_qpos = col_result['obs_qpos'][i][j]
                print(f"  IK solution {j} qpos: {col_qpos}, collision with obs ids: {col_obs}")
                self._update_pykin_robot_state(col_qpos)
                pc_robot = self.gen_pc_from_pykin_robot(sample_points=self.mesh_sample_points, 
                                                        link_names=self.mesh_sample_links)
                pc_robot_path = f"ply/pc_robot_t{self._task_idx}_{j}.ply"
                self.save_pc(pc_robot, f"{obs_dir}/{pc_robot_path}")
                print("Generated Point Clouds: camera={}K, robot={}K, total={}K\n".format(len(pc_cam['xyz'])/1000.0, 
                                                                                        len(pc_robot['xyz'])/1000.0, 
                                                                                        (len(pc_cam['xyz'])+len(pc_robot['xyz']))/1000.0))
                # pc_cam = self.gen_pc_from_camera(seg_ids=col_obs, env_idx=i)
                # self.pointcloud_vis_debug([pc_cam, pc_robot])
                pc_cams.append(pc_cam_path)
                pc_robots.append(pc_robot_path)
        
        pc_cams = np.asarray(pc_cams)
        pc_robots = np.asarray(pc_robots)
        eef = col_result['obs_eef'].reshape(-1, *col_result['obs_eef'].shape[2:])
        qpos = col_result['obs_qpos'].reshape(-1, *col_result['obs_qpos'].shape[2:])
        collision = col_result['obs_collision'].reshape(-1, *col_result['obs_collision'].shape[2:])
        
        obs_data = {
            'eef': eef,
            'qpos': qpos,
            'collision': collision,
            'pc_cam': pc_cams,
            'pc_robot': pc_robots
        }
        
        print("Final dataset shapes: eef {}, qpos {}, collision {}, pc_cams {}, pc_robots {}".format(
            eef.shape, qpos.shape, collision.shape, pc_cams.shape, pc_robots.shape
        ))
        
        return obs_data
        
    def solve(self):
        log = {}

        # self.set_target_color()
        self._solution_video = []
        self._video_frame = 0

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()
        
        # obs_dir = f"{ASSET_PATH}/Obstruction/{self.cfg['task']['scene_config_path'][0]}"
        obs_dir = f"./Obstruction/{self.cfg['task']['scene_config_path'][0]}"
        print("obs_dir:", obs_dir)
        if not os.path.exists(obs_dir):
            os.makedirs(obs_dir)

        # Sample Good Grasp Pose
        sampled_grasp_pose = self.sample_all_obj_annotated_grasp_pose() # shape: (num_env, max_seed, 7)
        print("Sampled grasp pose:", sampled_grasp_pose.shape)
        
        # Solve IK
        ik_result = self.solve_ik(sampled_grasp_pose)
        print("Success IK solutions:", int(ik_result["grasp_success"].sum()))
        
        # Collision Checking
        self.update_pykin_world_collider_pose()
        col_result = self.collision_check(ik_result)
        print("Collision check result:", len(col_result['obs_qpos'][0]))
        # plot_collision_statistics(col_result['obs_collision'][0])

        # Make obs data
        obs_data = self.make_obs_data(col_result, obs_dir)
       
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(f"{obs_dir}/obstruction_data.h5", "w") as f:
            f.create_dataset("eef", data=np.asarray(obs_data['eef'], dtype=np.float32))
            f.create_dataset("qpos", data=np.asarray(obs_data['qpos'], dtype=np.float32))
            f.create_dataset("collision", data=np.asarray(obs_data['collision'], dtype=np.bool8))
            f.create_dataset("pc_cam", data=np.asarray(obs_data['pc_cam'], dtype=str_dtype)) 
            f.create_dataset("pc_robot", data=np.asarray(obs_data['pc_robot'], dtype=str_dtype)) 
        exit()      

        # self.set_default_color()

        return image_to_video(self._solution_video), log
        
    """
    Debug Visualization
    """
    def grasp_vis_debug(self, poses, grasp_pose, grasp_success, env_idx=0):
        """
        Visualize grasp pose and environment in a separate window using trimesh.
        
        Args:
            poses: dict of pose tensors for different entities in the environment
            grasp_pose: Pose tensor of shape (num_grasp_pose, 7) representing candidate grasp poses
            grasp_success: Boolean tensor of shape (num_grasp_pose,) indicating IK success for each grasp pose
            env_idx: Index of the environment to visualize
        """
        
        scene = trimesh.Scene()
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)

        table_pose = poses['table']
        dq = torch.concat([table_pose['quat'][..., -1:], table_pose['quat'][..., :-1]], dim=-1)
        dq, dt = dq.cpu().numpy(), table_pose['pos'].cpu().numpy()

        table_translation = tr.translation_matrix(dt[env_idx])
        table_rotation = tr.quaternion_matrix(dq[env_idx])

        table = trimesh.creation.box(extents=self.table_asset[env_idx]['dim'], transform=table_translation @ table_rotation)
        scene.add_geometry(table)

        scene_pose = poses['scene']

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

        object_poses = poses['object']
        oq = torch.concat([object_poses['quat'][..., -1:], object_poses['quat'][..., :-1]], dim=-1)
        oq, ot = oq.cpu().numpy(), object_poses['pos'].cpu().numpy()

        # vis objects
        for i, o in enumerate(self.object_asset[env_idx]):
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

        scene.show()

    def collision_vis_debug(self, objs_in_collision, env_idx=0):
        print("Visualizing collision with objects:", objs_in_collision)
        scene = trimesh.Scene()
        
        # vis axis
        axis = trimesh.creation.axis()
        scene.add_geometry(axis)
        # vis robot
        scene = apply_robot_to_scene(trimesh_scene=scene, 
                                    robot=self.pykin_robot, 
                                    geom=self.pykin_robot_collision.geom)
        # vis world collision objects
        _objs = self.pykin_world_collision[env_idx]._objs     
        for c_name, c_info in self.pykin_world_collision_info[env_idx].items():
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
        
        scene.show()

"""
Util Functions
"""
def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


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