import os
import time

import numpy as np
import trimesh
from pykin.collision.collision_manager import CollisionManager
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm


def _load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(list(mesh.geometry.values()))
    if isinstance(mesh, (list, tuple)):
        return trimesh.util.concatenate(mesh)
    raise TypeError(f"Unsupported mesh type for {file_path}: {type(mesh)}")


class PyKinCollisionChecker:
    def __init__(self, collision_model_cfg, mesh_sample_links, joint_distance_threshold=0.3):
        self.collision_model_cfg = collision_model_cfg
        self.mesh_sample_links = tuple(mesh_sample_links)
        self.joint_distance_threshold = float(joint_distance_threshold)

        self.pykin_robot = self._build_robot(collision_model_cfg)
        self.robot_collision = CollisionManager(is_robot=True)
        self.robot_collision.setup_robot_collision(
            self.pykin_robot,
            geom=collision_model_cfg["geom"],
        )

        self.world_collisions = []
        self.world_name_to_obs_id = []

    def _build_robot(self, collision_model_cfg):
        robot_path = collision_model_cfg["robot_path"]
        if not os.path.isabs(robot_path):
            robot_path = os.path.abspath(robot_path)

        offset = collision_model_cfg["offset"]
        robot = SingleArm(
            robot_path,
            offset=Transform(pos=tuple(offset["pos"]), rot=tuple(offset["rot"])),
            has_gripper=collision_model_cfg["has_gripper"],
            gripper_name=collision_model_cfg["gripper_model"],
        )
        robot.setup_link_name(
            base_name=collision_model_cfg["base_name"],
            eef_name=collision_model_cfg["eef_name"],
        )
        return robot

    def build_task_worlds(self, snapshot, object_assets, scene_assets):
        pose = snapshot["pose"]
        object_quat = torch_xyzw_to_np_wxyz(pose["object"]["quat"])
        object_pos = pose["object"]["pos"].cpu().numpy()
        scene_quat = torch_xyzw_to_np_wxyz(pose["scene"]["quat"])
        scene_pos = pose["scene"]["pos"].cpu().numpy()

        self.world_collisions = []
        self.world_name_to_obs_id = []

        for env_idx in range(len(object_assets)):
            world_collision = CollisionManager()
            name_to_obs_id = {}

            for obj_idx in snapshot["task_cand_obj_index"][env_idx]:
                name = f"obj_{obj_idx}"
                mesh = _load_mesh(object_assets[env_idx][obj_idx]["file"])
                transform = Transform(
                    pos=object_pos[env_idx][obj_idx],
                    rot=object_quat[env_idx][obj_idx],
                ).h_mat
                world_collision.add_object(name, gtype="mesh", gparam=mesh, h_mat=transform)
                name_to_obs_id[name] = snapshot["new_id_map"][env_idx][obj_idx]

            for scene_file in scene_assets[env_idx]["files"]:
                name = os.path.splitext(os.path.basename(scene_file))[0]
                mesh = _load_mesh(scene_file)
                transform = Transform(
                    pos=scene_pos[env_idx],
                    rot=scene_quat[env_idx],
                ).h_mat
                world_collision.add_object(name, gtype="mesh", gparam=mesh, h_mat=transform)
                name_to_obs_id[name] = snapshot["scene_id"]

            self.world_collisions.append(world_collision)
            self.world_name_to_obs_id.append(name_to_obs_id)

    def _update_robot_state(self, goal_qpos, open_gripper=True):
        goal_qpos = np.asarray(goal_qpos, dtype=np.float32)
        if goal_qpos.shape[0] > 7:
            goal_qpos = goal_qpos[:7]

        goal_eef = self.pykin_robot.forward_kin(goal_qpos)[self.pykin_robot.eef_name]
        self.pykin_robot.set_transform(goal_qpos)
        if open_gripper:
            self.pykin_robot.open_gripper()

        for link, info in self.pykin_robot.info[self.robot_collision.geom].items():
            if link in self.robot_collision._objs:
                self.robot_collision.set_transform(name=link, h_mat=info[3])

        return np.concatenate([goal_eef.pos, goal_eef.rot[[3, 0, 1, 2]]]).astype(np.float32)

    def sample_robot_point_cloud(self, goal_qpos, sample_points, seed=None):
        self._update_robot_state(goal_qpos)

        robot_meshes = []
        for link, info in self.pykin_robot.info[self.robot_collision.geom].items():
            if link not in self.mesh_sample_links:
                continue

            gtype, mesh_data, h_mat = info[1], info[2], info[3]
            if gtype != "mesh":
                continue

            meshes = mesh_data if isinstance(mesh_data, list) else [mesh_data]
            for mesh in meshes:
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                mesh_copy = mesh.copy()
                mesh_copy.apply_transform(h_mat)
                if len(mesh_copy.vertices) > 0:
                    robot_meshes.append(mesh_copy)

        if not robot_meshes:
            return np.empty((0, 3), dtype=np.float32)

        merged_mesh = trimesh.util.concatenate(robot_meshes)
        if seed is not None:
            np.random.seed(seed)
        xyz, _ = trimesh.sample.sample_surface(merged_mesh, sample_points)
        return np.asarray(xyz, dtype=np.float32)

    def _collision_ids_from_pairs(self, env_idx, collision_pairs):
        collision_ids = set()
        for co1, co2 in collision_pairs:
            if co1 not in self.mesh_sample_links:
                continue
            obs_id = self.world_name_to_obs_id[env_idx].get(co2)
            if obs_id is not None:
                collision_ids.add(obs_id)
        return collision_ids

    def _joint_distance(self, q1, q2):
        diff = q1 - q2
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(np.sqrt(np.sum(diff * diff)))

    def _greedy_threshold_joint_sampling(self, q_list, candidate_indices):
        candidate_indices = list(candidate_indices)
        if not candidate_indices:
            return []

        selected_indices = []
        for idx in candidate_indices:
            if not selected_indices:
                selected_indices.append(idx)
                continue
            if all(
                self._joint_distance(q_list[idx], q_list[selected_idx]) >= self.joint_distance_threshold
                for selected_idx in selected_indices
            ):
                selected_indices.append(idx)
        return selected_indices

    def _filter_candidate_ids(self, qpose_per_env, collision_per_env, scene_id):
        col_obs_dict = {}
        for idx, col_mask in enumerate(collision_per_env):
            col_obs = tuple(np.where(col_mask)[0])
            col_obs_dict.setdefault(col_obs, []).append(idx)

        total_filtered_ids = []
        for collision_signature, candidate_ids in col_obs_dict.items():
            if collision_signature == (scene_id,):
                continue
            total_filtered_ids.extend(
                self._greedy_threshold_joint_sampling(qpose_per_env, candidate_ids)
            )
        return total_filtered_ids

    def check_ik_results(self, ik_result, scene_id):
        if not self.world_collisions:
            raise RuntimeError("Task worlds are not built. Call build_task_worlds() first.")

        num_collision_bits = scene_id + 1
        collision_check_time = []
        obs_qpos, obs_eef, obs_collision = [], [], []

        for env_idx, world_collision in enumerate(self.world_collisions):
            success_mask = ik_result["grasp_success"][env_idx].detach().cpu().numpy()
            grasp_pose_ik = ik_result["grasp_pose_ik"][env_idx].detach().cpu().numpy()

            qpose_per_env, eef_per_env, collision_per_env = [], [], []
            for sample_idx, is_success in enumerate(success_mask):
                if not is_success:
                    continue

                goal_qpos = grasp_pose_ik[sample_idx, :7]
                goal_eef = self._update_robot_state(goal_qpos)

                start = time.time()
                in_collision, names = self.robot_collision.in_collision_other(
                    world_collision,
                    return_names=True,
                )
                collision_check_time.append(time.time() - start)

                collision_ids = self._collision_ids_from_pairs(
                    env_idx,
                    names if in_collision else [],
                )
                collision_mask = np.zeros(num_collision_bits, dtype=np.bool_)
                if collision_ids:
                    collision_mask[list(collision_ids)] = True

                qpose_per_env.append(goal_qpos.astype(np.float32))
                eef_per_env.append(goal_eef)
                collision_per_env.append(collision_mask)

            if qpose_per_env:
                qpose_per_env = np.asarray(qpose_per_env, dtype=np.float32)
                eef_per_env = np.asarray(eef_per_env, dtype=np.float32)
                collision_per_env = np.asarray(collision_per_env, dtype=np.bool_)
                filtered_ids = self._filter_candidate_ids(
                    qpose_per_env,
                    collision_per_env,
                    scene_id,
                )
                obs_qpos.append(qpose_per_env[filtered_ids])
                obs_eef.append(eef_per_env[filtered_ids])
                obs_collision.append(collision_per_env[filtered_ids])
            else:
                obs_qpos.append(np.empty((0, 7), dtype=np.float32))
                obs_eef.append(np.empty((0, 7), dtype=np.float32))
                obs_collision.append(np.empty((0, num_collision_bits), dtype=np.bool_))

        mean_time = float(np.mean(collision_check_time)) if collision_check_time else 0.0
        return {
            "obs_qpos": obs_qpos,
            "obs_eef": obs_eef,
            "obs_collision": obs_collision,
            "timing": {
                "mean": mean_time,
                "count": len(collision_check_time),
            },
        }


def torch_xyzw_to_np_wxyz(quat_tensor):
    quat_array = quat_tensor.cpu().numpy()
    return np.concatenate([quat_array[..., -1:], quat_array[..., :-1]], axis=-1)
