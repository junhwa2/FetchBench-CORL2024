
import numpy as np
import os
import torch
import trimesh.transformations as tr
import trimesh
import time
import h5py

# cuRobo
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
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

from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import FetchMeshCurobo
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo_go import (
    create_gripper_marker, make_camera_transform,
)
from isaacgymenvs.tasks.fetch.utils.singulation_planner import (
    plan_with_grasps, draw_and_or_graph, step_color_rgb,
)


SPHERE_TYPE = {
    0: SphereFitType.SAMPLE_SURFACE,
    1: SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
}


def image_to_video(obs_buf):
    video = []
    for s, images in enumerate(obs_buf):
        steps = []
        for e, imgs in enumerate(images):
            steps.append(np.concatenate(imgs, axis=0))
        video.append(np.concatenate(steps, axis=1))
    return video


class FetchMeshCuroboGORun(FetchMeshCurobo):
    def update_cuRobo_motion_gen_config(self, attach_goal_obj=True, obs_obj_id=None):
        """Configure motion-gen world + (optionally) attach an obj to the EE.

        Args:
            attach_goal_obj: True to attach (carry) the obj, False to detach.
            obs_obj_id: optional new_id from obs_data. When provided we attach
                that obj instead of the task's final goal. Mirrors the
                `obs_obj_id` arg on `_enable_goal_obj_collision_checking`.
        """
        self.update_cuRobo_world_collider_pose()

        q, qd = (self.states["q"].clone().to(self.tensor_args.device),
                 self.states["qd"].clone().to(self.tensor_args.device))

        if attach_goal_obj:
            # as robot_cfg is unique, batch_env operation is not supported for attached object

            for i in range(self.num_envs):
                cu_js = JointState(
                    position=q[i, :-2],
                    velocity=qd[i, :-2],
                    acceleration=q[i, :-2] * 0.0,
                    jerk=q[i, :-2] * 0.0,
                    joint_names=self.robot_joint_names
                )
                if obs_obj_id is None:
                    goal_obj_idx = self.task_obj_index[i][self.get_task_idx()].cpu().numpy()
                else:
                    goal_obj_idx = int(self._obs_ids[obs_obj_id])

                # setup attached_object pre_transform pose
                ee_pose_inv = self.motion_generators[i].compute_kinematics(cu_js).ee_pose.inverse()

                mesh_idx = self.motion_generator_colliders[i].get_mesh_idx(f'env_{i}_obj_{goal_obj_idx}', 0)
                curr_mesh_pose_tensor = self.motion_generator_colliders[i].get_mesh_pose_tensor()[:, mesh_idx]
                curr_mesh_pose = Pose(curr_mesh_pose_tensor[:, :3], curr_mesh_pose_tensor[:, 3:7]).inverse()
                obj_init_pose = self.motion_generator_colliders[i].world_model.get_obstacle(f'env_{i}_obj_{goal_obj_idx}').pose
                obj_init_pose = Pose.from_list(obj_init_pose, self.tensor_args)

                offset_pos = to_torch([[0.0, 0.0, self.cfg["solution"]["cuRobo"]["attach_object_z_offset"]]], device=self.tensor_args.device, dtype=torch.float)
                offset_quat = to_torch([[1.0, 0.0, 0.0, 0.0]], device=self.tensor_args.device, dtype=torch.float)
                offset_pose = Pose(offset_pos.repeat(self.num_envs, 1), offset_quat.repeat(self.num_envs, 1))

                attached_obj_pre_transform = (
                    ee_pose_inv.multiply(offset_pose).multiply(curr_mesh_pose).multiply(obj_init_pose.inverse()))

                self.motion_generators[i].attach_objects_to_robot_custom(attached_obj_pre_transform,
                                                                         [f'env_{i}_obj_{goal_obj_idx}'],
                                                                         surface_sphere_radius=self.cfg["solution"]["cuRobo"]["surface_sphere_radius"],
                                                                         sphere_fit_type=SPHERE_TYPE[self.cfg["solution"]["cuRobo"]["sphere_approx_method"]])
                self.motion_generators[i].reset()

        else:
            self._enable_goal_obj_collision_checking(True, obs_obj_id=obs_obj_id)

            for m in self.motion_generators:
                m.detach_object_from_robot()
                m.reset()

        if self.debug_viz and self.viewer:
            for i in range(self.num_envs):
                cu_js = JointState(
                    position=q[i, :-2],
                    velocity=qd[i, :-2],
                    acceleration=q[i, :-2] * 0.0,
                    jerk=q[i, :-2] * 0.0,
                    joint_names=self.robot_joint_names
                )

                eef_pose = self.motion_generators[i].compute_kinematics(cu_js).ee_pose.get_numpy_matrix()[0]
                self.cuRobo_vis_debug(self.motion_generators[i], eef_pose=eef_pose)

    """
    Sample Grasp Pose
    """

    def _approx_material_color(self, obj_file):
        """Return a `gymapi.Vec3` average RGB of the obj's mesh.obj surface.

        Isaac Gym has no API to clear `set_rigid_body_color` and restore the
        URDF material once it's been overridden. We approximate by sampling
        the trimesh-loaded mesh's `face_colors` (`TextureVisuals` are first
        converted via `to_color()`, which samples the texture per-face).
        The result is a single flat color close to the obj's average
        appearance — not the full texture, but visibly distinct from a flat
        default blue / target red. Fallback to `default_obj_color` on error.
        """
        try:
            mesh = trimesh.load(obj_file, process=False, force='mesh')
            if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                mesh.visual = mesh.visual.to_color()
            fc = np.asarray(mesh.visual.face_colors)
            if fc.size == 0:
                return self.default_obj_color
            rgb = fc[:3] if fc.ndim == 1 else np.mean(fc[:, :3], axis=0)
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
            return gymapi.Vec3(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        except Exception as e:
            print(f"[_approx_material_color] {obj_file}: "
                  f"{type(e).__name__}: {e}")
            return self.default_obj_color

    def _get_or_cache_material_color(self, env_idx, old_id):
        """Memoized lookup of the obj's approximate material color."""
        if not hasattr(self, '_material_color_cache'):
            self._material_color_cache = {}
        key = (env_idx, old_id)
        if key not in self._material_color_cache:
            obj_file = self.object_asset[env_idx][old_id].get('file')
            self._material_color_cache[key] = (
                self._approx_material_color(obj_file)
                if obj_file else self.default_obj_color)
        return self._material_color_cache[key]

    def _color_planner_final_target(self, color):
        """Apply `color` to the planner's chosen final target.

        When `color is None`, restore the target to its cached material-
        average color (close to URDF appearance) instead of painting a fixed
        value — used by `set_default_color` so previously-targeted objs
        don't stick around as flat blue across tasks.

        Returns True when the active plan is available (so the caller knows
        the planner-aware path was taken); False when there is no plan yet
        (e.g. called before `solve` finished planning) so the caller can
        fall back to the base-class behavior.
        """
        if not getattr(self, '_active_plan', None):
            return False
        final_new_id = self._active_plan[-1][0]
        for i, env_ptr in enumerate(self.envs):
            old_id = int(self._obs_ids[final_new_id])
            applied = (color if color is not None
                       else self._get_or_cache_material_color(i, old_id))
            self.gym.set_rigid_body_color(
                env_ptr, self.objects[i][old_id],
                0, gymapi.MESH_VISUAL, applied,
            )
        return True

    def set_target_color(self):
        """Color the planner-chosen final target red.

        Falls back to base behavior (`task_obj_index` obj painted red) when
        there is no active plan yet — e.g. called before `solve()` reaches
        the planning step.
        """
        if not self._color_planner_final_target(self.task_obj_color):
            super().set_target_color()

    def set_default_color(self):
        """Un-highlight the planner-chosen final target.

        When `env.preserve_obj_material=True` we restore the target to its
        approximate material color (per-obj cached flat average) instead of
        a fixed `default_obj_color`. The URDF texture itself cannot be
        restored — Isaac Gym has no API for it — but the cached average
        keeps each obj close to its natural tone and avoids the
        red-accumulation problem across tasks.
        """
        if self.cfg["env"].get("preserve_obj_material", False):
            # `None` → use material-avg cache inside _color_planner_final_target.
            if not self._color_planner_final_target(None):
                super().set_default_color()
            return
        if not self._color_planner_final_target(self.default_obj_color):
            super().set_default_color()

    def eval(self):
        """Planner-aware success criterion.

        Overrides the base eval whose `success` checks the task-config goal
        obj. Here, success means:
          (1) the singulation plan ran to completion (all steps OK), AND
          (2) the *planner's* chosen final target is currently lifted to the
              robot (same z/x threshold logic as base eval, just applied to
              the planner's target rather than `task_obj_index[task_idx]`).
        Base z/x/e/task_repeat results are preserved under their original keys
        for cross-reference; the planner-specific ones are added under
        `planner_*`.
        """
        base = super().eval()
        # No active plan (solve early-returned before planning): leave the
        # base success alone, which still uses the task-goal criterion.
        if not getattr(self, '_active_plan', None):
            return base

        plan_ok = bool(getattr(self, '_plan_success', False))
        final_new_id = self._active_plan[-1][0]
        final_old_id = int(self._obs_ids[final_new_id])

        # Mirror base eval's world-frame CoM computation.
        curr_pos = self.states['obj_pos'].clone()      # (num_envs, num_objs, 3)
        curr_rot = self.states['obj_quat'].clone()
        com      = self.obj_ref_point.clone()
        curr_obj_to_world = (curr_pos.reshape(-1, 3)
                             + quat_apply(curr_rot.reshape(-1, 4), com.reshape(-1, 3)))
        curr_obj_to_world = curr_obj_to_world.reshape(self.num_envs, -1, 3)

        robot_to_world = self._robot_base_state.clone()[..., :3]
        target_to_robot = curr_obj_to_world[:, final_old_id] - robot_to_world

        z_ok = (target_to_robot[..., -1] >= self.cfg['eval']['z_threshold']).cpu().numpy()
        x_ok = (target_to_robot[..., 0]  <= self.cfg['eval']['x_threshold']).cpu().numpy()
        held = z_ok & x_ok

        base['planner_plan_complete'] = np.array([plan_ok] * self.num_envs, dtype=bool)
        base['planner_z_threshold']   = z_ok
        base['planner_x_threshold']   = x_ok
        base['planner_held']          = held
        # Overwrite success with planner-aware criterion.
        base['success'] = held & plan_ok
        return base

    def _enable_goal_obj_collision_checking(self, enable=True, obs_obj_id=None):
        """Toggle collision-check for the obj we're about to grasp.

        Args:
            enable: True to re-enable, False to disable.
            obs_obj_id: optional new_id from obs_data (i.e. the kb obj id used
                by the singulation planner). When provided we look up the
                corresponding pykin/isaac old_id via `self._obs_ids[obs_obj_id]`
                instead of falling back to the task's final goal index. This
                lets us toggle collision for a plan step that is not the
                eventual final target (e.g. an intermediate clearing obj).
        """
        for i in range(self.num_envs):
            if obs_obj_id is None:
                goal_idx = self.task_obj_index[i][self.get_task_idx()].cpu().numpy()
            else:
                goal_idx = int(self._obs_ids[obs_obj_id])
            self.ik_collision.enable_obstacle(f'env_{i}_obj_{goal_idx}', enable=enable, env_idx=i)
            self.motion_generator_colliders[i].enable_obstacle(f'env_{i}_obj_{goal_idx}', enable=enable)
            self.motion_generators[i].reset()

    def sample_goal_obj_collision_free_grasp_pose(self):
        # Use IK solver to solve for candidate grasp pose
        annotated_grasp_pose = self._sample_goal_obj_annotated_grasp_pose() # shape: (num_env, max_seed, 7)

        # Base boolean True mask to combine IK success with additional per-env validity checks.
        result_holder = torch.ones((self.num_envs, 1), device=self.tensor_args.device, dtype=torch.bool)
        # Per-env default joint template used as a safe IK fallback (including gripper joints).
        ik_holder = (self.robot_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).to(self.tensor_args.device)).unsqueeze(1)

        grasp_poses, pre_grasp_poses = [], []
        grasp_success, pre_grasp_success = [], []
        grasp_pose_ik = []

        # Check collision-free IK at pre-grasp pose
        for i in range(annotated_grasp_pose.shape[1]):
            grasp_candidate = annotated_grasp_pose[:, i]
            grasp_pose = Pose(grasp_candidate[..., :3], grasp_candidate[..., 3:7])
            pre_grasp_offset_pos = to_torch([0, 0, -self.cfg["solution"]["pre_grasp_offset"]],
                                            device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_pos = pre_grasp_offset_pos.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_quat = pre_grasp_offset_quat.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset = Pose(pre_grasp_offset_pos, pre_grasp_offset_quat)
            pre_grasp_pose = grasp_pose.multiply(pre_grasp_offset) # shape: (num_env, 7)

            grasp_poses.append(grasp_pose)
            pre_grasp_poses.append(pre_grasp_pose)

            ik_result = self.ik_solver.solve_batch_env(pre_grasp_pose)  # ik_result: success (num_envs, 1), solution (num_envs, 1, dof)
            torch.cuda.synchronize() # Synchronize to ensure async CUDA IK finishes before using results 

            pre_grasp_success.append(result_holder & ik_result.success) # shape: (num_envs, 1)

        # Check collision-free IK at grasp pose (disable goal obj)
        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(False)

        for i in range(annotated_grasp_pose.shape[1]):
            grasp_pose = grasp_poses[i]
            ik_result = self.ik_solver.solve_batch_env(grasp_pose)
            torch.cuda.synchronize()
            
            grasp_success.append(result_holder & ik_result.success)

            # ik_result.success: (num_envs, 1), ik_result.success.float().unsqueeze(-1): (num_envs, 1, 1)
            # Falls back to default joints on IK failure to keep downstream motion inputs stable.
            ik = (ik_result.solution * ik_result.success.float().unsqueeze(-1) +
                  (1. - ik_result.success.float().unsqueeze(-1)) * ik_holder[..., :-2]) 
            # ik: (num_envs, 1, dof), ik_holder[..., :-2]: (num_envs, 1, 2)
            # Append gripper joint states from ik_holder.
            grasp_pose_ik.append(torch.concat([ik, ik_holder[..., -2:]], dim=-1))

        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(True)

        grasp_poses, pre_grasp_poses = Pose.vstack(grasp_poses, dim=1), Pose.vstack(pre_grasp_poses, dim=1)
        grasp_success, pre_grasp_success = torch.cat(grasp_success, dim=1), torch.cat(pre_grasp_success, dim=1)
        grasp_pose_ik = torch.cat(grasp_pose_ik, dim=1) # success grasp_pose?

        res = {
            'grasp_poses': grasp_poses, # shape: (num_envs, max_seed, 7)
            'pre_grasp_poses': pre_grasp_poses, # shape: (num_envs, max_seed, 7)
            'grasp_success': grasp_success, # shape: (num_envs, max_seed)
            'pre_grasp_success': pre_grasp_success, # shape: (num_envs, max_seed)
            'grasp_ik': grasp_pose_ik # shape: (num_envs, max_seed, dof)
        }
        if self.debug_viz and self.viewer:
            print("Visualizing grasp poses and IK results in cuRobo debug viz...")
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                success_ik = torch.masked_select(grasp_pose_ik[i], mask=grasp_success[i].unsqueeze(-1))
                success_ik = success_ik.reshape(-1, grasp_pose_ik.shape[-1])

                if len(success_ik) == 0:
                    ik_poses = None
                else:
                    ik_poses = self.ik_solver.fk(success_ik).ee_pose # success

                self.grasp_vis_debug(pose, grasp_poses[i], pre_grasp_poses[i], ik_poses, env_idx=i)
        return res

    def sample_plan_step_grasp_pose(self):
        """IK-check the grasp poses associated with the first step of the active plan.

        Mirrors `sample_goal_obj_collision_free_grasp_pose` but the candidate
        grasps are not freshly sampled from `obj_grasp_poses`; instead they
        are the obstruction-pipeline grasps recorded in
        `self._obs_grasp` (robot frame, wxyz quat — see fetch_mesh_curobo_go.py
        line 1071-1073, 1199, 671) at the indices listed by the planner for
        the first step of `self._active_plan`.

        Returns dict with the same schema as
        `sample_goal_obj_collision_free_grasp_pose`:
            grasp_poses, pre_grasp_poses           # Pose,  (num_envs, K, 7)
            grasp_success, pre_grasp_success       # bool,  (num_envs, K)
            grasp_ik                               # float, (num_envs, K, dof)

        Assumes num_envs == 1 (obstruction h5 is per-env).
        """
        assert self.num_envs == 1, "sample_plan_step_grasp_pose assumes num_envs == 1"
        assert getattr(self, '_active_plan', None), \
            "no active plan — call after plan_with_grasps succeeded"

        step_idx = getattr(self, '_current_step_idx', 0)
        step_new_id, grasp_ids = self._active_plan[step_idx]
        if len(grasp_ids) == 0:
            raise ValueError(f"plan step for obj{step_new_id} has zero grasp ids")

        # Robot-frame, wxyz grasps straight from obstruction h5.
        grasps_np = np.asarray(self._obs_grasp)[np.asarray(grasp_ids, dtype=np.int64)]  # (K, 7)
        annotated_grasp_pose = torch.from_numpy(grasps_np).to(self.tensor_args.device).float()
        annotated_grasp_pose = annotated_grasp_pose.unsqueeze(0)  # (1, K, 7)

        result_holder = torch.ones((self.num_envs, 1), device=self.tensor_args.device, dtype=torch.bool)
        ik_holder = (self.robot_default_dof_pos.unsqueeze(0)
                     .repeat(self.num_envs, 1)
                     .to(self.tensor_args.device)).unsqueeze(1)

        grasp_poses, pre_grasp_poses = [], []
        grasp_success, pre_grasp_success = [], []
        grasp_pose_ik = []

        # Pre-grasp IK loop.
        for i in range(annotated_grasp_pose.shape[1]):
            grasp_candidate = annotated_grasp_pose[:, i]
            grasp_pose = Pose(grasp_candidate[..., :3], grasp_candidate[..., 3:7])
            pre_grasp_offset_pos = to_torch([0, 0, -self.cfg["solution"]["pre_grasp_offset"]],
                                            device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_pos = pre_grasp_offset_pos.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_quat = pre_grasp_offset_quat.unsqueeze(dim=0).repeat(self.num_envs, 1)
            pre_grasp_offset = Pose(pre_grasp_offset_pos, pre_grasp_offset_quat)
            pre_grasp_pose = grasp_pose.multiply(pre_grasp_offset)

            grasp_poses.append(grasp_pose)
            pre_grasp_poses.append(pre_grasp_pose)

            ik_result = self.ik_solver.solve_batch_env(pre_grasp_pose)
            torch.cuda.synchronize()
            pre_grasp_success.append(result_holder & ik_result.success)

        # Grasp IK loop — disable collision against the obj for this plan step.
        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(False, obs_obj_id=step_new_id)

        for i in range(annotated_grasp_pose.shape[1]):
            grasp_pose = grasp_poses[i]
            ik_result = self.ik_solver.solve_batch_env(grasp_pose)
            torch.cuda.synchronize()

            grasp_success.append(result_holder & ik_result.success)
            ik = (ik_result.solution * ik_result.success.float().unsqueeze(-1) +
                  (1. - ik_result.success.float().unsqueeze(-1)) * ik_holder[..., :-2])
            grasp_pose_ik.append(torch.concat([ik, ik_holder[..., -2:]], dim=-1))

        if self.cfg["solution"]["disable_grasp_obj_ik_collision"]:
            self._enable_goal_obj_collision_checking(True, obs_obj_id=step_new_id)

        grasp_poses, pre_grasp_poses = Pose.vstack(grasp_poses, dim=1), Pose.vstack(pre_grasp_poses, dim=1)
        grasp_success, pre_grasp_success = torch.cat(grasp_success, dim=1), torch.cat(pre_grasp_success, dim=1)
        grasp_pose_ik = torch.cat(grasp_pose_ik, dim=1)

        res = {
            'grasp_poses':       grasp_poses,        # (num_envs, K, 7)
            'pre_grasp_poses':   pre_grasp_poses,    # (num_envs, K, 7)
            'grasp_success':     grasp_success,      # (num_envs, K)
            'pre_grasp_success': pre_grasp_success,  # (num_envs, K)
            'grasp_ik':          grasp_pose_ik,      # (num_envs, K, dof)
        }
        if self.debug_viz and self.viewer:
            pose = self._get_pose_in_robot_frame()
            for i in range(self.num_envs):
                success_ik = torch.masked_select(grasp_pose_ik[i], mask=grasp_success[i].unsqueeze(-1))
                success_ik = success_ik.reshape(-1, grasp_pose_ik.shape[-1])
                ik_poses = None if len(success_ik) == 0 else self.ik_solver.fk(success_ik).ee_pose
                self.grasp_vis_debug(pose, grasp_poses[i], pre_grasp_poses[i], ik_poses, env_idx=i)
        return res

    """
    Motion Generation
    """

    def load_prediction_h5(self, folder_name, env_idx=0):
        """Read the prediction .h5 file for the currently active scene + task_idx.

        Path layout:
            $ASSET_PATH/<folder_name>/<scene_type>/<scene_name>/obstruction_data_t<task_idx>_pred.h5

        `<scene_type>/<scene_name>` is taken from `scene_config_path`
        (e.g. `benchmark_eval/LargeShelf/LargeShelfSceneFactory_27`) with the
        leading benchmark split (`benchmark_eval/`, `benchmark_train/`, ...)
        stripped so it matches the on-disk layout under `$ASSET_PATH/<folder>/`.

        Args:
            folder_name: subdirectory under $ASSET_PATH that holds the predictions
                (e.g. "20260516_vorm").
            env_idx: which env's scene_config_path to use (defaults to 0).

        Returns:
            dict with every dataset in the h5 file loaded as a numpy array, plus
            a `_path` key holding the resolved absolute path for logging.
        """
        asset_path = os.environ["ASSET_PATH"]
        scene_rel_path = self.cfg["task"]["scene_config_path"][env_idx]
        task_idx = self.get_task_idx()

        # Strip the leading "benchmark_*/" split — the prediction tree only
        # keeps "<scene_type>/<scene_name>".
        parts = scene_rel_path.split("/")
        if parts and parts[0].startswith("benchmark_"):
            parts = parts[1:]
        scene_sub = "/".join(parts)

        h5_path = os.path.join(
            asset_path,
            folder_name,
            scene_sub,
            f"obstruction_data_t{task_idx}_pred.h5",
        )

        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Prediction h5 not found: {h5_path}")

        data = {"_path": h5_path}
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                data[key] = f[key][()]

        print(f"[load_prediction_h5] loaded {h5_path} "
              f"(keys: {[k for k in data.keys() if not k.startswith('_')]})")
        return data

    def load_obstruction_h5(self, folder_name, env_idx=0):
        """Read the ground-truth obstruction .h5 file for the active scene + task_idx.

        `folder_name` is expected to already include the benchmark split (e.g.
        "Obstruction_260513_60_111/benchmark_eval"), so the leading
        "benchmark_*/" segment of `scene_config_path` is stripped before being
        appended — otherwise it would be duplicated on disk.

        Path layout:
            $ASSET_PATH/<folder_name>/<scene_type>/<scene_name>/obstruction_data_t<task_idx>.h5

        e.g. with `folder_name='Obstruction_260513_60_111/benchmark_eval'` and
        `scene_config_path='benchmark_eval/LargeShelf/LargeShelfSceneFactory_27'`:
            $ASSET_PATH/Obstruction_260513_60_111/benchmark_eval/LargeShelf/
                LargeShelfSceneFactory_27/obstruction_data_t<task_idx>.h5

        Args:
            folder_name: subdirectory under $ASSET_PATH that holds the obstruction
                dataset, including the benchmark split (e.g.
                "Obstruction_260513_60_111/benchmark_eval").
            env_idx: which env's scene_config_path to use (defaults to 0).

        Returns:
            dict with every dataset in the h5 file loaded as a numpy array
            (`collision`, `eef`, `grasp`, `pc_cam`, `pc_robot`, `qpos`), plus a
            `_path` key holding the resolved absolute path for logging.
        """
        asset_path = os.environ["ASSET_PATH"]
        scene_rel_path = self.cfg["task"]["scene_config_path"][env_idx]
        task_idx = self.get_task_idx()

        # `folder_name` already encodes the benchmark split, so drop the leading
        # "benchmark_*/" from scene_config_path to avoid duplicating it.
        parts = scene_rel_path.split("/")
        if parts and parts[0].startswith("benchmark_"):
            parts = parts[1:]
        scene_sub = "/".join(parts)

        h5_path = os.path.join(
            asset_path,
            folder_name,
            scene_sub,
            f"obstruction_data_t{task_idx}.h5",
        )

        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Obstruction h5 not found: {h5_path}")

        data = {"_path": h5_path}
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                data[key] = f[key][()]

        print(f"[load_obstruction_h5] loaded {h5_path} "
              f"(keys: {[k for k in data.keys() if not k.startswith('_')]})")
        return data

    # ---- Planner-target cache (task_config.npz in-place edit) -------------
    # GT mode picks a target via rank_final_targets and writes it back to the
    # scene's task_config.npz so subsequent pred-mode runs lock onto the same
    # target. We use the same file the loader reads at sim init (via the
    # `$ASSET_PATH/Task -> Task_*` symlink) — no extra cache directory.

    def _dump_kb_and_graph(self, kb, final_target, min_clear, plan, log):
        """Write `kb/t<idx>.{json,txt}` and `graph/t<idx>.{dot,png}`.

        Safe to call whether BC succeeded or not — when `plan` is falsy the
        AND-OR graph is rendered without plan highlighting and the KB payload
        records `bc_plan_length=0`, `plan=[]`. Callers should pass:
          * BC success → real `final_target` + `plan`
          * GT-mode BC fail → rank-top candidate as `final_target`, plan=None
          * Pred-mode BC fail → cached target's new_id as `final_target`,
                                plan=None
        `final_target=None` aborts both dumps (nothing meaningful to root).
        """
        if final_target is None:
            print("[planner] KB / AND-OR graph dump skipped (no target)")
            return
        plan_for_dump = list(plan) if plan else []
        exp_root = os.path.join(
            "runs", self.cfg.get("experiment_name", "default"),
        )
        task_tag = f"t{self.get_task_idx()}"

        # (a) AND-OR graph rooted at `final_target`. plan_for_dump may be []
        # — draw_and_or_graph treats empty plan as "no highlight", same as
        # passing `plan=None`.
        graph_path = os.path.join(exp_root, "graph", f"{task_tag}.png")
        try:
            written = draw_and_or_graph(kb, final_target, graph_path,
                                        plan=plan_for_dump or None)
            print(f"[planner] AND-OR graph: {written}")
        except Exception as e:
            print(f"[planner] AND-OR graph dump failed: {type(e).__name__}: {e}")

        # (b) KB snapshot: round-trippable JSON + quick-read TXT.
        kb_dir = os.path.join(exp_root, "kb")
        os.makedirs(kb_dir, exist_ok=True)
        kb_json_path = os.path.join(kb_dir, f"{task_tag}.json")
        kb_txt_path  = os.path.join(kb_dir, f"{task_tag}.txt")
        kb_payload = {
            "task_idx":      self.get_task_idx(),
            "final_target":  int(final_target),
            "min_clear":     int(min_clear) if min_clear is not None else None,
            "bc_plan_time":  float(log.get("bc_plan_time", 0.0)),
            "bc_plan_length": int(log.get("bc_plan_length", 0)),
            "plan":          [[int(o), [int(g) for g in gs]] for o, gs in plan_for_dump],
            "kb":            {str(int(k)): {
                                  "status": v["status"],
                                  "clauses": [[int(x) for x in c] for c in v["clauses"]],
                                  "grasp_indices": [[int(g) for g in gs] for gs in v["grasp_indices"]],
                              } for k, v in kb.items()},
        }
        try:
            import json
            with open(kb_json_path, "w", encoding="utf-8") as f:
                json.dump(kb_payload, f, indent=2, ensure_ascii=False)
            with open(kb_txt_path, "w", encoding="utf-8") as f:
                f.write(f"task_idx={self.get_task_idx()}  final_target=obj{final_target} "
                        f"(min_clear={min_clear})\n")
                f.write(f"bc_plan_time={kb_payload['bc_plan_time']:.4f}s  "
                        f"bc_plan_length={kb_payload['bc_plan_length']}\n")
                if plan_for_dump:
                    f.write("plan: " + " -> ".join(
                        f"{o}:[{','.join(str(g) for g in gs)}]" for o, gs in plan_for_dump
                    ) + "\n\n")
                else:
                    f.write("plan: (BC FAILED — no plan)\n\n")
                f.write(self.format_grasp_kb(kb) + "\n")
            print(f"[planner] KB dump: {kb_json_path}, {kb_txt_path}")
        except Exception as e:
            print(f"[planner] KB dump failed: {type(e).__name__}: {e}")

    def _scene_task_config_path(self, env_idx=0):
        return os.path.join(
            os.environ["ASSET_PATH"], "Task",
            self.cfg["task"]["scene_config_path"][env_idx],
            "task_config.npz",
        )

    def _scene_rearrange_config_path(self, env_idx=0):
        return os.path.join(
            os.environ["ASSET_PATH"], "Task",
            self.cfg["task"]["scene_config_path"][env_idx],
            "rearrange_config.npz",
        )

    def _lookup_obj_label(self, old_id, env_idx=0):
        """Per-obj label string from the scene's rearrange_config.npz.

        Mirrors loader._path/rearrange_config.npz['object_labels'][0][old_id]
        (e.g. "cup_0", "book_3") — falls back to "obj_<old_id>" on any error.
        """
        try:
            rc = np.load(self._scene_rearrange_config_path(env_idx))
            labels = rc['object_labels']
            # object_labels has shape (n_scene_compositions, n_objs); scene 0.
            return str(labels[0][int(old_id)])
        except Exception as e:
            print(f"[_lookup_obj_label] fallback for obj {old_id}: "
                  f"{type(e).__name__}: {e}")
            return f"obj_{int(old_id)}"

    def load_planner_target_from_config(self, env_idx=0):
        """Read the cached planner target from the scene's task_config.npz.

        Returns:
            (target_old_id, has_marker)
                target_old_id : int — task_obj_index[task_idx]
                has_marker    : bool — True iff `planner_chosen` flag is set
                                       in the file (= written by GT mode)
        """
        path = self._scene_task_config_path(env_idx)
        data = np.load(path, allow_pickle=True)
        target_old = int(np.asarray(data['task_obj_index'])[self.get_task_idx()])
        has_marker = bool(data['planner_chosen'].item()) \
                     if 'planner_chosen' in data.files else False
        return target_old, has_marker

    def save_planner_target_to_config(self, target_old_id, target_label,
                                      env_idx=0):
        """Overwrite task_obj_index[task_idx] + task_obj_label[task_idx] in
        the scene's task_config.npz and set the `planner_chosen=True` marker.

        Other fields are preserved exactly (task_init_state, task_camera_pose,
        task_cand_obj_index/label, ...). The file is the very same one the
        loader reads at sim init, so the new target takes effect on the next
        env construction.
        """
        path = self._scene_task_config_path(env_idx)
        data = dict(np.load(path, allow_pickle=True))
        # task_obj_index might be int ndarray; task_obj_label is object dtype.
        idx_arr = np.asarray(data['task_obj_index']).copy()
        lbl_arr = np.asarray(data['task_obj_label']).copy()
        idx_arr[self.get_task_idx()] = int(target_old_id)
        lbl_arr[self.get_task_idx()] = str(target_label)
        data['task_obj_index']  = idx_arr
        data['task_obj_label']  = lbl_arr
        data['planner_chosen']  = np.array(True)
        np.savez(path, **data)
        print(f"[save_planner_target] {path}  "
              f"task_idx={self.get_task_idx()} → obj_old={target_old_id} "
              f"label='{target_label}'")

    @staticmethod
    def build_grasp_kb(collision, target):
        """Build the singulation KB consumed by `singulation_planner`.

        Categorizes each candidate target into one of three states:
          (i)   'ungraspable' — every grasp for that target hits the scene
                                (last column of `collision`), so no movable
                                object can ever rescue it.
          (ii)  'obstructed'  — at least one grasp survives the scene, but
                                all surviving grasps still collide with some
                                movable object(s).
          (iii) 'graspable'   — at least one surviving grasp collides with
                                nothing besides (optionally) the target itself.

        For each grasp `g`:
            obstacles(g) = {j : collision[g, j], j != target[g], j != scene_col}

        Grasps with the same obstacle tuple are deduplicated and their grasp
        indices merged. Strict-superset clauses are then absorbed
        (`(a) ∨ (a∧b) → (a)`).

        Args:
            collision: (G, K) bool array — per-grasp collision per object.
                       Last column is treated as the scene/fixed aggregate.
            target:    (G,)  int  array — target movable id per grasp.

        Returns:
            dict[int, dict] keyed by obj_id with the schema:
                {
                    'status':        'graspable' | 'obstructed' | 'ungraspable',
                    'clauses':       List[List[int]],   # ICRA-style DNF
                    'grasp_indices': List[List[int]],   # parallel to clauses
                }
            * 'graspable'   → clauses=[],            grasp_indices=[[g, ...]]
            * 'obstructed'  → clauses=[[a,b],[c]],   grasp_indices=[[..],[..]]
            * 'ungraspable' → clauses=[],            grasp_indices=[]

        Objects that never appear in `target` are omitted (the planner treats
        missing keys as dead-ends, identical to 'ungraspable').
        """
        collision = np.asarray(collision, dtype=bool)              # (G, K)
        target    = np.asarray(target,    dtype=np.int64)          # (G,)
        if collision.ndim != 2 or target.shape != (collision.shape[0],):
            raise ValueError(
                f"shape mismatch: collision={collision.shape}, target={target.shape} "
                f"(expected collision=(G,K), target=(G,))"
            )
        scene_col = collision.shape[1] - 1
        keep = ~collision[:, scene_col]                            # drop scene-blocked

        # Per-grasp obstacle tuple (target self + scene col excluded).
        buckets = {}  # obj_id -> {obstacle_tuple: [grasp_indices]}
        for g in np.where(keep)[0]:
            t = int(target[g])
            obstacles = tuple(int(j) for j in np.where(collision[g])[0]
                              if j != t and j != scene_col)
            buckets.setdefault(t, {}).setdefault(obstacles, []).append(int(g))

        kb = {}
        for t in sorted(buckets.keys()):
            items = list(buckets[t].items())                       # [(tuple, [g...])]
            # Special case: any empty tuple → object is currently graspable.
            graspable_grasps = []
            for clause, grasps in items:
                if clause == ():
                    graspable_grasps.extend(grasps)
            if graspable_grasps:
                bucket = sorted(graspable_grasps)
                kb[t] = {
                    'status': 'graspable',
                    'clauses':            [],
                    'grasp_indices':      [bucket],
                    'full_clauses':       [],
                    'full_grasp_indices': [bucket],
                }
                continue
            # `full_*` keeps every distinct obstacle bucket; `clauses` /
            # `grasp_indices` apply absorption (drop strict-superset clauses)
            # for the planner default. Both views share the same status
            # ordering so swapping at plan time is just a field rename.
            #
            # Sort key per clause i:
            #   1. len(clause)        ↑ shortest first (BC cost)
            #   2. -len(grasps)       ↑ grasp-richer first  ← BC tiebreak
            #   3. clause tuple       ↑ lex order for determinism
            # Effect: among clauses of equal length, BC sees grasp-richer
            # clauses first → conjugate_dnfs enumerates them first → ties in
            # Node cost resolve in favor of the higher-grasp clause via
            # Python's first-inserted-wins `min(open_list)` behavior.
            sort_key = lambda i: (len(items[i][0]), -len(items[i][1]),
                                  items[i][0])
            all_idx = sorted(range(len(items)), key=sort_key)
            full_clauses = [list(items[i][0]) for i in all_idx]
            full_grasps  = [sorted(items[i][1]) for i in all_idx]

            sets = [set(c) for c, _ in items]
            kept = [i for i, si in enumerate(sets)
                    if not any(j != i and sets[j] < si for j in range(len(sets)))]
            kept.sort(key=sort_key)
            kb[t] = {
                'status': 'obstructed',
                'clauses':            [list(items[i][0]) for i in kept],
                'grasp_indices':      [sorted(items[i][1]) for i in kept],
                'full_clauses':       full_clauses,
                'full_grasp_indices': full_grasps,
            }

        # Targets that appeared in `target` but had every grasp scene-blocked.
        for t in np.unique(target):
            t = int(t)
            if t not in kb:
                kb[t] = {'status': 'ungraspable',
                         'clauses': [], 'grasp_indices': [],
                         'full_clauses': [], 'full_grasp_indices': []}

        return dict(sorted(kb.items()))

    @staticmethod
    def format_grasp_kb(kb):
        """Pretty-print a KB dict from `build_grasp_kb`."""
        counts = {'graspable': 0, 'obstructed': 0, 'ungraspable': 0}
        for entry in kb.values():
            counts[entry['status']] += 1
        header = (f"[grasp_kb] graspable={counts['graspable']}, "
                  f"obstructed={counts['obstructed']}, "
                  f"ungraspable={counts['ungraspable']}")
        lines = [header]
        for obj_id, entry in kb.items():
            status = entry['status'].upper()
            if entry['status'] == 'graspable':
                expr = f"grasps={entry['grasp_indices'][0]}"
            elif entry['status'] == 'ungraspable':
                expr = "scene blocks every grasp"
            else:
                expr = " OR ".join(
                    "(" + " AND ".join(str(x) for x in clause) + ")"
                    for clause in entry['clauses']
                ) + f"  grasps={entry['grasp_indices']}"
            lines.append(f"  obj {obj_id:>3}: {status:<11}  {expr}")
        return "\n".join(lines)

    # Known signal names for `rank_final_targets`. Each entry: (sign, desc)
    # where `sign = +1` means "smaller value first" (ascending) and
    # `sign = -1` means "larger value first" (handled by negating in the key).
    _TIEBREAK_SIGNALS = {
        'volume':         (+1, 'obj mesh volume — smaller first'),
        'avg_min_grasps': (-1, 'mean grasp count over min_clear clauses — larger first'),
    }

    @staticmethod
    def rank_final_targets(kb, tie_signals=None, tiebreak_order=('volume',)):
        """Rank candidates from hardest to easiest by `min_clear`.

        min_clear(o) = 0                                        if 'graspable'
                     = min(len(c) for c in kb[o]['clauses'])    if 'obstructed'
                     = -inf  (excluded from candidates)         if 'ungraspable'

        Sort key (primary → tiebreak):
          1. -min_clear                       — biggest `min_clear` first
          2..k. signals in `tiebreak_order`   — see `_TIEBREAK_SIGNALS` for
                                                per-signal direction
          k+1. obj_id                         — final deterministic tiebreak

        Args:
            kb: KB dict from `build_grasp_kb`.
            tie_signals: optional dict[obj_id -> dict[signal_name -> float]]
                providing per-candidate values for each named signal. Missing
                obj_ids or missing signal names fall back to a sign-aware
                "worst" sentinel (+inf for ascending, -inf for descending) so
                the candidate gets pushed to the back of its mc-tier.
            tiebreak_order: tuple of signal names from `_TIEBREAK_SIGNALS`,
                applied in order after the primary -mc key. Empty tuple →
                only obj_id breaks ties.

        Returns:
            List[(obj_id, min_clear)] in descending difficulty order, or [].
        """
        # Validate signal names up front so a typo fails loud, not silently.
        for sig in tiebreak_order:
            if sig not in FetchMeshCuroboGORun._TIEBREAK_SIGNALS:
                raise ValueError(
                    f"unknown tiebreak signal '{sig}'; must be one of "
                    f"{sorted(FetchMeshCuroboGORun._TIEBREAK_SIGNALS)}")

        ranked = []
        for obj_id in sorted(kb.keys()):
            entry = kb[obj_id]
            if entry['status'] == 'ungraspable':
                continue
            mc = (0 if entry['status'] == 'graspable'
                  else min(len(c) for c in entry['clauses']))
            keys = [-mc]
            for sig in tiebreak_order:
                sign, _ = FetchMeshCuroboGORun._TIEBREAK_SIGNALS[sig]
                worst = float('inf') if sign > 0 else float('-inf')
                v = (float(tie_signals.get(int(obj_id), {}).get(sig, worst))
                     if tie_signals is not None else worst)
                # sign=+1 (asc) → push v as-is; sign=-1 (desc) → negate so
                # ascending sort yields descending semantic order.
                keys.append(v if sign > 0 else -v)
            keys.append(int(obj_id))
            ranked.append((tuple(keys), int(obj_id), int(mc)))
        ranked.sort(key=lambda r: r[0])
        return [(o, mc) for _, o, mc in ranked]

    @staticmethod
    def select_final_target(kb, tie_signals=None, tiebreak_order=('volume',)):
        """Top-1 helper around `rank_final_targets` (the hardest candidate).

        Returns:
            (obj_id, min_clear) or (None, None) if no graspable target exists.
        """
        ranked = FetchMeshCuroboGORun.rank_final_targets(
            kb, tie_signals=tie_signals, tiebreak_order=tiebreak_order)
        return ranked[0] if ranked else (None, None)

    def _compute_obj_volumes(self, env_idx=0):
        """Per-obj mesh volume (m^3), keyed by Isaac/pykin `old_id`.

        Uses `trimesh.Trimesh.volume` when the mesh is watertight (true volume),
        else falls back to its axis-aligned bounding-box volume (rough upper
        bound but always positive/finite). Cached per env on `self` since the
        underlying assets don't change task-to-task.

        Returns:
            dict[int, float] — old_id → volume. Missing/broken meshes get +inf.
        """
        cache = getattr(self, '_obj_volume_cache', None)
        if cache is None:
            cache = {}
            self._obj_volume_cache = cache
        if env_idx in cache:
            return cache[env_idx]
        volumes = {}
        for old_id, asset in enumerate(self.object_asset[env_idx]):
            mesh = asset.get('mesh') if isinstance(asset, dict) else None
            v = float('inf')
            if mesh is not None:
                try:
                    if getattr(mesh, 'is_watertight', False):
                        v = float(mesh.volume)
                    if not np.isfinite(v) or v <= 0:
                        v = float(mesh.bounding_box.volume)
                except Exception as e:
                    print(f"[_compute_obj_volumes] fallback for old_id={old_id}: "
                          f"{type(e).__name__}: {e}")
                    try:
                        v = float(mesh.bounding_box.volume)
                    except Exception:
                        v = float('inf')
            volumes[int(old_id)] = v
        cache[env_idx] = volumes
        return volumes

    def _execute_plan_step(self, step_idx, log, release_after, computing_time_ref):
        """Run one (obj, grasp_ids) plan step end-to-end.

        Mirrors the original single-shot grasp/carry pipeline, but:
          * collision/attach are toggled per-step via `obs_obj_id`
          * log keys are namespaced as `step{step_idx}_<key>`
          * intermediate steps optionally release the obj at the free-space
            drop pose, final step keeps holding (controlled by `release_after`)

        Args:
            step_idx: index into `self._active_plan`
            log:      dict to write per-step entries into
            release_after: True → open_gripper after carry-out (intermediate),
                           False → keep holding (final target)
            computing_time_ref: 1-element list used as mutable counter so the
                                caller can aggregate planner time across steps

        Returns:
            True on full success of the step, False on any IK / motion failure.
        """
        obs_obj_id, _ = self._active_plan[step_idx]
        self._current_step_idx = step_idx
        key = lambda k: f'step{step_idx}_{k}'

        # Map planner-new-id → isaac old-id for state lookups. The tracking
        # baseline for execute-success is captured later, *after* close_gripper,
        # so we don't compare to the robot's home pose.
        target_old_id = int(self._obs_ids[obs_obj_id])

        def _snap(stage):
            """Snapshot obj_pos / eef_pos at this point in the step pipeline."""
            self._refresh()
            op = self.states['obj_pos'][0, target_old_id].clone()
            ep = self.states['eef_pos'][0].clone()
            d  = float((op - ep).norm())
            log[key(f'dbg_{stage}_obj_pos')] = op.cpu().numpy()
            log[key(f'dbg_{stage}_eef_pos')] = ep.cpu().numpy()
            log[key(f'dbg_{stage}_dist')]    = d
            print(f"  [dbg step{step_idx} {stage:>7}] "
                  f"obj={op.cpu().numpy().round(3).tolist()}  "
                  f"eef={ep.cpu().numpy().round(3).tolist()}  "
                  f"dist={d:.3f}")
            return op, ep, d

        _snap('pre')

        # 1. IK
        self.update_cuRobo_world_collider_pose()
        ik_result = self.sample_plan_step_grasp_pose()
        if not bool(ik_result['grasp_success'].any()):
            log[key('plan_success')] = 0
            log[key('plan_failure')] = 'no_ik'
            return False

        # 2. Grasp motion
        if self.cfg["solution"]["direct_grasp"]:
            ik_success = ik_result['grasp_success']
            self.update_cuRobo_world_collider_pose()
            t0 = time.time()
            traj, success, poses, _ = self.motion_gen_to_grasp_pose(
                ik_result['grasp_poses'], mask=ik_success)
            log[key('grasp_plan_success')] = success
            computing_time_ref[0] += time.time() - t0

            self.follow_motion_trajs(traj, gripper_state=0)
            log[key('grasp_execute_error')] = self.get_end_effect_error(poses)
        else:
            ik_success = ik_result['grasp_success'] & ik_result['pre_grasp_success']
            log[key('ik_plan_success')] = ik_success.any(dim=-1).cpu().numpy()
            t0 = time.time()
            traj, success, poses, _ = self.motion_gen_to_grasp_pose(
                ik_result['pre_grasp_poses'], mask=ik_success)
            log[key('pre_grasp_plan_success')] = success
            computing_time_ref[0] += time.time() - t0

            self.follow_motion_trajs(traj, gripper_state=0)
            log[key('pre_grasp_execute_error')] = self.get_end_effect_error(poses)

            if self.cfg["solution"]["move_offset_method"] == 'motion_planning':
                self.update_cuRobo_world_collider_pose()
                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(False, obs_obj_id=obs_obj_id)

                t0 = time.time()
                traj, success, poses, _ = self.motion_gen_by_z_offset(
                    z=self.cfg["solution"]["pre_grasp_offset"], mask=success)
                computing_time_ref[0] += time.time() - t0

                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(True, obs_obj_id=obs_obj_id)
                log[key('grasp_plan_success')] = success
                self.follow_motion_trajs(traj, gripper_state=0)
                log[key('grasp_execute_error')] = self.get_end_effect_error(poses)
            elif self.cfg["solution"]["move_offset_method"] == 'cartesian_linear':
                offset = np.array([0, 0, self.cfg["solution"]["pre_grasp_offset"] *
                                   self.cfg["solution"]["grasp_overshoot_ratio"]])
                self.follow_cartesian_linear_motion(offset, gripper_state=0)

        if not bool(np.asarray(success).any()):
            log[key('plan_success')] = 0
            log[key('plan_failure')] = 'grasp_motion_fail'
            return False

        # 3. Close gripper, retract
        self.close_gripper()
        log[key('grasp_finger_obj_contact')] = self.finger_goal_obj_contact()
        # Snapshot at this moment becomes the baseline for tracking-error:
        # we want "did obj follow EEF *during the carry*", not "did obj follow
        # EEF from the robot's home pose".
        post_close_obj_pos = self.states['obj_pos'][0, target_old_id].clone()
        post_close_eef_pos = self.states['eef_pos'][0].clone()
        _snap('close')

        if self.cfg["solution"]["retract_offset"] > 0:
            offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
            log[key('retract_finger_obj_contact')] = self.finger_goal_obj_contact()
            _snap('retract')

        # 4. Attach + carry to free space (the drop pose)
        attach = self.cfg["solution"]["attach_goal_obj"]
        self.update_cuRobo_motion_gen_config(attach_goal_obj=attach, obs_obj_id=obs_obj_id)
        if self.cfg["solution"]["disable_grasp_obj_motion_gen"] and (not attach):
            self._enable_goal_obj_collision_checking(False, obs_obj_id=obs_obj_id)

        t0 = time.time()
        traj, success, poses, results = self.motion_gen_to_free_space(mask=success)
        computing_time_ref[0] += time.time() - t0

        self.update_cuRobo_motion_gen_config(attach_goal_obj=False, obs_obj_id=obs_obj_id)
        log[key('fetch_plan_success')] = success
        log[key('fetch_plan_failure')] = [None if r is None else r.status for r in results]

        if not bool(np.asarray(success).any()):
            log[key('plan_success')] = 0
            log[key('plan_failure')] = 'fetch_motion_fail'
            return False

        self.follow_motion_trajs(traj, gripper_state=-1)
        log[key('fetch_execute_error')] = self.get_end_effect_error(poses)
        post_obj_pos, post_eef_pos, _ = _snap('carry')

        # === All motion planning succeeded ===
        log[key('plan_success')] = 1

        # === Execute success check — right after carry, BEFORE release.
        # Held = the obj→EEF *distance* stays roughly constant from close to
        # carry. This tolerates rotation within the gripper (which changes the
        # direction but not the radius) and is invariant to per-obj URDF-origin
        # offsets. A real slip / drop changes the distance substantially.
        # `disp` (obj absolute travel) is still gated to rule out no-op cases.
        # min_disp is fixed: rules out no-op steps where the obj didn't
        # actually move. 0.05 m chosen empirically — small enough to admit
        # legitimate short carries, large enough to reject jitter.
        min_disp       = 0.05
        dist_drift_tol = self.cfg["solution"].get("execute_dist_drift_tol", 0.10)
        delta_obj   = post_obj_pos - post_close_obj_pos
        disp        = float(delta_obj.norm())
        dist_close  = float((post_close_obj_pos - post_close_eef_pos).norm())
        dist_carry  = float((post_obj_pos       - post_eef_pos      ).norm())
        dist_drift  = abs(dist_carry - dist_close)
        moved       = disp       > min_disp
        held        = dist_drift < dist_drift_tol
        print(f"  [dbg step{step_idx}    chk ] "
              f"disp={disp:.3f}  dist_close={dist_close:.3f}  "
              f"dist_carry={dist_carry:.3f}  drift={dist_drift:.3f}")
        log[key('dbg_disp')]       = disp
        log[key('dbg_dist_close')] = dist_close
        log[key('dbg_dist_carry')] = dist_carry
        log[key('dbg_dist_drift')] = dist_drift
        if moved and held:
            log[key('execute_success')] = 1
        else:
            log[key('execute_success')] = 0
            reasons = []
            if not moved: reasons.append(f'no_movement(disp={disp:.3f})')
            if not held:  reasons.append(f'slipped(drift={dist_drift:.3f})')
            log[key('execute_failure')] = ','.join(reasons)
            return False

        # 5. Release intermediate obj at the drop pose; keep holding final.
        if release_after:
            self.open_gripper()

        return True

    def solve(self):
        # set goal obj color
        log = {}

        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.

        # Reset any plan/obs state carried over from a previous task so the
        # color overrides don't accidentally target last task's obj if this
        # task aborts before reaching the planning step.
        self._active_plan = None
        self._obs_grasp   = None
        self._obs_ids     = None
        self._current_step_idx = 0
        self._plan_success = False

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()
        
        # Load obstruction GT first — it gives the target list, grasp poses,
        # and the per-grasp collision GT used for both the planner and the
        # consistency check below.
        obs_folder = self.cfg["solution"]["obs_folder"]
        obs_data = self.load_obstruction_h5(obs_folder)

        # KB collision source: obs GT by default, or learned predictions when
        # `pred_folder` is set. The prediction file MUST carry a
        # `collision_label` matching obs `collision` exactly (same scene,
        # same grasp ordering) — otherwise the KB would be built against a
        # different problem than what the rest of the pipeline executes.
        pred_folder = self.cfg["solution"].get("pred_folder")
        if pred_folder is None:
            print("[planner] pred_folder=null → KB built from obs_data['collision']")
            kb_collision = obs_data['collision']
        else:
            pred_data = self.load_prediction_h5(pred_folder)
            assert np.array_equal(
                np.asarray(pred_data['collision_label']),
                np.asarray(obs_data['collision'])
            ), (f"pred_data['collision_label'] does not match obs_data['collision'] "
                f"(pred shape={np.asarray(pred_data['collision_label']).shape}, "
                f"obs shape={np.asarray(obs_data['collision']).shape}) — the "
                f"prediction file was generated against a different obs file.")
            print(f"[planner] pred_folder={pred_folder} → KB built from "
                  f"pred_data['collision_pred']")
            kb_collision = pred_data['collision_pred']

        # Build singulation KB and pick final target = obj with largest min_clear.
        # KB always carries both minimal (absorbed) and full (raw) clause sets.
        # The active view is picked by `solution.use_full_obs`:
        #   False (default) → minimal clauses only (smaller search, current behavior)
        #   True            → all clauses including supersets (more grasp diversity)
        kb = self.build_grasp_kb(kb_collision, obs_data['target'])
        use_full_obs = bool(self.cfg["solution"].get("use_full_obs", False))
        if use_full_obs:
            for entry in kb.values():
                entry['clauses']       = entry.get('full_clauses', entry['clauses'])
                entry['grasp_indices'] = entry.get('full_grasp_indices',
                                                   entry['grasp_indices'])
        print(f"[planner] use_full_obs={use_full_obs}")
        print(self.format_grasp_kb(kb))

        # Three nested feasibility flags surfaced separately in the CSV.
        log['target_exist']         = 0
        log['bc_plan_exist']        = 0
        log['motion_plan_success']  = 0
        log['bc_plan_length']       = 0
        log['bc_plan_time']         = 0.0

        # `_obs_ids[new_id] = old_id` (isaac/pykin); precompute reverse map
        # so we can swap between planner-frame and scene-frame obj ids.
        obs_ids_arr = np.asarray(obs_data['obs_ids'])
        old_to_new = {int(old): int(new) for new, old in enumerate(obs_ids_arr)}

        total_budget = float(self.cfg["solution"].get("plan_time_limit", 2.0))

        final_target, min_clear, plan = None, None, None
        winning_dt = 0.0

        if pred_folder is None:
            # ============================================================
            # GT mode: pick target via rank_final_targets fallback loop;
            # save the chosen target back to task_config.npz so subsequent
            # pred-mode runs lock onto the same target.
            # ============================================================
            # Per-candidate tiebreak signals. Order applied is yaml-controlled
            # via `solution.target_tiebreak_order` (see _TIEBREAK_SIGNALS for
            # supported names and per-signal direction). KB keys are new_id;
            # volume cache is old_id-keyed, so map via obs_ids_arr.
            old_volumes = self._compute_obj_volumes(env_idx=0)
            tie_signals = {}
            for kb_obj_id, entry in kb.items():
                if entry['status'] == 'graspable':
                    # graspable obj has exactly one (empty-obstacle) bucket;
                    # treat its grasp count as the "min-clear grasp pool".
                    amg = float(len(entry['grasp_indices'][0]))
                elif entry['status'] == 'obstructed':
                    cur_mc = min(len(c) for c in entry['clauses'])
                    min_buckets = [g for c, g in zip(entry['clauses'],
                                                     entry['grasp_indices'])
                                   if len(c) == cur_mc]
                    amg = (float(np.mean([len(g) for g in min_buckets]))
                           if min_buckets else 0.0)
                else:
                    amg = 0.0
                old_id = int(obs_ids_arr[int(kb_obj_id)])
                tie_signals[int(kb_obj_id)] = {
                    'volume':         float(old_volumes.get(old_id, float('inf'))),
                    'avg_min_grasps': amg,
                }

            tiebreak_order = tuple(self.cfg["solution"].get(
                "target_tiebreak_order", ["volume", "avg_min_grasps"]))

            ranked = self.rank_final_targets(
                kb, tie_signals=tie_signals, tiebreak_order=tiebreak_order)
            if not ranked:
                print("[planner] no graspable target — every candidate is ungraspable")
                return image_to_video(self._solution_video), log
            log['target_exist'] = 1

            per_try_budget = max(total_budget / max(len(ranked), 1), 0.05)
            print(f"[planner] candidates (hard→easy, tiebreak={list(tiebreak_order)}, "
                  f"budget={total_budget:.2f}s ({per_try_budget:.2f}s/try):")
            for o, mc in ranked:
                sigs = tie_signals.get(int(o), {})
                print(f"  obj {o:>3}: mc={mc}  "
                      f"vol={sigs.get('volume', float('inf')):.6f}  "
                      f"avg_min_g={sigs.get('avg_min_grasps', 0.0):.2f}")

            total_dt = 0.0
            for attempt_idx, (obj_id, mc) in enumerate(ranked):
                t_attempt = time.time()
                ok, candidate_plan = plan_with_grasps(
                    kb, obj_id, time_limit=per_try_budget,
                )
                dt = time.time() - t_attempt
                total_dt += dt
                if ok:
                    final_target, min_clear, plan = obj_id, mc, candidate_plan
                    winning_dt = dt
                    print(f"[planner] try {attempt_idx}/{len(ranked)-1}: "
                          f"obj{obj_id} (mc={mc}) → OK in {dt:.3f}s")
                    break
                print(f"[planner] try {attempt_idx}/{len(ranked)-1}: "
                      f"obj{obj_id} (mc={mc}) → FAIL in {dt:.3f}s")

            # GT mode persists *the rank-top candidate* as the planner's
            # chosen target even if BC failed for all candidates — target
            # choice is a KB-level decision independent of BC success.
            target_for_cache_new = (final_target if final_target is not None
                                    else ranked[0][0])
            target_old = int(obs_ids_arr[target_for_cache_new])
            self.save_planner_target_to_config(
                target_old, self._lookup_obj_label(target_old))

            if plan is None:
                print(f"[planner] all {len(ranked)} candidates failed BC "
                      f"(total_dt={total_dt:.3f}s)")
                # Dump KB + AND-OR graph rooted at the rank-top candidate so
                # the failure is inspectable. `target_for_cache_new` was set
                # above to ranked[0][0] (which is also what we cached).
                self._dump_kb_and_graph(
                    kb, int(target_for_cache_new), None, None, log)
                return image_to_video(self._solution_video), log
        else:
            # ============================================================
            # Pred mode: target is dictated by the cached choice from a
            # prior GT run — no rank/fallback. Plan once on cached target.
            # ============================================================
            target_old, has_marker = self.load_planner_target_from_config()
            if not has_marker:
                raise RuntimeError(
                    f"[planner] task_config.npz at "
                    f"{self._scene_task_config_path()} has no "
                    f"`planner_chosen` marker — run GT mode (pred_folder=null) "
                    f"first to populate the cached target.")

            if target_old not in old_to_new:
                # Cached target's old_id not present in current obs_ids
                # (rare scene/h5 mismatch). Soft-fail with target_exist=0.
                print(f"[planner] cached target old_id={target_old} not in "
                      f"current obs_ids — soft-fail (target_exist=0)")
                return image_to_video(self._solution_video), log

            final_target = old_to_new[target_old]
            # Recompute min_clear from the *pred* KB for logging — same obj
            # may have a different difficulty under prediction vs GT.
            entry = kb.get(final_target)
            if entry is None or entry['status'] == 'ungraspable':
                min_clear = None
            elif entry['status'] == 'graspable':
                min_clear = 0
            else:
                min_clear = min(len(c) for c in entry['clauses'])

            log['target_exist'] = 1
            print(f"[planner] pred mode: cached target obj_old={target_old} "
                  f"→ new_id={final_target} (min_clear={min_clear})")

            t_attempt = time.time()
            ok, plan = plan_with_grasps(
                kb, final_target, time_limit=total_budget,
            )
            winning_dt = time.time() - t_attempt
            if not ok:
                print(f"[planner] BC failed for cached target obj{final_target} "
                      f"(dt={winning_dt:.3f}s) — no fallback")
                log['bc_plan_time'] = winning_dt
                # Dump KB + AND-OR graph rooted at the cached target so the
                # prediction-vs-GT divergence (which obj got mis-classified)
                # is inspectable from runs/.../kb and graph/.
                self._dump_kb_and_graph(
                    kb, int(final_target), min_clear, None, log)
                return image_to_video(self._solution_video), log

        log['bc_plan_time']   = winning_dt
        log['bc_plan_exist']  = 1
        log['bc_plan_length'] = len(plan)
        log['plan']           = plan
        print(f"[planner] final_target=obj{final_target} (min_clear={min_clear}) "
              f"plan ({len(plan)} steps, bc_plan_time={winning_dt:.3f}s): {plan}")

        # Stash artifacts needed by sample_plan_step_grasp_pose below.
        assert self.num_envs == 1, "GORun.solve assumes num_envs == 1 (per-env obstruction h5)"
        self._active_plan = plan
        self._obs_grasp   = np.asarray(obs_data['grasp'])    # (num_grasp, 7), robot frame, wxyz
        self._obs_ids     = np.asarray(obs_data['obs_ids'])  # new_id -> pykin/isaac old_id

        # Snapshot world poses at plan time so kb_plan_vis_debug renders the
        # *planning* state even if called after sim has stepped. Layout per
        # row: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)].
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # Reproduce Isaac Gym's cameras for this task: the per-task cams from
        # task_camera_init_state + the fixed vis cam wired in fetch_base.py:686.
        # Each entry is a flat 6-vec [eye(3), target(3)]. env 0 by the
        # num_envs==1 assert above.
        task_cams = list(self.task_camera_init_state[0][self._task_idx])
        vis_cam   = np.array([-2.0, 0.0, 2.5, 0.0, 0.0, 0.5], dtype=np.float64)
        cam_poses = [np.asarray(p, dtype=np.float64).reshape(-1) for p in task_cams]
        cam_poses.append(vis_cam)

        self._plan_snapshot = {
            'robot':  self._robot_base_state.detach().cpu().numpy().copy(),  # (n_env, 13)
            'qpos':   self._q[:, :7].detach().cpu().numpy().copy(),          # (n_env, 7) arm joints
            'table':  self._table_base_state.detach().cpu().numpy().copy(),  # (n_env, 13)
            'scene':  self._scene_base_state.detach().cpu().numpy().copy(),  # (n_env, 13)
            'object': self._obj_state.detach().cpu().numpy().copy(),         # (n_env, num_obj, 13)
            'cams':   cam_poses,                                              # list of (6,) [eye(3), target(3)]
        }

        self.set_target_color()

        # Dump planner artifacts under runs/<experiment_name>/ for inspection.
        # Shared with the BC-failure paths so failed plans still leave a KB
        # snapshot + AND-OR graph behind for debugging (e.g. seeing which obj
        # got mis-classified as ungraspable in pred mode).
        self._dump_kb_and_graph(kb, int(final_target), min_clear, plan, log)

        # (c) Plan visualization: kb_vis/t<idx>.png — target obj red,
        #     per-step representative grasp (first index) as gripper marker.
        try:
            self.kb_plan_vis_debug(plan, final_target, env_idx=0)
        except Exception as e:
            print(f"[planner] kb_plan_vis dump failed: {type(e).__name__}: {e}")

        # input("next?")

        # Execute plan steps in order. Intermediate steps drop the obj at the
        # free-space target and release; final step keeps holding (standard
        # fetch benchmark). Abort on first step failure.
        computing_time_ref = [computing_time]
        for step_idx, (obj_id, _) in enumerate(self._active_plan):
            is_final = (step_idx == len(self._active_plan) - 1)
            release  = not is_final
            print(f"\n[step {step_idx}/{len(self._active_plan)-1}] "
                  f"obj={obj_id}  release_after={release}")
            ok_step = self._execute_plan_step(step_idx, log,
                                              release_after=release,
                                              computing_time_ref=computing_time_ref)
            if not ok_step:
                print(f"[step {step_idx}] FAILED — aborting plan")
                break
        else:
            log['motion_plan_success'] = 1
            self._plan_success = True
        computing_time = computing_time_ref[0]

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()
        
        return image_to_video(self._solution_video), log

    """
    Debug Visualization
    """
    def _ensure_pykin_robot(self):
        """Lazy-init pykin SingleArm for visualization. Mirrors the setup in
        fetch_mesh_curobo_go.py:138-146 but only on first call — keeps the
        Run task lean for runs that never hit kb_plan_vis_debug.
        """
        if getattr(self, "pykin_robot", None) is not None:
            return
        from isaacgymenvs.tasks.fetch.utils.load_utils import get_franka_panda_asset
        from pykin.robots.single_arm import SingleArm
        from pykin.kinematics.transform import Transform
        from pykin.collision.collision_manager import CollisionManager
        robot_path = get_franka_panda_asset(type='franka_r3_cvx_pykin')
        self.pykin_robot = SingleArm(
            os.path.join(robot_path['asset_root'], robot_path['urdf_file']),
            offset=Transform(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
            has_gripper=True,
            gripper_name="panda_r3_gripper")
        self.pykin_robot.setup_link_name(base_name='panda_link0', eef_name='panda_link7')
        cm = CollisionManager(is_robot=True)
        cm.setup_robot_collision(self.pykin_robot, geom="collision")
        self._pykin_robot_geom = "collision"

    def kb_plan_vis_debug(self, plan, final_target, env_idx=0):
        """Render the scene + per-step representative grasp markers to PNG.

        Color language is shared with the AND-OR graph via
        `singulation_planner.step_color_rgb`: the last step (target) is red,
        earlier steps cycle through a fixed palette. Plan objs are *tinted*
        with their step color (alpha=200, slightly translucent), while their
        grasp marker is drawn in the same color but fully solid — same color
        language as the AND-OR graph's obj font, with the shape difference
        (obj mesh vs gripper outline) plus the alpha gap separating obj from
        grasp. Non-plan candidate objs keep their original material, faded
        (alpha=90) when `solution.kb_plan_vis.fade_obstacles` is True.
        Saved to runs/<experiment_name>/kb_vis/t<task_idx>.png.

        Args:
            plan: List of (obj_id_new, [grasp_idx, ...]) tuples from
                  `plan_with_grasps`. Empty list → no-op.
            final_target: new_id of the final target obj (red tint).
            env_idx: env to render (assumes num_envs == 1 in solve()).
        """
        if not plan:
            print("[kb_plan_vis] empty plan — skipped")
            return

        # Read from the plan-time snapshot stashed in solve() so the picture
        # reflects the world state the planner saw, not whatever sim has
        # stepped into since. Layout per row: [pos(3), quat_xyzw(4), vel(6)].
        if not hasattr(self, "_plan_snapshot"):
            print("[kb_plan_vis] no plan snapshot — solve() didn't reach the "
                  "stash. Skipping.")
            return
        snap = self._plan_snapshot

        # yaml toggles (defaults reproduce previous behavior).
        vis_cfg = self.cfg.get("solution", {}).get("kb_plan_vis", {}) or {}
        show_robot_table    = bool(vis_cfg.get("show_robot_table", True))
        fade_obstacles      = bool(vis_cfg.get("fade_obstacles",   True))
        gripper_tube_radius = float(vis_cfg.get("gripper_tube_radius", 0.0035))
        # Grasp marker = obj's step color × this factor. Fixed at 0.7 —
        # "shade darker, not muddy", keeps the same hue family as the obj
        # tint so motion intent reads at a glance.
        grasp_color_darken  = 0.7

        def _xyzw_to_wxyz(q):  # trimesh quaternion_matrix expects wxyz
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.axis())

        # Robot base in world — needed for grasp marker (robot→world); the
        # robot-link transforms also use it but only when rendered.
        rbs = snap['robot'][env_idx]
        T_world_robot = (tr.translation_matrix(rbs[:3])
                         @ tr.quaternion_matrix(_xyzw_to_wxyz(rbs[3:7])))

        # --- Table (same translucency as robot) -------------------------
        if show_robot_table:
            tbs = snap['table'][env_idx]
            T_table = (tr.translation_matrix(tbs[:3])
                       @ tr.quaternion_matrix(_xyzw_to_wxyz(tbs[3:7])))
            table_mesh = trimesh.creation.box(
                extents=self.table_asset[env_idx]['dim'], transform=T_table)
            table_mesh.visual.face_colors = [180, 180, 220, 90]
            scene.add_geometry(table_mesh)

        # --- Scene meshes -----------------------------------------------
        sbs = snap['scene'][env_idx]
        T_scene = (tr.translation_matrix(sbs[:3])
                   @ tr.quaternion_matrix(_xyzw_to_wxyz(sbs[3:7])))
        for f in self.scene_asset[env_idx]['files']:
            mesh = trimesh.load(f).apply_transform(T_scene)
            scene.add_geometry(mesh)

        # --- Candidate objects (target tinted red) ----------------------
        # `_obs_ids` is new_id → old_id (isaac/pykin); reverse for lookup.
        old_to_new = {int(old): int(new) for new, old in enumerate(self._obs_ids)}
        obj_state = snap['object'][env_idx]  # (num_objs, 13)

        # Objects involved in the plan (target + intermediate obstacles) stay
        # opaque; everything else fades to the same ghost alpha as the
        # robot/table so the eye is drawn to the plan.
        plan_obj_new_ids = {int(o) for o, _ in plan}
        cand_old_ids = self.task_cand_obj_index[env_idx][self.get_task_idx()]
        for i in cand_old_ids:
            T_obj = (tr.translation_matrix(obj_state[i, :3])
                     @ tr.quaternion_matrix(_xyzw_to_wxyz(obj_state[i, 3:7])))
            new_id = old_to_new.get(int(i))
            in_plan = new_id is not None and new_id in plan_obj_new_ids

            # Re-load the mesh from file so textures/materials survive — the
            # pre-loaded `object_asset[i]['mesh']` was `trimesh.util.concatenate`d
            # at scene-build time, which drops per-geometry visuals. Loading
            # with process=False keeps multi-material Scenes intact.
            obj_file = self.object_asset[env_idx][i].get('file')
            loaded = (trimesh.load(obj_file, process=False)
                      if obj_file else self.object_asset[env_idx][i]['mesh'].copy())

            # Normalize to a list of Trimesh pieces preserving their visuals.
            if isinstance(loaded, trimesh.Scene):
                pieces = list(loaded.geometry.values())
            elif isinstance(loaded, trimesh.Trimesh):
                pieces = [loaded]
            else:
                pieces = list(loaded) if hasattr(loaded, '__iter__') else [loaded]

            # Pre-compute this obj's step color if it's part of the plan.
            plan_step_rgb = None
            if in_plan:
                step_idx = next(i for i, (o, _) in enumerate(plan)
                                if int(o) == int(new_id))
                plan_step_rgb = step_color_rgb(step_idx, len(plan))

            for piece in pieces:
                if not isinstance(piece, trimesh.Trimesh):
                    continue
                piece = piece.copy().apply_transform(T_obj)
                if plan_step_rgb is not None:
                    # Plan obj: tint with step color (alpha=200, slightly
                    # translucent). Same color as the grasp marker below; the
                    # obj-vs-gripper shape difference + the alpha gap (200 vs
                    # solid 255) keep them visually separable.
                    if isinstance(piece.visual, trimesh.visual.texture.TextureVisuals):
                        piece.visual = piece.visual.to_color()
                    piece.visual.face_colors = [
                        plan_step_rgb[0], plan_step_rgb[1], plan_step_rgb[2], 200]
                elif fade_obstacles:
                    # Non-plan: keep original material, drop alpha to ghost
                    # the obj. Convert TextureVisuals so the alpha channel
                    # can be rewritten without losing per-face RGB.
                    if isinstance(piece.visual, trimesh.visual.texture.TextureVisuals):
                        piece.visual = piece.visual.to_color()
                    fc = np.asarray(piece.visual.face_colors).copy()
                    if fc.ndim == 1:           # single broadcast color
                        fc = np.tile(fc, (len(piece.faces), 1))
                    if fc.shape[-1] == 3:      # add alpha channel if RGB-only
                        fc = np.concatenate(
                            [fc, np.full((fc.shape[0], 1), 255, dtype=fc.dtype)],
                            axis=-1)
                    fc[:, 3] = 90              # match robot/table ghost alpha
                    piece.visual.face_colors = fc
                # else (non-plan && !fade_obstacles): original material, opaque.
                scene.add_geometry(piece)

        # --- Robot in initial pose (semi-transparent) ------------------
        # Snapshot's qpos is the joint state at plan time. pykin's set_transform
        # updates info[geom][link][3] = h_mat (in robot frame), which we then
        # carry to world via T_world_robot. Alpha < 255 → translucent so the
        # robot reads as a "ghost" overlay rather than blocking objects/grasps.
        # Gated by `solution.kb_plan_vis.show_robot_table`.
        if show_robot_table:
            try:
                self._ensure_pykin_robot()
                self.pykin_robot.set_transform(snap['qpos'][env_idx])
                robot_color = [180, 180, 220, 90]  # light blue, ~35% alpha
                for link, info in self.pykin_robot.info[self._pykin_robot_geom].items():
                    if info[1] != "mesh":
                        continue
                    h_mat = info[3]
                    T_link = T_world_robot @ h_mat
                    pieces = info[2] if isinstance(info[2], list) else [info[2]]
                    for piece in pieces:
                        if not isinstance(piece, trimesh.Trimesh):
                            continue
                        pc = piece.copy()
                        if isinstance(pc.visual, trimesh.visual.texture.TextureVisuals):
                            pc.visual = pc.visual.to_color()
                        pc.visual.face_colors = robot_color
                        scene.add_geometry(pc, transform=T_link)
            except Exception as e:
                print(f"[kb_plan_vis] robot render skipped: {type(e).__name__}: {e}")

        # --- Per-step representative grasp marker -----------------------
        # `_obs_grasp` is wxyz in *robot* frame — convert to world by composing
        # with the robot base pose. vis_rot aligns the marker frame with the
        # gripper convention (see grasp_vis_debug in fetch_mesh_curobo_go.py).
        vis_rot = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        n_grasp = self._obs_grasp.shape[0]
        n_drawn = 0
        for step_idx, (obj_id_new, grasp_ids) in enumerate(plan):
            if not grasp_ids:
                continue
            g = int(grasp_ids[0])
            if not (0 <= g < n_grasp):
                print(f"[kb_plan_vis] step {step_idx}: grasp idx {g} OOB "
                      f"(num_grasp={n_grasp}) — skipped")
                continue
            pos  = self._obs_grasp[g, :3]
            quat = self._obs_grasp[g, 3:7]  # wxyz
            T_robot_grasp = (tr.translation_matrix(pos)
                             @ tr.quaternion_matrix(quat))
            T_grasp = T_world_robot @ T_robot_grasp @ vis_rot
            # Same color FAMILY as the obj tint above, but darkened by a
            # fixed factor so the gripper outline pops against the obj that
            # shares its base hue (alpha=200 tint vs solid darker gripper).
            base = step_color_rgb(step_idx, len(plan))
            color = [max(0, min(255, int(round(c * grasp_color_darken))))
                     for c in base]
            scene.add_geometry(
                create_gripper_marker(color, tube_radius=gripper_tube_radius)
                .apply_transform(T_grasp))
            n_drawn += 1

        # --- Render one PNG per Isaac Gym camera -----------------------
        # Match the sim's intrinsics so the trimesh shots line up with what
        # the cameras actually see in env (FOV + aspect ratio). Resolution is
        # scaled up 2× to keep grasp markers legible at output size.
        cam_cfg = self.cfg["env"]["cam"]
        cam_w, cam_h = int(cam_cfg["width"]), int(cam_cfg["height"])
        scale = 2
        out_res = (cam_w * scale, cam_h * scale)
        # trimesh.Camera accepts (h_fov_x, v_fov_y); derive v from h by aspect.
        hfov = float(cam_cfg["hov"])
        vfov = float(np.degrees(
            2.0 * np.arctan(np.tan(np.radians(hfov) / 2.0) * cam_h / cam_w)))
        scene.camera.fov = (hfov, vfov)
        scene.camera.resolution = out_res

        out_dir = os.path.join(
            "runs", self.cfg.get("experiment_name", "default"), "kb_vis")
        os.makedirs(out_dir, exist_ok=True)
        task_tag = f"t{self.get_task_idx()}"

        saved = []
        for j, pose in enumerate(snap['cams']):
            eye, tgt = pose[:3], pose[3:6]
            scene.camera_transform = make_camera_transform(np.asarray(eye),
                                                           np.asarray(tgt))
            png = scene.save_image(resolution=out_res)
            out_path = os.path.join(out_dir, f"{task_tag}_cam{j}.png")
            with open(out_path, "wb") as f:
                f.write(png)
            saved.append(out_path)

        print(f"[kb_plan_vis] saved {len(saved)} views to {out_dir}/{task_tag}_cam*.png "
              f"(plan_steps={len(plan)}, grasp_markers={n_drawn}, "
              f"fov={hfov:.1f}°x{vfov:.1f}°, res={out_res}, "
              f"show_robot_table={show_robot_table}, fade_obstacles={fade_obstacles}, "
              f"gripper_tube_radius={gripper_tube_radius})")

