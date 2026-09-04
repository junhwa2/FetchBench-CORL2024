"""Multi-view variant of the closed-loop server task.

NEW FILE ONLY — does not edit server_task.py or any FetchBench source.

Coverage finding (see KNOWN_ISSUES.md "Perception coverage").  The observed
point cloud that VORM/GraspGen consume is built from the TWO scene cameras
(FetchBase num_cam=2), and `filter_pointcloud` already merges all point-cloud
cameras (fetch_ptd.py:98-100) — so `server_cam_idx` does NOT limit the cloud;
it only picks the RGB/VLM image.  The real coverage limit is camera PLACEMENT:
the scene defines exactly two poses, both at the +x support edge looking back
at the scene (trimesh_scene.py:396 `assert i <= 1, "left & right"`).  They
share nearly the same front viewpoint, so back / mutually occluded objects get
0-7 points and fall through the 8-point GraspGen gate (grasp_gen.py:177) and
the 'valid' object filter (fetch_mesh_curobo_go.py:1241-1265).

FetchPointCloudBase builds the cloud engine from `self.cameras[idx][:-1]`,
deliberately DROPPING the elevated 'vis' camera at world [-2,0,2.5]->[0,0,0.5]
(fetch_base.py:693-697, fetch_ptd.py:23).  That vis camera is exactly the
complementary top/back viewpoint that resolves front-to-back occlusion.  This
subclass rebuilds the cloud engine to INCLUDE it -> a 3-view cloud, no new
sensors, no edits.

Depth note: the vis camera sits ~3.2 m from the scene, beyond the default
depth_max=2.5 m, so its points would be culled (point_cloud_utils.py:39).  The
paired config FetchMeshCuroboGORunVLMServerMV.yaml raises depth_max; the 1.06 m
workspace sphere crop in _filter_pc removes anything outside the workspace
afterward, so the wider range cannot leak background into the cloud.
"""
from isaacgymenvs.tasks.fetch.utils.point_cloud_utils import CameraPointCloud
from isaacgymenvs.tasks.fetch.vlm_closed_loop.server_task import (
    FetchMeshCuroboGORunVLMServer,
)


class FetchMeshCuroboGORunVLMServerMV(FetchMeshCuroboGORunVLMServer):
    """Server task whose point cloud merges the 2 scene cams + the vis cam."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # Parent (FetchPointCloudBase) built self.cam_point_clouds from
        # self.cameras[idx][:-1] — front scene cams only, vis cam dropped.
        # Rebuild with the FULL per-env handle list so the elevated vis view
        # also feeds the segmented cloud.  Reassigning self.cam_point_clouds
        # replaces the parent's engine; the extra wrappers over the shared
        # front-cam GPU tensors are harmless.
        ptd_cam_handles = [self.cameras[idx] for idx in range(len(self.envs))]
        graphics_device = (self.graphics_device_id
                           if self.graphics_device_id >= 0 else 'cpu')
        self.cam_point_clouds = CameraPointCloud(
            self.sim, self.gym, self.envs, ptd_cam_handles,
            camera_params=self.cfg["env"]["cam"],
            depth_max=self.cfg["env"]["cam"]["depth_max"],
            depth_min=self.cfg["env"]["cam"]["depth_min"],
            graphics_device=graphics_device,
            compute_device=self.device,
        )


# --- self-register so tasks/__init__.py stays untouched ------------------- #
from isaacgymenvs.tasks import isaacgym_task_map  # noqa: E402

isaacgym_task_map["FetchMeshCuroboGORunVLMServerMV"] = \
    FetchMeshCuroboGORunVLMServerMV
