"""Closed-loop server task for the VLM+VORM pipeline (see VLM_SIM_RPC_SPEC.md).

`FetchMeshCuroboGORunVLMServer` turns the FetchBench sim into an interactive
server for the ThinkGrasp-style loop driven by vorm_pipeline (`loop_runner_mj`):

    reset(task_idx) -> observe -> decide_goal (sim-side Gemini-ER VLM)
                    -> [master plans a grasp] -> execute -> observe -> ...

It reuses two things wholesale and edits neither:
  * FetchMeshCuroboGORunVLM  -> the VLM machinery (Gemini-ER client, category
    label building, seg->old_id resolution).
  * FetchMeshCuroboGORun     -> `_execute_plan_step` (curobo reach + close +
    carry) and `gen_pc_from_camera` / `save_pc` (segmented point cloud in the
    robot base frame, xyz/rgb/id — the same .ply contract vorm's
    `perception.from_ply` reads).

The novelty here is `server_execute`: it runs the sim's grasp pipeline on an
**externally supplied** 6-DoF grasp (from the master's GraspGen), by injecting
`_obs_grasp` / `_obs_ids` / `_active_plan` and calling the unmodified
`_execute_plan_step`. This is the only wiring needed because `obs_obj_id`
throughout GORun is just an index into `self._obs_ids` -> isaac old_id.

The RPC transport lives in `server_entry.py` + `sim_rpc.py`; this class only exposes pure
`server_*` methods returning numpy/py data. Nothing here writes RPC files.

Registration mutates the shared `isaacgym_task_map` on import, so
`tasks/__init__.py` stays untouched.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..fetch_mesh_curobo_go_run_vlm import FetchMeshCuroboGORunVLM
from ..fetch_ptd import FetchPointCloudBase
from ..fetch_mesh_curobo_go import FetchMeshCuroboGO
from ..utils.vlm_target import (
    SEG_ID_OFFSET,
    VLMCallError,
    resolve_point,
    visible_pixel_count,
)


def _mat16_to_pos_wxyz(m16: List[float]) -> np.ndarray:
    """4x4 row-major homogeneous matrix -> [x,y,z, qw,qx,qy,qz] (curobo wxyz).

    Matches the layout of `obs_data['grasp']` rows that GORun feeds straight
    into IK (robot base frame). Uses a numerically stable matrix->quaternion.
    """
    m = np.asarray(m16, dtype=np.float64).reshape(4, 4)
    R = m[:3, :3]
    t = m[:3, 3]
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=np.float32)


def _exec_diag(log: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, JSON-safe per-stage summary of a `_execute_plan_step` log.

    Turns the namespaced step0_* entries (numpy arrays of per-grasp success,
    etc.) into scalars so the master can see WHERE a step failed: IK ->
    grasp-motion -> carry-motion -> execute-success (slip/no-move) check.
    """
    def scalarize(v):
        try:
            arr = np.asarray(v)
            if arr.dtype == bool:
                return {"n_true": int(arr.sum()), "n": int(arr.size)}
            if arr.ndim == 0:
                return v.item() if hasattr(v, "item") else v
        except Exception:
            pass
        if isinstance(v, (bool, int, float, str)) or v is None:
            return v
        return str(v)

    keys = ("ik_plan_success", "pre_grasp_plan_success", "grasp_plan_success",
            "fetch_plan_success", "plan_success", "plan_failure",
            "execute_success", "execute_failure", "grasp_finger_obj_contact",
            "dbg_disp", "dbg_dist_drift")
    out: Dict[str, Any] = {}
    for k in keys:
        kk = "step0_" + k
        if kk in log:
            out[k] = scalarize(log[kk])
    return out


def _to_grasp7(pose) -> np.ndarray:
    """Accept a grasp as either a 16-vec (4x4) or a 7-vec [pos, wxyz]."""
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size == 16:
        return _mat16_to_pos_wxyz(arr)
    if arr.size == 7:
        return arr.astype(np.float32)
    raise ValueError("grasp pose must have 16 (4x4) or 7 (pos+wxyz) entries, "
                     "got {}".format(arr.size))


class FetchMeshCuroboGORunVLMServer(FetchMeshCuroboGORunVLM, FetchPointCloudBase):
    """Closed-loop server. Multiple-inherits FetchPointCloudBase so the
    depth-based point-cloud engine (`self.cam_point_clouds`, `get_camera_data`)
    is set up in the cooperative __init__ chain alongside GORun's curobo stack.
    MRO: [Server, GORunVLM, GORun, FetchMeshCurobo, FetchSolutionBase,
    FetchPointCloudBase, FetchBase, VecTask].

    The three segmented-cloud helpers below are BORROWED from FetchMeshCuroboGO
    as plain functions (bound here as methods) rather than inherited, so GO's
    heavy __init__ (a second IK solver, pykin, the obs_data_gen config block)
    never runs. Their module globals (IsaacGymId, PC_BOUND_TYPE, o3d, plt)
    resolve against fetch_mesh_curobo_go, so nothing else needs importing. They
    only require `self.pc_bound_option` / `self.pc_voxel_size` / `self.debug_viz`,
    set lazily by `_ensure_pc_cfg`.
    """

    gen_pc_from_camera = FetchMeshCuroboGO.gen_pc_from_camera
    _get_seg_color = FetchMeshCuroboGO._get_seg_color
    _filter_pc = FetchMeshCuroboGO._filter_pc
    save_pc = FetchMeshCuroboGO.save_pc      # writes the xyz/rgb/id .ply vorm reads

    def _ensure_pc_cfg(self) -> None:
        if not hasattr(self, "pc_bound_option"):
            v = self._vlm_cfg()
            self.pc_bound_option = v.get("pc_bound_option", "panda_w_gripper_sphere")
            self.pc_voxel_size = float(v.get("pc_voxel_size", 0.015))
        if not hasattr(self, "debug_viz"):
            self.debug_viz = False

    # ------------------------------------------------------------------ #
    # VLM client (reveal-reasoning by default for the closed loop)
    # ------------------------------------------------------------------ #
    def _vlm_client(self):
        """Override the Phase-1 factory: for the closed loop, build a
        RevealERClient (action + point) when backend=gemini and vlm.reveal is
        on. Oracle / reveal-off fall back to the stock pointing client, and
        `server_decide_goal` then uses the category proxy for is_target."""
        if getattr(self, "_vlm_client_obj", None) is None:
            from .reveal_client import build_reveal_client
            self._vlm_client_obj = build_reveal_client(self._vlm_cfg())
        return self._vlm_client_obj

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _settle(self, steps: Optional[int] = None) -> None:
        if steps is None:
            steps = int(self.cfg["solution"].get("init_steps", 60))
        for _ in range(int(steps)):
            self.env_physics_step()
            self.post_phy_step()

    def _server_cam_idx(self) -> int:
        """Fixed primary camera for BOTH the point cloud RGB and the VLM view.

        Deliberately target-agnostic (never uses ground truth to pick a camera)
        so the closed loop cannot leak the target's location the way Phase-1's
        `pick_camera` intentionally does.
        """
        return int(self._vlm_cfg().get("server_cam_idx", 0))

    def _render_primary(self, env_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rgb_all, seg_all = self.get_camera_image(rgb=True, seg=True)
        cam = self._server_cam_idx()
        rgb = np.ascontiguousarray(
            np.asarray(rgb_all[env_idx][cam])[..., :3]).astype(np.uint8)
        seg = np.asarray(seg_all[env_idx][cam])
        if seg.ndim == 3:
            seg = seg[..., 0]
        return rgb, seg

    def _save_vlm_io(self, rgb, seg, point_frac, chosen_old, seg_offset,
                     task_idx, step, action, reason):
        """Save BOTH what the VLM actually received and an annotated overlay.

        `<stem>_input.png`   : the exact clean RGB fed to the VLM (no marks).
        `<stem>_overlay.png` : same image + red dot (VLM point), green box
                               (chosen object), blue box (GT target), caption.
        Written under `_vlm_log_dir()` (vlm_target_logs/<server exp>/)."""
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:  # noqa: BLE001
            print("[VLMServer] PIL unavailable, skip image save: {}".format(exc))
            return
        d = self._vlm_log_dir()
        scene_tag = str(self.cfg["task"]["scene_config_path"][0]).replace("/", "_")
        stem = os.path.join(d, "{}_t{}_s{}".format(scene_tag, int(task_idx), int(step)))
        rgb_u8 = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)

        # 1) clean input — byte-for-byte what gets PNG-encoded to the VLM
        Image.fromarray(rgb_u8).save(stem + "_input.png")

        # 2) overlay
        img = Image.fromarray(rgb_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        h, w = rgb_u8.shape[:2]
        if point_frac is not None:
            px, py = int(point_frac[0] * (w - 1)), int(point_frac[1] * (h - 1))
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(255, 0, 0), width=3)
        gt_old = int(getattr(self, "_server_gt_target_old", -1))
        for old_id, color in ((chosen_old, (0, 255, 0)), (gt_old, (0, 128, 255))):
            if old_id is None or int(old_id) < 0:
                continue
            ys, xs = np.where(seg == int(old_id) + int(seg_offset))
            if xs.size:
                draw.rectangle([int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max())], outline=color, width=3)
        draw.text((5, 5), "{} | {}".format(action, (reason or "")[:70]),
                  fill=(255, 255, 0))
        img.save(stem + "_overlay.png")
        print("[VLMServer] saved VLM I/O: {}_{{input,overlay}}.png".format(stem))

    def _save_task_video(self, fps: int = 10):
        """Flush the accumulated per-task frames to ONE mp4 under the VLM log
        dir: `vlm_target_logs/<exp>/<scene_tag>_t<task_idx>.mp4`.

        Frames are collected by follow_motion_trajs -> log_video during execute;
        the buffer is reset per task in server_reset. Called after every execute
        step so the file grows into the full-task video (later steps append to
        the same buffer, so the mp4 is simply rewritten). Best-effort: an empty
        buffer (e.g. a plan that failed before any physics step, like no_ik) or
        any encoder error is swallowed so video never breaks the loop. Returns
        the mp4 path or None."""
        frames = getattr(self, "_solution_video", None)
        if not frames:
            return None
        try:
            import imageio.v3 as iio
            from ..fetch_mesh_curobo import image_to_video
            vid = image_to_video(frames)
            if not vid:
                return None
            d = self._vlm_log_dir()
            scene_tag = str(self.cfg["task"]["scene_config_path"][0]).replace("/", "_")
            path = os.path.join(d, "{}_t{}.mp4".format(
                scene_tag, int(getattr(self, "_server_task_idx", 0))))
            iio.imwrite(path, np.stack(vid, axis=0), fps=int(fps))
            return path
        except Exception as exc:  # noqa: BLE001 - video is best-effort
            print("[VLMServer] video save failed: {}".format(exc))
            return None

    def _qpos7(self, env_idx: int = 0) -> np.ndarray:
        self._refresh()
        return self._q[env_idx, :7].detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # RPC-facing methods (pure data in/out; no file IO)
    # ------------------------------------------------------------------ #
    def server_reset(self, task_idx: int) -> Dict[str, Any]:
        """Place the scene at `task_idx`'s initial state, settle, and return the
        benchmark instruction + ground-truth target (for scoring only)."""
        assert self.num_envs == 1, "VLMServer assumes num_envs == 1"
        self.reset_task(int(task_idx))
        self._settle()

        # Fresh per-task video buffer. The closed loop never calls the normal
        # solve() (which is where GORun resets this), so without an explicit
        # reset here the frame buffer would keep growing across every task since
        # server start. follow_motion_trajs -> log_video appends frames during
        # each execute; server_execute flushes a single mp4 per task.
        self._solution_video = []
        self._video_frame = 0

        env_idx = 0
        gt_old = self._gt_target_old(env_idx, int(task_idx))
        cand_old = self._candidate_old_ids(env_idx, int(task_idx))

        # Build the instruction the same way Phase 1 does, but the qualifier is
        # grounded against the *initial* view. The GT id and category are kept
        # server-side for scoring; only the instruction text crosses to the VLM.
        _, seg = self._render_primary(env_idx)
        vlm_cfg = self._vlm_cfg()
        seg_offset = int(vlm_cfg.get("seg_offset", SEG_ID_OFFSET))
        if str(vlm_cfg.get("label_source", "category")) == "category":
            target_label, _disamb = self._build_category_label(
                gt_old, cand_old, seg, seg_offset, env_idx,
                normalize=bool(vlm_cfg.get("normalize_category", True)),
                ambiguity_scope=str(vlm_cfg.get("ambiguity_scope", "visible")),
                min_visible_px=int(vlm_cfg.get("min_visible_px", 30)),
            )
            tgt_cat = self._lookup_obj_category(gt_old, env_idx)
        else:
            target_label = self._lookup_obj_label(gt_old, env_idx)
            tgt_cat = target_label
        template = vlm_cfg.get("instruction_template", "Pick up the {label}.")

        self._server_task_idx = int(task_idx)
        self._server_gt_target_old = int(gt_old)
        self._server_target_category = self._normalize_category(tgt_cat)
        self._server_instruction = template.format(label=target_label)
        self._server_step = 0

        return {
            "task_idx": int(task_idx),
            "scene": str(self.cfg["task"]["scene_config_path"][env_idx]),
            "instruction": self._server_instruction,
            "target_seg_id": int(gt_old) + seg_offset,   # scoring only
            "target_category": self._lookup_obj_category(gt_old, env_idx),
            # number of tasks actually loaded for this scene (capped by
            # env.numTasks); the task-config arrays are per-scene, len == #tasks.
            "num_tasks": int(len(self.task_obj_index[env_idx])),
        }

    def server_observe(self, env_idx: int = 0) -> Dict[str, Any]:
        """Render RGB + segmented base-frame point cloud + arm joints.

        Returns in-memory arrays; the RPC layer serialises rgb->png, pc->ply
        (via `save_pc`, the vorm-compatible xyz/rgb/id contract), qpos->npy.
        """
        self._ensure_pc_cfg()
        rgb, _seg = self._render_primary(env_idx)
        pc = self.gen_pc_from_camera(env_idx)          # {xyz, rgb, id(seg)}
        qpos = self._qpos7(env_idx)
        h, w = rgb.shape[:2]
        return {
            "step": int(getattr(self, "_server_step", 0)),
            "rgb": rgb,
            "pc": pc,
            "qpos": qpos,
            "cam": {"width": int(w), "height": int(h),
                    "cam_idx": self._server_cam_idx()},
        }

    def server_decide_goal(self, instruction: Optional[str] = None
                           ) -> Dict[str, Any]:
        """Run the sim-side Gemini-ER VLM on the current view and resolve its
        point to a scene object (seg id). This is where the reveal reasoning
        happens: with the target fully occluded the model must point at an
        occluder to remove; VORM cannot do this (no cloud for a hidden object).
        """
        env_idx = 0
        instr = instruction or getattr(self, "_server_instruction", "")
        task_idx = int(getattr(self, "_server_task_idx", self.get_task_idx()))
        vlm_cfg = self._vlm_cfg()
        seg_offset = int(vlm_cfg.get("seg_offset", SEG_ID_OFFSET))

        rgb, seg = self._render_primary(env_idx)
        cand_old = self._candidate_old_ids(env_idx, task_idx)
        # Only objects still visible are pointable this step; a removed/carried
        # object leaves the candidate set naturally (0 visible px).
        context = {"seg": seg, "cand_old_ids": cand_old, "seed": task_idx}
        client = self._vlm_client()

        vlm_error = None
        point_frac = None
        vlm_action = None       # 'grasp_target' | 'remove_occluder' | None
        vlm_reason = None
        vlm_seen = None         # model's enumerated visible objects (grounding CoT)
        vlm_found = None        # model's own target_found flag

        if hasattr(client, "decide"):
            # Reveal-reasoning path: the model returns its action AND the point.
            # is_target then comes from the model's own decision, not a proxy.
            try:
                dec = client.decide(rgb, instr, context=context)
                point_frac = dec.get("point")
                vlm_action = dec.get("action")
                vlm_reason = dec.get("reason")
                vlm_seen = dec.get("visible_objects")
                vlm_found = dec.get("target_found")
            except (VLMCallError, Exception) as exc:  # noqa: BLE001
                vlm_error = "{}: {}".format(type(exc).__name__, exc)
            pts = [point_frac]
        else:
            # Plain pointing fallback (oracle / reveal disabled).
            try:
                pts = client.point_samples(
                    rgb, instr, context=context,
                    n=max(1, int(vlm_cfg.get("num_samples", 1))))
            except (VLMCallError, Exception) as exc:  # noqa: BLE001
                pts = [None]
                vlm_error = "{}: {}".format(type(exc).__name__, exc)

        window = int(vlm_cfg.get("resolve_window", 15))
        allow_snap = bool(vlm_cfg.get("allow_centroid_snap", True))
        chosen_old, snapped, method = None, False, "no_point"
        for p in pts:
            rec = resolve_point(p, seg, cand_old, seg_offset,
                                window=window, allow_centroid_snap=allow_snap)
            if rec.get("old_id") is not None:
                chosen_old = int(rec["old_id"])
                snapped = bool(rec.get("snapped", False))
                method = rec.get("method", "point")
                point_frac = list(p) if p is not None else point_frac
                break

        # Optionally persist exactly what the VLM saw (clean input) + an
        # annotated overlay, for both resolved and unresolved decisions.
        if bool(vlm_cfg.get("save_vlm_io", False)):
            self._save_vlm_io(rgb, seg, point_frac, chosen_old, seg_offset,
                              task_idx, int(getattr(self, "_server_step", 0)),
                              vlm_action, vlm_reason)

        if chosen_old is None:
            return {"goal_seg_id": -1, "is_target": False, "vlm_point": None,
                    "reason": vlm_reason or "no_resolvable_point",
                    "action": vlm_action,
                    "target_visible": bool(visible_pixel_count(
                        seg, self._server_gt_target_old, seg_offset) > 0),
                    "snapped": False, "vlm_error": vlm_error, "method": method}

        # is_target: prefer the model's explicit action; if the client can't
        # decide (oracle / reveal off), fall back to a category-match proxy that
        # never uses the GT object id (it only checks whether the *named*
        # category was selected).
        if vlm_action == "grasp_target":
            is_target = True
        elif vlm_action == "remove_occluder":
            is_target = False
        else:
            chosen_cat = self._normalize_category(
                self._lookup_obj_category(chosen_old, env_idx))
            is_target = bool(chosen_cat and chosen_cat ==
                             getattr(self, "_server_target_category", None))

        return {
            "goal_seg_id": int(chosen_old) + seg_offset,
            "is_target": is_target,
            "vlm_point": list(point_frac) if point_frac is not None else None,
            "reason": vlm_reason or ("category_match" if is_target
                                     else "occluder_to_remove"),
            "action": vlm_action,
            "visible_objects": vlm_seen,     # what the model reported seeing
            "target_found": vlm_found,       # model's own found flag
            "target_visible": bool(visible_pixel_count(
                seg, self._server_gt_target_old, seg_offset) > 0),  # diagnostic
            "snapped": bool(snapped),
            "vlm_error": vlm_error,
            "method": method,
        }

    def server_execute(self, goal_seg_id: int, grasp_poses: List[Any],
                       place_pose: Optional[List[float]] = None,
                       release: Optional[bool] = None,
                       env_idx: int = 0) -> Dict[str, Any]:
        """Execute ONE externally-supplied grasp-and-carry on `goal_seg_id`.

        grasp_poses: list of candidate grasps (each 4x4-flat or 7-vec, robot
        base frame). The sim IK-filters them and reaches the first feasible one
        exactly as a normal plan step would. `place_pose` is reserved (v0 uses
        the sim's free-space drop for removals); `release` defaults to "drop it"
        for an occluder and "keep holding" for the target.
        """
        assert self.num_envs == 1, "VLMServer assumes num_envs == 1"
        seg_offset = int(self._vlm_cfg().get("seg_offset", SEG_ID_OFFSET))
        goal_old = int(goal_seg_id) - seg_offset
        gt_old = int(getattr(self, "_server_gt_target_old", -1))

        grasps7 = np.stack([_to_grasp7(g) for g in grasp_poses], axis=0) \
            if len(grasp_poses) else np.empty((0, 7), dtype=np.float32)
        if grasps7.shape[0] == 0:
            return {"success": False, "grasped_seg_id": -1,
                    "target_grasped": False, "done": False,
                    "failure": "no_grasp_supplied"}

        goal_is_target = (goal_old == gt_old)
        if release is None:
            release = not goal_is_target

        # --- inject a single-step plan the unmodified executor can run -------
        # obs_obj_id/new_id throughout GORun is only ever `self._obs_ids[id] ->
        # old_id`, so a length-1 obs_ids + a one-step plan is sufficient. The
        # curobo collision world already holds every object, so motion planning
        # still avoids all obstacles.
        self._obs_grasp = grasps7.astype(np.float32)          # (K, 7) wxyz
        self._obs_ids = np.asarray([goal_old], dtype=np.int64)  # new_id 0 -> old
        self._active_plan = [(0, list(range(grasps7.shape[0])))]
        self._current_step_idx = 0

        log: Dict[str, Any] = {}
        ok = False
        try:
            ok = bool(self._execute_plan_step(
                0, log, release_after=bool(release),
                computing_time_ref=[0.0]))
        except Exception as exc:  # noqa: BLE001 - reported to master, loop survives
            log["exec_exception"] = "{}: {}".format(type(exc).__name__, exc)
            ok = False

        self._server_step = int(getattr(self, "_server_step", 0)) + 1

        # Per-task video: rewrite the single mp4 with everything executed so far
        # in this task (buffer was reset in server_reset, accumulates per step).
        vid_path = self._save_task_video(fps=10)
        if vid_path:
            print("[VLMServer] saved task video: {}".format(vid_path))

        target_grasped = bool(ok and goal_is_target and (not release))
        return {
            "success": bool(ok),
            "grasped_seg_id": int(goal_seg_id) if ok else -1,
            "target_grasped": target_grasped,
            "done": target_grasped,
            "release": bool(release),
            "goal_is_target": bool(goal_is_target),
            "n_grasps_tried": int(grasps7.shape[0]),
            "step_failure": log.get("step0_plan_failure")
            or log.get("step0_execute_failure") or log.get("exec_exception"),
            # Per-stage breakdown so a failure can be pinned to IK vs grasp-motion
            # vs carry-motion vs the execute-success (slip/no-move) check.
            "diag": _exec_diag(log),
        }


# --- self-register so tasks/__init__.py stays untouched ------------------- #
try:
    from isaacgymenvs.tasks import isaacgym_task_map

    isaacgym_task_map["FetchMeshCuroboGORunVLMServer"] = FetchMeshCuroboGORunVLMServer
except Exception as _exc:  # pragma: no cover
    print("[GORunVLMServer] task registration deferred: {}: {}"
          .format(type(_exc).__name__, _exc))
