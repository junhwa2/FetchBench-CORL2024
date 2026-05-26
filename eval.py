# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method fcl --pitch 0.025
# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method grn
# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method cbn
# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method cbn --robot_to_model_t -0.7 0.0 -0.1
# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method scn
# python eval.py --obstruction_root asset_release_v1.1/Obstruction_mini_1/benchmark_eval --method scn --robot_to_model_t -0.7 0.0 -0.1
import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import open3d as o3d
import torch
import trimesh
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelSpecificity,
)
from pykin.collision.collision_manager import CollisionManager
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm

GRN_PATH = "third_party/geometric_reasoning_networks"
sys.path.append(GRN_PATH)
from Network import GRN

SCN_PATH = "third_party/SceneCollisionNet/scenecollisionnet"
CBN_PATH = "third_party/cabinet/src"
sys.path.append(f"{SCN_PATH}")
sys.path.append(f"{CBN_PATH}")
try:
    from policy.robot import Robot
    from policy.collision_checker_custom import CabinetSceneCollisionChecker, NNSceneCollisionChecker
except ImportError:
    Robot = CabinetSceneCollisionChecker = NNSceneCollisionChecker = None

# Result type alias: (label, pred, time, category, t_idx)
# category: per-row str array; t_idx: per-row int array (parsed from filename obstruction_data_t<N>.h5)
ArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_T_RE = re.compile(r"_t(\d+)\.(?:h5|pt)$")


def _scene_category(h5_path: Path, root: Path) -> str:
    rel_parts = h5_path.resolve().relative_to(root).parts
    return rel_parts[0] if len(rel_parts) > 1 else "<root>"


def _t_index(h5_path: Path) -> int:
    m = _T_RE.search(h5_path.name)
    if m is None:
        raise ValueError(f"Cannot parse t-index from filename: {h5_path.name}")
    return int(m.group(1))


def _pred_out_path(log_dir: Path, root: Path, h5_path: Path) -> Path:
    """Mirror the input scene hierarchy under log_dir: e.g.
    <root>/<Category>/<Scene>/obstruction_data_t<N>.h5
    -> <log_dir>/<Category>/<Scene>/obstruction_data_t<N>_pred.h5"""
    rel = h5_path.relative_to(root)
    return log_dir / rel.parent / (h5_path.stem + "_pred.h5")


def _write_pred_h5(out_path: Path, **datasets) -> None:
    """Write a per-scene-task prediction h5. Pass any of:
    collision_label/pred/prob, feasibility_label/pred/prob. None values are
    skipped. label/pred are stored as int8, prob as float32 (matching the
    reference *_pred.h5 layout, minus valid_mask)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as lf:
        for k, v in datasets.items():
            if v is None:
                continue
            dtype = np.float32 if k.endswith("_prob") else np.int8
            lf.create_dataset(k, data=np.asarray(v).astype(dtype))


def _parse_t_ratio(spec: str, total: int) -> List[Tuple[str, int, int]]:
    """Parse a 3-digit ratio like '111' or '123' and split [0..total-1] into
    (low, mid, high) bins proportionally. Returns [(name, lo, hi)] inclusive."""
    if len(spec) != 3 or not spec.isdigit():
        raise ValueError(f"--t-ratio must be 3 digits (e.g., '111' or '123'), got: {spec!r}")
    ratios = [int(d) for d in spec]
    total_ratio = sum(ratios)
    if total_ratio == 0:
        raise ValueError("--t-ratio sum must be positive")
    boundaries = [0]
    acc = 0
    for r in ratios:
        acc += r
        boundaries.append(round(total * acc / total_ratio))
    boundaries[-1] = total
    names = ["low", "mid", "high"]
    return [(names[i], boundaries[i], boundaries[i + 1] - 1) for i in range(3)]


def build_pykin_world_from_ply(
    ply_path: Path,
    pitch: float,
) -> Tuple[CollisionManager, Dict[int, trimesh.Trimesh], Dict[int, np.ndarray]]:
    pcd = o3d.t.io.read_point_cloud(str(ply_path))
    if "positions" not in pcd.point:
        raise ValueError(f"positions not found in point cloud: {ply_path}")
    if "id" not in pcd.point:
        raise ValueError(f"id not found in point cloud: {ply_path}")

    xyz = pcd.point["positions"].numpy().astype(np.float32)
    rgb = pcd.point["colors"].numpy().astype(np.float64) if "colors" in pcd.point else np.zeros((xyz.shape[0], 3), dtype=np.float64)
    if rgb.size > 0 and rgb.max() <= 1.0:
        rgb = rgb * 255.0
    pid = pcd.point["id"].numpy().astype(np.int32).reshape(-1)

    world_collision = CollisionManager()
    mesh_map: Dict[int, trimesh.Trimesh] = {}
    color_map: Dict[int, np.ndarray] = {}
    h_mat = Transform().h_mat

    for obj_id in np.unique(pid):
        obj_id = int(obj_id)
        pts = xyz[pid == obj_id]
        if pts.shape[0] < 4:
            continue
        try:
            mesh = trimesh.voxel.ops.points_to_marching_cubes(trimesh.PointCloud(pts).vertices, pitch=pitch)
            world_collision.add_object(name=str(obj_id), gtype="mesh", gparam=mesh, h_mat=h_mat)
            mesh_map[obj_id] = mesh
            obj_rgb = rgb[pid == obj_id]
            if obj_rgb.shape[0] > 0:
                color_map[obj_id] = np.clip(np.mean(obj_rgb, axis=0), 0.0, 255.0)
        except Exception as exc:
            print(f"[WARN] Failed mesh conversion for id={obj_id} at {ply_path}: {exc}")

    return world_collision, mesh_map, color_map


def build_pykin_robot_from_pc_robot_ply(
    ply_path: Path,
    pitch: float,
) -> Tuple[CollisionManager, Optional[trimesh.Trimesh]]:
    """Build a pykin robot CollisionManager from a per-row pc_robot ply via marching
    cubes. The pc_robot ply already encodes the current qpos (it's the rendered
    robot surface), so no URDF/FK is needed. Returns (manager, mesh); mesh is
    None when the point cloud is too sparse or meshing fails."""
    pcd = o3d.t.io.read_point_cloud(str(ply_path))
    if "positions" not in pcd.point:
        raise ValueError(f"positions not found in point cloud: {ply_path}")
    xyz = pcd.point["positions"].numpy().astype(np.float32)

    manager = CollisionManager(is_robot=True)
    if xyz.shape[0] < 4:
        print(f"[WARN] pc_robot too sparse to mesh ({xyz.shape[0]} pts): {ply_path}")
        return manager, None
    try:
        mesh = trimesh.voxel.ops.points_to_marching_cubes(
            trimesh.PointCloud(xyz).vertices, pitch=pitch
        )
    except Exception as exc:
        print(f"[WARN] Failed pc_robot mesh conversion at {ply_path}: {exc}")
        return manager, None
    manager.add_object(name="robot", gtype="mesh", gparam=mesh, h_mat=Transform().h_mat)
    return manager, mesh


def pad_and_concat(
    all_labels: List[np.ndarray],
    all_preds: List[np.ndarray],
    all_times: List[np.ndarray],
    all_categories: List[np.ndarray],
    all_t_idx: List[np.ndarray],
) -> ArrayTriple:
    max_cols = max(x.shape[1] for x in all_labels)
    for i in range(len(all_labels)):
        if all_labels[i].shape[1] < max_cols:
            out = np.zeros((all_labels[i].shape[0], max_cols), dtype=np.bool_)
            out[:, :all_labels[i].shape[1]] = all_labels[i]
            all_labels[i] = out
        if all_preds[i].shape[1] < max_cols:
            out = np.zeros((all_preds[i].shape[0], max_cols), dtype=all_preds[i].dtype)
            out[:, :all_preds[i].shape[1]] = all_preds[i]
            all_preds[i] = out
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_times),
        np.concatenate(all_categories),
        np.concatenate(all_t_idx),
    )


def visualize_trimesh_pred_collision(
    pykin_robot: Optional[SingleArm],
    pykin_robot_collision: CollisionManager,
    pred_ids: np.ndarray,
    mesh_map: Dict[int, trimesh.Trimesh],
    world_color_map: Dict[int, np.ndarray],
    robot_color: np.ndarray,
    robot_mesh: Optional[trimesh.Trimesh] = None,
) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis())

    if pykin_robot is not None:
        # URDF mode: iterate link meshes placed by FK
        for link, info in pykin_robot.info[pykin_robot_collision.geom].items():
            gtype, mesh_data, h_mat = info[1], info[2], info[3]
            if gtype != "mesh":
                continue
            meshes = mesh_data if isinstance(mesh_data, list) else [mesh_data]
            for mesh in meshes:
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                m = mesh.copy()
                m.apply_transform(h_mat)
                if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                    m.visual = m.visual.to_color()
                rc = np.clip(robot_color, 0.0, 255.0).astype(np.uint8)
                m.visual.face_colors = [int(rc[0]), int(rc[1]), int(rc[2]), 220]
                scene.add_geometry(m)
    elif robot_mesh is not None:
        # pc mode: single marching-cubes mesh built from pc_robot ply
        m = robot_mesh.copy()
        if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
            m.visual = m.visual.to_color()
        rc = np.clip(robot_color, 0.0, 255.0).astype(np.uint8)
        m.visual.face_colors = [int(rc[0]), int(rc[1]), int(rc[2]), 220]
        scene.add_geometry(m)

    for obj_id in pred_ids.tolist():
        obj_id = int(obj_id)
        if obj_id not in mesh_map:
            continue
        m = mesh_map[obj_id].copy()
        if isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
            m.visual = m.visual.to_color()
        wc = np.clip(world_color_map.get(obj_id, np.array([220.0, 60.0, 60.0])), 0.0, 255.0).astype(np.uint8)
        m.visual.face_colors = [int(wc[0]), int(wc[1]), int(wc[2]), 220]
        scene.add_geometry(m)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        scene_pickle = Path(tmp_file.name)
    with scene_pickle.open("wb") as f:
        pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)
    subprocess.run(
        [sys.executable, "-c",
         "import pickle,sys\nwith open(sys.argv[1],'rb') as f: scene=pickle.load(f)\nscene.show()",
         str(scene_pickle)],
        check=False,
    )
    try:
        scene_pickle.unlink(missing_ok=True)
    except OSError:
        pass


def build_fcl_arrays(
    obstruction_root: str,
    pitch: float,
    log_dir: Optional[Path],
    visualize: bool = False,
    robot_geom: str = "urdf",
) -> Tuple[ArrayTriple, ArrayTriple]:
    if robot_geom not in ("urdf", "pc"):
        raise ValueError(f"Unknown robot_geom: {robot_geom!r} (expected 'urdf' or 'pc')")
    root = Path(obstruction_root).resolve()
    h5_files = sorted(root.rglob("obstruction_data_t*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No obstruction_data_t*.h5 found under: {root}")

    # URDF-mode shared robot kinematics; in pc mode the robot CollisionManager is
    # built per row from pc_robot ply (cached below), so URDF setup is skipped.
    pykin_robot: Optional[SingleArm] = None
    pykin_robot_collision_urdf: Optional[CollisionManager] = None
    if robot_geom == "urdf":
        config_file = root.parent / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"config.yaml not found: {config_file}")
        with open(config_file) as f:
            rcm = yaml.safe_load(f)['obs_data_gen']['robot_collision_model']
        pykin_robot = SingleArm(
            f_name=f'InfiniGym/{rcm["robot_path"]}',
            offset=Transform(pos=rcm['offset']['pos'], rot=rcm['offset']['rot']),
            has_gripper=rcm['has_gripper'],
            gripper_name=rcm['gripper_name'],
        )
        pykin_robot.setup_link_name(base_name=rcm['base_name'], eef_name=rcm['eef_name'])
        pykin_robot_collision_urdf = CollisionManager(is_robot=True)
        pykin_robot_collision_urdf.setup_robot_collision(pykin_robot, geom=rcm['geom'])

    world_cache: Dict[str, Tuple[CollisionManager, Dict[int, trimesh.Trimesh], Dict[int, np.ndarray]]] = {}
    robot_cache: Dict[str, Tuple[CollisionManager, Optional[trimesh.Trimesh]]] = {}
    all_c_labels: List[np.ndarray] = []
    all_c_preds: List[np.ndarray] = []
    all_c_times: List[np.ndarray] = []
    all_c_cats: List[np.ndarray] = []
    all_c_tidx: List[np.ndarray] = []
    all_f_labels: List[np.ndarray] = []
    all_f_preds: List[np.ndarray] = []
    all_f_times: List[np.ndarray] = []
    all_f_cats: List[np.ndarray] = []
    all_f_tidx: List[np.ndarray] = []

    for h5_path in h5_files:
        print(f"[INFO] Processing {h5_path}")
        category = _scene_category(h5_path, root)
        t_idx = _t_index(h5_path)
        with h5py.File(h5_path, "r") as f:
            qpos = np.asarray(f["qpos"], dtype=np.float32)
            collision_label = np.asarray(f["collision"], dtype=np.bool_)
            pc_cam_list = [
                v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in np.asarray(f["pc_cam"])
            ]
            pc_robot_list = [
                v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in np.asarray(f["pc_robot"])
            ]

        collision_pred = np.zeros_like(collision_label, dtype=np.float32)
        collision_time = np.zeros(len(qpos), dtype=np.float32)
        scene_dir = h5_path.parent

        for i, (row_qpos, pc_cam_rel, pc_robot_rel) in enumerate(zip(qpos, pc_cam_list, pc_robot_list)):
            t0 = time()
            robot_mesh: Optional[trimesh.Trimesh] = None
            if robot_geom == "urdf":
                q = np.asarray(row_qpos, dtype=np.float32)
                if q.shape[0] > 7:
                    q = q[:7]
                pykin_robot.set_transform(q)
                pykin_robot.open_gripper()
                for link, info in pykin_robot.info[pykin_robot_collision_urdf.geom].items():
                    if link in pykin_robot_collision_urdf._objs:
                        pykin_robot_collision_urdf.set_transform(name=link, h_mat=info[3])
                pykin_robot_collision = pykin_robot_collision_urdf
            else:  # pc
                robot_key = str((scene_dir / pc_robot_rel).resolve())
                if robot_key not in robot_cache:
                    robot_cache[robot_key] = build_pykin_robot_from_pc_robot_ply(Path(robot_key), pitch)
                pykin_robot_collision, robot_mesh = robot_cache[robot_key]

            cache_key = str((scene_dir / pc_cam_rel).resolve())
            if cache_key not in world_cache:
                world_cache[cache_key] = build_pykin_world_from_ply(Path(cache_key), pitch)
            pykin_world_collision, world_mesh_map, world_color_map = world_cache[cache_key]

            if len(pykin_robot_collision._objs) == 0:
                # pc mode with empty/sparse robot ply: no collidable robot geometry
                collision_time[i] = time() - t0
                continue

            result, names = pykin_robot_collision.in_collision_other(pykin_world_collision, return_names=True)
            collision_time[i] = time() - t0

            if result:
                for _, world_name in names:
                    try:
                        obj_id = int(world_name)
                        if 0 <= obj_id < collision_pred.shape[1]:
                            collision_pred[i, obj_id] = 1.0
                    except ValueError:
                        continue

                if visualize:
                    pred_ids = np.where(collision_pred[i])[0]
                    label_ids = np.where(collision_label[i])[0]
                    if pred_ids.tolist() != label_ids.tolist():
                        print(f"\t[DEBUG] Mismatch at row {i}: label={label_ids.tolist()}, pred={pred_ids.tolist()}")
                        pcd_robot = o3d.t.io.read_point_cloud(str(scene_dir / pc_robot_rel))
                        if "colors" in pcd_robot.point:
                            robot_colors = pcd_robot.point["colors"].numpy().astype(np.float64)
                            if robot_colors.size > 0 and robot_colors.max() <= 1.0:
                                robot_colors = robot_colors * 255.0
                            robot_color = np.clip(np.mean(robot_colors, axis=0), 0.0, 255.0)
                        else:
                            robot_color = np.array([30.0, 160.0, 30.0])
                        visualize_trimesh_pred_collision(
                            pykin_robot, pykin_robot_collision, pred_ids,
                            world_mesh_map, world_color_map, robot_color,
                            robot_mesh=robot_mesh,
                        )

        feas_label = (~collision_label.any(axis=1)).reshape(-1, 1).astype(np.float32)
        feas_pred = (~collision_pred.astype(bool).any(axis=1)).reshape(-1, 1).astype(np.float32)
        feas_time = collision_time

        if log_dir is not None:
            # FCL is binary (no probability), so collision_prob == collision_pred.
            _write_pred_h5(
                _pred_out_path(log_dir, root, h5_path),
                collision_label=collision_label,
                collision_pred=collision_pred,
                collision_prob=collision_pred,
                feasibility_label=feas_label,
                feasibility_pred=feas_pred,
                feasibility_prob=feas_pred,
            )
        cats_c = np.full(collision_label.shape[0], category, dtype=object)
        cats_f = np.full(feas_label.shape[0], category, dtype=object)
        tidx_c = np.full(collision_label.shape[0], t_idx, dtype=np.int32)
        tidx_f = np.full(feas_label.shape[0], t_idx, dtype=np.int32)
        all_c_labels.append(collision_label)
        all_c_preds.append(collision_pred)
        all_c_times.append(collision_time)
        all_c_cats.append(cats_c)
        all_c_tidx.append(tidx_c)
        all_f_labels.append(feas_label)
        all_f_preds.append(feas_pred)
        all_f_times.append(feas_time)
        all_f_cats.append(cats_f)
        all_f_tidx.append(tidx_f)

    c_result = pad_and_concat(all_c_labels, all_c_preds, all_c_times, all_c_cats, all_c_tidx)
    f_result = pad_and_concat(all_f_labels, all_f_preds, all_f_times, all_f_cats, all_f_tidx)
    return c_result, f_result


def estimate_robot_to_model_t(
    h5_files: List[Path],
    bounds_wrt_model: np.ndarray,
    method_name: str,
    max_points: int = 2_000_000,
    z_bin: float = 0.01,
) -> Tuple[float, float, float]:
    """Estimate the robot->model translation (single global value) by aligning the
    dataset's ply point distribution (robot frame) to the network's trained box
    `bounds_wrt_model`. Geometry-only: x/y are centered on the robust p2-p98
    midpoint; z slides a box-height window to the position of maximum point
    coverage (the scene is taller than the box, so it cannot be centered).
    `pc_cam` points are assumed to be in the robot frame (pure-translation fit)."""
    # collect unique ply paths referenced by pc_cam
    ply_paths: set = set()
    for h5_path in h5_files:
        scene_dir = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            for v in np.asarray(f["pc_cam"]):
                rel = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                ply_paths.add(str((scene_dir / rel).resolve()))
    ply_list = sorted(ply_paths)
    if not ply_list:
        raise ValueError("No pc_cam ply paths found; cannot auto-fit robot_to_model_t.")

    # read & aggregate points, subsampling per file to stay under max_points
    per_file_cap = max(1, max_points // len(ply_list))
    rng = np.random.default_rng(0)
    chunks: List[np.ndarray] = []
    for p in ply_list:
        pts_p = o3d.t.io.read_point_cloud(p).point["positions"].numpy().astype(np.float32)
        if pts_p.shape[0] > per_file_cap:
            pts_p = pts_p[rng.choice(pts_p.shape[0], per_file_cap, replace=False)]
        chunks.append(pts_p)
    pts = np.concatenate(chunks, axis=0)

    bm = bounds_wrt_model

    # robust per-axis center for x, y (scene fits inside the box on x/y)
    p2 = np.percentile(pts, 2, axis=0)
    p98 = np.percentile(pts, 98, axis=0)
    center_xy = 0.5 * (p2 + p98)

    # z: the scene is taller than the box, so it cannot be centered. Slide a
    # box-height window over the sorted z values and pick the position that
    # captures the most points (max coverage); align that window to the box.
    z = pts[:, 2]
    w_z = float(bm[2, 1] - bm[2, 0])
    zs = np.sort(z)
    counts = np.searchsorted(zs, zs + w_z, side="right") - np.arange(zs.shape[0])
    i_best = int(counts.argmax())
    z_win_lo = float(zs[i_best])
    z_cov = float(counts[i_best]) / zs.shape[0]

    # z histogram mode, diagnostics only (typically the table surface)
    z_lo, z_hi = np.percentile(z, [1, 99])
    n_bins = max(1, int(round((z_hi - z_lo) / z_bin)))
    hist, edges = np.histogram(z, bins=n_bins, range=(z_lo, z_hi))
    b = int(hist.argmax())
    z_table = 0.5 * (edges[b] + edges[b + 1])

    t_x = float(0.5 * (bm[0, 0] + bm[0, 1]) - center_xy[0])
    t_y = float(0.5 * (bm[1, 0] + bm[1, 1]) - center_xy[1])
    t_z = float(bm[2, 0] - z_win_lo)
    t = (t_x, t_y, t_z)

    # diagnostic: fraction of points landing inside the trained box after t
    pm = pts + np.array(t, dtype=np.float32)
    inside = np.all(
        (pm >= bm[:, 0].astype(np.float32)) & (pm <= bm[:, 1].astype(np.float32)), axis=1,
    )
    frac = float(inside.mean())
    print(
        f"[INFO] auto-fit robot_to_model_t [{method_name}] from {len(ply_list)} plys, "
        f"{pts.shape[0]} points\n"
        f"       scene robot-frame p2-p98: "
        f"x[{p2[0]:.3f},{p98[0]:.3f}] y[{p2[1]:.3f},{p98[1]:.3f}] z[{z_lo:.3f},{z_hi:.3f}]\n"
        f"       z histogram mode (table?)={z_table:.3f}; "
        f"z coverage window=[{z_win_lo:.3f},{z_win_lo + w_z:.3f}] covers {z_cov*100:.1f}% of points\n"
        f"       estimated t=({t_x:.4f}, {t_y:.4f}, {t_z:.4f}); "
        f"{frac*100:.1f}% of points inside bounds_wrt_model after t"
    )
    if frac < 0.5:
        print(
            "[WARN] <50% of points fit the trained box after translation — pure "
            "translation may be insufficient (rotation/scale mismatch?)."
        )
    return t


def build_cbn_arrays(
    obstruction_root: str,
    log_dir: Optional[Path],
    robot_to_model_t: Tuple[float, float, float] = (-0.7, 0.0, -0.2),
    auto_fit: bool = False,
) -> Tuple[ArrayTriple, Tuple[float, float, float]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot = Robot(
        "third_party/SceneCollisionNet/data/panda/panda.urdf",
        "right_gripper",
        device=device,
    )
    checker = CabinetSceneCollisionChecker(
        "third_party/cabinet/checkpoints/cabinet_collision",
        robot,
        device=device,
    )

    bounds_wrt_model = np.array([[-0.5, 0.5], [-0.8, 0.8], [-0.06, 0.44]])

    root = Path(obstruction_root).resolve()
    h5_files = sorted(root.rglob("obstruction_data_t*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No obstruction_data_t*.h5 found under: {root}")

    if auto_fit:
        robot_to_model_t = estimate_robot_to_model_t(h5_files, bounds_wrt_model, "CabiNet")

    robot_to_model = np.eye(4, dtype=np.float64)
    robot_to_model[:3, 3] = robot_to_model_t

    bounds_wrt_robot = bounds_wrt_model.copy()
    bounds_wrt_robot[:3, 0] -= robot_to_model[:3, 3]
    bounds_wrt_robot[:3, 1] -= robot_to_model[:3, 3]
    print(
        f"[INFO] CabiNet bounds\n"
        f"       model : x{bounds_wrt_model[0].tolist()}  y{bounds_wrt_model[1].tolist()}  z{bounds_wrt_model[2].tolist()}\n"
        f"       robot : x{bounds_wrt_robot[0].tolist()}  y{bounds_wrt_robot[1].tolist()}  z{bounds_wrt_robot[2].tolist()}"
    )

    all_f_labels: List[np.ndarray] = []
    all_f_preds: List[np.ndarray] = []
    all_f_times: List[np.ndarray] = []
    all_f_cats: List[np.ndarray] = []
    all_f_tidx: List[np.ndarray] = []

    for h5_path in h5_files:
        print(f"[INFO] Processing {h5_path} with CabiNet model")
        category = _scene_category(h5_path, root)
        t_idx = _t_index(h5_path)
        with h5py.File(h5_path, "r") as f:
            qpos = np.asarray(f["qpos"], dtype=np.float32)
            collision_label = np.asarray(f["collision"], dtype=np.bool_)
            pc_cam_list = [
                v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in np.asarray(f["pc_cam"])
            ]

        scene_dir = h5_path.parent
        n = len(qpos)
        feas_pred = np.zeros((n, 1), dtype=np.float32)
        feas_time = np.zeros(n, dtype=np.float32)
        feas_label = (
            (~collision_label.any(axis=1, keepdims=True)).astype(np.float32)
            if collision_label.ndim > 1
            else (~collision_label).reshape(-1, 1).astype(np.float32)
        )

        scene_to_indices: Dict[str, List[int]] = {}
        for i, pc_cam_rel in enumerate(pc_cam_list):
            key = str((scene_dir / pc_cam_rel).resolve())
            scene_to_indices.setdefault(key, []).append(i)

        for pc_cam_path, indices in scene_to_indices.items():
            pcd = o3d.t.io.read_point_cloud(pc_cam_path)
            scene_pc = pcd.point["positions"].numpy().astype(np.float32)

            checker.set_scene({
                "scene_pc": scene_pc,
                "object_pc": np.array([], dtype=np.float32),
                "robot_to_model": robot_to_model,
                "model_to_robot": np.linalg.inv(robot_to_model),
            })

            qs_batch = qpos[indices]
            if qs_batch.shape[1] < robot.dof:
                pad = np.full((len(qs_batch), robot.dof - qs_batch.shape[1]), 0.04, dtype=np.float32)
                qs_batch = np.concatenate([qs_batch, pad], axis=1)
            qs_tensor = torch.from_numpy(qs_batch[:, :robot.dof]).to(device)

            t0 = time()
            with torch.no_grad():
                coll = checker(qs_tensor, threshold=0.45)  # (B,) bool: True = collision
            t_per_query = (time() - t0) / len(indices)

            coll_np = coll.cpu().numpy()
            for j, idx in enumerate(indices):
                feas_pred[idx, 0] = 1.0 - float(coll_np[j])  # 1.0 = no collision = feasible
                feas_time[idx] = t_per_query

        if log_dir is not None:
            # CBN/SCN produce only pose-level feasibility (binary): prob == pred.
            _write_pred_h5(
                _pred_out_path(log_dir, root, h5_path),
                feasibility_label=feas_label,
                feasibility_pred=feas_pred,
                feasibility_prob=feas_pred,
            )
        all_f_labels.append(feas_label)
        all_f_preds.append(feas_pred)
        all_f_times.append(feas_time)
        all_f_cats.append(np.full(feas_label.shape[0], category, dtype=object))
        all_f_tidx.append(np.full(feas_label.shape[0], t_idx, dtype=np.int32))

    return (
        pad_and_concat(all_f_labels, all_f_preds, all_f_times, all_f_cats, all_f_tidx),
        tuple(robot_to_model_t),
    )


def build_scn_arrays(
    obstruction_root: str,
    log_dir: Optional[Path],
    robot_to_model_t: Tuple[float, float, float] = (-0.7, 0.0, -0.2),
    auto_fit: bool = False,
) -> Tuple[ArrayTriple, Tuple[float, float, float]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot = Robot(
        f"third_party/SceneCollisionNet/data/panda/panda.urdf",
        "right_gripper",
        device=device,
    )
    checker = NNSceneCollisionChecker(
        f"third_party/SceneCollisionNet/weights/scene_coll_nn",
        robot,
        device=device,
    )

    bounds_wrt_model = np.array([[-0.5, 0.5], [-0.8, 0.8], [0.24, 0.60]])

    root = Path(obstruction_root).resolve()
    h5_files = sorted(root.rglob("obstruction_data_t*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No obstruction_data_t*.h5 found under: {root}")

    if auto_fit:
        robot_to_model_t = estimate_robot_to_model_t(h5_files, bounds_wrt_model, "SceneCollisionNet")

    robot_to_model = np.eye(4, dtype=np.float64)
    robot_to_model[:3, 3] = robot_to_model_t

    bounds_wrt_robot = bounds_wrt_model.copy()
    bounds_wrt_robot[:3, 0] -= robot_to_model[:3, 3]
    bounds_wrt_robot[:3, 1] -= robot_to_model[:3, 3]
    print(
        f"[INFO] SceneCollisionNet bounds\n"
        f"       model : x{bounds_wrt_model[0].tolist()}  y{bounds_wrt_model[1].tolist()}  z{bounds_wrt_model[2].tolist()}\n"
        f"       robot : x{bounds_wrt_robot[0].tolist()}  y{bounds_wrt_robot[1].tolist()}  z{bounds_wrt_robot[2].tolist()}"
    )

    all_f_labels: List[np.ndarray] = []
    all_f_preds: List[np.ndarray] = []
    all_f_times: List[np.ndarray] = []
    all_f_cats: List[np.ndarray] = []
    all_f_tidx: List[np.ndarray] = []

    for h5_path in h5_files:
        print(f"[INFO] Processing {h5_path} with SceneCollisionNet model")
        category = _scene_category(h5_path, root)
        t_idx = _t_index(h5_path)
        with h5py.File(h5_path, "r") as f:
            qpos = np.asarray(f["qpos"], dtype=np.float32)
            collision_label = np.asarray(f["collision"], dtype=np.bool_)
            pc_cam_list = [
                v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                for v in np.asarray(f["pc_cam"])
            ]

        scene_dir = h5_path.parent
        n = len(qpos)
        feas_pred = np.zeros((n, 1), dtype=np.float32)
        feas_time = np.zeros(n, dtype=np.float32)
        feas_label = (
            (~collision_label.any(axis=1, keepdims=True)).astype(np.float32)
            if collision_label.ndim > 1
            else (~collision_label).reshape(-1, 1).astype(np.float32)
        )

        scene_to_indices: Dict[str, List[int]] = {}
        for i, pc_cam_rel in enumerate(pc_cam_list):
            key = str((scene_dir / pc_cam_rel).resolve())
            scene_to_indices.setdefault(key, []).append(i)

        for pc_cam_path, indices in scene_to_indices.items():
            pcd = o3d.t.io.read_point_cloud(pc_cam_path)
            scene_pc = pcd.point["positions"].numpy().astype(np.float32)

            checker.set_scene({
                "scene_pc": scene_pc,
                "object_pc": np.array([], dtype=np.float32),
                "robot_to_model": robot_to_model,
                "model_to_robot": np.linalg.inv(robot_to_model),
            })

            qs_batch = qpos[indices]
            if qs_batch.shape[1] < robot.dof:
                pad = np.full((len(qs_batch), robot.dof - qs_batch.shape[1]), 0.04, dtype=np.float32)
                qs_batch = np.concatenate([qs_batch, pad], axis=1)
            qs_tensor = torch.from_numpy(qs_batch[:, :robot.dof]).to(device)

            t0 = time()
            with torch.no_grad():
                coll = checker(qs_tensor, threshold=0.45)  # (B,) bool: True = collision
            t_per_query = (time() - t0) / len(indices)

            coll_np = coll.cpu().numpy()
            for j, idx in enumerate(indices):
                feas_pred[idx, 0] = 1.0 - float(coll_np[j])  # 1.0 = no collision = feasible
                feas_time[idx] = t_per_query

        if log_dir is not None:
            # CBN/SCN produce only pose-level feasibility (binary): prob == pred.
            _write_pred_h5(
                _pred_out_path(log_dir, root, h5_path),
                feasibility_label=feas_label,
                feasibility_pred=feas_pred,
                feasibility_prob=feas_pred,
            )
        all_f_labels.append(feas_label)
        all_f_preds.append(feas_pred)
        all_f_times.append(feas_time)
        all_f_cats.append(np.full(feas_label.shape[0], category, dtype=object))
        all_f_tidx.append(np.full(feas_label.shape[0], t_idx, dtype=np.int32))

    return (
        pad_and_concat(all_f_labels, all_f_preds, all_f_times, all_f_cats, all_f_tidx),
        tuple(robot_to_model_t),
    )


def build_grn_arrays(
    obstruction_root: str,
    log_dir: Optional[Path],
) -> Tuple[ArrayTriple, ArrayTriple]:
    root = Path(obstruction_root).resolve()
    pt_files = sorted(root.rglob("grouped_data_t*.pt"))
    h5_files = sorted(root.rglob("grouped_data_t*.h5"))
    if not pt_files:
        raise FileNotFoundError(f"No grouped_data_t*.pt found under: {root}")
    if not h5_files:
        raise FileNotFoundError(f"No grouped_data_t*.h5 found under: {root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRN(device=device).to(device)
    model.load_state_dict(torch.load(f"{GRN_PATH}/GRN_models/GRN_panda.pt", map_location=device))
    model.eval()

    all_c_labels: List[np.ndarray] = []
    all_c_preds: List[np.ndarray] = []
    all_c_times: List[np.ndarray] = []
    all_c_cats: List[np.ndarray] = []
    all_c_tidx: List[np.ndarray] = []
    all_f_labels: List[np.ndarray] = []
    all_f_preds: List[np.ndarray] = []
    all_f_times: List[np.ndarray] = []
    all_f_cats: List[np.ndarray] = []
    all_f_tidx: List[np.ndarray] = []

    for pt_path, h5_path in zip(pt_files, h5_files):
        print(f"[INFO] Processing {pt_path} with GRN model")
        category = _scene_category(h5_path, root)
        t_idx = _t_index(h5_path)

        with h5py.File(h5_path, "r") as f:
            grn_collision = np.asarray(f["grn_collision"], dtype=np.bool_)
            grn_target = np.asarray(f["grn_target"], dtype=np.int32).reshape(-1)
            grn_gtype = np.asarray(f["grn_gtype"], dtype=np.int32).reshape(-1)

        scene_data, _ = torch.load(pt_path, weights_only=False)
        t0 = time()
        with torch.no_grad():
            batch = scene_data.to(device)
            f_preds, _, go_preds = model(batch, "test")
        inference_time = time() - t0

        masks = batch.mask.cpu().numpy()
        edge_index = batch.edge_index.cpu().numpy()
        blocking_masks = batch.blocking_mask.cpu().numpy()
        go_preds_np = go_preds.detach().cpu().numpy()   # (num_edges, 5)
        f_preds_np = f_preds.detach().cpu().numpy()     # (num_objects, 6)

        num_movables = int(masks.sum())
        num_label_cols = num_movables + 1
        num_grasp_poses = grn_collision.shape[0]

        preds = np.zeros((num_grasp_poses, num_label_cols), dtype=np.float32)
        for i in range(num_grasp_poses):
            tgt = int(grn_target[i])
            gtype = int(grn_gtype[i])
            if gtype >= go_preds_np.shape[1]:
                continue
            edge_mask = (edge_index[1] == tgt) & blocking_masks
            for src, pred_val in zip(edge_index[0, edge_mask], go_preds_np[edge_mask, gtype]):
                if src < num_movables:
                    preds[i, src] = pred_val
                else:
                    preds[i, num_movables] = max(preds[i, num_movables], pred_val)

        feas_preds = np.zeros((num_grasp_poses, 1), dtype=np.float32)
        for i in range(num_grasp_poses):
            tgt = int(grn_target[i])
            gtype = int(grn_gtype[i])
            if gtype < f_preds_np.shape[1]:
                feas_preds[i, 0] = f_preds_np[tgt, gtype]
        feas_label = (~grn_collision.any(axis=1)).reshape(-1, 1).astype(np.float32)

        file_times = np.full(num_grasp_poses, inference_time, dtype=np.float32)

        if log_dir is not None:
            # GRN outputs per-cell collision probability and per-pose feasibility
            # probability; pred = (prob >= 0.5).
            _write_pred_h5(
                _pred_out_path(log_dir, root, h5_path),
                collision_label=grn_collision[:, :num_label_cols],
                collision_pred=(preds >= 0.5),
                collision_prob=preds,
                feasibility_label=feas_label,
                feasibility_pred=(feas_preds >= 0.5),
                feasibility_prob=feas_preds,
            )
        cats = np.full(num_grasp_poses, category, dtype=object)
        tidx_arr = np.full(num_grasp_poses, t_idx, dtype=np.int32)
        all_c_labels.append(grn_collision[:, :num_label_cols])
        all_c_preds.append(preds)
        all_c_times.append(file_times)
        all_c_cats.append(cats)
        all_c_tidx.append(tidx_arr)
        all_f_labels.append(feas_label)
        all_f_preds.append(feas_preds)
        all_f_times.append(file_times)
        all_f_cats.append(cats)
        all_f_tidx.append(tidx_arr)

    c_result = pad_and_concat(all_c_labels, all_c_preds, all_c_times, all_c_cats, all_c_tidx)
    f_result = pad_and_concat(all_f_labels, all_f_preds, all_f_times, all_f_cats, all_f_tidx)
    return c_result, f_result


_METRIC_KEYS: List[str] = [
    "accuracy", "precision", "recall", "f1", "ap",
    "auroc", "specificity", "balanced_acc",
    "prevalence", "n_positive", "n_total",
]


def _fmt_metrics_msg(m: Dict[str, float]) -> str:
    parts = []
    for k in _METRIC_KEYS:
        v = m[k]
        parts.append(f"{k}={v}" if isinstance(v, int) else f"{k}={v:.6f}")
    return ", ".join(parts)


def _metric_row_values(m: Dict[str, float]) -> List:
    return [m[k] if isinstance(m[k], int) else f"{m[k]:.6f}" for k in _METRIC_KEYS]


def _collision_metrics_subset(c_label: np.ndarray, c_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    num_labels = c_label.shape[1]
    target = torch.from_numpy(c_label.astype(np.int64))
    pred_float = torch.from_numpy(c_pred.astype(np.float32))
    kwargs = {"num_labels": num_labels, "average": "micro"}
    kwargs_t = {**kwargs, "threshold": threshold}
    metrics = {
        # prevalence-sensitive
        "accuracy":    float(MultilabelAccuracy(**kwargs_t)(pred_float, target).item()),
        "precision":   float(MultilabelPrecision(**kwargs_t)(pred_float, target).item()),
        "recall":      float(MultilabelRecall(**kwargs_t)(pred_float, target).item()),
        "f1":          float(MultilabelF1Score(**kwargs_t)(pred_float, target).item()),
        "ap":          float(MultilabelAveragePrecision(**kwargs)(pred_float, target).item()),
        # prevalence-robust
        "auroc":       float(MultilabelAUROC(**kwargs)(pred_float, target).item()),
        "specificity": float(MultilabelSpecificity(**kwargs_t)(pred_float, target).item()),
        # prevalence stats
        "prevalence":  float(target.float().mean().item()),
        "n_positive":  int(target.sum().item()),
        "n_total":     int(target.numel()),
    }
    metrics["balanced_acc"] = 0.5 * (metrics["recall"] + metrics["specificity"])
    return metrics


_MATCHED_KEYS: List[str] = [
    "accuracy", "precision", "recall", "f1",
    "specificity", "balanced_acc",
]


def _binary_metrics_on_flat(
    flat_label: np.ndarray, flat_pred: np.ndarray, threshold: float
) -> Dict[str, float]:
    target = torch.from_numpy(flat_label.astype(np.int64))
    pred_float = torch.from_numpy(flat_pred.astype(np.float32))
    out = {
        "accuracy":    float(BinaryAccuracy(threshold=threshold)(pred_float, target).item()),
        "precision":   float(BinaryPrecision(threshold=threshold)(pred_float, target).item()),
        "recall":      float(BinaryRecall(threshold=threshold)(pred_float, target).item()),
        "f1":          float(BinaryF1Score(threshold=threshold)(pred_float, target).item()),
        "specificity": float(BinarySpecificity(threshold=threshold)(pred_float, target).item()),
    }
    out["balanced_acc"] = 0.5 * (out["recall"] + out["specificity"])
    return out


def _collision_metrics_matched(
    c_label: np.ndarray,
    c_pred: np.ndarray,
    threshold: float,
    target_prevalence: float,
    n_seeds: int = 10,
) -> Dict[str, float]:
    """Cell-level flatten then resample a single class so the flat prevalence matches
    target_prevalence. If target_prevalence is at or above the natural prevalence,
    negatives are downsampled (raises prevalence); if below, positives are
    downsampled (lowers prevalence). Repeat with n_seeds RNG seeds and return, for
    each metric in _MATCHED_KEYS, the across-seed mean ('<metric>_matched') and std
    ('<metric>_matched_std'), plus the post-resample sample counts
    ('matched_n_positive', 'matched_n_total').
    Raises ValueError if a class is empty or too small to reach target_prevalence."""
    flat_label = c_label.flatten().astype(np.int64)
    flat_pred = c_pred.flatten().astype(np.float32)
    pos_idx = np.where(flat_label == 1)[0]
    neg_idx = np.where(flat_label == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No positive samples in this subset; cannot match prevalence.")
    if n_neg == 0:
        raise ValueError("No negative samples in this subset; cannot match prevalence.")
    if not (0.0 < target_prevalence < 1.0):
        raise ValueError(f"target_prevalence must be in (0, 1), got {target_prevalence}")

    natural_prevalence = n_pos / (n_pos + n_neg)
    if target_prevalence >= natural_prevalence:
        # downsample negatives to raise prevalence toward the target
        n_pos_keep = n_pos
        n_neg_keep = int(round(n_pos * (1.0 - target_prevalence) / target_prevalence))
        if n_neg_keep > n_neg:
            raise ValueError(
                f"Not enough negatives to reach target_prevalence={target_prevalence:.6f}: "
                f"need {n_neg_keep}, have {n_neg}."
            )
        resample_neg = True
    else:
        # downsample positives to lower prevalence toward the target
        n_neg_keep = n_neg
        n_pos_keep = int(round(n_neg * target_prevalence / (1.0 - target_prevalence)))
        if n_pos_keep > n_pos:
            raise ValueError(
                f"Not enough positives to reach target_prevalence={target_prevalence:.6f}: "
                f"need {n_pos_keep}, have {n_pos}."
            )
        resample_neg = False

    per_seed: List[Dict[str, float]] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if resample_neg:
            sampled_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
            keep_idx = np.concatenate([pos_idx, sampled_neg])
        else:
            sampled_pos = rng.choice(pos_idx, size=n_pos_keep, replace=False)
            keep_idx = np.concatenate([sampled_pos, neg_idx])
        per_seed.append(_binary_metrics_on_flat(
            flat_label[keep_idx], flat_pred[keep_idx], threshold
        ))

    out: Dict[str, float] = {}
    for k in _MATCHED_KEYS:
        vals = [m[k] for m in per_seed]
        out[f"{k}_matched"] = float(np.mean(vals))
        out[f"{k}_matched_std"] = float(np.std(vals))
    out["matched_n_positive"] = int(n_pos_keep)
    out["matched_n_total"] = int(n_pos_keep + n_neg_keep)
    return out


def _feasibility_metrics_matched(
    f_label: np.ndarray,
    f_pred: np.ndarray,
    threshold: float,
    target_prevalence: float,
    n_seeds: int = 10,
) -> Dict[str, float]:
    """Same flatten-and-downsample procedure as `_collision_metrics_matched`,
    but spelled with the feasibility name for clarity at call sites."""
    return _collision_metrics_matched(f_label, f_pred, threshold, target_prevalence, n_seeds)


def _matched_csv_header_cols() -> List[str]:
    return (
        ["target_prevalence"]
        + [f"{k}_matched" for k in _MATCHED_KEYS]
        + [f"{k}_matched_std" for k in _MATCHED_KEYS]
        + ["matched_n_positive", "matched_n_total"]
    )


def _matched_csv_row_values(target_p: Optional[float], matched: Optional[Dict[str, float]]) -> List[str]:
    n_cols = 1 + 2 * len(_MATCHED_KEYS) + 2
    if matched is None or target_p is None:
        return [""] * n_cols
    return (
        [f"{target_p:.6f}"]
        + [f"{matched[f'{k}_matched']:.6f}" for k in _MATCHED_KEYS]
        + [f"{matched[f'{k}_matched_std']:.6f}" for k in _MATCHED_KEYS]
        + [str(matched["matched_n_positive"]), str(matched["matched_n_total"])]
    )


def _across_axis_f1_matched_stats(
    items: List[Dict],
) -> Tuple[Optional[float], Optional[float]]:
    """Mean/std of `f1_matched` across per-category or per-t-bin entries.
    Entries whose matched dict is None (e.g., a subgroup with no positives or
    too few negatives to reach the target prevalence) are skipped. Returns
    (None, None) when no usable entry remains."""
    vals = [
        b["matched"]["f1_matched"]
        for b in items
        if b.get("matched") is not None
    ]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _fmt_optional(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


_F1_SPREAD_COLS: List[str] = [
    "f1_matched_across_cats_mean",
    "f1_matched_across_cats_std",
    "f1_matched_across_tbins_mean",
    "f1_matched_across_tbins_std",
]


def _fmt_matched_msg(matched: Dict[str, float]) -> str:
    parts = [
        f"{k}={matched[f'{k}_matched']:.4f}±{matched[f'{k}_matched_std']:.4f}"
        for k in _MATCHED_KEYS
    ]
    parts.append(f"matched_n_positive={matched['matched_n_positive']}")
    parts.append(f"matched_n_total={matched['matched_n_total']}")
    return ", ".join(parts)


def compute_collision_metrics(
    c_result: ArrayTriple,
    run_id: str,
    obstruction_root: str,
    method_str: str,
    no_save: bool,
    t_bins: List[Tuple[str, int, int]],
    threshold: float,
    log_dir: Optional[Path] = None,
) -> None:
    c_label, c_pred, c_time, c_cat, c_tidx = c_result

    metrics = _collision_metrics_subset(c_label, c_pred, threshold)
    print(
        f"[collision][overall] {_fmt_metrics_msg(metrics)}, "
        f"collision_time={c_time.mean()*1000:.6f}ms"
    )

    # Per-category natural metrics (also keep label/pred slices for matched eval)
    per_cat: List[Dict] = []
    for cat in sorted(set(c_cat.tolist())):
        mask = c_cat == cat
        sub_label = c_label[mask]
        sub_pred = c_pred[mask]
        m = _collision_metrics_subset(sub_label, sub_pred, threshold)
        t_mean = float(c_time[mask].mean())
        per_cat.append({
            "name": cat,
            "natural": m,
            "t_mean": t_mean,
            "n": int(mask.sum()),
            "label": sub_label,
            "pred": sub_pred,
        })
        print(
            f"[collision][{cat}] n={int(mask.sum())} {_fmt_metrics_msg(m)}, "
            f"collision_time={t_mean*1000:.6f}ms"
        )

    # Matched metrics across categories (target = overall pooled prevalence)
    cat_target_p: Optional[float] = metrics["prevalence"] if per_cat else None
    for b in per_cat:
        b["target_prevalence"] = cat_target_p
        b["matched"] = None
        if cat_target_p is None:
            continue
        try:
            b["matched"] = _collision_metrics_matched(
                b["label"], b["pred"], threshold, cat_target_p, n_seeds=10,
            )
            print(
                f"[collision][{b['name']}] matched (target_p={cat_target_p:.6f}): "
                f"{_fmt_matched_msg(b['matched'])}"
            )
        except ValueError as e:
            print(f"[WARN] matched skipped for category={b['name']}: {e}")

    # Per-tbin natural metrics
    per_tbin: List[Dict] = []
    for name, lo, hi in t_bins:
        mask = (c_tidx >= lo) & (c_tidx <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"[collision][t:{name}({lo}-{hi})] n=0 (no rows)")
            continue
        sub_label = c_label[mask]
        sub_pred = c_pred[mask]
        m = _collision_metrics_subset(sub_label, sub_pred, threshold)
        t_mean = float(c_time[mask].mean())
        per_tbin.append({
            "name": name,
            "lo": lo,
            "hi": hi,
            "natural": m,
            "t_mean": t_mean,
            "n": n,
            "label": sub_label,
            "pred": sub_pred,
        })
        print(
            f"[collision][t:{name}({lo}-{hi})] n={n} {_fmt_metrics_msg(m)}, "
            f"collision_time={t_mean*1000:.6f}ms"
        )

    # Matched metrics across t-bins (target = overall pooled prevalence)
    tbin_target_p: Optional[float] = metrics["prevalence"] if per_tbin else None
    for b in per_tbin:
        b["target_prevalence"] = tbin_target_p
        b["matched"] = None
        if tbin_target_p is None:
            continue
        try:
            b["matched"] = _collision_metrics_matched(
                b["label"], b["pred"], threshold, tbin_target_p, n_seeds=10,
            )
            print(
                f"[collision][t:{b['name']}({b['lo']}-{b['hi']})] matched "
                f"(target_p={tbin_target_p:.6f}): {_fmt_matched_msg(b['matched'])}"
            )
        except ValueError as e:
            print(f"[WARN] matched skipped for t_bin={b['name']}: {e}")

    if no_save:
        return

    # Overall CSV: single pooled row. Per-cat / per-tbin matched F1 spread
    # (mean, std) is appended so cross-axis variability is visible alongside
    # the pooled metrics. Spread cells are blank if matching failed everywhere.
    cat_f1_mean, cat_f1_std = _across_axis_f1_matched_stats(per_cat)
    tbin_f1_mean, tbin_f1_std = _across_axis_f1_matched_stats(per_tbin)
    csv_path = Path("logs") / "collision_metrics.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method",
                             *_METRIC_KEYS, "avg_collision_time",
                             *_F1_SPREAD_COLS])
        writer.writerow([
            run_id, obstruction_root, method_str,
            *_metric_row_values(metrics), f"{c_time.mean():.6f}",
            _fmt_optional(cat_f1_mean), _fmt_optional(cat_f1_std),
            _fmt_optional(tbin_f1_mean), _fmt_optional(tbin_f1_std),
        ])
    print(f"[INFO] Saved collision metrics CSV to {csv_path}")

    per_xx_dir = log_dir if log_dir is not None else Path("logs")

    # Per-category CSV (existing columns kept; matched columns appended at end)
    per_cat_csv = per_xx_dir / "collision_per_category_metrics.csv"
    write_header = not per_cat_csv.exists() or per_cat_csv.stat().st_size == 0
    with per_cat_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method", "category", "n",
                             *_METRIC_KEYS, "avg_collision_time",
                             *_matched_csv_header_cols()])
        for b in per_cat:
            writer.writerow([
                run_id, obstruction_root, method_str, b["name"], b["n"],
                *_metric_row_values(b["natural"]), f"{b['t_mean']:.6f}",
                *_matched_csv_row_values(b["target_prevalence"], b["matched"]),
            ])
        # 'total' row: natural metrics on the full pooled data (matched cols blank)
        writer.writerow([
            run_id, obstruction_root, method_str, "total", c_label.shape[0],
            *_metric_row_values(metrics), f"{c_time.mean():.6f}",
            *_matched_csv_row_values(None, None),
        ])
    print(f"[INFO] Saved per-category collision metrics CSV to {per_cat_csv}")

    # Per-t-bin CSV (existing columns kept; matched columns appended at end)
    per_tbin_csv = per_xx_dir / "collision_per_t_bin_metrics.csv"
    write_header = not per_tbin_csv.exists() or per_tbin_csv.stat().st_size == 0
    with per_tbin_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method", "t_bin", "t_lo", "t_hi", "n",
                             *_METRIC_KEYS, "avg_collision_time",
                             *_matched_csv_header_cols()])
        for b in per_tbin:
            writer.writerow([
                run_id, obstruction_root, method_str, b["name"], b["lo"], b["hi"], b["n"],
                *_metric_row_values(b["natural"]), f"{b['t_mean']:.6f}",
                *_matched_csv_row_values(b["target_prevalence"], b["matched"]),
            ])
        # 'total' row: natural metrics on the full pooled data, t range = all bins
        t_lo_all = t_bins[0][1] if t_bins else 0
        t_hi_all = t_bins[-1][2] if t_bins else 0
        writer.writerow([
            run_id, obstruction_root, method_str, "total", t_lo_all, t_hi_all, c_label.shape[0],
            *_metric_row_values(metrics), f"{c_time.mean():.6f}",
            *_matched_csv_row_values(None, None),
        ])
    print(f"[INFO] Saved per-t-bin collision metrics CSV to {per_tbin_csv}")


def _feasibility_metrics_subset(f_label: np.ndarray, f_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    target = torch.from_numpy(f_label.reshape(-1).astype(np.int64))
    pred_float = torch.from_numpy(f_pred.reshape(-1).astype(np.float32))
    metrics = {
        # prevalence-sensitive
        "accuracy":    float(BinaryAccuracy(threshold=threshold)(pred_float, target).item()),
        "precision":   float(BinaryPrecision(threshold=threshold)(pred_float, target).item()),
        "recall":      float(BinaryRecall(threshold=threshold)(pred_float, target).item()),
        "f1":          float(BinaryF1Score(threshold=threshold)(pred_float, target).item()),
        "ap":          float(BinaryAveragePrecision()(pred_float, target).item()),
        # prevalence-robust
        "auroc":       float(BinaryAUROC()(pred_float, target).item()),
        "specificity": float(BinarySpecificity(threshold=threshold)(pred_float, target).item()),
        # prevalence stats
        "prevalence":  float(target.float().mean().item()),
        "n_positive":  int(target.sum().item()),
        "n_total":     int(target.numel()),
    }
    metrics["balanced_acc"] = 0.5 * (metrics["recall"] + metrics["specificity"])
    return metrics


def compute_feasibility_metrics(
    f_result: ArrayTriple,
    run_id: str,
    obstruction_root: str,
    method_str: str,
    no_save: bool,
    t_bins: List[Tuple[str, int, int]],
    threshold: float,
    log_dir: Optional[Path] = None,
) -> None:
    f_label, f_pred, f_time, f_cat, f_tidx = f_result

    metrics = _feasibility_metrics_subset(f_label, f_pred, threshold)
    print(
        f"[feasibility][overall] {_fmt_metrics_msg(metrics)}, "
        f"feasibility_time={f_time.mean()*1000:.6f}ms"
    )

    # Per-category natural metrics (also keep label/pred slices for matched eval)
    per_cat: List[Dict] = []
    for cat in sorted(set(f_cat.tolist())):
        mask = f_cat == cat
        sub_label = f_label[mask]
        sub_pred = f_pred[mask]
        m = _feasibility_metrics_subset(sub_label, sub_pred, threshold)
        t_mean = float(f_time[mask].mean())
        per_cat.append({
            "name": cat,
            "natural": m,
            "t_mean": t_mean,
            "n": int(mask.sum()),
            "label": sub_label,
            "pred": sub_pred,
        })
        print(
            f"[feasibility][{cat}] n={int(mask.sum())} {_fmt_metrics_msg(m)}, "
            f"feasibility_time={t_mean*1000:.6f}ms"
        )

    # Matched metrics across categories (target = overall pooled prevalence)
    cat_target_p: Optional[float] = metrics["prevalence"] if per_cat else None
    for b in per_cat:
        b["target_prevalence"] = cat_target_p
        b["matched"] = None
        if cat_target_p is None:
            continue
        try:
            b["matched"] = _feasibility_metrics_matched(
                b["label"], b["pred"], threshold, cat_target_p, n_seeds=10,
            )
            print(
                f"[feasibility][{b['name']}] matched (target_p={cat_target_p:.6f}): "
                f"{_fmt_matched_msg(b['matched'])}"
            )
        except ValueError as e:
            print(f"[WARN] matched skipped for category={b['name']}: {e}")

    # Per-tbin natural metrics
    per_tbin: List[Dict] = []
    for name, lo, hi in t_bins:
        mask = (f_tidx >= lo) & (f_tidx <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f"[feasibility][t:{name}({lo}-{hi})] n=0 (no rows)")
            continue
        sub_label = f_label[mask]
        sub_pred = f_pred[mask]
        m = _feasibility_metrics_subset(sub_label, sub_pred, threshold)
        t_mean = float(f_time[mask].mean())
        per_tbin.append({
            "name": name,
            "lo": lo,
            "hi": hi,
            "natural": m,
            "t_mean": t_mean,
            "n": n,
            "label": sub_label,
            "pred": sub_pred,
        })
        print(
            f"[feasibility][t:{name}({lo}-{hi})] n={n} {_fmt_metrics_msg(m)}, "
            f"feasibility_time={t_mean*1000:.6f}ms"
        )

    # Matched metrics across t-bins (target = overall pooled prevalence)
    tbin_target_p: Optional[float] = metrics["prevalence"] if per_tbin else None
    for b in per_tbin:
        b["target_prevalence"] = tbin_target_p
        b["matched"] = None
        if tbin_target_p is None:
            continue
        try:
            b["matched"] = _feasibility_metrics_matched(
                b["label"], b["pred"], threshold, tbin_target_p, n_seeds=10,
            )
            print(
                f"[feasibility][t:{b['name']}({b['lo']}-{b['hi']})] matched "
                f"(target_p={tbin_target_p:.6f}): {_fmt_matched_msg(b['matched'])}"
            )
        except ValueError as e:
            print(f"[WARN] matched skipped for t_bin={b['name']}: {e}")

    if no_save:
        return

    # Overall CSV: single pooled row. Per-cat / per-tbin matched F1 spread
    # (mean, std) is appended so cross-axis variability is visible alongside
    # the pooled metrics. Spread cells are blank if matching failed everywhere.
    cat_f1_mean, cat_f1_std = _across_axis_f1_matched_stats(per_cat)
    tbin_f1_mean, tbin_f1_std = _across_axis_f1_matched_stats(per_tbin)
    csv_path = Path("logs") / "feasibility_metrics.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method",
                             *_METRIC_KEYS, "avg_feasibility_time",
                             *_F1_SPREAD_COLS])
        writer.writerow([
            run_id, obstruction_root, method_str,
            *_metric_row_values(metrics), f"{f_time.mean():.6f}",
            _fmt_optional(cat_f1_mean), _fmt_optional(cat_f1_std),
            _fmt_optional(tbin_f1_mean), _fmt_optional(tbin_f1_std),
        ])
    print(f"[INFO] Saved feasibility metrics CSV to {csv_path}")

    per_xx_dir = log_dir if log_dir is not None else Path("logs")

    # Per-category CSV
    per_cat_csv = per_xx_dir / "feasibility_per_category_metrics.csv"
    write_header = not per_cat_csv.exists() or per_cat_csv.stat().st_size == 0
    with per_cat_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method", "category", "n",
                             *_METRIC_KEYS, "avg_feasibility_time",
                             *_matched_csv_header_cols()])
        for b in per_cat:
            writer.writerow([
                run_id, obstruction_root, method_str, b["name"], b["n"],
                *_metric_row_values(b["natural"]), f"{b['t_mean']:.6f}",
                *_matched_csv_row_values(b["target_prevalence"], b["matched"]),
            ])
        # 'total' row: natural metrics on the full pooled data (matched cols blank)
        writer.writerow([
            run_id, obstruction_root, method_str, "total", f_label.shape[0],
            *_metric_row_values(metrics), f"{f_time.mean():.6f}",
            *_matched_csv_row_values(None, None),
        ])
    print(f"[INFO] Saved per-category feasibility metrics CSV to {per_cat_csv}")

    # Per-t-bin CSV
    per_tbin_csv = per_xx_dir / "feasibility_per_t_bin_metrics.csv"
    write_header = not per_tbin_csv.exists() or per_tbin_csv.stat().st_size == 0
    with per_tbin_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "obstruction_root", "method", "t_bin", "t_lo", "t_hi", "n",
                             *_METRIC_KEYS, "avg_feasibility_time",
                             *_matched_csv_header_cols()])
        for b in per_tbin:
            writer.writerow([
                run_id, obstruction_root, method_str, b["name"], b["lo"], b["hi"], b["n"],
                *_metric_row_values(b["natural"]), f"{b['t_mean']:.6f}",
                *_matched_csv_row_values(b["target_prevalence"], b["matched"]),
            ])
        # 'total' row: natural metrics on the full pooled data, t range = all bins
        t_lo_all = t_bins[0][1] if t_bins else 0
        t_hi_all = t_bins[-1][2] if t_bins else 0
        writer.writerow([
            run_id, obstruction_root, method_str, "total", t_lo_all, t_hi_all, f_label.shape[0],
            *_metric_row_values(metrics), f"{f_time.mean():.6f}",
            *_matched_csv_row_values(None, None),
        ])
    print(f"[INFO] Saved per-t-bin feasibility metrics CSV to {per_tbin_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstruction_root", type=str,
                        default="asset_release_v1.1/Obstruction_mini_test/benchmark_eval")
    parser.add_argument("--no-save", dest="no_save", action="store_true", default=False,
                        help="If set, skip creating log_dir, saving config.json, per-file h5 dumps, and metrics CSVs (still prints to stdout).")
    parser.add_argument("--t-ratio", dest="t_ratio", type=str, default="111",
                        help="3-digit ratio for low:mid:high t-index bins, e.g. '111' (1:1:1) "
                             "or '123' (1:2:3). Total t-index range is split proportionally.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for binarizing float predictions in "
                             "accuracy/precision/recall/f1 metrics. AP uses the raw scores. Default: 0.5.")
    parser.add_argument("--method", type=str, default="fcl", choices=["fcl", "grn", "cbn", "scn"])
    parser.add_argument("--pitch", type=float, default=0.015)
    parser.add_argument("--robot_geom", type=str, default="urdf", choices=["urdf", "pc"],
                        help="fcl only: define the robot collision model from URDF+FK (urdf, default) "
                             "or from per-row pc_robot ply via marching cubes (pc).")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--robot_to_model_t", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=None, help="robot_to_model translation (x y z) for cbn/scn. "
                             "If omitted, it is auto-fit from the ply point distribution.")
    args = parser.parse_args()

    # method_str labels CSV rows and log_dir; fcl gets robot_geom and pitch baked in
    # so the two robot collision modes (urdf vs pc) are easy to compare in logs.
    if args.method == "fcl":
        method_str = f"fcl_{args.robot_geom}_{args.pitch}"
    else:
        method_str = args.method

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_str}"
    log_dir: Optional[Path] = None
    if not args.no_save:
        log_dir = Path("logs") / run_id
        log_dir.mkdir(parents=True, exist_ok=True)

    c_result: Optional[ArrayTriple] = None
    resolved_robot_to_model_t: Optional[Tuple[float, float, float]] = None

    if args.method == "fcl":
        c_result, f_result = build_fcl_arrays(
            args.obstruction_root, args.pitch, log_dir,
            visualize=args.visualize, robot_geom=args.robot_geom,
        )
    elif args.method == "grn":
        c_result, f_result = build_grn_arrays(args.obstruction_root, log_dir)
    elif args.method == "cbn":
        if args.robot_to_model_t:
            kwargs = {"robot_to_model_t": tuple(args.robot_to_model_t)}  # manual
        else:
            kwargs = {"auto_fit": True}                                 # auto-fit from ply
        f_result, resolved_robot_to_model_t = build_cbn_arrays(args.obstruction_root, log_dir, **kwargs)
    elif args.method == "scn":
        if args.robot_to_model_t:
            kwargs = {"robot_to_model_t": tuple(args.robot_to_model_t)}  # manual
        else:
            kwargs = {"auto_fit": True}                                 # auto-fit from ply
        f_result, resolved_robot_to_model_t = build_scn_arrays(args.obstruction_root, log_dir, **kwargs)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # config.json is written after the build so it can record the resolved
    # robot_to_model translation actually used (auto-fit estimate or manual value).
    if log_dir is not None:
        config_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "args": vars(args),
            "resolved_robot_to_model_t": (
                list(resolved_robot_to_model_t)
                if resolved_robot_to_model_t is not None else None
            ),
        }
        with (log_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_record, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved run config to {log_dir / 'config.json'}")

    total_tasks = int(f_result[4].max()) + 1
    t_bins = _parse_t_ratio(args.t_ratio, total_tasks)
    print(f"[INFO] t-bins (ratio={args.t_ratio}, total={total_tasks}): "
          + ", ".join(f"{n}:[{lo}-{hi}]" for n, lo, hi in t_bins))
    if c_result is not None:
        compute_collision_metrics(c_result, run_id, args.obstruction_root, method_str, args.no_save, t_bins, args.threshold, log_dir)
    compute_feasibility_metrics(f_result, run_id, args.obstruction_root, method_str, args.no_save, t_bins, args.threshold, log_dir)


if __name__ == "__main__":
    main()
