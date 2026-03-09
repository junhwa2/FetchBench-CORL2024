import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh


def load_scene_meshes(scene_dir: Path, rot_axis: str = "z", rot_deg: float = 0.0):
	metadata = np.load(scene_dir / "metadata.npy", allow_pickle=True)
	meshes = []
	for name in metadata:
		mesh_path = scene_dir / f"{name}.obj"
		mesh = trimesh.load_mesh(mesh_path)
		if isinstance(mesh, trimesh.Trimesh):
			meshes.append(mesh)
		else:
			meshes.extend(mesh.geometry.values())

	base_transform = np.array(
		[[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
		dtype=np.float32,
	)
	axis_map = {
		"x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
		"y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
		"z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
	}
	rot_transform = trimesh.transformations.rotation_matrix(
		angle=np.deg2rad(rot_deg),
		direction=axis_map[rot_axis],
		point=[0.0, 0.0, 0.0],
	)

	for mesh in meshes:
		mesh.apply_transform(base_transform)
		mesh.apply_transform(rot_transform)
	return meshes


def iter_support_entries(support_json_path: Path):
	with open(support_json_path, "r") as file:
		entries = json.load(file)
	for idx, entry in enumerate(entries):
		polygon = shapely.from_geojson(entry["polygon"])
		translation = np.array(entry["translation"], dtype=np.float32)
		label = entry.get("label", "")
		yield idx, polygon, translation, label


def add_polygon_to_scene(scene: trimesh.Scene, points_xyz: np.ndarray, radius: float):
	if len(points_xyz) < 2:
		return

	segments = np.stack([points_xyz[:-1], points_xyz[1:]], axis=1)
	path = trimesh.load_path(segments)
	scene.add_geometry(path)

	for point in points_xyz[:-1]:
		marker = trimesh.creation.uv_sphere(radius=radius)
		marker.apply_translation(point)
		scene.add_geometry(marker)


def main():
	parser = argparse.ArgumentParser(description="Visualize support.json polygons over scene meshes.")
	parser.add_argument(
		"--scene-dir",
		type=Path,
		required=True,
		help="Path to scene asset directory (contains metadata.npy/support.json)",
	)
	parser.add_argument("--label", type=str, default=None, help="Optional support label filter")
	parser.add_argument("--radius", type=float, default=0.01, help="Marker sphere radius")
	parser.add_argument(
		"--scene-rot-axis",
		type=str,
		choices=["x", "y", "z"],
		default="x",
		help="Additional rotation axis for scene meshes",
	)
	parser.add_argument(
		"--scene-rot-deg",
		type=float,
		default=270.0,
		help="Additional rotation angle (degrees) for scene meshes",
	)
	args = parser.parse_args()

	scene_dir = args.scene_dir
	support_json = scene_dir / "support.json"

	meshes = load_scene_meshes(scene_dir, rot_axis=args.scene_rot_axis, rot_deg=args.scene_rot_deg)

	vis_scene = trimesh.Scene()
	for mesh in meshes:
		vis_scene.add_geometry(mesh)

	any_added = False
	for idx, polygon, translation, label in iter_support_entries(support_json):
		if args.label is not None and label != args.label:
			continue

		if polygon.is_empty:
			print(f"[{idx}] label={label} -> empty polygon")
			continue

		area = polygon.area
		coords_xy = np.array(polygon.exterior.coords, dtype=np.float32)
		coords_xyz = np.column_stack(
			[coords_xy[:, 0], coords_xy[:, 1], np.zeros(len(coords_xy), dtype=np.float32)]
		)
		coords_xyz = coords_xyz + translation

		print(f"[{idx}] label={label}, area={area:.8f}, n_points={len(coords_xy)}")
		print(f"     translation={translation.tolist()}")
		print(f"     first_points={coords_xy[:4].tolist()}")

		add_polygon_to_scene(vis_scene, coords_xyz, radius=args.radius)
		any_added = True

	if not any_added:
		print("No support polygons were added to visualization. Check --label or support.json content.")

	vis_scene.show()


if __name__ == "__main__":
	main()



# python visualize_support_polygon.py --scene-dir asset_release/benchmark_scenes/Test/assets/scene_003/
# python visualize_support_polygon.py --scene-dir asset_release/benchmark_scenes/Test/assets/scene_003/ --scene-rot-axis y --scene-rot-deg 90