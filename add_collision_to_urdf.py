#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def _indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _parse_vec3(text, default):
    if text is None or text.strip() == "":
        return np.array(default, dtype=np.float64)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        return np.array(default, dtype=np.float64)
    return np.array(vals, dtype=np.float64)


def _rpy_to_rot(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _fmt_vec3(vec):
    return " ".join([f"{float(x):.10g}" for x in vec])


def _make_collision_box(origin_xyz, origin_rpy, size_xyz):
    collision = ET.Element("collision")

    origin = ET.Element("origin")
    origin.attrib["xyz"] = _fmt_vec3(origin_xyz)
    origin.attrib["rpy"] = _fmt_vec3(origin_rpy)

    geometry = ET.Element("geometry")
    box = ET.Element("box")
    box.attrib["size"] = _fmt_vec3(size_xyz)
    geometry.append(box)

    collision.append(origin)
    collision.append(geometry)
    return collision


def _select_largest_visual_box_candidate(visual_elems, urdf_dir):
    candidates = []
    for visual in visual_elems:
        visual_origin = visual.find("origin")
        visual_geometry = visual.find("geometry")
        if visual_geometry is None:
            continue

        mesh_elem = visual_geometry.find("mesh")
        if mesh_elem is None or "filename" not in mesh_elem.attrib:
            continue

        mesh_path = (urdf_dir / mesh_elem.attrib["filename"]).resolve()
        if not mesh_path.exists():
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        elif isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)

        if not isinstance(mesh, trimesh.Trimesh):
            continue

        bounds = mesh.bounds
        extents = np.maximum(bounds[1] - bounds[0], 1e-4)
        center = (bounds[0] + bounds[1]) / 2.0
        volume = float(np.prod(extents))

        if visual_origin is None:
            vis_xyz = np.zeros(3, dtype=np.float64)
            vis_rpy = np.zeros(3, dtype=np.float64)
        else:
            vis_xyz = _parse_vec3(visual_origin.attrib.get("xyz", None), default=[0.0, 0.0, 0.0])
            vis_rpy = _parse_vec3(visual_origin.attrib.get("rpy", None), default=[0.0, 0.0, 0.0])

        candidates.append({
            "extents": extents,
            "center": center,
            "vis_xyz": vis_xyz,
            "vis_rpy": vis_rpy,
            "rot": _rpy_to_rot(vis_rpy),
            "volume": volume,
        })

    if not candidates:
        return None

    return max(candidates, key=lambda c: c["volume"])


def _ensure_collision_from_visual(link_elem, mode, urdf_dir, replace_existing=False,
                                  hollow_open_face="+z", hollow_thickness_ratio=0.08,
                                  hollow_min_thickness=0.003):
    visual_elems = [child for child in list(link_elem) if child.tag == "visual"]
    collision_elems = [child for child in list(link_elem) if child.tag == "collision"]

    if mode == "skip":
        removed = 0
        if replace_existing and collision_elems:
            for coll in collision_elems:
                link_elem.remove(coll)
                removed += 1
        return 0, removed

    if not visual_elems:
        return 0, 0

    removed = 0
    if replace_existing and collision_elems:
        for coll in collision_elems:
            link_elem.remove(coll)
            removed += 1
        collision_elems = []

    if collision_elems:
        return 0, removed

    inserted = 0

    if mode in ("box", "hollow-box"):
        candidate = _select_largest_visual_box_candidate(visual_elems, urdf_dir)
        if candidate is None:
            mode = "mesh"
        else:
            ext = candidate["extents"]
            center = candidate["center"]
            vis_xyz = candidate["vis_xyz"]
            vis_rpy = candidate["vis_rpy"]
            rot = candidate["rot"]

            if mode == "box":
                world_center = vis_xyz + rot @ center
                link_elem.append(_make_collision_box(world_center, vis_rpy, ext))
                return 1, removed

            thickness = max(float(hollow_min_thickness), float(np.min(ext) * hollow_thickness_ratio))
            thickness = min(thickness, float(np.min(ext) * 0.45))

            ex, ey, ez = [float(v) for v in ext]

            wall_defs = {
                "+z": (np.array([0.0, 0.0, +(ez - thickness) / 2.0]), np.array([ex, ey, thickness])),
                "-z": (np.array([0.0, 0.0, -(ez - thickness) / 2.0]), np.array([ex, ey, thickness])),
                "+x": (np.array([+(ex - thickness) / 2.0, 0.0, 0.0]), np.array([thickness, ey, ez])),
                "-x": (np.array([-(ex - thickness) / 2.0, 0.0, 0.0]), np.array([thickness, ey, ez])),
                "+y": (np.array([0.0, +(ey - thickness) / 2.0, 0.0]), np.array([ex, thickness, ez])),
                "-y": (np.array([0.0, -(ey - thickness) / 2.0, 0.0]), np.array([ex, thickness, ez])),
            }

            if hollow_open_face not in wall_defs:
                hollow_open_face = "+z"

            for face_key, (offset, size) in wall_defs.items():
                if face_key == hollow_open_face:
                    continue
                world_center = vis_xyz + rot @ (center + offset)
                link_elem.append(_make_collision_box(world_center, vis_rpy, size))
                inserted += 1

            return inserted, removed

    for visual in visual_elems:
        collision = ET.Element("collision")

        visual_origin = visual.find("origin")
        visual_geometry = visual.find("geometry")
        if visual_geometry is None:
            continue

        if mode == "mesh":
            if visual_origin is not None:
                collision.append(copy.deepcopy(visual_origin))
            collision.append(copy.deepcopy(visual_geometry))
            link_elem.append(collision)
            inserted += 1
            continue

        mesh_elem = visual_geometry.find("mesh")
        if mesh_elem is None or "filename" not in mesh_elem.attrib:
            if visual_origin is not None:
                collision.append(copy.deepcopy(visual_origin))
            collision.append(copy.deepcopy(visual_geometry))
            link_elem.append(collision)
            inserted += 1
            continue

        mesh_file = mesh_elem.attrib["filename"]
        mesh_path = (urdf_dir / mesh_file).resolve()

        if not mesh_path.exists():
            if visual_origin is not None:
                collision.append(copy.deepcopy(visual_origin))
            collision.append(copy.deepcopy(visual_geometry))
            link_elem.append(collision)
            inserted += 1
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        elif isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)

        if not isinstance(mesh, trimesh.Trimesh):
            if visual_origin is not None:
                collision.append(copy.deepcopy(visual_origin))
            collision.append(copy.deepcopy(visual_geometry))
            link_elem.append(collision)
            inserted += 1
            continue

        bounds = mesh.bounds
        extents = np.maximum(bounds[1] - bounds[0], 1e-4)
        center = (bounds[0] + bounds[1]) / 2.0

        if visual_origin is None:
            vis_xyz = np.zeros(3, dtype=np.float64)
            vis_rpy = np.zeros(3, dtype=np.float64)
        else:
            vis_xyz = _parse_vec3(visual_origin.attrib.get("xyz", None), default=[0.0, 0.0, 0.0])
            vis_rpy = _parse_vec3(visual_origin.attrib.get("rpy", None), default=[0.0, 0.0, 0.0])

        rot = _rpy_to_rot(vis_rpy)
        col_xyz = vis_xyz + rot @ center

        origin = ET.Element("origin")
        origin.attrib["xyz"] = _fmt_vec3(col_xyz)
        origin.attrib["rpy"] = _fmt_vec3(vis_rpy)

        geometry = ET.Element("geometry")
        box = ET.Element("box")
        box.attrib["size"] = _fmt_vec3(extents)
        geometry.append(box)

        collision.append(origin)
        collision.append(geometry)
        link_elem.append(collision)
        inserted += 1

    return inserted, removed


def _non_fixed_child_links(root):
    links = set()
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "").strip().lower()
        if joint_type and joint_type != "fixed":
            child = joint.find("child")
            if child is None:
                continue
            child_link = child.attrib.get("link")
            if child_link:
                links.add(child_link)
    return links


def process_urdf(urdf_path: Path, in_place: bool, suffix: str, replace_existing: bool,
                 fixed_links_only: bool, fixed_mode: str, nonfixed_mode: str,
                 hollow_open_face: str, hollow_thickness_ratio: float,
                 hollow_min_thickness: float):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Not a URDF robot file: {urdf_path}")

    total_inserted = 0
    total_removed = 0
    link_count = 0
    skipped_non_fixed = 0

    non_fixed_children = _non_fixed_child_links(root)

    for link in root.findall("link"):
        link_count += 1
        link_name = link.attrib.get("name", "")
        if fixed_links_only and link_name in non_fixed_children:
            skipped_non_fixed += 1
            continue

        link_mode = nonfixed_mode if link_name in non_fixed_children else fixed_mode
        if link_mode == "skip" and link_name in non_fixed_children:
            skipped_non_fixed += 1

        inserted, removed = _ensure_collision_from_visual(
            link,
            mode=link_mode,
            urdf_dir=urdf_path.parent,
            replace_existing=replace_existing,
            hollow_open_face=hollow_open_face,
            hollow_thickness_ratio=hollow_thickness_ratio,
            hollow_min_thickness=hollow_min_thickness,
        )
        total_inserted += inserted
        total_removed += removed

    _indent(root)

    if in_place:
        out_path = urdf_path
    else:
        out_path = urdf_path.with_name(f"{urdf_path.stem}{suffix}{urdf_path.suffix}")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    return {
        "file": str(urdf_path),
        "out": str(out_path),
        "links": link_count,
        "collisions_added": total_inserted,
        "collisions_removed": total_removed,
        "skipped_non_fixed_links": skipped_non_fixed,
    }


def collect_urdfs(path: Path):
    if path.is_file():
        if path.suffix.lower() != ".urdf":
            raise ValueError(f"Input file is not .urdf: {path}")
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.urdf"))
    raise ValueError(f"Path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add collision tags to URDF links that have visual but no collision. "
            "Joints and all other tags remain unchanged."
        )
    )
    parser.add_argument("input", type=str, help="URDF file or directory")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source URDF(s). If omitted, write to *_with_collision.urdf",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_with_collision",
        help="Output suffix when not using --in-place (default: _with_collision)",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Remove existing collision tags and regenerate from visual tags",
    )
    parser.add_argument(
        "--fixed-links-only",
        action="store_true",
        help=(
            "Only add collisions to links that are not children of non-fixed joints "
            "(safer for articulated URDFs in PhysX)."
        ),
    )
    parser.add_argument(
        "--fixed-mode",
        type=str,
        default="mesh",
        choices=["skip", "mesh", "box", "hollow-box"],
        help="Collision generation mode for fixed links",
    )
    parser.add_argument(
        "--nonfixed-mode",
        type=str,
        default="skip",
        choices=["skip", "mesh", "box", "hollow-box"],
        help="Collision generation mode for links that are children of non-fixed joints",
    )
    parser.add_argument(
        "--hollow-open-face",
        type=str,
        default="+z",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="Which face to keep open when using hollow-box mode",
    )
    parser.add_argument(
        "--hollow-thickness-ratio",
        type=float,
        default=0.08,
        help="Wall thickness ratio relative to smallest box extent in hollow-box mode",
    )
    parser.add_argument(
        "--hollow-min-thickness",
        type=float,
        default=0.003,
        help="Minimum wall thickness (meters) in hollow-box mode",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    urdf_files = collect_urdfs(input_path)
    if not urdf_files:
        print("No URDF files found.")
        return

    nonfixed_mode = args.nonfixed_mode
    if args.fixed_links_only:
        nonfixed_mode = "skip"

    print(f"Processing {len(urdf_files)} URDF file(s)...")
    for urdf in urdf_files:
        result = process_urdf(
            urdf_path=urdf,
            in_place=args.in_place,
            suffix=args.suffix,
            replace_existing=args.replace_existing,
            fixed_links_only=args.fixed_links_only,
            fixed_mode=args.fixed_mode,
            nonfixed_mode=nonfixed_mode,
            hollow_open_face=args.hollow_open_face,
            hollow_thickness_ratio=args.hollow_thickness_ratio,
            hollow_min_thickness=args.hollow_min_thickness,
        )
        print(
            f"- {result['file']} -> {result['out']} | "
            f"links={result['links']} added={result['collisions_added']} removed={result['collisions_removed']} "
            f"skipped_non_fixed={result['skipped_non_fixed_links']}"
        )


if __name__ == "__main__":
    main()


# python add_collision_to_urdf.py asset_release/benchmark_scenes/Test/assets/scene_001/asset.urdf --fixed-mode mesh --nonfixed-mode hollow-box --hollow-open-face +z --suffix _safe_hollow_collision

# 두께 조절
# python add_collision_to_urdf.py asset_release/benchmark_scenes/Test/assets/scene_001/asset.urdf --fixed-mode mesh --nonfixed-mode hollow-box --hollow-open-face +z --hollow-thickness-ratio 0.06 --hollow-min-thickness 0.002 --suffix _safe_hollow_collision