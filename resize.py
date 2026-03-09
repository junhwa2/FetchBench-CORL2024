#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import numpy as np


def obj_vertex_extents(obj_path: str):
    """Read only 'v ' lines and return (mins, maxs, extents, vertex_count)."""
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    n = 0

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = map(float, parts[1:4])
                    except Exception:
                        continue
                    mins = np.minimum(mins, [x, y, z])
                    maxs = np.maximum(maxs, [x, y, z])
                    n += 1

    if n == 0:
        raise ValueError(f"[{obj_path}] 정점(v) 라인을 찾지 못했음")

    extents = maxs - mins
    return mins, maxs, extents, n


def scale_obj_vertices_to_match(
    src_obj_path: str,
    ref_obj_path: str,
    out_obj_path: str,
    keep_mtllib: bool = True,
):
    """
    Scale only 'v ' lines of src OBJ so that its max bbox extent matches ref OBJ.
    Keep all other lines unchanged (f/usemtl/vt/vn/...)
    """
    _, _, ext_ref, _ = obj_vertex_extents(ref_obj_path)
    _, _, ext_src, _ = obj_vertex_extents(src_obj_path)

    ref_max = float(ext_ref.max())
    src_max = float(ext_src.max())
    if src_max == 0:
        raise ValueError(f"[{src_obj_path}] bbox extent가 0이라 스케일 계산 불가")

    scale = ref_max / src_max

    out_lines = []
    with open(src_obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # mtllib은 원본 유지(기본) / 필요하면 유지하지 않고 그대로 둬도 됨
            if line.startswith("mtllib ") and keep_mtllib:
                out_lines.append(line)
                continue

            # 정점만 스케일
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = map(float, parts[1:4])
                        rest = parts[4:]  # w 같은 추가 필드 보존
                        v_line = "v {:.8f} {:.8f} {:.8f}".format(x * scale, y * scale, z * scale)
                        if rest:
                            v_line += " " + " ".join(rest)
                        out_lines.append(v_line + "\n")
                        continue
                    except Exception:
                        # 파싱 실패 시 원본 라인 그대로
                        pass

            out_lines.append(line)

    os.makedirs(os.path.dirname(out_obj_path) or ".", exist_ok=True)
    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return scale


def _parse_mtllib(obj_path: str):
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib "):
                return line.strip().split(maxsplit=1)[1]
    return None


def _parse_mtl_textures(mtl_path: str):
    texture_files = []
    keys = ("map_", "bump", "disp", "decal", "refl")
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(keys):
                parts = line.split()
                # Find last token that doesn't look like an option flag
                tex = None
                for token in reversed(parts[1:]):
                    if not token.startswith("-"):
                        tex = token
                        break
                if tex:
                    texture_files.append(tex)
    return texture_files


def copy_mtl_and_textures(src_obj_path: str, dst_dir: str):
    mtllib = _parse_mtllib(src_obj_path)
    if not mtllib:
        return None

    src_mtl_path = os.path.join(os.path.dirname(src_obj_path), mtllib)
    if not os.path.exists(src_mtl_path):
        print(f"[WARN] mtllib not found: {src_mtl_path}")
        return None

    dst_mtl_path = os.path.join(dst_dir, mtllib)
    os.makedirs(os.path.dirname(dst_mtl_path) or ".", exist_ok=True)
    with open(src_mtl_path, "rb") as fsrc, open(dst_mtl_path, "wb") as fdst:
        fdst.write(fsrc.read())

    tex_files = _parse_mtl_textures(src_mtl_path)
    for tex in tex_files:
        src_tex_path = os.path.join(os.path.dirname(src_mtl_path), tex)
        dst_tex_path = os.path.join(dst_dir, tex)
        if not os.path.exists(src_tex_path):
            print(f"[WARN] texture not found: {src_tex_path}")
            continue
        os.makedirs(os.path.dirname(dst_tex_path) or ".", exist_ok=True)
        with open(src_tex_path, "rb") as fsrc, open(dst_tex_path, "wb") as fdst:
            fdst.write(fsrc.read())

    return dst_mtl_path


def find_src_obj(shapenet_root: str, category: str, obj_id: str, search_recursive: bool = False):
    candidates = [
        os.path.join(shapenet_root, f"{obj_id}.obj"),
        os.path.join(shapenet_root, f"{category}_{obj_id}.obj"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if search_recursive:
        matches = glob.glob(os.path.join(shapenet_root, "**", f"{obj_id}.obj"), recursive=True)
        if matches:
            return matches[0]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="src OBJ를 ref(mesh.obj) 크기에 맞게 스케일링 (v 라인만 변경, 나머지는 보존)"
    )
    ap.add_argument("--ref", help="레퍼런스 OBJ 경로 (예: /path/to/mesh.obj)")
    ap.add_argument("--src", help="소스 OBJ 경로 또는 글롭(예: ./in/*.obj)")
    ap.add_argument("--outdir", help="출력 디렉토리 (예: ./scaled_out)")
    ap.add_argument("--suffix", default="_scaled", help="출력 파일명 접미사 (기본: _scaled)")
    ap.add_argument("--keep-mtllib", action="store_true", help="src의 mtllib 라인을 그대로 유지 (기본: False)")

    # batch mode for FetchBench
    ap.add_argument("--batch-benchmark", action="store_true", help="benchmark_objects 전체를 일괄 처리")
    ap.add_argument("--benchmark-root", default="./asset_release/benchmark_objects",
                    help="benchmark_objects 루트 경로")
    ap.add_argument("--shapenet-root", default="./ShapeNetSem/ShapeNetSem-backup/models-OBJ/models",
                    help="ShapeNetSem OBJ 루트 경로")
    ap.add_argument("--search-recursive", action="store_true",
                    help="ShapeNetSem 루트에서 재귀 검색으로 OBJ 찾기")
    args = ap.parse_args()

    if args.batch_benchmark:
        bench_root = args.benchmark_root
        shapenet_root = args.shapenet_root
        categories = [d for d in os.listdir(bench_root) if os.path.isdir(os.path.join(bench_root, d))]
        categories.sort()

        for cat in categories:
            cat_dir = os.path.join(bench_root, cat)
            obj_ids = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
            obj_ids.sort()

            for obj_id in obj_ids:
                ref_path = os.path.join(cat_dir, obj_id, "mesh.obj")
                if not os.path.exists(ref_path):
                    continue

                src_obj = find_src_obj(shapenet_root, cat, obj_id, search_recursive=args.search_recursive)
                if not src_obj:
                    print(f"[SKIP] src OBJ not found for {cat}/{obj_id}")
                    continue

                out_dir = os.path.join(cat_dir, obj_id)
                out_path = os.path.join(out_dir, "mesh.obj")

                scale = scale_obj_vertices_to_match(
                    src_obj_path=src_obj,
                    ref_obj_path=ref_path,
                    out_obj_path=out_path,
                    keep_mtllib=True,
                )

                copy_mtl_and_textures(src_obj, out_dir)
                print(f"[OK] {cat}/{obj_id} -> {out_path} | scale={scale}")

        return

    if not args.ref or not args.src or not args.outdir:
        raise SystemExit("--ref, --src, --outdir 는 필수입니다 (batch 모드 제외)")

    ref_path = args.ref
    src_list = sorted(glob.glob(args.src))
    if not src_list:
        raise SystemExit(f"--src 패턴에 해당하는 파일이 없음: {args.src}")

    for src_path in src_list:
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(args.outdir, f"{base}{args.suffix}.obj")

        scale = scale_obj_vertices_to_match(
            src_obj_path=src_path,
            ref_obj_path=ref_path,
            out_obj_path=out_path,
            keep_mtllib=args.keep_mtllib,
        )
        if args.keep_mtllib:
            copy_mtl_and_textures(src_path, args.outdir)
        print(f"[OK] {os.path.basename(src_path)} -> {out_path} | scale={scale}")


if __name__ == "__main__":
    main()
