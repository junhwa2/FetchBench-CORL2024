#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np


def find_scene_dirs(root: Path):
    scene_dirs = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        if category_dir.name in {"benchmark_train", "benchmark_eval"}:
            continue

        for child in sorted(category_dir.iterdir()):
            if child.is_dir() and "SceneFactory_" in child.name:
                scene_dirs.append(child)

    return scene_dirs


def split_items(items, train_ratio: float, seed: int):
    items = list(items)
    random.Random(seed).shuffle(items)
    train_count = int(len(items) * train_ratio)
    train_items = items[:train_count]
    eval_items = items[train_count:]
    return train_items, eval_items


def move_dirs(dirs, destination: Path, dry_run: bool):
    destination.mkdir(parents=True, exist_ok=True)

    for src in dirs:
        category = src.parent.name
        dst_parent = destination / category
        dst = dst_parent / src.name
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")

        if dry_run:
            print(f"[DRY RUN] move: {src} -> {dst}")
        else:
            dst_parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"move: {src} -> {dst}")


def remove_empty_dirs(root: Path, dry_run: bool):
    removed = []
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            next(path.iterdir())
            continue
        except StopIteration:
            if dry_run:
                print(f"[DRY RUN] remove empty dir: {path}")
            else:
                path.rmdir()
                print(f"remove empty dir: {path}")
            removed.append(path)
    return removed


def list_scene_dirs(benchmark_dir: Path):
    if not benchmark_dir.exists() or not benchmark_dir.is_dir():
        return []
    scene_dirs = []
    for category_dir in sorted(benchmark_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for child in sorted(category_dir.iterdir()):
            if child.is_dir() and "SceneFactory_" in child.name:
                scene_dirs.append(child)
    return scene_dirs


def infer_num_tasks(scene_dir: Path):
    task_config_path = scene_dir / "task_config.npz"
    if not task_config_path.exists():
        return ""

    try:
        with np.load(task_config_path, allow_pickle=True) as data:
            if "task_init_state" not in data:
                return ""
            task_init_state = data["task_init_state"]
            if getattr(task_init_state, "shape", None) is None or len(task_init_state.shape) == 0:
                return ""
            return int(task_init_state.shape[0])
    except Exception:
        return ""


def write_metadata_csv(benchmark_dir: Path, dry_run: bool):
    scene_dirs = list_scene_dirs(benchmark_dir)
    csv_path = benchmark_dir / "metadata.csv"

    if dry_run:
        print(f"[DRY RUN] write metadata: {csv_path} (rows: {len(scene_dirs)})")
        return

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Category", "ID", "Path", "NumTasks"])

        for idx, scene_dir in enumerate(scene_dirs):
            scene_id = scene_dir.name
            category = scene_dir.parent.name
            relative_path = f"{category}/{scene_id}"
            num_tasks = infer_num_tasks(scene_dir)
            writer.writerow([idx, category, scene_id, relative_path, num_tasks])

    print(f"write metadata: {csv_path} (rows: {len(scene_dirs)})")


def main():
    parser = argparse.ArgumentParser(
        description="Split SceneFactory directories into benchmark_train and benchmark_eval with 8:2 ratio."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory path (example: /home/kist/FetchBench-CORL2024/asset_release/Task/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without actually moving directories",
    )

    args = parser.parse_args()
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Invalid root directory: {root}")

    train_dir = root / "benchmark_train"
    eval_dir = root / "benchmark_eval"

    scene_dirs = find_scene_dirs(root)
    if not scene_dirs:
        print(f"No SceneFactory directories found under: {root}")
        write_metadata_csv(train_dir, args.dry_run)
        write_metadata_csv(eval_dir, args.dry_run)
        return

    train_items, eval_items = split_items(scene_dirs, args.train_ratio, args.seed)

    print(f"Root: {root}")
    print(f"Found scene dirs: {len(scene_dirs)}")
    print(f"Train ({args.train_ratio:.0%}): {len(train_items)}")
    print(f"Eval  ({1 - args.train_ratio:.0%}): {len(eval_items)}")

    move_dirs(train_items, train_dir, args.dry_run)
    move_dirs(eval_items, eval_dir, args.dry_run)
    removed_dirs = remove_empty_dirs(root, args.dry_run)

    if removed_dirs:
        print(f"Removed empty dirs: {len(removed_dirs)}")
    else:
        print("No empty directories to remove.")

    write_metadata_csv(train_dir, args.dry_run)
    write_metadata_csv(eval_dir, args.dry_run)

    if args.dry_run:
        print("Done (dry run).")
    else:
        print("Done.")


if __name__ == "__main__":
    main()

# python split_benchmark.py ~/FetchBench-CORL2024/asset_release/Task/JH --seed=1