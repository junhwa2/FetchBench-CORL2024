#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def count_stable_poses(npy_path: Path) -> int:
    data = np.load(npy_path, allow_pickle=True)
    if np.isscalar(data) or data.ndim == 0:
        return 1
    return int(data.shape[0])


def find_multi_stable_poses(objects_root: Path, min_count: int) -> tuple[list[tuple[str, str, int]], list[str], list[str]]:
    if not objects_root.exists():
        raise FileNotFoundError(f"Root not found: {objects_root}")

    matched: list[tuple[str, str, int]] = []
    missing: list[str] = []
    errors: list[str] = []

    for category_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        for item_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            npy_path = item_dir / "stable_poses.npy"
            rel = f"{category_dir.name}/{item_dir.name}"

            if not npy_path.is_file():
                missing.append(rel)
                continue

            try:
                count = count_stable_poses(npy_path)
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
                continue

            if count >= min_count:
                matched.append((category_dir.name, item_dir.name, count))

    return matched, missing, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find benchmark_objects/{category}/{id} entries whose stable_poses.npy "
            "contains at least N poses"
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("asset_release/benchmark_objects"),
        help="Path to benchmark_objects root (default: asset_release/benchmark_objects)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum number of stable poses to report (default: 2)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("asset_release/benchmark_objects/multi_stable_poses.txt"),
        help=(
            "Output txt path (default: asset_release/benchmark_objects/multi_stable_poses.txt). "
            "Writes lines as category/id,count"
        ),
    )
    args = parser.parse_args()

    matched, missing, errors = find_multi_stable_poses(args.root, args.min_count)

    print(f"Matched (>= {args.min_count}): {len(matched)}")
    for category, item_id, count in matched:
        print(f"{category}/{item_id},{count}")

    print(f"Missing stable_poses.npy: {len(missing)}")
    print(f"Load errors: {len(errors)}")

    if errors:
        print("\nFirst 20 load errors:")
        for line in errors[:20]:
            print(line)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{category}/{item_id},{count}" for category, item_id, count in matched]
        args.report.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Report written to: {args.report.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python dataset_tools/find_multi_stable_poses.py --min-count 2