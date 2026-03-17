#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_stable_pose_map(txt_path: Path) -> dict[tuple[str, str], int]:
    if not txt_path.exists():
        raise FileNotFoundError(f"stable pose list not found: {txt_path}")

    mapping: dict[tuple[str, str], int] = {}
    for line_no, raw in enumerate(txt_path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        try:
            left, count_str = line.rsplit(",", 1)
            category, item_id = left.split("/", 1)
            count = int(count_str)
        except Exception as exc:
            raise ValueError(f"Invalid format at line {line_no}: {line}") from exc

        mapping[(category, item_id)] = count

    return mapping


def add_stable_pose_column(metadata_csv: Path, stable_pose_txt: Path, output_csv: Path) -> tuple[int, int]:
    df = pd.read_csv(metadata_csv)

    if "Category" not in df.columns or "ID" not in df.columns:
        raise KeyError("metadata CSV must contain 'Category' and 'ID' columns")

    stable_map = load_stable_pose_map(stable_pose_txt)

    stable_values = []
    matched = 0
    for category, item_id in zip(df["Category"], df["ID"]):
        key = (str(category), str(item_id))
        value = stable_map.get(key)
        stable_values.append(value)
        if value is not None:
            matched += 1

    if "stable_pose" in df.columns:
        df["stable_pose"] = stable_values
    elif "remove" in df.columns:
        insert_idx = df.columns.get_loc("remove") + 1
        df.insert(insert_idx, "stable_pose", stable_values)
    else:
        df["stable_pose"] = stable_values

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return matched, len(df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add/update stable_pose column in metadata.csv using multi_stable_poses.txt "
            "(category/id,count)."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("asset_release/benchmark_objects/metadata.csv"),
        help="Path to metadata.csv (default: asset_release/benchmark_objects/metadata.csv)",
    )
    parser.add_argument(
        "--stable-pose-txt",
        type=Path,
        default=Path("asset_release/benchmark_objects/multi_stable_poses.txt"),
        help="Path to multi_stable_poses.txt (default: asset_release/benchmark_objects/multi_stable_poses.txt)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("asset_release/benchmark_objects/metadata.csv"),
        help=(
            "Output CSV path (default: asset_release/benchmark_objects/metadata.csv). "
            "Default behavior overwrites metadata.csv."
        ),
    )
    args = parser.parse_args()

    matched, total = add_stable_pose_column(args.metadata_csv, args.stable_pose_txt, args.output_csv)
    print(f"Rows matched with stable pose counts: {matched}/{total}")
    print(f"Saved: {args.output_csv.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python dataset_tools/add_stable_pose_to_metadata.py