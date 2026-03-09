#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_missing_jpgs(objects_root: Path) -> list[Path]:
    missing: list[Path] = []
    if not objects_root.exists():
        raise FileNotFoundError(f"Root not found: {objects_root}")

    for class_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        for item_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            item_name = item_dir.name
            expected = item_dir / f"{item_name}.jpg"
            if not expected.is_file():
                missing.append(expected)
    return missing


def copy_missing_jpgs(objects_root: Path, textures_root: Path) -> tuple[list[Path], list[Path]]:
    missing: list[Path] = []
    copied: list[Path] = []
    if not objects_root.exists():
        raise FileNotFoundError(f"Root not found: {objects_root}")
    if not textures_root.exists():
        raise FileNotFoundError(f"Textures root not found: {textures_root}")

    for class_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        for item_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            item_name = item_dir.name
            expected = item_dir / f"{item_name}.jpg"
            if expected.is_file():
                continue
            source = textures_root / f"{item_name}.jpg"
            if source.is_file():
                shutil.copy2(source, expected)
                copied.append(expected)
            else:
                missing.append(expected)

    return copied, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for missing {item}.jpg under benchmark_objects/{class}/{item}/"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("asset_release/benchmark_objects"),
        help="Path to benchmark_objects root (default: asset_release/benchmark_objects)",
    )
    parser.add_argument(
        "--textures-root",
        type=Path,
        default=Path("ShapeNetSem/ShapeNetSem-backup/models-textures/textures"),
        help=(
            "Path to textures root containing {item}.jpg "
            "(default: ShapeNetSem/ShapeNetSem-backup/models-textures/textures)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output file to save missing paths",
    )
    args = parser.parse_args()

    copied, missing = copy_missing_jpgs(args.root, args.textures_root)

    if copied:
        print(f"Copied {len(copied)} jpg(s) from textures:")
        for path in copied:
            print(path.as_posix())

    if missing:
        print(f"Missing {len(missing)} jpg(s) after copy attempt:")
        for path in missing:
            print(path.as_posix())
    elif not copied:
        print("All items have corresponding jpg files.")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(p.as_posix() for p in missing) + ("\n" if missing else ""))
        print(f"Report written to: {args.report.as_posix()}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
