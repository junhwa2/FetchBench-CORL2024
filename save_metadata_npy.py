#!/usr/bin/env python3
import argparse
import numpy as np

"""
dataset의 내용(data에 해당하는 scene 종류)에 맞춰 parser로 input 시 metadata.npy로 저장
e.g. cabinet, drawer, shelf, etc. -> metadata.npy에 저장
"""
def main():
    parser = argparse.ArgumentParser(description="Save a single name into .npy")
    parser.add_argument("--name", required=True, help="Name to store, e.g. cabinet")
    parser.add_argument("--out", default="metadata.npy", help="Output npy path")
    args = parser.parse_args()

    np.save(args.out, np.array([args.name], dtype=np.str_))
    print(f"saved: {args.out} -> [{args.name}]")


if __name__ == "__main__":
    main()

# python save_metadata_npy.py --name drawer --out asset_release/benchmark_scenes/Test/assets/scene_001/metadata.npy