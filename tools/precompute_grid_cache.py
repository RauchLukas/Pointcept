"""
Pre-compute CachedGridSample voxel caches for all scenes.

Run once before training to eliminate the first-epoch cache-build overhead
and to parallelise computation across CPU cores.

Usage:
    python tools/precompute_grid_cache.py \
        --data_root data/rohbau3d \
        --splits train val \
        --grid_size 0.04 \
        --workers 8
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np


def center_shift(coord, apply_z=True):
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    if apply_z:
        shift = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=coord.dtype
        )
    else:
        shift = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, 0], dtype=coord.dtype
        )
    return coord - shift


def fnv_hash_vec(arr):
    assert arr.ndim == 2
    arr = arr.astype(np.uint64, copy=True)
    hashed = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def process_scene(args):
    scene_dir, grid_size = args
    cache_dir = os.path.join(scene_dir, f"grid_cache_{grid_size}")

    if (
        os.path.isfile(os.path.join(cache_dir, "idx_sort.npy"))
        and os.path.isfile(os.path.join(cache_dir, "count.npy"))
        and os.path.isfile(os.path.join(cache_dir, "inverse.npy"))
    ):
        return scene_dir, "skipped (cache exists)"

    coord_path = os.path.join(scene_dir, "coord.npy")
    if not os.path.isfile(coord_path):
        return scene_dir, "skipped (no coord.npy)"

    coord = np.load(coord_path).astype(np.float32)
    coord = center_shift(coord, apply_z=True)

    scaled = coord / np.float32(grid_size)
    grid_coord = np.floor(scaled).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord

    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in [("idx_sort", idx_sort), ("count", count), ("inverse", inverse)]:
        tmp = os.path.join(cache_dir, f"{name}.tmp.npy")
        final = os.path.join(cache_dir, f"{name}.npy")
        np.save(tmp, arr)
        os.replace(tmp, final)

    n_points = coord.shape[0]
    n_voxels = len(count)
    return scene_dir, f"OK  ({n_points:,} pts -> {n_voxels:,} voxels)"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute grid cache for CachedGridSample"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val)",
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.04, help="Voxel grid size (default: 0.04)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (default: 4)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cache already exists",
    )
    args = parser.parse_args()

    scenes = []
    for split in args.splits:
        found = sorted(
            glob.glob(os.path.join(args.data_root, split, "site_*", "scene_*"))
        )
        scenes.extend(found)
        print(f"[{split}] found {len(found)} scenes")

    if not scenes:
        print("No scenes found – check --data_root and --splits.", file=sys.stderr)
        sys.exit(1)

    if args.force:
        for s in scenes:
            cache_dir = os.path.join(s, f"grid_cache_{args.grid_size}")
            for f in ("idx_sort.npy", "count.npy", "inverse.npy"):
                p = os.path.join(cache_dir, f)
                if os.path.isfile(p):
                    os.remove(p)

    tasks = [(s, args.grid_size) for s in scenes]

    print(
        f"\nProcessing {len(tasks)} scenes with grid_size={args.grid_size} "
        f"using {args.workers} workers ...\n"
    )
    t0 = time.time()

    if args.workers <= 1:
        results = [process_scene(t) for t in tasks]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(process_scene, tasks)

    for scene_dir, status in results:
        print(f"  {os.path.relpath(scene_dir, args.data_root):>50s}  {status}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
