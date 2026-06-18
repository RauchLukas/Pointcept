"""
Hyperparameter-sweep config generator for Pointcept.

Why a generator (and not ``--options``):
    Several values in the study config are *derived* at file-execution time
    (``epoch = 50 * loop``, every transform embeds ``grid_size`` / ``voxel_max``,
    and ``scheduler.max_lr`` embeds ``lr``).  Overriding the top-level variable
    via ``--options`` does NOT update those derived uses.  Instead we read the
    base config as text, substitute the top-level assignment lines, and write a
    full standalone config so every derived value is recomputed correctly.

Usage:
    # Phase 0 - learning-rate search (reduced loop, fixed batch_size=2):
    python tools/gen_sweep_configs.py --phase lr

    # Phase 1 - main sweep (edit BEST_LR + the grids below first):
    python tools/gen_sweep_configs.py --phase main

Outputs:
    - One config .py per setting under configs/<DATASET>/<OUT_SUBDIR>/
    - A manifest CSV mapping config name -> parameters
    - A launch script (scripts/<OUT_SUBDIR>_launch.sh) that calls scripts/train.sh
"""

import argparse
import csv
import itertools
import os
import re

# --------------------------------------------------------------------------- #
#  Paths / launch settings (edit to taste)
# --------------------------------------------------------------------------- #
DATASET = "rohbau3d"
BASE_CONFIG = "configs/rohbau3d/semseg-pt-v3m1-0-parameterstudy.py"
# IMPORTANT: in Pointcept the config ``batch_size`` is the TOTAL across all GPUs
# and must be divisible by NUM_GPU. The per-GPU batch is batch_size // NUM_GPU.
# This study uses 4-GPU DDP with per-GPU batch 1/2/3, i.e. TOTAL batch_size of
# 4/8/12. Set totals accordingly below (per_gpu * NUM_GPU).
NUM_GPU = 4  # GPUs per run, passed to scripts/train.sh -g
NUM_DEVICES = 4  # total GPUs available (only used by the single-GPU parallel launcher)

# Parameters this generator is allowed to substitute. Each MUST exist as a
# top-level ``name = ...`` assignment in the base config.
SUBSTITUTABLE = ["lr", "batch_size", "loop", "grid_size", "voxel_max", "mix_prob"]

# --------------------------------------------------------------------------- #
#  Phase 0: learning-rate search
#  Goal: find a good reference lr cheaply. Reduced loop is a fast proxy - the lr
#  *ranking* is usually stable across training length even if absolute mIoU is
#  not. Pick the best val mIoU, then use it as BEST_LR for the main phase.
# --------------------------------------------------------------------------- #
LR_PHASE = dict(
    out_subdir="sweep_lr",
    # batch_size=12 TOTAL = per-GPU 3 x 4 GPUs.
    fixed=dict(batch_size=12, loop=4, grid_size=0.04, voxel_max=64000, mix_prob=0),
    grid=dict(lr=[0.0003, 0.0006, 0.001, 0.003, 0.006]),
)

# --------------------------------------------------------------------------- #
#  Phase 1: main sweep. EDIT BEST_LR and the grids after Phase 0 finishes.
# --------------------------------------------------------------------------- #
BEST_LR = 0.003  # <-- set to the winner of the lr search
MAIN_PHASE = dict(
    out_subdir="sweep_main",
    fixed=dict(lr=BEST_LR, mix_prob=0),
    grid=dict(
        batch_size=[4, 8, 12],  # TOTAL = per-GPU 1/2/3 x 4 GPUs
        grid_size=[0.08],     # <-- add the grid sizes you want to test
        voxel_max=[80000],    # <-- add the voxel_max values you want to test
        loop=[16],            # <-- add the loop values you want to test
    ),
)


def _fmt(value):
    """Render a python literal for substitution."""
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _tag(value):
    """Filesystem-safe token for a value (0.08 -> 0p08)."""
    return str(value).replace(".", "p").replace("-", "m")


def substitute(text, overrides):
    for key, value in overrides.items():
        if key not in SUBSTITUTABLE:
            raise ValueError(f"'{key}' is not in SUBSTITUTABLE")
        pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        if not pattern.search(text):
            raise ValueError(
                f"Could not find top-level assignment '{key} = ...' in base config"
            )
        text = pattern.sub(f"{key} = {_fmt(value)}", text, count=1)
    return text


def fix_base_depth(text, extra_depth):
    """Adjust ``_base_`` relative paths for configs written into a subdirectory.

    The base config sits in ``configs/<DATASET>/`` and uses ``"../_base_/..."``.
    Generated configs sit ``extra_depth`` levels deeper, so each relative
    ``../`` prefix in ``_base_`` must gain ``extra_depth`` more ``../``.
    """
    if extra_depth <= 0:
        return text
    prefix = "../" * extra_depth
    return text.replace('"../_base_/', f'"{prefix}../_base_/')


def build_phase(phase_cfg, base_text, root):
    out_subdir = phase_cfg["out_subdir"]
    fixed = phase_cfg["fixed"]
    grid = phase_cfg["grid"]

    out_dir = os.path.join(root, "configs", DATASET, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    grid_keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in grid_keys]))

    manifest_rows = []
    launch_lines = [
        "#!/bin/sh",
        "set -e",
        "# Location-independent: invoke from anywhere (e.g. the directory above",
        "# Pointcept). train.sh itself cd's to the Pointcept root, so data_root",
        "# (../data) resolves correctly regardless of where you run this from.",
        'SCRIPT_DIR=$(dirname "$0")',
        "",
    ]

    for combo in combos:
        overrides = dict(fixed)
        overrides.update({k: v for k, v in zip(grid_keys, combo)})

        name_tokens = [f"{k}{_tag(v)}" for k, v in zip(grid_keys, combo)]
        exp_name = "__".join(name_tokens)
        config_name = f"{out_subdir}/{exp_name}"  # relative to configs/<DATASET>/

        # configs are written into configs/<DATASET>/<out_subdir>/ -> one level
        # deeper than the base config, so fix the _base_ relative path depth.
        extra_depth = len(out_subdir.split("/"))
        config_text = fix_base_depth(substitute(base_text, overrides), extra_depth)
        config_path = os.path.join(out_dir, f"{exp_name}.py")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(config_text)

        row = dict(config_name=config_name, exp_name=f"{out_subdir}/{exp_name}")
        row.update(overrides)
        manifest_rows.append(row)

        launch_lines.append(
            f'sh "$SCRIPT_DIR/train.sh" -d {DATASET} -c {config_name} '
            f"-n {out_subdir}/{exp_name} -g {NUM_GPU}"
        )

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = ["config_name", "exp_name"] + sorted(
        {k for row in manifest_rows for k in row.keys()} - {"config_name", "exp_name"}
    )
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    launch_path = os.path.join(root, "scripts", f"{out_subdir}_launch.sh")
    with open(launch_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(launch_lines) + "\n")

    print(f"[{out_subdir}] wrote {len(combos)} configs -> {out_dir}")
    print(f"[{out_subdir}] manifest -> {manifest_path}")
    print(f"[{out_subdir}] launch script (sequential) -> {launch_path}")

    # The per-GPU parallel launcher only makes sense when each run uses a single
    # GPU. With multi-GPU DDP (NUM_GPU > 1) each run already uses every GPU, so
    # runs must be sequential.
    if NUM_GPU == 1:
        parallel_lines = [
            "#!/bin/sh",
            "# Parallel launcher: runs configs concurrently, one per GPU, in waves",
            f"# of {NUM_DEVICES}. Invoke from anywhere (train.sh cd's to Pointcept root).",
            'SCRIPT_DIR=$(dirname "$0")',
            "",
        ]
        for i, row in enumerate(manifest_rows):
            device = i % NUM_DEVICES
            parallel_lines.append(
                f'CUDA_VISIBLE_DEVICES={device} sh "$SCRIPT_DIR/train.sh" -d {DATASET} '
                f'-c {row["config_name"]} -n {row["exp_name"]} -g 1 &'
            )
            if device == NUM_DEVICES - 1:
                parallel_lines.append("wait")
        parallel_lines.append("wait")

        parallel_path = os.path.join(
            root, "scripts", f"{out_subdir}_launch_parallel.sh"
        )
        with open(parallel_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(parallel_lines) + "\n")
        print(f"[{out_subdir}] launch script (parallel)   -> {parallel_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["lr", "main"], required=True, help="which phase to generate"
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_config_path = os.path.join(root, BASE_CONFIG)
    with open(base_config_path, "r", encoding="utf-8") as fh:
        base_text = fh.read()

    phase_cfg = LR_PHASE if args.phase == "lr" else MAIN_PHASE
    build_phase(phase_cfg, base_text, root)


if __name__ == "__main__":
    main()
