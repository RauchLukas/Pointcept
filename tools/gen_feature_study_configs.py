"""Generate Rohbau3D point-feature ablation configs and launch scripts.

The generated configs are standalone copies of the requested base config.  This
is intentional: values such as ``grid_size`` occur inside several transforms,
and changing only a top-level value with ``--options`` would not reliably update
hard-coded occurrences in the base config.

Examples:
    # Four independent ablations: XYZ, RGB, intensity, and normals
    python3 tools/gen_feature_study_configs.py

    # Add feature combinations and change fixed study parameters
    python3 tools/gen_feature_study_configs.py \
        --feature-sets xyz rgb intensity normals rgb+intensity rgb+intensity+normals \
        --seed 42 --grid-size 0.08 --voxel-max 80000 --batch-size 12 \
        --epoch 40 --loop 16 \
        --wandb-project pointcept_feature_ablation --force

Rohbau3D stores intensity under the key ``strength``.  The command-line name
remains ``intensity`` so that experiment and W&B names use familiar terminology.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shlex


DEFAULT_BASE_CONFIG = "configs/rohbau3d/semseg-pt-v3m1-0-parameterstudy.py"
DEFAULT_FEATURE_SETS = ("xyz", "rgb", "intensity", "normals")

FEATURES = {
    "xyz": ("coord", 3),
    "rgb": ("color", 3),
    "intensity": ("strength", 1),
    "normals": ("normal", 3),
}
ALIASES = {
    "color": "rgb",
    "colour": "rgb",
    "strength": "intensity",
    "normal": "normals",
}


def replace_once(text: str, pattern: str, replacement: str, description: str) -> str:
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {description} in the base config")
    return text


def replace_assignment(text: str, name: str, value: object) -> str:
    return replace_once(
        text,
        rf"^{re.escape(name)}\s*=.*$",
        f"{name} = {value!r}",
        f"top-level assignment for {name!r}",
    )


def parse_feature_set(spec: str) -> tuple[str, tuple[str, ...], int]:
    """Return canonical run name, Pointcept feat_keys, and channel count."""
    raw_names = [part.strip().lower() for part in re.split(r"[+,]", spec) if part.strip()]
    if not raw_names:
        raise argparse.ArgumentTypeError("feature set cannot be empty")

    canonical = []
    for raw_name in raw_names:
        name = ALIASES.get(raw_name, raw_name)
        if name not in FEATURES:
            choices = ", ".join(FEATURES)
            raise argparse.ArgumentTypeError(
                f"unknown feature {raw_name!r}; valid features are: {choices}"
            )
        if name not in canonical:
            canonical.append(name)

    feat_keys = tuple(FEATURES[name][0] for name in canonical)
    in_channels = sum(FEATURES[name][1] for name in canonical)
    return "+".join(canonical), feat_keys, in_channels


def python_tuple(values: tuple[str, ...]) -> str:
    if len(values) == 1:
        return f"({values[0]!r},)"
    return repr(values)


def adjust_base_paths(text: str, extra_depth: int) -> str:
    """Make relative ``_base_`` paths work from the generated subdirectory."""
    prefix = "../" * extra_depth
    return text.replace('"../_base_/', f'"{prefix}../_base_/')


def build_config(base_text: str, args: argparse.Namespace, feature_spec: str) -> tuple[str, dict]:
    feature_name, feat_keys, in_channels = parse_feature_set(feature_spec)
    text = base_text

    # Fixed parameters shared by all feature runs.
    for name, value in (
        ("seed", args.seed),
        ("batch_size", args.batch_size),
        ("loop", args.loop),
        ("eval_epoch", args.epoch),
        ("grid_size", args.grid_size),
        ("voxel_max", args.voxel_max),
        ("lr", args.lr),
        ("weight_decay", args.weight_decay),
        ("num_worker", args.num_worker),
        ("mix_prob", args.mix_prob),
        ("enable_wandb", args.enable_wandb),
        ("wandb_project", args.wandb_project),
    ):
        text = replace_assignment(text, name, value)

    if args.batch_size % args.num_gpu:
        raise ValueError("--batch-size must be divisible by --num-gpu")

    # ``--epoch`` is the number of epochs between evaluations. Pointcept's
    # top-level ``epoch`` is the total duration, hence eval_epoch * loop.
    text = replace_once(
        text,
        r"^epoch\s*=.*\neval_epoch\s*=.*$",
        f"eval_epoch = {args.epoch!r}\nepoch = eval_epoch * loop",
        "epoch/evaluation schedule assignments",
    )

    # Only the feature label is new. All training settings above replace their
    # existing parameter-study assignments, preserving its loop-based schedule.
    preamble = (
        f"\n# Point-feature-study label (generated).\n"
        f"feature_set = {feature_name!r}\n"
    )
    text = replace_once(
        text,
        r"^(_base_\s*=.*)$",
        rf"\1{preamble}",
        "_base_ declaration",
    )

    text = replace_once(
        text,
        r"^([ \t]*)in_channels\s*=\s*\d+.*$",
        rf"\g<1>in_channels={in_channels},  # generated from feature_set={feature_name!r}",
        "model in_channels setting",
    )

    feat_tuple = python_tuple(feat_keys)
    text, collect_count = re.subn(
        r"feat_keys\s*=\s*\(\s*[\"']color[\"']\s*,\s*[\"']normal[\"']\s*,\s*[\"']strength[\"']\s*,?\s*\)",
        f"feat_keys={feat_tuple}",
        text,
    )
    if collect_count < 3:
        raise ValueError(
            "Expected to update train, validation, and test Collect feat_keys "
            f"in the base config, but found {collect_count}"
        )

    # Make every geometry-resolution use follow the fixed study values.  The
    # base currently contains two hard-coded GridSample/GridCoord values in
    # addition to the top-level grid_size variable.
    text = re.sub(
        r"(?<!\n)grid_size\s*=\s*(?:0\.04|0\.05)",
        "grid_size=grid_size",
        text,
    )
    text = re.sub(
        r'(type=["\']SphereCrop["\']\s*,\s*point_max=)\d+(\s*,\s*mode=["\']random["\'])',
        r"\g<1>voxel_max\2",
        text,
    )

    text = adjust_base_paths(text, len(Path(args.out_subdir).parts))
    header = (
        "# AUTO-GENERATED by tools/gen_feature_study_configs.py; do not edit.\n"
        f"# Source: {args.base_config}\n"
    )
    metadata = {
        "feature_set": feature_name,
        "feat_keys": "+".join(feat_keys),
        "in_channels": in_channels,
    }
    return header + text, metadata


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to replace generated files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def validate_relative_component(value: str, option: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{option} must be a relative path without '..': {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-hyperparameter point-feature ablation runs."
    )
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--dataset", default="rohbau3d")
    parser.add_argument("--out-subdir", default="feature_study")
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=float, default=0.04)
    parser.add_argument("--voxel-max", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--epoch",
        type=int,
        default=40,
        help="epochs per evaluation cycle; total epochs are epoch * loop",
    )
    parser.add_argument("--loop", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=24)
    parser.add_argument("--num-gpu", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mix-prob", type=float, default=0.0)
    parser.add_argument("--wandb-project", default="pointcept_feature_study")
    parser.add_argument(
        "--no-wandb", dest="enable_wandb", action="store_false", help="disable W&B logging"
    )
    parser.set_defaults(enable_wandb=True)
    parser.add_argument("--force", action="store_true", help="replace existing generated files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_relative_component(args.dataset, "--dataset")
    validate_relative_component(args.out_subdir, "--out-subdir")
    if args.num_gpu <= 0:
        raise ValueError("--num-gpu must be positive")
    if args.grid_size <= 0 or args.voxel_max <= 0:
        raise ValueError("--grid-size and --voxel-max must be positive")
    if args.epoch <= 0 or args.loop <= 0:
        raise ValueError("--epoch and --loop must be positive")

    root = Path(__file__).resolve().parents[1]
    base_path = root / args.base_config
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    parsed_sets = [parse_feature_set(spec)[0] for spec in args.feature_sets]
    if len(parsed_sets) != len(set(parsed_sets)):
        raise ValueError("--feature-sets contains duplicate feature combinations")

    base_text = base_path.read_text(encoding="utf-8")
    out_dir = root / "configs" / args.dataset / args.out_subdir
    rows = []
    generated_config_paths = set()
    launch_lines = [
        "#!/bin/sh",
        "set -eu",
        "# Generated point-feature study; runs are sequential to avoid GPU overlap.",
        'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
        "",
    ]

    for feature_spec in args.feature_sets:
        config_text, metadata = build_config(base_text, args, feature_spec)
        run_name = metadata["feature_set"].replace("+", "_")
        config_name = f"{args.out_subdir}/{run_name}"
        config_path = out_dir / f"{run_name}.py"
        write_text(config_path, config_text, args.force)
        generated_config_paths.add(config_path)

        exp_name = config_name
        launch_lines.append(
            "sh \"$SCRIPT_DIR/train.sh\" "
            f"-d {shlex.quote(args.dataset)} -c {shlex.quote(config_name)} "
            f"-n {shlex.quote(exp_name)} -g {args.num_gpu}"
        )
        rows.append(
            {
                "config_name": config_name,
                "exp_name": exp_name,
                **metadata,
                "seed": args.seed,
                "grid_size": args.grid_size,
                "voxel_max": args.voxel_max,
                "batch_size": args.batch_size,
                "num_gpu": args.num_gpu,
                "epoch": args.epoch,
                "loop": args.loop,
                "total_epoch": args.epoch * args.loop,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "mix_prob": args.mix_prob,
                "wandb_project": args.wandb_project,
                "enable_wandb": args.enable_wandb,
            }
        )

    manifest_path = out_dir / "manifest.csv"
    manifest_fields = list(rows[0])
    # csv.writer requires a file-like object; StringIO keeps writes centralized.
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text(manifest_path, buffer.getvalue(), args.force)

    launch_path = root / "scripts" / f"{args.out_subdir.replace('/', '_')}_launch.sh"
    write_text(launch_path, "\n".join(launch_lines) + "\n", args.force)
    launch_path.chmod(0o755)

    print(f"Wrote {len(rows)} configs to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Launcher: {launch_path}")
    stale_configs = sorted(set(out_dir.glob("*.py")) - generated_config_paths)
    if stale_configs:
        print("Note: existing configs not selected by this invocation were left untouched:")
        for path in stale_configs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
