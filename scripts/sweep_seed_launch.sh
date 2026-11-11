#!/bin/sh
set -e
# Location-independent: invoke from anywhere (e.g. the directory above
# Pointcept). train.sh itself cd's to the Pointcept root, so data_root
# (../data) resolves correctly regardless of where you run this from.
SCRIPT_DIR=$(dirname "$0")

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep1 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep1 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep2 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep2 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep3 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep3 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep4 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep4 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep5 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep5 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep6 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep6 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep7 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep7 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep8 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep8 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep9 -n sweep_seed/batch_size12__grid_size0p04__voxel_max80000__loop4__seed42__rep9 -g 4
