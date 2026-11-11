#!/bin/sh
set -e
# Location-independent: invoke from anywhere (e.g. the directory above
# Pointcept). train.sh itself cd's to the Pointcept root, so data_root
# (../data) resolves correctly regardless of where you run this from.
SCRIPT_DIR=$(dirname "$0")

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c pointcept_hp_test_main/batch_size12__grid_size0p04__voxel_max80000__loop16 -n pointcept_hp_test_main/batch_size12__grid_size0p04__voxel_max80000__loop16 -g 4
