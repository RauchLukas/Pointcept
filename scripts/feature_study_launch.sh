#!/bin/sh
set -eu
# Generated point-feature study; runs are sequential to avoid GPU overlap.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/rgb -n feature_study/rgb -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/intensity -n feature_study/intensity -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/normals -n feature_study/normals -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/rgb_intensity -n feature_study/rgb_intensity -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/rgb_normals -n feature_study/rgb_normals -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/intensity_normals -n feature_study/intensity_normals -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c feature_study/rgb_intensity_normals -n feature_study/rgb_intensity_normals -g 4
