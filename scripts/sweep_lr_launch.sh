#!/bin/sh
set -e
# Location-independent: invoke from anywhere (e.g. the directory above
# Pointcept). train.sh itself cd's to the Pointcept root, so data_root
# (../data) resolves correctly regardless of where you run this from.
SCRIPT_DIR=$(dirname "$0")

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p004__weight_decay0p01 -n sweep_lr/lr0p004__weight_decay0p01 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p004__weight_decay0p05 -n sweep_lr/lr0p004__weight_decay0p05 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p004__weight_decay0p1 -n sweep_lr/lr0p004__weight_decay0p1 -g 4
