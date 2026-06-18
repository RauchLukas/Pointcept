#!/bin/sh
set -e
# Location-independent: invoke from anywhere (e.g. the directory above
# Pointcept). train.sh itself cd's to the Pointcept root, so data_root
# (../data) resolves correctly regardless of where you run this from.
SCRIPT_DIR=$(dirname "$0")

sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p0003 -n sweep_lr/lr0p0003 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p0006 -n sweep_lr/lr0p0006 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p001 -n sweep_lr/lr0p001 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p003 -n sweep_lr/lr0p003 -g 4
sh "$SCRIPT_DIR/train.sh" -d rohbau3d -c sweep_lr/lr0p006 -n sweep_lr/lr0p006 -g 4
