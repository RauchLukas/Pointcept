#!/bin/sh
set -e
cd "$(dirname "$0")/.." || exit

sh scripts/train.sh -d rohbau3d -c sweep_lr/lr0p0003 -n sweep_lr/lr0p0003 -g 4
sh scripts/train.sh -d rohbau3d -c sweep_lr/lr0p0006 -n sweep_lr/lr0p0006 -g 4
sh scripts/train.sh -d rohbau3d -c sweep_lr/lr0p001 -n sweep_lr/lr0p001 -g 4
sh scripts/train.sh -d rohbau3d -c sweep_lr/lr0p003 -n sweep_lr/lr0p003 -g 4
sh scripts/train.sh -d rohbau3d -c sweep_lr/lr0p006 -n sweep_lr/lr0p006 -g 4
