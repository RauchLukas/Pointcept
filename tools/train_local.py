"""
Main Training Script

Author: Lukas Ruach
"""
import sys
sys.path.insert(0,'Pointcept')

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import Trainer
from pointcept.engines.launch import launch

import wandb

import torch 
import random
import numpy as np

seed = 42

if seed:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"[WARNING] Fixed seed: Torch → {seed}")
    print(f"[WARNING] Fixed seed: Random → {seed}")
    print(f"[WARNING] Fixed seed: NumPy → {seed}")


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    # python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}

    my_args = [
        "--config-file", "Pointcept/configs/rohbau3d/semseg-pt-v3m1-0-base.py",
        "--num-gpus", "1",
        "--options", "save_path=Pointcept/exp/pt3_base",
            "resume=False",
    ]

    args = default_argument_parser().parse_args(my_args)
    cfg = default_config_parser(args.config_file, args.options)
    
    wandb.tensorboard.patch(root_logdir="Pointcept/exp/pt3_base")
    wandb.init(project="PT3_base", sync_tensorboard=True)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

    wandb.finish()

if __name__ == "__main__":
    main()

    print("Done.")
