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


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    # python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}

    my_args = [
        "--config-file", "Pointcept/configs/rohbau3d/semseg-pt-v2m1-1-base.py",
        "--num-gpus", "1",
        "--options", "save_path=Pointcept/exp/big_run",
            "resume=True", 
            "weight=Pointcept/exp/big_run/model/model_best.pth"
    ]

    args = default_argument_parser().parse_args(my_args)
    cfg = default_config_parser(args.config_file, args.options)
    
    wandb.tensorboard.patch(root_logdir="./exp/Rohbau3D/semseg-pt-v2m2-0-base-01")
    wandb.init(project="RB3D_BIG_RUN", sync_tensorboard=True)

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
