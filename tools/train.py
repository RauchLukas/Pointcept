"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

import torch
import wandb
wandb_grup_id = str(wandb.util.generate_id())

def main_worker(cfg):
    cfg = default_setup(cfg)

    machine_rank = torch.distributed.get_rank()
    if cfg["wandb"] and machine_rank == 0: 
        wandb.init(
            project=cfg.wandb["project"],
            config=dict(cfg),
            notes=cfg.wandb["note"],
            # group=cfg.wandb["experiment"],
        )

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    # wandb.tensorboard.patch(root_logdir="./exp/s3dis/semseg-pt-v2m2-0-base-01")
    # wandb.init(project="Scanner-MonacumOune-s3dis", sync_tensorboard=True)

    if cfg["wandb"]: 
        print("[DEBUG] tracking PyTorch with W&B")
        
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    print(">>>>>>>>> TRAIN.py <<<<<<<<<")
    main()