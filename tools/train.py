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


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    wandb_cfg = cfg.pop("wandb", None)
    if wandb_cfg: 
        if wandb_cfg.track: 
            import wandb
            print("Tracking Run with W&B, projectname: ", wandb_cfg.project)

            settings = wandb.Settings(disable_git=True)

            print("[DEBUG] W&B sync from tensorboard save path: ", cfg.save_path)
            wandb.tensorboard.patch(root_logdir=cfg.save_path)

            wandb.init(
                project=wandb_cfg.project,
                notes=wandb_cfg.notes,
                tags=wandb_cfg.tags,
                config=cfg,
                settings=settings
            )

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()