"""
Main Testing Script

Author: Lukas Rauch
"""

import sys
sys.path.insert(0,'Pointcept')

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):

    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    # args = default_argument_parser().parse_args()
    # cfg = default_config_parser(args.config_file, args.options)


    my_args = [
    "--config-file", "Pointcept/configs/rohbau3d/semseg-pt-v2m1-1-base.py",
    "--num-gpus", "1",
    "--options", 
        "save_path=Pointcept/exp/big_run", 
        "weight=Pointcept/exp/big_run/model/model_best.pth",

    ]

    args = default_argument_parser().parse_args(my_args)
    cfg = default_config_parser(args.config_file, args.options)

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

    print("Done.")
