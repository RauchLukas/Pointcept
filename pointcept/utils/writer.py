
from pathlib import Path
from typing import Union
from tensorboardX import SummaryWriter

from pointcept.utils.logger import get_root_logger
from pointcept.utils.config import Config





class WandBWriter(object):

    def __init__(
        self,
        save_path: str = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
        wandb_entity: str = None,
        wandb_config: Union[dict, Config] = None,
        wandb_group: str = None,
        wandb_name: str = None,
        wandb_id: str = None,
    ):

        self.logger = get_root_logger()

        self.wandb = None

        if use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config,
                group=wandb_group,
                # name=wandb_name if wandb_name else Path(save_path).name,
                dir=save_path,
                id=wandb_id,
            )


    def add_scalar(self, tag: str, val: float, step: float = None, commit: bool = False):
        if self.wandb:
            self.wandb.log({tag: val}, step=step, commit=commit)

    def close(self):
        if self.wandb:
            self.wandb.finish()
      



