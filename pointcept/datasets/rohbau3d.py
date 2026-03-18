"""
Rohbau3D Dataset

Author: Pointcept Contributors
"""

import os
import glob
import numpy as np
from collections.abc import Sequence

from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Rohbau3DDataset(DefaultDataset):
    """
    Expected scene directory layout:
    data_root/{split}/site_*/scene_*/
      - coord.npy            (N, 3)
      - color.npy            (N, 3), optional
      - normal.npy           (N, 3), optional
      - intensity.npy        (N,) or (N, 1), optional
      - class.npy            (N,) semantic label, optional in test split
      - segment.npy          (N,) instance label, optional
      - instance.npy         (N,) instance label, optional alias
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
    ]
    NUM_CLASSES = 18

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            data_list += glob.glob(
                os.path.join(self.data_root, split, "site_*", "scene_*")
            )
        return sorted(data_list)

    def get_data_name(self, idx):
        remain, scene_name = os.path.split(self.data_list[idx % len(self.data_list)])
        _, site_name = os.path.split(remain)
        return f"{site_name}-{scene_name}"

    def get_data(self, idx):
        scene_dir = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = dict(name=name, split=split)

        coord_path = os.path.join(scene_dir, "coord.npy")
        if not os.path.exists(coord_path):
            raise FileNotFoundError(f"coord.npy not found in: {scene_dir}")
        coord = np.load(coord_path).astype(np.float32)
        data_dict["coord"] = coord
        num_point = coord.shape[0]

        color_path = os.path.join(scene_dir, "color.npy")
        if os.path.exists(color_path):
            data_dict["color"] = np.load(color_path).astype(np.float32)
        else:
            data_dict["color"] = np.zeros((num_point, 3), dtype=np.float32)

        normal_path = os.path.join(scene_dir, "normal.npy")
        if os.path.exists(normal_path):
            data_dict["normal"] = np.load(normal_path).astype(np.float32)
        else:
            data_dict["normal"] = np.zeros((num_point, 3), dtype=np.float32)

        intensity_path = os.path.join(scene_dir, "intensity.npy")
        if os.path.exists(intensity_path):
            intensity = np.load(intensity_path).reshape([-1, 1]).astype(np.float32)
            data_dict["strength"] = intensity
        else:
            data_dict["strength"] = np.zeros((num_point, 1), dtype=np.float32)

        class_path = os.path.join(scene_dir, "class.npy")
        if os.path.exists(class_path):
            segment = np.load(class_path).reshape([-1]).astype(np.int32)
            if np.any((segment < 0) | (segment >= self.NUM_CLASSES)):
                raise ValueError(
                    f"Invalid class index in {class_path}, expected [0, {self.NUM_CLASSES - 1}]"
                )
        else:
            segment = np.ones(num_point, dtype=np.int32) * -1
        data_dict["segment"] = segment

        instance_path = os.path.join(scene_dir, "instance.npy")
        if not os.path.exists(instance_path):
            instance_path = os.path.join(scene_dir, "segment.npy")
        if os.path.exists(instance_path):
            instance = np.load(instance_path).reshape([-1]).astype(np.int32)
        else:
            instance = np.ones(num_point, dtype=np.int32) * -1
        data_dict["instance"] = instance
        return data_dict


@DATASETS.register_module()
class Rohbauu3DDataset(Rohbau3DDataset):
    """Alias for typo-safe config compatibility."""

