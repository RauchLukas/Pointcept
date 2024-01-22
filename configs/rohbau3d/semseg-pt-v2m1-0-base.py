weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/rohbau3d"
num_worker = 16  # total worker in all gpu
batch_size = 1  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 1  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 1  # sche total eval & checkpoint epoch


sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# dataset settings
dataset_type = "Rohbau3DDataset"
data_root = "data/rohbau3d"
ignore_index = -1
num_classes = 11

# scheduler settings
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=num_classes,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
        'none',
        'ceiling',
        'floor',
        'wall',
        'beam',
        'column',
        'window',
        'door',
        'stairs',
        'equipment',
        'installation',
    ],
    train=dict(
        type = dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_discrete_coord=True,
            ),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=["coord", "color"],
            ),
        ],
        test_mode=False,
        # ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_discrete_coord=True,
            ),
            # dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
        # ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_discrete_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "color"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
