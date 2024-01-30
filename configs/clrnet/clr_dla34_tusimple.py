net = dict(
    type='Detector',
)

backbone = dict(
    type='DLAWrapper',
    dla='dla34',
    pretrained=True,
)

num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)

heads = dict(
    type='CLRHead', num_priors=192, refine_layers=3, fc_hidden_dim=64, sample_points=36
)

iou_loss_weight = 2.0
conf_loss_weight = 6.0
cls_loss_weight = 1.5
xyt_loss_weight = 0.1
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/dla34_tusimple"

neck = dict(
    type='PAFPN',
    in_channels=[128, 256, 512],
    out_channels=64,
    num_outs=3,
    attention=False,
)

test_parameters = dict(conf_threshold=0.40, nms_thres=50, nms_topk=max_lanes)

epochs = 70
batch_size = 40

optimizer = dict(type='AdamW', lr=1.0e-3)  # 3e-4 for batchsize 8
total_iter = (3626 // batch_size + 1) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = 10

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 0

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(
                name='Resize',
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(
                name='MultiplyAndAddToBrightness',
                parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                p=0.6,
            ),
            dict(name='AddToHueAndSaturation', parameters=dict(value=(-10, 10)), p=0.7),
            dict(
                name='OneOf',
                transforms=[
                    dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                    dict(name='MedianBlur', parameters=dict(k=(3, 5))),
                ],
                p=0.2,
            ),
            dict(
                name='Affine',
                parameters=dict(
                    translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                    rotate=(-10, 10),
                    scale=(0.8, 1.2),
                ),
                p=0.7,
            ),
            dict(
                name='Resize',
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(
                name='Resize',
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
        ],
        training=False,
    ),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/tusimple'
dataset_type = 'TuSimple'
test_json_file = 'data/tusimple/test_label.json'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
)

workers = 0
log_interval = 100
# seed = 0
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
lane_classes = 4
# There are 5 classes in label, but actually only 4 classes in training and testing
# 0: background, 1: continuous, 2: dashed, 3: double, 4: unknown, 5: solid dashed
# Note background is not included in label, class number in label starts from 1
# Merge 'continuous' and 'unknown' to 'continuous': (1, 4) -> 1
# Decrease remaining classes number, 'solid dashed': 5 -> 4
cls_merge = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
vis_cls_mapping = {
    1: {'name': 'continuous', 'color': (0, 255, 0)},
    2: {'name': 'dashed', 'color': (0, 255, 255)},
    3: {'name': 'double', 'color': (255, 255, 0)},
    4: {'name': 'solid dashed', 'color': (0, 0, 255)},
}
