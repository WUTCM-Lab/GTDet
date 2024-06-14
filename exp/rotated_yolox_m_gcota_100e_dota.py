evaluation = dict(
    interval=10, metric='mAP', save_best='auto', dynamic_intervals=[(85, 1)])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
optimizer = dict(
    type='SGD',
    lr=0.00125 / 2,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'le90'
img_scale = (1024, 1024)
model = dict(
    type='RotatedYOLOX',
    input_size=(1024, 1024),
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2),
    bbox_head=dict(
        type='RotatedYOLOXHead',
        num_classes=15,
        in_channels=192,
        feat_channels=192,
        separate_angle=False,
        with_angle_l1=True,
        angle_norm_factor=5,
        edge_swap='le90',
        loss_bbox=dict(
            type='RotatedIoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='RSimOTAAssigner_gau_distance', gau_weight=2.0, center_radius=3.0)),
    test_cfg=dict(
        score_thr=0.01, nms=dict(type='nms_rotated', iou_threshold=0.1)))
dataset_type = 'DOTADataset'
data_root = '/home/server4/datasets/split_ss_dota/'
train_pipeline = [
    dict(type='RMosaic', img_scale=(1024, 1024), pad_val=114.0),
    dict(
        type='PolyRandomAffine',
        version='le90',
        scaling_ratio_range=(0.1, 2),
        bbox_clip_border=False,
        border=(-512, -512)),
    dict(
        type='PolyMixUp',
        version='le90',
        bbox_clip_border=False,
        img_scale=(1024, 1024),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='FilterRotatedAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='DOTADataset',
        version='le90',
        ann_file= data_root +'train/annfiles/',
        img_prefix= data_root + 'train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='RMosaic', img_scale=(1024, 1024), pad_val=114.0),
        dict(
            type='PolyRandomAffine',
            version='le90',
            scaling_ratio_range=(0.1, 2),
            bbox_clip_border=False,
            border=(-512, -512)),
        dict(
            type='PolyMixUp',
            version='le90',
            bbox_clip_border=False,
            img_scale=(1024, 1024),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version='le90'),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterRotatedAnnotations',
            min_gt_bbox_wh=(1, 1),
            keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='DOTADataset',
            version='le90',
            ann_file= data_root + 'train/annfiles/',
            img_prefix= data_root + 'train/images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=True),
        pipeline=[
            dict(type='RMosaic', img_scale=(1024, 1024), pad_val=114.0),
            dict(
                type='PolyRandomAffine',
                version='le90',
                scaling_ratio_range=(0.1, 2),
                bbox_clip_border=False,
                border=(-512, -512)),
            dict(
                type='PolyMixUp',
                version='le90',
                bbox_clip_border=False,
                img_scale=(1024, 1024),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterRotatedAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DOTADataset',
        version='le90',
        ann_file= data_root +'val/annfiles/',
        img_prefix= data_root + 'val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DOTADataset',
        version='le90',
        ann_file= data_root +'val/annfiles/',
        img_prefix= data_root + 'val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
max_epochs = 100
num_last_epochs = 15
interval = 10
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,
        skip_type_keys=('RMosaic', 'PolyRandomAffine', 'PolyMixUp'),
        priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
work_dir = 'work-dir/0403-yolox-m-gcota-100e-dota'
auto_resume = True
gpu_ids = range(0, 2)
load_from = None
fp16 = dict(loss_scale='dynamic')