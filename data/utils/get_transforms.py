from monai import transforms

def get_transforms_fixsize(mode='test', cfg=None):
    img_keys = ["image", "label"]
    edge_keys = ['edge', 'pselab_edge']
    load_trans = [transforms.LoadImaged(keys=img_keys + edge_keys, image_only=False),
                  transforms.EnsureChannelFirstd(keys=img_keys + edge_keys),
                  transforms.Orientationd(keys=img_keys + edge_keys, axcodes="RAS"), ]

    if cfg['mod'] == 'CT':
        norm_trans = [transforms.ScaleIntensityRanged(keys=["image"],
                                               a_min=cfg['a_min'],
                                               a_max=cfg['a_max'],
                                               b_min=-1,
                                               b_max=1,
                                               clip=True),]
    elif cfg['mod'] == 'MRI':
        norm_trans = [transforms.ScaleIntensityRanged(keys=["image"],
                                               a_min=cfg['a_min'],
                                               a_max=cfg['a_max'],
                                               b_min=-1,
                                               b_max=1,
                                               clip=True),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=95, b_min=-1.0, b_max=1.0, clip=True)]
    else:
        norm_trans = [transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=95, b_min=-1.0, b_max=1.0, clip=True),]
    resize_trans = [
        transforms.Resized(keys=['image'],
                   spatial_size=[cfg['roi_x'], cfg['roi_y'], cfg['roi_z']],
                   mode=["trilinear"]),
        transforms.Resized(keys=['label'],
                   spatial_size=[cfg['roi_x'], cfg['roi_y'], cfg['roi_z']],
                   mode=["nearest"]),
        transforms.Resized(keys=edge_keys,
                   spatial_size=[cfg['roi_x'] * cfg['space_x'],
                                 cfg['roi_y'] * cfg['space_y'],
                                 cfg['roi_z'] * cfg['space_z']],
                   mode=["nearest"]*len(edge_keys)),]
    rand_trans = [
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=cfg['RandScaleIntensityd_prob']),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=cfg['RandShiftIntensityd_prob']),
            ]
    if 'train' in mode:
        return transforms.Compose(
                load_trans + norm_trans + rand_trans + resize_trans + [
                transforms.ToTensord(keys=img_keys+edge_keys, track_meta=True),
                transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]

            )
    if 'val' in mode:
        return transforms.Compose(
            load_trans + norm_trans + resize_trans + [
                transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]
                )
    if 'test' in mode:
        return transforms.Compose(
            load_trans + norm_trans + [
                transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]
            )
def get_transforms_fixspacing(mode='test', cfg=None):
    img_keys = ["image", "label"]
    edge_keys = ['edge', 'pselab_edge']
    load_trans = [transforms.LoadImaged(keys=img_keys + edge_keys, image_only=False),
                  transforms.EnsureChannelFirstd(keys=img_keys + edge_keys),
                  transforms.Orientationd(keys=img_keys + edge_keys, axcodes="RAS"), ]
    if cfg['mod'] == 'CT':
        norm_trans = [transforms.ScaleIntensityRanged(keys=["image"],
                                               a_min=cfg['a_min'],
                                               a_max=cfg['a_max'],
                                               b_min=-1,
                                               b_max=1,
                                               clip=True),]
    elif cfg['mod'] == 'MRI':
        norm_trans = [transforms.ScaleIntensityRanged(keys=["image"],
                                               a_min=cfg['a_min'],
                                               a_max=cfg['a_max'],
                                               b_min=-1,
                                               b_max=1,
                                               clip=True),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=95, b_min=-1.0, b_max=1.0, clip=True)]
    else:
        norm_trans = [transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=95, b_min=-1.0, b_max=1.0, clip=True),]
    # TESTSET keeps original size with spacing 1
    resize_trans = [transforms.Spacingd(keys=['image'], pixdim=[cfg['space_x'], cfg['space_y'], cfg['space_z']],
                                           mode=["bilinear"]),
                    transforms.Spacingd(keys=['label'], pixdim=[cfg['space_x'], cfg['space_y'], cfg['space_z']],
                                           mode=["nearest"]),
                    transforms.Spacingd(keys=edge_keys, pixdim=[1,1,1], mode=["nearest"]*len(edge_keys)),]
    if 'train' in mode:
        return transforms.Compose(
            load_trans + norm_trans + resize_trans + [
                    transforms.RandScaleIntensityd(keys="image",
                                                   factors=0.1,
                                                   prob=cfg['RandScaleIntensityd_prob']),
                    transforms.RandShiftIntensityd(keys="image",
                                                   offsets=0.1,
                                                   prob=cfg['RandShiftIntensityd_prob']),
                    transforms.ToTensord(keys=img_keys+edge_keys, track_meta=True),
                    transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]
            )
    if 'val' in mode:
        return transforms.Compose(
            load_trans + norm_trans + resize_trans + [
                    transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]
            )
    if 'test' in mode:
        return transforms.Compose(
            load_trans + norm_trans + [
                transforms.EnsureTyped(keys=img_keys+edge_keys),
                ]
            )
