"""SMC-UDA experiments configuration"""
import os.path as osp

from common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []
_C.TRAIN.ignore_label = -100
_C.TRAIN.AMP = False

# ---------------------------------------------------------------------------- #
# loss options
# ---------------------------------------------------------------------------- #
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.lambda_xm = 0.05
_C.TRAIN.LOSS.lambda_lovasz = 1

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()
_C.DATASET_SOURCE.point_num = 30000

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.TEST = tuple()
_C.DATASET_TARGET.point_num = 30000

# KITS
_C.DATASET_SOURCE.KITS = CN()
_C.DATASET_SOURCE.KITS.preprocess_dir = ''
_C.DATASET_SOURCE.KITS.merge_classes = True
# 3D augmentation
_C.DATASET_SOURCE.KITS.augmentation = CN()
_C.DATASET_SOURCE.KITS.augmentation.mod = 'CT'
_C.DATASET_SOURCE.KITS.augmentation.space_x = 2
_C.DATASET_SOURCE.KITS.augmentation.space_y = 2
_C.DATASET_SOURCE.KITS.augmentation.space_z = 2
_C.DATASET_SOURCE.KITS.augmentation.roi_x = 80
_C.DATASET_SOURCE.KITS.augmentation.roi_y = 160
_C.DATASET_SOURCE.KITS.augmentation.roi_z = 160
_C.DATASET_SOURCE.KITS.augmentation.RandScaleIntensityd_prob = 0.4
_C.DATASET_SOURCE.KITS.augmentation.RandShiftIntensityd_prob = 0.4
_C.DATASET_SOURCE.KITS.augmentation.a_min = -100
_C.DATASET_SOURCE.KITS.augmentation.a_max = 300

# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.KITS = CN(_C.DATASET_SOURCE.KITS)
_C.DATASET_TARGET.AMOS = CN(_C.DATASET_SOURCE.KITS)

_C.DATASET_TARGET.CHAOS_T1InPhase = CN(_C.DATASET_SOURCE.KITS)
_C.DATASET_TARGET.CHAOS_T1InPhase.augmentation.mod = 'MRI'
_C.DATASET_TARGET.CHAOS_T1InPhase.augmentation.a_min = 0
_C.DATASET_TARGET.CHAOS_T1InPhase.augmentation.a_max = 1200
_C.DATASET_TARGET.CHAOS_T2SPIR = CN(_C.DATASET_SOURCE.KITS)
_C.DATASET_TARGET.CHAOS_T2SPIR.augmentation.mod = 'MRI'
_C.DATASET_TARGET.CHAOS_T2SPIR.augmentation.a_min = 0
_C.DATASET_TARGET.CHAOS_T2SPIR.augmentation.a_max = 1200

_C.DATASET_TARGET.TESTSET = CN(_C.DATASET_SOURCE.KITS)
_C.DATASET_TARGET.TESTSET.augmentation.mod = 'MRI'
_C.DATASET_TARGET.TESTSET.augmentation.a_min = None
_C.DATASET_TARGET.TESTSET.augmentation.a_max = None

# ---------------------------------------------------------------------------- #
# Model IMG
# ---------------------------------------------------------------------------- #
_C.MODEL_IMG = CN()
_C.MODEL_IMG.TYPE = ''
_C.MODEL_IMG.CKPT_PATH = ''
_C.MODEL_IMG.NUM_CLASSES = 2
_C.MODEL_IMG.SRC_HEAD = False
# ---------------------------------------------------------------------------- #
# FCNResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_IMG.FCNResNet34 = CN()
_C.MODEL_IMG.FCNResNet34.pretrained = True
_C.MODEL_IMG.FCNResNet34.hiden_size = 128

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_EDGE = CN()
_C.MODEL_EDGE.TYPE = ''
_C.MODEL_EDGE.CKPT_PATH = ''
_C.MODEL_EDGE.NUM_CLASSES = 2
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_EDGE.SCN = CN()
_C.MODEL_EDGE.SCN.input_dims = 4
_C.MODEL_EDGE.SCN.hiden_size = 128  # number of unet features (multiplied in each layer)
_C.MODEL_EDGE.SCN.scale_list = [2, 4, 8, 16, 16, 16]


# ----------------------------------------------------------------------------- #
# KD options
# ----------------------------------------------------------------------------- #
_C.MODEL_KD = CN(_C.MODEL_EDGE)
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('/home/brownradai/Projects/kidney/smc_uda/output/@')
