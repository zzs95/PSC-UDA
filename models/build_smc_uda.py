from models.arch_smc_uda import NetImgSeg, NetEdgeSeg, xModalKD
# from models_dscml.Discriminator import FCDiscriminator
# from models_dscml.Discriminator import FCDiscriminator
from models.metric import SegIoU


def build_model_img(cfg):
    model = NetImgSeg(num_classes=cfg.MODEL_IMG.NUM_CLASSES,
                     backbone_img=cfg.MODEL_IMG.TYPE,
                     cfg=cfg,
                     train_src=cfg.MODEL_IMG.SRC_HEAD
                     )
    return model


def build_model_edge(cfg):
    model = NetEdgeSeg(num_classes=cfg.MODEL_EDGE.NUM_CLASSES,
                     backbone_edge=cfg.MODEL_EDGE.TYPE,
                     cfg=cfg,
                     )
    train_metric_src = SegIoU(cfg.MODEL_EDGE.NUM_CLASSES, name='seg_iou_src', ignore_index=cfg.TRAIN.ignore_label)
    train_metric_trg = SegIoU(cfg.MODEL_EDGE.NUM_CLASSES, name='seg_iou_trg', ignore_index=cfg.TRAIN.ignore_label)
    return model, train_metric_src, train_metric_trg

def build_KD(cfg):
    model = xModalKD(cfg)

    return model