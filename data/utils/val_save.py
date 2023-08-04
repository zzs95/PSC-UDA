import numpy as np
import logging
import time
import os
import torch
import torch.nn.functional as F

from data.utils.evaluate import Evaluator
from monai.transforms import *
import copy
from data.collate import decollect_scn_data
from monai.data import decollate_batch
def validate_save(cfg,
             model_img,
             model_edge,
             LH_KD,
             dataloader,
             val_metric_logger,
             logger=None,
             pselab_path='',):
    logger.info('Validation')
    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_edge = Evaluator(class_names) if model_edge else None

    post_transforms = Compose([
        EnsureTyped(keys=["pred", ]),
        Invertd(
            keys=["pred",],
            transform=dataloader.dataset.transform,
            orig_keys="edge",
            meta_keys=["pred_meta_dict"],
            orig_meta_keys="edge_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=os.path.join(pselab_path, 'val_output'),
                   output_postfix='',
                   separate_folder=False, resample=True, print_log=False),

    ])
    end = time.time()
    with torch.no_grad():
        for iteration, data_batch_trg in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            data_batch_trg['points'] = data_batch_trg['points'].cuda()
            data_batch_trg['batch_idx'] = data_batch_trg['batch_idx'].cuda()
            data_batch_trg['edge_label'] = data_batch_trg['edge_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()

            data_batch_trg = model_edge(data_batch_trg)
            pred_label_voxel_edge = data_batch_trg['logits'].argmax(1).cpu()

            if model_img != None:
                data_batch_trg = model_img(data_batch_trg)
                data_batch_trg = LH_KD(data_batch_trg)
                pred_label_voxel_img = data_batch_trg['img_pred_logits'].argmax(1).cpu()
                pred_label_voxel_edge = pred_label_voxel_edge | pred_label_voxel_img

            # get original point cloud from before voxelization
            seg_label = data_batch_trg['orig_seg_label']
            points_idx = data_batch_trg['orig_points_idx']
            # loop over batch
            pts_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind].numpy()
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                # case in batch, with different number of sampled point cloud
                curr_seg_label = seg_label[batch_ind]
                pts_idx_next = pts_idx + curr_points_idx.sum()
                pred_label_edge = pred_label_voxel_edge[pts_idx:pts_idx_next].numpy()

                # evaluate
                evaluator_edge.update(pred_label_edge, curr_seg_label)
                # save edge boundary
                pts_idx = pts_idx_next
                label_name = data_batch_trg['label_meta_dict']['filename_or_obj'][batch_ind].split('/')[-1]
                label_name = label_name.replace('.nii.gz', '.npy')
                boundary_npy_path = os.path.join(pselab_path, 'boundary', label_name)

                # update in validation
                pts_range_idx = data_batch_trg['orig_batch_idx'][batch_ind]
                pts = data_batch_trg['points'][pts_range_idx][:, :3]
                pts_top_boundary = pts.max(dim=0)[0].cpu().numpy()[None]
                pts_bottom_boundary = pts.min(dim=0)[0].cpu().numpy()[None]
                pts_boundary = np.concatenate([pts_bottom_boundary, pts_top_boundary], axis=0)
                np.save(boundary_npy_path, pts_boundary)
            # Saved pseudo label data topselab_path, batch_size is 1
            edge_indices = data_batch_trg['edge_indices'].cpu()
            edge_arr = copy.deepcopy(data_batch_trg['edge']).cpu()
            seg_prob_out_img = decollect_scn_data(torch.zeros_like(edge_arr), edge_indices, pred_label_voxel_edge )
            edge_arr.array = seg_prob_out_img.cpu()
            data_batch_trg['pred'] = edge_arr
            new_dict = {}
            for k in ['edge', 'pred']:
                new_dict[k] = data_batch_trg[k]
            # save in original space: 1x1x1
            d = [post_transforms(i) for i in decollate_batch(new_dict)]

            seg_loss_edge = F.cross_entropy(data_batch_trg['logits'], data_batch_trg['edge_label'])
            val_metric_logger.update(seg_loss_edge=seg_loss_edge)
            val_metric_logger.update(loss_total_trg=data_batch_trg['loss'],)
            if model_img != None:
                val_metric_logger.update(loss_pred_img_trg=data_batch_trg['loss_pred_img_trg'],)
                val_metric_logger.update(loss_seg_pts_trg=data_batch_trg['loss_seg_pts_trg'],)
                val_metric_logger.update(loss_seg_fuse_trg=data_batch_trg['loss_seg_fuse_trg'],)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_edge=evaluator_edge.overall_iou)
        eval_list = [('Edge', evaluator_edge)]
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

