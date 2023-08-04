#!/usr/bin/env python
import os
import os.path as osp
import argparse
import time
import socket
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import seed_everything
from common.solver.build import build_optimizer_base, build_scheduler
from common.utils.checkpoint import CheckpointerV2
from common.utils.logger import setup_logger
from common.utils.metric_logger import MetricLogger
from models.build_smc_uda import build_model_img, build_model_edge, build_KD
from data.build import build_dataloader
from data.utils.val_save import validate_save
from utils.file_and_folder_operations import *
from data.utils.edge_utils import save_edge_range
import numpy as np

root_path = os.getcwd()

def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger

def train(cfg, output_dir='', run_name='', logger=None):
    logger.info('train')
    seed_everything(cfg.RNG_SEED)
    # build edge model
    model_edge, train_metric_src, train_metric_trg = build_model_edge(cfg)
    logger.info('Build EDGE model:\n{}'.format(str(model_edge)))
    # num_params = sum(param.numel() for param in model_edge.parameters())
    # print('#Parameters: {:.2e}'.format(num_params))
    # build img model
    model_img = build_model_img(cfg)
    logger.info('Build IMG model:\n{}'.format(str(model_img)))
    # num_params = sum(param.numel() for param in model_img.parameters())
    # print('#Parameters: {:.2e}'.format(num_params))
    # build cross modal learning head
    LH_KD = build_KD(cfg)
    logger.info('Build KD model learning head:\n{}'.format(str(LH_KD)))
    # num_params = sum(param.numel() for param in LH_KD.parameters())
    # print('#Parameters: {:.2e}'.format(num_params))

    model_img = model_img.cuda()
    model_edge = model_edge.cuda()
    LH_KD = LH_KD.cuda()

    # build optimizer
    optimizer_img = build_optimizer_base(cfg, model_img)
    optimizer_edge = build_optimizer_base(cfg, model_edge)
    optimizer_KD = build_optimizer_base(cfg, LH_KD)

    # build lr scheduler
    scheduler_img = build_scheduler(cfg, optimizer_img)
    scheduler_edge = build_scheduler(cfg, optimizer_edge)
    scheduler_KD = build_scheduler(cfg, optimizer_KD)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_img = CheckpointerV2(model_img,
                                     optimizer=optimizer_img,
                                     scheduler=scheduler_img,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_img',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_img = checkpointer_img.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_edge = CheckpointerV2(model_edge,
                                     optimizer=optimizer_edge,
                                     scheduler=scheduler_edge,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_edge',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_edge = checkpointer_edge.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_KD = CheckpointerV2(LH_KD,
                                     optimizer=optimizer_KD,
                                     scheduler=scheduler_KD,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_KD',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_KD = checkpointer_KD.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None
    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_edge.get('iteration', 0)
    pselab_path_tmp = output_dir + '/val_pselab'
    maybe_rmtree(pselab_path_tmp)
    val_period = cfg.VAL.PERIOD
    if start_iteration >= val_period:
        last_pselab_path = output_dir + '/val_result_' + str(start_iteration)
        shutil.copytree(last_pselab_path, pselab_path_tmp)
    else:
        maybe_mkdir_p(pselab_path_tmp)
    cur_pselab_path = output_dir + '/val_result_' + str(start_iteration + val_period)
    cur_boundary_path = os.path.join(cur_pselab_path, 'boundary',)
    maybe_mkdir_p(cur_boundary_path)
    # build data loader
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration, crop_pselab=True,
                                            max_iteration=cfg.SCHEDULER.MAX_ITERATION , pselab_path=pselab_path_tmp,)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration, crop_pselab=True,
                                            max_iteration=cfg.SCHEDULER.MAX_ITERATION , pselab_path=pselab_path_tmp,)
    val_dataloader = build_dataloader(cfg, mode='val', domain='target', crop_pselab=True, pselab_path=pselab_path_tmp,) if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        'edge': checkpoint_data_edge.get(best_metric_name, None)
    }
    best_metric_iter = {'img': -1, 'edge': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_src, train_metric_trg])
    val_metric_logger = MetricLogger(delimiter='  ')


    def setup_train():
        # set training mode
        model_img.train()
        LH_KD.train()
        model_edge.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_img.eval()
        LH_KD.eval()
        model_edge.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.AMP:
        scaler = torch.cuda.amp.GradScaler()
    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)

    for iteration in range(start_iteration, max_iteration):
        data_time = time.time() - end
        optimizer_img.zero_grad()
        optimizer_edge.zero_grad()
        optimizer_KD.zero_grad()
        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        data_batch_src['points'] = data_batch_src['points'].cuda()
        data_batch_src['batch_idx'] = data_batch_src['batch_idx'].cuda()
        data_batch_src['edge_label'] = data_batch_src['edge_label'].cuda()
        data_batch_src['img'] = data_batch_src['img'].cuda()
        save_edge_range(data_batch_src, cur_boundary_path)

        if cfg.TRAIN.AMP:
            with torch.cuda.amp.autocast():
                data_batch_src = model_edge.forward(data_batch_src, src=True)
                data_batch_src = model_img.forward(data_batch_src, src=True)
                data_batch_src = LH_KD.forward(data_batch_src, src=True)
            scaler.scale(data_batch_src['loss'] ).backward()
        else:
            data_batch_src = model_edge.forward(data_batch_src, src=True)
            data_batch_src = model_img.forward(data_batch_src, src=True)
            data_batch_src = LH_KD.forward(data_batch_src, src=True)
            (data_batch_src['loss']).backward()

        with torch.no_grad():
            train_metric_src.update_dict(data_batch_src)
            train_metric_logger.update(loss_total_src=data_batch_src['loss'],)
            train_metric_logger.update(loss_pred_edge_src=data_batch_src['loss_pred_edge_src'],)
            train_metric_logger.update(loss_pred_img_src=data_batch_src['loss_pred_img_src'],)
            train_metric_logger.update(loss_seg_fuse_src=data_batch_src['loss_seg_fuse_src'],)
            train_metric_logger.update(loss_seg_pts_src=data_batch_src['loss_seg_pts_src'],)
        del data_batch_src

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        _, data_batch_trg = train_iter_trg.__next__()
        data_batch_trg['points'] = data_batch_trg['points'].cuda()
        data_batch_trg['batch_idx'] = data_batch_trg['batch_idx'].cuda()
        data_batch_trg['edge_label'] = data_batch_trg['edge_label'].cuda()
        data_batch_trg['img'] = data_batch_trg['img'].cuda()

        if cfg.TRAIN.AMP:
            with torch.cuda.amp.autocast():
                data_batch_trg = model_edge(data_batch_trg)
                data_batch_trg = model_img(data_batch_trg)
                data_batch_trg = LH_KD(data_batch_trg)
            # backward
            scaler.scale(data_batch_trg['loss'] ).backward()
        else:
            data_batch_trg = model_edge(data_batch_trg)
            data_batch_trg = model_img(data_batch_trg)
            data_batch_trg = LH_KD(data_batch_trg)
            # backward
            (data_batch_trg['loss'] ).backward()
        with torch.no_grad():
            train_metric_trg.update_dict(data_batch_trg)
            train_metric_logger.update(loss_total_trg=data_batch_trg['loss'],)
            train_metric_logger.update(loss_pred_img_trg=data_batch_trg['loss_pred_img_trg'],)
            train_metric_logger.update(loss_seg_pts_trg=data_batch_trg['loss_seg_pts_trg'],)
            train_metric_logger.update(loss_seg_fuse_trg=data_batch_trg['loss_seg_fuse_trg'],)
        # del data_batch_trg
        # ---------------------------------------------------------------------------- #
        # Optimizer
        # ---------------------------------------------------------------------------- #
        if cfg.TRAIN.AMP:
            scaler.step(optimizer_img)
            scaler.step(optimizer_edge)
            scaler.step(optimizer_KD)
            scaler.update()
        else:
            optimizer_img.step()
            optimizer_edge.step()
            optimizer_KD.step()
        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_edge.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_edge['iteration'] = cur_iter
            checkpoint_data_edge[best_metric_name] = best_metric['edge']
            checkpointer_edge.save('model_edge_{:06d}'.format(cur_iter), **checkpoint_data_edge)
            checkpointer_img.save('model_img_{:06d}'.format(cur_iter), **checkpoint_data_edge)
            checkpointer_KD.save('model_KD_{:06d}'.format(cur_iter), **checkpoint_data_edge)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()
            validate_save(cfg,
                     model_img,
                     model_edge,
                     LH_KD,
                     val_dataloader,
                     val_metric_logger,
                     logger=logger,
                     pselab_path=cur_pselab_path,)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['edge']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()
            maybe_rmtree(pselab_path_tmp)
            shutil.copytree(cur_pselab_path, pselab_path_tmp)

            cur_pselab_path = output_dir + '/val_result_' + str(cur_iter + val_period)
            cur_boundary_path = os.path.join(cur_pselab_path, 'boundary', )
            maybe_mkdir_p(cur_boundary_path)
        scheduler_img.step()
        scheduler_edge.step()
        end = time.time()

    for modality in ['edge']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                    best_metric_iter[modality]))

    return

def parse_args():
    parser = argparse.ArgumentParser(description='Edge testing')
    parser.add_argument(
        '--set_name',
        dest='set_name',
        default='t2',
        # default='AMOS',
        # default='t1in',
        help='set name',
        type=str,
    )
    parser.add_argument(
        '--exp_name',
        dest='exp_name',
        default='public',
        help='exp name',
        type=str,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_name = args.set_name
    exp_name = args.exp_name
    args.config_file = root_path+'/config/'+exp_name+'/'+set_name+'.yaml'
    from common.config import purge_cfg
    from config.base_config import cfg
    cfg.merge_from_file(args.config_file)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.split('config/')[-1])
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('SMC_UDA', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    train(cfg, output_dir, run_name, logger)



if __name__ == '__main__':
    main()
