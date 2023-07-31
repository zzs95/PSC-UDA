from torch.utils.data.sampler import RandomSampler, BatchSampler
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data.dataloader import DataLoader
from monai.data.dataloader import DataLoader
from monai.data.utils import list_data_collate
# from monai.data.utils import worker_init_fn

from yacs.config import CfgNode as CN

from common.utils.torch_util import worker_init_fn
from data.collate import get_collate_scn
from common.utils.sampler import IterationBasedBatchSampler
from data.kidney_dataloader import get_KiTS, get_CHAOS, get_TESTSET

def build_dataloader(cfg, mode='train', domain='source', start_iteration=0, max_iteration=None,
                     pselab_path=None, crop_pselab=False):
    dataset_cfg = cfg.get('DATASET_' + domain.upper())
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode

    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    if dataset_cfg.TYPE == 'KITS':
        # source domain
        dataset = get_KiTS(split, dataset_kwargs)
    elif 'CHAOS' in dataset_cfg.TYPE:
        dataset = get_CHAOS(split, dataset_kwargs)
    elif 'BTCV' in dataset_cfg.TYPE or 'AMOS' in dataset_cfg.TYPE:
        dataset = get_CHAOS(split, dataset_kwargs)
    elif dataset_cfg.TYPE == 'TESTSET':
        dataset = get_TESTSET(split, dataset_kwargs)

    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))
    dataset.class_names = ['no_kidney', 'kidney'] # for 2 classes

    collate_fn = get_collate_scn(is_train, point_num=dataset_cfg['point_num'], crop_pselab=crop_pselab, pselab_path=pselab_path)
    batch_size = cfg[mode.upper()].BATCH_SIZE
    if is_train:
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=cfg.DATALOADER.DROP_LAST)
        max_iteration = cfg.SCHEDULER.MAX_ITERATION if max_iteration==None else max_iteration
        batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iteration, start_iteration)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    return dataloader
