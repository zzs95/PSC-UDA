import copy
from monai import data
import numpy as np
from data.utils.get_transforms import get_transforms_fixsize # same size
from data.utils.get_transforms import get_transforms_fixspacing # different size with spacing 1
from utils.file_and_folder_operations import *
import torch
# torch.multiprocessing.set_start_method('fork', force=True)
# torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def get_public_set(mode, domain, cfg):
    data_dir = cfg['preprocess_dir']
    images = []
    segs = []
    edges = []

    imagestr = join(data_dir, 'imagesTr')
    labelsTr = join(data_dir, 'labelsTr')
    edgelabelstr = join(data_dir, 'edgelabelsTr') # with annotation
    img_files = subfiles(imagestr, join=False)
    for i, img_file in enumerate(img_files):
        img_path = join(imagestr, img_file)
        images.append(img_path)
        label_file = img_file[:-12] + '.nii.gz'
        label_path = join(labelsTr, label_file)
        segs.append(label_path)
        edge_path = join(edgelabelstr, label_file) # with annotation
        edges.append(edge_path)
    pselab_edges = copy.deepcopy(edges)
    datalist_json = [
        {"image": img, "label": seg, 'edge': edge, 'pselab_edge': pselab_edge, 'domain': domain} for img, seg, edge, pselab_edge in
        zip(images, segs, edges, pselab_edges)
    ]
    ds_transforms = get_transforms_fixsize(mode, cfg['augmentation'])
    ds = data.Dataset(data=datalist_json, transform=ds_transforms)

    return ds

def get_TESTSET(mode, cfg, ):
    # without edgelabels, do not split train/test/val
    # mode = 'test'
    data_dir = cfg['preprocess_dir']
    images = []
    edges = []

    imagestr = join(data_dir, 'imagesTr')
    edgelabelstr = join(data_dir, 'edgesTr')  # inference, without annotation
    img_files = subfiles(imagestr, join=False)
    for i, img_file in enumerate(img_files):
        img_path = join(imagestr, img_file)
        images.append(img_path)
        label_file = img_file[:-12] + '.nii.gz' # remove '_0000.nii.gz'
        edge_path = join(edgelabelstr, label_file)
        edges.append(edge_path)

    datalist_json = [
        {"image": img, "label": seg, 'edge': edge, 'pselab_edge': pselab_edge, 'domain': 'trg'} for img, seg, edge, pselab_edge in
        zip(images, edges, edges, edges)
    ]
    ds_transforms = get_transforms_fixspacing(mode, cfg['augmentation'])
    ds = data.Dataset(data=datalist_json, transform=ds_transforms)

    return ds
