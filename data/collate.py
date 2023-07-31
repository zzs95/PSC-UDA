import copy

import torch
from functools import partial
from data.utils.edge_utils import random_sample_edge_point, sample_point_feat, set_point_feat
from torch.utils.data.dataloader import default_collate
from monai.data.utils import list_data_collate, worker_init_fn
import numpy as np
from copy import deepcopy
import os
from monai.data.image_reader import NibabelReader
from monai import transforms
def get_collate_scn(is_train, point_num, crop_pselab=False, pselab_path=None):
    # if is_train:
    #     sample_mode = 'rand'
    # else:
    #     sample_mode = 'grid'
    sample_mode = 'rand'
    return partial(collate_scn_base,
                   sample_mode=sample_mode, point_num=point_num, crop_pselab=crop_pselab, pselab_path=pselab_path
                   )

def collate_scn_base(input_dict_list, sample_mode = 'rand', point_num=20000, crop_pselab=False, pselab_path=None):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """

    # input_dict = default_collate(input_dict_list)
    input_dict = list_data_collate(input_dict_list)  # meta tensor
    out_dict = collate_scn_data(input_dict, sample_mode=sample_mode, point_num=point_num, crop_pselab=crop_pselab, pselab_path=pselab_path)

    return out_dict

def read_trg_pselab_edge(pselab_edge_path):
    trans = transforms.Compose([transforms.LoadImaged(keys='pselab_edge', image_only=False),
        # transforms.Affined(keys='pselab_edge', affine=gt_edge.meta['affine'])
                                ])
    pselab_edge = trans({'pselab_edge':pselab_edge_path})
    return pselab_edge['pselab_edge']

def collate_scn_data(input_dict, sample_mode='rand', point_num=20000, if_window_batch=False, crop_pselab=False, pselab_path=None):
    '''
    :param input_dict: 'edge', 'image' 'label
    :return: out_dict
    '''

    batch_size = input_dict['edge'].shape[0]
    input_dict['edge'][input_dict['edge'] == 3] = 2  # for 2 classes
    edge = input_dict['edge']
    if crop_pselab:
        # calculate in 1x1x1 space
        label_files = input_dict['label_meta_dict']['filename_or_obj']
        edge_range_list = []
        for i_batch in range(batch_size):
            # progressive ROI
            edge_name = label_files[i_batch].split('/')[-1]
            boundary_npy_name = edge_name.replace('.nii.gz', '.npy')
            boundary_npy_path = os.path.join(pselab_path, 'boundary', boundary_npy_name)
            if os.path.exists(boundary_npy_path):
                edge_bottom_last, edge_top_last = np.load(boundary_npy_path)
                if input_dict['domain'][i_batch] == 'src':
                    pselab_edge_path = input_dict['pselab_edge'].meta['filename_or_obj'][i_batch]
                    pselab_edge = read_trg_pselab_edge(pselab_edge_path)
                elif input_dict['domain'][i_batch] == 'trg':
                    pselab_edge_path = os.path.join(pselab_path, 'val_output', edge_name)
                    pselab_edge = read_trg_pselab_edge(pselab_edge_path) # load last point cloud prediction
                pselab_edge_tmp = deepcopy(pselab_edge)
                pselab_nonz = (pselab_edge_tmp >= 2).nonzero() # get predicted kidney point cloud
                if len(pselab_nonz) < 20:
                    edge_bottom_cur = edge_bottom_last
                    edge_top_cur = edge_top_last
                else:
                    kidney_bottom = pselab_nonz.min(dim=0)[0].detach().cpu().numpy()
                    kidney_top = pselab_nonz.max(dim=0)[0].detach().cpu().numpy()
                    edge_bottom_dist = np.floor(((kidney_bottom - edge_bottom_last) * 0.1)).astype(int) # 0.1
                    edge_top_dist = np.floor(((edge_top_last - kidney_top) * 0.1)).astype(int)
                    edge_bottom_cur = edge_bottom_last + edge_bottom_dist
                    edge_top_cur = edge_top_last - edge_top_dist
                # space 1x1x1 to fixsize 160,320,320
                affine_matrix = input_dict['edge'][i_batch].affine
                inverse_affine_matrix = np.linalg.inv(affine_matrix)
                edge_bottom_cur = np.dot([[list(edge_bottom_cur) + [0]]], inverse_affine_matrix.T).astype(int)[0, 0, :3]
                edge_top_cur = np.dot([[list(edge_top_cur) + [0]]], inverse_affine_matrix.T).astype(int)[0, 0, :3]
            else:
                edge_bottom_cur = [0,0,0]
                edge_top_cur = edge.shape[2:] # 160,320,320

            edge_range = np.array([edge_bottom_cur, edge_top_cur]).T.astype(int)
            edge_range_list.append(edge_range)
        edge_range_list = np.array(edge_range_list)
        img_indices = random_sample_edge_point(edge, sample_mode=sample_mode+'_range', point_num=point_num, edge_range_list=edge_range_list)  # 320 px
    else:
        img_indices = random_sample_edge_point(edge, sample_mode=sample_mode, point_num=point_num)  # 320 px

    locs = img_indices[:, [2, 3, 4, 0]]
    feats = torch.ones([len(img_indices), 1], ).to(locs.device)
    # out_dict = {'x': [locs, feats]}
    out_dict = {}

    out_dict['points'] = torch.concat([locs[:,:3], feats], dim=1)
    out_dict['batch_idx'] = locs[:, 3]
    out_dict['batch_size'] = batch_size
    edge_label = edge - 1
    labels = sample_point_feat(img_indices, edge_label).squeeze().long()
    out_dict['edge_label'] = labels
    if 'label' in input_dict.keys():
        out_dict['mask_label'] = input_dict['label']

    out_dict['edge'] = edge  # for inference
    out_dict['edge_indices'] = deepcopy(img_indices)

    out_dict['img'] = input_dict['image']
    scale = round(input_dict['edge'].shape[-1] / input_dict['image'].shape[-1])
    img_indices[:,2:] = (deepcopy(img_indices)[:,2:] / scale).type(img_indices.dtype)
    out_dict['img_indices'] = img_indices
    # print(img_indices.max(), max(out_dict['img'].shape))
    # batch split
    batch_indexes = img_indices[:, 0]
    orig_seg_label = []
    orig_points_idx = []
    batch_idx = []
    spatial_shapes = []
    for i in range(batch_size):
        case_ind = (batch_indexes == i).nonzero(as_tuple=True)[0]
        batch_idx.append(case_ind)
        case_label = labels[case_ind]
        # case_indices = img_indices[case_ind]
        orig_seg_label.append(case_label)
        orig_points_idx.append(torch.ones_like(case_label, dtype=bool))
        spatial_shapes.append(edge[i,0].shape)
    out_dict['orig_batch_idx'] = batch_idx
    out_dict['orig_seg_label'] = orig_seg_label
    out_dict['orig_points_idx'] = orig_points_idx
    out_dict['spatial_shape'] = spatial_shapes
    # meta info
    if not if_window_batch:
        for k in input_dict.keys():
            if '_meta_dict' in k:
                out_dict[k] = input_dict[k]
    return out_dict


# def decollect_scn_data(input_dict):
#     # edge_arr = deepcopy(input_dict['edge'])
#     edge_arr = (input_dict['edge'] > 0).type(input_dict['edge'].dtype)
#     edge_pred = set_point_feat(input_dict['edge_indices'], edge_arr,
#                                input_dict['seg_label_pred'].type(input_dict['edge'].dtype)[:,None] + 1)  # 0:2 ->1:3
#     # orig_img_indices = []
#     # for batch_idx in input_dict['orig_batch_idx']:
#         # slice_starts = np.array([s.start for s in input_dict['unravel_slice'][0]])
#         # orig_img_ind = slice_starts[[0, 2, 3, 4]] + input_dict['img_indices'][batch_idx].cpu().numpy()[:, [0, 2, 3, 4]]
#         # orig_img_ind = input_dict['img_indices'][batch_idx].cpu().numpy()[:, [0, 2, 3, 4]]
#         # orig_img_indices.append(orig_img_ind)
#     # input_dict['slice_img_indices'] = orig_img_indices
#     return edge_pred

def decollect_scn_data(edge_arr, edge_indices, seg_label_pred ):
    edge_arr = (edge_arr > 0).type(edge_arr.dtype)
    edge_pred = set_point_feat(edge_indices, edge_arr,
                               seg_label_pred.type(edge_arr.dtype)[:,None] + 1)  # 0:2 ->1:3
    return edge_pred