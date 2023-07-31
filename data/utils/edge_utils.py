import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from utils.file_and_folder_operations import *

def de_meta(maybe_metatensor):
    try:
        maybe_metatensor = maybe_metatensor.as_tensor()
        return maybe_metatensor
    except:
        return maybe_metatensor

def random_sample_edge_point(edge, sample_mode='all', point_num=0, edge_range_list=None):
    '''
    :param edge: labeled or unlabeled edge [b,1,320,320,320]
    :param if_former_mask: if inference resample base on former
    :param point_num:
    :return:
    '''
    if_grid = False
    if_range = False
    batch_size = edge.shape[0]
    if sample_mode=='rand':
        if_use_label = False
        if_former_mask = False
    elif sample_mode=='rand_range':
        if_use_label = False
        if_former_mask = False
        if_range = True
    elif sample_mode=='former':
        if_use_label = True
        if_former_mask = True
    elif sample_mode=='gt':
        if_use_label = True
        if_former_mask = False
    elif sample_mode=='grid':
        if_use_label = False
        if_former_mask = False
        if_grid = True
    elif sample_mode=='all':
        # default: sample all
        if_use_label = False
        if_former_mask = False
        point_num = 0

    if if_use_label:
        # labeled sample
        if if_former_mask:
            edge = former_mask(edge)
        # class balance_sample
        # nonkidney_indices = torch.argwhere(dim=1)

        nonkidney_indices = torch.nonzero(edge == 1, as_tuple=False)
        Lkidney_indices = torch.nonzero(edge == 2, as_tuple=False)
        Rkidney_indices = torch.nonzero(edge == 3, as_tuple=False)
        nonkidney_indices = de_meta(nonkidney_indices)
        Lkidney_indices = de_meta(Lkidney_indices)
        Rkidney_indices = de_meta(Rkidney_indices)

        Lk_num_max = len(Lkidney_indices)
        Lkidney_indices = Lkidney_indices[torch.randperm(Lk_num_max)]  # if not random, will sample on batch 1
        Lkidney_indices = Lkidney_indices[0:min(int(batch_size*point_num*0.4), Lk_num_max)]
        # Rkidney_indices = torch.argwhere(edge==3)
        Rk_num_max = len(Rkidney_indices)
        Rkidney_indices = Rkidney_indices[torch.randperm(Rk_num_max)]
        Rkidney_indices = Rkidney_indices[0:min(int(batch_size*point_num*0.2), Rk_num_max)]

        nonk_num_max = len(nonkidney_indices)
        nonk_num = int(batch_size*point_num - len(Lkidney_indices) - len(Rkidney_indices))
        nonkidney_indices = nonkidney_indices[torch.randperm(nonk_num_max)]
        nonkidney_indices = nonkidney_indices[0:min(nonk_num, nonk_num_max)]
        # Lkidney_indices = torch.argwhere(edge==2)
        img_indices = torch.cat([nonkidney_indices, Lkidney_indices, Rkidney_indices], dim=0)
    else:
        if if_grid:
            # grid sample
            grid_size = 3
            edge_small = edge[:,:,::grid_size,::grid_size,::grid_size]
            all_indices = torch.nonzero(edge_small > 0, as_tuple=False)
            all_indices = de_meta(all_indices)
            all_indices[:, 2:] = all_indices[:, 2:] * grid_size
            num_max = len(all_indices)
            if point_num!=0:
                rand_num = int(batch_size * point_num )
                img_indices = all_indices[torch.randperm(num_max)][0:min(rand_num, num_max)]
            else:
                img_indices = all_indices
        else:
            # rand
            all_indices = torch.nonzero(edge > 0, as_tuple=False)
            all_indices = de_meta(all_indices)
            if if_range:
                all_indices_new = []
                for i in range(batch_size):
                    edge_range = edge_range_list[i]
                    batch_idx = all_indices[:,0] == i
                    pts = all_indices[batch_idx]
                    pts_idx = (pts[:,2] > edge_range[0,0]) & (pts[:,2] < edge_range[0,1]) & \
                            (pts[:,3] > edge_range[1,0]) & (pts[:,3] < edge_range[1,1]) & \
                            (pts[:,4] > edge_range[2,0]) & (pts[:,4] < edge_range[2,1])
                    all_indices_new.append(pts[pts_idx])
                del all_indices
                all_indices = torch.cat(all_indices_new, dim=0)
            num_max = len(all_indices)
            if point_num!=0:
                rand_num = int(batch_size * point_num )
                img_indices = all_indices[torch.randperm(num_max)[0:min(rand_num, num_max)]]
            else:
                # all sample
                img_indices = all_indices
                img_indices = img_indices[torch.randperm(len(img_indices))]
    return img_indices[img_indices[:,0].sort()[1]]

def sample_point_feat(img_indices, feats, ):
    '''
    Make sure the indices and feats are in the same scele.
    :param img_indices: [n, 5]
    :param feats: [b, 128, 64, 64, 64]
    :return:
    '''
    feats = de_meta(feats)
    edge_label = feats.permute(0, 2, 3, 4, 1)[
        img_indices[:, 0], img_indices[:, 2], img_indices[:, 3], img_indices[:, 4]]

    return edge_label

def set_point_feat(img_indices, edge, edge_label):
    '''
    :param img_indices: [n, 5]
    :param edge:[b, 1, 64, 64, 64]
    :param edge_label: [n, 1]
    :return:
    '''
    edge.permute(0, 2, 3, 4, 1)[
        img_indices[:, 0], img_indices[:, 2], img_indices[:, 3], img_indices[:, 4]] = edge_label
    return edge

def former_mask(edge):
    # former_edge = F.interpolate(edge, scale_factor=1/4, mode='trilinear', align_corners=False)
    # edge = F.interpolate(former_edge, scale_factor=4, mode='trilinear', align_corners=False)

    # former_edge = deepcopy(edge).to(edge.device)
    former_edge = edge
    former_edge_L_dilate = dilate_tr((former_edge==2).float())*1
    former_edge_R_dilate = dilate_tr((former_edge==3).float())*2
    edge_ones = (edge > 0).float()
    former_edge_L_Neighbors = edge_ones * former_edge_L_dilate
    former_edge_R_Neighbors = edge_ones * former_edge_R_dilate
    edge = edge_ones + (former_edge_L_Neighbors.int() | former_edge_R_Neighbors.int())
    # select the un detected points
    edge[former_edge > 1] = 0
    return edge

import torch.nn.functional as F
def dilate_tr(bin_img, kr=5):
    pad = kr
    ksize = kr * 2 + 1
    out = F.pad(bin_img, pad=[pad]*6, mode="constant", value=0)
    out = F.max_pool3d(out, kernel_size=ksize, stride=1, padding=0)
    return out

def erode_tr(bin_img, kernelRadius=5):
    out = 1 - dilate_tr(1 - bin_img, kernelRadius)
    return out

def save_edge_range(data_batch_src, cur_boundary_path):
    batch_size = data_batch_src['img'].shape[0]
    for i_batch in range(batch_size):
        label_name = data_batch_src['label_meta_dict']['filename_or_obj'][i_batch].split('/')[-1]
        label_name = label_name.replace('.nii.gz', '.npy')
        boundary_npy_path = join(cur_boundary_path, label_name)
        if not os.path.exists(boundary_npy_path):
            pts_idx = data_batch_src['orig_batch_idx'][i_batch]
            pts = data_batch_src['points'][pts_idx][:, :3]
            pts_top_boundary = pts.max(dim=0)[0].cpu().numpy()[None]
            pts_bottom_boundary = pts.min(dim=0)[0].cpu().numpy()[None]
            pts_boundary_tansformed = np.concatenate([pts_bottom_boundary, pts_top_boundary], axis=0)
            # space 160,320,320 to 1x1x1
            affine_matrix = data_batch_src['edge'].meta['affine'][i_batch]
            # affine_matrix = data_batch_src['edge'].meta['original_affine'][i_batch]
            pts_boundary_1 = np.concatenate([pts_boundary_tansformed, np.array([[0]]*batch_size)], axis=1)
            pts_boundary_1 = np.dot(pts_boundary_1, affine_matrix).astype(int)[:, :3]
            # print(pts_boundary_1)
            np.save(boundary_npy_path, pts_boundary_1)

