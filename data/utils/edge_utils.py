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
    if_range = False
    batch_size = edge.shape[0]
    if sample_mode=='rand':
        if_range = False
    elif sample_mode=='rand_range':
        if_range = True
    # rand
    all_indices = torch.nonzero(edge > 0, as_tuple=False)
    all_indices = de_meta(all_indices)
    if if_range:
        all_indices_new = []
        for i in range(batch_size):
            edge_range = edge_range_list[i]
            batch_idx = all_indices[:, 0] == i
            pts = all_indices[batch_idx]
            pts_idx = (pts[:, 2] > edge_range[0, 0]) & (pts[:, 2] < edge_range[0, 1]) & \
                      (pts[:, 3] > edge_range[1, 0]) & (pts[:, 3] < edge_range[1, 1]) & \
                      (pts[:, 4] > edge_range[2, 0]) & (pts[:, 4] < edge_range[2, 1])
            all_indices_new.append(pts[pts_idx])
        del all_indices
        all_indices = torch.cat(all_indices_new, dim=0)
    num_max = len(all_indices)
    if point_num != 0:
        rand_num = int(batch_size * point_num)
        img_indices = all_indices[torch.randperm(num_max)][0:min(rand_num, num_max)]
    else:
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
            # save coordinates in 160,320,320 size
            # pts_boundary = np.concatenate([pts_bottom_boundary, pts_top_boundary], axis=0)
            # np.save(boundary_npy_path, pts_boundary)

            # scale coordinates in 160,320,320 size to 1x1x1 spacing, save coordinates in 1x1x1 spacing
            pts_boundary_tansformed = np.concatenate([pts_bottom_boundary, pts_top_boundary], axis=0)
            affine_matrix = data_batch_src['edge'].meta['affine'][i_batch]
            # affine_matrix = data_batch_src['edge'].meta['original_affine'][i_batch]
            pts_boundary_1 = np.concatenate([pts_boundary_tansformed, np.array([[0]]*batch_size)], axis=1)
            pts_boundary_1 = np.dot(pts_boundary_1, affine_matrix).astype(int)[:, :3]
            # print(pts_boundary_1)
            np.save(boundary_npy_path, pts_boundary_1)


