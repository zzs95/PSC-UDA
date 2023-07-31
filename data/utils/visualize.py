import matplotlib.pyplot as plt
import numpy as np


def draw_points_image_labels(img, seg, edge, ):
    index = seg[0,0].nonzero().median(0)[0]
    plt.figure(figsize=[15, 15])
    plt.subplot(2, 2, 1)
    plt.imshow(img[0, 0, index[0]])
    plt.subplot(2, 2, 2)
    plt.imshow(img[0, 0, :, index[1]])
    plt.subplot(2, 2, 3)
    plt.imshow(img[0, 0, :, :, index[2]])
    plt.show()
    plt.figure(figsize=[15, 15])
    plt.subplot(2, 2, 1)
    plt.imshow(seg[0, 0, index[0]])
    plt.subplot(2, 2, 2)
    plt.imshow(seg[0, 0, :, index[1]])
    plt.subplot(2, 2, 3)
    plt.imshow(seg[0, 0, :, :, index[2]])
    plt.show()
    plt.figure(figsize=[15, 15])
    plt.subplot(2, 2, 1)
    plt.imshow(edge[0, 0, index[0]*4])
    plt.subplot(2, 2, 2)
    plt.imshow(edge[0, 0, :, index[1]*4])
    plt.subplot(2, 2, 3)
    plt.imshow(edge[0, 0, :, :, index[2]*4])
    plt.show()

import open3d as o3d
import pandas as pd
import torch
def vis_pc(data_batch_src):
    points = data_batch_src['img_indices'][data_batch_src['orig_batch_idx'][1]][:, 2:].cpu().numpy()

    # orig_image
    # seg_labels_onehot = pd.get_dummies(data_batch_src['orig_seg_label'][1]).values

    # sampled_image

    # pred_seg_label = preds_2d['seg_logit'].argmax(1)[data_batch_src['orig_batch_idx'][1]]
    pred_seg_label = preds_3d['seg_logit'].argmax(1)[data_batch_src['orig_batch_idx'][1]]
    pred_seg_label = pred_label[data_batch_src['orig_batch_idx'][1]]
    seg_labels_onehot = pd.get_dummies(pred_seg_label.cpu().numpy()).values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(seg_labels_onehot)
    o3d.visualization.draw_geometries([pcd])
import open3d as o3d
import pandas as pd
from data.utils.edge_utils import random_sample_edge_point, sample_point_feat
def vis_edge(edge):
    edge = torch.from_numpy(edge[None, None])
    edge[0,0,0,0,1] = 1
    edge[0,0,0,0,2] = 2
    edge[0,0,0,0,3] = 3
    img_indices = random_sample_edge_point(edge)
    points = img_indices[:, 2:].cpu().numpy()
    labels = sample_point_feat(img_indices, edge).squeeze().long() - 1
    # labels[0,0,0,0] = 1
    # labels[0,0,0,1] = 1
    # labels[0,0,0,2] = 2
    seg_labels_onehot = pd.get_dummies(labels.cpu().numpy()).values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(seg_labels_onehot)
    o3d.visualization.draw_geometries([pcd])

def vis_kidney(pred_kidney):
    import open3d as o3d
    import pandas as pd
    from data.utils.edge_utils import random_sample_edge_point, sample_point_feat
    edge_draw = pred_kidney[0, None] == 2
    img_indices = random_sample_edge_point(edge_draw)
    points = img_indices[:, 2:].cpu().numpy()
    labels = sample_point_feat(img_indices, edge_draw).squeeze().long() - 1
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    seg_labels_onehot = pd.get_dummies(labels.cpu().numpy()).values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(seg_labels_onehot)
    o3d.visualization.draw_geometries([pcd])

def vis3(edge_gt):
    import open3d as o3d
    import pandas as pd
    from data.utils.edge_utils import random_sample_edge_point, sample_point_feat
    edge_draw = deepcopy(edge_gt)
    edge_draw[pred_kidney.cpu() == 2] = 3
    edge_draw[edge_gt == 1] = 0
    edge_draw = (edge_draw.cpu()).float()
    img_indices = random_sample_edge_point(edge_draw)
    points = img_indices[:, 2:].cpu().numpy()
    labels = sample_point_feat(img_indices, edge_draw).squeeze().long() - 1
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    seg_labels_onehot = pd.get_dummies(labels.cpu().numpy()).values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(seg_labels_onehot)
    o3d.visualization.draw_geometries([pcd])

def vis_train(data_batch_trg):
    from data.utils.edge_utils import set_point_feat
    data_dict = data_batch_trg
    edge_arr = (data_dict['edge'] > 0).type(data_dict['edge'].dtype)
    edge_pred = set_point_feat(data_dict['edge_indices'], edge_arr,
                               data_dict['logits'].cpu().argmax(1).type(data_dict['edge'].dtype)[:,
                               None] + 1)  # 0:2 ->1:3

    import open3d as o3d
    import pandas as pd
    import copy
    from data.utils.edge_utils import random_sample_edge_point, sample_point_feat

    pred_kidney = edge_pred[0, None]
    edge_gt = data_dict['edge'][0, None]
    edge_draw = copy.deepcopy(data_dict['edge'])[0, None]
    edge_draw[pred_kidney.cpu() == 2] = 3
    edge_draw[edge_gt == 1] = 0
    edge_draw = (edge_draw.cpu()).float()
    img_indices = random_sample_edge_point(edge_draw, sample_mode='all')
    points = img_indices[:, 2:].cpu().numpy()
    labels = sample_point_feat(img_indices, edge_draw).squeeze().long() - 1
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    seg_labels_onehot = pd.get_dummies(labels.cpu().numpy()).values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(seg_labels_onehot)
    o3d.visualization.draw_geometries([pcd])

def vis_cluster(pred_pcd):
    cluster_labels = np.array(
        pred_pcd.cluster_dbscan(eps=3, min_points=1, print_progress=True))
    for label in range(cluster_labels.max()):
        label_index = np.where(cluster_labels == label)  # 提取分类为label的聚类点云下标
        label_pcd = pred_pcd.select_by_index(np.array(label_index)[0])  # 根据下标提取点云点
        print('label: ', str(label), '点云数：', len(label_pcd.points))

        # 可视化
        o3d.visualization.draw_geometries([label_pcd], "o3d dbscanclusting " + str(label) + " results", width=400,
                                          height=400)

def vis_nieghbors():
    k, idx, _ = edge_kdtree.search_knn_vector_3d(pred_pcd.points[i], knn=1000)
    neighbors = edge_points[idx, :]
    pcd_n = point2pcd(neighbors)
    pcd_n.paint_uniform_color([0.5, 0.5, 0.5])
    np.asarray(pcd_n.colors)[0, :] = [1, 0, 0]
    np.asarray(pcd_n.colors)[1:, :] = [0, 0, 1]
    o3d.visualization.draw_geometries([pcd_n])