import numpy as np
import torch
import open3d as o3d
from scipy import ndimage as ndi
# import kaolin as kal

def dilate_np(edge, ks = 7):
    dilated = ndi.binary_dilation(
        edge, iterations=ks
    )
    return dilated.astype(int)

def erode_np(edge, ks = 7):
    dilated = ndi.binary_erosion(
        edge, iterations=ks
    )
    return dilated.astype(int)

def remove_statistical_outlier(pred_edge_arr, nb_neighbors=50, std_ratio=0.3):
    pred_pcd = voxel2pcd(pred_edge_arr > 0)
    pred_pcd_1 = pred_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)[0]
    out_arr = pcd2voxel(pred_pcd_1, pred_edge_arr)
    return out_arr

def remove_radius_outlier(pred_edge_arr, nb_points_rate=0.5, radius=3):
    nb_points = int(cal_points_in_radius_2d(radius) * nb_points_rate)
    pred_pcd = voxel2pcd(pred_edge_arr > 0)
    pred_pcd_1 = pred_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)[0]
    out_arr = pcd2voxel(pred_pcd_1, pred_edge_arr)
    return out_arr

def voxel2face1(l_pred_edge_arr):
    l_pred_pcd = voxel2pcd(l_pred_edge_arr)
    l_pred_pcd.estimate_normals()
    radii = [4, 20]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        l_pred_pcd, o3d.utility.DoubleVector(radii))
    mesh = mesh.filter_smooth_simple(number_of_iterations=6)
    # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=1700)
    # mesh = mesh_out.simplify_vertex_clustering(voxel_size=2,
    #     contraction=o3d.geometry.SimplificationContraction.Average)
    # pcd = mesh.sample_points_uniformly(number_of_points=2500)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2)
    pcd.estimate_normals()
    # import pyvista as pv
    # import pymeshfix
    # points = pv.wrap(np.asarray(pcd.points))
    # surf = points.reconstruct_surface()
    # mesh = mesh_smp
    # v = np.asarray(mesh.vertices)
    # f = np.array(mesh.triangles)
    # tin = pymeshfix.PyTMesh()
    # tin.load_array(v, f)
    # mf = pymeshfix.MeshFix(v,f)
    #
    # mf = pymeshfix.MeshFix(surf)
    # mf.repair()
    # repaired = mf.mesh
    # tin.fill_small_boundaries()
    # tin.clean(max_iters=10, inner_loops=3)
    # vclean, fclean = tin.return_arrays()
    # surf = pv.PolyData(vclean, fclean)
    # pl = pv.Plotter()
    # # pl.add_mesh(pc, color='k', point_size=10)
    # pl.add_mesh(repaired)
    # pl.add_title('Reconstructed Surface')
    # pl.show()
    # radii = [4,10,20,30,40,50,60,80,90,100,200]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii))
    # mesh, _ = pcd.compute_convex_hull()
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1)
    mesh_watertight = None
    for i in range(5):
        alpha = 50 - i*10
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if mesh.is_watertight():
            mesh_watertight = mesh
            print(i)
        else:
            break
    if mesh_watertight==None:
        mesh_watertight = pcd.compute_convex_hull()
    mesh = mesh_watertight.simplify_quadric_decimation(target_number_of_triangles=300)
    l_voxel_arr = mesh2mask(mesh, l_pred_edge_arr)
    return l_voxel_arr

def voxel2face(l_pred_edge_arr):
    l_pred_pcd = voxel2pcd(l_pred_edge_arr)
    l_pred_pcd.estimate_normals()
    radii = [4, 20]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        l_pred_pcd, o3d.utility.DoubleVector(radii))
    mesh_out = mesh.filter_smooth_simple(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_out])
    pcd = mesh_out.sample_points_uniformly(number_of_points=2500)
    pcd = mesh_out.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
    mesh_watertight = None
    for i in range(5):
        alpha = 50 - i * 10
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if mesh.is_watertight():
            mesh_watertight = mesh
            print(i)
        else:
            break
    if mesh_watertight == None:
        mesh_watertight = pcd.compute_convex_hull()
    l_voxel_arr = mesh2mask(mesh_watertight, l_pred_edge_arr)
    return l_voxel_arr

def mesh2mask(mesh, arr):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1)
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    resolution = indices.max(0)+1
    voxelgrids = np.zeros(resolution)
    voxelgrids[indices[:, 0], indices[:, 1], indices[:, 2],] = 1
    mesh_ver = np.array(mesh.vertices).astype(float)
    origin = np.array(mesh_ver.min(0)).astype(int)
    end_ = origin + resolution
    less_bound = np.maximum(end_ - arr.shape, [0, 0, 0])
    end_ = end_ - less_bound
    l_voxel_arr = np.zeros_like(arr)
    l_voxel_arr[origin[0]:end_[0], origin[1]:end_[1], origin[2]:end_[2]] = voxelgrids
    return l_voxel_arr

# def voxel2mask(l_pred_edge_arr):
#     l_voxel_arr = voxel2hull(l_pred_edge_arr)
#     l_voxel_arr = ndi.binary_fill_holes(l_voxel_arr).astype(int)
#     return l_voxel_arr

# def voxel2hull(l_pred_edge_arr):
#     l_pred_pcd = voxel2pcd(l_pred_edge_arr)
#     mesh, _ = l_pred_pcd.compute_convex_hull()
#     mesh_ver = np.array(mesh.vertices).astype(float)[None]
#     vertices = torch.from_numpy(mesh_ver).cuda()
#     mesh_tan = np.array(mesh.triangles).astype(np.int64)
#     faces = torch.from_numpy(mesh_tan).cuda()
#     resolution = int(np.max(mesh_ver.max(1) - mesh_ver.min(1))) + 1
#     voxelgrids = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices, faces, resolution)
#     origin = np.array(mesh_ver.min(1)[0]).astype(int)
#     end_ = origin + resolution
#     less_bound = np.maximum(end_ - l_pred_edge_arr.shape, [0, 0, 0])
#     end_ = end_ - less_bound
#     l_voxel_arr = np.zeros_like(l_pred_edge_arr)
#     l_voxel_arr[origin[0]:end_[0], origin[1]:end_[1], origin[2]:end_[2]] = \
#         voxelgrids[0].cpu().numpy()[:resolution - less_bound[0], :resolution - less_bound[1],
#         :resolution - less_bound[2], ]
#     return l_voxel_arr

from data.utils.edge_utils import set_point_feat
def pcd2voxel(pred_pcd_1, pred_edge_arr):
    pc_ver = np.array(pred_pcd_1.points).astype(int)
    pc_ver = np.concatenate([np.zeros(len(pc_ver))[:,None], np.zeros(len(pc_ver))[:,None], pc_ver], axis=-1)
    out_arr = np.zeros_like(pred_edge_arr)[None,None]
    out_arr = set_point_feat(pc_ver, torch.from_numpy(out_arr), edge_label=1).numpy()[0,0]
    return out_arr

def voxel2pcd(edge_arr):
    edge_arr = edge_arr.astype(int)
    pred_point_cloud = np.array(np.nonzero(edge_arr > 0)).T
    pred_pcd = point2pcd(pred_point_cloud)
    return pred_pcd

def point2pcd(pred_point_cloud):
    pred_point_cloud = np.array(pred_point_cloud)
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_point_cloud[:, :3])
    return pred_pcd

def remove_curvature(pred_edge_arr, radius=30, thread=80):
    pred_pcd = voxel2pcd(pred_edge_arr > 0)
    curvature = caculate_surface_curvature(pred_pcd, radius=radius)

    curvature = curvature / curvature.max()
    knee_index = curvature > np.percentile(curvature, thread)
    pred_pcd_1 = pred_pcd.select_by_index(np.where(1-knee_index)[0])
    out_arr = pcd2voxel(pred_pcd_1, pred_edge_arr)
    return out_arr

def cluster_arr(l_edge_arr):
    l_edge_pcd = voxel2pcd(l_edge_arr >= 1)
    cluster_labels = cluster_pcd(l_edge_pcd)
    return cluster_labels

def cluster_pcd(l_edge_pcd):
    cluster_labels = np.array(l_edge_pcd.cluster_dbscan(eps=3, min_points=1, print_progress=False))
    return cluster_labels

def max_edge_part_pcd(l_edge_pcd):
    cluster_labels = cluster_pcd(l_edge_pcd)
    label_num = [np.sum(cluster_labels == i) for i in range(cluster_labels.max()+1)]
    label_index = np.where(cluster_labels == np.argmax(label_num))[0]
    label_pcd = l_edge_pcd.select_by_index(np.array(label_index))  # 根据下标提取点云点
    return label_pcd

def max_edge_part_arr(l_edge_arr):
    l_edge_pcd = voxel2pcd(l_edge_arr >= 1)
    label_pcd = max_edge_part_pcd(l_edge_pcd)
    l_edge_arr = pcd2voxel(label_pcd, l_edge_arr)
    return l_edge_arr

def neighbors_cluster(neighbors):
    pcd_n = point2pcd(neighbors)
    cluster_pcd_n = cluster_pcd(pcd_n)
    # neighbors_center = neighbors[0]
    neighbors_label = cluster_pcd_n[0]
    neighbors_index = np.where(neighbors_label == cluster_pcd_n)[0]
    pcd_neighbors = pcd_n.select_by_index(neighbors_index)
    return np.asarray(pcd_neighbors.points).tolist(), neighbors_index

# def remove_knee(l_edge_arr):
#     knee_conv17 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=17, padding='same')
#     knee_conv15 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=15, padding='same')
#     knee_conv13 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=13, padding='same')
#     knee_conv11 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=11, padding='same')
#     knee_conv9 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=9, padding='same')
#     knee_conv7 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=7, padding='same')
#     knee_conv5 = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, padding='same')
#     knee_conv5.weight.data[0] = torch.ones_like(knee_conv5.weight)
#     knee_conv7.weight.data[0] = torch.ones_like(knee_conv7.weight)
#     knee_conv9.weight.data[0] = torch.ones_like(knee_conv9.weight)
#     knee_conv11.weight.data[0] = torch.ones_like(knee_conv11.weight)
#     knee_conv13.weight.data[0] = torch.ones_like(knee_conv13.weight)
#     knee_conv15.weight.data[0] = torch.ones_like(knee_conv15.weight)
#     knee_conv17.weight.data[0] = torch.ones_like(knee_conv17.weight)
#     knee_bn = torch.nn.BatchNorm3d(1)
#
#     # remove knee points
#     edge_slice_ts = torch.from_numpy(l_edge_arr)
#     knee17 = knee_conv17(edge_slice_ts[None, None].float())
#     knee15 = knee_conv15(edge_slice_ts[None, None].float())
#     knee13 = knee_conv13(edge_slice_ts[None, None].float())
#     knee11 = knee_conv11(edge_slice_ts[None, None].float())
#     knee9 = knee_conv9(edge_slice_ts[None, None].float())
#     knee7 = knee_conv7(edge_slice_ts[None, None].float())
#     knee5 = knee_conv5(edge_slice_ts[None, None].float())
#     knee = knee17 + knee15 + knee13 + knee11 + knee9 + knee7 + knee5
#     knee = knee_bn(knee)
#     knee = knee.detach().numpy()[0, 0]
#     knee[l_edge_arr == 0] = 0
#     thread = np.percentile(knee[knee > 0], 90)
#     l_edge_arr[knee > thread] = 0
#     return l_edge_arr
def pca_compute(data, sort=True):
    """
     SVD分解计算点云的特征值与特征向量
    :param data: 输入数据
    :param sort: 是否将特征值特征向量进行排序
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)  # 求均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引
    return eigenvalues

def caculate_surface_curvature(cloud, radius = 50):
    """
    计算点云的表面曲率
    :param cloud: 输入点云
    :param radius: k近邻搜索的半径，默认值为：0.003m
    :return: 点云中每个点的表面曲率
    """
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        # k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
        k, idx, _ = kdtree.search_knn_vector_3d(cloud.points[i], knn = radius)
        neighbors = points[idx, :]
        neighbors, n_idx = neighbors_cluster(neighbors)
        w = pca_compute(neighbors)  # w为特征值
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
    return curvature

def cal_points_in_radius_3d(radius=5):
    num = 0
    for x in range(-radius, radius):
        for y in range(-radius, radius):
            for z in range(-radius, radius):
                dis = (x**2 + y**2 +z**2)**0.5
                if dis <= 5:
                    num += 1
    return num
def cal_points_in_radius_2d(radius=5):
    num = 0
    for x in range(-radius, radius):
        for y in range(-radius, radius):
            dis = (x**2 + y**2 )**0.5
            if dis <= 5:
                num += 1
    return num