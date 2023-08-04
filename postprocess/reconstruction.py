import os.path
from skimage import filters
from multiprocessing import Pool
from utils.sitk_utils import *
from postprocess.utils.mask_utils import *
from postprocess.utils.edge_utils import *

def process_file(file_name):
    gt_nii_path = join(gt_root, gt_setname, 'edgelabelsTr', file_name)
    edge_path = join(gt_root, gt_setname, 'edgesTr', file_name)
    pred_mask_nii_path = gt_nii_path.replace(gt_root, pred_root_exp)
    pred_edge_nii_path = pred_mask_nii_path.replace(gt_setname, gt_setname+'_edge')

    edge_itk = read_itk(edge_path)
    edge_arr = sitk.GetArrayFromImage(edge_itk)

    output_list = []
    kd_num_list = []

    for i in range(1, 15):
        val_folder = 'val_result_' + str(i) + '000'
        output_nii_path = join(set_path, val_folder, 'val_output', file_name)
        if os.path.exists(output_nii_path):
            output_itk = read_itk(output_nii_path)
            output_arr = sitk.GetArrayFromImage(output_itk)
            kd_output = output_arr == 2
            kd_output = np.array(kd_output).astype(int)
            kd_output_d = dilate_np(kd_output, 2)
            output_list.append(kd_output_d)
            kd_num = np.sum(kd_output)
            kd_num_list.append(kd_num)
    output_list = np.array(output_list)
    output_sum = output_list.sum(0)

    if np.sum(output_sum) > 0:
        val = filters.threshold_otsu(output_sum)
        output_edgemap = (output_sum > val) * edge_arr
        output_edgemap, cc_num = max_connected_components(output_edgemap)
    if np.sum(output_sum) == 0:
        print('no output')
        pred_mask = output_sum
    else:
        try:
            pred_hull = voxel2face(output_edgemap)
            pred_hull = dilate_np(pred_hull, ks=5)
            pred_mask = ndi.binary_fill_holes(pred_hull).astype(int)
            pred_mask = erode_np(pred_mask, ks=5)

        except:
            pred_mask = output_sum
            print('voxel2hull error')

    maybe_mkdir_p(os.path.split(pred_mask_nii_path)[0])
    pred_itk = sitk.GetImageFromArray(pred_mask)
    pred_itk.CopyInformation(edge_itk)
    write_nrrd(pred_itk, pred_mask_nii_path)
    maybe_mkdir_p(os.path.split(pred_edge_nii_path)[0])
    pred_edge_itk = sitk.GetImageFromArray(output_edgemap)
    pred_edge_itk.CopyInformation(edge_itk)
    write_nrrd(pred_edge_itk, pred_edge_nii_path)

if __name__ == '__main__':
    output_root = '../output/'
    pred_root = '../output_mask/'
    gt_root = '../Projects/kidney/kidney_public/sampled_lr'
    set_name_dict = {
        "AMOS": "Task133_AMOS_CT",
        "t1in": "Task037_CHAOS_T1InPhase",
        "t2": "Task037_CHAOS_T2SPIR", }
    setnames = {
                'public':[ 'AMOS', 't1in', 't1out', 't2' ],
                'source_edge':[ 'AMOS', 't1in', 't1out', 't2' ],
                'source_baseline':[ 'AMOS', 't1in', 't1out', 't2' ],
    }

    setnames = {
        'public':[ 't2',  ],
                }

    for exp_name in setnames.keys():
        for set_name in setnames[exp_name]:
            pred_root_exp = join(pred_root, exp_name, 'sampled_lr')
            set_path = join(output_root, exp_name, '' + set_name)
            gt_setname = set_name_dict[set_name]
            part_lr_names = subfiles(join(set_path, 'val_result_1000', 'val_output'), join=False)
            # for file_name in part_lr_names:
            #     process_file(file_name)
            p = Pool(1)
            for file_name in part_lr_names:
                p.apply_async(process_file, (file_name,),)
            p.close()
            p.join()
