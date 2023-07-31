import os.path
import numpy as np
from utils.file_and_folder_operations import *
from utils.sitk_utils import *
import medpy.metric.binary as mmb

def get_boundarybox(label):
    h0, h1, w0, w1, d0, d1 = calmin_max(np.nonzero(label))
    return [h0, w0, d0, h1, w1, d1]

def iou(box1, box2):
    '''
    box:[top, left, front, bottom, right, back]
    '''
    in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
    in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
    in_d = min(box1[5], box2[5]) - max(box1[2], box2[2])
    inter = 0 if in_h < 0 or in_w < 0 or in_d < 0 else in_h * in_w * in_d
    union = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2]) + \
            (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2]) - inter
    iou = inter / union
    return iou

def ap(box1, box2):
    # presicion
    # label_bbx, pred_bbx
    in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
    in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
    in_d = min(box1[5], box2[5]) - max(box1[2], box2[2])
    inter = 0 if in_h < 0 or in_w < 0 or in_d < 0 else in_h * in_w * in_d
    pred = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    ap = inter / pred
    return ap

def recall(box1, box2):
    # recall
    # label_bbx, pred_bbx
    in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
    in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
    in_d = min(box1[5], box2[5]) - max(box1[2], box2[2])
    inter = 0 if in_h < 0 or in_w < 0 or in_d < 0 else in_h * in_w * in_d
    gt = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    ap = inter / gt
    return ap


def cal_prosecss(label_file, setname, exp, gt_name):
    out_dict = {}
    out_dict[gt_name] = {}
    exp_result_set = join(exp_root, exp, 'sampled_lr', setname)
    exp_result_outputtr = subfolders(exp_result_set)[0]
    exp_result_files = subfiles(exp_result_outputtr)
    for exp_result_file in exp_result_files:
        exp_result_file_clean_subfix = exp_result_file.replace('_0000', '')
        if gt_name in exp_result_file_clean_subfix:
            print(setname, gt_name, exp )
            pred_file = exp_result_file
            label_arr = nib.load(label_file)
            label_arr = np.asanyarray(label_arr.get_fdata())
            label_arr = np.array(label_arr > 0).astype(int)

            pred_arr = nib.load(pred_file)
            pred_arr = np.asanyarray(pred_arr.get_fdata())
            pred_arr = np.array(pred_arr > 0).astype(int)

            out_dict[gt_name]['dice'] = mmb.dc(pred_arr, label_arr)
            try:
                out_dict[gt_name]['assd'] = mmb.assd(pred_arr, label_arr)
            except:
                out_dict[gt_name]['assd'] = 0
            try:
                out_dict[gt_name]['recall'] = mmb.recall(pred_arr, label_arr)
            except:
                out_dict[gt_name]['recall'] = 0
            try:
                out_dict[gt_name]['precision'] = mmb.precision(pred_arr, label_arr)
            except:
                out_dict[gt_name]['precision'] = 0
            try:
                out_dict[gt_name]['hd95'] = mmb.hd95(pred_arr, label_arr)
            except:
                out_dict[gt_name]['hd95'] = 0
            try:
                label_bbx = get_boundarybox(label_arr)
                pred_bbx = get_boundarybox(pred_arr)
                iou_score = iou(label_bbx, pred_bbx)
                out_dict[gt_name]['iou'] = iou_score
            except:
                out_dict[gt_name]['iou'] = 0
            # try:
            #     label_bbx = get_boundarybox(label_arr)
            #     pred_bbx = get_boundarybox(pred_arr)
            #     ap_score = ap(label_bbx, pred_bbx)
            #     out_dict[gt_name]['ap'] = ap_score
            # except:
            #     out_dict[gt_name]['ap'] = 0


    return out_dict
from multiprocessing import Pool
import fcntl

if __name__ == '__main__':
    gt_path = '/home/brownradai/Projects/kidney/kidney_public/sampled_lr'
    exp_root = '../output_mask'
    # setname_list = ['Task037_CHAOS_T1InPhase', 'Task037_CHAOS_T2SPIR', 'Task133_AMOS_CT'] #
    setname_list = ['Task037_CHAOS_T2SPIR',]
    exp_result_list = ['public',  ]
    metrics_dict = {}
    for setname in setname_list:
        for exp in exp_result_list:
            metrics_dict[setname] = {}
            csv_filename = exp_result_set = join(exp_root, exp, 'sampled_lr', setname+'_'+exp+'_scores.csv')
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
            with open(csv_filename, 'a+') as f:
                line = 'case_name,'
                for metric in ['dice', 'assd', 'recall', 'precision', 'hd95', 'iou',]:
                    line += metric+','
                line += '\n'
                f.write(line)
            def setmycallback(x):
                case_name = list(x.keys())[0]
                score_names = list(x[case_name].keys())
                with open(csv_filename, 'a+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 加锁
                    line = case_name + ','
                    for s_name in score_names:
                        scores = x[case_name][s_name]
                        # e_name_short = e_name.replace('_result', '')
                        # k = e_name_short + '_' + s_name
                        line += str(scores) +','
                    line += '\n'
                    f.write(line)
            labels_path = join(gt_path, setname, 'labelsTr')
            labels_files = subfiles(labels_path)
            p = Pool(36)  # 开启进程
            for label_file in labels_files:
                gt_name = os.path.split(label_file)[-1].replace('.nii.gz', '').replace('_0000', '')

                # out_dict = cal_prosecss(exp_result_list, label_file, setname, exp, gt_name)
                # setmycallback(out_dict)
                p.apply_async(cal_prosecss, (label_file, setname, exp, gt_name), callback=setmycallback)#调用进程
            p.close()
            p.join()
    print()


