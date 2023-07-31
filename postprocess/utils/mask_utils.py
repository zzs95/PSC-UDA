from skimage import measure
from postprocess.utils.edge_utils import *
def mask_connected_components(blobs):
    blobs_labels = measure.label(blobs, background=0)
    return blobs_labels

def max_connected_components(blobs):
    blobs_labels = measure.label(blobs, background=0)
    cc_num = []
    for i in range(1, np.max(blobs_labels)+1):
        cc_num.append(np.sum(blobs_labels == i))
    if len(cc_num) > 1:
        blobs_labels = np.array(blobs_labels == (np.argmax(cc_num)+1)).astype(int)
    else:
        blobs_labels = blobs_labels
    return blobs_labels, cc_num

def max_connected_components_weight(edge_sum):
    blobs = (edge_sum>0).astype(int)
    blobs_labels = measure.label(blobs, background=0)
    cc_num = []
    for i in range(1, np.max(blobs_labels)+1):
        cc_num.append(np.sum(edge_sum[blobs_labels == i]))
    if len(cc_num) > 1:
        blobs_labels = np.array(blobs_labels == (np.argmax(cc_num)+1)).astype(int)
    else:
        blobs_labels = blobs_labels
    return blobs_labels, cc_num
def mask_bbox(r_img_thread_arr):
    r_img_thread_arr = np.array(r_img_thread_arr).astype(int)

def max_cc_filter(r_img_thread_arr, r_pred_edge_arr):
    from skimage import measure
    cc_abels = measure.label(r_img_thread_arr, background=0)
    if cc_abels.max() > 1:
        cc_edge_sums = []
        for label in range(1, cc_abels.max() + 1):
            cc_d = dilate_np(cc_abels == label, ks=5)
            cc_pred_edge_arr = cc_d * r_pred_edge_arr
            cc_edge_sum = np.sum(cc_pred_edge_arr)
            cc_edge_sums.append(cc_edge_sum)
        cc_max_label = np.argmax(cc_edge_sums) + 1
        r_img_thread_arr = np.array(cc_abels == cc_max_label).astype(int)
    return r_img_thread_arr