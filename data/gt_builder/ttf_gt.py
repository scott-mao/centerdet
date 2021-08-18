import math
import numpy as np
import torch

# from icecream import ic
import matplotlib.pyplot as plt

def gaussian_truncate_2d(shape, sigma_x=1., sigma_y=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    # h_radius = math.ceil(h_radius)
    # w_radius = math.ceil(w_radius)
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_truncate_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top: h_radius + bottom,
                      w_radius - left: w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return gaussian


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas


def ttf_targets(meta, num_cls=80):
    # building targets for single image

    assert len(meta['gt_labels']) == len(meta['gt_bboxes'])
    img_sz = meta['img'].shape[:2]  # img_sz -> (h, w)

    fmap_size = [i // 4 for i in img_sz]
    alpha = 0.54
    objmax = 50
    objmin = 10
    # objs_num = len(meta['gt_labels'])

    heatmap = np.zeros((num_cls, fmap_size[0], fmap_size[1]), dtype=np.float32)  # heatmap
    fake_heatmap = np.zeros((fmap_size[0], fmap_size[1]), dtype=np.float32)
    reg_box = np.zeros((4, fmap_size[0], fmap_size[1]), dtype=np.float32)
    reg_weight = np.zeros((1, fmap_size[0], fmap_size[1]), dtype=np.float32)
    pos_weight = np.zeros((num_cls, fmap_size[0], fmap_size[1]), dtype=np.float16)


    gt_boxes = meta['gt_bboxes']
    labels = meta['gt_labels']

    boxes_areas_log = np.log(bbox_areas(gt_boxes))

    indices = np.argsort(-boxes_areas_log)

    gt_boxes = gt_boxes[indices]
    labels = labels[indices]
    boxes_areas_log = boxes_areas_log[indices]

    feat_gt_boxes = gt_boxes / 4
    feat_gt_boxes[:, [0, 2]] = np.clip(feat_gt_boxes[:, [0, 2]],
                                       a_min=0, a_max=fmap_size[1] - 1)
    feat_gt_boxes[:, [1, 3]] = np.clip(feat_gt_boxes[:, [1, 3]],
                                       a_min=0, a_max=fmap_size[0] - 1)
    feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                        feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
    ct_ints = (np.stack(((gt_boxes[:, 2] + gt_boxes[:, 0]) / 2,
                         (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2), axis=1) / 4).astype(np.int32)

    h_radiuses_alpha = (feat_hs / 2. * alpha).astype(np.int32)
    w_radiuses_alpha = (feat_ws / 2. * alpha).astype(np.int32)

    for i in range(indices.shape[0]):
        # filter smaller object
        if boxes_areas_log[i] < 1.3:
            continue

        cls_id = labels[i]

        maxline = max(feat_hs[i], feat_ws[i])
        gamma = (maxline - objmin) / (objmax - objmin) * 5
        gamma = min(max(0, gamma), 5) + 1
        draw_gaussian(pos_weight[cls_id], ct_ints[i], 2, k=gamma)


        # ic(fake_heatmap)
        fake_heatmap.fill(0)
        # ic(fake_heatmap)
        draw_truncate_gaussian(fake_heatmap,
                               ct_ints[i],
                               h_radiuses_alpha[i],
                               w_radiuses_alpha[i])

        heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
        box_target_inds = fake_heatmap > 0

        reg_box[:, box_target_inds] = gt_boxes[i][:, np.newaxis]

        local_heatmap = fake_heatmap[box_target_inds]
        # ct_div = max(local_heatmap.sum(), 0.8)
        ct_div = local_heatmap.sum()
        local_heatmap *= boxes_areas_log[i]
        reg_weight[0, box_target_inds] = local_heatmap / ct_div


    meta['ht_map'] = torch.from_numpy(heatmap)
    meta['reg_box'] = torch.from_numpy(reg_box)
    meta['weight'] = torch.from_numpy(reg_weight)
    meta['pos_weight'] = torch.from_numpy(pos_weight)
    # meta['warp_matrix'] = torch.from_numpy(meta['warp_matrix'])
    return meta


