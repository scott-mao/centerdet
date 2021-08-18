import math
import numpy as np
import cv2
import torch

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


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = math.ceil(det_size[0]), math.ceil(det_size[1])

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def center_targets(meta, num_cls=10):
    img_sz = meta['img'].shape[:2] # img_sz -> (h, w)
    fmap_size = [i // 4 for i in img_sz]
    assert len(meta['gt_labels'] == len(meta['gt_bboxes']))

    objs_num = len(meta['gt_labels'])

    heatmap = np.zeros((num_cls, fmap_size[0], fmap_size[1]), dtype=np.float32)  # heatmap
    regs = np.zeros((objs_num, 2), dtype=np.float32)   # width and height
    offsets = np.zeros((objs_num, 2), dtype=np.float32)  # regression
    inds = np.zeros((objs_num, 2), dtype=np.int64)
    # ind_masks = np.zeros((objs_num,), dtype=np.uint8)

    gt_boxes = meta['gt_bboxes']
    if objs_num > 0:
        gt_boxes = gt_boxes // 4
    else:
        pass  # TODO addd special condition
    labels = meta['gt_labels']

    for idx, (bbox, label) in enumerate(zip(gt_boxes, labels)):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if h > 0 and w > 0:
            obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            obj_c_int = obj_c.astype(np.int32)
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), 0.7))) # default gaussian_iou here is set as 0.7
            draw_gaussian(heatmap[label], obj_c_int, radius)
            # bboxes' widths and hights
            regs[idx] = 1. * w, 1. * h
            # center point offset
            offsets[idx] = obj_c - obj_c_int  # discretization error
            # obj index
            inds[idx] = obj_c_int

    meta['ht_map'] = torch.from_numpy(heatmap)
    meta['reg'] = torch.from_numpy(regs)
    meta['offset'] = torch.from_numpy(offsets)
    meta['ind_mask'] = torch.from_numpy(inds)
    return meta
