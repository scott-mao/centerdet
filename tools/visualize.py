import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from data.transform.warp import get_resize_matrix, warp_boxes
# def text_form(text):

def plot_bboxes(img, dets, class_names):
    for box in dets:
        x0, y0, x1, y1  = box[:4].astype(np.int)
        score, label = box[4], int(box[5])
        # color = self.cmap(i)[:3]
        color = (COLORS_TAB[label] * 255).astype(np.uint8).tolist()
        text = '{}:{:.2f}'.format(class_names[label], score)
        txt_color=(0, 0, 0) if np.mean(COLORS_TAB[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        cv2.rectangle(img,
                      (x0, y0 - txt_size[1] - 1),
                      (x0 + txt_size[0] + txt_size[1], y0 - 1), color, -1)
        cv2.putText(img, text, (x0, y0 - 1),
                    font, 0.5, txt_color, thickness=1)
    return img


def plot_results(img, dets, class_names, vis_thr=0.3, out_size=(320, 320), show_text=True):
    h, w = img.shape[:2]

    # pad_h, pad_w = out_size[0] -
    size_matrix = get_resize_matrix((w, h), out_size, keep_ratio=True)
    img = cv2.warpPerspective(img, size_matrix, dsize=out_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets[:, :4] = warp_boxes(dets[:, :4], size_matrix, out_size[1], out_size[0])
    dets = sorted(dets, key=lambda x: x[0])
    for box in dets:
        x0, y0, x1, y1 = box[:4].astype(np.int)
        score, label = box[4], int(box[5])
        if score < vis_thr:
            continue
        # color = self.cmap(i)[:3]
        color = (COLORS_TAB[label] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_4)

        if show_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = '{}:{:.2f}'.format(class_names[label], score)
            txt_color = (0, 0, 0) if np.mean(COLORS_TAB[label]) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            cv2.rectangle(img,
                          (x0, y0 - txt_size[1] - 1),
                          (x0 + txt_size[0] + txt_size[1], y0 - 1), color, -1)
            cv2.putText(img, text, (x0, y0 - 1),
                    font, 0.5, txt_color, thickness=1)
    return img


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for i in range(epochs):
        scheduler.step()
        if i == 0:
            print(optimizer.param_groups[0]['lr'])
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)



COLORS_TAB = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
