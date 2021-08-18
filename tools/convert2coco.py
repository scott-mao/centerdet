import json
import cv2
import os
import time
from tqdm import tqdm

cls_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def do_categories(cls):
    categories = list()
    for idx, cat in enumerate(cls):
        categories.append({
            'id': idx,
            'name': cat,
            'supercategory': 'None'
        })
    return categories


def converter():
    img_path = '../mchar/mchar_val'
    save_json = './mchar_coco_val.json'
    with open('../mchar/mchar_val.json', 'r') as fr:
        dumps = json.load(fr)
    fr.close()
    annotations = list()
    images = list()
    cat_id = 0
    info = {
        'year': 2021,
        'version': '1.0',
        "description": 'mcahr',
        "contributor": 'ttjjmm',
        'url': 'none',
        'date_created': '2021/03/12'
    }

    categories = do_categories(cls_name)
    for k, v in tqdm(dumps.items()):
        img_id = 100000 + int(k.split('.')[0])
        h, w = cv2.imread(os.path.join(img_path, k)).shape[:2]
        # converting images
        images.append({
            'id': img_id,
            'file_name': k,
            'width': w,
            'height': h
        })
        # converting annotations
        annoes = [v['left'], v['top'], v['width'], v['height'], v['label']]
        for idx, anno in enumerate(zip(*annoes)):
            cat_id += 1
            annotations.append({
                'id': cat_id,
                'image_id': img_id,
                'category_id': anno[-1],
                'iscrowd': 0,
                'ignore': 0,
                'area': anno[2] * anno[3],
                'segmentation': None,
                'bbox': list(anno[:4])
            })

    coco_json = {
        'info': info,
        'images': images,
        'type': 'instance',
        'annotations': annotations,
        'categories': categories
    }

    with open(save_json, 'w') as fw:
        json.dump(coco_json, fw)
    fw.close()

if __name__ == '__main__':
    converter()
