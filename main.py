from tools.config import cfg, load_config
from data.dataset import build_dataset
from model.arch import build_model
from torch.utils import data
from data.collate import collate_center, collate_ttf
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tools.utils import flops_info
import torch.optim as optim
from data.gt_builder.center_gt import center_targets
from tools import plot_lr_scheduler
from icecream import ic
# from data.dataset import build_loader
from tqdm import tqdm
from model.backbone.pp_mbnetv2 import MobileNetV2
from collections import OrderedDict

if __name__ == '__main__':
    # ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/mbv3_large.old.pth.tar', map_location='cpu')['state_dict']
    # for k, v in ckpt.items():
    #     print(k, v.shape)
    # load_config(cfg, 'config/centerdet_csp.yaml')
    # # print(cfg.data)
    # loader = build_loader(cfg.data.train, 'train')
    # for i in tqdm(loader):
    #     pass
    # --------------------------------- test dataset -------------------------------#
    # train_dataset = build_dataset(cfg.data.train, 'train')
    #
    # val_loader = data.DataLoader(train_dataset,
    #                              batch_size=4,
    #                              num_workers=8,
    #                              shuffle=False,
    #                              pin_memory=True,
    #                              collate_fn=collate_ttf)
    #

    # print(k['img'][0].shape)
    # print(k['ht_map'][0].shape)
    # box = k['reg_box'][0]
    # print(k['weight'].shape)
    # print(box[k['weight'][0] > 0])
    # plt.imshow(np.transpose(k['img'][0].numpy(), (1, 2, 0)))
    # plt.show()
    # plt.imshow(torch.sum(k['ht_map'][0], dim=0).numpy())
    # plt.show()
    # exit(1)


    # data = train_dataset[1531]
    # hm = torch.sum(data['ht_map'], dim=0).numpy()
    #
    # # hm = data['ht_map'][0].numpy()
    # # print(hm.shape)
    # img = data['img'].numpy()
    # bbox = data['gt_bboxes']
    # reg = data['reg_box']
    # print(reg.shape)
    # print(reg[:, 45, 45])
    # # print(bbox)
    # img = np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # for i in bbox:
    #     ic(i)
    #     cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (1, 0, 0), 1, cv2.LINE_8)
    # indx = data['weight'][0]
    # # print(indx > 0)
    # # for i in data['reg_box'][:, indx > 0]:
    # #     print(i)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(hm)
    # plt.show()



    # x = torch.sum(data['ht_map'], dim=0)
    # plt.imshow(x)
    # plt.show()
    #--------------------------------------------------------------------------------#

    # model = build_model(cfg.model)
    # inp = torch.randn((4, 3, 320, 320))
    # im = model(inp)
    # print(im[1].shape)

    new_state = OrderedDict()
    ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/MobileNetV2_ssld_pretrained.pth')
    m = MobileNetV2()

    key_map = {
        'conv1._conv.weight': 'conv1._conv.weight',
        'conv1._batch_norm.weight': 'conv1._batch_norm.weight',
        'conv1._batch_norm.bias': 'conv1._batch_norm.bias',
        'conv1._batch_norm.running_mean': 'conv1._batch_norm._mean',
        'conv1._batch_norm.running_var': 'conv1._batch_norm._variance',
    }

    for k, v in m.state_dict().items():

        key_points = k.split('.')
        if key_points[-1] == 'num_batches_tracked':
            new_state[k] = v
            continue

        if key_points[-1] == 'running_mean':
            tail = '_mean'
        elif key_points[-1] == 'running_var':
            tail = '_variance'
        else:
            tail = key_points[-1]

        if k in key_map.keys():
            new_key = key_map[k]
            # new_state[k] = ckpt[key_map[k]]
        elif len(key_points) == 6:
            new_key = 'conv{}.{}.{}.{}.{}'.format(int(key_points[1]) + 2, key_points[2], key_points[3], key_points[4], tail)
        elif len(key_points) == 7:
            conv_num = int(key_points[1]) + 2
            sub_conv_num = int(key_points[3]) + 2
            new_key = 'conv{}.conv{}_{}.{}.{}.{}'.format(conv_num,
                                                         conv_num,
                                                         sub_conv_num,
                                                         key_points[4],
                                                         key_points[5],
                                                         tail)
        else:
            raise RuntimeError

        if new_key not in ckpt:
            print(new_key)
            raise RuntimeError
        else:
            assert v.shape == ckpt[new_key].shape
            print(k)
            new_state[k] = ckpt[new_key]
    m.load_state_dict(new_state, strict=True)
    torch.save(m.state_dict(), './MobileNetV2_ssld_pretrained.pt')