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


if __name__ == '__main__':
    # ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/mbv3_large.old.pth.tar', map_location='cpu')['state_dict']
    # for k, v in ckpt.items():
    #     print(k, v.shape)
    load_config(cfg, 'config/centerdet_csp.yaml')
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

    model = build_model(cfg.model)
    inp = torch.randn((4, 3, 320, 320))
    im = model(inp)
    print(im[1].shape)
