from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)
# work
cfg.work = CfgNode(new_allowed=True)

cfg.log_dir = './'
cfg.weight_dir = './'
cfg.device = 'cuda:0'
# common params for NETWORK
cfg.model = CfgNode()
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.neck = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
# dataloader
# cfg.dataloader = CfgNode(new_allowed=True)
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 1

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


# if __name__ == '__main__':
#     import sys
#
#     with open(sys.argv[1], 'w') as f:
#         print(cfg, file=f)
