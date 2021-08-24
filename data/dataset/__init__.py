import copy
from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .iteration_batch_sampler import IterationBasedBatchSampler
from data.collate import collate_ttf, collate_center

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler


def build_dataset(cfg, mode):
    if 'loader' in cfg:
        cfg.pop('loader')
    dataset_cfg = copy.deepcopy(cfg)

    name = dataset_cfg.pop('name')
    if name == 'coco':
        return CocoDataset(mode=mode, **dataset_cfg)
    if name == 'xml_dataset':
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError('Unknown dataset type!')


def build_dataloader(cfg, mode, max_iter=300000, start_iter=0):

    # cfg_dataset = cfg.pop('dataset')
    cfg_loader = cfg.pop('loader')

    dataset = build_dataset(cfg, mode=mode)
    if cfg_loader.shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=cfg_loader.batch_size, drop_last=cfg_loader.drop_last)

    if max_iter is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler=batch_sampler, num_iterations=max_iter, start_iter=start_iter)

    data_loader = DataLoader(dataset,
                             num_workers=cfg_loader.num_workers,
                             batch_sampler=batch_sampler,
                             pin_memory=cfg_loader.pin_memory,
                             collate_fn=collate_ttf)

    return data_loader


