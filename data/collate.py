import torch
import torch.nn.functional as F

import re
from torch._six import container_abcs, string_classes, int_classes


np_str_obj_array_pattern = re.compile(r'[SaUO]')


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_function(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # TODO: support pytorch < 1.3
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            # return collate_function([torch.as_tensor(b) for b in batch])
            return batch
        elif elem.shape == ():  # scalars
            # return torch.as_tensor(batch)
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_function([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_function(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_function(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_center(batch):
    out = {
        'img': list(),
        'ht_map': list(),
        'reg': list(),
        'offset': list(),
        'ind_mask': list(),
        'img_info': list(),
        'warp_matrix': list()
    }
    for idx, item in enumerate(batch):
        for k in out:
            tensor = item[k]
            if k == 'ind_mask':
                ind = torch.zeros(size=(tensor.size(0), 3),
                                  dtype=tensor.dtype).fill_(idx)
                ind[:, 1:] = tensor
                out[k].append(ind)
            else:
                out[k].append(tensor)
            # del tensor
    # binding tensors in one batch
    # out = {k: torch.cat(out[k], dim=0) if k not in ('img', 'ht_map')
    #             else torch.stack(out[k], dim=0) for k in out}
    for k in out.keys():
        if k in ('img', 'ht_map'):
            out[k] = torch.stack(out[k], dim=0)
        elif k in ('reg', 'offset', 'ind_mask'):
            out[k] = torch.cat(out[k], dim=0)
    return out


def collate_ttf(batch):
    out = {
        'img': list(),
        'ht_map': list(),
        'reg_box': list(),
        'weight': list(),
        'img_info': list(),
        'warp_matrix': list(),
        'pos_weight': list()
    }

    out = {k: [i[k] for i in batch] for k in out}

    # binding tensors in one batch
    out = {k: torch.stack(out[k], dim=0) if k not in ('img_info', 'warp_matrix')
                else out[k] for k in out}

    u = {}
    for item in out['img_info']:
        for i in item.keys():
            if i not in u.keys():
                u[i] = list()
            u[i].append(item[i])

    out['img_info'] = u
    # for k in out.keys():
    #     if k in ('img', 'ht_map'):
    #         out[k] = torch.stack(out[k], dim=0)
    #     elif k in ('reg', 'offset', 'ind_mask'):
    #         out[k] = torch.cat(out[k], dim=0)
    return out