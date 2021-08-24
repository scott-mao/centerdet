import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
# import onnx
# from onnxsim import simplify
import logging
from model.arch import build_model
from tools import Logger, cfg, load_config, load_model_weight

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/PAFNet_lite.yaml',
                        help='train config file path')
    parser.add_argument('--model-path', type=str, default='../workspace/PAFNet_lite_exp3',
                        help='model file path')
    parser.add_argument('--save-path', type=str, default='../output/onnx',
                        help='exported onnx file path')
    parser.add_argument('--input_shape', type=tuple, default=(320, 320),
                        help='Model intput shape.')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='onnx model simplify')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


class CenterDet(nn.Module):
    def __init__(self, cfg_info, model_path, logger):
        super(CenterDet, self).__init__()
        self.model = build_model(cfg_info.model)
        checkpoint = torch.load(model_path, map_location='cpu')
        load_model_weight(self.model, checkpoint, logger)

    def forward(self, x):
        hetmap, reg_box = self.model(x)
        hetmap = torch.sigmoid(hetmap)
        hetmap = F.max_pool2d(hetmap, kernel_size=3, padding=1, stride=1)
        return hetmap, reg_box


def export_model():
    args = parse_args()
    load_config(cfg, args.config)
    ckpt_path = os.path.join(args.model_path, 'model_best/model_best.pt')
    model = CenterDet(cfg, ckpt_path, logger).to(args.device)
    model.eval()
    in_w, in_h = args.input_shape
    dummy = torch.zeros((1, 3, in_h, in_w)).to(args.device)

    model_name = os.path.basename(args.config).replace('yaml', 'onnx')
    onnx_path = os.path.join(args.save_path, model_name)

    torch.onnx.export(model, dummy, onnx_path,
                      verbose=True,
                      output_names=['heatmap', 'reg_box'],
                      input_names=['input'],
                      keep_initializers_as_inputs=True,
                      opset_version=11)

    if args.simplify:
        os.system('python3 -m onnxsim {} {}'.format(onnx_path, onnx_path))

# python3 -m onnxsim input_onnx_model output_onnx_model
# def simplify_model():
#     model_path = '../samples/model.onnx'
#     save_path = '../samples/model_sim.onnx'
#     os.system('python3 -m onnxsim {} {}'.format(model_path, save_path))




if __name__ == '__main__':
    export_model()
    # simplify_model()
