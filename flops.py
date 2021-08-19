import torch
from model.arch import build_model
from tools import cfg, load_config, get_model_complexity_info


def main(config, input_shape=(3, 320, 320)):
    model = build_model(config.model)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


if __name__ == '__main__':
    cfg_path = 'config/PAFNet_lite.yaml'
    load_config(cfg, cfg_path)
    main(config=cfg, input_shape=(3, 320, 320))
