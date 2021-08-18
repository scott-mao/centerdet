from .one_stage import OneStage

def build_model(model_cfg):
    if model_cfg.arch.name == 'OneStage':
        model = OneStage(model_cfg.arch.backbone, model_cfg.arch.neck, model_cfg.arch.head)
    else:
        raise NotImplementedError
    return model