from util.registry import Registry
MODEL = Registry('Model')

def get_model(cfg):
    model = MODEL.get_module(cfg.name)(**cfg.kwargs)
    return model
from .mynewmodel import mynewmodel
