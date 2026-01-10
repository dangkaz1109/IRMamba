from util.registry import Registry

# Tạo Registry
MODEL = Registry('Model')

def get_model(cfg):
    # Hàm tiện ích để lấy model từ config
    model = MODEL.get_module(cfg.name)(**cfg.kwargs)
    return model

# IMPORT FILE CỦA BẠN ĐỂ NÓ TỰ ĐĂNG KÝ
from .mynewmodel import mynewmodel