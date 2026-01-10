from argparse import Namespace

class cfg_model_mynewmodel(Namespace):
    def __init__(self):
        Namespace.__init__(self)
        self.model = Namespace()
        
        # Tên này PHẢI KHỚP với tên function được decorate @MODEL.register_module
        self.model.name = 'mynewmodel' 
        
        # Các tham số truyền vào __init__ của MyNewModel
        self.model.kwargs = dict(
            pretrained=False, 
            checkpoint_path='', 
            strict=True
        )