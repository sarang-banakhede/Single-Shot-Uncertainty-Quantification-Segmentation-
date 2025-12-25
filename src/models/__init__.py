from .unet import DualUNet, SharedEncoderDualDecoderUNet
from .transunet import DualTransUNet, SharedEncoderDualDecoderTransUNet

def build_model(config):
    """Initializes the requested model architecture."""
    name = config['model']['name']
    
    if name == 'dual_unet':
        return DualUNet()
    elif name == 'shared_unet':
        return SharedEncoderDualDecoderUNet()
    elif name == 'dual_transunet':
        return DualTransUNet(config)
    elif name == 'shared_transunet':
        return SharedEncoderDualDecoderTransUNet(config)
    else:
        raise ValueError(f"Model {name} not found.")