import torch
import torch.nn as nn
import os
from collections import OrderedDict

def get_device(config):
    """
    Configures device and handles DataParallel logic based on config.
    Returns: device, device_ids (list or None)
    """
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_ids = config['system'].get('gpu_ids', [])
        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))
        return device, gpu_ids
    return torch.device('cpu'), None

def load_robust_weights(model, checkpoint_path, device):
    """
    Loads weights robustly, handling 'module.' prefix mismatches from DataParallel.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print(f"Weights loaded successfully from {checkpoint_path}")
    return model