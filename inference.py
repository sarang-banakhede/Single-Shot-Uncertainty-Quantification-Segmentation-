import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.models import build_model
from src.utils.helpers import load_robust_weights

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(image_path, size):
    img = Image.open(image_path).convert('L')
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    return t(img).unsqueeze(0)

def get_model_config(model_name, img_size, n_classes, in_channels):
    """
    Configures the model dictionary based on the specific architecture type.
    """
    base_config = {
        'model': {
            'name': model_name,
            'num_classes': n_classes,
            'in_channels': in_channels,
            'base_filters': 32
        }
    }
    
    if 'TransUNet' in model_name:
        base_config['model'].update({
            'vit_name': 'R50-ViT-B_16',
            'vit_patches_size': 16,
            'n_skip': 3,
            'r50_pretrained': False,
            'img_size': img_size
        })
        
    return base_config

def save_visualization(img, gt, alpha, beta, mu, sigma, save_path, titles):
    has_gt = gt is not None
    n_cols = 6 if has_gt else 5
    
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    axes = axes.flatten()
    
    tensors = [img, alpha, beta, mu, sigma]
    if has_gt:
        tensors.insert(1, gt)
        
    for i, ax in enumerate(axes):
        data = tensors[i].squeeze().cpu().detach().numpy()
        
        if i == 0 or (has_gt and i == 1): 
            cmap = 'gray'
        else:
            cmap = 'inferno'
            
        ax.imshow(data, cmap=cmap)
        if i < len(titles):
            ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def run_inference(img_path, weight_path, save_dir, model_name, gt_path=None, titles=None):
    device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    img_size = 224 if 'TransUNet' in model_name else 256
    
    print(f"Configuring {model_name} with Image Size: {img_size}")
    config = get_model_config(model_name, img_size, 1, 1)
    
    try:
        model = build_model(config).to(device)
        model = load_robust_weights(model, weight_path, device)
        model.eval()
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    files = []
    path_obj = Path(img_path)
    if path_obj.is_dir():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            files.extend(list(path_obj.glob(ext)))
        files = sorted(files)
    else:
        files = [path_obj]

    print(f"Processing {len(files)} images...")

    for f in files:
        try:
            img_t = preprocess(f, img_size).to(device)
            
            gt_t = None
            if gt_path:
                gt_file = Path(gt_path)
                if gt_file.is_dir():
                    possible_gt = gt_file / f.name
                    if possible_gt.exists():
                        gt_t = preprocess(possible_gt, img_size).to(device)
                elif gt_file.exists():
                    gt_t = preprocess(gt_file, img_size).to(device)

            with torch.no_grad():
                out = model(img_t)
                if isinstance(out, (tuple, list)) and len(out) == 4:
                    alpha, beta, mu, sigma = out
                else:
                    print(f"Output mismatch for {f.name}")
                    continue

            final_name = Path(save_dir) / f"{f.stem}_{model_name}_result.png"
            
            current_titles = titles.copy()
            
            if gt_t is None and "Ground Truth" in current_titles:
                current_titles.remove("Ground Truth")
                
            save_visualization(img_t, gt_t, alpha, beta, mu, sigma, final_name, current_titles)
            print(f"Saved: {final_name}")

        except Exception as e:
            print(f"Error on {f.name}: {e}")

if __name__ == '__main__':

    IMAGE_INPUT_PATH = "/home/sarang/Desktop/Nature/Dataset/TN3K/trainval-image/0033.jpg"
    GROUND_TRUTH_PATH = "/home/sarang/Desktop/Nature/Dataset/TN3K/trainval-mask/0033.jpg" 
    
    MODEL_TYPE = 'shared_unet' 
    WEIGHTS_PATH = "/home/sarang/Desktop/Nature/ProbabilisticSegmentation/Outputs/Experiment_Shared_UNet/best_model.pth"
  
    OUTPUT_DIR = "/home/sarang/Desktop/Nature/ProbabilisticSegmentation/Outputs/Experiment_Shared_UNet/Inference_Results"
    
    PLOT_TITLES = ["Input Image", "Ground Truth", "Alpha Map", "Beta Map", "Prediction (Mu)", "Uncertainty (Sigma)"]

    run_inference(
        IMAGE_INPUT_PATH, 
        WEIGHTS_PATH, 
        OUTPUT_DIR, 
        MODEL_TYPE, 
        GROUND_TRUTH_PATH, 
        PLOT_TITLES
    )