import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SegmentationDataset(Dataset):
    """
    Generic Dataset for Binary Segmentation tasks.
    
    Expects:
        - image_dir: Directory containing input images (supports .jpg, .jpeg, .png).
        - mask_dir: Directory containing binary masks (same filename as images).
        
    Notes:
        - Images are converted to Grayscale ('L').
        - Masks are thresholded at 0.5 to ensure binary values (0.0 or 1.0).
        - If input images are RGB, change .convert('L') to .convert('RGB') in the code below.
    """
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        extensions = ['*.jpg', '*.jpeg', '*.png']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(glob(os.path.join(image_dir, ext))))
            
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  
        
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, img_name)

        if not os.path.exists(mask_path):
             base_name = os.path.splitext(img_name)[0]
             possible_masks = glob(os.path.join(self.mask_dir, f"{base_name}.*"))
             if possible_masks:
                 mask_path = possible_masks[0]
             else:
                 raise FileNotFoundError(f"Mask for {img_name} not found in {self.mask_dir}")

        mask = Image.open(mask_path).convert('L')
        
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def get_dataloaders(config):
    """Factory function to create DataLoaders based on config."""
    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    num_workers = config['system']['num_workers']
    
    t_img = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)
    t_mask = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST)
    
    train_ds = SegmentationDataset(
        config['data']['train_img_dir'], config['data']['train_mask_dir'], t_img, t_mask
    )
    test_ds = SegmentationDataset(
        config['data']['test_img_dir'], config['data']['test_mask_dir'], t_img, t_mask
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader