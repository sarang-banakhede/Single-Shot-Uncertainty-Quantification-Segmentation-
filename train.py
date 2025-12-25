import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.data.dataset import get_dataloaders
from src.models import build_model
from src.utils.losses import CombinedLoss
from src.utils.metrics import calculate_metrics_batch
from src.utils.helpers import get_device

def train(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = Path(config['output_dir']) / config['experiment_name']
    save_dir.mkdir(parents=True, exist_ok=True)
    device, gpu_ids = get_device(config)
    print(f"Running on: {device} | GPU IDs: {gpu_ids}")

    train_loader, test_loader = get_dataloaders(config)
    model = build_model(config).to(device)
    
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = CombinedLoss(lambda_seg=1.0, lambda_nll=0.5)
    
    best_dice = 0.0
    train_history = []
    test_history = []
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        model.train()
        train_metrics_sum = {}
        
        pbar = tqdm(train_loader, desc="Training")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            alpha, beta, mu, sigma_sq = model(imgs)
            loss, seg_loss, nll_loss = criterion(alpha, beta, mu, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                batch_metrics = calculate_metrics_batch(mu, sigma_sq, masks)
            
            train_metrics_sum['loss'] = train_metrics_sum.get('loss', 0) + loss.item()
            train_metrics_sum['seg_loss'] = train_metrics_sum.get('seg_loss', 0) + seg_loss.item()
            train_metrics_sum['nll_loss'] = train_metrics_sum.get('nll_loss', 0) + nll_loss.item()
            
            for k, v in batch_metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}
        avg_train_metrics['epoch'] = epoch

        model.eval()
        test_metrics_sum = {}
        
        with torch.no_grad():
            for imgs, masks in tqdm(test_loader, desc="Validation"):
                imgs, masks = imgs.to(device), masks.to(device)
                
                alpha, beta, mu, sigma_sq = model(imgs)
                loss, seg_loss, nll_loss = criterion(alpha, beta, mu, masks)
                batch_metrics = calculate_metrics_batch(mu, sigma_sq, masks)
                
                test_metrics_sum['loss'] = test_metrics_sum.get('loss', 0) + loss.item()
                test_metrics_sum['seg_loss'] = test_metrics_sum.get('seg_loss', 0) + seg_loss.item()
                test_metrics_sum['nll_loss'] = test_metrics_sum.get('nll_loss', 0) + nll_loss.item()
                
                for k, v in batch_metrics.items():
                    test_metrics_sum[k] = test_metrics_sum.get(k, 0) + v
        
        avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}
        avg_test_metrics['epoch'] = epoch
        
        print(f"Val Dice: {avg_test_metrics['dice']:.4f}")

        if avg_test_metrics['dice'] > best_dice:
            best_dice = avg_test_metrics['dice']
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(">>> Saved Best Model")
            
        if epoch % config['training']['save_freq'] == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_dir / checkpoint_name)
            print(f">>> Saved Periodic Checkpoint: {checkpoint_name}")
            
        train_history.append(avg_train_metrics)
        test_history.append(avg_test_metrics)
        
        pd.DataFrame(train_history).to_csv(save_dir / "train_metrics.csv", index=False)
        pd.DataFrame(test_history).to_csv(save_dir / "test_metrics.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    train(args.config)