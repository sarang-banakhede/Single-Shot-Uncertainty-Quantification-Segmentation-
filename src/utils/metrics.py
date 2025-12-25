import torch
import numpy as np
from scipy.stats import spearmanr

def calculate_metrics_batch(mu, sigma_sq, target, threshold=0.5):
    """
    Calculates both performance and uncertainty metrics for a batch.
    Returns a dictionary of scalar values.
    """
    pred_bin = (mu > threshold).float()
    target = target.float()
    
    # --- Performance ---
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    
    dice = (2. * intersection + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    # --- Uncertainty & Calibration ---
    
    # 1. Average Uncertainty
    avg_unc = sigma_sq.mean()
    
    # 2. Brier Score (Mean Squared Error for prob vs target)
    brier = ((mu - target) ** 2).mean()
    
    # Flatten for curve-based metrics
    mu_flat = mu.detach().cpu().numpy().flatten()
    target_flat = target.detach().cpu().numpy().flatten()
    sigma_flat = sigma_sq.detach().cpu().numpy().flatten()
    pred_bin_flat = pred_bin.detach().cpu().numpy().flatten()
    
    # 3. ECE (Expected Calibration Error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (mu_flat > bin_lower) & (mu_flat <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(mu_flat[in_bin]) - np.mean(target_flat[in_bin])) * prop_in_bin
            
    # 4. AUSE (Area Under Sparsification Error)
    # Error: Absolute difference between prediction and target
    errors = np.abs(pred_bin_flat - target_flat)
    
    # Sort by uncertainty (descending)
    sorted_indices = np.argsort(-sigma_flat)
    sorted_errors = errors[sorted_indices]
    
    n_pixels = len(sorted_errors)
    # Downsample for speed if needed
    step = max(1, n_pixels // 20)
    sparsification_curve = []
    for i in range(0, n_pixels, step):
        if i < n_pixels:
            sparsification_curve.append(sorted_errors[i:].mean())
    ause = np.trapz(sparsification_curve, dx=1.0/len(sparsification_curve))
    
    # 5. Spearman Rank Correlation (Error vs Uncertainty)
    # We want high uncertainty to correlate with high error.
    # Note: Using a subset if array is too large to speed up training
    if len(errors) > 10000:
        idx = np.random.choice(len(errors), 10000, replace=False)
        corr, _ = spearmanr(errors[idx], sigma_flat[idx])
    else:
        corr, _ = spearmanr(errors, sigma_flat)
        
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'avg_uncertainty': avg_unc.item(),
        'brier_score': brier.item(),
        'ece': float(ece),
        'ause': float(ause),
        'spearman_rho': float(corr)
    }