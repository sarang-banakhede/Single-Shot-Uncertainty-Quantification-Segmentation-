import matplotlib.pyplot as plt
import numpy as np

def save_inference_plot(image, alpha, beta, mu, sigma, save_path):
    """
    Saves a 5-panel figure: Input, Alpha, Beta, Mean Prediction, Uncertainty.
    """
    img_np = image.squeeze().cpu().numpy()
    alpha_np = alpha.squeeze().cpu().numpy()
    beta_np = beta.squeeze().cpu().numpy()
    mu_np = mu.squeeze().cpu().numpy()
    sigma_np = sigma.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(alpha_np, cmap='jet')
    axes[1].set_title('Alpha')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(beta_np, cmap='jet')
    axes[2].set_title('Beta')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(mu_np, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Prediction (Mu)')
    axes[3].axis('off')
    
    im4 = axes[4].imshow(sigma_np, cmap='inferno')
    axes[4].set_title('Uncertainty (Sigma^2)')
    axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()