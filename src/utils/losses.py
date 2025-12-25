import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class BetaBinomialNLL(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, alpha, beta, target):
        eps = 1e-6
        alpha = alpha + eps
        beta = beta + eps
        # Negative Log Likelihood
        log_prob = target * torch.log(alpha / (alpha + beta)) + \
                   (1 - target) * torch.log(beta / (alpha + beta))
        return -log_prob.mean()

class CombinedLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_nll=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.nll_loss = BetaBinomialNLL()
        self.lambda_seg = lambda_seg
        self.lambda_nll = lambda_nll
    
    def forward(self, alpha, beta, mu, target):
        dice = self.dice_loss(mu, target)
        bce = self.bce_loss(mu, target)
        l_seg = dice + bce
        l_nll = self.nll_loss(alpha, beta, target)
        total_loss = self.lambda_seg * l_seg + self.lambda_nll * l_nll
        return total_loss, l_seg, l_nll