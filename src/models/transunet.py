import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
from torch.nn.modules.utils import _pair
import numpy as np

def np2th(weights, conv=False):
    if conv: weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w, v = self.weight, torch.var_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - v[1]) / torch.sqrt(v[0] + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)
def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin; cmid = cmid or cout//4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6); self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6); self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6); self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if (stride != 1 or cin != cout):
            self.downsample, self.gn_proj = conv1x1(cin, cout, stride, bias=False), nn.GroupNorm(cout, cout)
    def forward(self, x):
        res = self.gn_proj(self.downsample(x)) if hasattr(self, 'downsample') else x
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(res + y)

class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor); self.width = width
        self.root = nn.Sequential(OrderedDict([('conv', StdConv2d(3, width, 7, 2, padding=3, bias=False)), ('gn', nn.GroupNorm(32, width, eps=1e-6)), ('relu', nn.ReLU(inplace=True))]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width, width*4, width))] + [(f'unit{i}', PreActBottleneck(width*4, width*4, width)) for i in range(2, block_units[0] + 1)]))),
            ('block2', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width*4, width*8, width*2, 2))] + [(f'unit{i}', PreActBottleneck(width*8, width*8, width*2)) for i in range(2, block_units[1] + 1)]))),
            ('block3', nn.Sequential(OrderedDict([('unit1', PreActBottleneck(width*8, width*16, width*4, 2))] + [(f'unit{i}', PreActBottleneck(width*16, width*16, width*4)) for i in range(2, block_units[2] + 1)])))
        ]))
    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x); features.append(x); x = nn.MaxPool2d(3, 2, 0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x); features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = config['model']['transunet'].get("grid") is not None
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config['model']['transunet']['resnet_layers'], width_factor=1)
            in_channels = self.hybrid_model.width * 16
        patch_size = _pair(config['model']['transunet']["patch_size"])
        n_patches = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1]) if not self.hybrid else (img_size[0]//16//config['model']['transunet']['grid'][0]) * (img_size[1]//16//config['model']['transunet']['grid'][1])
        self.patch_embeddings = nn.Conv2d(in_channels, config['model']['transunet']['hidden_size'], patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config['model']['transunet']['hidden_size']))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        features = None
        if self.hybrid: x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), features

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.embeddings = Embeddings(config, img_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['model']['transunet']['hidden_size'], nhead=config['model']['transunet']['num_heads'], dim_feedforward=config['model']['transunet']['mlp_dim'], dropout=0.1, activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['transunet']['num_layers'])
        self.norm = nn.LayerNorm(config['model']['transunet']['hidden_size'], eps=1e-6)
    def forward(self, x):
        emb, features = self.embeddings(x)
        return self.norm(self.encoder(emb)), features

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['model']['transunet']
        head_channels = 512
        self.conv_more = Conv2dReLU(self.config['hidden_size'], head_channels, 3, 1, use_batchnorm=True)
        in_ch = [head_channels] + list(self.config['decoder_channels'][:-1])
        out_ch = self.config['decoder_channels']
        skip_ch = self.config['skip_channels']
        self.blocks = nn.ModuleList([DecoderBlock(i, o, s) for i, o, s in zip(in_ch, out_ch, skip_ch)])
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = self.conv_more(hidden_states.transpose(1, 2).view(B, hidden, h, w))
        for i, blk in enumerate(self.blocks):
            skip = features[i] if features is not None and i < len(features) else None
            x = blk(x, skip)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, 1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip: x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class TransUNet(nn.Module):
    """Standard TransUNet"""
    def __init__(self, config, img_size=256, num_classes=1):
        super().__init__()
        self.transformer = Transformer(config, (img_size, img_size))
        self.decoder = DecoderCup(config)
        self.head = nn.Conv2d(config['model']['transunet']['decoder_channels'][-1], num_classes, 3, padding=1)
    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        enc, feats = self.transformer(x)
        return self.head(self.decoder(enc, feats))

class DualTransUNet(nn.Module):
    """Main Version: Two independent TransUNets"""
    def __init__(self, config):
        super().__init__()
        img_size = config['data']['img_size']
        self.model_alpha = TransUNet(config, img_size, 1)
        self.model_beta = TransUNet(config, img_size, 1)
    
    def forward(self, x):
        alpha = F.softplus(self.model_alpha(x)) + 1e-6
        beta = F.softplus(self.model_beta(x)) + 1e-6
        mu = alpha / (alpha + beta)
        sigma_sq = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        return alpha, beta, mu, sigma_sq

class SharedEncoderDualDecoderTransUNet(nn.Module):
    """Efficient Version: Shared Encoder, Two Decoders"""
    def __init__(self, config):
        super().__init__()
        img_size = config['data']['img_size']
        self.transformer = Transformer(config, (img_size, img_size))
        
        self.decoder_alpha = DecoderCup(config)
        self.head_alpha = nn.Conv2d(config['model']['transunet']['decoder_channels'][-1], 1, 3, padding=1)
        
        self.decoder_beta = DecoderCup(config)
        self.head_beta = nn.Conv2d(config['model']['transunet']['decoder_channels'][-1], 1, 3, padding=1)
        
    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        
        enc, feats = self.transformer(x)
        
        d_alpha = self.decoder_alpha(enc, feats)
        alpha = F.softplus(self.head_alpha(d_alpha)) + 1e-6
    
        d_beta = self.decoder_beta(enc, feats)
        beta = F.softplus(self.head_beta(d_beta)) + 1e-6
        
        mu = alpha / (alpha + beta)
        sigma_sq = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        return alpha, beta, mu, sigma_sq