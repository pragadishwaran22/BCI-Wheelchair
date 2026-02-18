import numpy as np
import torch
import torch.nn.functional as F

def gaussian_noise(x, std=0.05):
    """Add Gaussian noise for augmentation"""
    return x + torch.randn_like(x) * std * torch.rand(1).item()

def time_shift(x, max_shift=10):
    """Randomly shift signal in time"""
    batch, ch, time = x.shape
    shifts = torch.randint(-max_shift, max_shift+1, (batch,))
    
    shifted = []
    for i in range(batch):
        s = shifts[i].item()
        if s == 0:
            shifted.append(x[i])
        elif s > 0:
            shifted.append(torch.cat([x[i, :, -s:], x[i, :, :-s]], dim=1))
        else:
            s = abs(s)
            shifted.append(torch.cat([x[i, :, s:], x[i, :, :s]], dim=1))
    
    return torch.stack(shifted)

def frequency_mask(x, max_width=20, num_masks=2):
    """Apply frequency domain masking"""
    b, c, t = x.shape
    for _ in range(num_masks):
        width = torch.randint(1, max_width, (1,)).item()
        start = torch.randint(0, c - width, (1,)).item()
        x[:, start:start+width, :] = 0
    return x

def time_mask(x, max_width=20, num_masks=2):
    """Apply time domain masking"""
    b, c, t = x.shape
    for _ in range(num_masks):
        width = torch.randint(1, max_width, (1,)).item()
        start = torch.randint(0, t - width, (1,)).item()
        x[:, :, start:start+width] = 0
    return x

def amplitude_scale(x, min_scale=0.8, max_scale=1.2):
    """Randomly scale the signal amplitude"""
    scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
    return x * scale

def channel_shuffle(x, num_splits=4):
    """Randomly shuffle groups of channels"""
    b, c, t = x.shape
    split_size = c // num_splits
    if split_size < 2:
        return x
        
    x_splits = list(torch.split(x, split_size, dim=1))
    np.random.shuffle(x_splits)
    return torch.cat(x_splits, dim=1)

def mixup(x, y, alpha=0.2):
    """Perform mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def apply_augmentation(x, y=None, prob=0.5):
    """Apply random augmentations with given probability"""
    if np.random.rand() < prob:
        # Add subtle Gaussian noise
        x = x + torch.randn_like(x) * 0.02
    
    if np.random.rand() < prob:
        # Random time shift
        shift = torch.randint(-5, 6, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=-1)
    
    if np.random.rand() < prob:
        # Random scaling
        scale = torch.rand(1).item() * 0.4 + 0.8  # Scale between 0.8 and 1.2
        x = x * scale
    
    if np.random.rand() < 0.3:  # Less aggressive channel dropout
        # Randomly drop out a few channels
        mask = torch.rand(x.shape[1]) > 0.1
        x = x * mask.view(1, -1, 1)
    
    return x if y is None else (x, y)
