import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEEGNet1D(nn.Module):
    def __init__(self, in_channels=25, n_classes=4, dropout=0.5):
        super().__init__()
        
        # First block - temporal convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, stride=1, padding=32, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4, stride=4),
            nn.Dropout(dropout)
        )
        
        # Second block - separable convolution for spatial filtering
        self.depth_conv = nn.Sequential(
            nn.Conv1d(32, 32*2, kernel_size=16, groups=32, padding=8, bias=False),
            nn.BatchNorm1d(32*2),
            nn.ELU(),
            nn.AvgPool1d(4, stride=4),
            nn.Dropout(dropout)
        )
        
        # Third block - Learn temporal patterns
        self.point_conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification: use adaptive pooling so the model accepts variable input lengths
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # output shape (b, 128, 1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # Apply temporal convolution
        x = self.conv1(x)
        
        # Apply spatial filtering
        x = self.depth_conv(x)
        
        # Learn temporal patterns
        x = self.point_conv1(x)
        
        # Apply attention
        attn = self.attention(x)
        x = x * attn

        # Global pooling + classification (handles variable sequence lengths)
        x = self.global_pool(x).view(x.size(0), -1)  # (batch, 128)
        x = self.classifier(x)

        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class GlobalContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, 1)
        self.norm = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels),
            nn.LayerNorm(channels)
        )

    def forward(self, x):
        b, c, t = x.size()
        # Generate attention weights
        attn = self.conv(x)  # b, 1, t
        attn = self.norm(attn)  # b, 1, t
        
        # Apply attention
        context = torch.bmm(x, attn.transpose(1, 2))  # b, c, 1
        context = context.view(b, c)
        
        # Transform context
        transformed = self.fc(context)
        transformed = transformed.view(b, c, 1)
        
        return x + transformed.expand_as(x)  # residual connection
