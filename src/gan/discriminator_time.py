# src/gan/discriminator_time.py
import torch
import torch.nn as nn

class DiscriminatorTime(nn.Module):
    def __init__(self, n_channels=25, seq_len=1000, base_ch=32, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base_ch, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_ch, base_ch*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(base_ch*2, 1),
        )

    def forward(self, x):
        # x: (batch, channels, time)
        feat = self.net(x)
        out = self.fc(feat)
        return out
