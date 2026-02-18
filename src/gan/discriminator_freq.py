import torch
import torch.nn as nn

class DiscriminatorFreq(nn.Module):
    def __init__(self, n_freq_bins=100, n_channels=25, base_ch=32):
        super().__init__()
        # Input will be flattened PSD per channel or aggregated across channels
        # We'll accept an input shape (batch, n_channels, n_bins)
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_ch, base_ch*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch*2, 1)
        )

    def forward(self, psd):
        # psd: (batch, n_channels, n_bins)
        return self.net(psd)
