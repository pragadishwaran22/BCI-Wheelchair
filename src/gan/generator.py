# src/gan/generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, n_channels=25, seq_len=1000, base_ch=64, num_classes=4):
        super().__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(z_dim + num_classes, base_ch*8, kernel_size=4, stride=1),
            nn.BatchNorm1d(base_ch*8),
            nn.ReLU(True),
            # upsampling blocks to reach seq_len
            nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*4),
            nn.ReLU(True),
            nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_ch*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(base_ch*2, n_channels, kernel_size=4, stride=2, padding=1),
            # final: output shape (batch, n_channels, seq_len_approx)
        )

    def forward(self, z, labels):
        # z: (batch, z_dim, 1) or (batch, z_dim)
        if z.dim() == 2:
            z = z.unsqueeze(-1)
        label_vec = self.label_emb(labels).unsqueeze(-1)  # (batch, num_classes,1)
        x = torch.cat([z, label_vec], dim=1)
        out = self.net(x)
        # crop or pad time axis to seq_len if necessary
        if out.size(-1) > self.seq_len:
            out = out[..., :self.seq_len]
        elif out.size(-1) < self.seq_len:
            out = torch.nn.functional.pad(out, (0, self.seq_len - out.size(-1)))
        return out
