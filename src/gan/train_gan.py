# src/gan/train_gan.py
import torch
from torch import optim
import torch.nn.functional as F
from .generator import Generator
from .discriminator_time import DiscriminatorTime
from .discriminator_freq import DiscriminatorFreq

# helper to compute PSD (on torch tensors) using short-time welch approximate

def compute_psd_torch(x, sfreq=250, n_fft=256, hop_length=None, win_length=None):
    # x: (batch, channels, time)
    if hop_length is None:
        hop_length = n_fft // 2
    if win_length is None:
        win_length = n_fft
    batch, channels, time = x.shape
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    # reshape to (batch*channels, time) for torch.stft
    x2 = x.reshape(batch * channels, time)
    stft = torch.stft(
        x2,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    # stft: (batch*channels, freq_bins, frames)
    psd = (stft.abs() ** 2).mean(dim=-1)  # (batch*channels, freq_bins)
    psd = psd.reshape(batch, channels, psd.size(-1))  # (batch, channels, freq_bins)
    return psd


def train_gan_loop(real_loader, device, save_path, cfg):
    G = Generator(z_dim=cfg['z_dim'], n_channels=cfg['n_channels'], seq_len=cfg['seq_len'], num_classes=cfg['num_classes']).to(device)
    D_time = DiscriminatorTime(n_channels=cfg['n_channels'], seq_len=cfg['seq_len']).to(device)
    D_freq = DiscriminatorFreq(n_channels=cfg['n_channels'], n_freq_bins=cfg['n_freq_bins']).to(device)

    optG = optim.Adam(G.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    optD_time = optim.Adam(D_time.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    optD_freq = optim.Adam(D_freq.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))

    for epoch in range(cfg['epochs']):
        for real_x, labels in real_loader:
            real_x = real_x.to(device)  # (batch, channels, time)
            labels = labels.to(device)

            batch = real_x.size(0)
            # ============ Train Discriminators ============
            z = torch.randn(batch, cfg['z_dim'], device=device)
            fake = G(z, labels)

            # Time discriminator
            optD_time.zero_grad()
            real_out_t = D_time(real_x)
            fake_out_t = D_time(fake.detach())
            lossD_time = F.relu(1.0 - real_out_t).mean() + F.relu(1.0 + fake_out_t).mean()
            lossD_time.backward()
            optD_time.step()

            # Frequency discriminator: compute PSD
            optD_freq.zero_grad()
            real_psd = compute_psd_torch(real_x)
            fake_psd = compute_psd_torch(fake.detach())
            real_out_f = D_freq(real_psd)
            fake_out_f = D_freq(fake_psd)
            lossD_freq = F.relu(1.0 - real_out_f).mean() + F.relu(1.0 + fake_out_f).mean()
            lossD_freq.backward()
            optD_freq.step()

            # ============ Train Generator ============
            optG.zero_grad()
            fake = G(z, labels)
            out_t = D_time(fake)
            fake_psd = compute_psd_torch(fake)
            out_f = D_freq(fake_psd)
            # adversarial loss: want D outputs to be high (real)
            lossG_adv = -out_t.mean() + -out_f.mean()
            # optionally spectral loss: match PSD L2 to real
            loss_spec = F.mse_loss(fake_psd, compute_psd_torch(real_x))
            lossG = cfg['lambda_adv'] * lossG_adv + cfg['lambda_spec'] * loss_spec
            lossG.backward()
            optG.step()

        print(f"Epoch {epoch}: lossD_time={lossD_time.item():.4f}, lossD_freq={lossD_freq.item():.4f}, lossG={lossG.item():.4f}")
        # save checkpoints periodically
        if (epoch + 1) % cfg.get('save_every', 10) == 0:
            torch.save(G.state_dict(), f"{save_path}/G_epoch{epoch+1}.pth")
