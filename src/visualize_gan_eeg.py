import torch
import argparse
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
from gan.generator import Generator  # adjust path if needed


# ---------------------- Load Generator ----------------------
def load_generator(ckpt_path, device):
    print(f"Loading generator from {ckpt_path}")
    gen = Generator(z_dim=512, n_channels=22, seq_len=1000, num_classes=4).to(device)

    state = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


# ---------------------- Load Real EEG ----------------------
def load_real_eeg(mat_path):
    try:
        mat = sio.loadmat(mat_path)
        if 'data' in mat:
            d = mat['data'][0]
            X_all = []
            for i in range(len(d)):
                if isinstance(d[i][0][0][0], np.ndarray):
                    X_all.append(d[i][0][0][0])
            X_all = np.concatenate(X_all, axis=0)
            print(f"Loaded real EEG data shape: {X_all.shape}")
            return X_all
        else:
            raise KeyError("No key 'data' in MAT file.")
    except Exception as e:
        print(f"Could not load/plot comparison MAT file: {e}")
        return None


# ---------------------- Visualization ----------------------
def visualize(gen, num_samples, device, outdir, real_eeg=None):
    os.makedirs(outdir, exist_ok=True)

    # Generate synthetic EEG
    z = torch.randn(num_samples, 512, device=device)
    labels = torch.randint(0, 4, (num_samples,), device=device)
    with torch.no_grad():
        synth = gen(z, labels).cpu().numpy()  # (num, channels, time)

    print(f"Synth shape: {synth.shape}")

    # Plot heatmaps for synthetic EEGs
    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        plt.title(f"Synthetic EEG Heatmap {i+1}")
        plt.imshow(synth[i], aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude (µV)')
        plt.xlabel("Time")
        plt.ylabel("Channels (22)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"synthetic_heatmap_{i+1}.png"))
        plt.close()

    # Plot line waveform for first few channels of one synthetic sample
    plt.figure(figsize=(10, 6))
    for ch in range(min(5, synth.shape[1])):  # plot 5 channels
        plt.plot(synth[0, ch, :], label=f"Ch {ch+1}")
    plt.title("Synthetic EEG Waveform (First 5 Channels)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "synthetic_waveform.png"))
    plt.close()

    # Real EEG comparison
    if real_eeg is not None:
        # Heatmap
        plt.figure(figsize=(10, 6))
        plt.title("Example Real EEG Heatmap")
        plt.imshow(real_eeg[:1000, :22].T, aspect='auto', cmap='plasma')
        plt.colorbar(label='Amplitude (µV)')
        plt.xlabel("Time")
        plt.ylabel("Channels (22)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "real_eeg_heatmap.png"))
        plt.close()

        # Waveform (first 5 channels)
        plt.figure(figsize=(10, 6))
        for ch in range(min(5, 22)):
            plt.plot(real_eeg[:1000, ch], label=f"Ch {ch+1}")
        plt.title("Real EEG Waveform (First 5 Channels)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "real_eeg_waveform.png"))
        plt.close()

    print(f"Plots saved to {outdir}")


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to generator checkpoint")
    parser.add_argument("--num", type=int, default=6, help="Number of samples to generate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare", default=None, help="Path to .mat EEG file for comparison")
    parser.add_argument("--outdir", default="gan_viz_out")
    args = parser.parse_args()

    gen = load_generator(args.ckpt, device=args.device)
    real_eeg = load_real_eeg(args.compare) if args.compare else None
    visualize(gen, args.num, args.device, args.outdir, real_eeg)


if __name__ == "__main__":
    main()
