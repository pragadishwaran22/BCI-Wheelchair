import argparse
import glob
import os
import sys
import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.gan.generator import Generator


def find_latest_checkpoint(exp_dir: str) -> str:
    pattern = os.path.join(exp_dir, "G_epoch*.pth")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="",
                        help="Path to generator checkpoint .pth. If empty, auto-detect latest in experiments/.")
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--n_channels", type=int, default=22)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--allow_untrained", action="store_true",
                        help="If no checkpoint found, still visualize untrained generator output.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt.strip()
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(os.path.join(PROJECT_ROOT, "experiments"))

    if not ckpt_path or not os.path.isfile(ckpt_path):
        msg = "No checkpoint found. Use --ckpt to provide a .pth file, or run training to create one."
        if not args.allow_untrained:
            print(msg)
            return
        else:
            print(msg + " Proceeding with untrained generator (--allow_untrained).")

    print(f"Using checkpoint: {ckpt_path if ckpt_path else 'None (untrained)'}")

    G = Generator(z_dim=args.z_dim, n_channels=args.n_channels, seq_len=args.seq_len,
                  num_classes=args.num_classes).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(state)
    G.eval()

    z = torch.randn(args.samples, args.z_dim, device=device)
    labels = torch.arange(0, args.num_classes, device=device)[: args.samples]
    with torch.no_grad():
        synth = G(z, labels).cpu().numpy()  # (samples, channels, time)

    print("Generated EEG stats:")
    print("  Min:", float(synth.min()))
    print("  Max:", float(synth.max()))
    print("  Mean:", float(synth.mean()))
    print("  Std:", float(synth.std()))

    # Plot the first sample (transpose to time x channels)
    plt.figure(figsize=(10, 4))
    plot_len = min(500, synth.shape[-1])
    plt.plot(synth[0].T[:plot_len])
    plt.title(f"Generated EEG (first {plot_len} samples, {synth.shape[1]} channels)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()