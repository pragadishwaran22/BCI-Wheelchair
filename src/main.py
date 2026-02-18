import argparse
import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data_loader import load_csv_file, load_gdf_epochs, BCI2aDataset
from src.preprocess import normalize_epoch
from torch.utils.data import DataLoader
from src.data_loader import BCI2aDataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Optional GAN imports will be used when --with-gan is enabled
from src.gan.train_gan import train_gan_loop
from src.gan.generator import Generator

# high-level flow:
# 1. load epochs for a subject
# 2. preprocess and normalize
# 3. train GAN per-subject
# 4. generate synthetic trials
# 5. train classifier on real+synthetic
# 6. evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to data file or directory (.mat or .gdf format)')
    parser.add_argument('--subject', type=str, default='A01')
    parser.add_argument('--format', type=str, default='auto', 
                       choices=['auto', 'mat', 'gdf'],
                       help='File format (auto detects from extension)')
    parser.add_argument('--multi-subject', action='store_true',
                       help='Load multiple subjects from data directory')
    parser.add_argument('--with-gan', action='store_true',
                       help='Train GAN on training split and augment with synthetic data')
    parser.add_argument('--gan-epochs', type=int, default=100,
                       help='Number of GAN training epochs')
    parser.add_argument('--gan-batch-size', type=int, default=64,
                       help='Override batch size for GAN training')
    parser.add_argument('--synth-per-class', type=int, default=50,
                       help='Number of synthetic samples per class to generate (0 disables augmentation)')
    parser.add_argument('--gan-save-dir', type=str, default='experiments/gan',
                       help='Directory to save GAN checkpoints')
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.02,
                       help='Weight decay for optimizer')
    parser.add_argument('--augment', action='store_true',
                       help='Enable advanced data augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=None,
                       help='Override mixup alpha in config (None leaves default)')
    parser.add_argument('--no-mixup', action='store_true',
                       help='Disable mixup augmentation')
    parser.add_argument('--focal-gamma', type=float, default=None,
                       help='Override focal loss gamma (None leaves default)')
    parser.add_argument('--no-focal', action='store_true',
                       help='Disable focal loss')
    parser.add_argument('--patience', type=int, default=None,
                       help='Override early stopping patience (default in code)')
    parser.add_argument('--save-model-path', type=str, default=None,
                       help='If provided, save the trained model state_dict to this path')
    args = parser.parse_args()

    # Load data based on file extension or directory
    import os
    
    # Check if data path is a directory or file
    if os.path.isdir(args.data):
        # Load multiple subjects
        print(f"Loading multiple subjects from: {args.data}")
        csv_files = [f for f in os.listdir(args.data) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} .csv files")
        
        all_epochs = []
        all_labels = []
        
        for csv_file in sorted(csv_files):
            file_path = os.path.join(args.data, csv_file)
            print(f"\nLoading: {csv_file}")
            epochs, labels = load_csv_file(file_path, seq_len=1000, overlap=700)
            all_epochs.append(epochs)
            all_labels.append(labels)
        
        epochs = np.concatenate(all_epochs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print(f"\nTotal data loaded: {epochs.shape[0]} epochs")
        
    else:
        # Load single file
        file_ext = args.data.split('.')[-1].lower()
        
        if file_ext == 'csv' or args.format == 'csv':
            print(f"Loading .csv file: {args.data}")
            epochs, labels = load_csv_file(args.data, seq_len=1000, overlap=700)
        else:
            print(f"Loading .gdf file: {args.data}")
            event_map = {1: 0, 2: 1, 3: 2, 4: 3}  # map gdf events to class ids
            epochs, labels = load_gdf_epochs(args.data, event_id=event_map)
    # preprocess
    X = []
    for e in epochs:
        X.append(normalize_epoch(e))
    X = np.array(X).astype('float32')
    labels = np.array(labels)
    print(f'Loaded data: {X.shape}, labels: {labels.shape}')
    print(f'Label distribution: {np.bincount(labels)}')
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Split data: Train (70%) / Val (15%) / Test (15%)
    from sklearn.model_selection import train_test_split
    from src.train_classifier import train_classifier
    
    # Try stratified split, fall back to regular split if too few samples
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        print("Stratified split not possible (too few samples per class), using regular split")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, labels, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Configuration
    n_channels = X.shape[1]  # Will be e.g. 22 for BCI2a (no EOG)
    num_classes = int(np.unique(labels).size)
    
    # Adjust batch size based on dataset size
    if len(X) < 100:
        batch_size = 16
        epochs = 50
    elif len(X) < 500:
        batch_size = 32
        epochs = 120
    else:
        batch_size = 32
        epochs = 120
    
    # Build config and allow CLI overrides for common hyperparameters
    cfg = {
        'batch_size': args.batch_size if args.batch_size is not None else batch_size,
        'n_channels': n_channels,
        'num_classes': num_classes,
        'lr': args.lr if args.lr is not None else 0.0005,
        'epochs': args.epochs if args.epochs is not None else epochs,
        'dropout': args.dropout if args.dropout is not None else 0.6,
        'weight_decay': args.weight_decay if args.weight_decay is not None else 0.02,
    'patience': 12,
        'scheduler_patience': 4,
        # Augmentation and loss options (defaults can be overridden via CLI)
        'augment': args.augment,
        'augment_prob': 0.5,
        'mixup': True,
        'mixup_prob': 0.5,
        'mixup_alpha': 0.2,
        'use_focal': True,
        'focal_gamma': 2.0,
        'focal_use_class_weights': True
    }

    # Apply CLI overrides for mixup/focal if present
    if args.mixup_alpha is not None:
        cfg['mixup_alpha'] = float(args.mixup_alpha)
    if args.no_mixup:
        cfg['mixup'] = False
    if args.focal_gamma is not None:
        cfg['focal_gamma'] = float(args.focal_gamma)
    if args.no_focal:
        cfg['use_focal'] = False
    if args.patience is not None:
        cfg['patience'] = int(args.patience)
    print(f"Using {n_channels} EEG channels")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

    # =========================
    # Optional: Train GAN + Augment
    # =========================
    if args.with_gan and args.synth_per_class > 0:
        print("\n" + "="*60)
        print("TRAINING GAN FOR SYNTHETIC AUGMENTATION")
        print("="*60)

        # Prepare real training loader
        train_ds_real = BCI2aDataset(X_train, y_train)
        gan_bs = cfg['batch_size'] if args.gan_batch_size <= 0 else args.gan_batch_size
        train_loader_real = DataLoader(train_ds_real, batch_size=gan_bs, shuffle=True)

        # GAN config
        # For STFT with n_fft=256, onesided -> n_freq_bins = 256//2 + 1 = 129
        n_fft = 256
        n_freq_bins = n_fft // 2 + 1
        gan_cfg = {
            'z_dim': 128,
            'n_channels': n_channels,
            'seq_len': X_train.shape[2],
            'num_classes': 4,
            'n_freq_bins': n_freq_bins,
            'lr': 2e-4,
            'epochs': max(1, int(args.gan_epochs)),
            'lambda_adv': 1.0,
            'lambda_spec': 10.0,
            'save_every': 10
        }

        os.makedirs(args.gan_save_dir, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GAN config: z_dim={gan_cfg['z_dim']}, n_channels={gan_cfg['n_channels']}, seq_len={gan_cfg['seq_len']}, n_freq_bins={gan_cfg['n_freq_bins']}, epochs={gan_cfg['epochs']}, batch_size={gan_bs}")
        try:
            train_gan_loop(train_loader_real, device, args.gan_save_dir, gan_cfg)
        except Exception as e:
            # Print helpful diagnostics to aid debugging
            print("\n[ERROR] GAN training failed.")
            print(f"Data shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Device: {device}")
            print(f"GAN batch size: {gan_bs}")
            print(f"Exception: {type(e).__name__}: {str(e)}")
            raise

        # Load latest generator checkpoint if available; otherwise, keep current weights
        # Try highest epoch multiple of save_every up to gan_epochs
        chosen_epoch = None
        for ep in range(gan_cfg['epochs'], 0, -1):
            if ep % gan_cfg['save_every'] == 0 or ep == gan_cfg['epochs']:
                candidate = os.path.join(args.gan_save_dir, f"G_epoch{ep}.pth")
                if os.path.isfile(candidate):
                    chosen_epoch = ep
                    g_ckpt_path = candidate
                    break

        G = Generator(z_dim=gan_cfg['z_dim'], n_channels=n_channels, seq_len=X_train.shape[2], num_classes=4).to(device)
        if 'g_ckpt_path' in locals():
            print(f"Loading generator checkpoint: {g_ckpt_path}")
            G.load_state_dict(torch.load(g_ckpt_path, map_location=device))
        else:
            print("No generator checkpoint found; using current trained weights in memory (if any)")

        G.eval()

        # Generate synthetic samples per class and concatenate to training set
        synth_X_list, synth_y_list = [], []
        per_class = int(args.synth_per_class)
        for cls in range(4):
            z = torch.randn(per_class, gan_cfg['z_dim'], device=device)
            labels_t = torch.full((per_class,), cls, dtype=torch.long, device=device)
            with torch.no_grad():
                x_fake = G(z, labels_t).cpu().numpy().astype('float32')
            synth_X_list.append(x_fake)
            synth_y_list.append(np.full(per_class, cls, dtype='int64'))

        X_synth = np.concatenate(synth_X_list, axis=0)
        y_synth = np.concatenate(synth_y_list, axis=0)

        print(f"Generated synthetic samples: {X_synth.shape[0]} ({per_class} per class)")
        # Mix only into training set
        X_train = np.concatenate([X_train, X_synth], axis=0)
        y_train = np.concatenate([y_train, y_synth], axis=0)
        print(f"Augmented training set: {len(X_train)} samples")
    
    # Train the classifier
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    model = train_classifier(X_train, y_train, X_val, y_val, cfg, device)
    # Optionally save the trained model
    if args.save_model_path:
        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        try:
            torch.save(model.state_dict(), args.save_model_path)
            print(f"Saved trained model to: {args.save_model_path}")
        except Exception as e:
            print(f"Failed to save model to {args.save_model_path}: {e}")
    
    # Final evaluation on test set
    
    test_ds = BCI2aDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(p.tolist())
            test_trues.extend(yb.cpu().numpy().tolist())
    
    test_acc = (np.array(test_preds) == np.array(test_trues)).mean()
    test_kappa = cohen_kappa_score(test_trues, test_preds)
    try:
        test_precision = precision_score(test_trues, test_preds, average='macro')
        test_recall = recall_score(test_trues, test_preds, average='macro')
        test_f1 = f1_score(test_trues, test_preds, average='macro')
        test_report = classification_report(test_trues, test_preds)
    except Exception:
        test_precision = test_recall = test_f1 = 0.0
        test_report = ''
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS:")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Cohen's Kappa: {test_kappa:.4f}")
    print(f"  Precision (macro): {test_precision:.4f}")
    print(f"  Recall (macro): {test_recall:.4f}")
    print(f"  F1 (macro): {test_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(test_trues, test_preds))
    if test_report:
        print('\nPer-class report:\n')
        print(test_report)
    print(f"{'='*60}\n")
