# src/train_classifier.py
import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import cohen_kappa_score, classification_report, precision_score, recall_score, f1_score

from src.data_loader import BCI2aDataset
from src.classifier import SimpleEEGNet1D
from src.utils import apply_augmentation, mixup


def train_classifier(X_train, y_train, X_val, y_val, cfg, device):
    # Compute class counts and sample weights for balanced sampling
    num_classes = cfg['num_classes']
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    # per-sample weight = 1 / count[label]
    sample_weights = np.array([1.0 / max(1.0, class_counts[int(l)]) for l in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    # Initialize datasets and loaders (use sampler for balanced batches)
    train_ds = BCI2aDataset(X_train, y_train)
    val_ds = BCI2aDataset(X_val, y_val)
    batch_size = cfg.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleEEGNet1D(in_channels=cfg['n_channels'], n_classes=cfg['num_classes'], 
                          dropout=0.5).to(device)
    
    # Initialize weights for better gradient flow
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Use AdamW optimizer with OneCycle policy
    # Use AdamW optimizer with configured lr/weight_decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )

    # OneCycle learning rate scheduler (step per batch)
    steps_per_epoch = max(1, len(train_loader))
    max_lr = cfg.get('max_lr', cfg.get('lr', 1e-3) * 5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=cfg['epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_kappa = -1
    patience_counter = 0
    best_model_state = None
    
    print(f"Training with {len(X_train)} samples (balanced sampler), validating with {len(X_val)} samples")
    print(f"Using device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: AdamW with lr={cfg['lr']}, weight_decay={cfg.get('weight_decay', 0.01)}")
    print("-" * 60)

    # Compute class weights (used by loss if enabled)
    n_samples = float(len(y_train))
    class_weights_np = n_samples / (max(1.0, float(num_classes)) * np.clip(class_counts, 1.0, None))
    class_weights_np = class_weights_np / max(1e-6, class_weights_np.mean())
    class_weights_np = np.clip(class_weights_np, 0.5, 4.0)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    print(f"Class counts: {class_counts.tolist()} | Class weights: {class_weights_np.tolist()}")

    # Focal loss implementation (focuses on hard examples)
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None, reduction='mean'):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.reduction = reduction

        def forward(self, inputs, targets):
            # inputs: logits
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    use_focal = cfg.get('use_focal', True)
    focal_loss_fn = FocalLoss(gamma=cfg.get('focal_gamma', 2.0), weight=class_weights if cfg.get('focal_use_class_weights', True) else None)

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0.0
        train_steps = 0
        train_preds_epoch, train_trues_epoch = [], []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Optionally apply augmentations and mixup
            if cfg.get('augment', False):
                if cfg.get('mixup', False) and np.random.rand() < cfg.get('mixup_prob', 0.5):
                    mixed = mixup(xb, yb, alpha=cfg.get('mixup_alpha', 0.2))
                    xb, y_a, y_b, lam = mixed
                    xb = xb.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    logits = model(xb)
                    # Compute mixup loss (works with focal or CE)
                    if use_focal:
                        loss_a = focal_loss_fn(logits, y_a)
                        loss_b = focal_loss_fn(logits, y_b)
                    else:
                        loss_a = F.cross_entropy(logits, y_a, weight=class_weights, label_smoothing=0.1)
                        loss_b = F.cross_entropy(logits, y_b, weight=class_weights, label_smoothing=0.1)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    xb = apply_augmentation(xb, prob=cfg.get('augment_prob', 0.5))
                    logits = model(xb)
                    if use_focal:
                        loss = focal_loss_fn(logits, yb)
                    else:
                        loss = F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=0.1)
            else:
                logits = model(xb)
                if use_focal:
                    loss = focal_loss_fn(logits, yb)
                else:
                    loss = F.cross_entropy(logits, yb, weight=class_weights, label_smoothing=0.1)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # step scheduler per batch (OneCycleLR expects per-batch stepping)
            try:
                scheduler.step()
            except Exception:
                pass

            train_loss += loss.item()
            train_steps += 1

            # collect train metrics
            p_batch = logits.argmax(dim=1).detach().cpu().numpy()
            train_preds_epoch.extend(p_batch.tolist())
            train_trues_epoch.extend(yb.detach().cpu().numpy().tolist())

        # Evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                p = logits.argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(yb.cpu().numpy().tolist())

        acc = (np.array(preds) == np.array(trues)).mean()
        kappa = cohen_kappa_score(trues, preds)
        # additional metrics
        try:
            precision_macro = precision_score(trues, preds, average='macro')
            recall_macro = recall_score(trues, preds, average='macro')
            f1_macro = f1_score(trues, preds, average='macro')
        except Exception:
            precision_macro = 0.0
            recall_macro = 0.0
            f1_macro = 0.0

        # Compute train metrics for over/underfitting diagnosis
        train_acc = (np.array(train_preds_epoch) == np.array(train_trues_epoch)).mean() if len(train_preds_epoch) else 0.0
        train_kappa = cohen_kappa_score(train_trues_epoch, train_preds_epoch) if len(train_preds_epoch) else 0.0

        # Note: scheduler stepped per batch; get current lr for logging
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{cfg['epochs']}: train_loss={train_loss/train_steps:.4f}, "
              f"train_acc={train_acc:.4f}, train_kappa={train_kappa:.4f}, "
              f"val_acc={acc:.4f}, kappa={kappa:.4f}, "
              f"prec={precision_macro:.4f}, rec={recall_macro:.4f}, f1={f1_macro:.4f}, "
              f"lr={current_lr:.6f}")

        # Early stopping based on kappa
        if kappa > best_kappa:
            best_kappa = kappa
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  >>> New best kappa: {best_kappa:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.get('patience', 15):
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed. Best kappa: {best_kappa:.4f}")
    return model
