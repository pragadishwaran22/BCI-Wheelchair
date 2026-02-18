# scripts/eval_ensemble.py
# Load saved model checkpoints, build ensemble (average logits), evaluate on test set
import os
import sys
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# add repo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_csv_file, BCI2aDataset
from src.preprocess import normalize_epoch
from src.classifier import SimpleEEGNet1D
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

MODEL_PATHS = [
    'models/model_a_lr5e-4_wd5e-4.pth',
    'models/model_b_lr3e-4_wd1e-3.pth',
    'models/model_c_lr7e-4_wd2e-3.pth'
]
DATA_DIR = 'data'

# load all CSVs (same logic as main.py)
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
all_epochs = []
all_labels = []
for csv_file in sorted(csv_files):
    file_path = os.path.join(DATA_DIR, csv_file)
    epochs, labels = load_csv_file(file_path, seq_len=1000, overlap=700)
    all_epochs.append(epochs)
    all_labels.append(labels)

X = np.concatenate(all_epochs, axis=0)
y = np.concatenate(all_labels, axis=0)

# normalize same way
X2 = []
for e in X:
    X2.append(normalize_epoch(e))
X = np.array(X2).astype('float32')

# split (use same seeds/stratify)
try:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
except Exception:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Shapes:', X.shape, y.shape)
print('Train/Val/Test:', len(X_train), len(X_val), len(X_test))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

n_channels = X.shape[1]
num_classes = int(np.unique(y).size)

# build models and load weights
models = []
for p in MODEL_PATHS:
    if not os.path.isfile(p):
        raise FileNotFoundError(f'Model not found: {p}')
    m = SimpleEEGNet1D(in_channels=n_channels, n_classes=num_classes, dropout=0.5)
    state = torch.load(p, map_location='cpu')
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    models.append(m)

# create test loader
test_ds = BCI2aDataset(X_test, y_test)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

all_preds = []
all_trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits_sum = None
        for m in models:
            logits = m(xb)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits
        # average logits
        logits_avg = logits_sum / len(models)
        preds = logits_avg.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_trues.extend(yb.numpy().tolist())

acc = (np.array(all_preds) == np.array(all_trues)).mean()
kappa = cohen_kappa_score(all_trues, all_preds)
cm = confusion_matrix(all_trues, all_preds)

# additional metrics
precision_macro = precision_score(all_trues, all_preds, average='macro')
recall_macro = recall_score(all_trues, all_preds, average='macro')
f1_macro = f1_score(all_trues, all_preds, average='macro')
per_class_report = classification_report(all_trues, all_preds)

print('\nENSMBLE RESULTS')
print(f'  Accuracy: {acc:.4f} ({acc*100:.2f}%)')
print(f"  Cohen's Kappa: {kappa:.4f}")
print(f'  Precision (macro): {precision_macro:.4f}')
print(f'  Recall (macro): {recall_macro:.4f}')
print(f'  F1 (macro): {f1_macro:.4f}')
print('Confusion Matrix:')
print(cm)
print('\nPer-class report:\n')
print(per_class_report)

# Save results
with open('sweep_logs/ensemble_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'paths: {MODEL_PATHS}\n')
    f.write(f'Accuracy: {acc:.4f}\n')
    f.write(f'Kappa: {kappa:.4f}\n')
    f.write(f'Precision_macro: {precision_macro:.4f}\n')
    f.write(f'Recall_macro: {recall_macro:.4f}\n')
    f.write(f'F1_macro: {f1_macro:.4f}\n')
    f.write('Confusion matrix:\n')
    f.write(str(cm) + '\n')
    f.write('\nPer-class report:\n')
    f.write(per_class_report + '\n')

print('\nSaved ensemble summary to sweep_logs/ensemble_summary.txt')
