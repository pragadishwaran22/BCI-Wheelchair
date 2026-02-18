"""scripts/avg_and_eval.py
Average model weights from multiple checkpoints into a single model and evaluate.
"""
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
from sklearn.model_selection import train_test_split

MODEL_PATHS = [
    'models/sweep_top_a_lr5e-4_wd5e-4.pth',
    'models/sweep_top_b_lr3e-4_wd1e-3.pth',
    'models/sweep_top_c_lr7e-4_wd2e-3.pth'
]
DATA_DIR = 'data'


def load_all_data(seq_len=1000, overlap=700):
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    all_epochs = []
    all_labels = []
    for csv_file in sorted(csv_files):
        file_path = os.path.join(DATA_DIR, csv_file)
        epochs, labels = load_csv_file(file_path, seq_len=seq_len, overlap=overlap)
        all_epochs.append(epochs)
        all_labels.append(labels)

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # normalize
    X2 = []
    for e in X:
        X2.append(normalize_epoch(e))
    X = np.array(X2).astype('float32')

    # split
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    except Exception:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), X.shape[1:]


def average_state_dicts(paths):
    state_dicts = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Model not found: {p}')
        state = torch.load(p, map_location='cpu')
        state_dicts.append(state)

    # ensure keys align
    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        if sd.keys() != keys:
            raise ValueError('Checkpoint keys do not match; cannot average')

    avg_state = {}
    n = len(state_dicts)
    for k in keys:
        # sum tensors
        tensors = [sd[k].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)
        avg = torch.mean(stacked, dim=0)
        avg_state[k] = avg

    return avg_state


def evaluate_model(model, X_test, y_test, device='cpu'):
    model.to(device)
    model.eval()
    ds = BCI2aDataset(X_test, y_test)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(yb.numpy().tolist())

    acc = (np.array(all_preds) == np.array(all_trues)).mean()
    kappa = cohen_kappa_score(all_trues, all_preds)
    cm = confusion_matrix(all_trues, all_preds)
    precision_macro = precision_score(all_trues, all_preds, average='macro')
    recall_macro = recall_score(all_trues, all_preds, average='macro')
    f1_macro = f1_score(all_trues, all_preds, average='macro')
    per_class_report = classification_report(all_trues, all_preds)
    return acc, kappa, cm, precision_macro, recall_macro, f1_macro, per_class_report


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), ch_shape = load_all_data()
    print('Shapes:', X_train.shape, X_val.shape, X_test.shape)
    n_channels = ch_shape[0]
    num_classes = int(np.unique(np.concatenate([y_train, y_val, y_test])).size)

    # average
    print('Averaging checkpoints:', MODEL_PATHS)
    avg_state = average_state_dicts(MODEL_PATHS)

    # build model and load averaged weights
    model = SimpleEEGNet1D(in_channels=n_channels, n_classes=num_classes, dropout=0.5)
    model.load_state_dict(avg_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    acc, kappa, cm, precision_macro, recall_macro, f1_macro, per_class_report = evaluate_model(model, X_test, y_test, device=device)

    print('\nAVERAGED-MODEL RESULTS')
    print(f'  Accuracy: {acc:.4f} ({acc*100:.2f}%)')
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f'  Precision (macro): {precision_macro:.4f}')
    print(f'  Recall (macro): {recall_macro:.4f}')
    print(f'  F1 (macro): {f1_macro:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('\nPer-class report:\n')
    print(per_class_report)

    # save summary
    out = 'sweep_logs/avg_ensemble_summary.txt'
    with open(out, 'w', encoding='utf-8') as f:
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

    print(f'\nSaved averaged model summary to {out}')


if __name__ == '__main__':
    main()
