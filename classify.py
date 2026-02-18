import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_loader import load_csv_file, load_gdf_epochs, BCI2aDataset
from src.preprocess import normalize_epoch
from src.classifier import SimpleEEGNet1D

CLASS_NAMES = ['left', 'right', 'foot', 'tongue']

def get_class_name(idx, num_classes):
    if idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class_{idx}"

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/model_a_lr5e-4_wd5e-4.pth')
parser.add_argument('--data', required=True)
parser.add_argument('--gdf', action='store_true')
args = parser.parse_args()

print("=" * 60)
print("EEG CLASSIFICATION")
print("=" * 60)

# Load data
print(f"\n[1/4] Loading data from: {args.data}")
if args.gdf:
    print("   Format: GDF")
    epochs, labels = load_gdf_epochs(args.data, event_id={1:0, 2:1, 3:2, 4:3})
else:
    print("   Format: CSV")
    epochs, labels = load_csv_file(args.data, seq_len=1000, overlap=700)
print(f"   Loaded {len(epochs)} epochs")

# Preprocess
print(f"\n[2/4] Preprocessing data...")
X = np.array([normalize_epoch(e) for e in epochs]).astype('float32')
y_true = np.array(labels)
num_classes_data = len(np.unique(y_true))
print(f"   Shape: {X.shape} (samples, channels, time)")
print(f"   Classes in data: {num_classes_data}")

# Detect number of classes from checkpoint
print(f"\n[3/4] Loading model from: {args.model}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")
checkpoint = torch.load(args.model, map_location=device)
# Detect num_classes from checkpoint (check last layer bias shape)
if 'classifier.3.bias' in checkpoint:
    num_classes_model = checkpoint['classifier.3.bias'].shape[0]
else:
    num_classes_model = num_classes_data
print(f"   Model expects {num_classes_model} classes")
model = SimpleEEGNet1D(in_channels=X.shape[1], n_classes=num_classes_model, dropout=0.5)
model.load_state_dict(checkpoint)
model.to(device).eval()
print("   Model loaded successfully")

# Classify
print(f"\n[4/4] Classifying samples...")
loader = DataLoader(BCI2aDataset(X, y_true), batch_size=64, shuffle=False)
predictions = []
with torch.no_grad():
    for batch_idx, (xb, _) in enumerate(loader):
        preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
        predictions.extend(preds)
        if (batch_idx + 1) % 10 == 0:
            print(f"   Processed {(batch_idx + 1) * 64} samples...")

predictions = np.array(predictions)
accuracy = (predictions == y_true).mean()

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.2%} ({accuracy*100:.2f}%)")
print(f"Total samples: {len(predictions)}")
print(f"\nFirst 10 predictions:")
for i in range(min(10, len(predictions))):
    match = "✓" if predictions[i] == y_true[i] else "✗"
    pred_name = get_class_name(predictions[i], num_classes_model)
    true_name = get_class_name(y_true[i], num_classes_data)
    print(f"  Sample {i+1:3d}: {match} {pred_name:6s} (true: {true_name})")
print("=" * 60)

