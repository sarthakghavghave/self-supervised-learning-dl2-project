import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from scripts.load import PROCESSED_DIR, MODEL_DIR, make_loader, sample_labels
from src.models import Encoder, BaselineCNN
from scripts.train_cls import train_classifier, train_baseline

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

data = np.load(PROCESSED_DIR / 'splits.npz')
X_train = data['X_train']
X_val   = data['X_val']
X_test  = data['X_test']
y_train = data['y_train']
y_val   = data['y_val']
y_test  = data['y_test']

test_loader = make_loader(X_test, y_test, batch_size=128)

def evaluate(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

fractions = [0.01, 0.05, 0.10, 0.50, 1.0] 
results = {
    'BaselineCNN': {'acc': [], 'f1': []},
    'AE': {'acc': [], 'f1': []},
    'MAE': {'acc': [], 'f1': []}
}
device = "cpu"

for frac in fractions:
    print(f"\n{'='*40}")
    print(f" EXPERIMENT: {frac*100:g}% of Labeled Pool (n={int(len(X_val)*frac)})")
    print(f"{'='*40}")

    X_small, y_small = sample_labels(X_val, y_val, fraction=frac)    
    train_loader_small = make_loader(X_small, y_small, batch_size=128, shuffle=True)

    epochs = 30 if frac <= 0.05 else 25 if frac <= 0.1 else 20
    print(f"Training for {epochs} epochs...")

    print("\nTraining BaselineCNN...")
    model_baseline = BaselineCNN().to(device)
    model_baseline, _, _, _ = train_baseline(model_baseline, train_loader_small, test_loader, epochs=epochs, device=device)
    acc, f1 = evaluate(model_baseline, test_loader, device=device)
    results['BaselineCNN']['acc'].append(acc)
    results['BaselineCNN']['f1'].append(f1)
    
    print("\nTraining AE Classifier...")
    encoder_ae = Encoder()
    encoder_ae.load_state_dict(torch.load(MODEL_DIR / "encoder_ae.pth", weights_only=True))
    model_ae, _, _, _ = train_classifier(encoder_ae, train_loader_small, test_loader, epochs=epochs, device=device)
    acc, f1 = evaluate(model_ae, test_loader, device=device)
    results['AE']['acc'].append(acc)
    results['AE']['f1'].append(f1)

    print("\nTraining MAE Classifier...")
    encoder_mae = Encoder()
    encoder_mae.load_state_dict(torch.load(MODEL_DIR / "encoder_mae.pth", weights_only=True))
    model_mae, _, _, _ = train_classifier(encoder_mae, train_loader_small, test_loader, epochs=epochs, device=device)
    acc, f1 = evaluate(model_mae, test_loader, device=device)
    results['MAE']['acc'].append(acc)
    results['MAE']['f1'].append(f1)

print("\nPlotting results...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(fractions, results['BaselineCNN']['acc'], marker='o', linestyle='--', color='gray', label='Baseline (Scratch)')
plt.plot(fractions, results['AE']['acc'], marker='s', label='Encoder (AE)')
plt.plot(fractions, results['MAE']['acc'], marker='D', linewidth=2, label='Encoder (MAE)')
plt.xscale('log')
plt.xticks(fractions, [f"{f*100:g}%" for f in fractions])
plt.xlabel('Fraction of Labeled Data (Log Scale)')
plt.ylabel('Test Accuracy')
plt.title('Sample Efficiency: Accuracy')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(fractions, results['BaselineCNN']['f1'], marker='o', linestyle='--', color='gray', label='Baseline (Scratch)')
plt.plot(fractions, results['AE']['f1'], marker='s', label='Encoder (AE)')
plt.plot(fractions, results['MAE']['f1'], marker='D', linewidth=2, label='Encoder (MAE)')
plt.xscale('log')
plt.xticks(fractions, [f"{f*100:g}%" for f in fractions])
plt.xlabel('Fraction of Labeled Data (Log Scale)')
plt.ylabel('Macro F1 Score')
plt.title('Sample Efficiency: F1 Score')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.suptitle("Strictly Disjoint Evaluation: Self-Supervised vs. Baseline", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figures/disjoint_data_evaluation.png')
print("Saved comparison graph to figures/disjoint_data_evaluation.png")
