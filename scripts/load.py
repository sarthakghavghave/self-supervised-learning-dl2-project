import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset/raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_NAMES = {
    0: "Stand",       1: "Sit",          2: "Talk-sit",
    3: "Talk-stand",  4: "Stand-sit",    5: "Lay",
    6: "Lay-stand",   7: "Pick",         8: "Jump",
    9: "Push-up",    10: "Sit-up",      11: "Walk",
   12: "Walk-back",  13: "Walk-circle", 14: "Run",
   15: "Stair-up",   16: "Stair-down",  17: "Table-tennis",
}

def make_confusion(model, loader, title):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device="cpu")
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ACTIVITY_NAMES.values(), 
                yticklabels=ACTIVITY_NAMES.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

class HARDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def make_loader(X, y=None, batch_size=128, shuffle=False):
    ds = HARDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def sample_labels(X, y, fraction, random_state=42):
    if fraction >= 1.0:
        return X, y
    _, X_f, _, y_f = train_test_split(
        X, y, test_size=fraction, stratify=y, random_state=random_state
    )
    return X_f, y_f
