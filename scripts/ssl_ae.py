import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scripts.load import PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, make_loader, ACTIVITY_NAMES
from scripts.train_ssl import train_autoencoder, evaluate_reconstruction, evaluate_by_activity
from src.models import AutoEncoder

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

data = np.load(PROCESSED_DIR / 'splits.npz')
X_train_full = data['X_train']
X_val_true = data['X_val']
y_val_true = data['y_val']

X_train, X_val = train_test_split(X_train_full, test_size=0.1, random_state=42) 

train_loader_ssl = make_loader(X_train, batch_size=128, shuffle=True)
val_loader = make_loader(X_val, batch_size=128)
test_loader = make_loader(X_val_true, y=y_val_true, batch_size=128)

def plot_combined_history(ae_hist, mae_hist, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(ae_hist['train'], 'b-', label='AE Train')
    plt.plot(ae_hist['val'], 'b--', label='AE Val')
    plt.plot(mae_hist['train'], 'r-', label='MAE Train')
    plt.plot(mae_hist['val'], 'r--', label='MAE Val')
    plt.title("Learning Curves Comparison: AE vs MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(FIGURE_DIR / filename)
    plt.close()

def plot_activity_performance(ae_results, mae_results, filename):
    activities = sorted(ae_results.keys())
    names = [ACTIVITY_NAMES[a] for a in activities]
    ae_vals = [ae_results[a] for a in activities]
    mae_vals = [mae_results[a] for a in activities]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, ae_vals, width, label='AE', color='royalblue')
    plt.bar(x + width/2, mae_vals, width, label='MAE', color='indianred')
    
    plt.ylabel('MSE')
    plt.title('Reconstruction Error by Activity')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("\nTraining AE...")
ae_model, ae_history = train_autoencoder(train_loader_ssl, epochs=50, device=device, masked=False, val_loader=val_loader, patience=10)
torch.save(ae_model.state_dict(), MODEL_DIR / 'ae_full.pth')
torch.save(ae_model.encoder.state_dict(), MODEL_DIR / 'encoder_ae.pth')

print("\nTraining MAE...")
mae_model, mae_history = train_autoencoder(train_loader_ssl, epochs=100, device=device, masked=True, mask_ratio=0.5, val_loader=val_loader, patience=15)
torch.save(mae_model.state_dict(), MODEL_DIR / 'mae_full.pth')
torch.save(mae_model.encoder.state_dict(), MODEL_DIR / 'encoder_mae.pth')

print("\nGenerating Reporting Visuals...")
plot_combined_history(ae_history, mae_history, 'combined_learning_curves.png')

ae_activity_mse = evaluate_by_activity(ae_model, test_loader, device=device)
mae_activity_mse = evaluate_by_activity(mae_model, test_loader, device=device, masked=False)
plot_activity_performance(ae_activity_mse, mae_activity_mse, 'activity_wise_performance.png')

print("\nFinal Evaluation Metrics:")
ae_mse = evaluate_reconstruction(ae_model, test_loader, device=device)
print(f"Autoencoder MSE (Full Seq): {ae_mse:.6f}")

mae_mse_masked = evaluate_reconstruction(mae_model, test_loader, device=device, masked=True, mask_ratio=0.5)
mae_mse_full = evaluate_reconstruction(mae_model, test_loader, device=device, masked=False)

print(f"MAE MSE (Masked Patches): {mae_mse_masked:.6f}")
print(f"MAE MSE (Full Sequence):  {mae_mse_full:.6f}")
print(f"\nAll visuals saved to {FIGURE_DIR}")
