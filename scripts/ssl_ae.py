import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import random
from scripts.load import PROCESSED_DIR, MODEL_DIR, make_loader
from scripts.train_ssl import train_autoencoder, evaluate_reconstruction
from src.models import AutoEncoder

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

data = np.load(PROCESSED_DIR / 'splits.npz')
X_train = data['X_train']

train_loader_ssl = make_loader(X_train, batch_size=128, shuffle=True)

print("\nTraining AE...")
ae_model = train_autoencoder(train_loader_ssl, epochs=15, masked=False)
torch.save(ae_model.state_dict(), MODEL_DIR / 'ae_full.pth')
torch.save(ae_model.encoder.state_dict(), MODEL_DIR / 'encoder_ae.pth')

print("\nTraining MAE...")
mae_model = train_autoencoder(train_loader_ssl, epochs=15, masked=True, mask_ratio=0.3)
torch.save(mae_model.state_dict(), MODEL_DIR / 'mae_full.pth')
torch.save(mae_model.encoder.state_dict(), MODEL_DIR / 'encoder_mae.pth')

print("\nEvaluating...")
val_loader_ssl = make_loader(data['X_val'], batch_size=128)

ae_model = AutoEncoder()
ae_model.load_state_dict(torch.load(MODEL_DIR / 'ae_full.pth'))
ae_mse = evaluate_reconstruction(ae_model, val_loader_ssl)
print(f"\nAutoencoder Reconstruction MSE: {ae_mse:.6f}")

mae_model = AutoEncoder()
mae_model.load_state_dict(torch.load(MODEL_DIR / 'mae_full.pth'))
mae_mse = evaluate_reconstruction(mae_model, val_loader_ssl)
print(f"Masked Autoencoder Reconstruction MSE: {mae_mse:.6f}\n")

# Autoencoder Reconstruction MSE: 0.021345
# Masked Autoencoder Reconstruction MSE: 0.200484