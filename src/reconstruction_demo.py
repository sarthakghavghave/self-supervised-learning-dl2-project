import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.load import PROCESSED_DIR, MODEL_DIR
from src.models import AutoEncoder
from scripts.train_ssl import mask_input

data = np.load(PROCESSED_DIR / 'splits.npz')
X_test = data['X_test']
y_test = data['y_test']

model = AutoEncoder()
model.load_state_dict(torch.load(MODEL_DIR / 'mae_full.pth', weights_only=True))
model.eval()

idx = 100
sample = torch.FloatTensor(X_test[idx:idx+1])

masked_sample, mask = mask_input(sample, mask_ratio=0.4, mask_token=model.decoder.mask_token)

with torch.no_grad():
    reconstructed = model(masked_sample)

plt.figure(figsize=(12, 6))
orig = sample[0, 0].detach().numpy()
mask_attr = masked_sample[0, 0].detach().numpy()
pred = reconstructed[0, 0].detach().numpy()

plt.plot(orig, label='Original Signal', color='black', alpha=0.3, linewidth=2)
plt.plot(pred, label='MAE Reconstruction', color='blue', linewidth=2)
plt.scatter(np.where(mask[0, 0])[0], orig[mask[0, 0]], color='red', s=10, label='Masked Points', zorder=5)

plt.title(f"MAE Reconstruction (Sample {idx})")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/reconstruction_demo.png')
print("Demo saved to figures/reconstruction_demo.png")
