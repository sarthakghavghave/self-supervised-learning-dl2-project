import torch
import copy
import torch.nn as nn
from src.models import AutoEncoder

def mask_input(X, mask_ratio=0.3, patch_size=10, mask_token=None):
    B, C, L = X.shape
    if L % patch_size != 0:
        mask = torch.rand_like(X) < mask_ratio
        X_masked = X.clone()
        if mask_token is not None:
            X_masked[mask] = mask_token.unsqueeze(-1).expand(-1, -1, L)[mask]
        else:
            X_masked[mask] = 0
        return X_masked, mask

    num_patches = L // patch_size
    mask = torch.rand(B, 1, num_patches, device=X.device) < mask_ratio
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, patch_size)
    mask = mask.reshape(B, 1, L).expand(-1, C, -1)
    
    X_masked = X.clone()
    if mask_token is not None:
        mask_token_expanded = mask_token.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, num_patches)
        mask_token_expanded = mask_token_expanded.reshape(B, C, L)
        X_masked[mask] = mask_token_expanded[mask]
    else:
        X_masked[mask] = 0
    return X_masked, mask

def train_autoencoder(loader, epochs=10, lr=1e-3, device="cpu", masked=False, mask_ratio=0.3):
    model = AutoEncoder().to(device)
    mask_token = model.decoder.mask_token if masked else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X in loader:
            X = X.to(device)
            
            if masked:
                X_input, mask = mask_input(X, mask_ratio, mask_token=mask_token)
            else:
                X_input = X
                mask = torch.ones_like(X, dtype=torch.bool)

            optimizer.zero_grad()
            X_hat = model(X_input)

            loss = criterion(X_hat, X)
            loss = loss[mask].mean() if mask.any() else loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model

def evaluate_reconstruction(model, loader, device="cpu"):
    model.eval()
    total_mse = 0
    count = 0
    
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            X_hat = model(X)
            mse = nn.functional.mse_loss(X_hat, X, reduction='sum')
            total_mse += mse.item()
            count += X.numel()
    
    avg_mse = total_mse / count
    return avg_mse