import torch
import copy
import torch.nn as nn
from src.models import AutoEncoder

def mask_input(X, mask_ratio=0.3, patch_size=10, mask_token=None):
    B, C, L = X.shape
    num_patches = (L + patch_size - 1) // patch_size
    
    mask = torch.rand(B, 1, num_patches, device=X.device) < mask_ratio
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, patch_size).reshape(B, 1, -1)[:, :, :L]
    mask_expanded = mask_expanded.expand(-1, C, -1)
    
    X_masked = X.clone()
    if mask_token is not None:
        token_expanded = mask_token.expand(-1, L).unsqueeze(0).expand(B, -1, -1)
        X_masked_vals = token_expanded
        X_masked[mask_expanded] = X_masked_vals[mask_expanded]
    else:
        X_masked[mask_expanded] = 0
        
    return X_masked, mask_expanded

def train_autoencoder(loader, epochs=10, lr=1e-3, device="cpu", masked=False, mask_ratio=0.3, val_loader=None, patience=6):
    model = AutoEncoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    best_loss = float("inf")
    best_state = None
    history = {"train": [], "val": []}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            X = X.to(device)
            
            optimizer.zero_grad()
            m_ratio = mask_ratio if masked else 0.0
            X_hat, mask = model(X, mask_ratio=m_ratio)

            loss = criterion(X_hat, X)
            if masked and mask is not None:
                loss = loss[mask].mean()
            else:
                loss = loss.mean()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history["train"].append(avg_loss)

        metric_to_monitor = avg_loss
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_val in val_loader:
                    if isinstance(batch_val, (list, tuple)):
                        X_val = batch_val[0]
                    else:
                        X_val = batch_val
                    X_val = X_val.to(device)
                    m_ratio_v = mask_ratio if masked else 0.0
                    X_hat_v, mask_v = model(X_val, mask_ratio=m_ratio_v)
                    
                    loss_v = criterion(X_hat_v, X_val)
                    if masked and mask_v is not None:
                        loss_v = loss_v[mask_v].mean()
                    else:
                        loss_v = loss_v.mean()
                    val_loss += loss_v.item()
            avg_val = val_loss / len(val_loader)
            history["val"].append(avg_val)
            metric_to_monitor = avg_val
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val:.4f}")
        else:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if metric_to_monitor < best_loss:
            best_loss = metric_to_monitor
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def evaluate_reconstruction(model, loader, device="cpu", masked=False, mask_ratio=0.3):
    model.eval()
    total_mse = 0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            X = X.to(device)
            m_ratio = mask_ratio if masked else 0.0
            X_hat, mask = model(X, mask_ratio=m_ratio)
            
            mse = nn.functional.mse_loss(X_hat, X, reduction='none')
            if masked and mask is not None:
                mse = mse[mask].sum()
                total_mse += mse.item()
                count += mask.sum().item()
            else:
                total_mse += mse.sum().item()
                count += X.numel()
    
    avg_mse = total_mse / count
    return avg_mse

def evaluate_by_activity(model, loader, device="cpu", masked=False, mask_ratio=0.3):
    model.eval()
    activity_mse = {}
    activity_counts = {}
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X, y = batch
            else:
                continue
                
            X = X.to(device)
            m_ratio = mask_ratio if masked else 0.0
            X_hat, mask = model(X, mask_ratio=m_ratio)
            
            mse = nn.functional.mse_loss(X_hat, X, reduction='none')
            # Flatten to (B, C*L) then mean over features to get per-sample MSE
            mse_per_sample = mse.view(X.shape[0], -1).mean(dim=1)
            
            for i, label in enumerate(y):
                l = label.item()
                if l not in activity_mse:
                    activity_mse[l] = 0
                    activity_counts[l] = 0
                activity_mse[l] += mse_per_sample[i].item()
                activity_counts[l] += 1
                
    results = {l: activity_mse[l] / activity_counts[l] for l in activity_mse}
    return results
