import torch
import torch.nn as nn
import copy
from src.models import Classifier


def train_classifier(encoder, train_loader, val_loader, epochs=10, lr=1e-3, encoder_lr=1e-5, freeze_epochs=5, device="cpu", patience=12):
    model = Classifier(encoder).to(device)

    # Stage 1: Freeze Encoder (Linear Probing)
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler helps with fine-tuning stability
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Stage 2 Transition: Unfreeze Encoder (Fine-tuning)
        if epoch == freeze_epochs:
            print(f"--- Unfreezing Encoder at Epoch {epoch+1} ---")
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Recreate optimizer with discriminative learning rates
            optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'lr': encoder_lr},
                {'params': model.head.parameters(), 'lr': lr / 2}
            ], weight_decay=1e-4)

        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                total_val_loss += criterion(out, y).item()
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        val_accs.append(acc)
        val_losses.append(total_val_loss / len(val_loader))

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_accs


def train_baseline(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu", patience=10):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        correct, total = 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                total_val_loss += loss.item()
                
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        acc = correct / total
        val_accs.append(acc)
        
        print(f"Baseline Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Baseline Early stopping triggered at epoch {epoch+1}!")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_accs
