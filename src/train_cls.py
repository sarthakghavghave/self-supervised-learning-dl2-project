import torch
import torch.nn as nn
import copy
from src.models import Classifier


def train_classifier(encoder, train_loader, val_loader, epochs=10, lr=1e-3, encoder_lr=5e-4, device="cpu"):
    model = Classifier(encoder).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.head.parameters(), 'lr': lr}
    ], weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None

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

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model