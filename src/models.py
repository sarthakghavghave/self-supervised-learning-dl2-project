import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
                             
    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        z = self.bottleneck(lstm_out)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 6, kernel_size=5, padding=2)
        )

    def forward(self, x):
        x = self.expand(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        out = self.deconv(lstm_out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mask_token = nn.Parameter(torch.randn(6, 1))

    def apply_mask(self, x, mask_ratio, patch_size):
        B, C, L = x.shape
        num_patches = (L + patch_size - 1) // patch_size
        
        mask = torch.rand(B, 1, num_patches, device=x.device) < mask_ratio
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, patch_size).reshape(B, 1, -1)[:, :, :L]
        mask_expanded = mask_expanded.expand(-1, C, -1)
        
        x_masked = x.clone()
        token_expanded = self.mask_token.expand(-1, L).unsqueeze(0).expand(B, -1, -1)
        x_masked[mask_expanded] = token_expanded[mask_expanded]
        
        return x_masked, mask_expanded

    def forward(self, x, mask_ratio=0.0, patch_size=10):
        mask = None
        if mask_ratio > 0:
            x, mask = self.apply_mask(x, mask_ratio, patch_size)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mask

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=18, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        
        # Enhanced pooling: mean, max, and std-dev provide richer statistical summary
        # Input to head will be 32 (mean) + 32 (max) + 32 (std) = 96
        self.head = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (B, C, L)
        z = self.encoder(x) # (B, L, 32)
        
        # Hierarchical Global Pooling: Captures average behavior, spikes, and variability
        avg_pool = z.mean(dim=1)
        max_pool, _ = z.max(dim=1)
        std_pool = z.std(dim=1)
        
        # Combine statistics
        combined = torch.cat([avg_pool, max_pool, std_pool], dim=1) # (B, 96)
        
        return self.head(combined)

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x