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
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
                            
    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        return lstm_out 

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 6)
        
    def forward(self, x):
        out = self.fc(x)
        return out.permute(0, 2, 1)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=18):
        super().__init__()
        self.encoder = encoder
        
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)