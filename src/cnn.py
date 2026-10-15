"""
4-block 2D Convolutional Neural Network (AudioCNN2D) in PyTorch for audio deepfake detection.
Designed for 2D spectral audio representations (e.g. 64-bin log-mel spectrograms).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2D(nn.Module):
    """
    Constituent convolutional building block:
    Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D -> (Optional Dropout2D).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class AudioCNN2D(nn.Module):
    """
    4-Block PyTorch Convolutional Neural Network for 2D Audio Deepfake Classification.

    Architecture:
      Input: (Batch, Channels, Mel_Bins, Time_Frames) or (Batch, Mel_Bins, Time_Frames)
      - Block 1: ConvBlock2D(in_channels -> 32)
      - Block 2: ConvBlock2D(32 -> 64)
      - Block 3: ConvBlock2D(64 -> 128)
      - Block 4: ConvBlock2D(128 -> 256)
      - Global Pooling: AdaptiveAvgPool2d((1, 1))
      - Classifier: Flatten -> Dropout -> Linear(256, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 4 Convolutional Blocks
        self.block1 = ConvBlock2D(in_channels, 32, dropout=0.1)
        self.block2 = ConvBlock2D(32, 64, dropout=0.1)
        self.block3 = ConvBlock2D(64, 128, dropout=0.1)
        self.block4 = ConvBlock2D(128, 256, dropout=0.1)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward_features(self, x):
        """Extract spatial and pooled features from 4 conv blocks."""
        # If input has shape (Batch, Mel_Bins, Time), unsqueeze channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        """Standard forward pass returning raw logits."""
        feats = self.forward_features(x)
        # Pass features through classifier (skip initial flatten since forward_features flattens)
        # Apply dropout -> linear -> relu -> dropout -> linear
        out = self.classifier[1:](feats)
        return out

    def predict_proba(self, x):
        """Predict softmax class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x):
        """Predict integer class labels (0: real/bonafide, 1: fake/spoof)."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = AudioCNN2D(in_channels=1, num_classes=2)
    print(f"Instantiated AudioCNN2D architecture with {model.get_num_params():,} parameters.")

    # Test with dummy 64-bin, 4s log-mel spectrogram batch: (4, 64, 401)
    dummy_input = torch.randn(4, 64, 401)
    output = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    preds = model.predict(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {output.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions: {preds.tolist()}")
    print("SUCCESS: AudioCNN2D forward pass verified.")
