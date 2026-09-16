"""LeNet-5 matching the paper's reference implementation."""

import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0,
                 input_size: int = 28, in_channels: int = 1):
        super().__init__()
        # Dropout disabled matching the benchmark setup
        feature_size = input_size
        for _ in range(2):
            feature_size = (feature_size - 4) // 2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(16 * feature_size * feature_size, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        ]
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
