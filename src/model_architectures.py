"""Model architectures for PolymerCode AI plastic resin classification.

Includes:
1. EnhancedCNN: Deep 5-stage Convolutional Neural Network with Residual Skip
   Connections, Batch Normalization, and multi-stage ANN classification head.
2. CustomCNN: Original 4-stage baseline architecture for backwards compatibility.
3. build_vgg16: Transfer-learning VGG-16 baseline with global pooling and custom head.
"""

from __future__ import annotations
import torch
from torch import nn
from torchvision.models import vgg16


# ---------------------------------------------------------------------------
# 1. Enhanced CNN Components (Deeper 5-Stage Architecture + Residual Skips)
# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """Deep double-convolution block with batch normalization and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip / residual projection if channel dimension changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return self.pool(out)


class EnhancedCNN(nn.Module):
    """Deep 5-Stage Convolutional Neural Network for Plastic Resin Code Classification.

    Features:
    - 5 hierarchical stages: 32 -> 64 -> 128 -> 256 -> 512 channels.
    - Residual skip connections in each block to ensure smooth gradient flow on small datasets.
    - Adaptive Global Average Pooling.
    - Multi-stage Artificial Neural Network (ANN) classification head with
      Batch Normalization and Dropout for robust feature generalization.
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112 (3 -> 32 channels)
            ResidualConvBlock(3, 32, pool=True),
            # Stage 2: 112x112 -> 56x56 (32 -> 64 channels)
            ResidualConvBlock(32, 64, pool=True),
            # Stage 3: 56x56 -> 28x28 (64 -> 128 channels)
            ResidualConvBlock(64, 128, pool=True),
            # Stage 4: 28x28 -> 14x14 (128 -> 256 channels)
            ResidualConvBlock(128, 256, pool=True),
            # Stage 5 (New Deep Feature Stage): 14x14 -> 7x7 (256 -> 512 channels)
            ResidualConvBlock(256, 512, pool=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Advanced ANN classifier head with multi-stage feature projection
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# 2. Legacy CustomCNN Components (4-Stage Baseline for Backwards Compatibility)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Legacy 4-block Conv module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Legacy 4-stage Custom CNN baseline (35.8% accuracy)."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# 3. VGG-16 Transfer Learning Baseline
# ---------------------------------------------------------------------------
def build_vgg16(num_classes: int = 7) -> nn.Module:
    """Build VGG-16 with global pooling and lightweight ANN classifier head."""
    model = vgg16(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def build_model_by_key(model_key: str, num_classes: int = 7) -> nn.Module:
    """Factory helper to build any registered model."""
    key = model_key.lower().strip()
    if key in {"enhanced_cnn", "custom_cnn_enhanced"}:
        return EnhancedCNN(num_classes=num_classes)
    if key in {"custom_cnn", "baseline_cnn"}:
        return CustomCNN(num_classes=num_classes)
    if key in {"vgg16", "vgg16_balanced"}:
        return build_vgg16(num_classes=num_classes)
    raise ValueError(f"Unknown model key: {model_key}")
