import torch
import torch.nn as nn

class Block1(nn.Module):
    """Block1: Conv1D(kernel=80, stride=4) + BN + ReLU + MaxPool1D(4)"""
    def __init__(self, in_channels=1, out_channels=48):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=80, stride=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

    def forward(self, x):
        return self.block(x)

class IdentityBlock(nn.Module):
    """
    Bottleneck-style identity block as per the diagram:
    - Main path: Conv(N) → BN → ReLU → Conv(N) → BN → ReLU → Conv(4N) → BN
    - Shortcut: Conv(4N) → BN (always)
    - Add → ReLU
    """
    def __init__(self, in_channels, bottleneck_channels, kernel_size=9):
        super().__init__()
        out_channels = 4 * bottleneck_channels  # Final output channels

        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),

            nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),

            nn.Conv1d(bottleneck_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        self.block1 = Block1()  # (B, 1, 16384) → (B, 48, 1024)

        # Block group 1: 3x IdentityBlock(48) + MaxPool1D(4) → (B, 48, 256)
        self.group1 = nn.Sequential(
            IdentityBlock(48, 48),
            *[IdentityBlock(192, 48) for _ in range(2)],
            nn.MaxPool1d(4)
        )

        # Block group 2: 4x IdentityBlock(96) + MaxPool1D(4) → (B, 96, 64)
        self.group2 = nn.Sequential(
            IdentityBlock(192, 96),
            *[IdentityBlock(384, 96) for _ in range(3)],
            nn.MaxPool1d(4)
        )

        # Block group 3: 6x IdentityBlock(192) + MaxPool1D(4) → (B, 192, 16)
        self.group3 = nn.Sequential(
            IdentityBlock(384, 192),
            *[IdentityBlock(768, 192) for _ in range(5)],
            nn.MaxPool1d(4)
        )

        # Block group 4: 3x IdentityBlock(384) + GlobalAveragePooling1D → (B, 384)
        self.group4 = nn.Sequential(
            IdentityBlock(768, 384),
            *[IdentityBlock(1536, 384) for _ in range(2)],
            nn.AdaptiveAvgPool1d(1)  # → (B, 384, 1)
        )

        # Final dense layers: (B, 384) → (B, 1536) → (B, 12)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # → (B, 384)
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        # Input: (B, 1, 16384)
        x = self.block1(x)       # → (B, 48, 1024)
        x = self.group1(x)       # → (B, 48, 256)
        x = self.group2(x)       # → (B, 96, 64)
        x = self.group3(x)       # → (B, 192, 16)
        x = self.group4(x)       # → (B, 384, 1)
        x = self.classifier(x)   # → (B, 12)
        return x
