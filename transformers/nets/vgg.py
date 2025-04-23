import torch
import torch.nn as nn

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

    def forward(self, x):
        return self.block(x)


class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=pool_kernel)
        )

    def forward(self, x):
        return self.block(x)


class Block3(nn.Module):
    def __init__(self, input_size=16384, num_classes=12):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class VGG1D(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.block1_1 = Block1(1, 8)      # Input: (B, 1, 16384) → (B, 8, 4096)
        self.block1_2 = Block1(8, 16)     # → (B, 16, 1024)

        self.block2_1 = Block2(16, 32)    # → (B, 32, 512)
        self.block2_2 = Block2(32, 64)    # → (B, 64, 256)
        self.block2_3 = Block2(64, 128)   # → (B, 128, 128)
        self.block2_4 = Block2(128, 256)  # → (B, 256, 64)
        self.block2_5 = Block2(256, 512)  # → (B, 512, 32)
        self.block2_6 = Block2(512, 1024) # → (B, 1024, 16)

        self.flatten = nn.Flatten()
        self.block3 = Block3(1024 * 16, num_classes)

    def forward(self, x):
        x = self.block1_1(x)
        x = self.block1_2(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)
        x = self.block2_5(x)
        x = self.block2_6(x)

        x = self.flatten(x)
        x = self.block3(x)
        return x
