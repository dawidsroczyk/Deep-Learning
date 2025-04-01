import torch
import torch.nn as nn

class VGG_OUR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_OUR, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling for 32x32
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # Reduced from 4096
            nn.ReLU(True),
            nn.Dropout(0.3),     # Less dropout for small datasets
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        # Reduced pooling layers (3 instead of 5) and smaller channels
        cfg = [
            64, 64, 'M',  # Block 1: 64ch, 32x32 → 16x16
            128, 128, 'M', # Block 2: 128ch, 16x16 → 8x8
            256, 256, 256, 'M', # Block 3: 256ch, 8x8 → 4x4
            512, 512, 512   # Block 4: 512ch, 4x4 (no final pool)
        ]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)