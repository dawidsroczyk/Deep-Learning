import torch
import torch.nn as nn

class VGG_S(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_S, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Reduced from 512*7*7 to 256
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        # Reduced configuration: smaller channels and fewer layers
        cfg = [
            32, 32, 'M',       # Block 1: 2 convs, 32 channels
            64, 64, 'M',       # Block 2: 2 convs, 64 channels
            128, 128, 'M',     # Block 3: 2 convs, 128 channels
            256, 256, 'M',     # Block 4: 2 convs, 256 channels
            256, 256, 'M'      # Block 5: 2 convs, 256 channels
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
