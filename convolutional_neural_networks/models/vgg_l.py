import torch
import torch.nn as nn

class VGG_L(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_L, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 768),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(768, 384),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        cfg = [
            # Block 1 - 48 channels
            48, 48, 'M',
            # Block 2 - 96 channels
            96, 96, 'M',
            # Block 3 - 192 channels
            192, 192, 'M',
            # Block 4 - 256 channels (deeper)
            256, 256, 256, 'M'
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