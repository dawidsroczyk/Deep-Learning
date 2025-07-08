# Import necessary libraries
import torch
import torch.nn as nn

class VGG_XS(nn.Module):
    # A smaller variant of the VGG architecture for lightweight applications.
    def __init__(self, num_classes=10):
        # Initialize the VGG_XS model with reduced dimensions and layers.
        super(VGG_XS, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(192, 96),  # Reduced dimensions
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        # Define the forward pass through the network.
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        # Helper function to create the convolutional layers based on the configuration.
        cfg = [
            # Optimized channel progression
            32, 32, 'M',        # Block 1 (2 convs)
            64, 'M',            # Block 2 (1 conv)
            128, 128, 'M',      # Block 3 (2 convs)
            192, 192, 'M',      # Block 4 (2 convs)
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