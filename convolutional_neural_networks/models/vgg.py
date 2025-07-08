# based on: https://arxiv.org/pdf/1409.1556
# inspiration: 
# https://www.kaggle.com/code/vikram12301/vgg16-from-scratch-pytorch/notebook
# https://www.kaggle.com/code/vikram12301/vgg16-from-scratch-pytorch/notebook

# Import necessary libraries
import torch
import torch.nn as nn

class VGG(nn.Module):
    # Implementation of the VGG architecture for image classification.
    def __init__(self, num_classes=10):
        # Initialize the VGG model with feature extraction and classification layers.
        super(VGG, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
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
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512, 'M']
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
