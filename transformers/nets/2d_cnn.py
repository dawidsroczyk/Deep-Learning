from torchvision import models

def get_2d_cnn(model_name, num_classes=12, pretrained=False):
    """Get 2D CNN model with modified input and output layers"""
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        # Modify first layer for single channel input
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Modify classifier for our task
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Modify first layer for single channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final layer for our task
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=pretrained)
        # Modify first layer for single channel input
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        # Modify final layer for our task
        model.fc = nn.Linear(2048, num_classes)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model