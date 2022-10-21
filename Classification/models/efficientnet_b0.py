from torchvision import models


def get_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(num_classes=num_classes)
    return model

